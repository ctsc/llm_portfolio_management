"""Weekly LLM runner: plan tasks, run local models, submit and collect batches, assess runs.

Every call is appended to data/outputs/<week>/llm/<model>__<config>.jsonl right after it
returns, so a crashed job resumes without repeating finished calls. `llm_scores.csv` is
rebuilt from those records, which makes every entry point idempotent.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import pandas as pd

from portstudy import calendar as cal
from portstudy.config import LlmSettings, ModelRegistry, ModelSpec, StudyConfig
from portstudy.data.snapshot import Snapshot
from portstudy.io import read_json, write_csv_atomic, write_json_atomic
from portstudy.llm.cache import LlmCache, cache_key
from portstudy.llm.cost import CostTable, RunningCost, check_budget, estimate_week_cost
from portstudy.llm.prompts import PromptManifest, assemble_prompt, load_manifest
from portstudy.llm.providers.base import (
    BatchHandle,
    BatchItem,
    LlmProvider,
    LlmProviderError,
    RawResult,
    make_custom_id,
)
from portstudy.llm.providers.factory import make_provider
from portstudy.llm.schema import INVALID_OUTPUT, fallback_response
from portstudy.llm.validate import parse_response
from portstudy.paths import week_output_dir

ET = ZoneInfo("America/New_York")

SCORE_COLUMNS = [
    "model",
    "config_id",
    "ticker",
    "low",
    "likely",
    "high",
    "width",
    "rationale",
    "valid",
    "fallback_flag",
    "extra_keys",
    "had_thinking",
    "endpoint",
    "tokens_in",
    "tokens_cached",
    "tokens_out",
    "cost_usd",
    "latency_s",
    "model_version",
    "prompt_hash",
    "prompt_file_sha256",
]

RunState = Literal["ok", "failed", "incomplete"]


@dataclass(frozen=True)
class Task:
    model: str
    config_id: str
    ticker: str
    item: BatchItem


@dataclass
class RunStatus:
    model: str
    config_id: str
    state: RunState
    reason: str = ""
    n_tasks: int = 0
    n_done: int = 0
    n_invalid: int = 0
    invalid_rate: float = 0.0
    assessed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunSummary:
    model: str
    n_tasks: int = 0
    n_calls: int = 0
    n_cached: int = 0
    n_invalid: int = 0
    n_retries: int = 0
    cost_usd: float = 0.0
    per_config: dict[str, int] = field(default_factory=dict)


def lock_deadline(week_id: str, cfg: StudyConfig) -> dt.datetime:
    """Monday 9:00 am ET by default (30 minutes before the open) on the week's trade date."""
    hh, mm = (int(x) for x in cfg.timing.lock_deadline_time_et.split(":"))
    return dt.datetime.combine(cal.monday_open_day(week_id), dt.time(hh, mm), tzinfo=ET)


def _settings_for(cfg: StudyConfig, spec: ModelSpec) -> LlmSettings:
    return cfg.llm.settings.get(spec.id, spec.settings)


class WeekRunner:
    def __init__(
        self,
        week_id: str,
        cfg: StudyConfig,
        registry: ModelRegistry,
        snapshot: Snapshot,
        synthetic: bool = False,
        manifest: PromptManifest | None = None,
        providers: dict[str, LlmProvider] | None = None,
        cache: LlmCache | None = None,
        logger: logging.Logger | None = None,
        stub_seed: int = 0,
    ) -> None:
        self.week_id = week_id
        self.cfg = cfg
        self.registry = registry
        self.snapshot = snapshot
        self.synthetic = synthetic
        self.manifest = manifest or load_manifest()
        self.cache = cache or LlmCache()
        self.log = logger or logging.getLogger("portstudy.llm")
        self.table = CostTable(registry)
        self.stub_seed = stub_seed
        self._providers: dict[str, LlmProvider] = dict(providers or {})
        self._plans: dict[str, list[Task]] = {}
        self.folder = week_output_dir(week_id) / "llm"
        self.folder.mkdir(parents=True, exist_ok=True)

    # Planning -----------------------------------------------------------------------------

    def models(self) -> list[ModelSpec]:
        return self.registry.enabled(synthetic=self.synthetic)

    def provider(self, model_id: str) -> LlmProvider:
        if model_id not in self._providers:
            self._providers[model_id] = make_provider(self.registry.get(model_id), self.stub_seed)
        return self._providers[model_id]

    def plan_week(self, model: str | None = None) -> list[Task]:
        """Tasks (model, configuration, ticker) for every enabled model and configuration."""
        if model is None:
            return [t for m in self.models() for t in self.plan_week(m.id)]
        cached = self._plans.get(model)
        if cached is not None:
            return cached
        tasks = self._plan_model(model)
        self._plans[model] = tasks
        return tasks

    def _plan_model(self, model: str) -> list[Task]:
        ref = self.snapshot.reference.set_index("ticker")
        news = self.snapshot.news_text.set_index("ticker")["news_text"]
        tasks: list[Task] = []
        for spec in self.models():
            if spec.id != model:
                continue
            for conf in self.manifest.enabled():
                for tk in self.snapshot.included_tickers:
                    row = {
                        "ticker": tk,
                        "name": ref.loc[tk, "name"],
                        "sector": ref.loc[tk, "sector"],
                    }
                    prompt = assemble_prompt(
                        conf.id, row, str(news.get(tk, "")), self.cfg, self.manifest
                    )
                    item = BatchItem(
                        custom_id=make_custom_id(spec.id, conf.id, self.week_id, tk),
                        model=spec.id,
                        config_id=conf.id,
                        week_id=self.week_id,
                        ticker=tk,
                        prompt=prompt,
                    )
                    tasks.append(Task(spec.id, conf.id, tk, item))
        return tasks

    # Records ------------------------------------------------------------------------------

    def _records_path(self, model: str, config_id: str) -> Path:
        return self.folder / f"{model}__{config_id}.jsonl"

    def _read_records(self, model: str, config_id: str) -> dict[str, dict[str, Any]]:
        path = self._records_path(model, config_id)
        out: dict[str, dict[str, Any]] = {}
        if not path.exists():
            return out
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    out[str(rec["custom_id"])] = rec
        return out

    def _append_record(self, model: str, config_id: str, rec: dict[str, Any]) -> None:
        with self._records_path(model, config_id).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")

    def _handle_path(self, model: str, config_id: str) -> Path:
        return self.folder / f"batch_{model}__{config_id}.json"

    def _key(self, spec: ModelSpec, task: Task, settings: LlmSettings) -> str:
        return cache_key(
            spec.snapshot_id or spec.provider_model,
            task.item.prompt.prompt_hash,
            task.ticker,
            self.week_id,
            settings,
            model_id=spec.id,
        )

    # One call -----------------------------------------------------------------------------

    def _call_once(
        self, provider: LlmProvider, item: BatchItem, settings: LlmSettings
    ) -> RawResult:
        attempts = max(1, self.cfg.data.retries.attempts)
        last: Exception | None = None
        for _ in range(attempts):
            try:
                return provider.complete(item, settings)
            except LlmProviderError as exc:
                last = exc
                self.log.warning(
                    "provider_error",
                    extra={
                        "extra_fields": {
                            "week_id": self.week_id,
                            "custom_id": item.custom_id,
                            "error": str(exc),
                        }
                    },
                )
        return RawResult(custom_id=item.custom_id, text="", error=f"provider_error: {last}")

    def _finish(
        self,
        task: Task,
        spec: ModelSpec,
        settings: LlmSettings,
        raw: RawResult,
        retries: int,
        is_batch: bool,
    ) -> dict[str, Any]:
        resp = parse_response(raw.text, self.cfg.llm.rationale_token_cap, raw.stop_reason)
        if raw.error:
            resp = fallback_response(raw.text, raw.error, INVALID_OUTPUT)
        cost = (
            0.0
            if raw.from_cache
            else self.table.cost_usd(
                spec.id, raw.tokens_in, raw.tokens_cached, raw.tokens_out, is_batch
            )
        )
        rec: dict[str, Any] = {
            "custom_id": task.item.custom_id,
            "model": spec.id,
            "config_id": task.config_id,
            "ticker": task.ticker,
            "low": resp.low,
            "likely": resp.likely,
            "high": resp.high,
            "width": resp.high - resp.low if resp.valid else "",
            "rationale": resp.rationale,
            "valid": resp.valid,
            "fallback_flag": resp.fallback_flag,
            "extra_keys": resp.extra_keys,
            "had_thinking": resp.had_thinking,
            "endpoint": "",
            "error": resp.error,
            "raw": raw.text,
            "tokens_in": raw.tokens_in,
            "tokens_cached": raw.tokens_cached,
            "tokens_out": raw.tokens_out,
            "cost_usd": cost,
            "latency_s": raw.latency_s,
            "model_version": raw.model_version,
            "prompt_hash": task.item.prompt.prompt_hash,
            "prompt_file_sha256": task.item.prompt.prompt_file_sha256,
            "from_cache": raw.from_cache,
            "retries": retries,
        }
        return rec

    def _run_task(
        self,
        task: Task,
        spec: ModelSpec,
        settings: LlmSettings,
        provider: LlmProvider,
        seen_versions: set[str],
    ) -> dict[str, Any]:
        key = self._key(spec, task, settings)
        cached = self.cache.get(key)
        retries = 0
        if cached is not None:
            self.log.info(
                "cache_hit",
                extra={"extra_fields": {"week_id": self.week_id, "custom_id": task.item.custom_id}},
            )
            raw = cached
        else:
            raw = self._call_once(provider, task.item, settings)
            first = parse_response(raw.text, self.cfg.llm.rationale_token_cap, raw.stop_reason)
            # Invalid-output guardrail: retry exactly once, then fall back.
            if (not first.valid or raw.error) and self.cfg.llm.retry_invalid_once:
                retries = 1
                self.log.info(
                    "invalid_retry",
                    extra={
                        "extra_fields": {
                            "week_id": self.week_id,
                            "custom_id": task.item.custom_id,
                            "error": first.error or raw.error,
                        }
                    },
                )
                second = self._call_once(provider, task.item, settings)
                raw.tokens_in += second.tokens_in
                raw.tokens_out += second.tokens_out
                raw.tokens_cached += second.tokens_cached
                raw.latency_s += second.latency_s
                raw.text, raw.error, raw.stop_reason = second.text, second.error, second.stop_reason
                raw.model_version = second.model_version or raw.model_version
            self.cache.put(key, raw)
        if raw.model_version:
            if seen_versions and raw.model_version not in seen_versions:
                self.log.warning(
                    "model_version_changed",
                    extra={
                        "extra_fields": {
                            "week_id": self.week_id,
                            "model": spec.id,
                            "seen": sorted(seen_versions),
                            "returned": raw.model_version,
                        }
                    },
                )
            seen_versions.add(raw.model_version)
        return self._finish(task, spec, settings, raw, retries, is_batch=False)

    # Local run ----------------------------------------------------------------------------

    def run_local(self, model: str) -> RunSummary:
        """Resumable sequential run for one model, grouped by configuration, cache first."""
        spec = self.registry.get(model)
        settings = _settings_for(self.cfg, spec)
        provider = self.provider(model)
        tasks = self.plan_week(model)
        summary = RunSummary(model=model, n_tasks=len(tasks))
        seen_versions: set[str] = set()
        for conf in self.manifest.enabled():
            done = self._read_records(model, conf.id)
            for task in (t for t in tasks if t.config_id == conf.id):
                if task.item.custom_id in done:
                    summary.per_config[conf.id] = summary.per_config.get(conf.id, 0) + 1
                    continue
                rec = self._run_task(task, spec, settings, provider, seen_versions)
                self._append_record(model, conf.id, rec)
                summary.n_calls += 0 if rec["from_cache"] else 1
                summary.n_cached += 1 if rec["from_cache"] else 0
                summary.n_invalid += 0 if rec["valid"] else 1
                summary.n_retries += int(rec["retries"])
                summary.cost_usd += float(rec["cost_usd"])
                summary.per_config[conf.id] = summary.per_config.get(conf.id, 0) + 1
            self._record_cost(model, conf.id)
        self.write_scores()
        return summary

    # Batch run ----------------------------------------------------------------------------

    def pending_items(self, model: str, config_id: str) -> list[Task]:
        done = self._read_records(model, config_id)
        return [
            t
            for t in self.plan_week(model)
            if t.config_id == config_id and t.item.custom_id not in done
        ]

    def submit_api(self, model: str, allow_over_budget: bool = False) -> list[BatchHandle]:
        """Cost guard first, then one batch per configuration for the not-yet-answered items."""
        spec = self.registry.get(model)
        settings = _settings_for(self.cfg, spec)
        provider = self.provider(model)
        # Cache hits are answered now and never submitted.
        to_submit: dict[str, list[Task]] = {}
        for conf in self.manifest.enabled():
            pending = self.pending_items(model, conf.id)
            remaining: list[Task] = []
            for task in pending:
                key = self._key(spec, task, settings)
                cached = self.cache.get(key)
                if cached is not None:
                    self.log.info(
                        "cache_hit",
                        extra={
                            "extra_fields": {
                                "week_id": self.week_id,
                                "custom_id": task.item.custom_id,
                            }
                        },
                    )
                    self._append_record(
                        model, conf.id, self._finish(task, spec, settings, cached, 0, True)
                    )
                else:
                    remaining.append(task)
            if remaining:
                to_submit[conf.id] = remaining
        all_items = [t.item for ts in to_submit.values() for t in ts]
        estimate = estimate_week_cost(all_items, spec, settings, self.table)
        projected = check_budget(
            self.week_id, estimate["cost_usd"], self.cfg, allow_over_budget, self.log
        )
        handles: list[BatchHandle] = []
        for conf_id, tasks in to_submit.items():
            path = self._handle_path(model, conf_id)
            if path.exists():
                handle = BatchHandle.from_dict(read_json(path))
                if set(handle.custom_ids) == {t.item.custom_id for t in tasks}:
                    handles.append(handle)
                    continue
            handle = provider.submit_batch([t.item for t in tasks], settings)
            handle.extra["projected_study_usd"] = projected
            handle.extra["over_budget_allowed"] = allow_over_budget
            handle.extra["estimate_usd"] = estimate["cost_usd"]
            write_json_atomic(handle.to_dict(), path)
            self.log.info(
                "batch_submitted",
                extra={
                    "extra_fields": {
                        "week_id": self.week_id,
                        "model": model,
                        "config_id": conf_id,
                        "batch_id": handle.batch_id,
                        "n_items": len(tasks),
                    }
                },
            )
            handles.append(handle)
        return handles

    def poll_api(self, model: str) -> dict[str, str]:
        provider = self.provider(model)
        states: dict[str, str] = {}
        for conf in self.manifest.enabled():
            path = self._handle_path(model, conf.id)
            if not path.exists():
                states[conf.id] = "not_submitted" if self.pending_items(model, conf.id) else "done"
                continue
            handle = BatchHandle.from_dict(read_json(path))
            states[conf.id] = provider.poll_batch(handle).state
            write_json_atomic(handle.to_dict(), path)
        return states

    def collect_api(self, model: str, resubmit_failed: bool = True) -> RunSummary:
        """Collect finished batches, write records, resubmit failed items up to the retry count."""
        spec = self.registry.get(model)
        settings = _settings_for(self.cfg, spec)
        provider = self.provider(model)
        summary = RunSummary(model=model, n_tasks=len(self.plan_week(model)))
        seen_versions: set[str] = set()
        for conf in self.manifest.enabled():
            path = self._handle_path(model, conf.id)
            if not path.exists():
                continue
            handle = BatchHandle.from_dict(read_json(path))
            tasks = {t.item.custom_id: t for t in self.plan_week(model) if t.config_id == conf.id}
            # Providers that hold batch state in memory (the stub) re-attach the items after a
            # process restart; real providers keep the state server-side and ignore this.
            rehydrate = getattr(provider, "rehydrate_batch", None)
            if callable(rehydrate):
                rehydrate(handle, [tasks[c].item for c in handle.custom_ids if c in tasks])
            status = provider.poll_batch(handle)
            if status.state != "done":
                if status.state == "failed":
                    self.log.warning(
                        "batch_failed",
                        extra={
                            "extra_fields": {
                                "week_id": self.week_id,
                                "model": model,
                                "config_id": conf.id,
                            }
                        },
                    )
                continue
            results = {r.custom_id: r for r in provider.collect_batch(handle)}
            done = self._read_records(model, conf.id)
            failed: list[Task] = []
            for cid in handle.custom_ids:
                if cid in done or cid not in tasks:
                    continue
                task = tasks[cid]
                raw = results.get(cid) or RawResult(
                    custom_id=cid, text="", error="missing_from_batch"
                )
                if (
                    raw.error
                    and resubmit_failed
                    and int(handle.extra.get("retry", 0)) < self.cfg.llm.batch_retry_count
                ):
                    failed.append(task)
                    continue
                if raw.model_version:
                    if seen_versions and raw.model_version not in seen_versions:
                        self.log.warning(
                            "model_version_changed",
                            extra={
                                "extra_fields": {
                                    "week_id": self.week_id,
                                    "model": model,
                                    "returned": raw.model_version,
                                }
                            },
                        )
                    seen_versions.add(raw.model_version)
                key = self._key(spec, task, settings)
                self.cache.put(key, raw)
                rec = self._finish(task, spec, settings, raw, 0, is_batch=True)
                self._append_record(model, conf.id, rec)
                summary.n_calls += 1
                summary.n_invalid += 0 if rec["valid"] else 1
                summary.cost_usd += float(rec["cost_usd"])
            if failed:
                new = provider.submit_batch([t.item for t in failed], settings)
                new.extra["retry"] = int(handle.extra.get("retry", 0)) + 1
                write_json_atomic(new.to_dict(), path)
                self.log.warning(
                    "batch_resubmitted",
                    extra={
                        "extra_fields": {
                            "week_id": self.week_id,
                            "model": model,
                            "config_id": conf.id,
                            "n_items": len(failed),
                        }
                    },
                )
            else:
                path.unlink()
            self._record_cost(model, conf.id, extra=handle.extra)
        self.write_scores()
        return summary

    # Outputs ------------------------------------------------------------------------------

    def _record_cost(self, model: str, config_id: str, extra: dict[str, Any] | None = None) -> None:
        recs = list(self._read_records(model, config_id).values())
        if not recs:
            return
        df = pd.DataFrame(recs)
        handle_path = self._handle_path(model, config_id)
        if extra is None:
            extra = read_json(handle_path).get("extra", {}) if handle_path.exists() else {}
        RunningCost(self.week_id).record(
            {
                "model": model,
                "config_id": config_id,
                "n_calls": int((~df["from_cache"]).sum()),
                "n_cached": int(df["from_cache"].sum()),
                "tokens_in": int(df["tokens_in"].sum()),
                "tokens_cached": int(df["tokens_cached"].sum()),
                "tokens_out": int(df["tokens_out"].sum()),
                "cost_usd": float(df["cost_usd"].sum()),
                "projected_study_usd": float(extra.get("projected_study_usd", 0.0)),
                "over_budget_allowed": bool(extra.get("over_budget_allowed", False)),
            }
        )

    def write_scores(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for spec in self.models():
            for conf in self.manifest.enabled():
                rows.extend(self._read_records(spec.id, conf.id).values())
        if rows:
            df = pd.DataFrame(rows).loc[:, SCORE_COLUMNS]
            df = df.sort_values(["model", "config_id", "ticker"]).reset_index(drop=True)
        else:
            df = pd.DataFrame(columns=SCORE_COLUMNS)
        write_csv_atomic(df, week_output_dir(self.week_id) / "llm_scores.csv")
        return df

    # Assessment ---------------------------------------------------------------------------

    def assess_run(
        self,
        model: str,
        config_id: str,
        now: dt.datetime | None = None,
        deadline: dt.datetime | None = None,
    ) -> RunStatus:
        """DECISION(N1): invalid rate above the threshold, or not finished by the lock deadline."""
        spec = self.registry.get(model)
        tasks = [t for t in self.plan_week(model) if t.config_id == config_id]
        recs = self._read_records(model, config_id)
        done = [recs[t.item.custom_id] for t in tasks if t.item.custom_id in recs]
        n_invalid = sum(1 for r in done if not r["valid"])
        current = now or dt.datetime.now(tz=ET)
        status = RunStatus(
            model=model,
            config_id=config_id,
            state="ok",
            n_tasks=len(tasks),
            n_done=len(done),
            n_invalid=n_invalid,
            invalid_rate=(n_invalid / len(done)) if done else 0.0,
            assessed_at=current.isoformat(),
        )
        if len(done) < len(tasks):
            if current >= (deadline or lock_deadline(self.week_id, self.cfg)):
                status.state = "failed"
                status.reason = (
                    "batch_not_ready_by_lock"
                    if spec.kind == "api"
                    else "local_not_finished_by_lock"
                )
            else:
                status.state = "incomplete"
                status.reason = f"{len(done)}/{len(tasks)} done"
        elif status.invalid_rate > self.cfg.llm.invalid_rate_threshold:
            status.state = "failed"
            status.reason = (
                f"invalid_rate {status.invalid_rate:.3f} > {self.cfg.llm.invalid_rate_threshold}"
            )
        self._write_status(status)
        return status

    def _write_status(self, status: RunStatus) -> None:
        path = week_output_dir(self.week_id) / "run_status.json"
        data: dict[str, Any] = read_json(path) if path.exists() else {}
        data[f"{status.model}__{status.config_id}"] = status.to_dict()
        write_json_atomic(data, path)


def read_run_status(week_id: str) -> dict[str, RunStatus]:
    path = week_output_dir(week_id) / "run_status.json"
    if not path.exists():
        return {}
    data = read_json(path)
    return {k: RunStatus(**v) for k, v in data.items()}


def read_scores(week_id: str) -> pd.DataFrame:
    path = week_output_dir(week_id) / "llm_scores.csv"
    if not path.exists():
        return pd.DataFrame(columns=SCORE_COLUMNS)
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df["rationale"] = df["rationale"].fillna("").astype(str)
    df["valid"] = df["valid"].astype(str).str.lower().eq("true")
    return df
