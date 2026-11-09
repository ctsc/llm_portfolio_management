"""Weekly LLM runner: plan tasks, run local models, submit and collect batches, assess runs.

Every call is appended to data/outputs/<week>/llm/<model>__<config>.jsonl right after it
returns, so a crashed job resumes without repeating finished calls. `llm_scores.csv` is
rebuilt from those records, which makes every entry point idempotent.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import threading
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import pandas as pd

from portstudy import calendar as cal
from portstudy.config import LlmSettings, ModelRegistry, ModelSpec, StudyConfig
from portstudy.data.snapshot import Snapshot
from portstudy.io import read_json, write_csv_atomic, write_json_atomic, write_text_atomic
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
# Reasons `assess_run` writes itself; any other failed reason is a model-wide mark from J4/health.
_ASSESSED_FAILURE_PREFIXES = (
    "batch_not_ready_by_lock",
    "open_weight_not_finished_by_lock",
    "invalid_rate",
)


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
    endpoint: str = ""
    started_at: str = ""
    send_seed: bool | None = None
    json_mode: bool | None = None

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
    n_provider_retries: int = 0
    n_dropped: int = 0
    aborted: str | None = None
    cost_usd: float = 0.0
    per_config: dict[str, int] = field(default_factory=dict)


# Review fix M1: a task whose provider call fails (transport error, 429 after the provider's own
# retries, 5xx) is queued again up to this many times before it is dropped for the week.
MAX_PROVIDER_ERRORS = 2


class _CallCutoff(RuntimeError):
    pass


def lock_deadline(week_id: str, cfg: StudyConfig) -> dt.datetime:
    """Monday 9:00 am ET by default (30 minutes before the open) on the week's trade date."""
    hh, mm = (int(x) for x in cfg.timing.lock_deadline_time_et.split(":"))
    return dt.datetime.combine(cal.monday_open_day(week_id), dt.time(hh, mm), tzinfo=ET)


def _settings_for(cfg: StudyConfig, spec: ModelSpec) -> LlmSettings:
    return cfg.llm.settings.get(spec.id, spec.settings)


RUN_STATUS_LOCK = threading.RLock()


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
        clock: Callable[[], dt.datetime] | None = None,
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
        self.clock = clock
        self._providers: dict[str, LlmProvider] = dict(providers or {})
        self._plans: dict[str, list[Task]] = {}
        self.folder = week_output_dir(week_id) / "llm"
        self.folder.mkdir(parents=True, exist_ok=True)

    # Planning -----------------------------------------------------------------------------

    def models(self) -> list[ModelSpec]:
        return self.registry.enabled(synthetic=self.synthetic)

    def provider(self, model_id: str, config_id: str | None = None) -> LlmProvider:
        if model_id not in self._providers:
            status = (
                read_run_status(self.week_id).get(f"{model_id}__{config_id}") if config_id else None
            )
            self._providers[model_id] = make_provider(
                self.registry.get(model_id),
                self.stub_seed,
                gateway=self.cfg.llm.gateway,
                week_id=self.week_id,
                run_started_at=status.started_at if status else None,
            )
        return self._providers[model_id]

    def _now(self) -> dt.datetime:
        if self.clock:
            return self.clock()
        if self.synthetic:
            return lock_deadline(self.week_id, self.cfg) - dt.timedelta(
                minutes=self.cfg.pipeline.lock_margin_minutes + 1
            )
        return dt.datetime.now(tz=ET)

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
            endpoint=spec.serving,
        )

    def _finish(
        self,
        task: Task,
        spec: ModelSpec,
        settings: LlmSettings,
        raw: RawResult,
        retries: int,
        is_batch: bool,
    ) -> dict[str, Any]:
        resp = parse_response(
            raw.text,
            self.cfg.llm.rationale_token_cap,
            raw.stop_reason,
            reasoning_text=raw.reasoning_text,
        )
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
            "endpoint": raw.endpoint or ("api" if spec.kind == "api" else spec.serving),
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
            "send_seed": settings.send_seed if raw.send_seed is None else raw.send_seed,
            "json_mode": settings.json_mode if raw.json_mode is None else raw.json_mode,
        }
        return rec

    # Open-weight run ----------------------------------------------------------------------

    def _archive_source(self, model: str, config_id: str, endpoint: str) -> None:
        path = self._records_path(model, config_id)
        if not path.exists():
            return
        archive_dir = self.folder / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{model}__{config_id}.{endpoint}.jsonl"
        records: dict[str, dict[str, Any]] = {}
        if archive_path.exists():
            with archive_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        record = json.loads(line)
                        records[str(record["custom_id"])] = record
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    record = json.loads(line)
                    records[str(record["custom_id"])] = record
        text = "".join(
            json.dumps(record, sort_keys=True, default=str) + "\n" for record in records.values()
        )
        write_text_atomic(text, archive_path)
        path.unlink()

    def _prepare_open_weight_sources(
        self,
        model: str,
        spec: ModelSpec,
        tasks: list[Task],
    ) -> dict[str, dict[str, dict[str, Any]]]:
        statuses = read_run_status(self.week_id)
        now = self._now().isoformat()
        prepared: dict[str, dict[str, dict[str, Any]]] = {}
        for conf in self.manifest.enabled():
            key = f"{model}__{conf.id}"
            previous = statuses.get(key)
            records = self._read_records(model, conf.id)
            previous_endpoint = previous.endpoint if previous else ""
            if not previous_endpoint:
                endpoints = {str(record.get("endpoint", "")) for record in records.values()}
                endpoints.discard("")
                if len(endpoints) == 1:
                    previous_endpoint = next(iter(endpoints))
            if previous_endpoint and previous_endpoint != spec.serving:
                self._archive_source(model, conf.id, previous_endpoint)
                records = {}
            source_tasks = [task for task in tasks if task.config_id == conf.id]
            self._write_status(
                RunStatus(
                    model=model,
                    config_id=conf.id,
                    state="incomplete",
                    reason=f"{len(records)}/{len(source_tasks)} done",
                    n_tasks=len(source_tasks),
                    n_done=sum(task.item.custom_id in records for task in source_tasks),
                    assessed_at=now,
                    endpoint=spec.serving,
                    started_at=previous.started_at if previous and previous.started_at else now,
                    send_seed=(
                        previous.send_seed
                        if previous and previous.endpoint == spec.serving
                        else None
                    ),
                    json_mode=(
                        previous.json_mode
                        if previous and previous.endpoint == spec.serving
                        else None
                    ),
                )
            )
            prepared[conf.id] = records
        return prepared

    def _record_open_weight_result(
        self,
        task: Task,
        spec: ModelSpec,
        settings: LlmSettings,
        raw: RawResult,
        retries: int,
        key: str,
    ) -> dict[str, Any]:
        self.cache.put(key, raw)
        record = self._finish(task, spec, settings, raw, retries, is_batch=False)
        self._append_record(task.model, task.config_id, record)
        self._update_open_weight_status(task, record)
        return record

    def _update_open_weight_status(self, task: Task, record: dict[str, Any]) -> None:
        with RUN_STATUS_LOCK:
            statuses = read_run_status(self.week_id)
            key = f"{task.model}__{task.config_id}"
            status = statuses.get(key)
            if status is None:
                return
            status.n_done = len(self._read_records(task.model, task.config_id))
            status.reason = f"{status.n_done}/{status.n_tasks} done"
            status.endpoint = str(record.get("endpoint") or status.endpoint)
            status.send_seed = record.get("send_seed")
            status.json_mode = record.get("json_mode")
            self._write_status(status)

    def _persist_provider_features(
        self, model: str, endpoint: str, features: tuple[bool, bool]
    ) -> None:
        for status in read_run_status(self.week_id).values():
            if status.model == model and status.endpoint == endpoint:
                status.send_seed, status.json_mode = features
                self._write_status(status)

    @staticmethod
    def _merge_raw_results(first: RawResult, second: RawResult) -> RawResult:
        first.tokens_in += second.tokens_in
        first.tokens_out += second.tokens_out
        first.tokens_cached += second.tokens_cached
        first.latency_s += second.latency_s
        first.text = second.text
        first.error = second.error
        first.stop_reason = second.stop_reason
        first.reasoning_text = second.reasoning_text
        first.model_version = second.model_version or first.model_version
        first.endpoint = second.endpoint or first.endpoint
        first.send_seed = second.send_seed
        first.json_mode = second.json_mode
        return first

    def run_open_weight(self, model: str) -> RunSummary:
        """Run one open-weight model with bounded provider-only workers and main-thread writes."""
        spec = self.registry.get(model)
        if spec.kind != "open_weight":
            raise ValueError(f"{model} is not an open-weight model")
        tasks = self.plan_week(model)
        summary = RunSummary(model=model, n_tasks=len(tasks))
        settings = _settings_for(self.cfg, spec)
        first_config = tasks[0].config_id if tasks else None
        provider = self.provider(model, first_config)
        prior_statuses = read_run_status(self.week_id)
        prepared = self._prepare_open_weight_sources(model, spec, tasks)
        set_features = getattr(provider, "set_features", None)
        if callable(set_features):
            feature_statuses = [
                status
                for status in prior_statuses.values()
                if status.model == model and status.endpoint == spec.serving
            ]
            seed_values = [
                status.send_seed for status in feature_statuses if status.send_seed is not None
            ]
            json_values = [
                status.json_mode for status in feature_statuses if status.json_mode is not None
            ]
            if seed_values or json_values:
                set_features(
                    all(seed_values) if seed_values else settings.send_seed,
                    all(json_values) if json_values else settings.json_mode,
                )
        records = {custom_id for source in prepared.values() for custom_id in source}
        remaining: list[tuple[Task, str]] = []
        for task in tasks:
            if task.item.custom_id in records:
                continue
            key = self._key(spec, task, settings)
            cached = self.cache.get(key)
            if cached is not None:
                self.log.info(
                    "cache_hit",
                    extra={
                        "extra_fields": {
                            "week_id": self.week_id,
                            "custom_id": task.item.custom_id,
                            "endpoint": spec.serving,
                        }
                    },
                )
                record = self._finish(task, spec, settings, cached, 0, is_batch=False)
                self._append_record(model, task.config_id, record)
                self._update_open_weight_status(task, record)
                summary.n_cached += 1
                summary.n_invalid += 0 if record["valid"] else 1
                summary.cost_usd += float(record["cost_usd"])
                summary.per_config[task.config_id] = summary.per_config.get(task.config_id, 0) + 1
            else:
                remaining.append((task, key))

        cutoff = (
            dt.datetime.max.replace(tzinfo=dt.UTC)
            if self.synthetic
            else lock_deadline(self.week_id, self.cfg)
            - dt.timedelta(minutes=self.cfg.pipeline.lock_margin_minutes)
        )
        set_cutoff = getattr(provider, "set_call_cutoff", None)
        if callable(set_cutoff):
            set_cutoff(cutoff, self._now)
        workers = max(1, self.cfg.llm.concurrency.get(model, 4))
        pending: dict[Future[RawResult], tuple[Task, str, int, RawResult | None]] = {}
        versions: set[str] = set()
        next_index = 0
        fatal: Exception | None = None
        provider_errors: dict[str, int] = {}

        def call_provider(task: Task) -> RawResult:
            if self._now() >= cutoff:
                raise _CallCutoff("J6 cutoff reached before provider call")
            return provider.complete(task.item, settings)

        def submit(task: Task, key: str, retries: int, first: RawResult | None) -> None:
            future = executor.submit(call_provider, task)
            pending[future] = (task, key, retries, first)
            summary.n_calls += 1

        with ThreadPoolExecutor(max_workers=workers) as executor:
            while next_index < len(remaining) and len(pending) < workers:
                if self._now() >= cutoff:
                    break
                task, key = remaining[next_index]
                next_index += 1
                submit(task, key, 0, None)

            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    task, key, retries, first = pending.pop(future)
                    try:
                        raw = future.result()
                    except _CallCutoff:
                        continue
                    except LlmProviderError as exc:
                        features = (
                            getattr(provider, "send_seed", None),
                            getattr(provider, "json_mode", None),
                        )
                        if isinstance(features[0], bool) and isinstance(features[1], bool):
                            self._persist_provider_features(
                                model, spec.serving, (features[0], features[1])
                            )
                        n_errors = provider_errors.get(task.item.custom_id, 0) + 1
                        provider_errors[task.item.custom_id] = n_errors
                        retryable = getattr(exc, "retryable", True)
                        if not retryable and summary.aborted is None:
                            # Review fix M2: an authentication failure is model-wide; stop
                            # submitting and let J4 mark the model's sources failed.
                            summary.aborted = "authentication_failed"
                            self.log.error(
                                "provider_auth_failed",
                                extra={
                                    "extra_fields": {
                                        "week_id": self.week_id,
                                        "model": model,
                                        "error": str(exc),
                                    }
                                },
                            )
                        requeue = (
                            retryable
                            and n_errors <= MAX_PROVIDER_ERRORS
                            and fatal is None
                            and summary.aborted is None
                            and self._now() < cutoff
                        )
                        self.log.warning(
                            "provider_error",
                            extra={
                                "extra_fields": {
                                    "week_id": self.week_id,
                                    "custom_id": task.item.custom_id,
                                    "error": str(exc),
                                    "attempt": n_errors,
                                    "requeued": requeue,
                                }
                            },
                        )
                        if requeue:
                            submit(task, key, retries, first)
                            summary.n_provider_retries += 1
                        else:
                            summary.n_dropped += 1
                        continue
                    except Exception as exc:
                        fatal = exc
                        continue

                    if retries == 0:
                        response = parse_response(
                            raw.text,
                            self.cfg.llm.rationale_token_cap,
                            raw.stop_reason,
                            reasoning_text=raw.reasoning_text,
                        )
                        if (
                            (not response.valid or raw.error)
                            and self.cfg.llm.retry_invalid_once
                            and fatal is None
                            and self._now() < cutoff
                        ):
                            submit(task, key, 1, raw)
                            continue
                    else:
                        assert first is not None
                        raw = self._merge_raw_results(first, raw)

                    if raw.model_version:
                        if versions and raw.model_version not in versions:
                            self.log.warning(
                                "model_version_changed",
                                extra={
                                    "extra_fields": {
                                        "week_id": self.week_id,
                                        "model": model,
                                        "seen": sorted(versions),
                                        "returned": raw.model_version,
                                    }
                                },
                            )
                        versions.add(raw.model_version)
                    record = self._record_open_weight_result(
                        task, spec, settings, raw, retries, key
                    )
                    summary.n_invalid += 0 if record["valid"] else 1
                    summary.cost_usd += float(record["cost_usd"])
                    summary.n_retries += retries
                    summary.per_config[task.config_id] = (
                        summary.per_config.get(task.config_id, 0) + 1
                    )

                if fatal is None and summary.aborted is None:
                    while (
                        next_index < len(remaining)
                        and len(pending) < workers
                        and self._now() < cutoff
                    ):
                        task, key = remaining[next_index]
                        next_index += 1
                        submit(task, key, 0, None)

            for conf in self.manifest.enabled():
                self._record_cost(model, conf.id)
            self.write_scores()
            if fatal is not None:
                raise fatal
        return summary

    def run_local(self, model: str) -> RunSummary:
        """Compatibility alias for the open-weight runner."""
        return self.run_open_weight(model)

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
        source_status = read_run_status(self.week_id).get(f"{model}__{config_id}")
        if (
            source_status is not None
            and source_status.state == "failed"
            and not source_status.reason.startswith(_ASSESSED_FAILURE_PREFIXES)
            and len(done) < len(tasks)
        ):
            # Marked model-wide by the pre-J4 health check, an authentication abort or a crashed
            # model worker (N1): the run cannot complete, so the mark stands.
            return source_status
        endpoints = {str(record.get("endpoint", "")) for record in done}
        endpoints.discard("")
        if len(endpoints) > 1:
            self.log.warning(
                "mixed_endpoints_for_source",
                extra={
                    "extra_fields": {
                        "week_id": self.week_id,
                        "model": model,
                        "config_id": config_id,
                        "endpoints": sorted(endpoints),
                    }
                },
            )
        endpoint = (
            next(iter(endpoints))
            if len(endpoints) == 1
            else ("mixed" if endpoints else ("api" if spec.kind == "api" else spec.serving))
        )
        send_seed = next(
            (record["send_seed"] for record in done if record.get("send_seed") is not None),
            source_status.send_seed if source_status else None,
        )
        json_mode = next(
            (record["json_mode"] for record in done if record.get("json_mode") is not None),
            source_status.json_mode if source_status else None,
        )
        status = RunStatus(
            model=model,
            config_id=config_id,
            state="ok",
            n_tasks=len(tasks),
            n_done=len(done),
            n_invalid=n_invalid,
            invalid_rate=(n_invalid / len(done)) if done else 0.0,
            assessed_at=current.isoformat(),
            endpoint=endpoint,
            started_at=source_status.started_at if source_status else current.isoformat(),
            send_seed=send_seed,
            json_mode=json_mode,
        )
        if len(done) < len(tasks):
            if current >= (deadline or lock_deadline(self.week_id, self.cfg)):
                status.state = "failed"
                status.reason = (
                    "batch_not_ready_by_lock"
                    if spec.kind == "api"
                    else "open_weight_not_finished_by_lock"
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
        with RUN_STATUS_LOCK:  # PLAN3 phase 3: parallel model workers share this file
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
