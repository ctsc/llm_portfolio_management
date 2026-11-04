"""The weekly jobs J1 to J9. Each reads named inputs, writes named outputs, and is idempotent."""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from portstudy import calendar as cal
from portstudy.baseline.combine import build_baseline, read_baseline
from portstudy.config import repo_root
from portstudy.data import ledger
from portstudy.data.snapshot import make_snapshot, read_snapshot, snapshot_exists
from portstudy.data.universe import build_universe, universe_path
from portstudy.hashing import sha256_file, stable_hash
from portstudy.io import read_csv, read_json, write_json_atomic
from portstudy.llm.cache import LlmCache
from portstudy.llm.runner import WeekRunner, lock_deadline, read_scores
from portstudy.optimize import (
    WeekInputs,
    build_control_weights,
    build_q0_weights,
    build_weights,
    portfolio_ics,
    posterior_dir,
    posterior_for_q0,
    posterior_for_source,
    read_weights_meta,
    weights_dir,
)
from portstudy.paths import ledger_dir, snapshot_dir, week_output_dir
from portstudy.pipeline import fallback as fb
from portstudy.pipeline.context import RunContext
from portstudy.pipeline.status import JobStatus, hash_paths, read_status, run_job
from portstudy.risk.covariance import build_risk_model
from portstudy.risk.prior import build_prior
from portstudy.sim import PriceBook, WeekPlan, replay_all
from portstudy.types import PortfolioId, RatingSource, StrategyName
from portstudy.views.ic import ICStore, rank_ic, weekly_returns
from portstudy.views.residualize import valid_scores


class LockViolation(RuntimeError):
    pass


def _config_hash(ctx: RunContext) -> str:
    return stable_hash(
        {
            "study": ctx.cfg.model_dump(),
            "models": ctx.run_registry().model_dump(),
            "manifest": ctx.run_manifest().model_dump(),
            "synthetic": ctx.synthetic,
        }
    )


def _week_runner(ctx: RunContext, week_id: str) -> WeekRunner:
    return WeekRunner(
        week_id,
        ctx.cfg,
        ctx.run_registry(),
        read_snapshot(week_id),
        synthetic=ctx.synthetic,
        manifest=ctx.run_manifest(),
        providers=ctx.llm_providers or None,
        cache=LlmCache(),
        logger=ctx.log("llm"),
        stub_seed=ctx.stub_seed,
        clock=ctx.now,
    )


def _universe(ctx: RunContext, week_id: str) -> pd.DataFrame:
    """The frozen universe file; built once from the first cutoff and never rebuilt by code."""
    path = universe_path(ctx.cfg, synthetic=ctx.synthetic)
    if not path.exists():
        as_of = cal.cutoff_day(ctx.live_start_week if ctx.synthetic else week_id)
        return build_universe(as_of, ctx.cfg, ctx.providers(), out_path=path, print_table=False)
    return read_csv(path)


# J1 -------------------------------------------------------------------------------------------


def J1_snapshot(ctx: RunContext, week_id: str) -> JobStatus:
    uni_path = universe_path(ctx.cfg, synthetic=ctx.synthetic)
    inputs = stable_hash(
        {
            "week": week_id,
            "cfg": _config_hash(ctx),
            "universe": sha256_file(uni_path) if uni_path.exists() else None,
        }
    )

    def fn() -> dict[str, Any]:
        universe = _universe(ctx, week_id)
        ref = make_snapshot(week_id, ctx.cfg, ctx.providers(), universe, logger=ctx.log("snapshot"))
        snap = read_snapshot(week_id)
        return {
            "path": str(ref.path),
            "n_included": len(snap.included_tickers),
            "cutoff_date": ref.cutoff_date.isoformat(),
        }

    return run_job(
        week_id,
        "J1",
        inputs,
        fn,
        force=ctx.force,
        outputs=lambda: [snapshot_dir(week_id) / "manifest.json"],
    )


# J2 -------------------------------------------------------------------------------------------


def _snapshot_hash(week_id: str) -> str | None:
    path = snapshot_dir(week_id) / "manifest.json"
    return sha256_file(path) if path.exists() else None


def J2_baseline(ctx: RunContext, week_id: str) -> JobStatus:
    inputs = stable_hash({"snapshot": _snapshot_hash(week_id), "cfg": _config_hash(ctx)})

    def fn() -> dict[str, Any]:
        if not snapshot_exists(week_id):
            raise FileNotFoundError(f"no snapshot for {week_id}; run J1 first")
        snap = read_snapshot(week_id)
        base = build_baseline(week_id, ctx.cfg, snap, logger=ctx.log("baseline"))
        risk = build_risk_model(week_id, ctx.cfg.risk, snap, tickers=list(base["ticker"]))
        build_prior(week_id, ctx.cfg.risk, risk, snap.reference)
        return {"n_stocks": len(base), "alpha": risk.alpha}

    folder = week_output_dir(week_id)
    return run_job(
        week_id,
        "J2",
        inputs,
        fn,
        force=ctx.force,
        outputs=lambda: [
            folder / "baseline.csv",
            folder / "prior.csv",
            folder / "sigma_matrix.csv",
        ],
    )


# J3 to J5 (LLM) -----------------------------------------------------------------------------


def _llm_inputs(ctx: RunContext, week_id: str) -> str:
    return stable_hash(
        {
            "snapshot": _snapshot_hash(week_id),
            "cfg": _config_hash(ctx),
            "registry": ctx.run_registry().model_dump(),
            "prompts": hash_paths([repo_root() / "config" / "prompts_manifest.yaml"]),
        }
    )


def J3_api_submit(ctx: RunContext, week_id: str) -> JobStatus:
    def fn() -> dict[str, Any]:
        runner = _week_runner(ctx, week_id)
        out: dict[str, Any] = {}
        for model in ctx.api_models():
            handles = runner.submit_api(model, allow_over_budget=ctx.allow_over_budget)
            out[model] = [h.batch_id for h in handles]
        return out

    return run_job(week_id, "J3", _llm_inputs(ctx, week_id), fn, force=ctx.force)


def J4_open_weight_infer(ctx: RunContext, week_id: str) -> JobStatus:
    def fn() -> dict[str, Any]:
        # Kronos (inside J2) must be finished first: enforced with the status file, not time.
        if read_status(week_id, "J2").state != "done":
            raise RuntimeError(
                "J2 (baseline, including the forecaster) is not done; J4 waits for it"
            )
        runner = _week_runner(ctx, week_id)
        out: dict[str, Any] = {}
        for model in ctx.open_weight_models():
            s = runner.run_open_weight(model)
            out[model] = {"n_calls": s.n_calls, "n_cached": s.n_cached, "n_invalid": s.n_invalid}
        return out

    return run_job(week_id, "J4", _llm_inputs(ctx, week_id), fn, force=ctx.force)


def J5_api_collect(ctx: RunContext, week_id: str) -> JobStatus:
    def fn() -> dict[str, Any]:
        runner = _week_runner(ctx, week_id)
        out: dict[str, Any] = {}
        pending = False
        for model in ctx.api_models():
            runner.collect_api(model)
            states = runner.poll_api(model)
            out[model] = states
            if any(s not in ("done",) for s in states.values()):
                pending = True
        out["pending"] = pending
        return out

    return run_job(week_id, "J5", _llm_inputs(ctx, week_id), fn, force=True, pending_ok=True)


# J6 -------------------------------------------------------------------------------------------


def J6_validate(ctx: RunContext, week_id: str) -> JobStatus:
    scores_path = week_output_dir(week_id) / "llm_scores.csv"
    inputs = stable_hash(
        {
            "scores": sha256_file(scores_path) if scores_path.exists() else None,
            "j2": read_status(week_id, "J2").state,
            "now": ctx.now().isoformat(),
        }
    )

    def fn() -> dict[str, Any]:
        plan = fb.FallbackPlan(week_id=week_id)
        if read_status(week_id, "J2").state != "done":
            plan.no_rebalance = True
            plan.no_rebalance_reason = "baseline_unavailable"
            fb.write_plan(plan)
            return {"no_rebalance": True, "reason": plan.no_rebalance_reason}
        runner = _week_runner(ctx, week_id)
        now = ctx.now()
        # DECISION(N1): a run not finished `lock_margin_minutes` before the lock is a failure,
        # which leaves J7 time to write the weights before the deadline itself.
        deadline = lock_deadline(week_id, ctx.cfg) - dt.timedelta(
            minutes=ctx.cfg.pipeline.lock_margin_minutes
        )
        incomplete: list[str] = []
        for source in ctx.sources():
            status = runner.assess_run(source.model, source.config_id, now=now, deadline=deadline)
            if status.state == "ok":
                plan.statuses[source.key] = "normal"
            elif status.state == "failed":
                plan.statuses[source.key] = "fallback"
                plan.reasons[source.key] = status.reason
            else:
                incomplete.append(f"{source.key} ({status.reason})")
        if incomplete:
            # Before the margin deadline there is still time: do not write a plan yet.
            raise RuntimeError(
                "runs still incomplete before the lock margin: " + ", ".join(incomplete)
            )
        fb.write_plan(plan)
        n_fb = sum(1 for s in plan.statuses.values() if s == "fallback")
        return {"n_fallback": n_fb, "reasons": plan.reasons}

    return run_job(week_id, "J6", inputs, fn, force=True)


# J7 -------------------------------------------------------------------------------------------


def _previous_week_with_weights(week_id: str, portfolio: str) -> str | None:
    prev = cal.prev_week_id(week_id)
    for _ in range(60):
        if (weights_dir(prev) / f"{portfolio}.csv").exists():
            return prev
        prev = cal.prev_week_id(prev)
    return None


def J7_weights(ctx: RunContext, week_id: str) -> JobStatus:
    folder = week_output_dir(week_id)
    plan_file = folder / "fallback_plan.json"
    inputs = stable_hash(
        {
            "plan": sha256_file(plan_file) if plan_file.exists() else None,
            "baseline": hash_paths(
                [
                    folder / "baseline.csv",
                    folder / "prior.csv",
                    folder / "sigma_matrix.csv",
                    folder / "llm_scores.csv",
                ]
            ),
            "cfg": _config_hash(ctx),
        }
    )

    def fn() -> dict[str, Any]:
        now = ctx.now()
        deadline = lock_deadline(week_id, ctx.cfg)
        if now > deadline:
            raise LockViolation(
                f"lock time {now.isoformat()} is after the deadline {deadline.isoformat()}"
            )
        plan = fb.read_plan(week_id)
        if plan is None:
            raise FileNotFoundError("fallback_plan.json missing; run J6 first")
        log = ctx.log("weights")
        entries: list[dict[str, Any]] = []
        if plan.no_rebalance:
            entries.append(
                {"portfolio": "*", "status": "no_rebalance", "reason": plan.no_rebalance_reason}
            )
            fb.write_log(week_id, entries)
            write_json_atomic(
                {
                    "week_id": week_id,
                    "locked_at": now.isoformat(),
                    "no_rebalance": True,
                    "files": {},
                },
                weights_dir(week_id) / "locked.json",
            )
            return {"no_rebalance": True}
        inputs_week = WeekInputs.load(week_id)
        store = ICStore()
        ic_b, _ = portfolio_ics(week_id, ctx.cfg, store, ctx.live_start_week)
        summary: dict[str, Any] = {}
        q0_posterior = posterior_for_q0(inputs_week, ctx.cfg, ic_b, logger=log)
        q0_results: dict[str, Any] = {}
        for strategy in ctx.cfg.strategies.enabled:
            q0_key = PortfolioId.q0(strategy).key
            q0 = build_q0_weights(
                week_id,
                ctx.cfg,
                ic_b,
                inputs=inputs_week,
                previous_week=_previous_week_with_weights(week_id, q0_key),
                logger=log,
                strategy=strategy,
                posterior=q0_posterior,
            )
            q0_results[strategy] = q0
            summary[q0_key] = {"ic_b": ic_b, "solver": q0.solver}
        for name in ctx.cfg.strategies.controls:
            control = build_control_weights(week_id, name, ctx.cfg, inputs_week, log)
            summary[name] = {"solver": control.solver}

        def copy_q0(
            source: RatingSource,
            strategy: StrategyName,
            decision_source: str,
            reason: str = "",
            fallback: bool = False,
        ) -> None:
            key = PortfolioId.steered(strategy, source).key
            q0_key = PortfolioId.q0(strategy).key
            weight_folder = weights_dir(week_id)
            shutil.copyfile(weight_folder / f"{q0_key}.csv", weight_folder / f"{key}.csv")
            meta = read_json(weight_folder / f"{q0_key}.meta.json")
            source_scores = inputs_week.scores_for(source.model, source.config_id)
            endpoints = (
                {str(value) for value in source_scores["endpoint"] if str(value)}
                if "endpoint" in source_scores
                else set()
            )
            meta.update(
                {
                    "portfolio": key,
                    "strategy": strategy,
                    "source": source.key,
                    "endpoint": next(iter(endpoints))
                    if len(endpoints) == 1
                    else "mixed"
                    if endpoints
                    else "",
                    "decision_source": decision_source,
                    "use_llm_view": False,
                    "view_shift": 0.0,
                    "fallback": fallback,
                }
            )
            if reason:
                meta["reason"] = reason
            else:
                meta.pop("reason", None)
            write_json_atomic(meta, weight_folder / f"{key}.meta.json")
            if fallback:
                entries.append(
                    {
                        "portfolio": key,
                        "status": "fallback",
                        "reason": reason,
                        "decision_source": "fallback_q0",
                    }
                )

        q0_posterior_path = posterior_dir(week_id) / "Q0.csv"
        for source in ctx.sources():
            source_status = plan.status_for(source.key)
            _, ic_l = portfolio_ics(
                week_id,
                ctx.cfg,
                store,
                ctx.live_start_week,
                source.model,
                source.config_id,
                fallback_weeks=fb.fallback_weeks(source.key),
            )
            source_posterior = None
            if source_status != "fallback" and ctx.now() < deadline:
                source_posterior = posterior_for_source(
                    inputs_week, source, ctx.cfg, ic_b, ic_l, logger=log
                )
            else:
                shutil.copyfile(
                    q0_posterior_path,
                    posterior_dir(week_id) / f"{source.key}.csv",
                )
            for strategy in ctx.cfg.strategies.enabled:
                key = PortfolioId.steered(strategy, source).key
                reason = plan.reason_for(source.key)
                if source_status == "fallback":
                    copy_q0(source, strategy, "fallback_q0", reason, fallback=True)
                    summary[key] = {"fallback": True, "reason": reason}
                    continue
                if ctx.now() >= deadline:
                    copy_q0(source, strategy, "fallback_q0", "lock_deadline", fallback=True)
                    summary[key] = {"fallback": True, "reason": "lock_deadline"}
                    continue
                if source_posterior is None:
                    copy_q0(source, strategy, "llm")
                    summary[key] = {
                        "use_llm_view": False,
                        "solver": q0_results[strategy].solver,
                    }
                    continue
                q0 = q0_results[strategy]
                w = build_weights(
                    week_id,
                    source.model,
                    source.config_id,
                    ctx.cfg,
                    ic_b,
                    ic_l,
                    inputs=inputs_week,
                    previous_week=_previous_week_with_weights(
                        week_id, PortfolioId.steered(strategy, source).key
                    ),
                    logger=log,
                    strategy=strategy,
                    posterior=source_posterior,
                    q0_weights=q0.weights,
                )
                summary[key] = {
                    "ic_l": ic_l,
                    "use_llm_view": w.use_llm_view,
                    "solver": w.solver,
                    "n_invalid": w.n_invalid,
                }
                if w.fallback_flag:
                    entries.append(
                        {
                            "portfolio": key,
                            "status": "normal",
                            "reason": w.fallback_flag,
                            "decision_source": "llm",
                        }
                    )
        fb.write_log(week_id, entries)
        files = {
            p.name: sha256_file(p)
            for p in sorted(weights_dir(week_id).iterdir())
            if p.is_file() and p.name != "locked.json"
        }
        write_json_atomic(
            {
                "week_id": week_id,
                "locked_at": ctx.now().isoformat(),
                "deadline": deadline.isoformat(),
                "files": files,
            },
            weights_dir(week_id) / "locked.json",
        )
        return {"n_portfolios": len(files) // 2, "summary": summary}

    return run_job(
        week_id,
        "J7",
        inputs,
        fn,
        force=ctx.force,
        outputs=lambda: [weights_dir(week_id) / "locked.json"],
    )


# J8 -------------------------------------------------------------------------------------------


def weeks_with_weights() -> list[str]:
    from portstudy.paths import outputs_dir

    return sorted(p.parent.parent.name for p in outputs_dir().glob("*/weights/locked.json"))


def build_plans(ctx: RunContext, weeks: list[str]) -> dict[str, list[WeekPlan]]:
    """Weekly plans per portfolio from the stored weights and fallback logs."""
    keys = ctx.portfolio_keys()
    plans: dict[str, list[WeekPlan]] = {k: [] for k in keys}
    for w in weeks:
        for key in keys:
            status = fb.week_status_for(w, key)
            path = weights_dir(w) / f"{key}.csv"
            if status == "no_rebalance" or not path.exists():
                plans[key].append(WeekPlan(w, None, "hold", "no_rebalance"))
                continue
            weights = read_csv(path).set_index("ticker")["weight"].astype(float)
            pid = PortfolioId.parse(key)
            meta_path = weights_dir(w) / f"{key}.meta.json"
            decision_source = (
                read_weights_meta(w, key).get("decision_source", "") if meta_path.exists() else ""
            )
            if decision_source not in ("q0", "control", "llm", "fallback_q0"):
                decision_source = (
                    "q0"
                    if pid.is_q0
                    else "control"
                    if pid.is_control
                    else "fallback_q0"
                    if status == "fallback"
                    else "llm"
                )
            plans[key].append(WeekPlan(w, weights, decision_source, status))  # type: ignore[arg-type]
    return plans


def update_ic_history(ctx: RunContext, weeks: list[str], book: PriceBook) -> int:
    """Score every week whose return window has closed; idempotent upsert into the IC store."""
    led = ledger.read_ledger()
    actions = ledger.read_actions()
    last = book.last_date
    store = ICStore()
    rows: list[dict[str, Any]] = []
    for w in weeks:
        if last is None or cal.window_end_date(w) > last:
            continue
        rets = weekly_returns(led, w, actions).set_index("ticker")["ret"]
        base_path = week_output_dir(w) / "baseline.csv"
        if not base_path.exists():
            continue
        base = read_baseline(w).set_index("ticker")
        rows.append(
            {
                "week_id": w,
                "signal": "baseline",
                "model": "",
                "config_id": "",
                "rank_ic": rank_ic(base["z_b"], rets),
                "n_stocks": int(base["z_b"].notna().sum()),
            }
        )
        scores = read_scores(w)
        for source in ctx.sources():
            sub = scores[
                (scores["model"] == source.model) & (scores["config_id"] == source.config_id)
            ]
            signal = valid_scores(sub).set_index("ticker")["likely"]
            rows.append(
                {
                    "week_id": w,
                    "signal": "llm_raw",
                    "model": source.model,
                    "config_id": source.config_id,
                    "rank_ic": rank_ic(signal, rets) if len(signal) else float("nan"),
                    "n_stocks": len(signal),
                }
            )
            path = posterior_dir(w) / f"{source.key}.csv"
            posterior = read_csv(path).set_index("ticker") if path.exists() else pd.DataFrame()
            z_l = posterior["z_L"].dropna() if "z_L" in posterior else pd.Series(dtype=float)
            tilt = posterior["tilt"].dropna() if "tilt" in posterior else pd.Series(dtype=float)
            rows.extend(
                [
                    {
                        "week_id": w,
                        "signal": "llm_resid",
                        "model": source.model,
                        "config_id": source.config_id,
                        "rank_ic": rank_ic(z_l, rets) if len(z_l) else float("nan"),
                        "n_stocks": len(z_l),
                    },
                    {
                        "week_id": w,
                        "signal": "combined",
                        "model": source.model,
                        "config_id": source.config_id,
                        "rank_ic": rank_ic(tilt, rets) if len(tilt) else float("nan"),
                        "n_stocks": len(tilt),
                    },
                ]
            )
    if rows:
        store.upsert(rows)
    return len(rows)


def refresh_ledger(ctx: RunContext, through: dt.date) -> int:
    """Append raw prices for every ledger ticker up to `through` (never rewriting old days)."""
    led = ledger.read_ledger()
    uni_path = universe_path(ctx.cfg, synthetic=ctx.synthetic)
    tickers = set(led["ticker"].astype(str)) if len(led) else set()
    if uni_path.exists():
        tickers |= set(read_csv(uni_path)["ticker"].astype(str))
    tickers |= set(ctx.cfg.universe.etf_proxies)
    if not tickers:
        return 0
    last = max(led["date"]) if len(led) else through - dt.timedelta(days=7)
    if last >= through:
        return 0
    providers = ctx.providers()
    start = last + dt.timedelta(days=1)
    prices = providers.prices.get_prices(sorted(tickers), start, through)
    actions = providers.prices.get_corporate_actions(sorted(tickers), start, through)
    n = ledger.append_prices(prices, logger=ctx.log("ledger")) if len(prices) else 0
    if len(actions):
        ledger.append_actions(actions)
    return n


def J8_replay_score(ctx: RunContext, week_id: str) -> JobStatus:
    def fn() -> dict[str, Any]:
        # Marks run through the latest close: bring the ledger up to today first.
        today = ctx.now().date()
        n_new = refresh_ledger(ctx, cal.prev_trading_day(today, inclusive=True))
        weeks = [w for w in weeks_with_weights() if w <= week_id]
        if week_id not in weeks and (week_output_dir(week_id) / "fallback_plan.json").exists():
            weeks.append(week_id)
        weeks = sorted(set(weeks))
        book = PriceBook(ledger.read_ledger(), ledger.read_actions())
        hashes = replay_all(ctx.cfg, build_plans(ctx, weeks), book)
        n_ic = update_ic_history(ctx, weeks, book)
        return {
            "weeks": weeks,
            "n_portfolios": len(hashes),
            "ic_rows": n_ic,
            "ledger_rows_added": n_new,
            "ledger_last_date": book.last_date.isoformat() if book.last_date else None,
        }

    ledger_path = ledger_dir() / "prices_raw.csv"
    inputs = stable_hash(
        {
            "ledger": sha256_file(ledger_path) if ledger_path.exists() else None,
            "weights": hash_paths([weights_dir(w) / "locked.json" for w in weeks_with_weights()]),
        }
    )
    return run_job(week_id, "J8", inputs, fn, force=ctx.force)


# J9 -------------------------------------------------------------------------------------------


def J9_backup(ctx: RunContext, week_id: str) -> JobStatus:
    def fn() -> dict[str, Any]:
        from portstudy.backup import run_backup

        path = run_backup(week_id, ctx.cfg, logger=ctx.log("backup"))
        return {"path": str(path)}

    inputs = stable_hash(
        {
            "week": week_id,
            "j8": read_status(week_id, "J8").finished_at,
            "j7": read_status(week_id, "J7").finished_at,
        }
    )
    return run_job(week_id, "J9", inputs, fn, force=ctx.force)


JOBS = {
    "J1": J1_snapshot,
    "J2": J2_baseline,
    "J3": J3_api_submit,
    "J4": J4_open_weight_infer,
    "J5": J5_api_collect,
    "J6": J6_validate,
    "J7": J7_weights,
    "J8": J8_replay_score,
    "J9": J9_backup,
}


def run_named(ctx: RunContext, job: str, week_id: str) -> JobStatus:
    if job not in JOBS:
        raise KeyError(f"unknown job {job}; choose from {sorted(JOBS)}")
    return JOBS[job](ctx, week_id)


def current_week(now: dt.datetime) -> str:
    """The week whose cutoff Friday is the most recent one at or before `now` (ET)."""
    day = now.astimezone(dt.timezone(dt.timedelta(hours=-5))).date() if now.tzinfo else now.date()
    friday = day - dt.timedelta(days=(day.weekday() - 4) % 7)
    return cal.week_id(friday)


def read_weights_lock(week_id: str) -> dict[str, Any] | None:
    path = weights_dir(week_id) / "locked.json"
    return read_json(path) if path.exists() else None


def week_folder(week_id: str) -> Path:
    return week_output_dir(week_id)
