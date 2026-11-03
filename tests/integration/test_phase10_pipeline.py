"""Phase 10 tests: jobs run alone and idempotently, failure injection per job, fallback labels,
lock refusal, and run_weekend.py resume behaviour."""

from __future__ import annotations

import datetime as dt
import sys
from collections.abc import Callable
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from portstudy import calendar as cal
from portstudy.config import ModelRegistry, StudyConfig
from portstudy.data.providers.base import ProviderBundle
from portstudy.data.providers.synthetic import SyntheticWorld, synthetic_bundle
from portstudy.llm.cost import CostCapExceeded
from portstudy.llm.prompts import load_manifest
from portstudy.llm.providers.base import LlmProvider
from portstudy.llm.providers.stub import StubFailures, StubLlmProvider
from portstudy.llm.runner import lock_deadline, read_scores
from portstudy.optimize import read_weights, read_weights_meta
from portstudy.pipeline import fallback as fb
from portstudy.pipeline.context import RunContext, synthetic_clock
from portstudy.pipeline.jobs import LockViolation, build_plans, run_named
from portstudy.pipeline.status import JobFailed, all_statuses, read_status
from portstudy.sim.records import read_daily, read_trades
from portstudy.views.ic import ICStore, trailing_ic
from tests.conftest import PIPELINE_WEEKS
from tests.conftest import generous_budget as _cfg

ET = ZoneInfo("America/New_York")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_weekend  # noqa: E402

ALL_JOBS = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9"]

def make_ctx(cfg: StudyConfig, registry: ModelRegistry, world: SyntheticWorld, week: str,
             llm_providers: dict[str, LlmProvider] | None = None,
             clock: Callable[[], dt.datetime] | None = None, allow_over_budget: bool = False,
             data_providers: ProviderBundle | None = None, force: bool = False) -> RunContext:
    return RunContext(cfg=cfg, registry=registry, manifest=load_manifest(), synthetic=True,
                      allow_over_budget=allow_over_budget, force=force,
                      data_providers=data_providers or synthetic_bundle(world),
                      llm_providers=llm_providers or {}, clock=clock or synthetic_clock(week))


def run_week(ctx: RunContext, week: str, jobs: list[str] = ALL_JOBS) -> dict[str, str]:
    return {j: run_named(ctx, j, week).state for j in jobs}


def test_every_job_runs_alone_and_rerun_is_idempotent(pipeline_dir: Path, study_config: StudyConfig,
                                                     model_registry: ModelRegistry, world: SyntheticWorld,
                                                     inject_week: str) -> None:
    cfg = _cfg(study_config)
    ctx = make_ctx(cfg, model_registry, world, inject_week)
    first = run_week(ctx, inject_week)
    assert all(s == "done" for s in first.values()), first
    scores = read_scores(inject_week)
    n_models, n_conf = len(model_registry.enabled(synthetic=True)), len(load_manifest().enabled())
    assert len(scores) == n_models * n_conf * 100 and not scores.duplicated(["model", "config_id", "ticker"]).any()
    trades_before = {k: len(read_trades(k)) for k in ctx.portfolio_keys()}
    daily_before = read_daily("mvo__Q0")
    # Second pass: deterministic jobs are skipped, forced ones re-run without duplicating anything.
    second = run_week(make_ctx(cfg, model_registry, world, inject_week), inject_week)
    assert second["J1"] == "skipped" and second["J2"] == "skipped" and second["J4"] == "skipped"
    assert second["J7"] == "skipped"
    assert len(read_scores(inject_week)) == len(scores)
    for k, n in trades_before.items():
        assert len(read_trades(k)) == n
    assert not read_daily("mvo__Q0").duplicated(["date", "portfolio_id"]).any()
    assert len(read_daily("mvo__Q0")) == len(daily_before)
    # Status files exist for every job with timestamps and hashes.
    for job, st in all_statuses(inject_week).items():
        assert st.state in ("done", "skipped") and st.started_at and st.inputs_hash, job
    lock = pipeline_dir / "outputs" / inject_week / "weights" / "locked.json"
    assert lock.exists()


def test_j4_crash_then_resume(pipeline_dir: Path, study_config: StudyConfig, model_registry: ModelRegistry,
                              world: SyntheticWorld, inject_week: str) -> None:
    cfg = _cfg(study_config)
    crashing = StubLlmProvider("stub-ow-a", failures=StubFailures(crash_after_calls=50))
    ctx = make_ctx(cfg, model_registry, world, inject_week, llm_providers={"stub-ow-a": crashing})
    run_week(ctx, inject_week, ["J1", "J2", "J3"])
    with pytest.raises(JobFailed):
        run_named(ctx, "J4", inject_week)
    assert read_status(inject_week, "J4").state == "failed"
    assert "StubCrash" in (read_status(inject_week, "J4").error or "")
    healthy = StubLlmProvider("stub-ow-a")
    ctx2 = make_ctx(cfg, model_registry, world, inject_week, llm_providers={"stub-ow-a": healthy})
    assert run_named(ctx2, "J4", inject_week).state == "done"
    assert healthy.calls == 600 - 50
    rest = run_week(ctx2, inject_week, ["J5", "J6", "J7"])
    assert all(s == "done" for s in rest.values())
    assert fb.week_status_for(inject_week, "mvo__stub-ow-a__bare") == "normal"


def test_j5_batch_never_returns_is_a_labeled_fallback(pipeline_dir: Path, study_config: StudyConfig,
                                                      model_registry: ModelRegistry, world: SyntheticWorld,
                                                      inject_week: str) -> None:
    cfg = _cfg(study_config)
    stuck = StubLlmProvider("stub-api-a", failures=StubFailures(batch_never_returns=True))
    ctx = make_ctx(cfg, model_registry, world, inject_week, llm_providers={"stub-api-a": stuck})
    run_week(ctx, inject_week, ["J1", "J2", "J3", "J4"])
    assert run_named(ctx, "J5", inject_week).state == "pending"
    # Before the lock margin J6 refuses to decide.
    with pytest.raises(JobFailed):
        run_named(ctx, "J6", inject_week)
    # At the margin the batch counts as failed (N1) and J7 still writes before the lock.
    margin = lock_deadline(inject_week, cfg) - dt.timedelta(minutes=cfg.pipeline.lock_margin_minutes - 1)
    late = make_ctx(cfg, model_registry, world, inject_week, llm_providers={"stub-api-a": stuck}, clock=lambda: margin)
    assert run_named(late, "J6", inject_week).state == "done"
    plan = fb.read_plan(inject_week)
    assert plan is not None
    for conf in load_manifest().enabled():
        assert plan.status_for(f"mvo__stub-api-a__{conf.id}") == "fallback"
        assert plan.reasons[f"mvo__stub-api-a__{conf.id}"] == "batch_not_ready_by_lock"
        assert plan.status_for(f"mvo__stub-api-b__{conf.id}") == "normal"
    assert run_named(late, "J7", inject_week).state == "done"
    q0 = read_weights(inject_week, "mvo__Q0")
    fbw = read_weights(inject_week, "mvo__stub-api-a__bare")
    pd.testing.assert_frame_equal(q0, fbw)
    meta = read_weights_meta(inject_week, "mvo__stub-api-a__bare")
    assert meta["fallback"] is True and meta["decision_source"] == "fallback_q0"
    assert fb.week_status_for(inject_week, "mvo__stub-api-a__bare") == "fallback"
    assert fb.week_status_for(inject_week, "mvo__stub-api-b__bare") == "normal"
    # Excluded from clean weeks and from the trailing IC, but present in the as-traded records.
    weeks = cal.week_range(study_config.synthetic.first_week, PIPELINE_WEEKS + 1)
    assert fb.clean_weeks("mvo__stub-api-a__bare", weeks) == weeks[:PIPELINE_WEEKS]
    assert fb.fallback_weeks("mvo__stub-api-a__bare") == {inject_week}
    store = ICStore()
    store.upsert([{"week_id": w, "signal": "llm_resid", "model": "stub-api-a", "config_id": "bare",
                   "rank_ic": 0.9 if w == inject_week else 0.1, "n_stocks": 100} for w in cal.week_range(weeks[0], 12)])
    later = cal.week_range(weeks[0], 12)[9]
    with_excl = trailing_ic(store, "llm_resid", later, cfg.views, weeks[0], "stub-api-a", "bare",
                            fallback_weeks=fb.fallback_weeks("mvo__stub-api-a__bare"))
    assert with_excl == pytest.approx(0.1)
    assert run_named(late, "J8", inject_week).state == "done"
    trades = read_trades("mvo__stub-api-a__bare")
    wk = trades[trades["week_id"] == inject_week]
    assert len(wk) > 0 and (wk["decision_source"] == "fallback_q0").all()
    daily = read_daily("mvo__stub-api-a__bare")
    assert (daily.loc[daily["week_id"] == inject_week, "week_status"] == "fallback").all()


def test_j3_cost_cap_blocks_then_allow_over_budget(pipeline_dir: Path, study_config: StudyConfig,
                                                   model_registry: ModelRegistry, world: SyntheticWorld,
                                                   inject_week: str) -> None:
    tight = study_config.model_copy(update={"llm": study_config.llm.model_copy(
        update={"cost_goal_usd": 0.001, "cost_warn_usd": 0.001})})
    ctx = make_ctx(tight, model_registry, world, inject_week)
    run_week(ctx, inject_week, ["J1", "J2"])
    with pytest.raises(JobFailed) as info:
        run_named(ctx, "J3", inject_week)
    assert CostCapExceeded.__name__ in str(info.value)
    assert read_status(inject_week, "J3").state == "failed"
    ok = make_ctx(tight, model_registry, world, inject_week, allow_over_budget=True)
    assert run_named(ok, "J3", inject_week).state == "done"
    assert list((pipeline_dir / "outputs" / inject_week / "llm").glob("batch_*.json"))


def test_invalid_rate_above_threshold_falls_back(pipeline_dir: Path, study_config: StudyConfig,
                                                 model_registry: ModelRegistry, world: SyntheticWorld,
                                                 inject_week: str) -> None:
    cfg = _cfg(study_config)
    noisy = StubLlmProvider("stub-ow-b", failures=StubFailures(invalid_rate=0.5))
    ctx = make_ctx(cfg, model_registry, world, inject_week, llm_providers={"stub-ow-b": noisy})
    states = run_week(ctx, inject_week, ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8"])
    assert all(s == "done" for s in states.values())
    plan = fb.read_plan(inject_week)
    assert plan is not None
    for conf in load_manifest().enabled():
        assert plan.status_for(f"mvo__stub-ow-b__{conf.id}") == "fallback"
        assert "invalid_rate" in plan.reasons[f"mvo__stub-ow-b__{conf.id}"]
        assert plan.status_for(f"mvo__stub-ow-a__{conf.id}") == "normal"
    pd.testing.assert_frame_equal(
        read_weights(inject_week, "mvo__Q0"),
        read_weights(inject_week, "mvo__stub-ow-b__skills_1"),
    )
    log = fb.read_log(inject_week)
    assert sum(1 for e in log if e["status"] == "fallback") == len(load_manifest().enabled())


def test_j1_data_failure_gives_no_rebalance(pipeline_dir: Path, study_config: StudyConfig,
                                            model_registry: ModelRegistry, world: SyntheticWorld,
                                            inject_week: str) -> None:
    cfg = _cfg(study_config)
    bundle = synthetic_bundle(world)

    class Broken(type(bundle.prices)):  # type: ignore[misc]
        def get_prices(self, tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
            raise ConnectionError("provider down")

    broken = ProviderBundle(prices=Broken(world), news=bundle.news, reference=bundle.reference,
                            rates=bundle.rates, members=bundle.members)
    ctx = make_ctx(cfg, model_registry, world, inject_week, data_providers=broken)
    with pytest.raises(JobFailed):
        run_named(ctx, "J1", inject_week)
    with pytest.raises(JobFailed):
        run_named(ctx, "J2", inject_week)
    assert run_named(ctx, "J6", inject_week).state == "done"
    plan = fb.read_plan(inject_week)
    assert plan is not None and plan.no_rebalance and plan.no_rebalance_reason == "baseline_unavailable"
    assert run_named(ctx, "J7", inject_week).state == "done"
    assert fb.read_log(inject_week) == [{"portfolio": "*", "status": "no_rebalance", "reason": "baseline_unavailable"}]
    assert fb.week_status_for(inject_week, "mvo__Q0") == "no_rebalance"
    plans = build_plans(ctx, cal.week_range(study_config.synthetic.first_week, PIPELINE_WEEKS + 1))
    assert plans["mvo__Q0"][-1].weights is None and plans["mvo__Q0"][-1].week_status == "no_rebalance"
    # The replay still runs: positions are held, stops still apply, records are labeled.
    healthy = make_ctx(cfg, model_registry, world, inject_week)
    assert run_named(healthy, "J8", inject_week).state == "done"
    daily = read_daily("mvo__Q0")
    assert (daily.loc[daily["week_id"] == inject_week, "week_status"] == "no_rebalance").all()
    trades = read_trades("mvo__Q0")
    assert not ((trades["week_id"] == inject_week) & (trades["reason"] == "rebalance")).any()


def test_j7_lock_refuses_late_write(pipeline_dir: Path, study_config: StudyConfig, model_registry: ModelRegistry,
                                    world: SyntheticWorld, inject_week: str) -> None:
    cfg = _cfg(study_config)
    ctx = make_ctx(cfg, model_registry, world, inject_week)
    run_week(ctx, inject_week, ["J1", "J2", "J3", "J4", "J5", "J6"])
    late_clock = lock_deadline(inject_week, cfg) + dt.timedelta(minutes=1)
    late = make_ctx(cfg, model_registry, world, inject_week, clock=lambda: late_clock)
    with pytest.raises(JobFailed) as info:
        run_named(late, "J7", inject_week)
    assert LockViolation.__name__ in str(info.value)
    assert not (pipeline_dir / "outputs" / inject_week / "weights" / "locked.json").exists()


def test_run_weekend_resumes_after_each_step(pipeline_dir: Path, study_config: StudyConfig,
                                             model_registry: ModelRegistry, world: SyntheticWorld,
                                             inject_week: str, monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    import portstudy.pipeline.jobs as jobs_mod

    cfg = _cfg(study_config)
    # Point the script at the shared world and a generous budget without touching the files.
    monkeypatch.setattr(run_weekend, "default_context",
                        lambda **kw: make_ctx(cfg, model_registry, world, inject_week, clock=kw.get("clock"),
                                              allow_over_budget=kw.get("allow_over_budget", False)))
    assert run_weekend.main(["--synthetic", "--week", inject_week, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "step 1" in out and "J1: pending" in out

    # One sequential pass: the first run dies in J1, the rerun passes J1 and dies in J2, and so
    # on through J9. Every rerun must resume from the finished work, never redo it.
    victims = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J9"]
    state = {"next": 0}
    real_jobs = dict(jobs_mod.JOBS)

    def wrap(name: str) -> Callable[[RunContext, str], object]:
        def run_or_die(ctx: RunContext, week: str) -> object:
            if state["next"] < len(victims) and victims[state["next"]] == name:
                state["next"] += 1
                raise RuntimeError(f"injected hard stop in {name}")
            return real_jobs[name](ctx, week)
        return run_or_die

    for name in real_jobs:
        monkeypatch.setitem(jobs_mod.JOBS, name, wrap(name))
    for victim in victims:
        rc = run_weekend.main(["--synthetic", "--week", inject_week])
        out = capsys.readouterr().out
        assert rc == 1 and f"{victim} {inject_week} FAILED" in out and "rerun with:" in out, out
        # Everything before the victim is already done and gets skipped on the next pass.
        for earlier in ALL_JOBS[: ALL_JOBS.index(victim)]:
            if earlier in ("J5", "J6", "J8"):
                continue
            assert read_status(inject_week, earlier).state == "done", (victim, earlier)
    rc = run_weekend.main(["--synthetic", "--week", inject_week])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert f"Summary for {inject_week}" in out and "locked_at" in out
    assert all(st.state in ("done", "skipped") for job, st in all_statuses(inject_week).items() if job != "J8")
    n_models, n_conf = len(model_registry.enabled(synthetic=True)), len(load_manifest().enabled())
    assert len(read_scores(inject_week)) == n_models * n_conf * 100
