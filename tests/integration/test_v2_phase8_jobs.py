"""v2 phase 8 tests: J7 writes one file per portfolio, fallbacks copy the strategy's own Q0,
the lock deadline mid-loop labels unfinished sources, IC history has one row per source and
signal, and J7/J8 reruns are byte-identical."""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from portstudy.config import ModelRegistry, StudyConfig
from portstudy.data.providers.synthetic import SyntheticWorld, synthetic_bundle
from portstudy.llm.prompts import load_manifest
from portstudy.llm.providers.base import LlmProvider
from portstudy.llm.providers.stub import StubFailures, StubLlmProvider
from portstudy.llm.runner import lock_deadline
from portstudy.optimize import posterior_dir, read_weights, read_weights_meta, weights_dir
from portstudy.optimize.strategies.common import check_weights
from portstudy.pipeline import fallback as fb
from portstudy.pipeline.context import RunContext, synthetic_clock
from portstudy.pipeline.jobs import run_named
from portstudy.sim.records import read_trades
from portstudy.types import PortfolioId
from portstudy.views.ic import ICStore
from tests.conftest import generous_budget

TO_J7 = ["J1", "J2", "J3", "J4", "J5", "J6", "J7"]


def _ctx(
    cfg: StudyConfig,
    registry: ModelRegistry,
    world: SyntheticWorld,
    week: str,
    llm_providers: dict[str, LlmProvider] | None = None,
    clock: Callable[[], dt.datetime] | None = None,
    force: bool = False,
) -> RunContext:
    return RunContext(
        cfg=cfg,
        registry=registry,
        manifest=load_manifest(),
        synthetic=True,
        force=force,
        data_providers=synthetic_bundle(world),
        llm_providers=llm_providers or {},
        clock=clock or synthetic_clock(week),
    )


def _run(ctx: RunContext, week: str, jobs: list[str]) -> None:
    states = {job: run_named(ctx, job, week).state for job in jobs}
    assert all(state == "done" for state in states.values()), states


def _file_hashes(folder: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.iterdir())
        if p.is_file() and p.name != "locked.json"
    }


def _same_bytes(week: str, left: str, right: str) -> bool:
    folder = weights_dir(week)
    return (folder / f"{left}.csv").read_bytes() == (folder / f"{right}.csv").read_bytes()


def test_j7_writes_every_portfolio_and_each_passes_check_weights(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
) -> None:
    cfg = generous_budget(study_config)
    ctx = _ctx(cfg, model_registry, world, inject_week)
    _run(ctx, inject_week, TO_J7)
    keys = ctx.portfolio_keys()
    n_sources = len(ctx.sources())
    assert len(keys) == (n_sources + 1) * len(cfg.strategies.enabled) + len(cfg.strategies.controls)
    assert sorted(p.stem for p in weights_dir(inject_week).glob("*.csv")) == sorted(keys)
    for key in keys:
        weights = read_weights(inject_week, key).set_index("ticker")["weight"].astype(float)
        check_weights(weights, cfg.optimizer)
        meta = read_weights_meta(inject_week, key)
        pid = PortfolioId.parse(key)
        if pid.is_q0:
            assert meta["decision_source"] == "q0" and meta["view_shift"] == 0.0
        elif pid.is_control:
            assert meta["decision_source"] == "control" and meta["view_shift"] == 0.0
            assert meta["use_llm_view"] is False and meta["n_llm_view"] == 0
        else:
            assert meta["decision_source"] == "llm" and meta["strategy"] == pid.strategy
            assert pid.source is not None and meta["source"] == pid.source.key
            assert meta["view_shift"] >= 0.0
    posterior_files = {p.stem for p in posterior_dir(inject_week).glob("*.csv")}
    assert posterior_files == {"Q0", *(source.key for source in ctx.sources())}
    locked = pd.read_json(weights_dir(inject_week) / "locked.json", typ="series")
    assert len(locked["files"]) == 2 * len(keys)


def test_failed_source_copies_each_strategy_q0(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
) -> None:
    cfg = generous_budget(study_config)
    noisy = StubLlmProvider("stub-ow-b", failures=StubFailures(invalid_rate=0.5))
    ctx = _ctx(cfg, model_registry, world, inject_week, llm_providers={"stub-ow-b": noisy})
    _run(ctx, inject_week, TO_J7)
    failed = [source for source in ctx.sources() if source.model == "stub-ow-b"]
    assert failed
    for source in failed:
        for strategy in cfg.strategies.enabled:
            key = PortfolioId.steered(strategy, source).key
            q0_key = PortfolioId.q0(strategy).key
            assert _same_bytes(inject_week, key, q0_key)
            meta = read_weights_meta(inject_week, key)
            assert meta["decision_source"] == "fallback_q0" and meta["fallback"] is True
            assert meta["view_shift"] == 0.0 and meta["strategy"] == strategy
            assert "invalid_rate" in meta["reason"]
            assert fb.week_status_for(inject_week, key) == "fallback"
    healthy = PortfolioId.steered("mvo", next(s for s in ctx.sources() if s.model == "stub-ow-a"))
    assert read_weights_meta(inject_week, healthy.key)["decision_source"] == "llm"
    log = fb.read_log(inject_week)
    assert sum(1 for e in log if e["status"] == "fallback") == len(failed) * len(
        cfg.strategies.enabled
    )


def test_dropped_llm_view_is_byte_identical_to_the_strategy_q0(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
) -> None:
    base = generous_budget(study_config)
    # A zero cold-start IC drops every LLM view, so each steered portfolio must equal its Q0.
    cfg = base.model_copy(update={"views": base.views.model_copy(update={"cold_start_ic": 0.0})})
    ctx = _ctx(cfg, model_registry, world, inject_week)
    _run(ctx, inject_week, TO_J7)
    for strategy in cfg.strategies.enabled:
        for source in ctx.sources():
            key = PortfolioId.steered(strategy, source).key
            assert _same_bytes(inject_week, key, PortfolioId.q0(strategy).key), key
            meta = read_weights_meta(inject_week, key)
            assert meta["use_llm_view"] is False and meta["decision_source"] == "llm"
            assert meta["view_shift"] == 0.0 and meta["fallback"] is False
            assert fb.week_status_for(inject_week, key) == "normal"


def test_lock_deadline_mid_loop_labels_unfinished_sources(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
) -> None:
    cfg = generous_budget(study_config)
    ctx = _ctx(cfg, model_registry, world, inject_week)
    _run(ctx, inject_week, TO_J7[:-1])
    sources = ctx.sources()
    first, rest = sources[0], sources[1:]
    assert rest
    deadline = lock_deadline(inject_week, cfg)
    last_of_first = weights_dir(inject_week) / (
        f"{PortfolioId.steered(cfg.strategies.enabled[-1], first).key}.csv"
    )

    def clock() -> dt.datetime:
        # The deadline passes once the first source has every strategy written.
        if last_of_first.exists():
            return deadline + dt.timedelta(seconds=1)
        return deadline - dt.timedelta(minutes=1)

    late = _ctx(cfg, model_registry, world, inject_week, clock=clock)
    assert run_named(late, "J7", inject_week).state == "done"
    assert (weights_dir(inject_week) / "locked.json").exists()
    for strategy in cfg.strategies.enabled:
        done_key = PortfolioId.steered(strategy, first).key
        assert read_weights_meta(inject_week, done_key)["decision_source"] == "llm"
        for source in rest:
            key = PortfolioId.steered(strategy, source).key
            meta = read_weights_meta(inject_week, key)
            assert meta["decision_source"] == "fallback_q0" and meta["reason"] == "lock_deadline"
            assert meta["view_shift"] == 0.0
            assert _same_bytes(inject_week, key, PortfolioId.q0(strategy).key)
            assert fb.week_status_for(inject_week, key) == "fallback"
    log = fb.read_log(inject_week)
    assert sum(1 for e in log if e.get("reason") == "lock_deadline") == len(rest) * len(
        cfg.strategies.enabled
    )
    assert sorted(p.stem for p in weights_dir(inject_week).glob("*.csv")) == sorted(
        ctx.portfolio_keys()
    )


def test_ic_history_has_one_row_per_source_and_signal(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
) -> None:
    cfg = generous_budget(study_config)
    ctx = _ctx(cfg, model_registry, world, inject_week)
    _run(ctx, inject_week, [*TO_J7, "J8"])
    history = ICStore().read()
    assert len(history)
    assert not history.duplicated(["week_id", "signal", "model", "config_id"]).any()
    llm_rows = history[history["signal"] != "baseline"]
    per_week = llm_rows.groupby("week_id").size()
    assert (per_week == 3 * len(ctx.sources())).all(), per_week
    assert set(llm_rows["signal"]) == {"llm_raw", "llm_resid", "combined"}
    source_keys = {f"{m}__{c}" for m, c in zip(llm_rows["model"], llm_rows["config_id"])}
    assert source_keys == {source.key for source in ctx.sources()}
    baseline = history[history["signal"] == "baseline"]
    assert not baseline.duplicated("week_id").any()


def test_j7_and_j8_reruns_are_byte_identical(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
) -> None:
    cfg = generous_budget(study_config)
    ctx = _ctx(cfg, model_registry, world, inject_week)
    _run(ctx, inject_week, [*TO_J7, "J8"])
    weights_before = _file_hashes(weights_dir(inject_week))
    posterior_before = _file_hashes(posterior_dir(inject_week))
    trades_before = {key: read_trades(key) for key in ctx.portfolio_keys()}
    forced = _ctx(cfg, model_registry, world, inject_week, force=True)
    _run(forced, inject_week, ["J7", "J8"])
    assert _file_hashes(weights_dir(inject_week)) == weights_before
    assert _file_hashes(posterior_dir(inject_week)) == posterior_before
    for key, before in trades_before.items():
        pd.testing.assert_frame_equal(read_trades(key), before)
