"""v4 phase 3A: fixed view scale (A8), clean live start (A7) and the end-of-study closeout,
exercised through the real jobs on the shared synthetic pipeline run."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from portstudy import calendar as cal
from portstudy.config import ModelRegistry, StudyConfig
from portstudy.data.providers.synthetic import SyntheticWorld
from portstudy.eval.data import load_eval_data
from portstudy.io import read_json
from portstudy.optimize import read_weights_meta
from portstudy.paths import outputs_dir
from portstudy.pipeline.closeout import CloseoutNotReady, run_closeout
from portstudy.pipeline.context import ET, live_start_week_of
from portstudy.pipeline.jobs import run_named
from portstudy.sim.records import read_daily, read_trades
from portstudy.views.ic import ICStore, trailing_ic
from tests.conftest import PIPELINE_WEEKS, generous_budget
from tests.integration.test_phase10_pipeline import make_ctx, run_week


def _views(cfg: StudyConfig, **update: object) -> StudyConfig:
    return cfg.model_copy(update={"views": cfg.views.model_copy(update=update)})


def _timing(cfg: StudyConfig, **update: object) -> StudyConfig:
    return cfg.model_copy(update={"timing": cfg.timing.model_copy(update=update)})


def _clock_at(day: dt.date) -> dt.datetime:
    return dt.datetime.combine(day, dt.time(20, 0), tzinfo=ET)


def test_fixed_ic_keeps_views_when_stored_ics_are_negative(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
) -> None:
    """DECISION(A8): in fixed mode a negative trailing IC does not drop the LLM or baseline view."""
    cfg = _views(generous_budget(study_config), ic_mode="fixed", fixed_ic=0.02, cold_start_weeks=0)
    ctx = make_ctx(cfg, model_registry, world, inject_week)
    weeks = cal.week_range(study_config.synthetic.first_week, PIPELINE_WEEKS)
    rows: list[dict[str, object]] = []
    for w in weeks:
        rows.append(
            {
                "week_id": w,
                "signal": "baseline",
                "model": "",
                "config_id": "",
                "rank_ic": -0.3,
                "n_stocks": 50,
            }
        )
        for source in ctx.sources():
            rows.append(
                {
                    "week_id": w,
                    "signal": cfg.views.ic_signal_for_llm,
                    "model": source.model,
                    "config_id": source.config_id,
                    "rank_ic": -0.3,
                    "n_stocks": 50,
                }
            )
    ICStore().upsert(rows)
    states = run_week(ctx, inject_week, ["J1", "J2", "J3", "J4", "J5", "J6", "J7"])
    assert states["J7"] == "done"
    q0 = read_weights_meta(inject_week, "mvo__Q0")
    assert q0["ic_mode"] == "fixed" and q0["ic_b"] == pytest.approx(0.02)
    source = ctx.sources()[0]
    meta = read_weights_meta(inject_week, f"mvo__{source.key}")
    assert meta["ic_mode"] == "fixed"
    assert meta["ic_l"] == pytest.approx(0.02)
    assert meta["use_llm_view"] is True


def test_trailing_ic_ignores_weeks_before_the_live_start(
    data_dir: Path, study_config: StudyConfig
) -> None:
    cfg = _views(
        study_config,
        ic_mode="trailing",
        ic_history_start="live_start_week",
        cold_start_weeks=0,
        ic_min_weeks=1,
    )
    weeks = cal.week_range(study_config.synthetic.first_week, 6)
    store = ICStore()
    store.upsert(
        [
            {
                "week_id": w,
                "signal": "baseline",
                "model": "",
                "config_id": "",
                "rank_ic": 0.5 if i < 2 else 0.1,
                "n_stocks": 50,
            }
            for i, w in enumerate(weeks)
        ]
    )
    live = weeks[2]
    got = trailing_ic(store, "baseline", weeks[5], cfg.views, live)
    assert got == pytest.approx(0.1)  # the two pre-live 0.5 rows are ignored
    first = _views(cfg, ic_history_start="first_locked_week")
    assert trailing_ic(store, "baseline", weeks[5], first.views, live) > 0.1


def test_pilot_week_is_never_replayed_or_scored(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
) -> None:
    """DECISION(A7): the pilot week's weights stay on disk but every portfolio starts from cash
    at the first live week's Monday open, and the pilot is absent from the evaluation weeks."""
    weeks = cal.week_range(study_config.synthetic.first_week, PIPELINE_WEEKS)
    pilot, last = weeks[0], weeks[-1]
    cfg = _timing(generous_budget(study_config), pilot_week=pilot)
    assert live_start_week_of(cfg, synthetic=True) == weeks[1]
    ctx = make_ctx(cfg, model_registry, world, last, force=True)
    assert run_named(ctx, "J8", last).state == "done"
    key = "mvo__Q0"
    daily = read_daily(key)
    assert set(daily["week_id"].astype(str)) <= set(weeks[1:])
    assert daily["date"].min() == cal.monday_open_day(weeks[1])
    trades = read_trades(key)
    assert pilot not in set(trades["week_id"].astype(str))
    first_buys = trades[trades["week_id"].astype(str) == weeks[1]]
    assert len(first_buys) and set(first_buys["side"]) == {"buy"}  # started from cash
    data = load_eval_data(cfg, ctx.portfolio_keys(), synthetic=True)
    assert pilot not in data.weeks and weeks[1] in data.weeks


def test_closeout_refuses_early_then_completes(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
) -> None:
    weeks = cal.week_range(study_config.synthetic.first_week, PIPELINE_WEEKS)
    last = weeks[-1]
    cfg = generous_budget(study_config)
    required = cal.window_end_date(last)
    early = make_ctx(cfg, model_registry, world, last, force=True)
    with pytest.raises(CloseoutNotReady, match=required.isoformat()):
        run_closeout(early, last)
    assert not (outputs_dir() / "study_complete.json").exists()

    later = make_ctx(
        cfg,
        model_registry,
        world,
        last,
        force=True,
        clock=lambda: _clock_at(required + dt.timedelta(days=2)),
    )
    out = run_closeout(later, last)
    marker = read_json(out)
    assert marker["last_week"] == last and marker["n_portfolios"] > 0
    data = load_eval_data(cfg, later.portfolio_keys(), synthetic=True)
    assert data.weekly_returns.loc[last].notna().all()
