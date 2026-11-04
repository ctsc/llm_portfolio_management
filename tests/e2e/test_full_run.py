"""v2 phase 13: invariants of the 30-week rehearsal (tests/e2e/conftest.py::run_rehearsal), the
injected failure weeks, the eval over the clean weeks, and a 20-shuffle test in 4 shards."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portstudy.config import ModelRegistry, StudyConfig
from portstudy.data.providers.synthetic import SyntheticWorld
from portstudy.data.snapshot import read_snapshot
from portstudy.eval.data import load_eval_data
from portstudy.eval.report import build_report
from portstudy.eval.shuffle import collect_shards, plan_shards, run_shards_local
from portstudy.io import read_csv, read_json
from portstudy.llm import health
from portstudy.llm.runner import read_run_status
from portstudy.optimize import read_weights_meta
from portstudy.optimize.strategies.common import check_weights
from portstudy.pipeline import fallback as fb
from portstudy.pipeline.context import RunContext
from portstudy.pipeline.status import read_status
from portstudy.sim.records import read_trades
from tests.conftest import generous_budget
from tests.e2e.conftest import INJECTIONS, WeekendRunner, rehearsal_registry
from tests.e2e.test_full_run_synthetic import _csv_hashes

pytestmark = pytest.mark.e2e

SOLVER_STRATEGIES = ["mvo", "cvar", "robust", "resampled"]  # Top-K has no solver to fail
DELISTED = "SYN007"  # planted by the synthetic world


@pytest.fixture()
def rehearsal(
    e2e_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    e2e_weeks: list[str],
) -> tuple[Path, StudyConfig, RunContext, dict[str, str]]:
    cfg = generous_budget(study_config)
    ctx = WeekendRunner(cfg, rehearsal_registry(model_registry), world).context(e2e_weeks[-1])
    injected = {kind: e2e_weeks[i] for i, kind in INJECTIONS.items()}
    return e2e_dir, cfg, ctx, injected


def _steered(ctx: RunContext) -> list[str]:
    return ctx.steered_keys()


def test_portfolio_identities_and_weight_files_every_week(
    rehearsal: tuple[Path, StudyConfig, RunContext, dict[str, str]], e2e_weeks: list[str]
) -> None:
    root, cfg, ctx, _ = rehearsal
    keys = ctx.portfolio_keys()
    strategies = list(cfg.strategies.enabled)
    controls = list(cfg.strategies.controls)
    assert len(keys) == len(strategies) * (len(ctx.sources()) + 1) + len(controls)
    for w in e2e_weeks:
        folder = root / "outputs" / w / "weights"
        files = sorted(p.stem for p in folder.glob("*.csv"))
        assert files == sorted(keys), w  # one weight file per portfolio, nothing else
        assert sorted(k for k in files if k.endswith("__Q0")) == sorted(
            f"{s}__Q0" for s in strategies
        )
        lock = read_json(folder / "locked.json")
        assert set(lock["files"]) >= {f"{k}.csv" for k in keys}
        for key in files:
            weights = read_csv(folder / f"{key}.csv").set_index("ticker")["weight"].astype(float)
            check_weights(weights, cfg.optimizer)
            meta = read_weights_meta(w, key)
            if key in controls:
                assert meta["decision_source"] == "control"
                assert meta["use_llm_view"] is False and float(meta["view_shift"]) == 0.0
            elif key.endswith("__Q0"):
                assert meta["decision_source"] == "q0" and float(meta["view_shift"]) == 0.0
            else:
                assert meta["decision_source"] in ("llm", "fallback_q0")


def test_ic_history_has_one_row_per_source_and_signal(
    rehearsal: tuple[Path, StudyConfig, RunContext, dict[str, str]],
) -> None:
    root, _, ctx, _ = rehearsal
    ic = read_csv(root / "outputs" / "ic_history.csv")
    assert "strategy" not in ic.columns and "portfolio" not in ic.columns
    sources = {(s.model, s.config_id) for s in ctx.sources()}
    llm = ic[ic["signal"] != "baseline"]
    for (week, signal), rows in llm.groupby(["week_id", "signal"]):
        assert set(zip(rows["model"], rows["config_id"], strict=True)) == sources, (week, signal)
        assert len(rows) == len(sources), (week, signal)
    assert (ic[ic["signal"] == "baseline"].groupby("week_id").size() == 1).all()


def test_injected_source_failure_week(
    rehearsal: tuple[Path, StudyConfig, RunContext, dict[str, str]],
) -> None:
    _, cfg, ctx, injected = rehearsal
    w = injected["source_failure"]
    for strategy in cfg.strategies.enabled:
        for prompt in ctx.prompt_ids():
            assert fb.week_status_for(w, f"{strategy}__stub-ow-b__{prompt}") == "fallback"
            assert fb.week_status_for(w, f"{strategy}__stub-ow-a__{prompt}") == "normal"
            meta = read_weights_meta(w, f"{strategy}__stub-ow-b__{prompt}")
            assert meta["decision_source"] == "fallback_q0"
            q0 = read_csv(_weights_path(ctx, w, f"{strategy}__Q0")).set_index("ticker")["weight"]
            mine = read_csv(_weights_path(ctx, w, f"{strategy}__stub-ow-b__{prompt}"))
            pd.testing.assert_series_equal(
                mine.set_index("ticker")["weight"], q0, check_names=False
            )
    for control in cfg.strategies.controls:
        assert fb.week_status_for(w, control) == "normal"


def _weights_path(ctx: RunContext, week: str, key: str) -> Path:
    from portstudy.paths import week_output_dir

    return week_output_dir(week) / "weights" / f"{key}.csv"


def test_injected_solver_failure_week(
    rehearsal: tuple[Path, StudyConfig, RunContext, dict[str, str]], e2e_weeks: list[str]
) -> None:
    root, cfg, ctx, injected = rehearsal
    w = injected["solver_failure"]
    previous = e2e_weeks[e2e_weeks.index(w) - 1]
    for strategy in SOLVER_STRATEGIES:
        for key in [
            f"{strategy}__Q0",
            *[k for k in ctx.steered_keys() if k.startswith(f"{strategy}__")],
        ]:
            meta = read_weights_meta(w, key)
            assert str(meta.get("fallback_flag")).startswith("solver_failed"), key
            now = read_csv(root / "outputs" / w / "weights" / f"{key}.csv").set_index("ticker")[
                "weight"
            ]
            before = read_csv(root / "outputs" / previous / "weights" / f"{key}.csv").set_index(
                "ticker"
            )["weight"]
            # Previous weights carried over: identical when the universe is unchanged; otherwise
            # re-normalised and re-capped over this week's universe, with no new names bought.
            if set(now.index) == set(before.index):
                np.testing.assert_allclose(now.to_numpy(), before[now.index].to_numpy(), atol=1e-9)
            else:
                held_now = set(now.index[now > 0])
                held_before = set(before.index[before > 0])
                assert held_now <= held_before, key  # nothing new is bought on a failed solve
    assert not read_weights_meta(w, "topk__Q0").get("fallback_flag")
    for control in cfg.strategies.controls:
        assert not read_weights_meta(w, control).get("fallback_flag")
    data = load_eval_data(cfg, ctx.portfolio_keys(), weeks=e2e_weeks)
    failed = data.solver_failed.loc[w]
    assert failed[[k for k in failed.index if k.split("__")[0] in SOLVER_STRATEGIES]].all()
    assert not failed[
        [k for k in failed.index if k.startswith("topk__") or k in cfg.strategies.controls]
    ].any()


def test_injected_lock_cutoff_week(
    rehearsal: tuple[Path, StudyConfig, RunContext, dict[str, str]],
) -> None:
    root, cfg, ctx, injected = rehearsal
    w = injected["lock_cutoff"]
    lock = read_json(root / "outputs" / w / "weights" / "locked.json")
    locked_at, deadline = (dt.datetime.fromisoformat(lock[k]) for k in ("locked_at", "deadline"))
    assert locked_at > deadline  # J7 kept writing Q0 copies after the clock passed the lock
    reasons = {key: read_weights_meta(w, key).get("reason") for key in ctx.steered_keys()}
    cut = [k for k, r in reasons.items() if r == "lock_deadline"]
    finished = [k for k, r in reasons.items() if r != "lock_deadline"]
    assert cut and finished, reasons
    for key in cut:
        assert fb.week_status_for(w, key) == "fallback"
        strategy = key.split("__")[0]
        q0 = read_csv(root / "outputs" / w / "weights" / f"{strategy}__Q0.csv").set_index("ticker")[
            "weight"
        ]
        mine = read_csv(root / "outputs" / w / "weights" / f"{key}.csv").set_index("ticker")[
            "weight"
        ]
        pd.testing.assert_series_equal(mine, q0, check_names=False)
        trades = read_trades(key)
        assert (trades.loc[trades["week_id"] == w, "decision_source"] == "fallback_q0").all()
    for key in finished:
        assert fb.week_status_for(w, key) == "normal"
    for strategy in cfg.strategies.enabled:
        assert fb.week_status_for(w, f"{strategy}__Q0") == "normal"


def test_injected_gateway_health_failure_recovered_with_fallback_flag(
    rehearsal: tuple[Path, StudyConfig, RunContext, dict[str, str]],
) -> None:
    root, cfg, ctx, injected = rehearsal
    w = injected["gateway_health"]
    report = health.read_health(w)
    assert (
        report is not None and not report.failed
    )  # the rerun's check: stub-ow-a is off the gateway
    assert "stub-ow-a" in report.skipped
    assert [set(p["failed"]) for p in report.previous] == [{"stub-ow-a"}]
    statuses = read_run_status(w)
    for prompt in ctx.prompt_ids():
        assert statuses[f"stub-ow-a__{prompt}"].state == "ok", prompt
        assert statuses[f"stub-ow-a__{prompt}"].reason != health.HEALTH_REASON
    j4 = read_status(w, "J4")
    assert j4.state == "done" and j4.result is not None
    assert j4.result["stub-ow-a"].get("n_calls")
    for strategy in cfg.strategies.enabled:
        for prompt in ctx.prompt_ids():
            key = f"{strategy}__stub-ow-a__{prompt}"
            assert fb.week_status_for(w, key) == "normal", key
            assert read_weights_meta(w, key)["decision_source"] == "llm"
    lock = read_json(root / "outputs" / w / "weights" / "locked.json")
    assert dt.datetime.fromisoformat(lock["locked_at"]) <= dt.datetime.fromisoformat(
        lock["deadline"]
    )


def test_delisted_stock_leaves_the_universe_and_every_book(
    rehearsal: tuple[Path, StudyConfig, RunContext, dict[str, str]], e2e_weeks: list[str]
) -> None:
    _, _, ctx, _ = rehearsal
    present = [DELISTED in read_snapshot(w).included_tickers for w in e2e_weeks]
    assert present[0] and not present[-1]
    first_gone = present.index(False)
    assert not any(present[first_gone:]), "a delisted stock never comes back"
    for key in ctx.portfolio_keys():
        trades = read_trades(key)
        later = trades[trades["week_id"].isin(e2e_weeks[first_gone + 1 :])]
        assert not (later["ticker"] == DELISTED).any(), key


def test_eval_on_clean_weeks_shows_planted_width_and_view_effects(
    rehearsal: tuple[Path, StudyConfig, RunContext, dict[str, str]],
    e2e_weeks: list[str],
    tmp_path: Path,
) -> None:
    root, cfg, ctx, injected = rehearsal
    clean = [w for w in e2e_weeks if w not in injected.values()]
    data = load_eval_data(cfg, ctx.portfolio_keys(), weeks=clean)
    folder = tmp_path / "report"
    build_report(data, folder, skip_shuffle=True)
    assert (folder / "report.md").exists()
    h3b = pd.concat([read_csv(p) for p in folder.glob("h3b*.csv")])
    assert not h3b.empty
    gap = (h3b["mean_ic_narrow"] - h3b["mean_ic_wide"]).astype(float)
    assert gap.mean() > 0, h3b.to_dict("records")
    for strategy in cfg.strategies.enabled:
        steered = [k for k in ctx.steered_keys() if k.startswith(f"{strategy}__")]
        shift = data.view_shift[steered].astype(float).mean().mean()
        assert shift > 0, (strategy, shift)


def test_twenty_shuffles_in_four_shards(
    rehearsal: tuple[Path, StudyConfig, RunContext, dict[str, str]], e2e_weeks: list[str]
) -> None:
    _, cfg, ctx, injected = rehearsal
    clean = [w for w in e2e_weeks if w not in injected.values()]
    portfolio = "mvo__stub-ow-a__bare"
    shards = [s for s in plan_shards([portfolio], 20, 5) if s.portfolio == portfolio]
    assert len(shards) == 4 and [(s.start, s.stop) for s in shards] == [
        (0, 5),
        (5, 10),
        (10, 15),
        (15, 20),
    ]
    paths = run_shards_local(cfg, shards, clean, seed=cfg.eval.shuffle.seed, processes=2)
    assert len(paths) == 4 and all(read_json(p)["complete"] for p in paths)
    collected, skipped = collect_shards(shards)
    assert skipped == [] and len(collected) == 20
    assert sorted(collected["shuffle"]) == list(range(20))
    assert collected["total_return"].notna().all() and collected["sharpe"].notna().all()


def test_output_digest_is_recorded(e2e_dir: Path) -> None:
    """One digest over every output CSV; two runs of the suite must print the same value."""
    hashes = _csv_hashes(e2e_dir)
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    target = os.environ.get("PORTSTUDY_E2E_DIGEST_FILE")
    if target:
        Path(target).write_text(digest + "\n")
    assert len(hashes) > 30 * 40
