"""Phase 11 done-when: the report builds with no errors on the synthetic pipeline run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from portstudy.config import ModelRegistry, StudyConfig
from portstudy.data.providers.synthetic import SyntheticWorld, synthetic_bundle
from portstudy.eval.data import load_eval_data
from portstudy.eval.report import run_eval
from portstudy.llm.prompts import load_manifest
from portstudy.pipeline.context import RunContext
from tests.conftest import PIPELINE_WEEKS, generous_budget


def test_report_builds_on_synthetic_run(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
) -> None:
    cfg = generous_budget(study_config)
    ctx = RunContext(
        cfg=cfg,
        registry=model_registry,
        manifest=load_manifest(),
        synthetic=True,
        data_providers=synthetic_bundle(world),
    )
    data = load_eval_data(cfg, ctx.portfolio_keys())
    assert len(data.weeks) == PIPELINE_WEEKS
    assert set(data.weekly_returns.columns) == set(ctx.portfolio_keys())
    assert data.statuses.eq("normal").all().all()
    folder = run_eval(ctx, folder=pipeline_dir / "outputs" / "report")
    expected = [
        "report.md",
        "h1_ic_gap.csv",
        "h2_ic_gap.csv",
        "h2_portfolio_vs_q0_clean.csv",
        "h2_portfolio_vs_q0_as_traded.csv",
        "h3_skills_vs_controls.csv",
        "h4_earnings_week.csv",
        "h5_factors.csv",
        "fallback_counts.csv",
        "weekly_ic_long.csv",
        "portfolio_metrics_as_traded.csv",
        "fig_weekly_ic.png",
        "fig_h1_gaps.png",
        "fig_equity_curves.png",
        # v2 phase 10: H3b, H4 ranking, per-strategy equity curves
        "h3b_width.csv",
        "h4_strategy_ranking.csv",
        "fig_h3b_width.png",
        "fig_h4_ranking.png",
        *[f"fig_equity_{s}.png" for s in cfg.strategies.enabled],
    ]
    for name in expected:
        assert (folder / name).exists(), name
    md = (folder / "report.md").read_text(encoding="utf-8")
    assert "Minimum detectable gap" in md and "Fallback counts" in md and "inconclusive" in md
    assert "## Run summary" in md and "G3b:" in md and "K1:" in md and "P1:" in md
    for strategy in cfg.strategies.enabled:
        assert f"## Strategy: {strategy}" in md
    assert "## H3b:" in md and "strategy ranking" in md
    ranking = pd.read_csv(folder / "h4_strategy_ranking.csv")
    assert list(ranking["strategy"]) and set(ranking["strategy"]) <= set(cfg.strategies.enabled)
    assert not set(ranking["strategy"]) & set(cfg.strategies.controls)
    h2 = pd.read_csv(folder / "h2_portfolio_vs_q0_clean.csv")
    assert set(h2["strategy"]) == set(cfg.strategies.enabled)
    metrics_all = pd.read_csv(folder / "portfolio_metrics_as_traded.csv")
    assert {"mean_view_shift", "solver_failed_weeks", "fallback_weeks"} <= set(metrics_all.columns)
    assert set(cfg.strategies.controls) <= set(metrics_all["portfolio"])
    metrics = pd.read_csv(folder / "portfolio_metrics_as_traded.csv")
    assert set(metrics["portfolio"]) == set(ctx.portfolio_keys())
    assert metrics["total_cost_usd"].gt(0).all()
