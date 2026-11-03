"""Phase 11 done-when: the report builds with no errors on the synthetic pipeline run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from portstudy.config import ModelRegistry, StudyConfig
from portstudy.data.providers.synthetic import SyntheticWorld, synthetic_bundle
from portstudy.eval.data import load_eval_data
from portstudy.eval.report import run_eval
from portstudy.eval.shuffle import one_shuffle, shuffle_portfolio
from portstudy.llm.prompts import load_manifest
from portstudy.pipeline.context import RunContext
from tests.conftest import PIPELINE_WEEKS, generous_budget


def test_report_builds_on_synthetic_run(pipeline_dir: Path, study_config: StudyConfig,
                                        model_registry: ModelRegistry, world: SyntheticWorld) -> None:
    cfg = generous_budget(study_config)
    ctx = RunContext(cfg=cfg, registry=model_registry, manifest=load_manifest(), synthetic=True,
                     data_providers=synthetic_bundle(world))
    data = load_eval_data(cfg, ctx.portfolio_keys())
    assert len(data.weeks) == PIPELINE_WEEKS
    assert set(data.weekly_returns.columns) == set(ctx.portfolio_keys())
    assert data.statuses.eq("normal").all().all()
    folder = run_eval(ctx, skip_shuffle=False, n_shuffles=3, folder=pipeline_dir / "outputs" / "report")
    expected = ["report.md", "h1_ic_gap.csv", "h2_ic_gap.csv", "h2_portfolio_vs_q0_clean.csv",
                "h2_portfolio_vs_q0_as_traded.csv", "h3_skills_vs_controls.csv", "h4_earnings_week.csv",
                "h5_factors.csv", "h5_shuffle.csv", "fallback_counts.csv", "weekly_ic_long.csv",
                "portfolio_metrics_as_traded.csv", "fig_weekly_ic.png", "fig_h1_gaps.png",
                "fig_equity_curves.png", "fig_shuffle_histograms.png"]
    for name in expected:
        assert (folder / name).exists(), name
    md = (folder / "report.md").read_text(encoding="utf-8")
    assert "Minimum detectable gap" in md and "Fallback counts" in md and "inconclusive" in md
    shuffle = pd.read_csv(folder / "h5_shuffle.csv")
    shuffle_portfolios = set(ctx.portfolio_keys()) - set(cfg.strategies.controls)
    assert set(shuffle["portfolio"]) == shuffle_portfolios
    assert len(shuffle) == len(shuffle_portfolios) and (shuffle["n_shuffles"] == 3).all()
    assert shuffle["p_total_return"].between(0, 1).all()
    assert (shuffle["p_total_return_holm"] >= shuffle["p_total_return"] - 1e-12).all()
    metrics = pd.read_csv(folder / "portfolio_metrics_as_traded.csv")
    assert set(metrics["portfolio"]) == set(ctx.portfolio_keys())
    replayed_metrics = metrics.loc[~metrics["portfolio"].isin(cfg.strategies.controls)]
    assert replayed_metrics["total_cost_usd"].gt(0).all()
    # Rerunning a shuffle hits the cache and gives the identical result.
    weeks = data.weeks
    first = one_shuffle(cfg, "mvo__stub-ow-a__skills_1", weeks, cfg.eval.shuffle.seed, 0)
    again = one_shuffle(cfg, "mvo__stub-ow-a__skills_1", weeks, cfg.eval.shuffle.seed, 0)
    assert first == again
    dist = shuffle_portfolio(cfg, "mvo__Q0", weeks, n_shuffles=2, seed=cfg.eval.shuffle.seed)
    assert len(dist) == 2 and dist["total_return"].notna().all()
