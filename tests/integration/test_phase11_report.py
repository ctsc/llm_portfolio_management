"""Phase 11 done-when (updated in v4 phase 9): the evaluation data loads and the rewritten report
builds with no errors on the synthetic pipeline run."""

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
    metrics = pd.read_csv(folder / "portfolio_metrics.csv")
    assert set(metrics["portfolio"]) == set(ctx.portfolio_keys())
    assert metrics["total_cost_usd"].gt(0).all()
    assert {"mean_view_shift", "solver_failed_weeks", "fallback_weeks", "sharpe_excess"} <= set(
        metrics.columns
    )
    ranking = pd.read_csv(folder / "h4_strategy_ranking.csv")
    assert set(ranking["strategy"]) <= set(cfg.strategies.enabled)
    assert not set(ranking["strategy"]) & set(cfg.strategies.controls)
