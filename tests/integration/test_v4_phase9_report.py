"""v4 phase 9: `portstudy eval --synthetic` writes every section, CSV and figure, the Main results
table has one row per preregistered comparison, and `all_metrics.csv` is complete."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from portstudy.config import StudyConfig, load_preregistration
from portstudy.eval.report import SECTIONS
from portstudy.pipeline import cli

REPO_ROOT = Path(__file__).resolve().parents[2]
CSVS = [
    "main_results",
    "signal_tests",
    "portfolio_tests",
    "view_sensitivity",
    "model_change_weeks",
    "weekly_ic_long",
    "fallback_counts",
    "h3b_width",
    "portfolio_metrics",
    "portfolio_metrics_clean",
    "h4_strategy_ranking",
    "earnings_exposure",
    "tie_band_sensitivity",
    "data_integrity",
    "study_log_summary",
    "all_metrics",
]


def test_eval_cli_writes_the_full_report(
    pipeline_dir: Path, study_config: StudyConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    out = pipeline_dir / "outputs" / "report_v4"
    start = time.perf_counter()
    result = CliRunner().invoke(
        cli.app,
        [
            "eval",
            "--synthetic",
            "--config",
            str(REPO_ROOT / "config" / "study_test.yaml"),
            "--out",
            str(out),
        ],
    )
    elapsed = time.perf_counter() - start
    assert result.exit_code == 0, result.output
    print(f"report runtime {elapsed:.1f} s")
    md = (out / "report.md").read_text(encoding="utf-8")
    for heading in SECTIONS:
        assert heading in md, heading
    assert "shuffle" not in md.lower()
    for name in CSVS:
        assert (out / f"{name}.csv").exists(), name
    for fig in (
        "fig_weekly_ic.png",
        "fig_ic_gaps.png",
        "fig_equity_curves.png",
        "fig_h3b_width.png",
        "fig_h4_ranking.png",
        "fig_factor_loadings.png",
        *[f"fig_active_{s}.png" for s in study_config.strategies.enabled],
    ):
        assert (out / fig).exists(), fig
    assert not list(out.glob("*shuffle*"))
    prereg = load_preregistration(cfg=study_config)
    main = pd.read_csv(out / "main_results.csv")
    assert list(main["comparison_id"]) == [c.id for c in prereg.primary_comparisons]
    allm = pd.read_csv(out / "all_metrics.csv", keep_default_na=False, na_values=[""])
    assert {"source", "steered", "q0", "confidence_only", "control"} <= set(allm["unit_type"])
    nan = allm["estimate"].isna()
    assert (allm.loc[nan, "reason"].astype(str).str.len() > 0).all()
    assert elapsed < 300
