"""v4 phase 8: portfolio-level tests on hand-built evaluation data with planted effects."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portstudy.config import StudyConfig, load_preregistration
from portstudy.eval.data import EvalData
from portstudy.eval.portfolio_tests import factor_regression, h2_portfolio, portfolio_tests

REPO_ROOT = Path(__file__).resolve().parents[2]
N_WEEKS = 40


def _data(cfg: StudyConfig, effect: float, active_share: float, seed: int = 0) -> EvalData:
    rng = np.random.default_rng(seed)
    weeks = [f"2027-W{i + 1:02d}" for i in range(N_WEEKS)]
    q0 = rng.normal(0.002, 0.02, N_WEEKS)
    active = rng.random(N_WEEKS) < active_share
    steered = q0 + np.where(active, effect + rng.normal(0, 0.004, N_WEEKS), 0.0)
    off = q0.copy()  # views off every week: identical to Q0
    keys = ["mvo__Q0", "mvo__m1__bare", "mvo__m1__placebo", "EW"]
    weekly = pd.DataFrame(
        {
            "mvo__Q0": q0,
            "mvo__m1__bare": steered,
            "mvo__m1__placebo": off,
            "EW": rng.normal(0.001, 0.02, N_WEEKS),
        },
        index=weeks,
    )
    days = pd.bdate_range("2027-01-04", periods=N_WEEKS * 5)
    daily = {}
    for k in keys:
        r = np.repeat(weekly[k].to_numpy() / 5, 5)
        daily[k] = pd.DataFrame(
            {"date": [d.date() for d in days], "value": 1e5 * np.cumprod(1 + r)}
        )
    trades = {
        k: pd.DataFrame({"trade_value": [5e4], "cost_usd": [50.0], "reason": ["rebalance"]})
        for k in keys
    }
    act = pd.DataFrame({"mvo__m1__bare": active, "mvo__m1__placebo": False}, index=weeks)
    te = pd.DataFrame(
        {"mvo__m1__bare": np.where(active, 0.004, 0.0), "mvo__m1__placebo": 0.0}, index=weeks
    )
    return EvalData(
        cfg=cfg,
        portfolios=["mvo__m1__bare", "mvo__m1__placebo"],
        sources=["m1__bare", "m1__placebo"],
        portfolio_keys=keys,
        weeks=weeks,
        pilot_week=None,
        ic={},
        weekly_returns=weekly,
        daily_values=daily,
        trades=trades,
        statuses=pd.DataFrame("normal", index=weeks, columns=keys),
        spy_weekly=pd.Series(rng.normal(0, 0.02, N_WEEKS), index=weeks),
        spy_vol20=pd.Series(rng.uniform(0.1, 0.3, N_WEEKS), index=weeks),
        active_view=act,
        expected_te=te,
        uncertainty_ratio=pd.DataFrame(0.9, index=weeks, columns=act.columns),
        view_shift=pd.DataFrame(0.1, index=weeks, columns=act.columns),
    )


def _rows(rows: list[dict[str, object]], portfolio: str) -> dict[str, dict[str, object]]:
    return {str(r["analysis"]): r for r in rows if r["unit"] == portfolio}


def test_itt_dilution_property(study_config: StudyConfig) -> None:
    data = _data(study_config, effect=0.004, active_share=0.5, seed=3)
    rows = _rows(h2_portfolio(data), "mvo__m1__bare")
    itt, act = rows["intention-to-treat"], rows["active weeks"]
    assert float(itt["estimate"]) > 0 and float(act["estimate"]) > 0  # type: ignore[arg-type]
    assert float(act["estimate"]) > float(itt["estimate"])  # type: ignore[arg-type]
    assert abs(float(act["t"]) - float(itt["t"])) < 0.5 * abs(float(itt["t"]))  # type: ignore[arg-type]
    assert 0.3 < float(itt["share_active_weeks"]) < 0.7  # type: ignore[arg-type]


def test_views_off_every_week(study_config: StudyConfig) -> None:
    data = _data(study_config, effect=0.004, active_share=0.5)
    itt = _rows(h2_portfolio(data), "mvo__m1__placebo")["intention-to-treat"]
    assert itt["tracking_error"] == 0.0 and itt["estimate"] == 0.0
    assert np.isnan(float(itt["break_even_cost"]))  # type: ignore[arg-type]
    assert itt["break_even_reason"] == "extra turnover vs Q0 is not positive"


def test_planted_momentum_tilt_has_a_mom_loading_and_no_alpha() -> None:
    rng = np.random.default_rng(7)
    days = [dt.date(2027, 1, 4) + dt.timedelta(days=i) for i in range(400)]
    factors = pd.DataFrame(
        {n: rng.normal(0, 0.008, 400) for n in ("MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM")},
        index=days,
    )
    factors["RF"] = 0.0001
    active = 0.6 * factors["MOM"] + rng.normal(0, 0.001, 400)
    res = factor_regression(active, factors)
    assert res["beta_MOM"] == pytest.approx(0.6, abs=0.03) and res["t_MOM"] > 10
    assert abs(res["alpha_annual"]) < 0.02 and res["alpha_p"] > 0.01


def test_tables_families_and_view_sensitivity(study_config: StudyConfig) -> None:
    data = _data(study_config, effect=0.004, active_share=0.5)
    prereg = load_preregistration(REPO_ROOT / "config" / "preregistration_test.yaml")
    prereg = prereg.model_copy(update={"view_sensitivity_min_tracking_error": 0.01})
    out = portfolio_tests(data, prereg)
    tests = out["portfolio_tests"]
    h4 = tests[tests["hypothesis"] == "H4"]
    assert len(h4) and (h4["family"] == "exploratory").all()
    assert np.allclose(h4["p_adj"].dropna(), h4.loc[h4["p_adj"].notna(), "p"])
    assert (tests.loc[tests["hypothesis"] == "H2", "family"] == "secondary").all()
    sens = out["view_sensitivity"].set_index("strategy")
    assert sens.loc["mvo", "label"] == "insensitive to views by construction"  # median TE < 0.01
    assert set(tests.loc[tests["strategy"] == "mvo", "view_sensitivity"]) == {
        "insensitive to views by construction"
    }
    assert {"weekly return minus EW"} <= set(tests["comparison"])


def test_portfolio_tests_run_on_the_synthetic_pipeline(
    pipeline_dir: Path, study_config: StudyConfig
) -> None:
    from portstudy.config import load_model_registry
    from portstudy.eval.data import load_eval_data
    from portstudy.llm.prompts import load_manifest
    from portstudy.pipeline.context import RunContext

    ctx = RunContext(
        cfg=study_config, registry=load_model_registry(), manifest=load_manifest(), synthetic=True
    )
    data = load_eval_data(study_config, ctx.portfolio_keys(), synthetic=True)
    assert set(data.active_view.columns) >= set(data.portfolios)  # recorded for every source
    assert data.conf_only and data.factors is not None
    out = portfolio_tests(data, load_preregistration(cfg=study_config))
    tests = out["portfolio_tests"]
    assert {"H2", "H3", "H5"} <= set(tests["hypothesis"])
    assert (tests["comparison"] == "information: steered minus confidence-only").any()
    assert set(out["view_sensitivity"]["strategy"]) == set(study_config.strategies.enabled)
