"""Phase 4 tests: Ledoit-Wolf covariance, SPD guard, market weights, prior scaling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from portstudy.config import RiskConfig, StudyConfig
from portstudy.data.snapshot import read_snapshot
from portstudy.risk.covariance import (
    RiskModelError,
    assert_spd,
    build_risk_model,
    daily_returns,
    ledoit_wolf_covariance,
    read_risk_model,
)
from portstudy.risk.prior import build_prior, market_prior, market_weights


def _returns(n: int, t: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 0.01, (t, n)), columns=[f"S{i}" for i in range(n)])


def test_three_stock_toy_matches_sklearn_times_five() -> None:
    rets = _returns(3, 120, seed=1)
    cfg = RiskConfig()
    model = ledoit_wolf_covariance(rets, cfg)
    lw = LedoitWolf().fit(rets.to_numpy())
    np.testing.assert_allclose(model.sigma_matrix.to_numpy(), 5 * lw.covariance_, atol=1e-15)
    assert model.alpha == pytest.approx(lw.shrinkage_)
    np.testing.assert_allclose(model.sigma.to_numpy(), np.sqrt(np.diag(5 * lw.covariance_)))
    assert model.tickers == ["S0", "S1", "S2"]


def test_daily_returns_window() -> None:
    prices = pd.DataFrame(
        {"A": np.linspace(100, 110, 11)}, index=pd.bdate_range("2026-01-01", periods=11)
    )
    r = daily_returns(prices, 5)
    assert len(r) == 5
    assert r["A"].iloc[-1] == pytest.approx(110 / 109 - 1)


def test_more_stocks_than_days_is_still_positive_definite() -> None:
    rets = _returns(300, 252, seed=2)
    model = ledoit_wolf_covariance(rets, RiskConfig())
    eig = np.linalg.eigvalsh(model.sigma_matrix.to_numpy())
    assert eig[0] > 1e-10
    assert 0 < model.alpha < 1


def test_spd_guard_raises() -> None:
    with pytest.raises(RiskModelError):
        assert_spd(np.array([[1.0, 2.0], [2.0, 1.0]]), 1e-10)
    with pytest.raises(RiskModelError):
        assert_spd(np.array([[1.0, 0.0], [0.5, 1.0]]), 1e-10)


def test_market_weights_sum_to_one_and_prior_scales_with_delta() -> None:
    rets = _returns(4, 100, seed=3)
    model = ledoit_wolf_covariance(rets, RiskConfig())
    caps = pd.Series({"S0": 10.0, "S1": 30.0, "S2": np.nan, "S3": 60.0})
    w = market_weights(caps, model.tickers)
    assert w.sum() == pytest.approx(1.0)
    assert w["S2"] == 0.0 and w["S3"] == pytest.approx(0.6)
    pi1 = market_prior(model, w, 1.0)
    pi25 = market_prior(model, w, 2.5)
    np.testing.assert_allclose(pi25.to_numpy(), 2.5 * pi1.to_numpy())
    expected = model.sigma_matrix.to_numpy() @ w.to_numpy()
    np.testing.assert_allclose(pi1.to_numpy(), expected)


def test_risk_and_prior_saved_per_week(
    snapshot_data_dir: Path, study_config: StudyConfig, session_weeks: list[str]
) -> None:
    for w in session_weeks[:3]:
        snap = read_snapshot(w)
        model = build_risk_model(w, study_config.risk, snap)
        prior = build_prior(w, study_config.risk, model, snap.reference)
        back = read_risk_model(w)
        np.testing.assert_allclose(
            back.sigma_matrix.to_numpy(), model.sigma_matrix.to_numpy(), rtol=1e-10
        )
        assert back.tickers == model.tickers
        assert prior["w_mkt"].sum() == pytest.approx(1.0)
        assert list(prior["ticker"]) == model.tickers
        assert (snapshot_data_dir / "outputs" / w / "prior.csv").exists()
