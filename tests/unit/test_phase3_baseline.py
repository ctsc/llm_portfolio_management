"""Phase 3 tests: momentum, forecaster stub, combination, determinism, planted effect."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portstudy import calendar as cal
from portstudy.baseline.combine import build_baseline, combine, read_baseline
from portstudy.baseline.common import rank_u
from portstudy.baseline.kronos import StubForecaster, make_forecaster
from portstudy.baseline.momentum import momentum_features
from portstudy.config import MomentumLookbacks, StudyConfig
from portstudy.data.providers.synthetic import SyntheticWorld
from portstudy.data.snapshot import read_snapshot


def _toy_prices(n_days: int = 260) -> pd.DataFrame:
    # Three stocks with known straight-line price paths so each ratio is hand-checkable.
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    idx = np.arange(n_days, dtype=float)
    a = 100.0 + idx                      # rises 1 per day
    b = np.full(n_days, 50.0)            # flat
    c = 200.0 - 0.5 * idx                # falls 0.5 per day
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=dates)


def test_momentum_hand_computed() -> None:
    wide = _toy_prices()
    res = momentum_features(wide, MomentumLookbacks())
    f = res.features.set_index("ticker")
    n = len(wide)
    t = n - 1
    # A: P[t-21]/P[t-252] - 1 with P = 100 + idx
    assert f.loc["A", "M12_1"] == pytest.approx((100 + t - 21) / (100 + t - 252) - 1)
    assert f.loc["A", "M3"] == pytest.approx((100 + t) / (100 + t - 63) - 1)
    assert f.loc["A", "R1"] == pytest.approx(-((100 + t) / (100 + t - 21) - 1))
    assert f.loc["B", "M12_1"] == pytest.approx(0.0)
    assert f.loc["B", "R1"] == pytest.approx(0.0)
    assert f.loc["C", "M3"] < 0 and f.loc["C", "R1"] > 0
    # u ranks: A is top on M12_1 and M3 (3/3), bottom on R1 (1/3); m = mean.
    assert f.loc["A", "m"] == pytest.approx((1.0 + 1.0 + 1 / 3) / 3)
    assert res.excluded == []


def test_momentum_excludes_incomplete_history() -> None:
    wide = _toy_prices()
    wide.loc[wide.index[:30], "B"] = np.nan
    res = momentum_features(wide)
    assert res.excluded == ["B"]
    assert list(res.features["ticker"]) == ["A", "C"]


def test_rank_u_handles_ties() -> None:
    u = rank_u(pd.Series([1.0, 2.0, 2.0, 5.0], index=list("wxyz")))
    assert list(u) == pytest.approx([0.25, 0.625, 0.625, 1.0])


def test_combine_zb_is_standardized_and_deterministic() -> None:
    rng = np.random.default_rng(0)
    mom = pd.DataFrame({"ticker": [f"T{i}" for i in range(50)], "M12_1": 0.0, "M3": 0.0, "R1": 0.0,
                        "m": rng.random(50)})
    k = pd.Series(rng.normal(size=50), index=mom["ticker"])
    out = combine(mom, k)
    assert out["z_b"].mean() == pytest.approx(0.0, abs=1e-12)
    assert out["z_b"].std(ddof=1) == pytest.approx(1.0)
    pd.testing.assert_frame_equal(out, combine(mom, k))
    # Tickers missing a forecast are dropped from the baseline.
    out2 = combine(mom, k.drop("T3"))
    assert "T3" not in set(out2["ticker"]) and len(out2) == 49


def test_stub_forecaster_is_seeded() -> None:
    wide = _toy_prices()
    data = {c: pd.DataFrame({"open": wide[c], "high": wide[c], "low": wide[c], "close": wide[c],
                             "volume": 1.0}) for c in wide.columns}
    fc = StubForecaster()
    k1 = fc.forecast_return(data, seed=3)
    k2 = fc.forecast_return(data, seed=3)
    k3 = fc.forecast_return(data, seed=4)
    pd.testing.assert_series_equal(k1, k2)
    assert not np.allclose(k1.to_numpy(), k3.to_numpy())
    assert k1["A"] > k1["C"]  # rising stock has positive drift


def test_make_forecaster_from_config(study_config: StudyConfig) -> None:
    assert make_forecaster(study_config.baseline).name == "stub"


def test_baseline_builds_for_all_weeks_and_is_deterministic(snapshot_data_dir: Path, study_config: StudyConfig,
                                                            session_weeks: list[str],
                                                            world: SyntheticWorld) -> None:
    adj = world.prices[world.prices["ticker"].isin(world.stock_tickers)]
    wide = adj.drop_duplicates(["ticker", "date"]).pivot(index="date", columns="ticker", values="adj_close")
    ics = []
    for w in session_weeks:
        snap = read_snapshot(w)
        first = build_baseline(w, study_config, snap)
        second = build_baseline(w, study_config, snap)
        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(first, read_baseline(w), check_dtype=False)
        assert len(first) >= 90
        assert first["z_b"].mean() == pytest.approx(0.0, abs=1e-9)
        assert set(first["ticker"]) <= set(snap.included_tickers)
        # Rank IC of z_b against the realized Monday-to-Monday return.
        t0, t1 = cal.monday_open_day(w), cal.window_end_date(w)
        ret = (wide.loc[t1] / wide.loc[t0] - 1).rename("r")
        joined = pd.concat([first.set_index("ticker")["z_b"], ret], axis=1).dropna()
        ics.append(joined["z_b"].rank().corr(joined["r"].rank()))
    # The synthetic world plants a weak momentum effect, so the mean IC is positive.
    assert float(np.mean(ics)) > 0.0
