from __future__ import annotations

import hashlib
import logging
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from portstudy.baseline.combine import combine
from portstudy.baseline.common import rank_u
from portstudy.baseline.kronos import KronosForecaster, StubForecaster
from portstudy.config import KronosConfig
from portstudy.optimize import WeekInputs
from portstudy.risk.covariance import RiskModel


def _ohlcv() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2025-01-01", periods=5)
    close = np.full(5, 100.0)
    return {
        "AAA": pd.DataFrame(
            {
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": np.ones(5),
            },
            index=dates,
        )
    }


def _path(*closes: float) -> pd.DataFrame:
    return pd.DataFrame({"close": closes})


def _patch_sampling_rng(monkeypatch: pytest.MonkeyPatch) -> tuple[list[int], list[int]]:
    torch_seeds: list[int] = []
    numpy_seeds: list[int] = []
    fake_torch = SimpleNamespace(
        manual_seed=torch_seeds.append,
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(np.random, "seed", numpy_seeds.append)
    return torch_seeds, numpy_seeds


def _forecaster(predictor: object) -> KronosForecaster:
    forecaster = KronosForecaster(KronosConfig(context_candles=5), device="cpu")
    forecaster._predictor = predictor
    return forecaster


class _PathPredictor:
    def __init__(self, paths: list[pd.DataFrame]) -> None:
        self.paths = paths

    def predict_paths(
        self,
        frame: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        **kwargs: object,
    ) -> list[pd.DataFrame]:
        assert kwargs["sample_count"] == len(self.paths)
        return self.paths


class _RepeatedPredictor:
    def __init__(self, paths: list[pd.DataFrame]) -> None:
        self.paths = paths
        self.calls = 0

    def predict(
        self,
        frame: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        **kwargs: object,
    ) -> pd.DataFrame:
        assert kwargs["sample_count"] == 1
        prediction = self.paths[self.calls]
        self.calls += 1
        return prediction


class _BatchRepeatedPredictor:
    def __init__(self, paths: list[pd.DataFrame]) -> None:
        self.paths = paths
        self.calls = 0

    def predict_batch(
        self,
        frames: list[pd.DataFrame],
        x_timestamps: list[pd.Series],
        y_timestamps: list[pd.Series],
        **kwargs: object,
    ) -> list[pd.DataFrame]:
        assert len(frames) == 1
        assert kwargs["sample_count"] == 1
        prediction = self.paths[self.calls]
        self.calls += 1
        return [prediction]


def test_stub_spread_is_seeded_and_k_matches_v1_paths() -> None:
    dates = pd.bdate_range("2025-01-01", periods=260)
    closes = {
        "AAA": np.linspace(80.0, 120.0, len(dates)),
        "BBB": np.linspace(125.0, 100.0, len(dates)),
    }
    data = {
        ticker: pd.DataFrame(
            {
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": np.ones(len(dates)),
            },
            index=dates,
        )
        for ticker, close in closes.items()
    }
    forecaster = StubForecaster()
    actual = forecaster.forecast(data, horizon=5, n_samples=20, seed=7)
    repeated = forecaster.forecast(data, horizon=5, n_samples=20, seed=7)
    different_seed = forecaster.forecast(data, horizon=5, n_samples=20, seed=8)
    pd.testing.assert_frame_equal(actual, repeated)
    assert not np.array_equal(actual["k_sd"].to_numpy(), different_seed["k_sd"].to_numpy())
    assert (actual["k_sd"] > 0).all()

    expected_k: dict[str, float] = {}
    for ticker, close in closes.items():
        digest = hashlib.sha256(f"7:{ticker}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        returns = close[-22:]
        drift = float(np.mean(returns[1:] / returns[:-1] - 1.0))
        steps = rng.normal(drift, 0.004, size=(20, 5))
        expected_k[ticker] = float(np.prod(1.0 + steps, axis=1).mean() - 1.0)
    np.testing.assert_array_equal(actual.loc[list(expected_k), "k"], list(expected_k.values()))


def test_kronos_path_samples_and_repeated_calls_match(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    samples = [_path(105.0, 106.0), _path(110.0, 111.0), _path(95.0, 94.0)]
    path_seeds, path_numpy_seeds = _patch_sampling_rng(monkeypatch)
    with caplog.at_level(logging.INFO, logger="portstudy.baseline.kronos"):
        path_result = _forecaster(_PathPredictor(samples)).forecast(
            _ohlcv(), horizon=2, n_samples=3, seed=13
        )
    assert path_seeds == [13] and path_numpy_seeds == [13]
    path_logs = [record for record in caplog.records if record.message == "kronos_sampling_path"]
    assert len(path_logs) == 1
    assert path_logs[0].extra_fields["verification"] == "UNVERIFIED"

    caplog.clear()
    repeated_predictor = _RepeatedPredictor(samples)
    repeated_seeds, repeated_numpy_seeds = _patch_sampling_rng(monkeypatch)
    with caplog.at_level(logging.INFO, logger="portstudy.baseline.kronos"):
        repeated_result = _forecaster(repeated_predictor).forecast(
            _ohlcv(), horizon=2, n_samples=3, seed=13
        )
    assert repeated_predictor.calls == 3
    assert repeated_seeds == repeated_numpy_seeds == [13, 14, 15]
    assert (
        len([record for record in caplog.records if record.message == "kronos_sampling_path"]) == 1
    )
    pd.testing.assert_frame_equal(path_result, repeated_result)

    caplog.clear()
    batch_predictor = _BatchRepeatedPredictor(samples)
    batch_seeds, batch_numpy_seeds = _patch_sampling_rng(monkeypatch)
    with caplog.at_level(logging.INFO, logger="portstudy.baseline.kronos"):
        batch_result = _forecaster(batch_predictor).forecast(
            _ohlcv(), horizon=2, n_samples=3, seed=13
        )
    assert batch_predictor.calls == 3
    assert batch_seeds == batch_numpy_seeds == [13, 14, 15]
    assert (
        len([record for record in caplog.records if record.message == "kronos_sampling_path"]) == 1
    )
    pd.testing.assert_frame_equal(path_result, batch_result)

    returns = np.array([0.06, 0.11, -0.06])
    assert path_result.loc["AAA", "k"] == pytest.approx(returns.mean())
    assert path_result.loc["AAA", "k_sd"] == pytest.approx(returns.std(ddof=1))


def test_kronos_drops_nonpositive_paths_and_keeps_nan_spread(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    paths = [_path(105.0), _path(0.0, 110.0), _path(-1.0, 95.0)]
    predictor = _RepeatedPredictor(paths)
    _patch_sampling_rng(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="portstudy.baseline.kronos"):
        result = _forecaster(predictor).forecast(_ohlcv(), horizon=2, n_samples=3, seed=4)
    assert result.loc["AAA", "k"] == pytest.approx(0.05)
    assert np.isnan(result.loc["AAA", "k_sd"])
    assert (
        len([record for record in caplog.records if record.message == "kronos_path_dropped"]) == 2
    )


def test_combine_preserves_v1_zb_bytes_and_keeps_nan_spreads() -> None:
    rng = np.random.default_rng(12)
    tickers = [f"T{i}" for i in range(20)]
    momentum = pd.DataFrame(
        {
            "ticker": tickers,
            "M12_1": rng.normal(size=20),
            "M3": rng.normal(size=20),
            "R1": rng.normal(size=20),
            "m": rng.random(20),
        }
    )
    forecast = pd.DataFrame(
        {"k": rng.normal(size=20), "k_sd": rng.lognormal(size=20)},
        index=pd.Index(tickers, name="ticker"),
    )
    forecast.loc["T3", "k_sd"] = np.nan
    actual = combine(momentum, forecast)

    legacy = momentum.set_index("ticker").join(forecast["k"].rename("k"), how="inner")
    legacy = legacy.dropna(subset=["m", "k"]).sort_index()
    legacy["b"] = 0.5 * rank_u(legacy["m"]) + 0.5 * rank_u(legacy["k"])
    legacy_sd = legacy["b"].std(ddof=1)
    legacy["z_b"] = (legacy["b"] - legacy["b"].mean()) / legacy_sd if legacy_sd > 0 else 0.0
    expected = legacy.reset_index()["z_b"].to_numpy(dtype=np.float64)
    np.testing.assert_array_equal(actual["z_b"].to_numpy(dtype=np.float64), expected)
    assert actual.loc[actual["ticker"] == "T3", "k_sd"].isna().all()


def test_week_inputs_exposes_k_sd() -> None:
    tickers = ["AAA", "BBB"]
    baseline = pd.DataFrame({"k_sd": [0.01, np.nan]}, index=pd.Index(tickers, name="ticker"))
    risk = RiskModel(
        tickers=tickers,
        sigma_matrix=pd.DataFrame(index=tickers, columns=tickers, dtype=float),
        sigma=pd.Series(dtype=float),
        alpha=0.0,
        n_days=0,
    )
    inputs = WeekInputs(
        week_id="2026-W02",
        baseline=baseline,
        prior=pd.DataFrame(index=tickers),
        risk=risk,
        scores=pd.DataFrame(),
    )
    pd.testing.assert_series_equal(
        inputs.k_sd(),
        pd.Series([0.01, np.nan], index=pd.Index(tickers, name="ticker"), name="k_sd"),
    )
