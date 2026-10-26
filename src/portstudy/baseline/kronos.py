"""Price forecasters: the `ForecastProvider` interface, a seeded stub, and the Kronos wrapper.

k_i = mean over sampled paths of P_hat[t+horizon] / P[t] - 1
"""

from __future__ import annotations

import abc
import gc
import hashlib
from collections.abc import Mapping

import numpy as np
import pandas as pd

from portstudy.config import BaselineConfig, KronosConfig

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class ForecastProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def forecast_return(
        self,
        ohlcv_by_ticker: Mapping[str, pd.DataFrame],
        horizon: int = 5,
        n_samples: int = 20,
        seed: int = 0,
    ) -> pd.Series:
        """Expected `horizon`-day return per ticker, indexed by ticker."""


def _ticker_seed(seed: int, ticker: str) -> int:
    digest = hashlib.sha256(f"{seed}:{ticker}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


class StubForecaster(ForecastProvider):
    """Deterministic drift plus seeded noise. Needs no model and runs in milliseconds.

    Drift is the mean daily close-to-close return over the last `drift_days`, scaled to the
    horizon; noise is Gaussian with a per-ticker seed derived from (seed, ticker), so the
    result depends only on the inputs and the seed.
    """

    name = "stub"

    def __init__(self, drift_days: int = 21, noise_daily: float = 0.004) -> None:
        self.drift_days = drift_days
        self.noise_daily = noise_daily

    def forecast_return(
        self,
        ohlcv_by_ticker: Mapping[str, pd.DataFrame],
        horizon: int = 5,
        n_samples: int = 20,
        seed: int = 0,
    ) -> pd.Series:
        out: dict[str, float] = {}
        for ticker in sorted(ohlcv_by_ticker):
            close = ohlcv_by_ticker[ticker]["close"].astype(float).to_numpy()
            if len(close) < self.drift_days + 1 or close[-1] <= 0:
                continue
            rets = close[-self.drift_days - 1 :]
            drift = float(np.mean(rets[1:] / rets[:-1] - 1.0))
            rng = np.random.default_rng(_ticker_seed(seed, ticker))
            # Average of n_samples simulated paths of horizon daily steps.
            steps = rng.normal(drift, self.noise_daily, size=(n_samples, horizon))
            paths = np.prod(1.0 + steps, axis=1)
            out[ticker] = float(paths.mean() - 1.0)
        return pd.Series(out, name="k", dtype=float)


class KronosForecaster(ForecastProvider):
    """Wrapper over the vendored Kronos code (vendor/kronos) and Hugging Face weights.

    # UNVERIFIED: needs real-model smoke test (Phase 17). The vendored API surface
    # (KronosTokenizer, Kronos, KronosPredictor, predict_batch) is taken from the
    # shiyu-coder/Kronos README and must be confirmed against the pinned commit.
    """

    name = "kronos"

    def __init__(self, cfg: KronosConfig, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = device
        self._predictor: object | None = None

    def _load(self) -> object:
        if self._predictor is not None:
            return self._predictor
        import sys

        from portstudy.config import repo_root

        vendor = repo_root() / "vendor" / "kronos"
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore[import-not-found]

        tokenizer = KronosTokenizer.from_pretrained(self.cfg.tokenizer_id)
        model = Kronos.from_pretrained(self.cfg.model_id)
        self._predictor = KronosPredictor(model, tokenizer, device=self.device, max_context=512)
        return self._predictor

    def forecast_return(
        self,
        ohlcv_by_ticker: Mapping[str, pd.DataFrame],
        horizon: int = 5,
        n_samples: int = 20,
        seed: int = 0,
    ) -> pd.Series:
        import torch  # type: ignore[import-not-found]

        predictor = self._load()
        torch.manual_seed(seed)
        np.random.seed(seed)
        out: dict[str, float] = {}
        tickers = sorted(ohlcv_by_ticker)
        frames = []
        for tk in tickers:
            df = ohlcv_by_ticker[tk].iloc[-self.cfg.context_candles :]
            frames.append(df.loc[:, OHLCV_COLUMNS].reset_index(drop=True))
        try:
            # Use the batch method when present, otherwise predict one series at a time.
            x_stamps = [
                pd.Series(pd.to_datetime(ohlcv_by_ticker[tk].index[-self.cfg.context_candles :]))
                for tk in tickers
            ]
            y_stamps = [
                pd.Series(pd.bdate_range(x.iloc[-1], periods=horizon + 1)[1:]) for x in x_stamps
            ]
            if hasattr(predictor, "predict_batch"):
                preds = predictor.predict_batch(  # type: ignore[attr-defined]
                    frames,
                    x_stamps,
                    y_stamps,
                    pred_len=horizon,
                    T=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    sample_count=n_samples,
                    verbose=False,
                )
            else:
                preds = [
                    predictor.predict(  # type: ignore[attr-defined]
                        f,
                        xs,
                        ys,
                        pred_len=horizon,
                        T=self.cfg.temperature,
                        top_p=self.cfg.top_p,
                        sample_count=n_samples,
                        verbose=False,
                    )
                    for f, xs, ys in zip(frames, x_stamps, y_stamps)
                ]
            for tk, f, pred in zip(tickers, frames, preds):
                last = float(f["close"].iloc[-1])
                out[tk] = float(pred["close"].iloc[-1]) / last - 1.0
        finally:
            # Free GPU memory after use.
            self._predictor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return pd.Series(out, name="k", dtype=float)


def make_forecaster(cfg: BaselineConfig) -> ForecastProvider:
    """Provider chosen by config `baseline.forecaster: stub | kronos`."""
    if cfg.forecaster == "stub":
        return StubForecaster()
    return KronosForecaster(cfg.kronos)
