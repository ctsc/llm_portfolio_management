"""Price forecasters: the `ForecastProvider` interface, a seeded stub, and the Kronos wrapper.

k_i = mean over sampled paths of P_hat[t+horizon] / P[t] - 1
"""

from __future__ import annotations

import abc
import gc
import hashlib
import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

from portstudy.config import BaselineConfig, KronosConfig

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class ForecastProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def forecast(
        self,
        ohlcv_by_ticker: Mapping[str, pd.DataFrame],
        horizon: int = 5,
        n_samples: int = 20,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Return `k` and `k_sd` per ticker, indexed by ticker."""


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

    def forecast(
        self,
        ohlcv_by_ticker: Mapping[str, pd.DataFrame],
        horizon: int = 5,
        n_samples: int = 20,
        seed: int = 0,
    ) -> pd.DataFrame:
        out: dict[str, tuple[float, float]] = {}
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
            spread_rng = np.random.default_rng(_ticker_seed(seed + 1, ticker))
            out[ticker] = (
                float(paths.mean() - 1.0),
                float(spread_rng.lognormal(mean=np.log(0.01), sigma=0.5)),
            )
        frame = pd.DataFrame.from_dict(out, orient="index", columns=["k", "k_sd"], dtype=float)
        frame.index.name = "ticker"
        return frame


class KronosForecaster(ForecastProvider):
    """Wrapper over the vendored Kronos code (vendor/kronos) and Hugging Face weights.

    # UNVERIFIED: needs real smoke test (v2 phase 15). The vendored API surface
    # (KronosTokenizer, Kronos, KronosPredictor, predict_batch) is taken from the
    # shiyu-coder/Kronos README and must be confirmed against the pinned commit.
    """

    name = "kronos"
    logger = logging.getLogger("portstudy.baseline.kronos")

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

    @staticmethod
    def _path_return(
        prediction: pd.DataFrame,
        ticker: str,
        sample_index: int,
        last_close: float,
    ) -> float | None:
        if not np.isfinite(last_close):
            reason = "invalid_input_close"
        elif last_close <= 0:
            reason = "non_positive_input_close"
        else:
            reason = None
        if "close" not in prediction.columns:
            raise ValueError("Kronos prediction is missing the close column")
        closes = pd.to_numeric(prediction["close"], errors="coerce").to_numpy(dtype=float)
        if reason is None and (not len(closes) or not np.isfinite(closes).all()):
            reason = "invalid_close"
        elif reason is None and (closes <= 0).any():
            reason = "non_positive_close"
        if reason is not None:
            KronosForecaster.logger.warning(
                "kronos_path_dropped",
                extra={
                    "extra_fields": {
                        "ticker": ticker,
                        "sample_index": sample_index,
                        "reason": reason,
                    }
                },
            )
            return None
        return float(closes[-1] / last_close - 1.0)

    @classmethod
    def _summarize_paths(
        cls, predictions: list[pd.DataFrame], ticker: str, last_close: float
    ) -> tuple[float, float]:
        returns = [
            path_return
            for sample_index, prediction in enumerate(predictions)
            if (path_return := cls._path_return(prediction, ticker, sample_index, last_close))
            is not None
        ]
        k = float(np.mean(returns)) if returns else float("nan")
        k_sd = float(np.std(returns, ddof=1)) if len(returns) >= 2 else float("nan")
        return k, k_sd

    def forecast(
        self,
        ohlcv_by_ticker: Mapping[str, pd.DataFrame],
        horizon: int = 5,
        n_samples: int = 20,
        seed: int = 0,
    ) -> pd.DataFrame:
        tickers = sorted(ohlcv_by_ticker)
        if not tickers:
            return pd.DataFrame(columns=["k", "k_sd"], index=pd.Index([], name="ticker"))
        import torch  # type: ignore[import-not-found]

        predictor = self._load()
        frames = []
        for tk in tickers:
            df = ohlcv_by_ticker[tk].iloc[-self.cfg.context_candles :]
            frames.append(df.loc[:, OHLCV_COLUMNS].reset_index(drop=True))
        x_stamps = [
            pd.Series(pd.to_datetime(ohlcv_by_ticker[tk].index[-self.cfg.context_candles :]))
            for tk in tickers
        ]
        y_stamps = [
            pd.Series(pd.bdate_range(x.iloc[-1], periods=horizon + 1)[1:]) for x in x_stamps
        ]
        paths_by_ticker: dict[str, list[pd.DataFrame]] = {ticker: [] for ticker in tickers}
        try:
            # UNVERIFIED: the pinned predictor may expose individual paths through this adapter.
            path_method = getattr(predictor, "predict_paths", None)
            if callable(path_method):
                self.logger.info(
                    "kronos_sampling_path",
                    extra={
                        "extra_fields": {
                            "sampling_path": "per_path_samples",
                            "n_samples": n_samples,
                            "verification": "UNVERIFIED",
                        }
                    },
                )
                torch.manual_seed(seed)
                np.random.seed(seed)
                for ticker, frame, x_stamp, y_stamp in zip(tickers, frames, x_stamps, y_stamps):
                    sample_paths = path_method(
                        frame,
                        x_stamp,
                        y_stamp,
                        pred_len=horizon,
                        T=self.cfg.temperature,
                        top_p=self.cfg.top_p,
                        sample_count=n_samples,
                        verbose=False,
                    )
                    if not isinstance(sample_paths, (list, tuple)) or not all(
                        isinstance(path, pd.DataFrame) for path in sample_paths
                    ):
                        raise TypeError("Kronos predict_paths must return a sequence of DataFrames")
                    paths_by_ticker[ticker].extend(sample_paths)
            else:
                self.logger.info(
                    "kronos_sampling_path",
                    extra={
                        "extra_fields": {
                            "sampling_path": "repeated_single_sample",
                            "n_samples": n_samples,
                            "verification": "UNVERIFIED",
                        }
                    },
                )
                for sample_index in range(n_samples):
                    sample_seed = seed + sample_index
                    if hasattr(predictor, "predict_batch"):
                        torch.manual_seed(sample_seed)
                        np.random.seed(sample_seed)
                        predictions = predictor.predict_batch(  # type: ignore[attr-defined]
                            frames,
                            x_stamps,
                            y_stamps,
                            pred_len=horizon,
                            T=self.cfg.temperature,
                            top_p=self.cfg.top_p,
                            sample_count=1,
                            verbose=False,
                        )
                        if len(predictions) != len(tickers):
                            raise ValueError(
                                "Kronos predict_batch returned an unexpected ticker count"
                            )
                    else:
                        predictions = []
                        for frame, x_stamp, y_stamp in zip(frames, x_stamps, y_stamps):
                            torch.manual_seed(sample_seed)
                            np.random.seed(sample_seed)
                            predictions.append(
                                predictor.predict(  # type: ignore[attr-defined]
                                    frame,
                                    x_stamp,
                                    y_stamp,
                                    pred_len=horizon,
                                    T=self.cfg.temperature,
                                    top_p=self.cfg.top_p,
                                    sample_count=1,
                                    verbose=False,
                                )
                            )
                    for ticker, prediction in zip(tickers, predictions):
                        paths_by_ticker[ticker].append(prediction)
            out: dict[str, tuple[float, float]] = {}
            for ticker, frame in zip(tickers, frames):
                last_close = float(frame["close"].iloc[-1])
                out[ticker] = self._summarize_paths(paths_by_ticker[ticker], ticker, last_close)
        finally:
            # Free GPU memory after use.
            self._predictor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result = pd.DataFrame.from_dict(out, orient="index", columns=["k", "k_sd"], dtype=float)
        result.index.name = "ticker"
        return result


def make_forecaster(cfg: BaselineConfig) -> ForecastProvider:
    """Provider chosen by config `baseline.forecaster: stub | kronos`."""
    if cfg.forecaster == "stub":
        return StubForecaster()
    return KronosForecaster(cfg.kronos)
