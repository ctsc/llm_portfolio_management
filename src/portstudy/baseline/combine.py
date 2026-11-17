"""Combine momentum and the forecaster into the quant baseline score z_b.

b   = 0.5 * u(m) + 0.5 * u(k)
z_b = (b - mean(b)) / std(b, ddof=1)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from portstudy.baseline.common import complete_tickers, rank_u
from portstudy.baseline.kronos import ForecastProvider, make_forecaster
from portstudy.baseline.momentum import momentum_features
from portstudy.config import StudyConfig
from portstudy.data.snapshot import Snapshot
from portstudy.io import read_csv, read_json, write_csv_atomic, write_json_atomic
from portstudy.paths import week_output_dir

BASELINE_COLUMNS = ["ticker", "M12_1", "M3", "R1", "m", "k", "k_sd", "b", "z_b"]


def combine(momentum: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Join momentum features with forecast values and compute `b` and `z_b`."""
    df = momentum.set_index("ticker").join(forecast.loc[:, ["k", "k_sd"]], how="inner")
    df = df.dropna(subset=["m", "k"]).sort_index()
    if len(df) < 2:
        return pd.DataFrame(columns=BASELINE_COLUMNS)
    df["b"] = 0.5 * rank_u(df["m"]) + 0.5 * rank_u(df["k"])
    sd = df["b"].std(ddof=1)
    df["z_b"] = (df["b"] - df["b"].mean()) / sd if sd > 0 else 0.0
    return df.reset_index().loc[:, BASELINE_COLUMNS]


def ohlcv_by_ticker(snapshot: Snapshot, tickers: list[str], n_days: int) -> dict[str, pd.DataFrame]:
    """DECISION(E4): adjusted OHLCV per ticker (open, high, low scaled by adj_close / close)."""
    p = snapshot.prices[snapshot.prices["ticker"].isin(tickers)].copy()
    ratio = p["adj_close"] / p["close"]
    out: dict[str, pd.DataFrame] = {}
    for tk, g in p.groupby("ticker", sort=True):
        g = g.sort_values("date").tail(n_days)
        r = ratio.loc[g.index]
        frame = pd.DataFrame(
            {
                "open": g["open"] * r,
                "high": g["high"] * r,
                "low": g["low"] * r,
                "close": g["adj_close"],
                "volume": g["volume"],
            }
        )
        frame.index = pd.DatetimeIndex(pd.to_datetime(g["date"]))
        out[str(tk)] = frame
    return out


def build_baseline(
    week_id: str,
    cfg: StudyConfig,
    snapshot: Snapshot,
    forecaster: ForecastProvider | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Compute and save `data/outputs/<week>/baseline.csv`. Idempotent for the same inputs."""
    log = logger or logging.getLogger("portstudy.baseline")
    lb = cfg.baseline.momentum_lookbacks
    wide = snapshot.adj_close_wide()
    tickers = complete_tickers(wide, lb.long)
    mom = momentum_features(wide.loc[:, tickers], lb)
    for tk in mom.excluded:
        log.info(
            "baseline_excluded",
            extra={
                "extra_fields": {"week_id": week_id, "ticker": tk, "reason": "incomplete_history"}
            },
        )
    fc = forecaster or make_forecaster(cfg.baseline)
    kc = cfg.baseline.kronos
    data = ohlcv_by_ticker(snapshot, tickers, kc.context_candles)
    kronos_fallback = False
    reason = ""
    try:
        forecast = fc.forecast(
            data,
            horizon=kc.horizon,
            n_samples=kc.n_samples,
            seed=kc.seed if cfg.baseline.forecaster == "kronos" else cfg.baseline.stub_seed,
        )
    except Exception as exc:
        if cfg.baseline.on_kronos_failure == "hold":
            raise
        # DECISION(K3): the forecaster alone failed; the baseline falls back to the momentum
        # rank. With every k_sd NaN, views/build.py::_baseline_ratio gives ratio 1.0 for every
        # stock, so K1 keeps a neutral spread.
        log.exception(
            "kronos_failed",
            extra={"extra_fields": {"week_id": week_id, "error": str(exc)}},
        )
        kronos_fallback, reason = True, str(exc)
        df = mom.features.set_index("ticker").sort_index()
        df["k"] = float("nan")
        df["k_sd"] = float("nan")
        df["b"] = rank_u(df["m"])
        sd = df["b"].std(ddof=1)
        df["z_b"] = (df["b"] - df["b"].mean()) / sd if sd > 0 else 0.0
        out = df.reset_index().loc[:, BASELINE_COLUMNS]
    else:
        out = combine(mom.features, forecast)
    write_csv_atomic(out, week_output_dir(week_id) / "baseline.csv")
    write_json_atomic(
        {
            "week_id": week_id,
            "forecaster": cfg.baseline.forecaster,
            "kronos_fallback": kronos_fallback,
            "reason": reason,
            "n_stocks": int(len(out)),
        },
        week_output_dir(week_id) / "baseline_meta.json",
    )
    return out


def read_baseline(week_id: str) -> pd.DataFrame:
    return read_csv(week_output_dir(week_id) / "baseline.csv")


def read_baseline_meta(week_id: str) -> dict[str, Any]:
    path = week_output_dir(week_id) / "baseline_meta.json"
    if not path.exists():
        return {"kronos_fallback": False}
    return dict(read_json(path))
