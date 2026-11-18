"""Stock-week characteristics for the residual IC (DECISION(G9)), built only from saved data.

One row per stock and week, every value as of that week's cutoff (nothing after it is used):
baseline score, 12-1 month momentum and 1-month reversal, log market cap and sector, 63-day
volatility, 252-day beta to SPY, and trailing 252-day loadings on HML, RMW and CMA from a
regression of each stock's daily excess returns on the French factors. Numeric columns are also
z-scored across stocks each week (`<column>_z`).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

CHARACTERISTIC_COLUMNS = [
    "baseline_score",
    "momentum_12_1",
    "reversal_1m",
    "log_market_cap",
    "sector",
    "volatility_63d",
    "beta_252d",
    "loading_hml",
    "loading_rmw",
    "loading_cma",
]
NUMERIC = [c for c in CHARACTERISTIC_COLUMNS if c != "sector"]


def _loadings(
    stock_ret: pd.Series, factors: pd.DataFrame, min_days: int = 60
) -> tuple[float, float, float]:
    joined = pd.concat([stock_ret.rename("r"), factors], axis=1, join="inner").dropna()
    if len(joined) < min_days:
        return float("nan"), float("nan"), float("nan")
    y = joined["r"] - joined["RF"]
    names = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
    x = np.column_stack([np.ones(len(joined)), joined[names].to_numpy(dtype=float)])
    coef, *_ = np.linalg.lstsq(x, y.to_numpy(dtype=float), rcond=None)
    return float(coef[3]), float(coef[4]), float(coef[5])


def week_characteristics(
    cutoff: dt.date,
    adj_close: pd.DataFrame,
    reference: pd.DataFrame,
    baseline: pd.DataFrame,
    factors: pd.DataFrame | None,
    spy_close: pd.Series,
) -> pd.DataFrame:
    """Characteristics for one week from a wide adjusted-close table (dates by tickers)."""
    px = adj_close[adj_close.index <= cutoff].sort_index()
    rets = px.pct_change()
    spy = spy_close[spy_close.index <= cutoff].sort_index().pct_change()
    fac = factors[factors.index <= cutoff] if factors is not None else None
    ref = reference.set_index("ticker")
    base = baseline.set_index("ticker") if len(baseline) else pd.DataFrame()
    rows = []
    for tk in px.columns:
        p = px[tk].dropna()
        r = rets[tk].dropna()
        r252 = r.tail(252)
        mom = float(p.iloc[-22] / p.iloc[-253] - 1.0) if len(p) >= 253 else float("nan")
        rev = float(-(p.iloc[-1] / p.iloc[-22] - 1.0)) if len(p) >= 22 else float("nan")
        joined = pd.concat([r252.rename("s"), spy.rename("m")], axis=1, join="inner").dropna()
        beta = (
            float(np.cov(joined["s"], joined["m"])[0, 1] / joined["m"].var())
            if len(joined) >= 60 and joined["m"].var() > 0
            else float("nan")
        )
        hml, rmw, cma = _loadings(r252, fac) if fac is not None else (float("nan"),) * 3
        cap = ref["market_cap"].get(tk) if "market_cap" in ref else None
        rows.append(
            {
                "ticker": tk,
                "baseline_score": float(base["z_b"].get(tk, np.nan)) if len(base) else np.nan,
                "momentum_12_1": mom,
                "reversal_1m": rev,
                "log_market_cap": float(np.log(cap)) if cap and float(cap) > 0 else np.nan,
                "sector": str(ref["sector"].get(tk, "")) if "sector" in ref else "",
                "volatility_63d": float(r.tail(63).std(ddof=1) * np.sqrt(252))
                if len(r) >= 20
                else np.nan,
                "beta_252d": beta,
                "loading_hml": hml,
                "loading_rmw": rmw,
                "loading_cma": cma,
            }
        )
    out = pd.DataFrame(rows, columns=["ticker", *CHARACTERISTIC_COLUMNS])
    for col in NUMERIC:
        v = out[col].astype(float)
        sd = v.std(ddof=1)
        out[f"{col}_z"] = (v - v.mean()) / sd if sd and np.isfinite(sd) else 0.0
    return out


def characteristics_panel(weeks: list[str], factors: pd.DataFrame | None) -> pd.DataFrame:
    """The stock-week panel for every week with a snapshot and a baseline."""
    from portstudy import calendar as cal
    from portstudy.baseline.combine import read_baseline
    from portstudy.data import ledger
    from portstudy.data.snapshot import read_snapshot, snapshot_exists
    from portstudy.paths import week_output_dir

    led = ledger.read_ledger()
    spy = led[led["ticker"] == "SPY"].set_index("date")["close"].astype(float)
    frames = []
    for w in weeks:
        if not snapshot_exists(w) or not (week_output_dir(w) / "baseline.csv").exists():
            continue
        snap = read_snapshot(w)
        part = week_characteristics(
            cal.cutoff_day(w), snap.adj_close_wide(), snap.reference, read_baseline(w), factors, spy
        )
        part.insert(0, "week_id", w)
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
