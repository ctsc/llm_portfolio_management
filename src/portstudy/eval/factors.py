"""H5 factor regressions on daily returns (DECISION(G5)).

Factor stand-ins from the ETF proxies in the ledger: market = SPY minus T-bill, size = IWM
minus SPY, value = IWD minus IWF, profitability = QUAL minus SPY; investment is unavailable
and reported as a gap. If an official Fama-French daily file is present at
`eval.fama_french_file`, MKT-RF, SMB, HML, RMW, CMA, and RF are used instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from portstudy.data import ledger
from portstudy.data.snapshot import read_snapshot, snapshot_exists
from portstudy.eval.tests import default_maxlags

ETF_FACTORS = ["MKT_RF", "SMB", "HML", "RMW"]
FF_FACTORS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]


def daily_rf_from_snapshots(weeks: list[str]) -> pd.Series:
    """Daily risk-free rate from the T-bill series stored in the snapshots (percent, annual)."""
    frames = []
    for w in weeks:
        if snapshot_exists(w):
            r = read_snapshot(w).rates
            if len(r):
                frames.append(r)
    if not frames:
        return pd.Series(dtype=float)
    rates = pd.concat(frames).drop_duplicates("date").set_index("date")["rate"].astype(float)
    return (rates / 100.0 / 252.0).sort_index().rename("rf")


def etf_factors(rf: pd.Series) -> pd.DataFrame:
    led = ledger.read_ledger()
    etfs = led[led["ticker"].isin(["SPY", "IWM", "IWD", "IWF", "QUAL"])]
    closes = etfs.pivot(index="date", columns="ticker", values="close").astype(float).sort_index()
    rets = closes.pct_change().dropna(how="all")
    rf_aligned = rf.reindex(rets.index).ffill().fillna(0.0)
    out = pd.DataFrame(index=rets.index)
    out["MKT_RF"] = rets["SPY"] - rf_aligned
    out["SMB"] = rets["IWM"] - rets["SPY"]
    out["HML"] = rets["IWD"] - rets["IWF"]
    out["RMW"] = rets["QUAL"] - rets["SPY"]
    out["RF"] = rf_aligned
    return out.dropna()


def fama_french_factors(path: Path) -> pd.DataFrame:
    """Read the Ken French daily five-factor CSV (percent units) into decimal returns."""
    raw = pd.read_csv(path, skiprows=3)
    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw = raw[raw["date"].astype(str).str.fullmatch(r"\d{8}")]
    raw["date"] = pd.to_datetime(raw["date"].astype(str), format="%Y%m%d").dt.date
    df = raw.set_index("date").astype(float) / 100.0
    df = df.rename(columns={"Mkt-RF": "MKT_RF"})
    return df.loc[:, [*FF_FACTORS, "RF"]]


def regress_portfolio(
    daily_values: pd.DataFrame, factors: pd.DataFrame, names: list[str]
) -> dict[str, float]:
    """Alpha, betas, R squared, HAC errors for one portfolio's daily excess returns."""
    values = daily_values.set_index("date")["value"].astype(float).sort_index()
    rets = values.pct_change().dropna()
    joined = pd.concat([rets.rename("ret"), factors], axis=1, join="inner").dropna()
    out: dict[str, float] = {"n_days": float(len(joined))}
    if len(joined) < 20:
        return out
    y = joined["ret"] - joined["RF"]
    x = sm.add_constant(joined[names])
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": default_maxlags(len(joined))})
    out["alpha_daily"] = float(model.params["const"])
    out["alpha_annual"] = float(model.params["const"] * 252)
    out["alpha_t"] = float(model.tvalues["const"])
    out["alpha_p"] = float(model.pvalues["const"])
    for n in names:
        out[f"beta_{n}"] = float(model.params[n])
        out[f"t_{n}"] = float(model.tvalues[n])
    out["r_squared"] = float(model.rsquared)
    return out


def factor_table(
    daily_by_portfolio: dict[str, pd.DataFrame], weeks: list[str], fama_french_file: str | None
) -> tuple[pd.DataFrame, str]:
    if fama_french_file and Path(fama_french_file).exists():
        factors, names, source = (
            fama_french_factors(Path(fama_french_file)),
            FF_FACTORS,
            "fama_french",
        )
    else:
        factors, names, source = (
            etf_factors(daily_rf_from_snapshots(weeks)),
            ETF_FACTORS,
            "etf_proxies",
        )
    rows = []
    for key, daily in daily_by_portfolio.items():
        if len(daily) == 0:
            continue
        rows.append({"portfolio": key, **regress_portfolio(daily, factors, names)})
    df = pd.DataFrame(rows)
    if source == "etf_proxies" and len(df):
        df["beta_CMA"] = np.nan  # investment factor unavailable from ETF proxies
    return df, source
