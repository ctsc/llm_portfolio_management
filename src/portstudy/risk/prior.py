"""Market prior from market-cap weights over the study's stocks only (DECISION(E5)).

w_mkt = market_cap / sum(market_cap)
pi    = delta * Sigma @ w_mkt
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portstudy.config import RiskConfig
from portstudy.io import read_csv, write_csv_atomic
from portstudy.paths import week_output_dir
from portstudy.risk.covariance import RiskModel


def market_weights(market_cap: pd.Series, tickers: list[str]) -> pd.Series:
    """DECISION(E5): renormalized market-cap weights over `tickers`. Missing caps get zero."""
    caps = market_cap.reindex(tickers).astype(float).fillna(0.0).clip(lower=0.0)
    total = float(caps.sum())
    if total <= 0:
        raise ValueError("market caps sum to zero; cannot build market weights")
    return (caps / total).rename("w_mkt")


def market_prior(risk: RiskModel, w_mkt: pd.Series, delta: float) -> pd.Series:
    sigma = risk.sigma_matrix.loc[risk.tickers, risk.tickers].to_numpy()
    w = w_mkt.reindex(risk.tickers).to_numpy(dtype=float)
    pi = delta * sigma @ w
    return pd.Series(np.asarray(pi, dtype=float), index=risk.tickers, name="pi")


def build_prior(
    week_id: str, cfg: RiskConfig, risk: RiskModel, reference: pd.DataFrame
) -> pd.DataFrame:
    """Compute and save `data/outputs/<week>/prior.csv` with `ticker, market_cap, w_mkt, pi`."""
    caps = reference.set_index("ticker")["market_cap"]
    w = market_weights(caps, risk.tickers)
    pi = market_prior(risk, w, cfg.delta)
    out = pd.DataFrame(
        {
            "ticker": risk.tickers,
            "market_cap": caps.reindex(risk.tickers).to_numpy(),
            "w_mkt": w.to_numpy(),
            "pi": pi.to_numpy(),
            "sigma": risk.sigma.to_numpy(),
        }
    )
    write_csv_atomic(out, week_output_dir(week_id) / "prior.csv")
    return out


def read_prior(week_id: str) -> pd.DataFrame:
    return read_csv(week_output_dir(week_id) / "prior.csv")
