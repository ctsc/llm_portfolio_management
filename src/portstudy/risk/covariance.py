"""Ledoit-Wolf shrinkage covariance from daily simple returns, scaled to a weekly horizon.

Sigma_daily = (1 - alpha) S + alpha (tr(S)/N) I   (sklearn LedoitWolf, alpha = shrinkage_)
Sigma       = weekly_scale * Sigma_daily
sigma_i     = sqrt(Sigma_ii)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from portstudy.baseline.common import complete_tickers
from portstudy.config import RiskConfig
from portstudy.data.snapshot import Snapshot
from portstudy.io import read_csv, write_csv_atomic, write_json_atomic
from portstudy.paths import week_output_dir


class RiskModelError(ValueError):
    """Raised when the covariance is not symmetric positive definite."""


@dataclass
class RiskModel:
    tickers: list[str]
    sigma_matrix: pd.DataFrame
    sigma: pd.Series
    alpha: float
    n_days: int

    @property
    def n(self) -> int:
        return len(self.tickers)


def daily_returns(adj_close: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Daily simple returns over the last `lookback_days` trading days (lookback_days rows)."""
    tail = adj_close.sort_index().iloc[-(lookback_days + 1) :]
    rets: pd.DataFrame = tail.pct_change().iloc[1:]
    return rets


def ledoit_wolf_covariance(returns: pd.DataFrame, cfg: RiskConfig) -> RiskModel:
    """Fit Ledoit-Wolf on a returns table (days by tickers) and scale to weekly."""
    tickers = [str(c) for c in returns.columns]
    x = returns.to_numpy(dtype=float)
    lw = LedoitWolf(assume_centered=False).fit(x)
    sigma_daily = np.asarray(lw.covariance_, dtype=float)
    sigma = cfg.weekly_scale * sigma_daily
    sigma = 0.5 * (sigma + sigma.T)
    assert_spd(sigma, cfg.min_eigenvalue)
    frame = pd.DataFrame(sigma, index=tickers, columns=tickers)
    return RiskModel(
        tickers=tickers,
        sigma_matrix=frame,
        sigma=pd.Series(np.sqrt(np.diag(sigma)), index=tickers, name="sigma"),
        alpha=float(lw.shrinkage_),
        n_days=len(returns),
    )


def assert_spd(matrix: np.ndarray, min_eigenvalue: float) -> None:
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise RiskModelError("covariance is not symmetric")
    smallest = float(np.linalg.eigvalsh(matrix)[0])
    if smallest <= min_eigenvalue:
        raise RiskModelError(f"covariance smallest eigenvalue {smallest:.3e} <= {min_eigenvalue}")


def build_risk_model(
    week_id: str, cfg: RiskConfig, snapshot: Snapshot, tickers: list[str] | None = None
) -> RiskModel:
    """Fit the weekly covariance for the common universe and save it under the week's outputs."""
    wide = snapshot.adj_close_wide()
    universe = tickers or complete_tickers(wide, cfg.cov_lookback_days)
    rets = daily_returns(wide.loc[:, universe], cfg.cov_lookback_days)
    model = ledoit_wolf_covariance(rets, cfg)
    folder = week_output_dir(week_id)
    write_csv_atomic(model.sigma_matrix, folder / "sigma_matrix.csv", index=True)
    write_csv_atomic(model.sigma.rename_axis("ticker").reset_index(), folder / "sigma.csv")
    write_json_atomic(
        {"week_id": week_id, "alpha": model.alpha, "n_days": model.n_days, "n_tickers": model.n},
        folder / "risk_meta.json",
    )
    return model


def read_risk_model(week_id: str) -> RiskModel:
    folder = week_output_dir(week_id)
    matrix = read_csv(folder / "sigma_matrix.csv", index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    import json

    meta = json.loads((folder / "risk_meta.json").read_text(encoding="utf-8"))
    tickers = [str(t) for t in matrix.index]
    return RiskModel(
        tickers=tickers,
        sigma_matrix=matrix,
        sigma=pd.Series(np.sqrt(np.diag(matrix.to_numpy())), index=tickers, name="sigma"),
        alpha=float(meta["alpha"]),
        n_days=int(meta["n_days"]),
    )
