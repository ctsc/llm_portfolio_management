"""Weekly IC tables per signal and portfolio, restricted to clean weeks per portfolio."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from portstudy.types import PortfolioId
from portstudy.views.ic import ICStore

SIGNALS = ["baseline", "llm_raw", "llm_resid", "combined"]


def ic_wide(store: ICStore, signal: str, portfolios: list[str]) -> pd.DataFrame:
    """Weeks by portfolio table of rank ICs for one signal (baseline has a single column)."""
    df = store.read()
    df = df[df["signal"] == signal]
    if signal == "baseline":
        s = df.set_index("week_id")["rank_ic"].astype(float).sort_index()
        return s.to_frame("baseline")
    cols = {}
    for key in portfolios:
        pid = PortfolioId.parse(key)
        sub = df[(df["model"] == pid.model) & (df["config_id"] == pid.config_id)]
        cols[key] = sub.set_index("week_id")["rank_ic"].astype(float)
    return pd.DataFrame(cols).sort_index()


def ic_tables(
    store: ICStore,
    portfolios: list[str],
    clean_weeks: Callable[[str], list[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """One wide table per signal. Rows (weeks) that are not clean for a portfolio become NaN."""
    out: dict[str, pd.DataFrame] = {}
    for signal in SIGNALS:
        wide = ic_wide(store, signal, portfolios)
        if signal != "baseline" and clean_weeks is not None:
            for key in portfolios:
                if key in wide.columns:
                    keep = set(clean_weeks(key))
                    wide.loc[~wide.index.isin(keep), key] = float("nan")
        out[signal] = wide
    return out


def long_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for signal, wide in tables.items():
        for col in wide.columns:
            part = pd.DataFrame(
                {
                    "week_id": wide.index,
                    "signal": signal,
                    "portfolio": str(col),
                    "rank_ic": wide[col].to_numpy(dtype=float),
                }
            )
            rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["week_id", "signal", "portfolio", "rank_ic"])
    out: pd.DataFrame = pd.concat(rows, ignore_index=True)
    return out.loc[:, ["week_id", "signal", "portfolio", "rank_ic"]]
