"""The common ticker set used by the baseline, the covariance model, and the prior.

A stock is in the common universe for a week when it is included in the snapshot and has
a complete adjusted-close history over the longest lookback (252 days plus the cutoff day).
"""

from __future__ import annotations

import pandas as pd

from portstudy.data.snapshot import Snapshot


def common_universe(snapshot: Snapshot, lookback_days: int = 252) -> list[str]:
    wide = snapshot.adj_close_wide()
    return complete_tickers(wide, lookback_days)


def complete_tickers(wide: pd.DataFrame, lookback_days: int) -> list[str]:
    """Tickers with `lookback_days + 1` non-missing, positive adjusted closes at the end."""
    need = lookback_days + 1
    if len(wide) < need:
        return []
    tail = wide.iloc[-need:]
    ok = tail.notna().all(axis=0) & (tail > 0).all(axis=0)
    return sorted(str(t) for t in tail.columns[ok.to_numpy()])


def rank_u(x: pd.Series) -> pd.Series:
    """u(x) = rank(x, method="average") / N, ascending ranks across stocks."""
    return x.rank(method="average") / len(x)
