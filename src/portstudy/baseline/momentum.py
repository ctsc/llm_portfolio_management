"""Momentum features from adjusted close.

M12_1 = P[t-21] / P[t-252] - 1
M3    = P[t]    / P[t-63]  - 1
R1    = -(P[t]  / P[t-21]  - 1)
u(x)  = rank(x, method="average") / N
m     = mean(u(M12_1), u(M3), u(R1))

`t` is the cutoff day index (the last row of the wide price table).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from portstudy.baseline.common import complete_tickers, rank_u
from portstudy.config import MomentumLookbacks

MOMENTUM_COLUMNS = ["ticker", "M12_1", "M3", "R1", "m"]


@dataclass
class MomentumResult:
    features: pd.DataFrame
    excluded: list[str] = field(default_factory=list)


def momentum_features(
    adj_close: pd.DataFrame, lookbacks: MomentumLookbacks | None = None
) -> MomentumResult:
    """Compute momentum features from a wide adjusted-close table (dates by tickers).

    Stocks missing any of the history needed are excluded and listed in `excluded`.
    """
    lb = lookbacks or MomentumLookbacks()
    wide = adj_close.sort_index()
    need = lb.long + 1
    keep = complete_tickers(wide, lb.long)
    excluded = sorted(set(str(c) for c in wide.columns) - set(keep))
    if not keep or len(wide) < need:
        return MomentumResult(
            pd.DataFrame(columns=MOMENTUM_COLUMNS), excluded or list(wide.columns)
        )
    p = wide.loc[:, keep]
    p_t = p.iloc[-1]
    m12_1 = p.iloc[-1 - lb.skip] / p.iloc[-1 - lb.long] - 1.0
    m3 = p_t / p.iloc[-1 - lb.medium] - 1.0
    r1 = -(p_t / p.iloc[-1 - lb.short] - 1.0)
    m = (rank_u(m12_1) + rank_u(m3) + rank_u(r1)) / 3.0
    out = pd.DataFrame(
        {
            "ticker": keep,
            "M12_1": m12_1.to_numpy(),
            "M3": m3.to_numpy(),
            "R1": r1.to_numpy(),
            "m": m.to_numpy(),
        }
    )
    return MomentumResult(out.reset_index(drop=True), excluded)
