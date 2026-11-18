"""Multiple-testing families (DECISION(G7)).

The preregistered primary comparisons are Holm-adjusted together; every other test row is in
one Benjamini-Hochberg secondary family; hypotheses listed as exploratory (H4) keep their raw p.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from portstudy.eval.tests import holm


def fdr_bh(pvalues: list[float] | np.ndarray | pd.Series) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN inputs stay NaN and do not count)."""
    arr = np.asarray(pvalues, dtype=float)
    out = np.full(arr.shape, np.nan)
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return out
    _, adjusted, _, _ = multipletests(arr[mask], method="fdr_bh")
    out[mask] = adjusted
    return out


def assign_families(
    df: pd.DataFrame,
    primary_ids: list[str] | set[str],
    exploratory: list[str] | set[str] | tuple[str, ...] = ("H4",),
    p_col: str = "p",
    id_col: str = "comparison_id",
    hypothesis_col: str = "hypothesis",
    significance: float = 0.05,
) -> pd.DataFrame:
    """Fill `family` (primary, secondary, exploratory), `p_adj` and `significant`."""
    out = df.copy()
    if not len(out):
        for col in ("family", "p_adj", "significant"):
            out[col] = pd.Series(dtype=object)
        return out
    ids = out[id_col].astype(str) if id_col in out else pd.Series("", index=out.index)
    hyp = (
        out[hypothesis_col].astype(str) if hypothesis_col in out else pd.Series("", index=out.index)
    )
    primary = ids.isin(set(primary_ids))
    explore = ~primary & hyp.apply(lambda h: any(h.startswith(e) for e in exploratory))
    out["family"] = np.where(primary, "primary", np.where(explore, "exploratory", "secondary"))
    p = out[p_col].astype(float)
    out["p_adj"] = np.nan
    out.loc[primary, "p_adj"] = holm(p[primary].to_numpy())
    secondary = out["family"] == "secondary"
    out.loc[secondary, "p_adj"] = fdr_bh(p[secondary].to_numpy())
    out.loc[explore, "p_adj"] = p[explore]
    out["significant"] = out["p_adj"] < significance
    return out
