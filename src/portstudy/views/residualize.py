"""Residualize the likely rating on the baseline score z_b (DECISION(E2) uses this signal's IC).

s = a + c z_b + e ; z_L = standardize(e) with ddof=1 ; zeros if the residual std < 1e-9.
Invalid outputs are excluded so their stored fallback range is never residualized.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RESID_STD_FLOOR = 1e-9


def valid_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Rows with a valid output for one (model, configuration)."""
    valid = (
        scores["valid"].astype(str).str.lower().eq("true")
        if scores["valid"].dtype == object
        else scores["valid"].astype(bool)
    )
    flag = (
        scores["fallback_flag"]
        if "fallback_flag" in scores.columns
        else pd.Series("", index=scores.index)
    )
    ok = valid & (flag.isna() | (flag.astype(str) == ""))
    return scores.loc[ok, ["ticker", "low", "likely", "high", "width"]].reset_index(drop=True)


def residualize(scores: pd.DataFrame, z_b: pd.Series) -> pd.Series:
    """Cross-sectional OLS of valid scores on z_b with an intercept; standardized residual.

    `scores` must already be filtered to valid outputs (see `valid_scores`). Only tickers
    present in both inputs are used. Returns `z_L` indexed by ticker.
    """
    s = scores.set_index("ticker")["likely"].astype(float)
    common = s.index.intersection(z_b.index)
    s = s.loc[common]
    x = z_b.loc[common].astype(float)
    if len(common) < 3:
        return pd.Series(0.0, index=common, name="z_L")
    design = np.column_stack([np.ones(len(common)), x.to_numpy()])
    coef, *_ = np.linalg.lstsq(design, s.to_numpy(), rcond=None)
    resid = s.to_numpy() - design @ coef
    sd = float(np.std(resid, ddof=1))
    if sd < RESID_STD_FLOOR:
        return pd.Series(0.0, index=common, name="z_L")
    z = (resid - resid.mean()) / sd
    return pd.Series(z, index=common, name="z_L")
