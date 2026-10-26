"""Statistical tests used by the evaluation. Every test takes a seed and is deterministic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests


@dataclass
class MeanTest:
    mean: float
    se: float
    t: float
    p: float
    n: int
    maxlags: int


def default_maxlags(n: int) -> int:
    """floor(4 * (T/100)^(2/9)), at least 1."""
    return max(1, int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


def newey_west_mean(x: pd.Series | np.ndarray, maxlags: int | None = None) -> MeanTest:
    """OLS of x on a constant with HAC (Bartlett) standard errors."""
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(arr)
    if n < 2:
        return MeanTest(
            float(arr.mean()) if n else float("nan"), float("nan"), float("nan"), float("nan"), n, 0
        )
    lags = maxlags if maxlags is not None else default_maxlags(n)
    model = sm.OLS(arr, np.ones((n, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return MeanTest(
        mean=float(model.params[0]),
        se=float(model.bse[0]),
        t=float(model.tvalues[0]),
        p=float(model.pvalues[0]),
        n=n,
        maxlags=lags,
    )


def sign_test(x: pd.Series | np.ndarray) -> tuple[int, int, float]:
    """Two-sided binomial test on the count of positives among the non-zero values."""
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    nonzero = arr[arr != 0]
    k, n = int((nonzero > 0).sum()), len(nonzero)
    if n == 0:
        return 0, 0, float("nan")
    return k, n, float(stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue)


def sign_flip_permutation(x: pd.Series | np.ndarray, n: int = 10000, seed: int = 0) -> float:
    """Two-sided p-value: randomly flip signs and compare the mean to the observed one."""
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(arr) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    observed = abs(arr.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, len(arr)))
    means = np.abs((signs * arr).mean(axis=1))
    return float((1 + np.sum(means >= observed - 1e-15)) / (n + 1))


def holm(pvalues: list[float] | np.ndarray | pd.Series) -> np.ndarray:
    """Holm step-down adjusted p-values (NaN inputs stay NaN and do not count)."""
    arr = np.asarray(pvalues, dtype=float)
    out = np.full(arr.shape, np.nan)
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return out
    _, adjusted, _, _ = multipletests(arr[mask], method="holm")
    out[mask] = adjusted
    return out


def min_detectable_gap(sd: float, n: int) -> float:
    """2.8 * sd / sqrt(T): the gap detectable with 80 percent power at the 5 percent level."""
    if n <= 0 or not np.isfinite(sd):
        return float("nan")
    return 2.8 * sd / math.sqrt(n)


def bootstrap_ci(
    x: pd.Series | np.ndarray, n: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(arr) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def paired_summary(
    gap: pd.Series, tie_band: float, seed: int, n_perm: int = 10000, n_boot: int = 2000
) -> dict[str, float | int]:
    """Win/tie/loss counts, mean gap with Newey-West and bootstrap, sign and permutation tests."""
    g = gap.dropna().astype(float)
    nw = newey_west_mean(g)
    k, n_nonzero, p_sign = sign_test(g)
    lo, hi = bootstrap_ci(g, n=n_boot, seed=seed)
    return {
        "n_weeks": int(len(g)),
        "wins": int((g > tie_band).sum()),
        "ties": int(((g >= -tie_band) & (g <= tie_band)).sum()),
        "losses": int((g < -tie_band).sum()),
        "mean_gap": nw.mean,
        "se_nw": nw.se,
        "t_nw": nw.t,
        "p_nw": nw.p,
        "ci_low": lo,
        "ci_high": hi,
        "p_sign": p_sign,
        "p_perm": sign_flip_permutation(g, n=n_perm, seed=seed),
        "sd_gap": float(g.std(ddof=1)) if len(g) > 1 else float("nan"),
        "min_detectable_gap": min_detectable_gap(float(g.std(ddof=1)), len(g))
        if len(g) > 1
        else float("nan"),
    }


def rank_ic_series(scores: pd.Series, returns: pd.Series) -> float:
    """Spearman rank IC (same rule as views.ic.rank_ic: NaN under 10 stocks or zero variance)."""
    from portstudy.views.ic import rank_ic

    return rank_ic(scores, returns)
