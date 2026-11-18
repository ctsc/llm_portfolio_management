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


def mean_test(x: pd.Series | np.ndarray, maxlags: int | None = None) -> dict[str, float]:
    """DECISION(G12): mean with Newey-West and plain standard errors; the larger one gives the
    t-statistic, and the p-value uses Student-t with T - 1 degrees of freedom."""
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(arr)
    out = {"n": float(n), "mean": float(arr.mean()) if n else float("nan")}
    if n < 2:
        return {
            **out,
            "se_nw": float("nan"),
            "se_plain": float("nan"),
            "se": float("nan"),
            "t": float("nan"),
            "p": float("nan"),
            "df": float("nan"),
        }
    nw = newey_west_mean(arr, maxlags)
    se_plain = float(arr.std(ddof=1) / math.sqrt(n))
    se = max(nw.se, se_plain) if np.isfinite(nw.se) else se_plain
    tstat = out["mean"] / se if se > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(tstat), n - 1)) if np.isfinite(tstat) else float("nan")
    return {
        **out,
        "se_nw": nw.se,
        "se_plain": se_plain,
        "se": se,
        "t": tstat,
        "p": p,
        "df": float(n - 1),
    }


def t_critical(n: int, level: float = 0.95) -> float:
    """Two-sided Student-t critical value with n - 1 degrees of freedom."""
    return float(stats.t.ppf(1 - (1 - level) / 2, n - 1)) if n > 1 else float("nan")


EXACT_CHUNK = 1 << 16


def sign_flip_p(
    x: pd.Series | np.ndarray, exact_max_t: int = 24, n: int = 10000, seed: int = 0
) -> tuple[float, str]:
    """Two-sided sign-flip p-value of the mean: exact enumeration of all 2^T patterns when
    T <= exact_max_t (in chunks, bounded memory), Monte Carlo otherwise."""
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    t = len(arr)
    if t == 0:
        return float("nan"), "none"
    observed = abs(arr.mean()) - 1e-12 * (1 + abs(arr.mean()))
    if t <= exact_max_t:
        total = 1 << t
        bits = np.arange(t, dtype=np.int64)
        count = 0
        for start in range(0, total, EXACT_CHUNK):
            idx = np.arange(start, min(total, start + EXACT_CHUNK), dtype=np.int64)
            signs = ((idx[:, None] >> bits) & 1) * 2.0 - 1.0
            count += int(np.sum(np.abs(signs @ arr) / t >= observed))
        return count / total, "exact"
    return sign_flip_permutation(arr, n=n, seed=seed), "monte_carlo"


def stationary_bootstrap_indices(
    t: int, n: int, block_mean: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap: blocks of geometric length with mean `block_mean`."""
    p = 1.0 / max(block_mean, 1.0)
    idx = np.empty((n, t), dtype=np.int64)
    idx[:, 0] = rng.integers(0, t, n)
    jump = rng.random((n, t)) < p
    fresh = rng.integers(0, t, (n, t))
    for k in range(1, t):
        idx[:, k] = np.where(jump[:, k], fresh[:, k], (idx[:, k - 1] + 1) % t)
    return idx


def block_bootstrap_ci(
    x: pd.Series | np.ndarray,
    n: int = 2000,
    seed: int = 0,
    block_mean: float | None = None,
    level: float = 0.95,
) -> tuple[float, float]:
    """Stationary block bootstrap percentile-t interval for the mean (block mean T^(1/3) by
    default)."""
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    t = len(arr)
    if t < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = block_mean if block_mean is not None else t ** (1.0 / 3.0)
    sample = arr[stationary_bootstrap_indices(t, n, b, rng)]
    mean, se = arr.mean(), arr.std(ddof=1) / math.sqrt(t)
    if se == 0:
        return float(mean), float(mean)
    se_star = sample.std(axis=1, ddof=1) / math.sqrt(t)
    ok = se_star > 0
    t_star = (sample.mean(axis=1)[ok] - mean) / se_star[ok]
    lo_q, hi_q = np.quantile(t_star, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(mean - hi_q * se), float(mean - lo_q * se)


def paired_summary(
    gap: pd.Series,
    tie_band: float,
    seed: int,
    n_perm: int = 10000,
    n_boot: int = 2000,
    exact_max_t: int = 24,
    block_mean: float | None = None,
) -> dict[str, float | int | str]:
    """DECISION(G12) summary of a weekly gap: win/tie/loss, mean with Newey-West and Student-t,
    stationary block bootstrap percentile-t CI, sign test, sign-flip p (exact up to
    `exact_max_t` weeks), minimum detectable gap. `p_perm` and `p_nw` keep their old names."""
    g = gap.dropna().astype(float)
    mt = mean_test(g)
    k, n_nonzero, p_sign = sign_test(g)
    lo, hi = block_bootstrap_ci(g, n=n_boot, seed=seed, block_mean=block_mean)
    p_flip, method = sign_flip_p(g, exact_max_t=exact_max_t, n=n_perm, seed=seed)
    sd = float(g.std(ddof=1)) if len(g) > 1 else float("nan")
    mdg = min_detectable_gap(sd, len(g)) if len(g) > 1 else float("nan")
    return {
        "n": int(len(g)),
        "n_weeks": int(len(g)),
        "wins": int((g > tie_band).sum()),
        "ties": int(((g >= -tie_band) & (g <= tie_band)).sum()),
        "losses": int((g < -tie_band).sum()),
        "mean_gap": mt["mean"],
        "se_nw": mt["se_nw"],
        "se_plain": mt["se_plain"],
        "t": mt["t"],
        "t_nw": mt["t"],
        "p": mt["p"],
        "p_nw": mt["p"],
        "ci_low": lo,
        "ci_high": hi,
        "p_sign": p_sign,
        "p_signflip": p_flip,
        "p_perm": p_flip,
        "signflip_method": method,
        "sd_gap": sd,
        "mdg": mdg,
        "min_detectable_gap": mdg,
    }


# Sharpe ratio tests ------------------------------------------------------------------------


def _sharpe_moments(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Difference of per-period Sharpe ratios, its gradient and the moment series (LW 2008)."""
    mu1, mu2 = x.mean(), y.mean()
    g1, g2 = (x**2).mean(), (y**2).mean()
    v1, v2 = g1 - mu1**2, g2 - mu2**2
    diff = mu1 / math.sqrt(v1) - mu2 / math.sqrt(v2)
    grad = np.array([g1 / v1**1.5, -g2 / v2**1.5, -mu1 / (2 * v1**1.5), mu2 / (2 * v2**1.5)])
    series = np.column_stack([x - mu1, y - mu2, x**2 - g1, y**2 - g2])
    return float(diff), grad, series


def _hac_cov(series: np.ndarray, lags: int) -> np.ndarray:
    t = len(series)
    cov: np.ndarray = np.asarray(series.T @ series / t, dtype=float)
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = series[lag:].T @ series[:-lag] / t
        cov += w * (gamma + gamma.T)
    return cov


def _sharpe_diff_se(x: np.ndarray, y: np.ndarray, lags: int) -> tuple[float, float]:
    diff, grad, series = _sharpe_moments(x, y)
    var = float(grad @ _hac_cov(series, lags) @ grad) / len(x)
    return diff, math.sqrt(max(var, 0.0))


def ledoit_wolf_sharpe_test(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    block_size: int | None = None,
    level: float = 0.95,
) -> dict[str, float]:
    """Ledoit and Wolf (2008): studentized circular block bootstrap test of the difference of two
    Sharpe ratios on paired (excess) returns. Per-period Sharpe difference, HAC standard error,
    two-sided p-value and a symmetric percentile-t CI."""
    pair = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna().to_numpy(dtype=float)
    t = len(pair)
    nan = {
        "sharpe_diff": float("nan"),
        "se": float("nan"),
        "p": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "n": float(t),
    }
    if t < 4 or pair[:, 0].std() == 0 or pair[:, 1].std() == 0:
        return nan
    a, b = pair[:, 0], pair[:, 1]
    lags = default_maxlags(t)
    diff, se = _sharpe_diff_se(a, b, lags)
    if se == 0 or np.allclose(a, b):
        return {**nan, "sharpe_diff": diff, "se": se, "p": 1.0, "ci_low": diff, "ci_high": diff}
    block = block_size or max(1, int(round(t ** (1.0 / 3.0))))
    rng = np.random.default_rng(seed)
    n_blocks = -(-t // block)
    t_stats = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, t, n_blocks)
        idx = ((starts[:, None] + np.arange(block)) % t).ravel()[:t]
        try:
            d_star, se_star = _sharpe_diff_se(a[idx], b[idx], lags)
        except (ValueError, ZeroDivisionError):
            t_stats[i] = np.nan
            continue
        t_stats[i] = abs(d_star - diff) / se_star if se_star > 0 else np.nan
    t_stats = t_stats[np.isfinite(t_stats)]
    observed = abs(diff) / se
    p = float((1 + np.sum(t_stats >= observed)) / (1 + len(t_stats)))
    q = float(np.quantile(t_stats, level)) if len(t_stats) else float("nan")
    return {
        "sharpe_diff": diff,
        "se": se,
        "p": p,
        "ci_low": diff - q * se,
        "ci_high": diff + q * se,
        "n": float(t),
    }


def jobson_korkie_memmel(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> dict[str, float]:
    """Jobson and Korkie (1981) with Memmel's (2003) correction, per-period Sharpe ratios."""
    pair = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna().to_numpy(dtype=float)
    t = len(pair)
    if t < 3:
        return {"z": float("nan"), "p": float("nan")}
    a, b = pair[:, 0], pair[:, 1]
    s1, s2 = a.mean() / a.std(ddof=1), b.mean() / b.std(ddof=1)
    rho = float(np.corrcoef(a, b)[0, 1])
    theta = (2 - 2 * rho + 0.5 * (s1**2 + s2**2) - s1 * s2 * rho**2) / t
    if theta <= 0:
        return {"z": float("nan"), "p": 1.0}
    z = (s1 - s2) / math.sqrt(theta)
    return {"z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


def probabilistic_sharpe(returns: pd.Series | np.ndarray, sr_benchmark: float = 0.0) -> float:
    """Bailey and Lopez de Prado (2012): P(true per-period SR > sr_benchmark) given the sample
    SR, its length, skewness and (non-excess) kurtosis."""
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    t = len(r)
    if t < 3 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(t - 1) / math.sqrt(denom)))


EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period SR among `n_trials` independent trials under the null."""
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(returns: pd.Series | np.ndarray, n_trials: int, sr_variance: float) -> float:
    """Bailey and Lopez de Prado (2014): PSR against the expected maximum SR of `n_trials`."""
    return probabilistic_sharpe(returns, expected_max_sharpe(n_trials, sr_variance))


def rank_ic_series(scores: pd.Series, returns: pd.Series) -> float:
    """Spearman rank IC (same rule as views.ic.rank_ic: NaN under 10 stocks or zero variance)."""
    from portstudy.views.ic import rank_ic

    return rank_ic(scores, returns)
