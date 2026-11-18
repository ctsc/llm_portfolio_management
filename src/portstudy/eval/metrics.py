"""Portfolio metrics (DECISION(G10)): risk-free-adjusted Sharpe and its probabilistic and
Lo-adjusted versions, Sortino, drawdown, Calmar, realized CVaR, information ratio and tracking
error, certainty-equivalent return, hit rate, break-even cost, and optional `arch` wrappers for
SPA, StepM and the model confidence set.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from portstudy.eval.tests import probabilistic_sharpe

WEEKS_PER_YEAR = 52
DAYS_PER_YEAR = 252


def weekly_rf(rf_daily: pd.Series, weeks: dict[str, tuple[Any, Any]]) -> pd.Series:
    """Weekly risk-free return per week id, compounding the daily rate over [start, end)."""
    out = {}
    for week, (start, end) in weeks.items():
        part = rf_daily[(rf_daily.index >= start) & (rf_daily.index < end)]
        out[week] = float(np.prod(1.0 + part.to_numpy(dtype=float)) - 1.0) if len(part) else 0.0
    return pd.Series(out, dtype=float)


def excess(returns: pd.Series, rf: pd.Series | float) -> pd.Series:
    if isinstance(rf, pd.Series):
        return returns - rf.reindex(returns.index).fillna(0.0)
    return returns - rf


def sharpe_annual(excess_returns: pd.Series, periods: int = WEEKS_PER_YEAR) -> float:
    r = excess_returns.dropna().astype(float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods))


def lag1_autocorr(returns: pd.Series) -> float:
    r = returns.dropna().astype(float)
    if len(r) < 3 or r.std(ddof=1) == 0:
        return float("nan")
    return float(np.corrcoef(r.to_numpy()[1:], r.to_numpy()[:-1])[0, 1])


def lo_adjusted_sharpe(excess_returns: pd.Series, periods: int = WEEKS_PER_YEAR) -> float:
    """Lo (2002) annualization with AR(1) autocorrelations rho^k (reported, not tested)."""
    r = excess_returns.dropna().astype(float)
    rho = lag1_autocorr(r)
    if len(r) < 3 or not np.isfinite(rho) or r.std(ddof=1) == 0:
        return float("nan")
    q = periods
    sr = r.mean() / r.std(ddof=1)
    denom = q + 2 * sum((q - k) * rho**k for k in range(1, q))
    if denom <= 0:
        return float("nan")
    return float(sr * q / math.sqrt(denom))


def sortino(excess_daily: pd.Series, periods: int = DAYS_PER_YEAR) -> float:
    r = excess_daily.dropna().astype(float)
    downside = math.sqrt(float(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2))) if len(r) else 0.0
    if len(r) < 2 or downside == 0:
        return float("nan")
    return float(r.mean() / downside * math.sqrt(periods))


def max_drawdown(values: pd.Series) -> float:
    v = values.dropna().astype(float)
    if not len(v):
        return float("nan")
    return float((v / v.cummax() - 1.0).min())


def calmar(daily_returns: pd.Series, values: pd.Series, periods: int = DAYS_PER_YEAR) -> float:
    r = daily_returns.dropna().astype(float)
    mdd = max_drawdown(values)
    if not len(r) or not np.isfinite(mdd) or mdd == 0:
        return float("nan")
    annual = float(np.prod(1.0 + r.to_numpy()) ** (periods / len(r)) - 1.0)
    return annual / abs(mdd)


def cvar(daily_returns: pd.Series, level: float = 0.95) -> float:
    """Realized CVaR: the mean loss over the worst (1 - level) share of days, as a positive loss."""
    r = np.sort(daily_returns.dropna().to_numpy(dtype=float))
    if not len(r):
        return float("nan")
    k = max(1, int(math.ceil((1 - level) * len(r))))
    return float(-r[:k].mean())


def tracking_error(active: pd.Series, periods: int = WEEKS_PER_YEAR) -> float:
    a = active.dropna().astype(float)
    return float(a.std(ddof=1) * math.sqrt(periods)) if len(a) > 1 else float("nan")


def information_ratio(active: pd.Series, periods: int = WEEKS_PER_YEAR) -> float:
    a = active.dropna().astype(float)
    if len(a) < 2 or a.std(ddof=1) == 0:
        return float("nan")
    return float(a.mean() / a.std(ddof=1) * math.sqrt(periods))


def certainty_equivalent(returns: pd.Series, gamma: float, periods: int = WEEKS_PER_YEAR) -> float:
    """Annualized mean-variance certainty-equivalent return: (mean - gamma/2 var) * periods."""
    r = returns.dropna().astype(float)
    if len(r) < 2:
        return float("nan")
    return float((r.mean() - gamma / 2 * r.var(ddof=1)) * periods)


def hit_rate(values: pd.Series) -> tuple[float, float]:
    """Share of positive values and the two-sided binomial p-value against 0.5 (zeros dropped)."""
    v = values.dropna().astype(float)
    v = v[v != 0]
    if not len(v):
        return float("nan"), float("nan")
    k = int((v > 0).sum())
    return k / len(v), float(stats.binomtest(k, len(v), 0.5).pvalue)


def break_even_cost(
    net_active_return: float, extra_cost: float, extra_turnover: float
) -> tuple[float, str]:
    """Trading cost per unit traded at which the gross active return vs Q0 disappears:
    (net active + extra cost) / extra turnover. NaN with a reason when it is undefined."""
    if not np.isfinite(extra_turnover) or extra_turnover <= 0:
        return float("nan"), "extra turnover vs Q0 is not positive"
    gross = net_active_return + extra_cost
    if gross <= 0:
        return float("nan"), "gross active return vs Q0 is not positive"
    return float(gross / extra_turnover), ""


def return_metrics(
    weekly: pd.Series,
    weekly_rf_rate: pd.Series | float,
    daily_returns: pd.Series,
    daily_rf: pd.Series | float,
    values: pd.Series,
    cer_gamma: list[float],
) -> dict[str, float]:
    """The G10 metric set for one portfolio."""
    ex_w = excess(weekly.dropna(), weekly_rf_rate)
    ex_d = excess(daily_returns.dropna(), daily_rf)
    hr, hr_p = hit_rate(ex_w)
    out = {
        "sharpe_excess": sharpe_annual(ex_w),
        "sharpe_lo": lo_adjusted_sharpe(ex_w),
        "autocorr_lag1": lag1_autocorr(ex_w),
        "psr": probabilistic_sharpe(ex_w),
        "volatility": float(weekly.dropna().std(ddof=1) * math.sqrt(WEEKS_PER_YEAR))
        if weekly.dropna().size > 1
        else float("nan"),
        "sortino": sortino(ex_d),
        "max_drawdown_daily": max_drawdown(values),
        "calmar": calmar(daily_returns, values),
        "cvar_95": cvar(daily_returns),
        "hit_rate": hr,
        "hit_rate_p": hr_p,
    }
    for g in cer_gamma:
        out[f"cer_gamma_{g:g}"] = certainty_equivalent(weekly, g)
    return out


# Optional arch wrappers (the `stats` extra) -------------------------------------------------


def arch_available() -> bool:
    try:
        import arch  # noqa: F401
    except ImportError:
        return False
    return True


def spa_test(
    benchmark_losses: pd.Series, model_losses: pd.DataFrame, seed: int = 0
) -> dict[str, float] | None:
    """Hansen SPA: does any model beat the benchmark (losses, lower is better)? None w/o arch."""
    if not arch_available():
        return None
    from arch.bootstrap import SPA

    spa = SPA(benchmark_losses.to_numpy(), model_losses.to_numpy(), seed=seed)
    spa.compute()
    p = spa.pvalues
    return {
        "p_lower": float(p["lower"]),
        "p_consistent": float(p["consistent"]),
        "p_upper": float(p["upper"]),
    }


def stepm(
    benchmark_losses: pd.Series, model_losses: pd.DataFrame, seed: int = 0
) -> list[str] | None:
    """Romano-Wolf StepM: models that beat the benchmark. None without arch."""
    if not arch_available():
        return None
    from arch.bootstrap import StepM

    sm = StepM(benchmark_losses.to_numpy(), model_losses, seed=seed)
    sm.compute()
    return [str(c) for c in sm.superior_models]


def model_confidence_set(
    losses: pd.DataFrame, size: float = 0.1, seed: int = 0
) -> list[str] | None:
    """Hansen MCS: the set of plausibly best models. None without arch."""
    if not arch_available():
        return None
    from arch.bootstrap import MCS

    mcs = MCS(losses, size=size, seed=seed)
    mcs.compute()
    return [str(c) for c in mcs.included]
