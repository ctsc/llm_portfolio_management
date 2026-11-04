"""Long-only mean-variance with a position cap, solved with cvxpy.

maximize  w' mu - (delta/2) w' Sigma w
s.t.      sum(w) = 1,  0 <= w <= max_weight

Solvers are tried in the configured order (OSQP, then CLARABEL). On failure the previous
week's weights are returned with `fallback_flag = "solver_failed"`.

The problem is compiled once per (tickers, Sigma, delta, cap) with mu as a cvxpy Parameter and
reused for every later mu (resampled draws, shuffles, each source in a week); `use_cache=False`
builds the same parametrized problem fresh, so both paths share one canonicalization.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from portstudy.config import OptimizerConfig
from portstudy.optimize.problem_cache import CachedProblem, cached_problem, matrix_sha256


class OptimizerError(RuntimeError):
    pass


@dataclass
class MVResult:
    weights: pd.Series
    solver: str
    objective: float
    fallback_flag: str | None = None


def _normalize_capped_weights(raw: np.ndarray, cap: float) -> np.ndarray:
    weights: np.ndarray = np.clip(raw, 0.0, cap)
    total = float(weights.sum())
    if total > 1.0:
        weights /= total
    elif total < 1.0:
        headroom = cap - weights
        weights += (1.0 - total) * headroom / headroom.sum()
    return weights


def _cholesky_factor(sigma: np.ndarray) -> np.ndarray:
    sym = 0.5 * (sigma + sigma.T)
    try:
        return np.linalg.cholesky(sym)
    except np.linalg.LinAlgError:
        jitter = 1e-10 * np.trace(sym) / len(sym)
        return np.linalg.cholesky(sym + jitter * np.eye(len(sym)))


def solve_mean_variance(
    mu: pd.Series,
    sigma_matrix: pd.DataFrame,
    delta: float,
    cfg: OptimizerConfig,
    previous: pd.Series | None = None,
    solvers: Sequence[str] | None = None,
    use_cache: bool = True,
) -> MVResult:
    tickers = [str(t) for t in mu.index]
    n = len(tickers)
    if n < cfg.min_stocks:
        raise OptimizerError(f"{n} stocks available, need at least {cfg.min_stocks} for the cap")
    if cfg.max_weight * n < 1.0:
        raise OptimizerError("max_weight times the stock count is below 1; infeasible")
    sigma = sigma_matrix.loc[tickers, tickers].to_numpy(dtype=float)
    m = mu.to_numpy(dtype=float)

    def build() -> CachedProblem:
        chol = _cholesky_factor(sigma)
        mu_param = cp.Parameter(n, name="mu")
        w = cp.Variable(n)
        risk = cp.sum_squares(chol.T @ w)  # w' Sigma w through the Cholesky factor
        objective = cp.Maximize(mu_param @ w - (delta / 2.0) * risk)
        constraints = [cp.sum(w) == 1, w >= 0, w <= cfg.max_weight]
        return CachedProblem(cp.Problem(objective, constraints), w, mu_param)

    key = ("mvo", tuple(tickers), matrix_sha256(sigma), float(delta), float(cfg.max_weight))
    compiled = cached_problem(key, build) if use_cache else build()
    compiled.mu.value = m
    problem, w = compiled.problem, compiled.weights
    last_error = ""
    for name in solvers or cfg.solvers:
        try:
            problem.solve(solver=name, verbose=False, warm_start=False)
        except Exception as exc:  # solver missing or numerical failure: try the next one
            last_error = f"{name}: {exc}"
            continue
        if problem.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            raw = np.asarray(w.value, dtype=float)
            raw = _normalize_capped_weights(raw, cfg.max_weight)
            return MVResult(
                pd.Series(raw, index=tickers, name="weight"), name, float(problem.value)
            )
        last_error = f"{name}: status {problem.status}"
    if previous is None:
        raise OptimizerError(f"all solvers failed ({last_error}) and no previous weights exist")
    prev = previous.reindex(tickers).fillna(0.0)
    total = float(prev.sum())
    prev = prev / total if total > 0 else pd.Series(1.0 / n, index=tickers)
    return MVResult(prev.rename("weight"), "none", float("nan"), fallback_flag="solver_failed")
