"""Mean-CVaR optimization using common posterior scenarios."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from portstudy.config import StudyConfig
from portstudy.optimize.mean_variance import OptimizerError
from portstudy.optimize.strategies.common import (
    StrategyInputs,
    StrategyResult,
    cap_and_renormalize,
    cholesky_with_jitter,
    clean_weights,
    fallback_chain,
    solve_with,
    standard_normals,
    validate_inputs,
)


def _sample_cvar(losses: np.ndarray, beta: float) -> float:
    tail_count = (1.0 - beta) * len(losses)
    ordered = np.sort(losses)[::-1]
    complete = int(np.floor(tail_count))
    remainder = tail_count - complete
    tail_sum = float(ordered[:complete].sum())
    if remainder > 0.0:
        tail_sum += remainder * float(ordered[complete])
    return tail_sum / tail_count


def run(inputs: StrategyInputs, cfg: StudyConfig) -> StrategyResult:
    validate_inputs(inputs, cfg)
    config = cfg.strategies.cvar
    tickers = inputs.tickers
    n = len(tickers)
    mu = inputs.mu.reindex(tickers).to_numpy(dtype=float)
    sigma = inputs.sigma.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
    cov_mu = inputs.cov_mu.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
    try:
        factor, jitter = cholesky_with_jitter(sigma + cov_mu)
    except OptimizerError as exc:
        return fallback_chain(inputs, cfg, {"solver_error": str(exc)})

    z = standard_normals(inputs.week_id, "cvar", (config.n_scenarios, n), config.seed)
    scenarios = mu[None, :] + z @ factor.T
    weights = cp.Variable(n)
    zeta = cp.Variable()
    excess = cp.Variable(config.n_scenarios)
    cvar = zeta + cp.sum(excess) / ((1.0 - config.beta) * config.n_scenarios)
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= cfg.optimizer.max_weight,
        excess >= -scenarios @ weights - zeta,
        excess >= 0,
    ]
    problem = cp.Problem(cp.Maximize(mu @ weights - config.lam * cvar), constraints)
    try:
        solver = solve_with(problem, config.solvers)
        raw_weights = pd.Series(
            np.asarray(weights.value, dtype=float), index=tickers, name="weight"
        )
        capped = cap_and_renormalize(raw_weights, cfg.optimizer.max_weight)
        if not isinstance(capped, pd.Series):
            raise TypeError("series input must produce series output")
        result_weights = clean_weights(capped, cfg.optimizer)
    except (OptimizerError, TypeError, ValueError) as exc:
        return fallback_chain(inputs, cfg, {"solver_error": str(exc), "covariance_jitter": jitter})

    losses = -(scenarios @ result_weights.to_numpy(dtype=float))
    in_sample_cvar = _sample_cvar(losses, config.beta)
    return_term = float(mu @ result_weights.to_numpy(dtype=float))
    cvar_term = float(config.lam * in_sample_cvar)
    return StrategyResult(
        result_weights,
        solver,
        float(problem.value),
        diagnostics={
            "in_sample_cvar": in_sample_cvar,
            "return_term": return_term,
            "cvar_term": cvar_term,
            "covariance_jitter": jitter,
        },
    )
