"""Robust mean-variance optimization over posterior mean uncertainty."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from portstudy.config import StudyConfig
from portstudy.optimize.mean_variance import OptimizerError
from portstudy.optimize.problem_cache import CachedProblem, cached_problem, matrix_sha256
from portstudy.optimize.strategies.common import (
    StrategyInputs,
    StrategyResult,
    cap_and_renormalize,
    cholesky_with_jitter,
    clean_weights,
    fallback_chain,
    solve_with,
    validate_inputs,
)


def run(inputs: StrategyInputs, cfg: StudyConfig) -> StrategyResult:
    validate_inputs(inputs, cfg)
    tickers = inputs.tickers
    n = len(tickers)
    mu = inputs.mu.reindex(tickers).to_numpy(dtype=float)
    sigma = inputs.sigma.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
    cov_mu = inputs.cov_mu.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
    try:
        factor, jitter = cholesky_with_jitter(cov_mu)
    except OptimizerError as exc:
        return fallback_chain(inputs, cfg, {"solver_error": str(exc)})

    kappa = cfg.strategies.robust.kappa
    cap = cfg.optimizer.max_weight

    def build() -> CachedProblem:
        mu_param = cp.Parameter(n, name="mu")
        w = cp.Variable(n)
        return_term = mu_param @ w
        uncertainty_term = kappa * cp.norm(factor.T @ w, 2)
        risk_term = (inputs.delta / 2.0) * cp.quad_form(w, 0.5 * (sigma + sigma.T))
        problem = cp.Problem(
            cp.Maximize(return_term - uncertainty_term - risk_term),
            [cp.sum(w) == 1, w >= 0, w <= cap],
        )
        return CachedProblem(problem, w, mu_param)

    # R (the cov_mu factor) is part of the key: it changes with every source's views.
    key = (
        "robust",
        tuple(tickers),
        matrix_sha256(sigma),
        matrix_sha256(cov_mu),
        float(kappa),
        float(inputs.delta),
        float(cap),
    )
    compiled = cached_problem(key, build)
    compiled.mu.value = mu
    problem, weights = compiled.problem, compiled.weights
    try:
        solver = solve_with(problem, cfg.strategies.robust.solvers)
        raw_weights = pd.Series(
            np.asarray(weights.value, dtype=float), index=tickers, name="weight"
        )
        capped = cap_and_renormalize(raw_weights, cfg.optimizer.max_weight)
        if not isinstance(capped, pd.Series):
            raise TypeError("series input must produce series output")
        result_weights = clean_weights(capped, cfg.optimizer)
    except (OptimizerError, TypeError, ValueError) as exc:
        return fallback_chain(inputs, cfg, {"solver_error": str(exc), "covariance_jitter": jitter})

    values = result_weights.to_numpy(dtype=float)
    return StrategyResult(
        result_weights,
        solver,
        float(problem.value),
        diagnostics={
            "return_term": float(mu @ values),
            "uncertainty_term": float(
                cfg.strategies.robust.kappa * np.linalg.norm(factor.T @ values)
            ),
            "risk_term": float((inputs.delta / 2.0) * values @ sigma @ values),
            "covariance_jitter": jitter,
        },
    )
