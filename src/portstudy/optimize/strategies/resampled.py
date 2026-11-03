"""MVO over common posterior-mean draws."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portstudy.config import StudyConfig
from portstudy.optimize.mean_variance import OptimizerError, solve_mean_variance
from portstudy.optimize.strategies.common import (
    StrategyInputs,
    StrategyResult,
    cap_and_renormalize,
    check_weights,
    cholesky_with_jitter,
    clean_weights,
    fallback_chain,
    standard_normals,
    validate_inputs,
)


def run(inputs: StrategyInputs, cfg: StudyConfig) -> StrategyResult:
    validate_inputs(inputs, cfg)
    config = cfg.strategies.resampled
    tickers = inputs.tickers
    n = len(tickers)
    mu = inputs.mu.reindex(tickers).to_numpy(dtype=float)
    sigma = inputs.sigma.reindex(index=tickers, columns=tickers)
    cov_mu = inputs.cov_mu.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
    try:
        factor, jitter = cholesky_with_jitter(cov_mu)
    except OptimizerError as exc:
        return fallback_chain(inputs, cfg, {"solver_error": str(exc)})

    z = standard_normals(inputs.week_id, "resampled", (config.n_draws, n), config.seed)
    draws = mu[None, :] + z @ factor.T
    solutions: list[np.ndarray] = []
    objectives: list[float] = []
    solvers: list[str] = []
    for draw in draws:
        try:
            result = solve_mean_variance(
                pd.Series(draw, index=tickers),
                sigma,
                inputs.delta,
                cfg.optimizer,
                solvers=cfg.optimizer.solvers,
            )
            check_weights(result.weights, cfg.optimizer)
        except (OptimizerError, np.linalg.LinAlgError):
            continue
        solutions.append(result.weights.to_numpy(dtype=float))
        objectives.append(result.objective)
        solvers.append(result.solver)

    failed = config.n_draws - len(solutions)
    diagnostics = {
        "failed_draws": failed,
        "weight_dispersion": (
            float(np.std(np.stack(solutions), axis=0).mean()) if solutions else float("nan")
        ),
        "covariance_jitter": jitter,
        "solvers": sorted(set(solvers)),
    }
    if failed / config.n_draws > config.max_failed_share or not solutions:
        return fallback_chain(inputs, cfg, diagnostics)
    try:
        averaged = pd.Series(np.mean(solutions, axis=0), index=tickers, name="weight")
        capped = cap_and_renormalize(averaged, cfg.optimizer.max_weight)
        if not isinstance(capped, pd.Series):
            raise TypeError("series input must produce series output")
        result_weights = clean_weights(capped, cfg.optimizer)
    except OptimizerError as exc:
        diagnostics["solver_error"] = str(exc)
        return fallback_chain(inputs, cfg, diagnostics)

    return StrategyResult(
        result_weights,
        "resampled",
        float(np.mean(objectives)),
        diagnostics=diagnostics,
    )
