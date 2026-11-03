"""Mean-variance strategy backed by the v1 optimizer."""

from __future__ import annotations

import numpy as np

from portstudy.config import StudyConfig
from portstudy.optimize.mean_variance import OptimizerError, solve_mean_variance
from portstudy.optimize.strategies.common import (
    StrategyInputs,
    StrategyResult,
    check_weights,
    fallback_chain,
    validate_inputs,
)


def run(inputs: StrategyInputs, cfg: StudyConfig) -> StrategyResult:
    validate_inputs(inputs, cfg)
    mu = inputs.mu.reindex(inputs.tickers).astype(float)
    sigma = inputs.sigma.reindex(index=inputs.tickers, columns=inputs.tickers)
    try:
        result = solve_mean_variance(
            mu,
            sigma,
            inputs.delta,
            cfg.optimizer,
            solvers=cfg.optimizer.solvers,
        )
        check_weights(result.weights, cfg.optimizer)
    except (OptimizerError, np.linalg.LinAlgError) as exc:
        return fallback_chain(inputs, cfg, {"solver_error": str(exc)})
    return StrategyResult(result.weights, result.solver, result.objective)
