"""Registry and shared interface for strategy and control portfolios."""

from __future__ import annotations

from collections.abc import Callable

from portstudy.config import StudyConfig
from portstudy.optimize.strategies import cvar, equal_weight, hrp, mvo, resampled, robust, topk
from portstudy.optimize.strategies.common import (
    StrategyInputs,
    StrategyResult,
    check_weights,
)

StrategyRunner = Callable[[StrategyInputs, StudyConfig], StrategyResult]

STRATEGIES: dict[str, StrategyRunner] = {
    "mvo": mvo.run,
    "cvar": cvar.run,
    "robust": robust.run,
    "resampled": resampled.run,
    "topk": topk.run,
}
CONTROLS: dict[str, StrategyRunner] = {
    "HRP": hrp.run,
    "EW": equal_weight.run,
}


def run_strategy(name: str, inputs: StrategyInputs, cfg: StudyConfig) -> StrategyResult:
    try:
        strategy = STRATEGIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown strategy {name!r}; expected one of {tuple(STRATEGIES)}") from exc
    result = strategy(inputs, cfg)
    check_weights(result.weights, cfg.optimizer)
    return result


def run_control(name: str, inputs: StrategyInputs, cfg: StudyConfig) -> StrategyResult:
    try:
        control = CONTROLS[name]
    except KeyError as exc:
        raise ValueError(f"unknown control {name!r}; expected one of {tuple(CONTROLS)}") from exc
    result = control(inputs, cfg)
    check_weights(result.weights, cfg.optimizer)
    return result


__all__ = [
    "CONTROLS",
    "STRATEGIES",
    "StrategyInputs",
    "StrategyResult",
    "check_weights",
    "run_control",
    "run_strategy",
]
