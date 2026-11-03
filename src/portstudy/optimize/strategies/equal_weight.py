"""Forecast-free equal-weight control."""

from __future__ import annotations

import pandas as pd

from portstudy.config import StudyConfig
from portstudy.optimize.strategies.common import (
    StrategyInputs,
    StrategyResult,
    check_weights,
    validate_universe,
)


def run(inputs: StrategyInputs, cfg: StudyConfig) -> StrategyResult:
    validate_universe(inputs, cfg)
    weights = pd.Series(1.0 / len(inputs.tickers), index=inputs.tickers, name="weight")
    check_weights(weights, cfg.optimizer)
    return StrategyResult(weights, "deterministic", float("nan"))
