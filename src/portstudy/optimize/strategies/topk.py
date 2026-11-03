"""Deterministic Top-K allocation by posterior mean."""

from __future__ import annotations

import pandas as pd

from portstudy.config import StudyConfig
from portstudy.optimize.strategies.common import (
    StrategyInputs,
    StrategyResult,
    cap_and_renormalize,
    check_weights,
    validate_inputs,
)


def run(inputs: StrategyInputs, cfg: StudyConfig) -> StrategyResult:
    validate_inputs(inputs, cfg)
    mu = inputs.mu.reindex(inputs.tickers)
    k = min(cfg.strategies.topk.k, len(inputs.tickers))
    order = sorted(inputs.tickers, key=lambda ticker: (-float(mu.loc[ticker]), ticker))
    selected = order[:k]
    weights = pd.Series(0.0, index=inputs.tickers, name="weight")
    weights.loc[selected] = 1.0 / k
    capped = cap_and_renormalize(weights, cfg.optimizer.max_weight)
    if not isinstance(capped, pd.Series):
        raise TypeError("series input must produce series output")
    check_weights(capped, cfg.optimizer)
    return StrategyResult(
        capped,
        "deterministic",
        float(mu.reindex(capped.index).to_numpy(dtype=float) @ capped.to_numpy(dtype=float)),
        diagnostics={"selected": selected},
    )
