"""Hierarchical risk parity control."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from portstudy.config import StudyConfig
from portstudy.optimize.mean_variance import OptimizerError
from portstudy.optimize.strategies.common import (
    StrategyInputs,
    StrategyResult,
    cap_and_renormalize,
    check_weights,
    validate_sigma,
    validate_universe,
)


def _cluster_variance(cluster: list[int], covariance: np.ndarray) -> float:
    block = covariance[np.ix_(cluster, cluster)]
    inverse_variance = 1.0 / np.diag(block)
    allocation = inverse_variance / inverse_variance.sum()
    return float(allocation @ block @ allocation)


def run(inputs: StrategyInputs, cfg: StudyConfig) -> StrategyResult:
    validate_universe(inputs, cfg)
    validate_sigma(inputs)
    tickers = inputs.tickers
    covariance = inputs.sigma.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    variances = np.diag(covariance)
    if not np.isfinite(variances).all() or (variances <= 0.0).any():
        raise OptimizerError("HRP requires positive finite variances")
    correlation = covariance / np.sqrt(np.outer(variances, variances))
    correlation = np.clip(0.5 * (correlation + correlation.T), -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    distance = np.sqrt(np.maximum((1.0 - correlation) / 2.0, 0.0))
    tree = linkage(
        squareform(distance, checks=False),
        method=cfg.strategies.hrp.linkage,
    )
    order = leaves_list(tree).tolist()

    allocation = np.ones(len(tickers), dtype=float)
    clusters = [order]
    while clusters:
        cluster = clusters.pop()
        if len(cluster) < 2:
            continue
        midpoint = len(cluster) // 2
        left, right = cluster[:midpoint], cluster[midpoint:]
        left_variance = _cluster_variance(left, covariance)
        right_variance = _cluster_variance(right, covariance)
        total_variance = left_variance + right_variance
        if not np.isfinite(total_variance) or total_variance <= 0.0:
            raise OptimizerError("HRP cluster variance must be positive and finite")
        left_weight = 1.0 - left_variance / total_variance
        allocation[left] *= left_weight
        allocation[right] *= 1.0 - left_weight
        clusters.extend((left, right))

    pre_cap = pd.Series(allocation, index=tickers, name="weight")
    pre_cap /= float(pre_cap.sum())
    capped = cap_and_renormalize(pre_cap, cfg.optimizer.max_weight)
    if not isinstance(capped, pd.Series):
        raise TypeError("series input must produce series output")
    check_weights(capped, cfg.optimizer)
    return StrategyResult(
        capped,
        "single_linkage",
        float("nan"),
        diagnostics={
            "linkage_order": [tickers[i] for i in order],
            "pre_cap_weights": pre_cap.to_dict(),
        },
    )
