"""Confidence-only counterfactual (DECISION(G18)).

For a source-week, the posterior with the same LLM view uncertainty `Omega_L` but a neutral rating
(`z_L = 0`, so `q_L = mu_Q0`). Under E7 its mean equals Q0's posterior mean; what changes is the
posterior covariance `M_conf`, which shows how much uncertainty the views remove on their own.
One linear solve; no optimizer runs here.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd

from portstudy.optimize.black_litterman import Posterior, black_litterman
from portstudy.views.build import Views


def conf_only_views(views: Views) -> Views:
    """The same views with every LLM view moved onto the Q0 posterior mean."""
    if not views.use_llm_view or not views.llm_tickers:
        return views
    center = views.mu_q0.reindex(views.llm_tickers).astype(float)
    return dataclasses.replace(views, q_l=center)


def conf_only_posterior(
    views: Views, sigma_matrix: pd.DataFrame, logger: logging.Logger | None = None
) -> Posterior:
    return black_litterman(conf_only_views(views), sigma_matrix, logger=logger)


def uncertainty_ratio(conf: Posterior, q0: Posterior) -> float:
    """trace(M_conf) / trace(M_Q0): below 1 when the LLM views add confidence."""
    denom = float(np.trace(q0.cov_mu.to_numpy(dtype=float)))
    if denom <= 0:
        return float("nan")
    return float(np.trace(conf.cov_mu.to_numpy(dtype=float))) / denom
