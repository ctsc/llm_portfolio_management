"""Black-Litterman posterior with stacked identity views.

P      = [I; I_L]          (I_L selects the stocks that carry an LLM view)
q      = [q_b ; q_L]
Omega  = diag([Omega_b ; Omega_L])
A      = inv(tau*Sigma) + P' inv(Omega) P
b      = inv(tau*Sigma) pi + P' inv(Omega) q
mu_BL  = solve(A, b)
M      = inv(A)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

from portstudy.optimize.mean_variance import OptimizerError
from portstudy.views.build import Views


@dataclass
class Posterior:
    mu_bl: pd.Series
    tilt: pd.Series
    n_views: int
    cov_mu: pd.DataFrame


def _factor(a: np.ndarray, logger: logging.Logger) -> np.ndarray:
    try:
        return np.linalg.cholesky(a)
    except np.linalg.LinAlgError:
        jitter = 1e-12 * float(np.trace(a)) / len(a)
        logger.warning(
            "black_litterman_cholesky_jitter",
            extra={"extra_fields": {"jitter": jitter, "dimension": len(a)}},
        )
        try:
            return np.linalg.cholesky(a + jitter * np.eye(len(a)))
        except np.linalg.LinAlgError as exc:
            raise OptimizerError(
                "Black-Litterman posterior precision is not positive definite"
            ) from exc


def black_litterman(
    views: Views,
    sigma_matrix: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> Posterior:
    """Posterior for a portfolio. Dropped views (flags False) simply contribute no rows."""
    tickers = views.tickers
    sigma = sigma_matrix.loc[tickers, tickers].to_numpy(dtype=float)
    pi = views.pi.reindex(tickers).to_numpy(dtype=float)
    n = len(tickers)
    identity = np.eye(n)
    prior_cov = views.tau * sigma
    prior_precision = np.linalg.solve(prior_cov, identity)
    a = prior_precision.copy()
    b = prior_precision @ pi
    n_views = 0
    if views.use_baseline_view:
        inv_ob = 1.0 / views.omega_b.reindex(tickers).to_numpy(dtype=float)
        a[np.diag_indices(n)] += inv_ob
        b += inv_ob * views.q_b.reindex(tickers).to_numpy(dtype=float)
        n_views += n
    if views.use_llm_view and views.llm_tickers:
        idx = np.array([tickers.index(t) for t in views.llm_tickers])
        inv_ol = 1.0 / views.omega_l.reindex(views.llm_tickers).to_numpy(dtype=float)
        a[idx, idx] += inv_ol
        b[idx] += inv_ol * views.q_l.reindex(views.llm_tickers).to_numpy(dtype=float)
        n_views += len(idx)
    a = 0.5 * (a + a.T)
    log = logger or logging.getLogger("portstudy.optimize.black_litterman")
    factor = _factor(a, log)
    forward_mean = solve_triangular(factor, b, lower=True)
    mu = solve_triangular(factor.T, forward_mean, lower=False)
    forward_cov = solve_triangular(factor, identity, lower=True)
    cov_mu_array = solve_triangular(factor.T, forward_cov, lower=False)
    cov_mu_array = 0.5 * (cov_mu_array + cov_mu_array.T)
    mu_s = pd.Series(mu, index=tickers, name="mu_bl")
    return Posterior(
        mu_bl=mu_s,
        tilt=(mu_s - views.pi.reindex(tickers)).rename("tilt"),
        n_views=n_views,
        cov_mu=pd.DataFrame(cov_mu_array, index=tickers, columns=tickers),
    )


def black_litterman_q0(
    views: Views,
    sigma_matrix: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> Posterior:
    """Q0 and any portfolio with `use_llm_view = False`: P = I, q = q_b, Omega = Omega_b."""
    q0 = Views(
        tickers=views.tickers,
        pi=views.pi,
        sigma=views.sigma,
        tau=views.tau,
        ic_b=views.ic_b,
        ic_l=0.0,
        k_sd=views.k_sd,
        ratio=views.ratio,
        q_b=views.q_b,
        omega_b=views.omega_b,
        use_baseline_view=views.use_baseline_view,
        q_l=pd.Series(dtype=float),
        omega_l=pd.Series(dtype=float),
        width=pd.Series(dtype=float),
        c=pd.Series(dtype=float),
        use_llm_view=False,
        llm_tickers=[],
        notes=list(views.notes),
    )
    return black_litterman(q0, sigma_matrix, logger=logger)
