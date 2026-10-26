"""Black-Litterman posterior with stacked identity views.

P      = [I; I_L]          (I_L selects the stocks that carry an LLM view)
q      = [q_b ; q_L]
Omega  = diag([Omega_b ; Omega_L])
A      = inv(tau*Sigma) + P' inv(Omega) P
b      = inv(tau*Sigma) pi + P' inv(Omega) q
mu_BL  = solve(A, b)

Because Omega is diagonal, P' inv(Omega) P is the diagonal matrix D = diag(1/Omega_b) plus
1/Omega_L scattered onto the LLM stocks, and P' inv(Omega) q = d is a vector. The system is
solved as (I + tau Sigma D) mu = pi + tau Sigma d, which avoids inverting Sigma.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portstudy.views.build import Views


@dataclass
class Posterior:
    mu_bl: pd.Series
    tilt: pd.Series
    n_views: int


def _posterior(
    sigma: np.ndarray, pi: np.ndarray, tau: float, d_diag: np.ndarray, d_vec: np.ndarray
) -> np.ndarray:
    n = len(pi)
    ts = tau * sigma
    a = np.eye(n) + ts * d_diag[None, :]  # (I + tau Sigma D) with D diagonal
    b = pi + ts @ d_vec
    mu: np.ndarray = np.linalg.solve(a, b)
    return mu


def black_litterman(views: Views, sigma_matrix: pd.DataFrame) -> Posterior:
    """Posterior for a portfolio. Dropped views (flags False) simply contribute no rows."""
    tickers = views.tickers
    sigma = sigma_matrix.loc[tickers, tickers].to_numpy(dtype=float)
    pi = views.pi.reindex(tickers).to_numpy(dtype=float)
    n = len(tickers)
    d_diag = np.zeros(n)
    d_vec = np.zeros(n)
    n_views = 0
    if views.use_baseline_view:
        inv_ob = 1.0 / views.omega_b.reindex(tickers).to_numpy(dtype=float)
        d_diag += inv_ob
        d_vec += inv_ob * views.q_b.reindex(tickers).to_numpy(dtype=float)
        n_views += n
    if views.use_llm_view and views.llm_tickers:
        idx = np.array([tickers.index(t) for t in views.llm_tickers])
        inv_ol = 1.0 / views.omega_l.reindex(views.llm_tickers).to_numpy(dtype=float)
        d_diag[idx] += inv_ol
        d_vec[idx] += inv_ol * views.q_l.reindex(views.llm_tickers).to_numpy(dtype=float)
        n_views += len(idx)
    if n_views == 0:
        mu = pi.copy()
    else:
        mu = _posterior(sigma, pi, views.tau, d_diag, d_vec)
    mu_s = pd.Series(mu, index=tickers, name="mu_bl")
    return Posterior(
        mu_bl=mu_s, tilt=(mu_s - views.pi.reindex(tickers)).rename("tilt"), n_views=n_views
    )


def black_litterman_q0(views: Views, sigma_matrix: pd.DataFrame) -> Posterior:
    """Q0 and any portfolio with `use_llm_view = False`: P = I, q = q_b, Omega = Omega_b."""
    q0 = Views(
        tickers=views.tickers,
        pi=views.pi,
        sigma=views.sigma,
        tau=views.tau,
        ic_b=views.ic_b,
        ic_l=0.0,
        q_b=views.q_b,
        omega_b=views.omega_b,
        use_baseline_view=views.use_baseline_view,
        q_l=pd.Series(dtype=float),
        omega_l=pd.Series(dtype=float),
        use_llm_view=False,
        llm_tickers=[],
        notes=list(views.notes),
    )
    return black_litterman(q0, sigma_matrix)
