"""Build the baseline and LLM views for Black-Litterman.

q_b     = pi + IC_b * sigma * z_b
q_L     = pi + IC_L * sigma * z_L
Omega_b = tau * diag(Sigma)
Omega_L = tau * diag(Sigma) / clip(conf, 0.1, 1.0)

Per-stock rule: a stock with no valid LLM output keeps only the baseline view.
Zero-IC rule: IC_L == 0 drops every LLM view row (`use_llm_view = False`) so the portfolio's
weights equal Q0's for the week. The same applies to the baseline view when IC_b == 0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from portstudy.config import ViewsConfig


@dataclass
class Views:
    tickers: list[str]
    pi: pd.Series
    sigma: pd.Series
    tau: float
    ic_b: float
    ic_l: float
    q_b: pd.Series
    omega_b: pd.Series
    use_baseline_view: bool
    q_l: pd.Series
    omega_l: pd.Series
    use_llm_view: bool
    llm_tickers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.tickers)


def range_certainty(width: pd.Series, cfg: ViewsConfig) -> pd.Series:
    """Map range width to view certainty using DECISION(E1)."""
    lo, hi = cfg.confidence_clip
    return (1.0 - width.astype(float) / 10.0).clip(lo, hi)


def build_views(
    pi: pd.Series,
    sigma_vec: pd.Series,
    z_b: pd.Series,
    ic_b: float,
    cfg: ViewsConfig,
    z_l: pd.Series | None = None,
    confidence: pd.Series | None = None,
    ic_l: float = 0.0,
    logger: logging.Logger | None = None,
    label: str = "",
) -> Views:
    """Build views for one portfolio. `z_l` and `confidence` cover the valid-output stocks."""
    log = logger or logging.getLogger("portstudy.views")
    tickers = [str(t) for t in pi.index]
    sigma = sigma_vec.reindex(tickers).astype(float)
    pi = pi.astype(float)
    zb = z_b.reindex(tickers).fillna(0.0).astype(float)
    var = sigma**2  # diag(Sigma)
    notes: list[str] = []

    use_b = ic_b > 0
    q_b = pi + ic_b * sigma * zb
    omega_b = cfg.tau * var
    if not use_b:
        notes.append("baseline_view_dropped")
        log.info(
            "baseline_view_dropped", extra={"extra_fields": {"portfolio": label, "ic_b": ic_b}}
        )

    llm_tickers: list[str] = []
    use_l = False
    q_l = pd.Series(dtype=float)
    omega_l = pd.Series(dtype=float)
    if z_l is not None and len(z_l) and ic_l > 0:
        llm_tickers = [t for t in tickers if t in set(z_l.index)]
        if llm_tickers:
            zl = z_l.reindex(llm_tickers).astype(float)
            conf = (
                confidence.reindex(llm_tickers).astype(float)
                if confidence is not None
                else pd.Series(1.0, index=llm_tickers)
            )
            lo, hi = cfg.confidence_clip
            conf = conf.fillna(lo).clip(lo, hi)
            q_l = pi.loc[llm_tickers] + ic_l * sigma.loc[llm_tickers] * zl
            omega_l = cfg.tau * var.loc[llm_tickers] / conf
            use_l = True
    if not use_l:
        reason = "ic_zero" if (z_l is not None and len(z_l)) else "no_valid_outputs"
        notes.append(f"llm_view_dropped:{reason}")
        log.info(
            "llm_view_dropped",
            extra={"extra_fields": {"portfolio": label, "ic_l": ic_l, "reason": reason}},
        )
        llm_tickers = []
    return Views(
        tickers=tickers,
        pi=pi,
        sigma=sigma,
        tau=cfg.tau,
        ic_b=ic_b,
        ic_l=ic_l,
        q_b=q_b,
        omega_b=omega_b,
        use_baseline_view=use_b,
        q_l=q_l,
        omega_l=omega_l,
        use_llm_view=use_l,
        llm_tickers=llm_tickers,
        notes=notes,
    )


def views_frame(views: Views) -> pd.DataFrame:
    """Flat per-stock table: ticker, pi, sigma, q_b, omega_b, q_L, omega_L, has_llm_view."""
    df = pd.DataFrame(
        {
            "ticker": views.tickers,
            "pi": views.pi.to_numpy(),
            "sigma": views.sigma.to_numpy(),
            "q_b": views.q_b.to_numpy(),
            "omega_b": views.omega_b.to_numpy(),
        }
    )
    df["q_L"] = views.q_l.reindex(views.tickers).to_numpy() if views.use_llm_view else np.nan
    df["omega_L"] = (
        views.omega_l.reindex(views.tickers).to_numpy() if views.use_llm_view else np.nan
    )
    df["has_llm_view"] = df["ticker"].isin(views.llm_tickers) if views.use_llm_view else False
    return df
