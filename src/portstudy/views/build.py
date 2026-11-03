"""Build the baseline and LLM views for Black-Litterman.

q_b     = pi + IC_b * sigma * z_b
q_L     = pi + IC_L * sigma * z_L
Omega_b = tau * diag(Sigma) * clip(k_sd / median(k_sd), 0.25, 4.0)
Omega_L = tau * diag(Sigma) / clip(1 - width / 10, 0.1, 1.0)

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
    k_sd: pd.Series
    ratio: pd.Series
    q_b: pd.Series
    omega_b: pd.Series
    use_baseline_view: bool
    q_l: pd.Series
    omega_l: pd.Series
    width: pd.Series
    c: pd.Series
    use_llm_view: bool
    llm_tickers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.tickers)


def width_to_c(width: pd.Series, cfg: ViewsConfig) -> pd.Series:
    """Map a validated rating width to its DECISION(E1) scale."""
    values = width.astype(float)
    if not np.isfinite(values.to_numpy()).all() or ((values < 0) | (values > 10)).any():
        raise AssertionError("validated rating widths must be within [0, 10]")
    lo, hi = cfg.width_conf_clip
    return (1.0 - values / 10.0).clip(lo, hi).rename("c")


def _baseline_ratio(
    k_sd: pd.Series | None,
    tickers: list[str],
    cfg: ViewsConfig,
    log: logging.Logger,
    label: str,
) -> tuple[pd.Series, pd.Series]:
    spread = (
        k_sd.reindex(tickers).astype(float)
        if k_sd is not None
        else pd.Series(np.nan, index=tickers, dtype=float)
    )
    valid = np.isfinite(spread.to_numpy()) & (spread.to_numpy() >= 0)
    ratio = pd.Series(1.0, index=tickers, name="ratio")
    if not valid.any():
        log.info(
            "baseline_spread_unavailable",
            extra={"extra_fields": {"portfolio": label, "reason": "all_missing"}},
        )
        return spread.rename("k_sd"), ratio

    median = float(spread.loc[valid].median())
    if median <= 1e-12:
        log.info(
            "baseline_spread_degenerate",
            extra={
                "extra_fields": {
                    "portfolio": label,
                    "median_k_sd": median,
                    "threshold": 1e-12,
                }
            },
        )
        return spread.rename("k_sd"), ratio

    if not valid.all():
        log.info(
            "baseline_spread_missing",
            extra={
                "extra_fields": {
                    "portfolio": label,
                    "tickers": [tickers[i] for i, is_valid in enumerate(valid) if not is_valid],
                }
            },
        )
    lo, hi = cfg.kronos_spread_clip
    ratio.iloc[np.flatnonzero(valid)] = (spread.iloc[np.flatnonzero(valid)] / median).clip(lo, hi)
    return spread.rename("k_sd"), ratio


def build_views(
    pi: pd.Series,
    sigma_vec: pd.Series,
    z_b: pd.Series,
    ic_b: float,
    cfg: ViewsConfig,
    k_sd: pd.Series | None = None,
    z_l: pd.Series | None = None,
    ic_l: float = 0.0,
    width: pd.Series | None = None,
    logger: logging.Logger | None = None,
    label: str = "",
) -> Views:
    """Build baseline and valid LLM views for one portfolio."""
    log = logger or logging.getLogger("portstudy.views")
    tickers = [str(t) for t in pi.index]
    sigma = sigma_vec.reindex(tickers).astype(float)
    pi = pi.astype(float)
    zb = z_b.reindex(tickers).fillna(0.0).astype(float)
    var = sigma**2  # diag(Sigma)
    notes: list[str] = []
    spreads, ratio = _baseline_ratio(k_sd, tickers, cfg, log, label)

    use_b = ic_b > 0
    q_b = pi + ic_b * sigma * zb
    omega_b = cfg.tau * var * ratio
    if not use_b:
        notes.append("baseline_view_dropped")
        log.info(
            "baseline_view_dropped", extra={"extra_fields": {"portfolio": label, "ic_b": ic_b}}
        )

    llm_tickers: list[str] = []
    use_l = False
    q_l = pd.Series(dtype=float)
    omega_l = pd.Series(dtype=float)
    llm_width = pd.Series(dtype=float, name="width")
    c = pd.Series(dtype=float, name="c")
    if width is not None:
        width_tickers = [t for t in tickers if t in set(width.index)]
        llm_width = width.reindex(width_tickers).astype(float)
        if llm_width.isna().any():
            raise AssertionError("validated rating widths must be present")
        if len(llm_width):
            c = width_to_c(llm_width, cfg)
    if z_l is not None and len(z_l) and ic_l > 0:
        llm_tickers = [t for t in tickers if t in set(z_l.index)]
        if llm_tickers:
            if width is None or not set(llm_tickers).issubset(llm_width.index):
                raise ValueError("width is required for every LLM view")
            zl = z_l.reindex(llm_tickers).astype(float)
            q_l = pi.loc[llm_tickers] + ic_l * sigma.loc[llm_tickers] * zl
            omega_l = cfg.tau * var.loc[llm_tickers] / c.reindex(llm_tickers)
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
        k_sd=spreads,
        ratio=ratio,
        q_b=q_b,
        omega_b=omega_b,
        use_baseline_view=use_b,
        q_l=q_l,
        omega_l=omega_l,
        width=llm_width,
        c=c,
        use_llm_view=use_l,
        llm_tickers=llm_tickers,
        notes=notes,
    )


def views_frame(views: Views) -> pd.DataFrame:
    """Flat per-stock table for view values and uncertainty inputs."""
    df = pd.DataFrame(
        {
            "ticker": views.tickers,
            "pi": views.pi.to_numpy(),
            "sigma": views.sigma.to_numpy(),
            "q_b": views.q_b.to_numpy(),
            "omega_b": views.omega_b.to_numpy(),
            "k_sd": views.k_sd.to_numpy(),
            "ratio": views.ratio.to_numpy(),
        }
    )
    df["q_L"] = views.q_l.reindex(views.tickers).to_numpy() if views.use_llm_view else np.nan
    df["omega_L"] = (
        views.omega_l.reindex(views.tickers).to_numpy() if views.use_llm_view else np.nan
    )
    df["width"] = views.width.reindex(views.tickers).to_numpy()
    df["c"] = views.c.reindex(views.tickers).to_numpy()
    df["has_llm_view"] = df["ticker"].isin(views.llm_tickers) if views.use_llm_view else False
    return df
