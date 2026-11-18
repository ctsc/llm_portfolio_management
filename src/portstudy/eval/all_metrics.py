"""Every metric for every unit in long form (DECISION(G22)): `results/eval/all_metrics.csv`.

One row per (unit, metric) with estimate, CI, p, `p_adj`, family, T and a `reason` whenever the
value is NaN. Units: each source, steered portfolio, Q0, confidence-only portfolio, HRP and equal
weight, plus pooled rows. `check_completeness` lists any expected (unit, metric) pair with no row;
the evaluation fails on a non-empty list.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

COLUMNS = [
    "unit",
    "unit_type",
    "metric",
    "estimate",
    "ci_low",
    "ci_high",
    "p",
    "p_adj",
    "family",
    "n",
    "reason",
]

PORTFOLIO_METRICS = [
    "sharpe_excess",
    "psr",
    "sharpe_lo",
    "sortino",
    "calmar",
    "max_drawdown_daily",
    "cvar_95",
    "volatility",
    "cer_gamma_2",
    "cer_gamma_5",
    "turnover",
    "total_cost_usd",
    "total_return",
    "hit_rate",
    "alpha_annual",
]
STEERED_METRICS = [
    *PORTFOLIO_METRICS,
    "active_return",
    "information_ratio",
    "tracking_error",
    "lw_sharpe_diff",
    "break_even_cost",
    "share_active_weeks",
    "view_shift",
    "expected_te_vs_q0",
    "conf_only_uncertainty_ratio",
]
SOURCE_METRICS = [
    "rank_ic",
    "ic_gap_vs_baseline",
    "residual_ic",
    "icir",
    "ic_hit_rate",
    "quintile_spread",
    "tilt_ic_gap_vs_q0",
]


def _unit_type(key: str, data: Any) -> str:
    if key in data.sources:
        return "source"
    if key in data.controls:
        return "control"
    if key in data.conf_only:
        return "confidence_only"
    if key.endswith("__Q0"):
        return "q0"
    if key in data.source_of:
        return "steered"
    return "pooled"


def _put(
    rows: list[dict[str, Any]],
    data: Any,
    unit: str,
    metric: str,
    value: Any,
    reason: str = "",
    **extra: Any,
) -> None:
    try:
        est = float(value)
    except (TypeError, ValueError):
        est = float("nan")
    if not np.isfinite(est) and not reason:
        reason = extra.pop("why", "") or "not estimable from the available weeks"
    rows.append(
        {
            "unit": unit,
            "unit_type": _unit_type(unit, data),
            "metric": metric,
            "estimate": est,
            "reason": "" if np.isfinite(est) else reason,
            **extra,
        }
    )


def _test_fields(r: pd.Series) -> dict[str, Any]:
    return {k: r.get(k, np.nan) for k in ("ci_low", "ci_high", "p", "p_adj", "family", "n")}


def build_all_metrics(
    data: Any,
    metrics: pd.DataFrame,
    signal: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    met = metrics.set_index("portfolio") if len(metrics) else pd.DataFrame()
    h5 = (
        portfolio[portfolio["comparison"] == "excess return factor regression"].set_index("unit")
        if len(portfolio)
        else pd.DataFrame()
    )
    for key in data.portfolio_keys:
        for metric in PORTFOLIO_METRICS:
            if metric == "alpha_annual":
                r = h5.loc[key] if key in h5.index else None
                _put(
                    rows,
                    data,
                    key,
                    metric,
                    r["alpha_annual"] if r is not None else np.nan,
                    (r.get("reason", "") if r is not None else "no factor regression row"),
                    p=r["alpha_p"] if r is not None else np.nan,
                    n=r.get("n_days", np.nan) if r is not None else np.nan,
                )
                continue
            value = met.loc[key, metric] if key in met.index and metric in met.columns else np.nan
            _put(
                rows,
                data,
                key,
                metric,
                value,
                n=met.loc[key, "n_weeks"] if key in met.index else np.nan,
            )
    itt = (
        portfolio[
            portfolio["comparison"].str.startswith(
                "steered minus Q0 weekly return (intention-to-treat)"
            )
        ]
        if len(portfolio)
        else pd.DataFrame()
    )
    itt = itt[~itt["comparison"].str.contains("excluding")] if len(itt) else itt
    itt = itt.set_index("unit") if len(itt) else itt
    for key in data.portfolios:
        r = itt.loc[key] if len(itt) and key in itt.index else None
        tf = _test_fields(r) if r is not None else {}
        _put(
            rows,
            data,
            key,
            "active_return",
            r["estimate"] if r is not None else np.nan,
            "" if r is not None else "no H2 row (no Q0 or no weekly returns)",
            **tf,
        )
        for metric in (
            "information_ratio",
            "tracking_error",
            "lw_sharpe_diff",
            "break_even_cost",
            "share_active_weeks",
        ):
            reason = ""
            if metric == "break_even_cost" and r is not None:
                reason = str(r.get("break_even_reason", ""))
            _put(
                rows,
                data,
                key,
                metric,
                r[metric] if r is not None else np.nan,
                reason,
                p=r["lw_p"] if r is not None and metric == "lw_sharpe_diff" else np.nan,
            )
        for metric, table in (
            ("view_shift", data.view_shift),
            ("expected_te_vs_q0", data.expected_te),
            ("conf_only_uncertainty_ratio", data.uncertainty_ratio),
        ):
            v = table[key].mean() if key in table else np.nan
            _put(rows, data, key, metric, v, "not recorded in the weight metas")
    by = {}
    if len(signal):
        for comp, metric in (
            ("LLM rank IC above zero", "rank_ic"),
            ("LLM minus baseline rank IC (same stocks)", "ic_gap_vs_baseline"),
            ("residual IC (Fama-MacBeth)", "residual_ic"),
            ("top-minus-bottom quintile spread", "quintile_spread"),
            ("steered tilt minus Q0 tilt rank IC (same stocks)", "tilt_ic_gap_vs_q0"),
        ):
            sub = signal[
                (signal["comparison"] == comp) & (signal["hypothesis"].isin(["H1", "H2"]))
            ].set_index("unit")
            by[metric] = sub
    for s in data.sources:
        for metric in SOURCE_METRICS:
            if metric in ("icir", "ic_hit_rate"):
                sub = by.get("ic_gap_vs_baseline", pd.DataFrame())
                r = sub.loc[s] if s in sub.index else None
                col = metric
                _put(
                    rows,
                    data,
                    s,
                    metric,
                    r.get(col, np.nan) if r is not None else np.nan,
                    "" if r is not None else "no H1 row for this source",
                    p=r.get("ic_hit_p", np.nan)
                    if r is not None and metric == "ic_hit_rate"
                    else np.nan,
                )
                continue
            sub = by.get(metric, pd.DataFrame())
            r = sub.loc[s] if s in sub.index else None
            _put(
                rows,
                data,
                s,
                metric,
                r["estimate"] if r is not None else np.nan,
                "" if r is not None else "no signal row for this source",
                **(_test_fields(r) if r is not None else {}),
            )
    for table in (signal, portfolio):
        if not len(table):
            continue
        pooled = table[
            table["unit"].astype(str).str.startswith("pooled")
            | table["comparison"].astype(str).str.startswith("pooled")
        ]
        for _, r in pooled.iterrows():
            _put(
                rows,
                data,
                f"pooled:{r['unit']}",
                r["comparison"],
                r.get("estimate", np.nan),
                **_test_fields(r),
                comparison_id=r.get("comparison_id", ""),
            )
    out = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in out:
            out[col] = np.nan if col not in ("reason", "family") else ""
    return out.loc[:, [*COLUMNS, *[c for c in out.columns if c not in COLUMNS]]]


def expected_pairs(data: Any) -> list[tuple[str, str]]:
    pairs = [(k, m) for k in data.portfolio_keys for m in PORTFOLIO_METRICS]
    pairs += [
        (k, m) for k in data.portfolios for m in STEERED_METRICS if m not in PORTFOLIO_METRICS
    ]
    pairs += [(s, m) for s in data.sources for m in SOURCE_METRICS]
    return pairs


def check_completeness(data: Any, table: pd.DataFrame) -> list[str]:
    have = set(zip(table["unit"], table["metric"], strict=False))
    missing = [f"{u}: {m}" for u, m in expected_pairs(data) if (u, m) not in have]
    nan_no_reason = table[table["estimate"].isna() & (table["reason"].fillna("") == "")]
    missing += [f"{r.unit}: {r.metric} is NaN without a reason" for r in nan_no_reason.itertuples()]
    return missing
