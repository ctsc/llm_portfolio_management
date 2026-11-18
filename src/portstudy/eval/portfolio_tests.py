"""Portfolio-level hypothesis tests (PLAN4 Phase 8): the portfolio halves of H2 and H3, H4, H5,
model comparisons, the information-versus-confidence split (G19), view sensitivity (G13),
model-change sensitivity (G21) and the comparisons with equal weight and HRP.

Weekly series are the open-to-open returns from Phase 3A (paired gaps, IR, Sharpe tests);
daily close values feed the factor regressions only. Intention-to-treat (G8) uses the realized
gap in every clean week, never forcing inactive weeks to zero. Every row carries `hypothesis`,
`comparison`, `unit`, `comparison_id` and gets `family` and `p_adj` from `eval/multiple.py`.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from portstudy.eval import metrics as m
from portstudy.eval.signal_tests import pooled_regression
from portstudy.eval.tests import deflated_sharpe, ledoit_wolf_sharpe_test, paired_summary
from portstudy.types import PortfolioId

FACTOR_NAMES = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
CONF_NOTE = (
    "part of the difference from Q0 comes from lower uncertainty alone (G18); see the "
    "uncertainty ratio"
)


def _row(hypothesis: str, comparison: str, unit: str, **fields: Any) -> dict[str, Any]:
    return {
        "hypothesis": hypothesis,
        "comparison": comparison,
        "unit": unit,
        "comparison_id": fields.pop("comparison_id", ""),
        **fields,
    }


def _summary(gap: pd.Series, cfg: Any) -> dict[str, Any]:
    e = cfg.eval
    s = paired_summary(
        gap.dropna(),
        e.tie_band,
        e.seed,
        n_perm=e.permutation_samples,
        n_boot=e.bootstrap_samples,
        exact_max_t=e.signflip_exact_max_t,
        block_mean=e.bootstrap_block_mean,
    )
    return {"estimate": s["mean_gap"], **s}


def daily_returns(data: Any, key: str) -> pd.Series:
    daily = data.daily_values.get(key, pd.DataFrame())
    if not len(daily):
        return pd.Series(dtype=float)
    v: pd.Series = daily.set_index("date")["value"].astype(float).sort_index()
    out: pd.Series = v.pct_change().dropna()
    return out


def factor_regression(
    y: pd.Series, factors: pd.DataFrame | None, subtract_rf: bool = False
) -> dict[str, Any]:
    """Daily OLS of `y` on the six factors with Newey-West SEs (lags floor(4 (T/100)^(2/9)));
    alpha daily and annualized (x252), its t and p, and every loading."""
    names = [c for c in FACTOR_NAMES if factors is not None and c in factors.columns]
    nan: dict[str, Any] = {
        "alpha_daily": np.nan,
        "alpha_annual": np.nan,
        "alpha_t": np.nan,
        "alpha_p": np.nan,
        "n_days": 0,
    }
    if factors is None or not names:
        return {**nan, "reason": "no factor data"}
    idx = pd.Index([pd.Timestamp(d).date() for d in y.index])
    joined = pd.concat(
        [pd.Series(y.to_numpy(), index=idx, name="y"), factors], axis=1, join="inner"
    ).dropna(subset=["y", *names])
    if subtract_rf and "RF" in joined:
        joined["y"] = joined["y"] - joined["RF"]
    if len(joined) < len(names) + 10:
        return {**nan, "n_days": len(joined), "reason": "too few days with factor data"}
    lags = int(math.floor(4 * (len(joined) / 100) ** (2 / 9)))
    x = sm.add_constant(joined[names].astype(float))
    fit = sm.OLS(joined["y"].astype(float), x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    df_resid = len(joined) - len(names) - 1
    t_alpha = float(fit.tvalues["const"])
    out = {
        "alpha_daily": float(fit.params["const"]),
        "alpha_annual": float(fit.params["const"] * 252),
        "alpha_t": t_alpha,
        "alpha_p": float(2 * stats.t.sf(abs(t_alpha), df_resid)),
        "n_days": len(joined),
        "nw_lags": lags,
        "reason": "",
    }
    for n in names:
        out[f"beta_{n}"] = float(fit.params[n])
        out[f"t_{n}"] = float(fit.tvalues[n])
    return out


def _weekly(data: Any, key: str) -> pd.Series:
    return data.weekly_returns[key] if key in data.weekly_returns else pd.Series(dtype=float)


def _clean(data: Any, key: str, s: pd.Series) -> pd.Series:
    mask = data.clean_mask(key).reindex(s.index).fillna(False).astype(bool)
    out: pd.Series = s[mask]
    return out


def _turnover(data: Any, key: str) -> float:
    trades = data.trades.get(key, pd.DataFrame())
    daily = data.daily_values.get(key, pd.DataFrame())
    avg = float(daily["value"].mean()) if len(daily) else float("nan")
    return float(trades["trade_value"].sum() / avg) if len(trades) and avg else 0.0


def _cost_share(data: Any, key: str) -> float:
    trades = data.trades.get(key, pd.DataFrame())
    daily = data.daily_values.get(key, pd.DataFrame())
    avg = float(daily["value"].mean()) if len(daily) else float("nan")
    return float(trades["cost_usd"].sum() / avg) if len(trades) and avg else 0.0


def compare(data: Any, a: str, b: str, weeks_mask: pd.Series | None = None) -> dict[str, Any]:
    """Paired weekly comparison of portfolio `a` with `b`: summary, IR, TE, Ledoit-Wolf."""
    gap = (_weekly(data, a) - _weekly(data, b)).dropna()
    if weeks_mask is not None:
        gap = gap[weeks_mask.reindex(gap.index).fillna(False).astype(bool)]
    lw = ledoit_wolf_sharpe_test(
        _weekly(data, a).reindex(gap.index),
        _weekly(data, b).reindex(gap.index),
        n_boot=min(data.cfg.eval.bootstrap_samples, 1000),
        seed=data.cfg.eval.seed,
    )
    return {
        **_summary(gap, data.cfg),
        "information_ratio": m.information_ratio(gap),
        "tracking_error": m.tracking_error(gap) if len(gap) > 1 else float("nan"),
        "lw_sharpe_diff": lw["sharpe_diff"],
        "lw_p": lw["p"],
    }


# View sensitivity (G13) ------------------------------------------------------------------------


def view_sensitivity(data: Any, threshold: float | None) -> pd.DataFrame:
    rows = []
    for strategy in data.strategies():
        keys = [k for k in data.steered_for(strategy) if k in data.expected_te]
        te = (
            pd.concat([_clean(data, k, data.expected_te[k]) for k in keys])
            if keys
            else (pd.Series(dtype=float))
        )
        ratio = (
            pd.concat([data.uncertainty_ratio[k] for k in keys if k in data.uncertainty_ratio])
            if keys
            else pd.Series(dtype=float)
        )
        median_te = float(te.median()) if te.notna().any() else float("nan")
        insensitive = threshold is not None and np.isfinite(median_te) and median_te < threshold
        shift = (
            pd.concat([data.view_shift[k] for k in keys if k in data.view_shift])
            if keys
            else (pd.Series(dtype=float))
        )
        rows.append(
            {
                "strategy": strategy,
                "median_expected_te": median_te,
                "threshold": threshold if threshold is not None else float("nan"),
                "label": "insensitive to views by construction"
                if insensitive
                else ("sensitive" if threshold is not None else "threshold not set"),
                "mean_uncertainty_ratio": float(ratio.mean())
                if ratio.notna().any()
                else float("nan"),
                "mean_view_shift": float(shift.mean()) if shift.notna().any() else float("nan"),
                "note": CONF_NOTE if strategy in ("robust", "resampled", "cvar") else "",
            }
        )
    return pd.DataFrame(rows)


# H2 portfolio ----------------------------------------------------------------------------------


def h2_portfolio(data: Any, excluded: dict[str, list[str]] | None = None) -> list[dict[str, Any]]:
    """Steered vs own Q0: ITT on clean weeks and active weeks only (G8), with IR, TE, the
    Ledoit-Wolf test, turnover gap, break-even cost, share of active weeks and the daily
    active-return factor regression. `excluded` drops weeks per model (G21 sensitivity)."""
    rows = []
    for key in data.portfolios:
        strategy = data.strategy_of.get(key)
        q0 = data.q0_key(strategy) if strategy else ""
        if q0 not in data.weekly_returns or key not in data.weekly_returns:
            continue
        clean = data.clean_mask(key).reindex(data.weeks).fillna(False).astype(bool)
        model = PortfolioId.parse(key).model
        if excluded and model in excluded:
            clean = clean & ~clean.index.isin(excluded[model])
        active = (
            data.active_view[key].reindex(data.weeks).fillna(False).astype(bool)
            if (key in data.active_view)
            else pd.Series(False, index=data.weeks)
        )
        gap = (_weekly(data, key) - _weekly(data, q0)).reindex(data.weeks)
        inactive_gap = gap[clean & ~active].dropna()
        share_active = float(active[clean].mean()) if clean.any() else float("nan")
        turnover_gap = _turnover(data, key) - _turnover(data, q0)
        net_active = float(gap[clean].dropna().sum())
        be, be_reason = m.break_even_cost(
            net_active, _cost_share(data, key) - _cost_share(data, q0), turnover_gap
        )
        active_daily = (daily_returns(data, key) - daily_returns(data, q0)).dropna()
        fr = factor_regression(active_daily, data.factors) if not excluded else {}
        base = {
            "strategy": strategy,
            "source": data.source_of.get(key, ""),
            "portfolio": key,
            "share_active_weeks": share_active,
            "mean_abs_inactive_gap": float(inactive_gap.abs().mean())
            if len(inactive_gap)
            else float("nan"),
            "turnover_gap": turnover_gap,
            "break_even_cost": be,
            "break_even_reason": be_reason,
        }
        label = " excluding model_change weeks" if excluded else ""
        for analysis, mask in (("intention-to-treat", clean), ("active weeks", clean & active)):
            if excluded and analysis == "active weeks":
                continue
            rows.append(
                _row(
                    "H2",
                    f"steered minus Q0 weekly return ({analysis}){label}",
                    key,
                    analysis=analysis,
                    **base,
                    **compare(data, key, q0, mask),
                    **(
                        {f"active_{k}": v for k, v in fr.items()}
                        if analysis == "intention-to-treat"
                        else {}
                    ),
                )
            )
    return rows


def info_conf_split(data: Any) -> list[dict[str, Any]]:
    """G19: steered minus Q0 = information (steered minus CONF) + confidence (CONF minus Q0)."""
    rows = []
    for key in data.portfolios:
        pid = PortfolioId.parse(key)
        if pid.source is None:
            continue
        conf = PortfolioId.conf(pid.strategy, pid.source).key  # type: ignore[arg-type]
        q0 = data.q0_key(pid.strategy)
        if conf not in data.weekly_returns:
            if pid.strategy == "cvar":
                ratio = (
                    data.uncertainty_ratio[key].mean()
                    if key in data.uncertainty_ratio
                    else (float("nan"))
                )
                rows.append(
                    _row(
                        "H2",
                        "information vs confidence split",
                        key,
                        p=float("nan"),
                        note="no confidence-only control for Mean-CVaR; uncertainty "
                        f"ratio {ratio:.3f}",
                    )
                )
            continue
        clean = data.clean_mask(key).reindex(data.weeks).fillna(False).astype(bool)
        rows.append(
            _row(
                "H2",
                "information: steered minus confidence-only",
                key,
                **compare(data, key, conf, clean),
            )
        )
        rows.append(
            _row(
                "H2", "confidence: confidence-only minus Q0", key, **compare(data, conf, q0, clean)
            )
        )
    return rows


# H3 portfolio, model comparisons, controls -----------------------------------------------------


def _by_cell(data: Any) -> dict[tuple[str, str, str], str]:
    out = {}
    for key in data.portfolios:
        pid = PortfolioId.parse(key)
        out[(pid.model, pid.config_id, pid.strategy)] = key
    return out


def h3_portfolio(data: Any) -> list[dict[str, Any]]:
    rows = []
    cells = _by_cell(data)
    stacked = []
    for (model, prompt, strategy), key in cells.items():
        q0 = data.q0_key(strategy)
        active = (_weekly(data, key) - _weekly(data, q0)).reindex(data.weeks)
        clean = data.clean_mask(key).reindex(data.weeks).fillna(False).astype(bool)
        stacked.append(
            pd.DataFrame(
                {
                    "week_id": data.weeks,
                    "y": active.where(clean).to_numpy(),
                    "model": model,
                    "prompt": prompt,
                    "strategy": strategy,
                }
            )
        )
        if not prompt.startswith("skills"):
            continue
        for control in ("placebo", "bare"):
            other = cells.get((model, control, strategy))
            if other is None:
                continue
            both = clean & data.clean_mask(other).reindex(data.weeks).fillna(False).astype(bool)
            hyp = "H5" if control == "placebo" else "H3"
            rows.append(
                _row(
                    hyp,
                    f"{prompt} minus {control} weekly return",
                    key,
                    model=model,
                    strategy=strategy,
                    prompt_id=prompt,
                    control=control,
                    note="the placebo controls for prompt length and any change to the views"
                    if control == "placebo"
                    else "",
                    **compare(data, key, other, both),
                )
            )
    if stacked:
        panel = pd.concat(stacked, ignore_index=True)
        skills = {str(p) for p in panel["prompt"].unique() if str(p).startswith("skills")}
        for strat, g in panel.groupby("strategy"):
            for control in ("placebo", "bare"):
                sub = g[g["prompt"].isin(skills | {control})].assign(
                    skills=lambda d: d["prompt"].isin(skills).astype(float)
                )
                rows.append(
                    _row(
                        "H3",
                        f"pooled skills minus {control} active return (model and week FE)",
                        str(strat),
                        **pooled_regression(sub, "y ~ skills + C(model) + C(week_id)", "skills"),
                    )
                )
    return rows


def model_portfolio(data: Any, prompt: str) -> list[dict[str, Any]]:
    rows = []
    cells = _by_cell(data)
    models = sorted({mo for (mo, p, _s) in cells if p == prompt})
    for strategy in data.strategies():
        for m1, m2 in itertools.combinations(models, 2):
            a, b = cells.get((m1, prompt, strategy)), cells.get((m2, prompt, strategy))
            if a and b:
                rows.append(
                    _row(
                        "H3",
                        f"{m1} minus {m2} weekly return on {prompt}",
                        f"{strategy}|{m1}|{m2}",
                        **compare(data, a, b),
                    )
                )
    return rows


def controls_comparison(data: Any) -> list[dict[str, Any]]:
    rows = []
    q0s = [data.q0_key(s) for s in data.strategies()]
    for key in [*data.portfolios, *q0s]:
        for control in data.controls:
            if key in data.weekly_returns and control in data.weekly_returns:
                rows.append(
                    _row("H2", f"weekly return minus {control}", key, **compare(data, key, control))
                )
    return rows


# H4 (exploratory) and H5 -----------------------------------------------------------------------


def h4_exploratory(data: Any) -> list[dict[str, Any]]:
    rows = []
    gaps = {}
    for strategy in data.strategies():
        q0 = data.q0_key(strategy)
        series = [
            (_weekly(data, k) - _weekly(data, q0)).where(
                data.clean_mask(k).reindex(data.weeks).fillna(False).astype(bool)
            )
            for k in data.steered_for(strategy)
        ]
        if series:
            gaps[strategy] = pd.concat(series, axis=1).mean(axis=1).reindex(data.weeks)
    for s1, s2 in itertools.combinations(gaps, 2):
        rows.append(
            _row(
                "H4",
                f"{s1} minus {s2} mean active return",
                f"{s1}|{s2}",
                **_summary(gaps[s1] - gaps[s2], data.cfg),
            )
        )
    up = (data.spy_weekly.reindex(data.weeks) > 0).astype(float)
    vol = data.spy_vol20.reindex(data.weeks)
    high = (vol > vol.median()).astype(float)
    for strategy, gap in gaps.items():
        for regime, flag in (("market up", up), ("high volatility", high)):
            df = pd.DataFrame({"y": gap, "regime": flag}).dropna()
            if len(df) < 4 or df["regime"].nunique() < 2:
                continue
            fit = sm.OLS(df["y"], sm.add_constant(df["regime"])).fit()
            b, se = float(fit.params["regime"]), float(fit.bse["regime"])
            n = len(df)
            tval = b / se if se > 0 else float("nan")
            crit = stats.t.ppf(0.975, n - 1)
            rows.append(
                _row(
                    "H4",
                    f"active return by regime: {regime}",
                    strategy,
                    estimate=b,
                    se=se,
                    t=tval,
                    p=float(2 * stats.t.sf(abs(tval), n - 1)),
                    ci_low=b - crit * se,
                    ci_high=b + crit * se,
                    n=n,
                )
            )
    return rows


def h5_factor_rows(data: Any) -> list[dict[str, Any]]:
    """Each portfolio's daily excess return on the six factors (H5); the active-return
    regressions are joined into the H2 rows. Headline Sharpe gets the deflated Sharpe ratio."""
    rows = []
    for key in data.portfolio_keys:
        fr = factor_regression(daily_returns(data, key), data.factors, subtract_rf=True)
        rows.append(
            _row(
                "H5",
                "excess return factor regression",
                key,
                estimate=fr["alpha_annual"],
                p=fr["alpha_p"],
                t=fr["alpha_t"],
                factor_source=data.factor_source,
                **fr,
            )
        )
    sharpe = {k: m.sharpe_annual(_weekly(data, k)) for k in data.portfolios}
    sharpe = {k: v for k, v in sharpe.items() if np.isfinite(v)}
    if sharpe:
        best = max(sharpe, key=lambda k: sharpe[k])
        per_period = [v / math.sqrt(52) for v in sharpe.values()]
        dsr = deflated_sharpe(
            _weekly(data, best),
            len(data.portfolios),
            float(np.var(per_period, ddof=1)) if len(per_period) > 1 else 0.0,
        )
        t_best = sharpe[best] / math.sqrt(52) * math.sqrt(max(_weekly(data, best).notna().sum(), 1))
        rows.append(
            _row(
                "H5",
                "headline portfolio deflated Sharpe",
                best,
                estimate=dsr,
                p=float("nan"),
                sharpe=sharpe[best],
                n_trials=len(data.portfolios),
                t_above_3=bool(abs(t_best) > 3),
            )
        )
    return rows


def any_source_tests(data: Any) -> list[dict[str, Any]]:
    """Optional `arch` SPA and StepM per strategy on weekly active returns (losses = -active)."""
    if not m.arch_available():
        return [
            _row(
                "H2",
                "SPA and StepM (any source beats Q0)",
                "all",
                p=float("nan"),
                note="arch not installed; skipped",
            )
        ]
    rows = []
    for strategy in data.strategies():
        q0 = data.q0_key(strategy)
        act = pd.DataFrame(
            {k: _weekly(data, k) - _weekly(data, q0) for k in data.steered_for(strategy)}
        ).dropna()
        if act.shape[0] < 5 or act.shape[1] < 1:
            continue
        bench = pd.Series(0.0, index=act.index)
        spa = m.spa_test(bench, -act, seed=data.cfg.eval.seed)
        winners = m.stepm(bench, -act, seed=data.cfg.eval.seed) or []
        rows.append(
            _row(
                "H2",
                "SPA: any source beats Q0",
                strategy,
                p=spa["p_consistent"] if spa else float("nan"),
                stepm_superior=";".join(winners),
            )
        )
    return rows


def portfolio_tests(data: Any, prereg: Any) -> dict[str, pd.DataFrame]:
    """Every portfolio-level table, families assigned."""
    from portstudy.eval.multiple import assign_families

    rows = [
        *h2_portfolio(data),
        *info_conf_split(data),
        *h3_portfolio(data),
        *model_portfolio(data, prereg.main_model_comparison_prompt),
        *controls_comparison(data),
        *h4_exploratory(data),
        *h5_factor_rows(data),
        *any_source_tests(data),
    ]
    if data.model_change:
        rows.extend(h2_portfolio(data, excluded=data.model_change))
    tests = pd.DataFrame(rows)
    if "p" not in tests:
        tests["p"] = np.nan
    ids = {c.id for c in prereg.primary_comparisons}
    tests = assign_families(
        tests,
        ids,
        data.cfg.eval.multiple_testing.exploratory,
        significance=data.cfg.eval.significance,
    )
    sens = view_sensitivity(data, prereg.view_sensitivity_min_tracking_error)
    insensitive = set(sens.loc[sens["label"].str.startswith("insensitive"), "strategy"])
    if "strategy" in tests:
        tests["view_sensitivity"] = np.where(
            tests["strategy"].isin(insensitive), "insensitive to views by construction", ""
        )
    return {
        "portfolio_tests": tests,
        "view_sensitivity": sens,
        "model_change_weeks": pd.DataFrame(
            [{"model": k, "weeks": ";".join(v)} for k, v in data.model_change.items()],
            columns=["model", "weeks"],
        ),
    }
