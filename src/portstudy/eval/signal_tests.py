"""Signal-level hypothesis tests (PLAN4 Phase 7): H1, the signal halves of H2 and H3, model
comparisons and H3b, on a long stock-week panel.

Every IC gap is computed each week on the stocks that have a value in both signals (G14). The
residual IC (G9) is a weekly Fama-MacBeth slope of next-week return ranks on the standardized
score and the preregistered controls, tested over weeks with Newey-West and t(T-1). Pooled
tests are OLS on source-week rows with model (and week) fixed effects and standard errors
clustered by week; p-values use t with (weeks - 1) degrees of freedom. Every row carries
`hypothesis` and `comparison_id` and goes through `eval/multiple.py`.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from portstudy.eval.tests import mean_test, paired_summary
from portstudy.views.ic import MIN_STOCKS_FOR_IC, rank_ic

NUMERIC_CONTROLS = {
    "baseline_score": "baseline_score_z",
    "momentum_12_1": "momentum_12_1_z",
    "log_market_cap": "log_market_cap_z",
    "volatility_63d": "volatility_63d_z",
    "beta_252d": "beta_252d_z",
    "loading_hml": "loading_hml_z",
    "loading_rmw": "loading_rmw_z",
    "loading_cma": "loading_cma_z",
}


# Weekly building blocks ----------------------------------------------------------------------


def same_stock_ic(week: pd.DataFrame, a: str, b: str) -> tuple[float, float, int]:
    """Rank IC of `a` and of `b` against `ret` on the stocks valid in both."""
    sub = week[[a, b, "ret"]].dropna()
    if len(sub) < MIN_STOCKS_FOR_IC:
        return float("nan"), float("nan"), len(sub)
    return rank_ic(sub[a], sub["ret"]), rank_ic(sub[b], sub["ret"]), len(sub)


def weekly_gap(panel: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Per week: IC(a), IC(b) and IC(a) - IC(b) on the same stocks, and the stock count."""
    rows = []
    for week, g in panel.groupby("week_id", sort=True):
        ia, ib, n = same_stock_ic(g, a, b)
        rows.append({"week_id": week, "ic_a": ia, "ic_b": ib, "gap": ia - ib, "n_stocks": n})
    return pd.DataFrame(rows).set_index("week_id") if rows else pd.DataFrame()


def weekly_ic(panel: pd.DataFrame, score: str) -> pd.Series:
    out = {}
    for week, g in panel.groupby("week_id", sort=True):
        sub = g[[score, "ret"]].dropna()
        out[week] = rank_ic(sub[score], sub["ret"]) if len(sub) else float("nan")
    return pd.Series(out, dtype=float)


def _z(v: pd.Series) -> pd.Series:
    sd = v.std(ddof=1)
    return (v - v.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else v * 0.0


def _design(
    g: pd.DataFrame, score: str, controls: list[str]
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Standardized rank-return target, design [1, z(score), z(controls)..., sector dummies],
    and the rows used. The score is column 1."""
    cols = [score] + [NUMERIC_CONTROLS[c] for c in controls if c in NUMERIC_CONTROLS]
    cols = [c for c in cols if c in g.columns]
    use_sector = "sector" in controls and "sector" in g.columns
    sub = g[[*cols, "ret", *(["sector"] if use_sector else [])]].dropna()
    y = sub["ret"].rank(method="average").to_numpy(dtype=float)
    y = (y - y.mean()) / (y.std(ddof=1) or 1.0)
    parts = [np.ones(len(sub)), _z(sub[score].astype(float)).to_numpy()]
    for c in cols[1:]:
        v = sub[c].astype(float)
        if len(v) > 1 and v.std(ddof=1) > 0:
            parts.append(_z(v).to_numpy())
    if use_sector:
        dummies = pd.get_dummies(sub["sector"].astype(str), drop_first=True, dtype=float)
        parts.extend(dummies[c].to_numpy() for c in dummies.columns)
    return y, np.column_stack(parts), sub


def fama_macbeth(panel: pd.DataFrame, score: str, controls: list[str]) -> pd.Series:
    """Weekly slope on the standardized score, controlling for `controls` (G9)."""
    out = {}
    for week, g in panel.groupby("week_id", sort=True):
        y, x, _ = _design(g, score, controls)
        if len(y) < max(MIN_STOCKS_FOR_IC, x.shape[1] + 5):
            continue
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        out[week] = float(coef[1])
    return pd.Series(out, dtype=float)


def residualized_ic(panel: pd.DataFrame, score: str, controls: list[str]) -> pd.Series:
    """Cross-check: residualize the score on the controls each week, then rank IC."""
    out = {}
    for week, g in panel.groupby("week_id", sort=True):
        _, x, sub = _design(g, score, controls)
        if len(sub) < max(MIN_STOCKS_FOR_IC, x.shape[1] + 5):
            continue
        s = x[:, 1]
        others = np.delete(x, 1, axis=1)
        coef, *_ = np.linalg.lstsq(others, s, rcond=None)
        out[week] = rank_ic(pd.Series(s - others @ coef, index=sub.index), sub["ret"])
    return pd.Series(out, dtype=float)


def quintile_spread(panel: pd.DataFrame, score: str) -> pd.Series:
    """Per week: mean return of the top score quintile minus the bottom quintile."""
    out = {}
    for week, g in panel.groupby("week_id", sort=True):
        sub = g[[score, "ret"]].dropna()
        if len(sub) < 10:
            continue
        q = sub[score].rank(method="first", pct=True)
        out[week] = float(sub.loc[q > 0.8, "ret"].mean() - sub.loc[q <= 0.2, "ret"].mean())
    return pd.Series(out, dtype=float)


def ic_profile(ic: pd.Series) -> dict[str, Any]:
    """Mean IC, ICIR, hit rate and its binomial p."""
    v = ic.dropna()
    nz = v[v != 0]
    k = int((nz > 0).sum())
    return {
        "mean_ic": float(v.mean()) if len(v) else float("nan"),
        "icir": float(v.mean() / v.std(ddof=1))
        if len(v) > 1 and v.std(ddof=1) > 0
        else float("nan"),
        "ic_hit_rate": k / len(nz) if len(nz) else float("nan"),
        "ic_hit_p": float(stats.binomtest(k, len(nz), 0.5).pvalue) if len(nz) else float("nan"),
    }


def series_test(series: pd.Series) -> dict[str, Any]:
    mt = mean_test(series)
    return {"estimate": mt["mean"], "se": mt["se"], "t": mt["t"], "p": mt["p"], "n": mt["n"]}


def pooled_regression(
    df: pd.DataFrame, formula: str, term: str, cluster: str = "week_id"
) -> dict[str, Any]:
    """OLS with standard errors clustered by `cluster`; t-test of `term` with G - 1 df."""
    data = df.dropna(subset=["y"])
    groups = data[cluster].nunique()
    nan = {
        "estimate": float("nan"),
        "se": float("nan"),
        "t": float("nan"),
        "p": float("nan"),
        "n": float(len(data)),
        "n_weeks": float(groups),
    }
    if len(data) < 4 or groups < 3:
        return nan
    codes = pd.factorize(data[cluster])[0]
    try:
        fit = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": codes})
    except (ValueError, np.linalg.LinAlgError):
        return nan
    if term not in fit.params:
        return nan
    est, se = float(fit.params[term]), float(fit.bse[term])
    tval = est / se if se > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(tval), groups - 1)) if np.isfinite(tval) else float("nan")
    return {
        "estimate": est,
        "se": se,
        "t": tval,
        "p": p,
        "n": float(len(data)),
        "n_weeks": float(groups),
    }


# Panels from EvalData ------------------------------------------------------------------------


def likely_col(source: str) -> str:
    return f"likely:{source}"


def tilt_panel(weeks: list[str], sources: list[str]) -> pd.DataFrame:
    """Per (week, ticker): Q0's tilt and each source's steered tilt (`mu_BL - pi`)."""
    from portstudy.io import read_csv
    from portstudy.optimize import posterior_dir

    frames = []
    for w in weeks:
        folder = posterior_dir(w)
        q0 = folder / "Q0.csv"
        if not q0.exists():
            continue
        part = read_csv(q0).set_index("ticker")[["tilt"]].rename(columns={"tilt": "tilt:Q0"})
        for s in sources:
            path = folder / f"{s}.csv"
            if path.exists():
                part[f"tilt:{s}"] = read_csv(path).set_index("ticker")["tilt"]
        part["week_id"] = w
        frames.append(part.reset_index())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _row(
    hypothesis: str, comparison: str, unit: str, *, cid: str = "", **fields: Any
) -> dict[str, Any]:
    return {
        "hypothesis": hypothesis,
        "comparison": comparison,
        "unit": unit,
        "comparison_id": cid,
        **fields,
    }


def _gap_row(
    hypothesis: str,
    comparison: str,
    unit: str,
    gap: pd.DataFrame,
    cfg: Any,
    cid: str = "",
    **extra: Any,
) -> dict[str, Any]:
    e = cfg.eval
    g = gap["gap"] if len(gap) else pd.Series(dtype=float)
    summary = paired_summary(
        g,
        e.tie_band,
        e.seed,
        n_perm=e.permutation_samples,
        n_boot=e.bootstrap_samples,
        exact_max_t=e.signflip_exact_max_t,
        block_mean=e.bootstrap_block_mean,
    )
    return _row(
        hypothesis,
        comparison,
        unit,
        cid=cid,
        estimate=summary["mean_gap"],
        mean_n_stocks=float(gap["n_stocks"].mean()) if len(gap) else float("nan"),
        **summary,
        **extra,
    )


def _primary_ids(prereg: Any) -> dict[str, Any]:
    return {
        c.kind if c.kind != "pooled_prompt_ic_gap" else f"{c.kind}:{c.control_prompt}": c
        for c in prereg.primary_comparisons
    }


def _model_of(source: str) -> str:
    return source.split("__")[0]


def _prompt_of(source: str) -> str:
    return source.split("__")[1]


def signal_tests(data: Any, prereg: Any) -> pd.DataFrame:
    """Every signal-level row (H1, H2 signal, H3 signal, model comparisons, H3b), families set."""
    from portstudy.eval.multiple import assign_families

    cfg = data.cfg
    panel = data.stock_week
    controls = list(prereg.residual_ic_controls)
    prim = _primary_ids(prereg)
    rows: list[dict[str, Any]] = []
    if not len(panel) or "baseline_score" not in panel.columns:
        return pd.DataFrame(columns=["hypothesis", "comparison", "unit", "comparison_id", "p"])

    # H1: LLM likely vs the baseline score, same stocks; residual IC; profile.
    stacked = []
    for s in data.sources:
        col = likely_col(s)
        if col not in panel:
            continue
        clean = data.source_clean_mask(s)
        sp = panel[panel["week_id"].isin([w for w, ok in clean.items() if ok])]
        gap = weekly_gap(sp, col, "baseline_score")
        fm = fama_macbeth(sp, col, controls)
        prof = ic_profile(weekly_ic(sp, col))
        spread = quintile_spread(sp, col)
        rows.append(_gap_row("H1", "LLM minus baseline rank IC (same stocks)", s, gap, cfg, **prof))
        rows.append(
            _row(
                "H1",
                "residual IC (Fama-MacBeth)",
                s,
                **series_test(fm),
                cross_check_resid_ic=float(residualized_ic(sp, col, controls).mean()),
            )
        )
        rows.append(_row("H1", "top-minus-bottom quintile spread", s, **series_test(spread)))
        rows.append(_row("H1", "LLM rank IC above zero", s, **series_test(weekly_ic(sp, col))))
        if len(gap):
            part = gap.reset_index()[["week_id", "gap", "ic_a"]].assign(
                source=s, model=_model_of(s), prompt=_prompt_of(s)
            )
            stacked.append(part)
    src_week = pd.concat(stacked, ignore_index=True) if stacked else pd.DataFrame()
    if len(src_week):
        df = src_week.assign(y=src_week["gap"])
        c = prim.get("pooled_source_ic_gap")
        rows.append(
            _row(
                "H1",
                "pooled LLM minus baseline IC (model FE)",
                "pooled",
                cid=c.id if c else "",
                **pooled_regression(df, "y ~ 1 + C(model, Sum)", "Intercept"),
            )
        )
        df = src_week.assign(y=src_week["ic_a"])
        rows.append(
            _row(
                "H1",
                "pooled LLM rank IC above zero (model FE)",
                "pooled",
                **pooled_regression(df, "y ~ 1 + C(model, Sum)", "Intercept"),
            )
        )

    # H2 signal: steered tilt vs Q0 tilt, same stocks (G14).
    tp = tilt_panel(data.weeks, data.sources)
    if len(tp):
        tp = tp.merge(
            panel[
                [
                    "week_id",
                    "ticker",
                    "ret",
                    *[c for c in panel.columns if c.endswith("_z") or c == "sector"],
                ]
            ],
            on=["week_id", "ticker"],
            how="inner",
        )
        tilts = []
        for s in data.sources:
            col = f"tilt:{s}"
            if col not in tp:
                continue
            clean = data.source_clean_mask(s)
            sp = tp[tp["week_id"].isin([w for w, ok in clean.items() if ok])]
            gap = weekly_gap(sp, col, "tilt:Q0")
            ratio = data.conf_only_ratio(s)
            rows.append(
                _gap_row(
                    "H2",
                    "steered tilt minus Q0 tilt rank IC (same stocks)",
                    s,
                    gap,
                    cfg,
                    mean_conf_only_uncertainty_ratio=ratio,
                )
            )
            rows.append(
                _row(
                    "H2",
                    "steered tilt residual IC (Fama-MacBeth)",
                    s,
                    **series_test(
                        fama_macbeth(sp, col, [c for c in controls if c != "baseline_score"])
                    ),
                )
            )
            if len(gap):
                tilts.append(gap.reset_index()[["week_id", "gap"]].assign(model=_model_of(s)))
        if tilts:
            df = pd.concat(tilts, ignore_index=True)
            c = prim.get("pooled_tilt_ic_gap")
            rows.append(
                _row(
                    "H2",
                    "pooled steered minus Q0 tilt IC (model FE)",
                    "pooled",
                    cid=c.id if c else "",
                    **pooled_regression(
                        df.assign(y=df["gap"]), "y ~ 1 + C(model, Sum)", "Intercept"
                    ),
                )
            )

    # H3 signal: per model skills vs bare and vs placebo; pooled with model and week FE.
    if len(src_week):
        ic = src_week.assign(y=src_week["ic_a"])
        skills = {p for p in ic["prompt"].unique() if p.startswith("skills")}
        for model, g in ic.groupby("model"):
            for control in ("bare", "placebo"):
                ctrl = g[g["prompt"] == control].set_index("week_id")
                for p in sorted(skills):
                    sk = g[g["prompt"] == p].set_index("week_id")
                    pair = data.stock_week
                    a, b = likely_col(f"{model}__{p}"), likely_col(f"{model}__{control}")
                    if a in pair and b in pair and len(ctrl) and len(sk):
                        rows.append(
                            _gap_row(
                                "H3",
                                f"{p} minus {control} rank IC (same stocks)",
                                f"{model}",
                                weekly_gap(pair, a, b),
                                cfg,
                                prompt_id=p,
                                control=control,
                            )
                        )
        for control in ("bare", "placebo"):
            sub = ic[ic["prompt"].isin(skills | {control})].assign(
                skills=lambda d: d["prompt"].isin(skills).astype(float)
            )
            c = prim.get(f"pooled_prompt_ic_gap:{control}")
            rows.append(
                _row(
                    "H3",
                    f"pooled skills minus {control} IC (model and week FE)",
                    "pooled",
                    cid=c.id if c else "",
                    **pooled_regression(sub, "y ~ skills + C(model) + C(week_id)", "skills"),
                )
            )

        # Model comparisons on the preregistered prompt, and pooled model effects.
        main = prereg.main_model_comparison_prompt
        models = sorted(ic.loc[ic["prompt"] == main, "model"].unique())
        for m1, m2 in itertools.combinations(models, 2):
            a, b = likely_col(f"{m1}__{main}"), likely_col(f"{m2}__{main}")
            if a in panel and b in panel:
                rows.append(
                    _gap_row(
                        "H3",
                        f"{m1} minus {m2} rank IC on {main}",
                        f"{m1}|{m2}",
                        weekly_gap(panel, a, b),
                        cfg,
                    )
                )
        if len(models) > 1:
            for m in models[1:]:
                df = ic.assign(is_m=(ic["model"] == m).astype(float))
                rows.append(
                    _row(
                        "H3",
                        f"pooled {m} minus {models[0]} IC (prompt and week FE)",
                        "pooled",
                        **pooled_regression(
                            df[df["model"].isin([models[0], m])],
                            "y ~ is_m + C(prompt) + C(week_id)",
                            "is_m",
                        ),
                    )
                )

    # H3b: narrow minus wide (kept in hypotheses.h3b_width), confidence interaction, calibration.
    for s in data.sources:
        col, wcol = likely_col(s), f"width:{s}"
        if col not in panel or wcol not in panel:
            continue
        inter, calib = {}, {}
        for week, g in panel.groupby("week_id", sort=True):
            sub = g[[col, wcol, "ret"]].dropna()
            if len(sub) < MIN_STOCKS_FOR_IC or sub[wcol].nunique() < 2:
                continue
            z = _z(sub[col].astype(float))
            conf = (1 - sub[wcol] / 10).clip(0.1, 1.0)
            cz = _z(conf)
            y = sub["ret"].rank().to_numpy(dtype=float)
            x = np.column_stack([np.ones(len(sub)), z, cz, z * cz])
            coef, *_ = np.linalg.lstsq(x, (y - y.mean()) / (y.std(ddof=1) or 1), rcond=None)
            inter[week] = float(coef[3])
            err = (sub[col].rank(pct=True) - sub["ret"].rank(pct=True)).abs()
            calib[week] = float(stats.spearmanr(sub[wcol], err).statistic)
        rows.append(
            _row(
                "H3b",
                "score by confidence interaction (Fama-MacBeth)",
                s,
                **series_test(pd.Series(inter, dtype=float)),
            )
        )
        rows.append(
            _row(
                "H3b",
                "width vs absolute rank error (Spearman)",
                s,
                **series_test(pd.Series(calib, dtype=float)),
            )
        )

    out = pd.DataFrame(rows)
    if "p" not in out:
        out["p"] = math.nan
    ids = {c.id for c in prereg.primary_comparisons}
    return assign_families(
        out, ids, cfg.eval.multiple_testing.exploratory, significance=cfg.eval.significance
    )
