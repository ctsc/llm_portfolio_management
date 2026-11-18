"""Write the evaluation report (PLAN4 Phase 9): CSV for every table, `report.md` and PNG figures.

Order: header, Main results (the preregistered primary family with effect, CI, p, Holm p and
minimum detectable effect), H1, H2 (intention-to-treat and active weeks side by side), H3, model
comparisons, H3b, H4 (exploratory), H5, controls vs HRP and equal weight, portfolio metrics,
tie-band sensitivity, view sensitivity (G13), model-change sensitivity (G21), data integrity, the
study log summary and how to read a null. `all_metrics.csv` (G22) holds every metric for every
unit; the evaluation fails when an expected (unit, metric) pair is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from portstudy.config import Preregistration, load_preregistration  # noqa: E402
from portstudy.eval.all_metrics import build_all_metrics, check_completeness  # noqa: E402
from portstudy.eval.data import EvalData, load_eval_data  # noqa: E402
from portstudy.eval.hypotheses import (  # noqa: E402
    earnings_exposure,
    h3b_width,
    h4_splits,
    h4_strategy_ranking,
    portfolio_metrics,
)
from portstudy.eval.multiple import assign_families  # noqa: E402
from portstudy.eval.portfolio_tests import portfolio_tests  # noqa: E402
from portstudy.eval.rank_ic import long_table  # noqa: E402
from portstudy.eval.signal_tests import signal_tests  # noqa: E402
from portstudy.io import read_csv, write_csv_atomic, write_text_atomic  # noqa: E402
from portstudy.paths import data_dir, report_dir  # noqa: E402
from portstudy.pipeline.context import RunContext  # noqa: E402
from portstudy.types import PortfolioId  # noqa: E402

SECTIONS = [
    "## Header",
    "## Main results",
    "## H1: LLM signal vs the baseline",
    "## H2: steered portfolios vs their own Q0",
    "## H3: skills vs bare and placebo",
    "## Model comparisons",
    "## H3b: range width",
    "## H4 (exploratory)",
    "## H5: factor model and the placebo control",
    "## Controls: HRP and equal weight",
    "## Portfolio metrics",
    "## Tie-band sensitivity",
    "## View sensitivity (G13)",
    "## Model-change sensitivity (G21)",
    "## Data integrity",
    "## Study log summary",
    "## How to read a null",
]
TEST_COLS = ["comparison", "unit", "estimate", "ci_low", "ci_high", "p", "p_adj", "family", "n"]


class EvalIncompleteError(RuntimeError):
    """`all_metrics.csv` is missing an expected (unit, metric) pair."""


def _md_table(df: pd.DataFrame, cols: list[str] | None = None, floatfmt: str = ".4f") -> str:
    if len(df) == 0:
        return "_no rows_\n"
    use = df.loc[:, cols] if cols else df
    header = "| " + " | ".join(str(c) for c in use.columns) + " |"
    sep = "|" + "|".join(["---"] * len(use.columns)) + "|"
    lines = [header, sep]
    for _, row in use.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:{floatfmt}}" if pd.notna(v) else "")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _save(df: pd.DataFrame, name: str, folder: Path) -> None:
    write_csv_atomic(df, folder / f"{name}.csv")


def _fig_ic_series(data: EvalData, folder: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    base = data.ic["baseline"]["baseline"]
    ax.plot(range(len(base)), base.to_numpy(), label="baseline", color="black", linewidth=2)
    raw = data.ic["llm_raw"]
    for key in raw.columns:
        ax.plot(range(len(raw)), raw[key].reindex(base.index).to_numpy(), alpha=0.5, label=key)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xticks(range(len(base)))
    ax.set_xticklabels(list(base.index), rotation=90, fontsize=6)
    ax.set_ylabel("weekly rank IC")
    ax.legend(fontsize=5, ncol=3)
    fig.tight_layout()
    fig.savefig(folder / "fig_weekly_ic.png", dpi=120)
    plt.close(fig)


def _fig_bars_ci(df: pd.DataFrame, label_col: str, ylabel: str, path: Path) -> None:
    """Bars of `estimate` with their CI (`ci_low`, `ci_high`)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    if len(df):
        est = df["estimate"].astype(float).fillna(0.0)
        lo = (est - df["ci_low"].astype(float)).clip(lower=0).fillna(0.0)
        hi = (df["ci_high"].astype(float) - est).clip(lower=0).fillna(0.0)
        ax.bar(
            df[label_col].astype(str), est, yerr=[lo.to_numpy(), hi.to_numpy()], color="steelblue"
        )
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_loadings(h5: pd.DataFrame, data: EvalData, path: Path) -> None:
    """Mean factor loading of each strategy's portfolios (excess-return regressions)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    cols = [c for c in h5.columns if c.startswith("beta_")]
    if len(h5) and cols:
        h5 = h5.assign(strategy=h5["unit"].map(lambda k: str(k).split("__")[0]))
        means = h5.groupby("strategy")[cols].mean()
        means.T.plot(kind="bar", ax=ax)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_ylabel("factor loading")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_equity(data: EvalData, keys: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for key in keys:
        daily = data.daily_values.get(key, pd.DataFrame())
        if len(daily) == 0:
            continue
        pid = PortfolioId.parse(key)
        ax.plot(
            pd.to_datetime(daily["date"]),
            daily["value"],
            label=key,
            color="black" if pid.is_q0 else "grey" if pid.is_control else None,
            linewidth=2 if pid.is_q0 else 1.5 if pid.is_control else 0.8,
            linestyle="--" if pid.is_control else "-",
            alpha=0.9,
        )
    ax.set_ylabel("portfolio value")
    ax.legend(fontsize=5, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_equity(data: EvalData, folder: Path) -> None:
    _plot_equity(data, list(data.daily_values), folder / "fig_equity_curves.png")
    for strategy in data.strategies():
        keys = [data.q0_key(strategy), *data.steered_for(strategy), *data.controls]
        _plot_equity(data, keys, folder / f"fig_equity_{strategy}.png")


def _fig_h3b(h3b: pd.DataFrame, folder: Path) -> None:
    if len(h3b) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(h3b))
    ax.bar([i - 0.2 for i in x], h3b["mean_ic_narrow"], width=0.4, label="narrow", color="seagreen")
    ax.bar([i + 0.2 for i in x], h3b["mean_ic_wide"], width=0.4, label="wide", color="indianred")
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(h3b["source"]), rotation=90, fontsize=6)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_ylabel("mean weekly rank IC by width group")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "fig_h3b_width.png", dpi=120)
    plt.close(fig)


def _fig_h4_ranking(ranking: pd.DataFrame, folder: Path) -> None:
    if len(ranking) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    err = [
        (ranking["mean_gap"] - ranking["ci_low"]).clip(lower=0).fillna(0).to_numpy(),
        (ranking["ci_high"] - ranking["mean_gap"]).clip(lower=0).fillna(0).to_numpy(),
    ]
    ax.bar(ranking["strategy"], ranking["mean_gap"], yerr=err, color="steelblue")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_ylabel("mean weekly gap vs own Q0")
    fig.tight_layout()
    fig.savefig(folder / "fig_h4_ranking.png", dpi=120)
    plt.close(fig)


def _cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _mde(r: Any) -> float:
    mdg = r.get("mdg", np.nan)
    if pd.notna(mdg):
        return float(mdg)
    se = r.get("se", np.nan)
    return float(2.8 * se) if pd.notna(se) else float("nan")


def main_results(signal: pd.DataFrame, prereg: Preregistration) -> pd.DataFrame:
    """One row per preregistered comparison (missing ones are kept with a reason)."""
    rows = []
    prim = (
        signal[signal["family"] == "primary"].set_index("comparison_id")
        if len(signal)
        else (pd.DataFrame())
    )
    for c in prereg.primary_comparisons:
        if c.id in prim.index:
            r = prim.loc[c.id]
            rows.append(
                {
                    "comparison_id": c.id,
                    "hypothesis": c.hypothesis,
                    "kind": c.kind,
                    "estimate": r.get("estimate"),
                    "ci_low": r.get("ci_low", np.nan),
                    "ci_high": r.get("ci_high", np.nan),
                    "se": r.get("se", np.nan),
                    "p": r.get("p"),
                    "p_holm": r.get("p_adj"),
                    "n": r.get("n"),
                    "n_weeks": r.get("n_weeks", np.nan),
                    "min_detectable_effect": _mde(r),
                    "reason": "",
                }
            )
        else:
            rows.append(
                {
                    "comparison_id": c.id,
                    "hypothesis": c.hypothesis,
                    "kind": c.kind,
                    "reason": "no estimate (no sources or weeks for this comparison)",
                }
            )
    df = pd.DataFrame(rows)
    if "se" in df:
        crit = 1.96
        df["ci_low"] = df["ci_low"].fillna(df["estimate"] - crit * df["se"])
        df["ci_high"] = df["ci_high"].fillna(df["estimate"] + crit * df["se"])
    return df


def tie_band_sensitivity(data: EvalData) -> pd.DataFrame:
    rows = []
    for key in data.portfolios:
        strategy = data.strategy_of.get(key)
        q0 = data.q0_key(strategy) if strategy else ""
        if q0 not in data.weekly_returns:
            continue
        gap = data.weekly_returns[key] - data.weekly_returns[q0]
        gap = gap[data.clean_mask(key).reindex(gap.index).fillna(False).astype(bool)].dropna()
        for band in data.cfg.eval.tie_band_sensitivity:
            rows.append(
                {
                    "portfolio": key,
                    "tie_band": band,
                    "wins": int((gap > band).sum()),
                    "ties": int(((gap >= -band) & (gap <= band)).sum()),
                    "losses": int((gap < -band).sum()),
                }
            )
    return pd.DataFrame(rows)


def data_integrity(data: EvalData) -> pd.DataFrame:
    """Per week: universe size, removals, large-move flags and data-error skips (Phase 3A)."""
    from portstudy.data.snapshot import read_snapshot, snapshot_exists

    rows: list[dict[str, Any]] = []
    for w in data.weeks:
        if not snapshot_exists(w):
            rows.append({"week_id": w, "reason": "no snapshot"})
            continue
        man: dict[str, Any] = dict(read_snapshot(w).manifest)
        removed = man.get("removed") or []
        skipped = man.get("skipped") or []
        rows.append(
            {
                "week_id": w,
                "n_included": man.get("n_included", np.nan),
                "n_removed": len(removed) if isinstance(removed, list) else 0,
                "n_large_move": man.get("n_large_move", 0),
                "n_skipped": len(skipped) if isinstance(skipped, list) else 0,
                "reason": "",
            }
        )
    return pd.DataFrame(rows)


def study_log_summary() -> pd.DataFrame:
    from portstudy.pipeline.study_log import read_study_log

    log = read_study_log()
    cols = [
        "week_id",
        "failed_job",
        "universe_size",
        "n_removed",
        "n_fallback_rows",
        "litellm_cost_total",
        "arctic_gpu_hours_total",
        "audit_ok",
    ]
    return log.loc[:, _cols(log, cols)] if len(log) else pd.DataFrame(columns=cols)


def repeat_agreement() -> pd.DataFrame:
    path = data_dir() / "outputs" / "pilot_repeatability.csv"
    return read_csv(path) if path.exists() else pd.DataFrame()


def build_report(data: EvalData, folder: Path, prereg: Preregistration) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    signal = signal_tests(data, prereg)
    ports = portfolio_tests(data, prereg)
    tables["signal_tests"] = signal
    tables["portfolio_tests"] = ports["portfolio_tests"]
    tables["view_sensitivity"] = ports["view_sensitivity"]
    tables["model_change_weeks"] = ports["model_change_weeks"]
    tables["main_results"] = main_results(signal, prereg)
    tables["weekly_ic_long"] = long_table(data.ic) if data.ic else pd.DataFrame()
    tables["fallback_counts"] = data.fallback_counts()
    h3b = h3b_width(data)
    tables["h3b_width"] = (
        assign_families(h3b.assign(hypothesis="H3b"), set(), p_col="p_perm") if len(h3b) else h3b
    )
    tables["portfolio_metrics"] = portfolio_metrics(data, clean_only=False)
    tables["portfolio_metrics_clean"] = portfolio_metrics(data, clean_only=True)
    ranking = h4_strategy_ranking(data)
    tables["h4_strategy_ranking"] = ranking
    for name, df in h4_splits(data).items():
        tables[f"h4_{name}"] = df.assign(family="exploratory") if len(df) else df
    tables["earnings_exposure"] = earnings_exposure(data)
    tables["tie_band_sensitivity"] = tie_band_sensitivity(data)
    tables["data_integrity"] = data_integrity(data)
    tables["study_log_summary"] = study_log_summary()
    tables["repeat_agreement"] = repeat_agreement()
    allm = build_all_metrics(data, tables["portfolio_metrics"], signal, ports["portfolio_tests"])
    tables["all_metrics"] = allm
    for name, df in tables.items():
        write_csv_atomic(df, folder / f"{name}.csv")
    pt = ports["portfolio_tests"]
    _fig_ic_series(data, folder)
    h1 = signal[signal["comparison"] == "LLM minus baseline rank IC (same stocks)"]
    _fig_bars_ci(h1, "unit", "IC gap vs baseline", folder / "fig_ic_gaps.png")
    _fig_equity(data, folder)
    itt = pt[pt["comparison"] == "steered minus Q0 weekly return (intention-to-treat)"]
    for strategy in data.strategies():
        _fig_bars_ci(
            itt[itt["strategy"] == strategy] if "strategy" in itt else itt,
            "unit",
            "mean weekly active return",
            folder / f"fig_active_{strategy}.png",
        )
    _fig_h3b(h3b, folder)
    _fig_h4_ranking(ranking, folder)
    _fig_loadings(
        pt[pt["comparison"] == "excess return factor regression"],
        data,
        folder / "fig_factor_loadings.png",
    )
    write_text_atomic(_markdown(data, tables, prereg), folder / "report.md")
    missing = check_completeness(data, allm)
    if missing:
        raise EvalIncompleteError(
            f"all_metrics.csv is missing {len(missing)} pairs, for example: {missing[:5]}"
        )
    return {"folder": str(folder), "tables": sorted(tables), "factor_source": data.factor_source}


def _header(data: EvalData, prereg: Preregistration) -> list[str]:
    cfg = data.cfg
    models = sorted({s.split("__")[0] for s in data.sources})
    prompts = sorted({s.split("__")[1] for s in data.sources})
    st = data.statuses[[k for k in data.portfolios if k in data.statuses]]
    fb_share = float((st == "fallback").to_numpy().mean()) if st.size else float("nan")
    act = data.active_view[[k for k in data.portfolios if k in data.active_view]]
    act_share = float(act.to_numpy().mean()) if act.size else float("nan")
    frozen = prereg.frozen_on if prereg.frozen else "not frozen"
    lines = [
        SECTIONS[0],
        "",
        f"- Weeks: {len(data.weeks)} ({data.weeks[0] if data.weeks else ''} to "
        f"{data.weeks[-1] if data.weeks else ''}); pilot week excluded: {data.pilot_week}",
        f"- Models: {', '.join(models)}; prompts: {', '.join(prompts)}; strategies: "
        f"{', '.join(data.strategies())}; controls: {', '.join(data.controls)}; "
        f"confidence-only portfolios: {len(data.conf_only)}",
        f"- Factor source: {data.factor_source or 'none'}",
        f"- Pre-registration: {frozen}",
        f"- Share of steered portfolio-weeks on fallback: {fb_share:.3f}; with an active LLM "
        f"view: {act_share:.3f}",
        "- Decision values in force: "
        + "; ".join(
            f"{d.id}={d.value}"
            for d in cfg.decisions
            if d.id in ("B4", "E7", "A8", "S4", "S8", "G7", "G12", "G13", "N4")
        ),
        "",
    ]
    return lines


def _test_section(title: str, df: pd.DataFrame, note: str = "") -> list[str]:
    out = [title, ""]
    if note:
        out += [note, ""]
    out.append(
        _md_table(
            df,
            _cols(
                df,
                TEST_COLS + ["mean_n_stocks", "analysis", "share_active_weeks", "view_sensitivity"],
            ),
        )
    )
    return out


def _markdown(data: EvalData, tables: dict[str, pd.DataFrame], prereg: Preregistration) -> str:
    s, p = tables["signal_tests"], tables["portfolio_tests"]
    out: list[str] = ["# Evaluation report", "", *_header(data, prereg)]
    main = tables["main_results"]
    out += [
        SECTIONS[1],
        "",
        "The four preregistered pooled comparisons, Holm-adjusted together "
        "(DECISION(G7)). Every other test is in the Benjamini-Hochberg secondary family; H4 "
        "is exploratory.",
        "",
        _md_table(
            main,
            _cols(
                main,
                [
                    "comparison_id",
                    "hypothesis",
                    "estimate",
                    "ci_low",
                    "ci_high",
                    "p",
                    "p_holm",
                    "n_weeks",
                    "min_detectable_effect",
                    "reason",
                ],
            ),
        ),
    ]
    out += _test_section(
        SECTIONS[2],
        s[s["hypothesis"] == "H1"],
        "Every IC gap is on the stocks valid in both signals each week (G14).",
    )
    h2 = p[
        p["comparison"].str.startswith("steered minus Q0")
        & ~p["comparison"].str.contains("excluding")
    ]
    out += _test_section(
        SECTIONS[3],
        pd.concat([s[s["hypothesis"] == "H2"], h2, p[p["comparison"].str.contains("confidence")]]),
        "Intention-to-treat (all clean weeks, realized gap every week) and "
        "active weeks only are listed side by side (G8); the information vs "
        "confidence split covers Robust and Resampled MVO (G19).",
    )
    h3 = p[(p["hypothesis"] == "H3") & ~p["comparison"].str.contains(" minus .* on ", regex=True)]
    out += _test_section(
        SECTIONS[4],
        pd.concat([s[(s["hypothesis"] == "H3") & ~s["comparison"].str.contains(" on ")], h3]),
    )
    out += _test_section(
        SECTIONS[5],
        pd.concat(
            [
                s[
                    s["comparison"].str.contains(" on ")
                    | s["comparison"].str.startswith("pooled m")
                ],
                p[p["comparison"].str.contains(" on ")],
            ]
        ),
    )
    out += _test_section(SECTIONS[6], pd.concat([s[s["hypothesis"] == "H3b"], tables["h3b_width"]]))
    out += _test_section(
        SECTIONS[7],
        p[p["hypothesis"] == "H4"],
        "Exploratory: no multiple-testing adjustment. Ranking and splits are in "
        "`h4_strategy_ranking.csv` and `h4_*.csv`.",
    )
    out += _test_section(
        SECTIONS[8],
        p[p["hypothesis"] == "H5"],
        "The placebo controls for prompt length and any change to the views; "
        "chance is handled by the tests and intervals, market and style luck by "
        "Q0 and the factor model.",
    )
    out += _test_section(SECTIONS[9], p[p["comparison"].str.startswith("weekly return minus")])
    met = tables["portfolio_metrics"]
    out += [
        SECTIONS[10],
        "",
        _md_table(
            met,
            _cols(
                met,
                [
                    "portfolio",
                    "sharpe_excess",
                    "psr",
                    "sortino",
                    "calmar",
                    "max_drawdown_daily",
                    "cvar_95",
                    "volatility",
                    "cer_gamma_2",
                    "turnover",
                    "total_cost_usd",
                    "fallback_weeks",
                ],
            ),
        ),
    ]
    out += [SECTIONS[11], "", _md_table(tables["tie_band_sensitivity"])]
    out += [SECTIONS[12], "", _md_table(tables["view_sensitivity"])]
    out += [
        SECTIONS[13],
        "",
        "Flagged weeks per model (the H2 tests are repeated without them):",
        "",
        _md_table(tables["model_change_weeks"]),
        "Pilot repeat agreement (G20):",
        "",
        _md_table(tables["repeat_agreement"]),
    ]
    out += [
        SECTIONS[14],
        "",
        _md_table(tables["data_integrity"]),
        _md_table(tables["fallback_counts"]),
    ]
    out += [SECTIONS[15], "", _md_table(tables["study_log_summary"])]
    null_lines = [
        f"- {r.comparison_id}: minimum detectable effect "
        f"{getattr(r, 'min_detectable_effect', float('nan')):.4f}"
        for r in main.itertuples()
    ]
    out += [
        SECTIONS[16],
        "",
        "A result whose interval includes zero is inconclusive, not "
        "evidence of no effect, when the minimum detectable effect is larger than an "
        "economically meaningful effect. Strategies labeled insensitive to views by "
        "construction say nothing about whether the LLM helps.",
        "",
        *null_lines,
        "",
    ]
    return "\n".join(out)


def run_eval(ctx: RunContext, folder: Path | None = None, factor_source: str | None = None) -> Path:
    cfg = ctx.cfg
    if factor_source:
        cfg = cfg.model_copy(
            update={"eval": cfg.eval.model_copy(update={"factor_source": factor_source})}
        )
    data = load_eval_data(cfg, ctx.portfolio_keys(), synthetic=ctx.synthetic)
    target = folder or report_dir()
    build_report(data, target, load_preregistration(cfg=cfg))
    return target
