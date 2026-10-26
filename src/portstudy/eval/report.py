"""Write the evaluation report: CSV and Markdown tables plus PNG figures.

Always shows as-traded and clean-weeks-only results with fallback counts per portfolio, and
states the minimum detectable gap: gaps below it are inconclusive, not evidence of no effect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from portstudy.eval.data import EvalData, load_eval_data  # noqa: E402
from portstudy.eval.factors import factor_table  # noqa: E402
from portstudy.eval.hypotheses import (  # noqa: E402
    earnings_exposure,
    h1_llm_raw_vs_baseline,
    h2_combined_vs_baseline,
    h2_portfolio_vs_q0,
    h3_skills_vs_controls,
    h4_splits,
    portfolio_metrics,
)
from portstudy.eval.rank_ic import long_table  # noqa: E402
from portstudy.eval.shuffle import shuffle_test  # noqa: E402
from portstudy.io import write_csv_atomic, write_text_atomic  # noqa: E402
from portstudy.paths import report_dir  # noqa: E402
from portstudy.pipeline.context import RunContext  # noqa: E402


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


def _fig_gaps(h1: pd.DataFrame, folder: Path) -> None:
    if len(h1) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(h1["portfolio"], h1["mean_gap"], yerr=1.96 * h1["se_nw"].fillna(0), color="steelblue")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_ylabel("mean IC gap (llm_raw minus baseline)")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    fig.tight_layout()
    fig.savefig(folder / "fig_h1_gaps.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = pd.Series(0.0, index=h1.index)
    for col, color in (("wins", "seagreen"), ("ties", "lightgrey"), ("losses", "indianred")):
        ax.bar(h1["portfolio"], h1[col], bottom=bottom, label=col, color=color)
        bottom = bottom + h1[col]
    ax.set_ylabel("weeks")
    ax.legend()
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    fig.tight_layout()
    fig.savefig(folder / "fig_h1_win_tie_loss.png", dpi=120)
    plt.close(fig)


def _fig_equity(data: EvalData, folder: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for key, daily in data.daily_values.items():
        if len(daily) == 0:
            continue
        ax.plot(
            pd.to_datetime(daily["date"]),
            daily["value"],
            label=key,
            color="black" if key == "Q0" else None,
            linewidth=2 if key == "Q0" else 0.8,
            alpha=0.9,
        )
    ax.set_ylabel("portfolio value")
    ax.legend(fontsize=5, ncol=3)
    fig.tight_layout()
    fig.savefig(folder / "fig_equity_curves.png", dpi=120)
    plt.close(fig)


def _fig_shuffle(dists: pd.DataFrame, table: pd.DataFrame, folder: Path) -> None:
    if len(dists) == 0:
        return
    keys = list(table["portfolio"])
    n = len(keys)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.4 * rows), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        d = dists[dists["portfolio"] == key]["total_return"]
        ax.hist(d, bins=20, color="lightgrey")
        real = table.loc[table["portfolio"] == key, "real_total_return"].iloc[0]
        ax.axvline(real, color="red")
        ax.set_title(key, fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(folder / "fig_shuffle_histograms.png", dpi=120)
    plt.close(fig)


def build_report(
    data: EvalData, folder: Path, skip_shuffle: bool = False, n_shuffles: int | None = None
) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    tables["weekly_ic_long"] = long_table(data.ic)
    tables["fallback_counts"] = data.fallback_counts()
    tables["h1_ic_gap"] = h1_llm_raw_vs_baseline(data)
    tables["h2_ic_gap"] = h2_combined_vs_baseline(data)
    tables["h2_portfolio_vs_q0_clean"] = h2_portfolio_vs_q0(data, clean_only=True)
    tables["h2_portfolio_vs_q0_as_traded"] = h2_portfolio_vs_q0(data, clean_only=False)
    tables["h3_skills_vs_controls"] = h3_skills_vs_controls(data)
    tables["portfolio_metrics_as_traded"] = portfolio_metrics(data, clean_only=False)
    tables["portfolio_metrics_clean"] = portfolio_metrics(data, clean_only=True)
    for name, df in h4_splits(data).items():
        tables[f"h4_{name}"] = df
    tables["earnings_exposure"] = earnings_exposure(data)
    factors, source = factor_table(data.daily_values, data.weeks, data.cfg.eval.fama_french_file)
    tables["h5_factors"] = factors
    shuffle_table = pd.DataFrame()
    dists = pd.DataFrame()
    if not skip_shuffle:
        shuffle_table, dists = shuffle_test(
            data.cfg, data.portfolios, data.weeks, n_shuffles=n_shuffles
        )
        tables["h5_shuffle"] = shuffle_table
        tables["h5_shuffle_distributions"] = dists
    for name, df in tables.items():
        _save(df, name, folder)
    _fig_ic_series(data, folder)
    _fig_gaps(tables["h1_ic_gap"], folder)
    _fig_equity(data, folder)
    _fig_shuffle(dists, shuffle_table, folder)
    md = _markdown(data, tables, source, skip_shuffle)
    write_text_atomic(md, folder / "report.md")
    return {"folder": str(folder), "tables": sorted(tables), "factor_source": source}


def _markdown(
    data: EvalData, tables: dict[str, pd.DataFrame], factor_source: str, skip_shuffle: bool
) -> str:
    out: list[str] = ["# Evaluation report", ""]
    out.append(
        f"Weeks evaluated: {len(data.weeks)} ({data.weeks[0] if data.weeks else ''} to "
        f"{data.weeks[-1] if data.weeks else ''}). Pilot week excluded: {data.pilot_week}."
    )
    out.append("")
    out.append("## Fallback counts per portfolio")
    out.append(_md_table(tables["fallback_counts"]))
    gap_cols = [
        "portfolio",
        "n_weeks",
        "wins",
        "ties",
        "losses",
        "mean_gap",
        "se_nw",
        "p_nw",
        "p_perm",
        "p_holm",
        "significant",
        "min_detectable_gap",
    ]
    for title, key in (
        ("## H1: LLM raw IC minus baseline IC (clean weeks)", "h1_ic_gap"),
        ("## H2: combined IC minus baseline IC (clean weeks)", "h2_ic_gap"),
        (
            "## H3: skills vs bare and skills vs placebo (Holm across the family)",
            "h3_skills_vs_controls",
        ),
    ):
        out.append(title)
        df = tables[key]
        out.append(_md_table(df, [c for c in gap_cols if c in df.columns]))
        if len(df):
            mdg = df["min_detectable_gap"].median()
            below = df[(~df["significant"]) & (df["mean_gap"].abs() < df["min_detectable_gap"])]
            out.append(
                f"Minimum detectable gap (median across rows, from the observed weekly sd): "
                f"{mdg:.4f}. {len(below)} of {len(df)} non-significant gaps are below it and are "
                "inconclusive, not evidence of no effect."
            )
            out.append("")
    out.append("## H2: portfolio weekly return minus Q0")
    out.append("Clean weeks only:")
    port_cols = [
        "portfolio",
        "n_weeks",
        "wins",
        "ties",
        "losses",
        "mean_gap",
        "p_nw",
        "p_holm",
        "significant",
        "total_return",
        "sharpe",
        "max_drawdown",
        "turnover",
        "total_cost_usd",
    ]
    out.append(
        _md_table(
            tables["h2_portfolio_vs_q0_clean"],
            [c for c in port_cols if c in tables["h2_portfolio_vs_q0_clean"].columns],
        )
    )
    out.append("As traded (fallback weeks included):")
    out.append(
        _md_table(
            tables["h2_portfolio_vs_q0_as_traded"],
            [c for c in port_cols if c in tables["h2_portfolio_vs_q0_as_traded"].columns],
        )
    )
    out.append("## Portfolio metrics")
    out.append("As traded:")
    out.append(_md_table(tables["portfolio_metrics_as_traded"]))
    out.append("Clean weeks only:")
    out.append(_md_table(tables["portfolio_metrics_clean"]))
    out.append("## H4 (exploratory): splits")
    for name in ("market_direction", "volatility", "earnings_week"):
        key = f"h4_{name}"
        if key in tables and len(tables[key]):
            df = tables[key]
            out.append(f"### {name}")
            cols = [
                c
                for c in [
                    "group",
                    "portfolio",
                    "n_weeks",
                    "usable_weeks",
                    "wins",
                    "ties",
                    "losses",
                    "mean_gap",
                    "p_nw",
                    "p_holm",
                ]
                if c in df.columns
            ]
            out.append(_md_table(df, cols))
            if name == "earnings_week":
                out.append(
                    "Usable weeks per portfolio are listed above; outside earnings season this "
                    "split is mostly descriptive."
                )
                out.append("")
    out.append("### Earnings exposure (descriptive)")
    out.append(_md_table(tables["earnings_exposure"]))
    out.append(f"## H5: factor regression on daily returns (source: {factor_source})")
    if factor_source == "etf_proxies":
        out.append("Investment factor (CMA) unavailable from ETF proxies: reported as a gap.")
        out.append("")
    out.append(_md_table(tables["h5_factors"]))
    out.append("## H5: shuffled-ratings test")
    if skip_shuffle:
        out.append("_skipped (run without --skip-shuffle)_")
        out.append("")
    else:
        cols = [
            "portfolio",
            "n_shuffles",
            "real_total_return",
            "shuffle_mean_total_return",
            "percentile_total_return",
            "p_total_return",
            "p_total_return_holm",
            "significant_total_return",
            "real_sharpe",
            "p_sharpe_holm",
        ]
        out.append(
            _md_table(tables["h5_shuffle"], [c for c in cols if c in tables["h5_shuffle"].columns])
        )
    out.append("## Figures")
    for f in (
        "fig_weekly_ic.png",
        "fig_h1_gaps.png",
        "fig_h1_win_tie_loss.png",
        "fig_equity_curves.png",
        "fig_shuffle_histograms.png",
    ):
        out.append(f"- {f}")
    out.append("")
    return "\n".join(out)


def run_eval(
    ctx: RunContext,
    skip_shuffle: bool = False,
    n_shuffles: int | None = None,
    folder: Path | None = None,
) -> Path:
    data = load_eval_data(ctx.cfg, ctx.portfolio_keys())
    target = folder or report_dir()
    build_report(data, target, skip_shuffle=skip_shuffle, n_shuffles=n_shuffles)
    return target
