"""H1 to H4 over rating sources and strategies. Every function returns a tidy DataFrame.

Families (config `eval.holm_families`): H1 and H2 IC gaps are per source and Holm-adjusted
across all sources; H2 portfolio returns pair each steered portfolio with its own strategy's Q0
and are Holm-adjusted within each strategy; H3 compares every `skills` prompt type with `bare`
and `placebo` per model (Holm per model); H3b splits each source-week at the median range width
(DECISION(G3b)); H4 ranks strategies by their mean weekly gap vs their own Q0 (exploratory,
no Holm). Controls (HRP, EW) appear in metric and factor tables only.

The primary p-value for each paired weekly test is the exact sign-flip permutation p (`p_perm`),
which keeps its nominal size at 20 to 30 weeks; the Newey-West p (`p_nw`) is reported beside it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portstudy.eval.data import EvalData, likely_column, max_drawdown, sharpe, width_column
from portstudy.eval.tests import bootstrap_ci, holm, paired_summary, rank_ic_series
from portstudy.llm.prompts import load_manifest
from portstudy.types import RatingSource

CONTROL_IDS = ("bare", "placebo")


def _holm_within(df: pd.DataFrame, group: str | None, significance: float) -> pd.DataFrame:
    """Holm-adjust `p_perm` across the whole frame, or within each value of `group`."""
    if not len(df):
        return df
    df["p_holm"] = float("nan")
    if group is None:
        df["p_holm"] = holm(df["p_perm"].to_numpy())
    else:
        for value in df[group].unique():
            mask = df[group] == value
            df.loc[mask, "p_holm"] = holm(df.loc[mask, "p_perm"].to_numpy())
    df["significant"] = df["p_holm"] < significance
    return df


def _gap_rows(
    data: EvalData,
    gaps: dict[str, pd.Series],
    family: str,
    label: str,
    groups: dict[str, str] | None = None,
    group_name: str = "group",
) -> pd.DataFrame:
    rows = []
    for key, gap in gaps.items():
        summary = paired_summary(
            gap,
            data.cfg.eval.tie_band,
            data.cfg.eval.seed,
            n_perm=data.cfg.eval.permutation_samples,
            n_boot=data.cfg.eval.bootstrap_samples,
        )
        row = {"family": family, "comparison": label, "portfolio": key, **summary}
        if groups is not None:
            row[group_name] = groups.get(key, "")
        rows.append(row)
    df = pd.DataFrame(rows)
    return _holm_within(df, group_name if groups is not None else None, data.cfg.eval.significance)


def h1_llm_raw_vs_baseline(data: EvalData) -> pd.DataFrame:
    """H1: weekly gap IC_llm_raw - IC_baseline per rating source; Holm across all sources."""
    base = data.ic["baseline"]["baseline"]
    raw = data.ic["llm_raw"]
    gaps = {k: (raw[k] - base).dropna() for k in data.sources if k in raw}
    df = _gap_rows(data, gaps, "H1_ic_gap", "llm_raw minus baseline")
    if len(df):
        df["source"] = df["portfolio"]
    return df


def h2_combined_vs_baseline(data: EvalData) -> pd.DataFrame:
    """H2 (IC): weekly gap IC_combined - IC_baseline per source (E3: combined = ranked BL tilt)."""
    base = data.ic["baseline"]["baseline"]
    comb = data.ic["combined"]
    gaps = {k: (comb[k] - base).dropna() for k in data.sources if k in comb}
    df = _gap_rows(data, gaps, "H2_ic_gap", "combined minus baseline")
    if len(df):
        df["source"] = df["portfolio"]
    return df


def portfolio_metrics(data: EvalData, clean_only: bool) -> pd.DataFrame:
    """Return, Sharpe, drawdown, turnover, cost, stops, mean view shift, fallback and
    solver-failed weeks for every portfolio, controls included."""
    rows = []
    for key in data.portfolio_keys:
        weekly = data.weekly_returns[key] if key in data.weekly_returns else pd.Series(dtype=float)
        is_steered = key in data.source_of
        if clean_only and is_steered:
            weekly = weekly[data.clean_mask(key).reindex(weekly.index).fillna(False)]
        daily = data.daily_values.get(key, pd.DataFrame())
        trades = data.trades.get(key, pd.DataFrame())
        values = daily["value"] if len(daily) else pd.Series(dtype=float)
        avg_value = float(values.mean()) if len(values) else float("nan")
        statuses = data.statuses[key] if key in data.statuses else pd.Series(dtype=object)
        failed = data.solver_failed[key] if key in data.solver_failed else pd.Series(dtype=bool)
        shift = data.view_shift[key] if key in data.view_shift else pd.Series(dtype=float)
        rows.append(
            {
                "portfolio": key,
                "strategy": data.strategy_of.get(key, key if key in data.controls else ""),
                "source": data.source_of.get(key, ""),
                "n_weeks": int(weekly.notna().sum()),
                "total_return": float(np.prod(1.0 + weekly.dropna().to_numpy(dtype=float)) - 1.0)
                if len(weekly)
                else float("nan"),
                "sharpe": sharpe(weekly),
                "max_drawdown": max_drawdown(values),
                "turnover": float(trades["trade_value"].sum() / avg_value)
                if len(trades) and avg_value
                else float("nan"),
                "total_cost_usd": float(trades["cost_usd"].sum()) if len(trades) else 0.0,
                "n_stop_loss": int((trades["reason"] == "stop_loss").sum()) if len(trades) else 0,
                "n_take_profit": int((trades["reason"] == "take_profit").sum())
                if len(trades)
                else 0,
                "mean_view_shift": float(shift.mean()) if shift.notna().any() else float("nan"),
                "fallback_weeks": int((statuses == "fallback").sum()),
                "solver_failed_weeks": int(failed.sum()) if len(failed) else 0,
            }
        )
    return pd.DataFrame(rows)


def h2_portfolio_vs_q0(data: EvalData, clean_only: bool = True) -> pd.DataFrame:
    """H2 (portfolios): paired weekly return minus the strategy's own Q0 on the source's clean
    weeks; Holm within each strategy."""
    gaps: dict[str, pd.Series] = {}
    groups: dict[str, str] = {}
    for key in data.portfolios:
        strategy = data.strategy_of.get(key)
        if key not in data.weekly_returns or strategy is None:
            continue
        q0_key = data.q0_key(strategy)
        if q0_key not in data.weekly_returns:
            continue
        gap = data.weekly_returns[key] - data.weekly_returns[q0_key]
        if clean_only:
            gap = gap[data.clean_mask(key).reindex(gap.index).fillna(False)]
        gaps[key] = gap.dropna()
        groups[key] = strategy
    df = _gap_rows(
        data, gaps, "H2_portfolio_vs_q0", "weekly return minus Q0", groups, group_name="strategy"
    )
    if len(df):
        df["source"] = df["portfolio"].map(data.source_of)
        metrics = portfolio_metrics(data, clean_only).set_index("portfolio")
        for col in (
            "sharpe",
            "max_drawdown",
            "turnover",
            "total_cost_usd",
            "total_return",
            "mean_view_shift",
            "solver_failed_weeks",
        ):
            df[col] = df["portfolio"].map(metrics[col])
    return df


def skills_prompt_ids() -> list[str]:
    """Enabled prompt configurations of kind `skills`, from the manifest."""
    return [conf.id for conf in load_manifest().by_kind("skills")]


def h3_skills_vs_controls(data: EvalData, signal: str = "llm_raw") -> pd.DataFrame:
    """H3 (G3): per model, every `skills` prompt type vs bare and vs placebo; Holm per model."""
    table = data.ic[signal]
    skills = set(skills_prompt_ids())
    by_model: dict[str, dict[str, str]] = {}
    for key in data.sources:
        source = RatingSource.parse(key)
        by_model.setdefault(source.model, {})[source.config_id] = key
    gaps: dict[str, pd.Series] = {}
    groups: dict[str, str] = {}
    for model, confs in by_model.items():
        for conf, key in confs.items():
            if conf not in skills or key not in table:
                continue
            for control in CONTROL_IDS:
                ckey = confs.get(control)
                if ckey is None or ckey not in table:
                    continue
                label = f"{model}: {conf} vs {control}"
                gaps[label] = (table[key] - table[ckey]).dropna()
                groups[label] = model
    df = _gap_rows(data, gaps, "H3_skills_vs_controls", f"{signal} IC gap", groups, "model")
    if len(df):
        df["prompt_id"] = df["portfolio"].str.split(": ").str[1].str.split(" vs ").str[0]
        df["control"] = df["portfolio"].str.split(" vs ").str[1]
    return df


def h3b_width(data: EvalData) -> pd.DataFrame:
    """H3b (G3b): per source and week, rank IC of `likely` for stocks with width at or below the
    median minus the IC for the rest; weeks with a group under `eval.h3b.min_group_size` or with
    every width equal are skipped and counted. Holm across sources."""
    sw = data.stock_week
    min_group = data.cfg.eval.h3b.min_group_size
    rows = []
    gaps_by_source: dict[str, pd.Series] = {}
    detail: dict[str, dict[str, object]] = {}
    for source in data.sources:
        lcol, wcol = likely_column(source), width_column(source)
        if len(sw) == 0 or lcol not in sw.columns or wcol not in sw.columns:
            continue
        clean = data.source_clean_mask(source)
        gaps: list[float] = []
        narrow_ics: list[float] = []
        wide_ics: list[float] = []
        skipped_small, skipped_equal, skipped_nan = 0, 0, 0
        for week, group in sw.groupby("week_id"):
            if not bool(clean.get(week, False)):
                continue
            valid = group[[lcol, wcol, "ret"]].dropna()
            if len(valid) == 0:
                continue
            widths = valid[wcol].to_numpy(dtype=float)
            if np.all(widths == widths[0]):
                skipped_equal += 1
                continue
            median = float(np.median(widths))
            narrow = valid[valid[wcol] <= median]
            wide = valid[valid[wcol] > median]
            if len(narrow) < min_group or len(wide) < min_group:
                skipped_small += 1
                continue
            ic_narrow = rank_ic_series(narrow[lcol], narrow["ret"])
            ic_wide = rank_ic_series(wide[lcol], wide["ret"])
            if pd.isna(ic_narrow) or pd.isna(ic_wide):
                skipped_nan += 1
                continue
            gaps.append(ic_narrow - ic_wide)
            narrow_ics.append(ic_narrow)
            wide_ics.append(ic_wide)
        gaps_by_source[source] = pd.Series(gaps, dtype=float)
        detail[source] = {
            "usable_weeks": len(gaps),
            "skipped_small_group": skipped_small,
            "skipped_equal_width": skipped_equal,
            "skipped_no_ic": skipped_nan,
            "mean_ic_narrow": float(np.mean(narrow_ics)) if narrow_ics else float("nan"),
            "mean_ic_wide": float(np.mean(wide_ics)) if wide_ics else float("nan"),
        }
    df = _gap_rows(data, gaps_by_source, "H3b_width", "IC narrow minus wide")
    if len(df):
        df["source"] = df["portfolio"]
        for col in (
            "usable_weeks",
            "skipped_small_group",
            "skipped_equal_width",
            "skipped_no_ic",
            "mean_ic_narrow",
            "mean_ic_wide",
        ):
            df[col] = pd.Series([detail[s][col] for s in df["portfolio"]], index=df.index)
        rows = [df]
    return pd.concat(rows, ignore_index=True) if rows else df


def strategy_weekly_gaps(data: EvalData) -> tuple[dict[str, pd.Series], dict[str, int]]:
    """Per strategy: weekly mean return gap vs its Q0 across the sources clean that week, and the
    count of weeks dropped because no source was clean."""
    gaps: dict[str, pd.Series] = {}
    dropped: dict[str, int] = {}
    for strategy in data.strategies():
        q0_key = data.q0_key(strategy)
        steered = [k for k in data.steered_for(strategy) if k in data.weekly_returns]
        if q0_key not in data.weekly_returns or not steered:
            continue
        weekly: dict[str, float] = {}
        n_dropped = 0
        diffs = {
            key: (data.weekly_returns[key] - data.weekly_returns[q0_key]).astype(float)
            for key in steered
        }
        for week in data.weekly_returns.index:
            values = []
            for key in steered:
                clean = bool(data.clean_mask(key).get(week, False))
                gap = diffs[key].loc[week]
                if clean and pd.notna(gap):
                    values.append(float(gap))
            if values:
                weekly[week] = float(np.mean(values))
            else:
                n_dropped += 1
        gaps[strategy] = pd.Series(weekly, dtype=float)
        dropped[strategy] = n_dropped
    return gaps, dropped


def h4_strategy_ranking(data: EvalData) -> pd.DataFrame:
    """H4 (G4), exploratory: strategies ranked by their mean weekly gap vs their own Q0, with a
    bootstrap CI and the sign-flip p; no Holm."""
    gaps, dropped = strategy_weekly_gaps(data)
    rows = []
    for strategy, gap in gaps.items():
        summary = paired_summary(
            gap,
            data.cfg.eval.tie_band,
            data.cfg.eval.seed,
            n_perm=data.cfg.eval.permutation_samples,
            n_boot=data.cfg.eval.bootstrap_samples,
        )
        lo, hi = bootstrap_ci(gap, n=data.cfg.eval.bootstrap_samples, seed=data.cfg.eval.seed)
        rows.append(
            {
                "family": "H4_strategy_ranking",
                "strategy": strategy,
                "n_sources": len(data.steered_for(strategy)),
                "dropped_weeks": dropped[strategy],
                **summary,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("mean_gap", ascending=False, kind="stable").reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
    return df


def h4_splits(data: EvalData) -> dict[str, pd.DataFrame]:
    """H4 (G4), exploratory: the v1 IC splits per source, and the same market direction and
    volatility splits on each strategy's weekly gap vs its Q0."""
    out: dict[str, pd.DataFrame] = {}
    base = data.ic["baseline"]["baseline"]
    raw = data.ic["llm_raw"]
    gap_all = pd.DataFrame({k: raw[k] - base for k in data.sources if k in raw})
    strategy_gaps, _ = strategy_weekly_gaps(data)

    def split_rows(name: str, mask: pd.Series, label_true: str, label_false: str) -> pd.DataFrame:
        frames = []
        for flag, label in ((True, label_true), (False, label_false)):
            weeks = mask[mask == flag].index
            gaps = {k: gap_all.loc[gap_all.index.isin(weeks), k].dropna() for k in gap_all.columns}
            df = _gap_rows(data, gaps, f"H4_{name}", label)
            df["split"] = name
            df["group"] = label
            df["level"] = "source"
            frames.append(df)
            by_strategy = {s: g[g.index.isin(weeks)].dropna() for s, g in strategy_gaps.items()}
            sdf = _gap_rows(data, by_strategy, f"H4_{name}_strategy", label)
            sdf["split"] = name
            sdf["group"] = label
            sdf["level"] = "strategy"
            frames.append(sdf)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if len(data.spy_weekly):
        out["market_direction"] = split_rows(
            "market_direction", data.spy_weekly > 0, "market_up", "market_down"
        )
    if len(data.spy_vol20):
        median = data.spy_vol20.median()
        out["volatility"] = split_rows("volatility", data.spy_vol20 > median, "high_vol", "low_vol")
    out["earnings_week"] = h4_earnings(data)
    return out


def h4_earnings(data: EvalData) -> pd.DataFrame:
    """IC inside vs outside the earnings window per source and week; small groups give no IC."""
    sw = data.stock_week
    if len(sw) == 0:
        return pd.DataFrame()
    rows = []
    for source in data.sources:
        col = likely_column(source)
        if col not in sw.columns:
            continue
        clean = data.source_clean_mask(source)
        gaps = []
        n_usable = 0
        for w, g in sw.groupby("week_id"):
            if not bool(clean.get(w, False)):
                continue
            inside, outside = g[g["earnings_in_window"]], g[~g["earnings_in_window"]]
            if (
                len(inside) < data.cfg.eval.min_group_size
                or len(outside) < data.cfg.eval.min_group_size
            ):
                continue
            ic_in = rank_ic_series(inside[col], inside["ret"])
            ic_out = rank_ic_series(outside[col], outside["ret"])
            if pd.notna(ic_in) and pd.notna(ic_out):
                gaps.append(ic_in - ic_out)
                n_usable += 1
        summary = paired_summary(
            pd.Series(gaps, dtype=float),
            data.cfg.eval.tie_band,
            data.cfg.eval.seed,
            n_perm=data.cfg.eval.permutation_samples,
            n_boot=data.cfg.eval.bootstrap_samples,
        )
        rows.append(
            {
                "family": "H4_earnings_week",
                "comparison": "IC inside minus outside earnings window",
                "portfolio": source,
                "source": source,
                "usable_weeks": n_usable,
                **summary,
            }
        )
    df = pd.DataFrame(rows)
    if len(df):
        df = _holm_within(df, None, data.cfg.eval.significance)
        df["split"] = "earnings_week"
        df["group"] = "inside_minus_outside"
        df["level"] = "source"
    return df


def earnings_exposure(data: EvalData) -> pd.DataFrame:
    """Descriptive: share of weight and of stop sales that fell in earnings-window stocks."""
    sw = data.stock_week
    rows = []
    if len(sw) == 0:
        return pd.DataFrame()
    flags = sw.set_index(["week_id", "ticker"])["earnings_in_window"]
    from portstudy.io import read_csv
    from portstudy.optimize import weights_dir

    for key in data.portfolio_keys:
        weight_in, weight_total = 0.0, 0.0
        for w in data.weeks:
            path = weights_dir(w) / f"{key}.csv"
            if not path.exists():
                continue
            wdf = read_csv(path)
            f = [bool(flags.get((w, t), False)) for t in wdf["ticker"]]
            weight_in += float(wdf.loc[f, "weight"].sum())
            weight_total += float(wdf["weight"].sum())
        trades = data.trades.get(key, pd.DataFrame())
        stops = (
            trades[trades["reason"].isin(["stop_loss", "take_profit"])]
            if len(trades)
            else pd.DataFrame()
        )
        n_stop_in = sum(
            bool(flags.get((w, t), False))
            for w, t in zip(stops.get("week_id", []), stops.get("ticker", []))
        )
        rows.append(
            {
                "portfolio": key,
                "weight_share_in_earnings_window": weight_in / weight_total
                if weight_total
                else float("nan"),
                "stop_sales": int(len(stops)),
                "stop_sales_in_earnings_window": int(n_stop_in),
            }
        )
    return pd.DataFrame(rows)
