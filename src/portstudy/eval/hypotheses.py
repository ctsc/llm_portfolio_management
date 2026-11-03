"""H1 to H4. Every function returns a tidy DataFrame; p-values are Holm-adjusted per family.

The primary p-value for each paired weekly test is the exact sign-flip permutation p (`p_perm`),
which keeps its nominal size at 20 to 30 weeks; the Newey-West p (`p_nw`) is reported beside it.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from portstudy.eval.data import EvalData, max_drawdown, sharpe
from portstudy.eval.tests import holm, paired_summary, rank_ic_series
from portstudy.types import PortfolioId, RatingSource, StrategyName


def _gap_rows(data: EvalData, gaps: dict[str, pd.Series], family: str, label: str) -> pd.DataFrame:
    rows = []
    for key, gap in gaps.items():
        summary = paired_summary(
            gap,
            data.cfg.eval.tie_band,
            data.cfg.eval.seed,
            n_perm=data.cfg.eval.permutation_samples,
            n_boot=data.cfg.eval.bootstrap_samples,
        )
        rows.append({"family": family, "comparison": label, "portfolio": key, **summary})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm(df["p_perm"].to_numpy())
        df["significant"] = df["p_holm"] < data.cfg.eval.significance
    return df


def h1_llm_raw_vs_baseline(data: EvalData) -> pd.DataFrame:
    """H1: weekly gap IC_llm_raw - IC_baseline per model and configuration (one Holm family)."""
    base = data.ic["baseline"]["baseline"]
    raw = data.ic["llm_raw"]
    gaps = {k: (raw[k] - base).dropna() for k in data.sources if k in raw}
    return _gap_rows(data, gaps, "H1_ic_gap", "llm_raw minus baseline")


def h2_combined_vs_baseline(data: EvalData) -> pd.DataFrame:
    """H2 (IC): weekly gap IC_combined - IC_baseline (E3: combined = ranked BL tilt)."""
    base = data.ic["baseline"]["baseline"]
    comb = data.ic["combined"]
    gaps = {k: (comb[k] - base).dropna() for k in data.sources if k in comb}
    return _gap_rows(data, gaps, "H2_ic_gap", "combined minus baseline")


def portfolio_metrics(data: EvalData, clean_only: bool) -> pd.DataFrame:
    rows = []
    for key in data.portfolio_keys:
        weekly = data.weekly_returns[key] if key in data.weekly_returns else pd.Series(dtype=float)
        if clean_only and not PortfolioId.parse(key).is_q0:
            weekly = weekly[data.clean_mask(key).reindex(weekly.index).fillna(False)]
        daily = data.daily_values.get(key, pd.DataFrame())
        trades = data.trades.get(key, pd.DataFrame())
        values = daily["value"] if len(daily) else pd.Series(dtype=float)
        avg_value = float(values.mean()) if len(values) else float("nan")
        rows.append(
            {
                "portfolio": key,
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
            }
        )
    return pd.DataFrame(rows)


def h2_portfolio_vs_q0(data: EvalData, clean_only: bool = True) -> pd.DataFrame:
    """H2 (portfolios): paired weekly return difference against Q0 (one Holm family)."""
    gaps = {}
    for key in data.portfolios:
        if key not in data.weekly_returns:
            continue
        strategy = cast(StrategyName, PortfolioId.parse(key).strategy)
        q0_key = PortfolioId.q0(strategy).key
        if q0_key not in data.weekly_returns:
            continue
        gap = data.weekly_returns[key] - data.weekly_returns[q0_key]
        if clean_only:
            gap = gap[data.clean_mask(key).reindex(gap.index).fillna(False)]
        gaps[key] = gap.dropna()
    df = _gap_rows(data, gaps, "H2_portfolio_vs_q0", "weekly return minus Q0")
    metrics = portfolio_metrics(data, clean_only).set_index("portfolio")
    if len(df):
        for col in ("sharpe", "max_drawdown", "turnover", "total_cost_usd", "total_return"):
            df[col] = df["portfolio"].map(metrics[col])
    return df


def h3_skills_vs_controls(data: EvalData, signal: str = "llm_raw") -> pd.DataFrame:
    """H3 (G3): per model, skills vs bare and skills vs placebo; Holm across the full family.

    Any configuration of kind `skills` other than the controls is compared against both.
    """
    table = data.ic[signal]
    by_model: dict[str, dict[str, str]] = {}
    for key in data.sources:
        source = RatingSource.parse(key)
        by_model.setdefault(source.model, {})[source.config_id] = key
    gaps: dict[str, pd.Series] = {}
    for model, confs in by_model.items():
        for conf, key in confs.items():
            if conf in ("bare", "placebo") or key not in table:
                continue
            for control in ("bare", "placebo"):
                ckey = confs.get(control)
                if ckey is None or ckey not in table:
                    continue
                gaps[f"{model}: {conf} vs {control}"] = (table[key] - table[ckey]).dropna()
    df = _gap_rows(data, gaps, "H3_skills_vs_controls", f"{signal} IC gap")
    if len(df):
        df["model"] = df["portfolio"].str.split(":").str[0]
    return df


def h4_splits(data: EvalData) -> dict[str, pd.DataFrame]:
    """H4 (G4), exploratory: H1 gaps within market direction, volatility, and earnings splits."""
    out: dict[str, pd.DataFrame] = {}
    base = data.ic["baseline"]["baseline"]
    raw = data.ic["llm_raw"]
    gap_all = pd.DataFrame({k: raw[k] - base for k in data.sources if k in raw})

    def split_rows(name: str, mask: pd.Series, label_true: str, label_false: str) -> pd.DataFrame:
        frames = []
        for flag, label in ((True, label_true), (False, label_false)):
            weeks = mask[mask == flag].index
            gaps = {k: gap_all.loc[gap_all.index.isin(weeks), k].dropna() for k in gap_all.columns}
            df = _gap_rows(data, gaps, f"H4_{name}", label)
            df["split"] = name
            df["group"] = label
            frames.append(df)
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
    """IC inside vs outside the earnings window per week; small groups give no IC."""
    sw = data.stock_week
    if len(sw) == 0:
        return pd.DataFrame()
    rows = []
    usable: dict[str, int] = {}
    for key in data.portfolios:
        if key not in sw.columns:
            continue
        gaps = []
        n_usable = 0
        for w, g in sw.groupby("week_id"):
            if w not in data.clean_mask(key).index or not bool(data.clean_mask(key).get(w, False)):
                continue
            inside, outside = g[g["earnings_in_window"]], g[~g["earnings_in_window"]]
            if (
                len(inside) < data.cfg.eval.min_group_size
                or len(outside) < data.cfg.eval.min_group_size
            ):
                continue
            ic_in = rank_ic_series(inside[key], inside["ret"])
            ic_out = rank_ic_series(outside[key], outside["ret"])
            if pd.notna(ic_in) and pd.notna(ic_out):
                gaps.append(ic_in - ic_out)
                n_usable += 1
        usable[key] = n_usable
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
                "portfolio": key,
                "usable_weeks": n_usable,
                **summary,
            }
        )
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm(df["p_perm"].to_numpy())
        df["significant"] = df["p_holm"] < data.cfg.eval.significance
        df["split"] = "earnings_week"
        df["group"] = "inside_minus_outside"
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
