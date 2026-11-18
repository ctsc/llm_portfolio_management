"""DECISION(S8): calibrate Mean-CVaR `lam` and Resampled MVO `draw_scale` on the synthetic world so
LLM views can move those strategies, by a fixed rule (no judgment needed).

For each synthetic week (J1 and J2 run on the synthetic world at `universe.size` stocks) and each
strategy, the expected tracking error against the strategy's own Q0,
sqrt((w - w_Q0)' Sigma (w - w_Q0)), is measured under three rating inputs: every rating 0
(confidence only), random noise ratings, and skilled ratings (built from the next week's
realized return plus noise). Rule: from the grid [1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01], pick
the largest Mean-CVaR `lam` (and separately the largest Resampled `draw_scale`) whose median
tracking error under noise ratings is at least 25% of MVO's; if none qualifies, keep the smallest
grid value and say so. Writes data/outputs/calibration/view_sensitivity.csv and summary.md.

Usage: python scripts/calibrate_view_sensitivity.py [--weeks 12] [--size 100] [--config PATH]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portstudy import calendar as cal
from portstudy.config import StudyConfig, load_model_registry, load_study_config
from portstudy.data.providers.factory import build_providers
from portstudy.io import write_csv_atomic, write_text_atomic
from portstudy.llm.prompts import load_manifest
from portstudy.optimize import WeekInputs, _q0_views, expected_te
from portstudy.optimize.black_litterman import Posterior, black_litterman, black_litterman_q0
from portstudy.optimize.strategies import run_strategy
from portstudy.optimize.strategies.common import StrategyInputs
from portstudy.paths import outputs_dir
from portstudy.pipeline.context import RunContext, synthetic_clock
from portstudy.pipeline.jobs import run_named
from portstudy.views.build import build_views
from portstudy.views.residualize import residualize

GRID = [1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01]
SHARE_OF_MVO = 0.25
INPUTS = ("zero", "noise", "skilled")
STRATEGIES = ("mvo", "cvar", "robust", "resampled", "topk")


def ratings(kind: str, tickers: list[str], fwd: pd.Series, seed: int) -> pd.DataFrame:
    """Integer range ratings in [-5, 5] with width 2 for every stock."""
    rng = np.random.default_rng(seed)
    if kind == "zero":
        likely = np.zeros(len(tickers), dtype=int)
    elif kind == "noise":
        likely = rng.integers(-5, 6, size=len(tickers))
    else:
        r = fwd.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
        z = (r - r.mean()) / (r.std(ddof=1) or 1.0)
        likely = np.clip(np.round(2.0 * z + rng.normal(0.0, 1.0, len(tickers))), -5, 5).astype(int)
    low = np.clip(likely - 1, -5, 5)
    high = np.clip(likely + 1, -5, 5)
    return pd.DataFrame(
        {"ticker": tickers, "low": low, "likely": likely, "high": high, "width": high - low}
    )


def _weights(cfg: StudyConfig, strategy: str, inp: WeekInputs, post: Posterior) -> pd.Series:
    si = StrategyInputs(
        week_id=inp.week_id,
        tickers=inp.tickers,
        mu=post.mu_bl,
        sigma=inp.risk.sigma_matrix,
        cov_mu=post.cov_mu,
        delta=cfg.risk.delta,
        previous=None,
        q0_weights=None,
    )
    return run_strategy(strategy, si, cfg).weights


def _with(cfg: StudyConfig, strategy: str, value: float) -> StudyConfig:
    s = cfg.strategies
    if strategy == "cvar":
        s = s.model_copy(update={"cvar": s.cvar.model_copy(update={"lam": value})})
    elif strategy == "resampled":
        s = s.model_copy(update={"resampled": s.resampled.model_copy(update={"draw_scale": value})})
    return cfg.model_copy(update={"strategies": s})


def week_rows(cfg: StudyConfig, week: str, fwd: pd.Series, grids: bool) -> list[dict[str, Any]]:
    inp = WeekInputs.load(week)
    log = logging.getLogger("portstudy.calibration")
    ic = cfg.views.fixed_ic
    q0_views = _q0_views(inp, cfg, ic, log)
    q0_post = black_litterman_q0(q0_views, inp.risk.sigma_matrix)
    posts: dict[str, Posterior] = {}
    for kind in INPUTS:
        seed = int(hashlib.sha256(f"{week}:{kind}".encode()).hexdigest()[:8], 16)
        scores = ratings(kind, inp.tickers, fwd, seed=seed)
        z_l = residualize(scores, inp.z_b())
        views = build_views(
            inp.pi(),
            inp.sigma_vec(),
            inp.z_b(),
            ic,
            cfg.views,
            k_sd=inp.k_sd(),
            z_l=z_l,
            ic_l=ic,
            width=scores.set_index("ticker")["width"].astype(float),
            mu_q0=q0_post.mu_bl,
        )
        posts[kind] = black_litterman(views, inp.risk.sigma_matrix)
    rows = []
    sigma = inp.risk.sigma_matrix
    for strategy in STRATEGIES:
        values: list[float | None] = [None]
        if grids and strategy in ("cvar", "resampled"):
            values = [None, *GRID]  # current setting under every input, grid under noise
        for value in values:
            c = cfg if value is None else _with(cfg, strategy, value)
            w_q0 = _weights(c, strategy, inp, q0_post)
            kinds = INPUTS if value is None else ("noise",)
            for kind in kinds:
                w = _weights(c, strategy, inp, posts[kind])
                rows.append(
                    {
                        "week_id": week,
                        "strategy": strategy,
                        "setting": "current" if value is None else value,
                        "input": kind,
                        "te": expected_te(w, w_q0, sigma),
                    }
                )
    return rows


def choose(table: pd.DataFrame) -> tuple[dict[str, float], dict[str, str]]:
    med = table.groupby(["strategy", "setting", "input"])["te"].median()
    mvo = float(med[("mvo", "current", "noise")])
    chosen: dict[str, float] = {}
    notes: dict[str, str] = {}
    for strategy in ("cvar", "resampled"):
        ok = [v for v in GRID if float(med[(strategy, v, "noise")]) >= SHARE_OF_MVO * mvo]
        if ok:
            chosen[strategy] = max(ok)
            notes[strategy] = f"largest grid value meeting {SHARE_OF_MVO:.0%} of MVO"
        else:
            chosen[strategy] = min(GRID)
            notes[strategy] = "no grid value met the rule; kept the smallest"
    return chosen, notes


def calibrate(
    cfg: StudyConfig, n_weeks: int, out_dir: Path
) -> tuple[pd.DataFrame, dict[str, float]]:
    registry = load_model_registry()
    ctx = RunContext(cfg=cfg, registry=registry, manifest=load_manifest(), synthetic=True)
    providers = build_providers(cfg, synthetic=True)
    weeks = cal.week_range(cfg.synthetic.first_week, n_weeks)
    rows: list[dict[str, Any]] = []
    for week in weeks:
        ctx.clock = synthetic_clock(week)
        ctx.data_providers = providers
        for job in ("J1", "J2"):
            run_named(ctx, job, week)
        start, end = cal.monday_open_day(week), cal.window_end_date(week)
        snap_tickers = list(WeekInputs.load(week).tickers)
        px = providers.prices.get_prices(snap_tickers, start, end)
        px = px.drop_duplicates(["date", "ticker"])
        opens = px.pivot(index="date", columns="ticker", values="open")
        fwd = (opens.iloc[-1] / opens.iloc[0] - 1.0) if len(opens) >= 2 else pd.Series(dtype=float)
        rows.extend(week_rows(cfg, week, fwd, grids=True))
        print(f"{week}: done", flush=True)
    table = pd.DataFrame(rows)
    chosen, notes = choose(table)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(table, out_dir / "view_sensitivity.csv")
    med = (
        table.groupby(["strategy", "setting", "input"])["te"]
        .median()
        .reset_index()
        .rename(columns={"te": "median_te"})
    )
    lines = [
        f"# View sensitivity calibration ({dt.date.today().isoformat()})",
        "",
        f"Synthetic world, {len(weeks)} weeks, {cfg.universe.size} stocks; rule: largest grid "
        f"value whose median expected tracking error under noise ratings is at least "
        f"{SHARE_OF_MVO:.0%} of MVO's.",
        "",
        "| strategy | setting | input | median expected TE vs Q0 |",
        "|---|---|---|---|",
        *[
            f"| {r.strategy} | {r.setting} | {r.input} | {r.median_te:.6f} |"
            for r in med.itertuples()
        ],
        "",
        f"- Mean-CVaR `lam`: {chosen['cvar']} ({notes['cvar']})",
        f"- Resampled `draw_scale`: {chosen['resampled']} ({notes['resampled']})",
    ]
    write_text_atomic("\n".join(lines) + "\n", out_dir / "summary.md")
    return table, chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weeks", type=int, default=12)
    parser.add_argument("--size", type=int, default=None, help="default: universe.size")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_study_config(Path(args.config) if args.config else None)
    if args.size is not None:
        cfg = cfg.model_copy(
            update={"universe": cfg.universe.model_copy(update={"size": args.size})}
        )
    _, chosen = calibrate(cfg, args.weeks, outputs_dir() / "calibration")
    print(f"chosen: cvar lam {chosen['cvar']}, resampled draw_scale {chosen['resampled']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
