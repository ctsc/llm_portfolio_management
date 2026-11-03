"""H5 shuffled-ratings test (DECISION(G6)).

For every portfolio (each model and configuration) plus Q0: for each clean week permute the
LLM ranges across stocks with a seeded RNG, rebuild
residualization, views, Black-Litterman, and mean-variance, replay the simulator from the
ledger, and repeat `n_shuffles` times. View strength is held fixed: every shuffle uses the
real portfolio's stored weekly IC_L (IC_b for Q0) and `use_llm_view` flags, so only the
assignment of scores to stocks is random. For Q0 the baseline score z_b is shuffled.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from portstudy.config import StudyConfig
from portstudy.eval.data import sharpe, weekly_portfolio_returns
from portstudy.eval.tests import holm
from portstudy.io import read_json, write_json_atomic
from portstudy.optimize import WeekInputs, read_weights, read_weights_meta, weights_dir
from portstudy.optimize.black_litterman import black_litterman, black_litterman_q0
from portstudy.optimize.mean_variance import solve_mean_variance
from portstudy.paths import outputs_dir
from portstudy.pipeline import fallback as fb
from portstudy.sim import PriceBook, WeekPlan, ledger_book, replay
from portstudy.sim.records import daily_frame, read_daily
from portstudy.types import PortfolioId, StrategyName
from portstudy.views.build import build_views
from portstudy.views.residualize import residualize, valid_scores


@dataclass
class WeekState:
    """Everything a shuffle needs for one week, loaded once."""

    week_id: str
    status: str
    inputs: WeekInputs | None
    ic_b: float
    ic_l: float
    use_llm_view: bool
    stored_weights: pd.Series | None
    q0_weights: pd.Series | None
    scores: pd.DataFrame = field(default_factory=pd.DataFrame)


def _load_states(cfg: StudyConfig, portfolio: str, weeks: list[str]) -> list[WeekState]:
    pid = PortfolioId.parse(portfolio)
    states: list[WeekState] = []
    for w in weeks:
        status = fb.week_status_for(w, portfolio)
        path = weights_dir(w) / f"{portfolio}.csv"
        if status == "no_rebalance" or not path.exists():
            states.append(WeekState(w, "no_rebalance", None, 0.0, 0.0, False, None, None))
            continue
        stored = read_weights(w, portfolio).set_index("ticker")["weight"].astype(float)
        q0_key = PortfolioId.q0(cast(StrategyName, pid.strategy)).key
        q0 = read_weights(w, q0_key).set_index("ticker")["weight"].astype(float)
        meta = read_weights_meta(w, portfolio)
        inputs = WeekInputs.load(w) if status == "normal" else None
        scores = (
            inputs.scores_for(pid.model, pid.config_id)
            if (inputs is not None and not pid.is_q0)
            else pd.DataFrame()
        )
        states.append(
            WeekState(
                w,
                status,
                inputs,
                float(meta.get("ic_b", 0.0)),
                float(meta.get("ic_l", 0.0)),
                bool(meta.get("use_llm_view", False)),
                stored,
                q0,
                scores,
            )
        )
    return states


def _shuffled_weights(
    cfg: StudyConfig, state: WeekState, rng: np.random.Generator, is_q0: bool
) -> pd.Series:
    inp = state.inputs
    assert inp is not None
    pi, sigma_vec, z_b = inp.pi(), inp.sigma_vec(), inp.z_b()
    sigma_matrix = inp.risk.sigma_matrix
    if is_q0:
        shuffled_zb = pd.Series(rng.permutation(z_b.to_numpy()), index=z_b.index)
        views = build_views(pi, sigma_vec, shuffled_zb, state.ic_b, cfg.views, k_sd=inp.k_sd())
        post = black_litterman_q0(views, sigma_matrix)
    else:
        if not state.use_llm_view or state.q0_weights is None:
            return state.q0_weights if state.q0_weights is not None else pd.Series(dtype=float)
        valid = valid_scores(state.scores)
        perm = rng.permutation(len(valid))
        shuffled = valid.copy()
        shuffled[["low", "likely", "high", "width"]] = valid[
            ["low", "likely", "high", "width"]
        ].to_numpy()[perm]
        z_l = residualize(shuffled, z_b)
        width = shuffled.set_index("ticker")["width"].astype(float)
        views = build_views(
            pi,
            sigma_vec,
            z_b,
            state.ic_b,
            cfg.views,
            k_sd=inp.k_sd(),
            z_l=z_l,
            ic_l=state.ic_l,
            width=width,
        )
        if not views.use_llm_view:
            return state.q0_weights
        post = black_litterman(views, sigma_matrix)
    mv = solve_mean_variance(
        post.mu_bl, sigma_matrix, cfg.risk.delta, cfg.optimizer, previous=state.stored_weights
    )
    return mv.weights


def _plans_for_shuffle(
    cfg: StudyConfig, states: list[WeekState], seed: int, portfolio: str, index: int
) -> list[WeekPlan]:
    digest = hashlib.sha256(f"{seed}:{portfolio}:{index}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    is_q0 = PortfolioId.parse(portfolio).is_q0
    plans: list[WeekPlan] = []
    for st in states:
        if st.status == "no_rebalance" or st.inputs is None:
            plans.append(WeekPlan(st.week_id, None, "hold", "no_rebalance"))
        elif st.status == "fallback":
            plans.append(WeekPlan(st.week_id, st.stored_weights, "fallback_q0", "fallback"))
        else:
            w = _shuffled_weights(cfg, st, rng, is_q0)
            plans.append(WeekPlan(st.week_id, w, "q0" if is_q0 else "llm", "normal"))
    return plans


def _metrics(daily: pd.DataFrame, cfg: StudyConfig) -> dict[str, float]:
    weekly = weekly_portfolio_returns(daily, cfg.simulator.initial_value)
    total = (
        float(daily["value"].iloc[-1] / cfg.simulator.initial_value - 1.0)
        if len(daily)
        else float("nan")
    )
    return {"total_return": total, "sharpe": sharpe(weekly)}


def shuffle_cache_dir(portfolio: str) -> Path:
    path = outputs_dir() / "shuffle" / portfolio
    path.mkdir(parents=True, exist_ok=True)
    return path


def one_shuffle(
    cfg: StudyConfig,
    portfolio: str,
    weeks: list[str],
    seed: int,
    index: int,
    states: list[WeekState] | None = None,
    book: PriceBook | None = None,
) -> dict[str, float]:
    """Run shuffle `index` for a portfolio, cached by seed so reruns resume."""
    cache = shuffle_cache_dir(portfolio) / f"{seed}_{index}.json"
    if cache.exists():
        return dict(read_json(cache))
    sts = states if states is not None else _load_states(cfg, portfolio, weeks)
    plans = _plans_for_shuffle(cfg, sts, seed, portfolio, index)
    sim = replay(f"shuffle_{portfolio}_{index}", plans, book or ledger_book(), cfg.simulator)
    result = _metrics(daily_frame(sim.daily), cfg)
    write_json_atomic(result, cache)
    return result


def _worker(args: tuple[StudyConfig, str, list[str], int, int]) -> dict[str, float]:
    cfg, portfolio, weeks, seed, index = args
    return one_shuffle(cfg, portfolio, weeks, seed, index)


def shuffle_portfolio(
    cfg: StudyConfig,
    portfolio: str,
    weeks: list[str],
    n_shuffles: int | None = None,
    seed: int | None = None,
    processes: int | None = None,
) -> pd.DataFrame:
    n = n_shuffles if n_shuffles is not None else cfg.eval.shuffle.n_shuffles
    s = seed if seed is not None else cfg.eval.shuffle.seed
    procs = processes if processes is not None else cfg.eval.shuffle.processes
    if procs > 1:
        with ProcessPoolExecutor(max_workers=procs) as pool:
            results = list(pool.map(_worker, [(cfg, portfolio, weeks, s, i) for i in range(n)]))
    else:
        states = _load_states(cfg, portfolio, weeks)
        book = ledger_book()
        results = [one_shuffle(cfg, portfolio, weeks, s, i, states, book) for i in range(n)]
    df = pd.DataFrame(results)
    df.insert(0, "shuffle", range(n))
    df.insert(0, "portfolio", portfolio)
    return df


def real_metrics(cfg: StudyConfig, portfolio: str, weeks: list[str]) -> dict[str, float]:
    daily = read_daily(portfolio)
    daily = daily[daily["week_id"].isin(weeks)] if len(daily) else daily
    return _metrics(daily, cfg)


def shuffle_test(
    cfg: StudyConfig,
    portfolios: list[str],
    weeks: list[str],
    n_shuffles: int | None = None,
    seed: int | None = None,
    processes: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every portfolio plus Q0; p = (1 + count at or above real) / (n + 1); Holm across all."""
    rows = []
    dists = []
    strategies = dict.fromkeys(
        PortfolioId.parse(key).strategy
        for key in portfolios
        if not PortfolioId.parse(key).is_control
    )
    q0_keys = [PortfolioId.q0(cast(StrategyName, strategy)).key for strategy in strategies]
    keys = q0_keys + [key for key in portfolios if not PortfolioId.parse(key).is_control]
    for key in keys:
        dist = shuffle_portfolio(cfg, key, weeks, n_shuffles, seed, processes)
        dists.append(dist)
        real = real_metrics(cfg, key, weeks)
        n = len(dist)
        row = {
            "portfolio": key,
            "n_shuffles": n,
            "real_total_return": real["total_return"],
            "real_sharpe": real["sharpe"],
            "shuffle_mean_total_return": float(dist["total_return"].mean()),
            "shuffle_mean_sharpe": float(dist["sharpe"].mean()),
        }
        for metric in ("total_return", "sharpe"):
            at_or_above = int((dist[metric] >= real[metric]).sum()) if pd.notna(real[metric]) else n
            row[f"p_{metric}"] = (1 + at_or_above) / (n + 1)
            row[f"percentile_{metric}"] = (
                float((dist[metric] < real[metric]).mean())
                if pd.notna(real[metric])
                else float("nan")
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    if len(table):
        table["p_total_return_holm"] = holm(table["p_total_return"].to_numpy())
        table["p_sharpe_holm"] = holm(table["p_sharpe"].to_numpy())
        table["significant_total_return"] = table["p_total_return_holm"] < cfg.eval.significance
    return table, pd.concat(dists, ignore_index=True) if dists else pd.DataFrame()


def dump_shuffle_json(table: pd.DataFrame, path: Path) -> None:
    path.write_text(
        json.dumps(table.to_dict(orient="records"), indent=2, default=float), encoding="utf-8"
    )
