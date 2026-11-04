"""H5 shuffled-ratings test (DECISION(G6)), sharded for a Slurm array.

For every steered portfolio plus each strategy's Q0: for each clean week permute the LLM range
rows (low, likely, high, width) across stocks with an RNG seeded by SHA-256 of
(seed, portfolio, index, week), rebuild residualization, views, Black-Litterman, and the
portfolio's own strategy, replay the simulator from the ledger, and repeat `n_shuffles` times.
View strength is held fixed: every shuffle uses the real portfolio's stored weekly IC_L (IC_b for
Q0) and `use_llm_view` flags, so only the assignment of ratings to stocks is random. For Q0 the
baseline score z_b and the Kronos spread k_sd are permuted together. CVaR and resampled MVO draw
their common random numbers from (week, strategy), so the identity permutation reproduces the
real weights.

Shuffles are grouped into shards (`Shard(portfolio, start, stop)`); `run_shard` writes one JSON
file per shard under `data/shuffle/<portfolio>/<start>_<stop>.json` with a completion marker, so
shards run in any order, any number of times, and a half-written file is simply rerun.
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
from portstudy.optimize.black_litterman import Posterior, black_litterman, black_litterman_q0
from portstudy.optimize.strategies import run_strategy
from portstudy.optimize.strategies.common import StrategyInputs
from portstudy.paths import data_dir, outputs_dir
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


RANGE_COLUMNS = ["low", "likely", "high", "width"]
SHARD_COLUMNS = ["shard", "portfolio", "start", "stop"]


def week_rng(seed: int, portfolio: str, index: int, week_id: str) -> np.random.Generator:
    """RNG for one (shuffle index, week): SHA-256 of (seed, portfolio, index, week)."""
    digest = hashlib.sha256(f"{seed}:{portfolio}:{index}:{week_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _run_portfolio_strategy(
    cfg: StudyConfig, state: WeekState, post: Posterior, strategy: str
) -> pd.Series:
    inp = state.inputs
    assert inp is not None
    inputs = StrategyInputs(
        week_id=state.week_id,
        tickers=inp.tickers,
        mu=post.mu_bl,
        sigma=inp.risk.sigma_matrix,
        cov_mu=post.cov_mu,
        delta=cfg.risk.delta,
        previous=state.stored_weights,
        q0_weights=state.q0_weights,
    )
    return run_strategy(strategy, inputs, cfg).weights


def shuffled_weights(
    cfg: StudyConfig,
    state: WeekState,
    rng: np.random.Generator | None,
    portfolio: str,
) -> pd.Series:
    """Weights for one week after permuting the ratings (identity when `rng` is None)."""
    inp = state.inputs
    assert inp is not None
    pid = PortfolioId.parse(portfolio)
    strategy = str(pid.strategy)
    pi, sigma_vec, z_b, k_sd = inp.pi(), inp.sigma_vec(), inp.z_b(), inp.k_sd()
    sigma_matrix = inp.risk.sigma_matrix
    if pid.is_q0:
        if rng is not None:
            perm = rng.permutation(len(z_b))
            z_b = pd.Series(z_b.to_numpy()[perm], index=z_b.index, name=z_b.name)
            k_sd = pd.Series(k_sd.to_numpy()[perm], index=k_sd.index, name=k_sd.name)
        views = build_views(pi, sigma_vec, z_b, state.ic_b, cfg.views, k_sd=k_sd)
        post = black_litterman_q0(views, sigma_matrix)
    else:
        if not state.use_llm_view or state.q0_weights is None:
            return state.q0_weights if state.q0_weights is not None else pd.Series(dtype=float)
        valid = valid_scores(state.scores)
        shuffled = valid.copy()
        if rng is not None:
            perm = rng.permutation(len(valid))
            shuffled[RANGE_COLUMNS] = valid[RANGE_COLUMNS].to_numpy()[perm]
        z_l = residualize(shuffled, z_b)
        width = shuffled.set_index("ticker")["width"].astype(float)
        views = build_views(
            pi,
            sigma_vec,
            z_b,
            state.ic_b,
            cfg.views,
            k_sd=k_sd,
            z_l=z_l,
            ic_l=state.ic_l,
            width=width,
        )
        if not views.use_llm_view:
            return state.q0_weights
        post = black_litterman(views, sigma_matrix)
    return _run_portfolio_strategy(cfg, state, post, strategy)


def _plans_for_shuffle(
    cfg: StudyConfig,
    states: list[WeekState],
    seed: int,
    portfolio: str,
    index: int,
    identity: bool = False,
) -> list[WeekPlan]:
    is_q0 = PortfolioId.parse(portfolio).is_q0
    plans: list[WeekPlan] = []
    for st in states:
        if st.status == "no_rebalance" or st.inputs is None:
            plans.append(WeekPlan(st.week_id, None, "hold", "no_rebalance"))
        elif st.status == "fallback":
            plans.append(WeekPlan(st.week_id, st.stored_weights, "fallback_q0", "fallback"))
        else:
            rng = None if identity else week_rng(seed, portfolio, index, st.week_id)
            w = shuffled_weights(cfg, st, rng, portfolio)
            plans.append(WeekPlan(st.week_id, w, "q0" if is_q0 else "llm", "normal"))
    return plans


def identity_weights(cfg: StudyConfig, portfolio: str, weeks: list[str]) -> dict[str, pd.Series]:
    """Weights from the identity permutation per clean week (must equal the stored weights)."""
    out: dict[str, pd.Series] = {}
    for st in _load_states(cfg, portfolio, weeks):
        if st.status == "normal" and st.inputs is not None:
            out[st.week_id] = shuffled_weights(cfg, st, None, portfolio)
    return out


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


# Shards ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Shard:
    portfolio: str
    start: int
    stop: int  # exclusive

    @property
    def path(self) -> Path:
        return shard_dir(self.portfolio) / f"{self.start}_{self.stop}.json"


def shard_dir(portfolio: str) -> Path:
    path = data_dir() / "shuffle" / portfolio
    path.mkdir(parents=True, exist_ok=True)
    return path


def shuffle_portfolios(portfolios: list[str]) -> list[str]:
    """Each strategy's Q0 followed by every steered portfolio; controls never shuffle."""
    strategies = dict.fromkeys(
        PortfolioId.parse(key).strategy
        for key in portfolios
        if not PortfolioId.parse(key).is_control
    )
    q0_keys = [PortfolioId.q0(cast(StrategyName, strategy)).key for strategy in strategies]
    steered = [
        key
        for key in portfolios
        if not PortfolioId.parse(key).is_control and not PortfolioId.parse(key).is_q0
    ]
    return list(dict.fromkeys(q0_keys + steered))


def plan_shards(portfolios: list[str], n_shuffles: int, shard_size: int) -> list[Shard]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    shards = []
    for key in shuffle_portfolios(portfolios):
        for start in range(0, n_shuffles, shard_size):
            shards.append(Shard(key, start, min(start + shard_size, n_shuffles)))
    return shards


def write_shard_plan(shards: list[Shard], path: Path) -> pd.DataFrame:
    table = pd.DataFrame(
        [
            {"shard": i, "portfolio": s.portfolio, "start": s.start, "stop": s.stop}
            for i, s in enumerate(shards)
        ],
        columns=SHARD_COLUMNS,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n")
    return table


def read_shard_plan(path: Path) -> list[Shard]:
    table = pd.read_csv(path)
    return [
        Shard(portfolio, start, stop)
        for portfolio, start, stop in zip(
            table["portfolio"].astype(str).tolist(),
            table["start"].astype(int).tolist(),
            table["stop"].astype(int).tolist(),
        )
    ]


def shard_is_complete(shard: Shard) -> bool:
    if not shard.path.exists():
        return False
    try:
        payload = read_json(shard.path)
    except (OSError, ValueError):
        return False
    return bool(payload.get("complete")) and len(payload.get("results", [])) == (
        shard.stop - shard.start
    )


def run_shard(
    cfg: StudyConfig,
    shard: Shard,
    weeks: list[str],
    seed: int | None = None,
    force: bool = False,
) -> Path:
    """Run every shuffle index in the shard and write its file atomically with a completion
    marker. A complete file is skipped; a file without the marker is rerun."""
    if shard_is_complete(shard) and not force:
        return shard.path
    s = seed if seed is not None else cfg.eval.shuffle.seed
    states = _load_states(cfg, shard.portfolio, weeks)
    if not any(st.status == "normal" for st in states):
        payload = {
            "portfolio": shard.portfolio,
            "start": shard.start,
            "stop": shard.stop,
            "seed": s,
            "weeks": weeks,
            "results": [],
            "skipped": "no clean weeks",
            "complete": True,
        }
        write_json_atomic(payload, shard.path)
        return shard.path
    book = ledger_book()
    results = []
    for index in range(shard.start, shard.stop):
        plans = _plans_for_shuffle(cfg, states, s, shard.portfolio, index)
        sim = replay(f"shuffle_{shard.portfolio}_{index}", plans, book, cfg.simulator)
        results.append({"shuffle": index, **_metrics(daily_frame(sim.daily), cfg)})
    payload = {
        "portfolio": shard.portfolio,
        "start": shard.start,
        "stop": shard.stop,
        "seed": s,
        "weeks": weeks,
        "results": results,
        "complete": True,
    }
    write_json_atomic(payload, shard.path)
    return shard.path


def _shard_worker(args: tuple[StudyConfig, Shard, list[str], int | None]) -> str:
    cfg, shard, weeks, seed = args
    return str(run_shard(cfg, shard, weeks, seed))


def run_shards_local(
    cfg: StudyConfig,
    shards: list[Shard],
    weeks: list[str],
    seed: int | None = None,
    processes: int = 1,
) -> list[Path]:
    if processes > 1:
        with ProcessPoolExecutor(max_workers=processes) as pool:
            jobs = [(cfg, s, weeks, seed) for s in shards]
            return [Path(p) for p in pool.map(_shard_worker, jobs)]
    return [run_shard(cfg, s, weeks, seed) for s in shards]


class MissingShards(RuntimeError):
    pass


def collect_shards(shards: list[Shard]) -> tuple[pd.DataFrame, list[str]]:
    """Per-shuffle metrics from every shard file; raises `MissingShards` listing every
    incomplete range. Portfolios skipped for lack of clean weeks are returned separately."""
    frames = []
    missing = []
    skipped = []
    for shard in shards:
        if not shard_is_complete(shard):
            missing.append(f"{shard.portfolio} [{shard.start}, {shard.stop})")
            continue
        payload = read_json(shard.path)
        if payload.get("skipped"):
            skipped.append(shard.portfolio)
            continue
        part = pd.DataFrame(payload["results"])
        part.insert(0, "portfolio", shard.portfolio)
        frames.append(part)
    if missing:
        raise MissingShards("missing or incomplete shards: " + "; ".join(missing))
    table = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["portfolio", "shuffle", "total_return", "sharpe"])
    )
    return table.sort_values(["portfolio", "shuffle"]).reset_index(drop=True), sorted(set(skipped))


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


def shuffle_table(
    cfg: StudyConfig, dists: pd.DataFrame, weeks: list[str], skipped: list[str] | None = None
) -> pd.DataFrame:
    """p = (1 + count at or above real) / (n + 1) per portfolio; Holm within each strategy
    (its steered portfolios plus its Q0), never across controls."""
    rows = []
    for key, dist in dists.groupby("portfolio", sort=False):
        real = real_metrics(cfg, str(key), weeks)
        n = len(dist)
        row = {
            "portfolio": str(key),
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
    for key in skipped or []:
        rows.append({"portfolio": key, "n_shuffles": 0, "skipped": "no clean weeks"})
    table = pd.DataFrame(rows)
    if len(table):
        table["strategy"] = [str(PortfolioId.parse(k).strategy) for k in table["portfolio"]]
        for metric in ("total_return", "sharpe"):
            table[f"p_{metric}_holm"] = float("nan")
            for strategy in table["strategy"].unique():
                mask = (table["strategy"] == strategy) & table[f"p_{metric}"].notna()
                raw = table.loc[mask, f"p_{metric}"].to_numpy()
                table.loc[mask, f"p_{metric}_holm"] = holm(raw)
        table["significant_total_return"] = table["p_total_return_holm"] < cfg.eval.significance
    return table


def shuffle_test(
    cfg: StudyConfig,
    portfolios: list[str],
    weeks: list[str],
    n_shuffles: int | None = None,
    seed: int | None = None,
    processes: int | None = None,
    shards: list[Shard] | None = None,
    require_complete: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every steered portfolio plus each strategy's Q0 from shard files. With
    `require_complete` the call fails listing missing shards (the `collect` command); otherwise
    missing shards are run here first (the in-process report path)."""
    n = n_shuffles if n_shuffles is not None else cfg.eval.shuffle.n_shuffles
    plan = shards if shards is not None else plan_shards(portfolios, n, cfg.eval.shuffle.shard_size)
    if not require_complete:
        procs = processes if processes is not None else cfg.eval.shuffle.processes
        run_shards_local(cfg, [s for s in plan if not shard_is_complete(s)], weeks, seed, procs)
    dists, skipped = collect_shards(plan)
    return shuffle_table(cfg, dists, weeks, skipped), dists


def dump_shuffle_json(table: pd.DataFrame, path: Path) -> None:
    path.write_text(
        json.dumps(table.to_dict(orient="records"), indent=2, default=float), encoding="utf-8"
    )
