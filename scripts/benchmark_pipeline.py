"""Per-strategy solve times at a chosen universe size, the implied weekly J7 time, and the
projected shuffle CPU hours, so the universe size (DECISION(B4)) and the ARCTIC request can
be checked before the freeze.

Usage:
  python scripts/benchmark_pipeline.py --n-stocks 300 --sources 3 --weeks 52
  python scripts/benchmark_pipeline.py --jobs            # v1 mode: seconds per job, one week

Inputs are synthetic (a factor covariance and seeded expected returns), so the numbers measure
the optimizers, not data loading. The table is printed and written to
`data/benchmarks/<date>.csv`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from portstudy.config import StudyConfig, load_study_config
from portstudy.optimize.black_litterman import Posterior
from portstudy.optimize.problem_cache import clear_problem_cache
from portstudy.optimize.strategies import run_control, run_strategy
from portstudy.optimize.strategies.common import StrategyInputs
from portstudy.paths import data_dir

BENCHMARK_COLUMNS = [
    "date",
    "n_stocks",
    "sources",
    "strategy",
    "solves",
    "first_solve_s",
    "mean_solve_s",
    "j7_s_per_week",
    "projected_sources",
    "j7_projected_s_per_week",
    "weeks",
    "n_shuffles",
    "shuffle_cpu_hours",
    "peak_memory_mb",
]
ARCTIC_REQUEST_CPU_HOURS = 15_000


def synthetic_inputs(
    n_stocks: int, n_sources: int, tau: float, seed: int
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, list[pd.Series]]:
    """A three-factor weekly covariance, a Q0 mean, and one tilted mean per source."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:03d}" for i in range(n_stocks)]
    loadings = rng.normal(0.0, 0.3, size=(n_stocks, 3))
    factor_var = np.diag([4e-4, 1e-4, 1e-4])
    idio = rng.uniform(4e-4, 1.6e-3, size=n_stocks)
    sigma = loadings @ factor_var @ loadings.T + np.diag(idio)
    frame = pd.DataFrame(sigma, index=tickers, columns=tickers)
    cov_mu = pd.DataFrame(tau * sigma, index=tickers, columns=tickers)
    q0 = pd.Series(rng.normal(0.001, 0.004, size=n_stocks), index=tickers)
    means = [q0] + [
        q0 + pd.Series(rng.normal(0.0, 0.003, size=n_stocks), index=tickers)
        for _ in range(n_sources)
    ]
    return tickers, frame, cov_mu, means


def time_strategies(
    cfg: StudyConfig,
    strategies: list[str],
    controls: list[str],
    n_stocks: int,
    n_sources: int,
    seed: int,
    week_id: str = "2026-W10",
) -> dict[str, list[float]]:
    """Seconds per solve, per strategy: Q0 plus every source; controls once each."""
    tickers, sigma, cov_mu, means = synthetic_inputs(n_stocks, n_sources, cfg.views.tau, seed)
    clear_problem_cache()
    times: dict[str, list[float]] = {name: [] for name in [*strategies, *controls]}
    for name in strategies:
        for mu in means:
            inputs = StrategyInputs(week_id, tickers, mu, sigma, cov_mu, cfg.risk.delta)
            start = time.perf_counter()
            run_strategy(name, inputs, cfg)
            times[name].append(time.perf_counter() - start)
    for name in controls:
        posterior = Posterior(means[0], pd.Series(0.0, index=tickers), 0, cov_mu * 0.0)
        inputs = StrategyInputs(week_id, tickers, posterior.mu_bl, sigma, cov_mu, cfg.risk.delta)
        start = time.perf_counter()
        run_control(name, inputs, cfg)
        times[name].append(time.perf_counter() - start)
    return times


def projection(
    times: dict[str, list[float]],
    strategies: list[str],
    controls: list[str],
    n_sources: int,
    projected_sources: int,
    weeks: int,
    n_shuffles: int,
) -> dict[str, float]:
    """Weekly J7 seconds now and at the full source count, plus shuffle CPU hours:
    one solve per week per shuffle per steered portfolio, at the mean strategy solve time."""
    mean = {name: float(np.mean(values)) for name, values in times.items() if values}
    strategy_mean = {name: mean[name] for name in strategies}
    control_total = sum(mean[name] for name in controls)
    j7_now = (n_sources + 1) * sum(strategy_mean.values()) + control_total
    j7_projected = (projected_sources + 1) * sum(strategy_mean.values()) + control_total
    steered = projected_sources * len(strategies)
    mean_solve = float(np.mean(list(strategy_mean.values()))) if strategy_mean else 0.0
    shuffle_hours = weeks * n_shuffles * steered * mean_solve / 3600.0
    return {
        "j7_s_per_week": j7_now,
        "j7_projected_s_per_week": j7_projected,
        "shuffle_cpu_hours": shuffle_hours,
        "steered_portfolios": float(steered),
        "mean_solve_s": mean_solve,
    }


def peak_memory_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def benchmark_table(
    cfg: StudyConfig,
    n_stocks: int,
    n_sources: int,
    strategies: list[str],
    controls: list[str],
    weeks: int,
    projected_sources: int,
    seed: int,
    today: dt.date,
) -> pd.DataFrame:
    times = time_strategies(cfg, strategies, controls, n_stocks, n_sources, seed)
    proj = projection(
        times,
        strategies,
        controls,
        n_sources,
        projected_sources,
        weeks,
        cfg.eval.shuffle.n_shuffles,
    )
    memory = peak_memory_mb()
    rows = []
    for name in [*strategies, *controls]:
        values = times[name]
        rows.append(
            {
                "date": today.isoformat(),
                "n_stocks": n_stocks,
                "sources": n_sources,
                "strategy": name,
                "solves": len(values),
                "first_solve_s": values[0],
                "mean_solve_s": float(np.mean(values)),
                "j7_s_per_week": proj["j7_s_per_week"],
                "projected_sources": projected_sources,
                "j7_projected_s_per_week": proj["j7_projected_s_per_week"],
                "weeks": weeks,
                "n_shuffles": cfg.eval.shuffle.n_shuffles,
                "shuffle_cpu_hours": proj["shuffle_cpu_hours"],
                "peak_memory_mb": memory,
            }
        )
    return pd.DataFrame(rows, columns=BENCHMARK_COLUMNS)


def write_benchmark(table: pd.DataFrame, folder: Path | None = None) -> Path:
    out_dir = folder or (data_dir() / "benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{table['date'].iloc[0]}.csv"
    table.to_csv(path, index=False, lineterminator="\n")
    return path


def run_jobs_once(week: str | None, data_folder: str | None) -> None:
    """v1 mode: wall-clock seconds per job for one synthetic week with the stub LLM."""
    from portstudy.pipeline.context import default_context, synthetic_clock
    from portstudy.pipeline.jobs import run_named

    folder = data_folder or tempfile.mkdtemp(prefix="portstudy_bench_")
    os.environ["PORTSTUDY_DATA_DIR"] = folder
    ctx = default_context(synthetic=True)
    ctx.cfg = ctx.cfg.model_copy(
        update={"llm": ctx.cfg.llm.model_copy(update={"cost_goal_usd": 1e9, "cost_warn_usd": 1e9})}
    )
    week_id = week or ctx.cfg.synthetic.first_week
    ctx.clock = synthetic_clock(week_id)
    t0 = time.perf_counter()
    ctx.providers()
    print(f"world load: {time.perf_counter() - t0:6.1f} s")
    total = 0.0
    for job in ctx.cfg.pipeline.jobs:
        t = time.perf_counter()
        status = run_named(ctx, job, week_id)
        elapsed = time.perf_counter() - t
        total += elapsed
        print(f"{job}: {elapsed:6.1f} s  ({status.state})")
    print(f"total: {total:6.1f} s for week {week_id} in {folder}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None, help="study YAML (default: PORTSTUDY_CONFIG)")
    parser.add_argument("--n-stocks", type=int, default=300)
    parser.add_argument("--sources", type=int, default=3)
    parser.add_argument("--strategies", default=None, help="comma list; default: all enabled")
    parser.add_argument("--controls", default=None, help="comma list; default: configured")
    parser.add_argument("--weeks", type=int, default=52)
    parser.add_argument(
        "--project-sources",
        type=int,
        default=51,
        help="source count for the projection (7 models x 7 prompt types + Q0)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None, help="default: data/benchmarks")
    parser.add_argument("--jobs", action="store_true", help="v1 mode: time J1 to J9 once")
    parser.add_argument("--week", default=None, help="week for --jobs")
    parser.add_argument("--data-dir", default=None, help="data dir for --jobs")
    args = parser.parse_args(argv)
    if args.jobs:
        run_jobs_once(args.week, args.data_dir)
        return 0
    cfg = load_study_config(Path(args.config) if args.config else None)
    strategies = args.strategies.split(",") if args.strategies else list(cfg.strategies.enabled)
    controls = (
        args.controls.split(",") if args.controls is not None else list(cfg.strategies.controls)
    )
    table = benchmark_table(
        cfg,
        args.n_stocks,
        args.sources,
        strategies,
        controls,
        args.weeks,
        args.project_sources,
        args.seed,
        dt.date.today(),
    )
    path = write_benchmark(table, Path(args.out_dir) if args.out_dir else None)
    shown = table.loc[:, ["strategy", "solves", "first_solve_s", "mean_solve_s"]]
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    first = table.iloc[0]
    print(f"J7 seconds per week at {args.sources} sources: {first['j7_s_per_week']:.1f}")
    print(
        f"J7 seconds per week at {args.project_sources} sources: "
        f"{first['j7_projected_s_per_week']:.1f}"
    )
    print(
        f"shuffle CPU hours ({args.weeks} weeks x {first['n_shuffles']} shuffles x "
        f"{args.project_sources * len(strategies)} steered portfolios): "
        f"{first['shuffle_cpu_hours']:.1f}"
    )
    if first["shuffle_cpu_hours"] > ARCTIC_REQUEST_CPU_HOURS:
        print(f"WARNING: projection exceeds the {ARCTIC_REQUEST_CPU_HOURS} CPU hour request")
    print(f"peak memory: {first['peak_memory_mb']:.0f} MB; table written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
