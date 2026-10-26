"""Wall-clock time per job for one synthetic week (100 stocks), to size the weekend window.

Usage: python scripts/benchmark_pipeline.py [--week 2026-W02] [--data-dir PATH]
Runs J1 to J9 once in a fresh data dir with the stub LLM and prints seconds per job. Real
model calls add their own time on top (measure them in Phase 17 B5).
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

from portstudy.pipeline.context import default_context, synthetic_clock
from portstudy.pipeline.jobs import run_named


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", default=None)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    data_dir = args.data_dir or tempfile.mkdtemp(prefix="portstudy_bench_")
    os.environ["PORTSTUDY_DATA_DIR"] = data_dir
    ctx = default_context(synthetic=True)
    ctx.cfg = ctx.cfg.model_copy(
        update={"llm": ctx.cfg.llm.model_copy(update={"cost_goal_usd": 1e9, "cost_warn_usd": 1e9})}
    )
    week = args.week or ctx.cfg.synthetic.first_week
    ctx.clock = synthetic_clock(week)
    t0 = time.perf_counter()
    ctx.providers()
    print(f"world load: {time.perf_counter() - t0:6.1f} s")
    total = 0.0
    for job in ctx.cfg.pipeline.jobs:
        t = time.perf_counter()
        status = run_named(ctx, job, week)
        elapsed = time.perf_counter() - t
        total += elapsed
        print(f"{job}: {elapsed:6.1f} s  ({status.state})")
    print(f"total: {total:6.1f} s for week {week} in {data_dir}")


if __name__ == "__main__":
    main()
