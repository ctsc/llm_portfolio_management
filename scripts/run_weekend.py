"""The one script the researcher runs by hand each weekend (DECISION(N2)). No scheduler.

Step 1: J8 and J9 for any earlier week whose replay or backup is not done.
Step 2: J1 to J6 for the current week in order, waiting on API batches (J5) with a visible
        countdown until done or the lock deadline minus the margin.
Step 3: J7, then J9.

Start Friday night if possible; a Saturday start leaves a smaller buffer for a batch retry.
On a failure the script stops, prints what failed, the fallback that will apply, and the exact
command to rerun that one job. Rerunning the script resumes where it stopped.

Usage: python scripts/run_weekend.py [--week 2026-W46] [--dry-run] [--synthetic]
                                     [--allow-over-budget] [--gateway-fallback] [--now ISO]
                                     [--poll-seconds N] [--from J3] [--until J5]
                                     [--check-walltime HH:MM:SS]

v2 phase 12: `--from <job>` resumes at that job (earlier jobs are left as they are), `--until`
stops after a job (login mode runs J3 to J5 on the login node), and before J4 a health check
sends one tiny call per gateway model; on a failure without `--gateway-fallback` the D2 steps
are printed and the affected sources are marked failed (N1) so the run can continue.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path
from typing import Any

from portstudy import calendar as cal
from portstudy.llm import health
from portstudy.llm.runner import lock_deadline
from portstudy.pipeline import fallback as fb
from portstudy.pipeline.context import RunContext, default_context, synthetic_clock
from portstudy.pipeline.jobs import current_week, read_weights_lock, run_named, weeks_with_weights
from portstudy.pipeline.status import JobFailed, read_status
from portstudy.types import PortfolioId

FALLBACK_HINTS = {
    "J1": "data failure: every portfolio holds last week's positions (no_rebalance)",
    "J2": "baseline failure: every portfolio holds last week's positions (no_rebalance)",
    "J3": "batch not submitted: the API portfolios trade the Q0 weights (fallback_q0)",
    "J4": (
        "open-weight run incomplete: those portfolios trade the Q0 weights "
        "(fallback_q0) at the lock"
    ),
    "J5": "batch not collected: those portfolios trade the Q0 weights (fallback_q0) at the lock",
    "J6": "validation did not finish: rerun J6 before the lock margin",
    "J7": "weights not written: fix and rerun J7 before the lock deadline",
    "J8": "replay or scoring not done: results lag until J8 runs",
    "J9": "backup not written: run J9 or `portstudy backup`",
}


def say(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def rerun_command(job: str, week: str, synthetic: bool, gateway_fallback: bool = False) -> str:
    return (
        f"portstudy run-job {job} --week {week}"
        + (" --synthetic" if synthetic else "")
        + (" --gateway-fallback" if gateway_fallback else "")
    )


def run(ctx: RunContext, job: str, week: str, synthetic: bool) -> str:
    """Run one job, print its outcome, and return the state."""
    try:
        status = run_named(ctx, job, week)
    except Exception as exc:  # a JobFailed, or a hard crash that left no status file
        say(f"{job} {week} FAILED: {exc}")
        say(f"fallback that will apply: {FALLBACK_HINTS.get(job, '')}")
        say(f"rerun with: {rerun_command(job, week, synthetic, ctx.gateway_fallback)}")
        raise JobFailed(job, week, str(exc)) from exc
    result = status.result or {}
    summary = _summarize(job, result)
    say(f"{job} {week}: {status.state}{(' ' + summary) if summary else ''}")
    return status.state


def _summarize(job: str, result: dict[str, Any]) -> str:
    if job == "J1":
        return f"stocks={result.get('n_included')} cutoff={result.get('cutoff_date')}"
    if job == "J2":
        return (
            f"stocks={result.get('n_stocks')} alpha={result.get('alpha', 0):.3f}" if result else ""
        )
    if job == "J4":
        parts = [
            f"{m}: calls={v.get('n_calls')} cached={v.get('n_cached')} invalid={v.get('n_invalid')}"
            for m, v in result.items()
            if isinstance(v, dict)
        ]
        return "; ".join(parts)
    if job == "J6":
        return f"fallbacks={result.get('n_fallback')} {result.get('reasons') or ''}"
    if job == "J7":
        return f"portfolios={result.get('n_portfolios')}"
    if job == "J8":
        return f"weeks={len(result.get('weeks', []))} ic_rows={result.get('ic_rows')}"
    if job == "J9":
        return f"path={result.get('path')}"
    return ""


def wait_for_batches(ctx: RunContext, week: str, synthetic: bool, poll_seconds: int) -> None:
    deadline = lock_deadline(week, ctx.cfg) - dt.timedelta(
        minutes=ctx.cfg.pipeline.lock_margin_minutes
    )
    while True:
        state = run(ctx, "J5", week, synthetic)
        if state != "pending":
            return
        now = ctx.now()
        remaining = deadline - now
        if remaining.total_seconds() <= 0:
            say("lock margin reached with batches still pending; J6 will label them as fallbacks")
            return
        if ctx.clock is not None and poll_seconds == 0:
            # A fixed clock never advances: nothing more can happen in this run.
            say("batches pending under a fixed clock; stopping here (rerun later)")
            return
        say(f"batches pending; next poll in {poll_seconds}s, {remaining} until the lock margin")
        time.sleep(poll_seconds)


def step1_catch_up(ctx: RunContext, week: str, synthetic: bool) -> None:
    earlier = [w for w in weeks_with_weights() if w < week]
    for w in earlier:
        for job in ("J8", "J9"):
            if read_status(w, job).state != "done":
                say(f"catch-up: {job} for {w}")
                run(ctx, job, w, synthetic)


def print_summary(week: str) -> None:
    lock = read_weights_lock(week)
    print()
    print(f"Summary for {week}")
    print(f"{'portfolio':32s} {'status':14s} {'weights file':40s} sha256")
    if lock is None:
        print("  no locked weights")
        return
    if lock.get("no_rebalance"):
        print("  no_rebalance for every portfolio (hold)")
        return
    for name, digest in sorted(lock["files"].items()):
        if not name.endswith(".csv"):
            continue  # meta files are locked too but have no status of their own
        key = name[:-4]
        status = fb.week_status_for(week, key) if not PortfolioId.parse(key).is_q0 else "normal"
        print(f"{key:32s} {status:14s} {name:40s} {digest[:16]}")
    print(f"locked_at: {lock.get('locked_at')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--allow-over-budget", action="store_true")
    parser.add_argument("--gateway-fallback", action="store_true")
    parser.add_argument("--config", default=None, help="study YAML path")
    parser.add_argument("--now", default=None, help="override the clock, ISO datetime with offset")
    parser.add_argument("--poll-seconds", type=int, default=None)
    parser.add_argument("--from", dest="from_job", default="J1", help="resume at this job")
    parser.add_argument("--until", dest="until_job", default="J9", help="stop after this job")
    parser.add_argument(
        "--check-walltime", default=None, help="Slurm walltime; warn if it ends after the lock"
    )
    args = parser.parse_args(argv)
    if args.from_job not in STEP_ORDER or args.until_job not in STEP_ORDER:
        parser.error(f"--from/--until must be one of {STEP_ORDER}")

    clock = None
    if args.now:
        fixed = dt.datetime.fromisoformat(args.now)
        clock = lambda: fixed  # noqa: E731
    elif args.synthetic and args.week:
        clock = synthetic_clock(args.week)
    ctx = default_context(
        synthetic=args.synthetic,
        allow_over_budget=args.allow_over_budget,
        clock=clock,
        config_path=Path(args.config) if args.config else None,
        gateway_fallback=args.gateway_fallback,
    )
    week = args.week or current_week(ctx.now())
    poll = (
        args.poll_seconds
        if args.poll_seconds is not None
        else (0 if args.synthetic else ctx.cfg.llm.api_poll_seconds)
    )

    now = ctx.now()
    if args.check_walltime:
        ends = now + health.parse_walltime(args.check_walltime)
        deadline = lock_deadline(week, ctx.cfg)
        if ends > deadline:
            say(
                f"warning: walltime {args.check_walltime} ends {ends.isoformat()} after the lock "
                f"deadline {deadline.isoformat()}; shorten the job or start earlier"
            )
        else:
            say(f"walltime {args.check_walltime} ends {ends.isoformat()} before the lock")
    if now.weekday() == 5:
        say("warning: Saturday start leaves a smaller buffer for a batch retry before the lock")
    say(
        f"week {week}: cutoff {cal.cutoff_day(week)}, trade date {cal.monday_open_day(week)}, "
        f"lock {lock_deadline(week, ctx.cfg).isoformat()}"
    )

    plan = [
        ("step 1", "J8/J9 for earlier weeks"),
        ("step 2", "J1 J2 J3 J4 J5(poll) J6"),
        ("step 3", "J7 J9"),
    ]
    if args.dry_run:
        for s, d in plan:
            print(f"{s}: {d}")
        for job in ctx.cfg.pipeline.jobs:
            print(f"  {job}: {read_status(week, job).state}")
        return 0
    jobs = jobs_between(args.from_job, args.until_job)
    say(f"jobs this run: {' '.join(jobs)}")
    explicit_j8 = "J8" in (args.from_job, args.until_job)
    try:
        # Review fix H4: the catch-up (J8 and J9 for earlier weeks) runs on every invocation, so
        # the ARCTIC flows that start at J3 or J6 still replay and score the earlier weeks.
        step1_catch_up(ctx, week, args.synthetic)
        for job in jobs:
            if job == "J4":
                gateway_health(ctx, week)
            if job == "J5":
                wait_for_batches(ctx, week, args.synthetic, poll)
            elif job == "J8" and not explicit_j8:
                continue  # this week's replay runs in next weekend's catch-up
            elif job in ("J1", "J2"):
                try:
                    run(ctx, job, week, args.synthetic)
                except JobFailed:
                    # Review fix H2: write the hold for the week now (J6 plans no_rebalance when
                    # J2 is not done, J7 locks it) so the replay never skips the week.
                    say(f"{job} failed: writing the no_rebalance plan and lock for {week}")
                    for hold_job in ("J6", "J7"):
                        run(ctx, hold_job, week, args.synthetic)
                    return 1
            else:
                run(ctx, job, week, args.synthetic)
    except JobFailed:
        return 1
    if "J9" in jobs:
        print_summary(week)
    return 0


STEP_ORDER = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9"]


def jobs_between(from_job: str, until_job: str) -> list[str]:
    start, stop = STEP_ORDER.index(from_job), STEP_ORDER.index(until_job)
    if stop < start:
        raise SystemExit(f"--until {until_job} comes before --from {from_job}")
    return STEP_ORDER[start : stop + 1]


def gateway_health(ctx: RunContext, week: str) -> health.HealthReport:
    """One tiny call per gateway model before J4; print the D2 steps and mark sources on failure."""
    from portstudy.pipeline.jobs import _week_runner

    report = health.gateway_health_check(ctx, week, _week_runner(ctx, week))
    if report.healthy:
        checked = ", ".join(report.ok) if report.ok else "no gateway models"
        say(f"gateway health check ok: {checked}")
        return report
    for line in health.d2_steps(week, report.failed):
        say(line)
    if ctx.gateway_fallback:
        say(
            "continuing with --gateway-fallback: fallback-eligible models use the fallback endpoint"
        )
        return report
    keys = health.mark_sources_failed(ctx, week, sorted(report.failed), health.HEALTH_REASON)
    say(f"marked failed (N1): {', '.join(keys)}; the rest of the week continues")
    return report


if __name__ == "__main__":
    sys.exit(main())
