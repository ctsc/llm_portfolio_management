"""The one script Carter runs by hand on the desktop each weekend (DECISION(N2)). No scheduler.

Step 1: J8 and J9 for any earlier week whose replay or backup is not done.
Step 2: J1 to J6 for the current week in order, waiting on API batches (J5) with a visible
        countdown until done or the lock deadline minus the margin.
Step 3: J7, then J9.

Start Friday night if possible; a Saturday start leaves a smaller buffer for a batch retry.
On a failure the script stops, prints what failed, the fallback that will apply, and the exact
command to rerun that one job. Rerunning the script resumes where it stopped.

Usage: python scripts/run_weekend.py [--week 2026-W46] [--dry-run] [--synthetic]
                                     [--allow-over-budget] [--now ISO]
                                     [--poll-seconds N] [--from J3] [--until J5] [--skip-preflight]

`--from <job>` resumes at that job (earlier jobs are left as they are) and `--until` stops after
a job. Before J4 a health check sends one tiny call per gateway model; on a failure the D2 lines
are printed and the affected sources are marked failed (N1) so the run can continue, and
`--from J4` retries them once the gateway is back.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from portstudy import calendar as cal
from portstudy.config import repo_root
from portstudy.data.snapshot import snapshot_exists
from portstudy.llm import health
from portstudy.llm.runner import lock_deadline
from portstudy.llm.usage import BudgetCheck, check_dollar_budget, usage_summary
from portstudy.logging_setup import current_run_id, new_run_id, set_run_id
from portstudy.pipeline import fallback as fb
from portstudy.pipeline import study_log
from portstudy.pipeline.audit import audit_week
from portstudy.pipeline.context import RunContext, default_context, synthetic_clock
from portstudy.pipeline.events import emit
from portstudy.pipeline.frozen import save_frozen_config
from portstudy.pipeline.jobs import current_week, read_weights_lock, run_named, weeks_with_weights
from portstudy.pipeline.keep_awake import start_keep_awake, stop_keep_awake
from portstudy.pipeline.preflight import live_ready
from portstudy.pipeline.progress import read_progress
from portstudy.pipeline.run_summary import write_run_summary
from portstudy.pipeline.status import JobFailed, read_status
from portstudy.types import PortfolioId

FALLBACK_HINTS = {
    "J1": "data failure: every portfolio holds last week's positions (no_rebalance)",
    "J2": "baseline failure: every portfolio holds last week's positions (no_rebalance)",
    "J3": "batch not submitted: the API portfolios trade the Q0 weights (fallback_q0)",
    "J4": (
        "open-weight run incomplete: rerun `python scripts/run_weekend.py --from J4` to resume "
        "from the saved records; at the lock margin unfinished sources trade Q0"
    ),
    "J5": "batch not collected: those portfolios trade the Q0 weights (fallback_q0) at the lock",
    "J6": (
        "fallback plan not written: rerun J6; it marks unfinished or failed sources as fallback "
        "and the week holds if J2 never ran"
    ),
    "J7": "weights not written: fix and rerun J7 before the lock deadline",
    "J8": "replay or scoring not done: results lag until J8 runs",
    "J9": "backup not written: run J9 or `portstudy backup`",
}


def say(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def rerun_command(job: str, week: str, synthetic: bool) -> str:
    return f"portstudy run-job {job} --week {week}" + (" --synthetic" if synthetic else "")


def run(ctx: RunContext, job: str, week: str, synthetic: bool) -> str:
    """Run one job, print its outcome, and return the state."""
    try:
        status = run_named(ctx, job, week)
    except Exception as exc:  # a JobFailed, or a hard crash that left no status file
        say(f"{job} {week} FAILED: {exc}")
        say(f"fallback that will apply: {FALLBACK_HINTS.get(job, '')}")
        say(f"rerun with: {rerun_command(job, week, synthetic)}")
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
        parts = []
        for m, v in result.items():
            if not isinstance(v, dict):
                continue
            if v.get("crashed"):
                parts.append(f"{m}: CRASHED {v['crashed']} (sources go to fallback)")
            elif v.get("skipped"):
                parts.append(f"{m}: skipped {v['skipped']}")
            else:
                parts.append(
                    f"{m}: calls={v.get('n_calls')} cached={v.get('n_cached')} "
                    f"invalid={v.get('n_invalid')} provider_error={v.get('n_provider_error')}"
                )
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


CONNECTION_MARKERS = ("ConnectError", "ConnectTimeout", "ConnectionError", "unreachable")


def wait_for_batches(ctx: RunContext, week: str, synthetic: bool, poll_seconds: int) -> None:
    deadline = lock_deadline(week, ctx.cfg) - dt.timedelta(
        minutes=ctx.cfg.pipeline.lock_margin_minutes
    )
    outage_started: float | None = None
    tolerance_s = 60.0 * ctx.cfg.llm.network_outage_tolerance_minutes
    while True:
        try:
            state = run(ctx, "J5", week, synthetic)
        except JobFailed as exc:
            # Network outage tolerance for batch polling: a connection error (VPN drop) is
            # retried within the window and never past the lock margin.
            text = str(exc)
            if not any(k in text for k in CONNECTION_MARKERS) or ctx.now() >= deadline:
                raise
            now_m = time.monotonic()
            outage_started = now_m if outage_started is None else outage_started
            if now_m - outage_started >= tolerance_s:
                raise
            say("provider unreachable while polling batches; check the GSU VPN (retrying in 30s)")
            time.sleep(30)
            continue
        outage_started = None
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
    parser.add_argument("--config", default=None, help="study YAML path")
    parser.add_argument("--now", default=None, help="override the clock, ISO datetime with offset")
    parser.add_argument("--poll-seconds", type=int, default=None)
    parser.add_argument("--from", dest="from_job", default="J1", help="resume at this job")
    parser.add_argument("--until", dest="until_job", default="J9", help="stop after this job")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip the live-readiness checks (check_live_ready.py is the standalone run)",
    )
    parser.add_argument(
        "--no-keep-awake",
        action="store_true",
        help="do not start the WSL helper that keeps Windows from sleeping",
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
    )
    week = args.week or current_week(ctx.now())
    poll = (
        args.poll_seconds
        if args.poll_seconds is not None
        else (0 if args.synthetic else ctx.cfg.llm.api_poll_seconds)
    )

    now = ctx.now()
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
    awake = None if args.no_keep_awake else start_keep_awake()
    try:
        return run_live(ctx, args, week, poll, argv)
    finally:
        stop_keep_awake(awake)


def run_live(
    ctx: RunContext, args: argparse.Namespace, week: str, poll: int, argv: list[str] | None
) -> int:
    """Everything after the dry-run check: preflight, jobs, audit, study log and summary."""
    set_run_id(new_run_id())
    started_at = ctx.now().isoformat()
    emit("run_start", week_id=week, argv=list(argv or sys.argv[1:]), synthetic=args.synthetic)
    say(f"run_id {current_run_id()}")
    if not args.synthetic:
        if not args.skip_preflight:
            report = live_ready(
                ctx.cfg,
                ctx.registry,
                ctx.manifest,
                root=repo_root(),
                env=os.environ,
                now=ctx.now(),
                week_id=week,
            )
            for line in report.failures:
                say(f"preflight FAIL: {line}")
            for line in report.warnings:
                say(f"preflight WARN: {line}")
            emit("preflight", week_id=week, failures=report.failures, warnings=report.warnings)
            if report.failures:
                say(
                    "preflight failed; fix the failures or rerun with --skip-preflight "
                    "(see scripts/check_live_ready.py)"
                )
                emit("run_end", week_id=week, exit_code=2, reason="preflight")
                return 2
        if week == ctx.cfg.timing.pilot_week:
            # Phase 3C frozen-config: the pilot run snapshots the configuration every later
            # week is diffed against.
            save_frozen_config(repo_root())
            say("pilot week: configuration frozen under data/frozen/")
    interrupted = [job for job in STEP_ORDER if read_status(week, job).state == "running"]
    if interrupted:
        say(f"previous run was interrupted during {' '.join(interrupted)}; resuming")
        emit("resume", week_id=week, interrupted=interrupted)
    jobs = jobs_between(args.from_job, args.until_job)
    say(f"jobs this run: {' '.join(jobs)}")
    code, failed_job, messages = run_jobs(ctx, week, jobs, args, poll)
    audit_problems: list[str] | None = None
    if code == 0 and "J9" in jobs:
        print_summary(week)
        audit_problems = audit_week(ctx, week)
        for line in audit_problems:
            say(f"audit: {line}")
        if not audit_problems:
            say("audit ok")
    finish_run(ctx, week, started_at, failed_job, messages, audit_problems)
    emit("run_end", week_id=week, exit_code=code, failed_job=failed_job)
    return code


def run_jobs(
    ctx: RunContext, week: str, jobs: list[str], args: argparse.Namespace, poll: int
) -> tuple[int, str, list[str]]:
    """Run the requested jobs; returns the exit code, the failing job and summary messages."""
    explicit_j8 = "J8" in (args.from_job, args.until_job)
    messages: list[str] = []
    current = "catch-up"
    try:
        # Review fix H4: the catch-up (J8 and J9 for earlier weeks) runs on every invocation, so
        # a rerun that starts at J3 or J6 still replays and scores the earlier weeks.
        step1_catch_up(ctx, week, args.synthetic)
        for job in jobs:
            current = job
            if job == "J4":
                budget = dollar_budget_check(ctx, week)
                if budget is not None:
                    say(budget.message)
                    if not budget.ok:
                        emit("budget_stop", week_id=week, message=budget.message)
                        return 1, "J4 (LiteLLM budget)", [budget.message]
                gateway_health(ctx, week)
                messages.extend(model_checks(ctx, week))
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
                    return 1, job, messages
            elif job == "J4":
                try:
                    run(ctx, job, week, args.synthetic)
                except JobFailed:
                    margin = lock_deadline(week, ctx.cfg) - dt.timedelta(
                        minutes=ctx.cfg.pipeline.lock_margin_minutes
                    )
                    if ctx.now() >= margin:
                        # N1: no time left to finish J4; J6 marks the unfinished sources as
                        # fallbacks and the week continues on their strategy's Q0.
                        say(
                            "J4 failed at the lock margin; continuing to J6 "
                            "(unfinished sources trade their strategy's Q0, N1)"
                        )
                        messages.append("J4 did not finish before the lock margin")
                        continue
                    raise
            else:
                run(ctx, job, week, args.synthetic)
            if job in ("J2", "J4", "J7"):
                messages.extend(report_progress(ctx, week))
    except JobFailed:
        return 1, current, messages
    return 0, "", messages


def model_checks(ctx: RunContext, week: str) -> list[str]:
    """G21 before J4: gateway model list and canaries. A flag is printed, never fatal."""
    from portstudy.llm import canary
    from portstudy.pipeline.jobs import _week_runner

    try:
        rows = canary.run_canary(ctx, week, _week_runner(ctx, week))
    except Exception as exc:  # a failed check must not stop the week
        say(f"model checks not run: {type(exc).__name__}: {exc}")
        emit("model_checks_failed", week_id=week, error=str(exc), traceback=traceback.format_exc())
        return [f"model checks not run: {exc}"]
    out = []
    for row in rows:
        say(f"canary {row['model']}: {row['matches']}/{row['n']} identical ({row['reasons']})")
        if row["model_change"]:
            line = f"model_change flag for {row['model']}: {row['reasons']}"
            say(f"warning: {line}")
            emit("flag", week_id=week, flag="model_change", model=row["model"])
            out.append(line)
    return out


def dollar_budget_check(ctx: RunContext, week: str) -> BudgetCheck | None:
    """DECISION(N7): stop before any gateway request when the LiteLLM budget would be passed."""
    gateway = [
        m for m in ctx.open_weight_models() if ctx.run_registry().get(m).serving == "gateway"
    ]
    if not gateway or not snapshot_exists(week):
        return None
    from portstudy.pipeline.jobs import _week_runner

    runner = _week_runner(ctx, week)
    n_planned = sum(runner.n_unrecorded(m) for m in gateway)
    if n_planned == 0:
        return None
    return check_dollar_budget(
        ctx.cfg, ctx.registry, week, n_planned, allow_over_budget=ctx.allow_over_budget
    )


def report_progress(ctx: RunContext, week: str) -> list[str]:
    """Print the long jobs' progress; warn when a projected finish passes the lock margin."""
    margin = lock_deadline(week, ctx.cfg) - dt.timedelta(
        minutes=ctx.cfg.pipeline.lock_margin_minutes
    )
    out: list[str] = []
    for job, entry in sorted(read_progress(week).items()):
        say(f"progress {job}: {entry['done']}/{entry['total']} in {entry['elapsed_s']:.0f}s")
        projected = entry.get("projected_finish")
        if projected and entry["done"] < entry["total"]:
            if dt.datetime.fromisoformat(projected) > margin:
                line = f"{job} is projected to finish {projected}, after the lock margin {margin}"
                say(f"warning: {line}")
                out.append(line)
    return out


def finish_run(
    ctx: RunContext,
    week: str,
    started_at: str,
    failed_job: str,
    messages: list[str],
    audit_problems: list[str] | None,
) -> None:
    """Study-log row, ARCTIC usage lines and run_summary.md, also after a failed run."""
    usage = usage_summary(ctx.cfg, ctx.registry, week)
    for line in usage.messages:
        say(line)
    try:
        study_log.upsert(study_log.week_row(ctx, week, started_at, failed_job, audit_problems))
    except Exception as exc:  # the log must never hide the run's own result
        say(f"study log not written: {type(exc).__name__}: {exc}")
        emit("study_log_failed", week_id=week, error=str(exc), traceback=traceback.format_exc())
    path, attention = write_run_summary(
        week, failed_job, [*messages, *usage.messages], audit_problems
    )
    say(f"run summary: {path}")
    if attention:
        say("needs attention:")
        for line in attention:
            say(f"  - {line}")


STEP_ORDER = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9"]


def jobs_between(from_job: str, until_job: str) -> list[str]:
    start, stop = STEP_ORDER.index(from_job), STEP_ORDER.index(until_job)
    if stop < start:
        raise SystemExit(f"--until {until_job} comes before --from {from_job}")
    return STEP_ORDER[start : stop + 1]


def gateway_health(ctx: RunContext, week: str) -> health.HealthReport:
    """One tiny call per gateway model before J4; print the D2 lines and mark sources on failure."""
    from portstudy.pipeline.jobs import _week_runner

    report = health.gateway_health_check(ctx, week, _week_runner(ctx, week))
    if report.healthy:
        checked = ", ".join(report.ok) if report.ok else "no gateway models"
        say(f"gateway health check ok: {checked}")
        return report
    for line in health.outage_lines(report.failed):
        say(line)
    keys = health.mark_sources_failed(ctx, week, sorted(report.failed), health.HEALTH_REASON)
    say(f"marked failed (N1): {', '.join(keys)}; the rest of the week continues")
    return report


if __name__ == "__main__":
    sys.exit(main())
