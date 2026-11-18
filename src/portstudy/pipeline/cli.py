"""`portstudy` command line: run-job, run-week, status, replay, backup, eval."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from pathlib import Path

import typer

from portstudy.pipeline.context import RunContext, default_context, synthetic_clock
from portstudy.pipeline.jobs import JOBS, current_week, run_named
from portstudy.pipeline.status import JobFailed, all_statuses, strategy_status_summary

app = typer.Typer(help="Weekly pipeline for the LLM skills study.", no_args_is_help=True)


def _clock(now: str | None, synthetic: bool, week: str | None) -> Callable[[], dt.datetime] | None:
    if now:
        fixed = dt.datetime.fromisoformat(now)
        return lambda: fixed
    if synthetic and week:
        return synthetic_clock(week)
    return None


def _context(
    synthetic: bool,
    allow_over_budget: bool,
    force: bool,
    now: str | None,
    week: str | None,
    config: str | None,
) -> RunContext:
    return default_context(
        synthetic=synthetic,
        allow_over_budget=allow_over_budget,
        force=force,
        clock=_clock(now, synthetic, week),
        config_path=Path(config) if config else None,
    )


def _week_or_current(week: str | None, ctx: RunContext) -> str:
    return week or current_week(ctx.now())


@app.command("run-job")
def run_job_cmd(
    job: str = typer.Argument(..., help="J1 .. J9"),
    week: str | None = typer.Option(None, "--week", help="ISO week id like 2026-W46"),
    synthetic: bool = typer.Option(
        False, "--synthetic", help="use the synthetic world and stub LLM"
    ),
    allow_over_budget: bool = typer.Option(False, "--allow-over-budget"),
    force: bool = typer.Option(False, "--force", help="rerun even if done with the same inputs"),
    now: str | None = typer.Option(
        None, "--now", help="override the clock (ISO datetime with offset)"
    ),
    config: str | None = typer.Option(None, "--config", help="study YAML path"),
) -> None:
    ctx = _context(synthetic, allow_over_budget, force, now, week, config)
    w = _week_or_current(week, ctx)
    from portstudy.logging_setup import new_run_id, set_run_id

    set_run_id(new_run_id())  # N6 (1): a job run alone gets its own run_id
    try:
        status = run_named(ctx, job.upper(), w)
    except JobFailed as exc:
        typer.echo(f"FAILED {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"{job.upper()} {w}: {status.state} {json.dumps(status.result or {}, default=str)[:400]}"
    )
    if status.state == "failed":
        raise typer.Exit(code=1)


@app.command("run-week")
def run_week_cmd(
    week: str | None = typer.Option(None, "--week"),
    synthetic: bool = typer.Option(False, "--synthetic"),
    allow_over_budget: bool = typer.Option(False, "--allow-over-budget"),
    force: bool = typer.Option(False, "--force"),
    now: str | None = typer.Option(None, "--now"),
    jobs: str = typer.Option("", "--jobs", help="comma-separated subset, default all in order"),
    config: str | None = typer.Option(None, "--config", help="study YAML path"),
) -> None:
    """Run the jobs in order for one week, stopping at the first failure."""
    ctx = _context(synthetic, allow_over_budget, force, now, week, config)
    w = _week_or_current(week, ctx)
    sequence = [j.strip().upper() for j in jobs.split(",") if j.strip()] or list(
        ctx.cfg.pipeline.jobs
    )
    for job in sequence:
        try:
            status = run_named(ctx, job, w)
        except JobFailed as exc:
            typer.echo(f"FAILED {exc}")
            typer.echo(
                f"rerun with: portstudy run-job {job} --week {w}"
                + (" --synthetic" if synthetic else "")
            )
            raise typer.Exit(code=1) from exc
        typer.echo(f"{job} {w}: {status.state}")
        if status.state == "pending":
            typer.echo(f"{job} is still pending (batch not ready); rerun later")
            raise typer.Exit(code=2)


@app.command("status")
def status_cmd(
    week: str | None = typer.Option(None, "--week"),
    synthetic: bool = typer.Option(False, "--synthetic"),
    config: str | None = typer.Option(None, "--config", help="study YAML path"),
) -> None:
    ctx = _context(synthetic, False, False, None, week, config)
    w = _week_or_current(week, ctx)
    for job, st in all_statuses(w).items():
        line = f"{job:3s} {st.state:8s}"
        if st.started_at:
            line += f" started {st.started_at}"
        if st.finished_at:
            line += f" finished {st.finished_at}"
        if st.error:
            line += f" error: {st.error}"
        typer.echo(line)
    from portstudy.pipeline.study_log import read_study_log

    log = read_study_log()
    if len(log):
        typer.echo("study log (last rows):")
        cols = ["week_id", "failed_job", "universe_size", "n_removed", "n_fallback_rows"]
        cols += ["arctic_gpu_hours_total", "litellm_cost_total", "audit_ok"]
        typer.echo(log.tail(5).loc[:, cols].to_string(index=False))
    for row in strategy_status_summary(w, ctx):
        mean_shift = row["mean_view_shift"]
        shift_text = f"{mean_shift:.6f}" if mean_shift is not None else "n/a"
        typer.echo(
            f"strategy {row['strategy']}: solver_failures={row['solver_failures']} "
            f"mean_view_shift={shift_text}"
        )
        for source in row["sources"]:
            valid_rate = source["valid_rate"]
            valid_text = f"{valid_rate:.3f}" if valid_rate is not None else "n/a"
            typer.echo(
                f"  source {source['source']}: endpoint={source['endpoint'] or 'n/a'} "
                f"valid_rate={valid_text} fallback_reason={source['fallback_reason'] or 'none'}"
            )


@app.command("replay")
def replay_cmd(
    week: str | None = typer.Option(
        None, "--week", help="replay through this week (default: latest)"
    ),
    synthetic: bool = typer.Option(False, "--synthetic"),
    config: str | None = typer.Option(None, "--config", help="study YAML path"),
) -> None:
    from portstudy.pipeline.jobs import weeks_with_weights

    ctx = _context(synthetic, False, True, None, week, config)
    weeks = weeks_with_weights()
    if not weeks:
        typer.echo("no locked weights found")
        raise typer.Exit(code=1)
    w = week or weeks[-1]
    status = run_named(ctx, "J8", w)
    typer.echo(f"J8 {w}: {status.state} {json.dumps(status.result or {}, default=str)[:400]}")


@app.command("backup")
def backup_cmd(
    week: str | None = typer.Option(None, "--week"),
    synthetic: bool = typer.Option(False, "--synthetic"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    config: str | None = typer.Option(None, "--config", help="study YAML path"),
) -> None:
    from portstudy.backup import run_backup

    ctx = _context(synthetic, False, True, None, week, config)
    w = _week_or_current(week, ctx)
    path = run_backup(w, ctx.cfg, dry_run=dry_run, logger=ctx.log("backup"))
    typer.echo(f"backup folder: {path}")


@app.command("eval")
def eval_cmd(
    synthetic: bool = typer.Option(False, "--synthetic"),
    config: str | None = typer.Option(None, "--config", help="study YAML path"),
    factor_source: str | None = typer.Option(
        None, "--factor-source", help="french or etf_proxy (overrides eval.factor_source)"
    ),
    out: str | None = typer.Option(None, "--out", help="report folder (default results/eval)"),
) -> None:
    """Write the evaluation report, every table as CSV and all_metrics.csv (PLAN4 Phase 9)."""
    from pathlib import Path

    from portstudy.eval.report import run_eval

    if factor_source not in (None, "french", "etf_proxy"):
        raise typer.BadParameter("--factor-source must be french or etf_proxy")
    ctx = _context(synthetic, False, True, None, None, config)
    folder = run_eval(ctx, Path(out) if out else None, factor_source)
    typer.echo(f"report written to {folder}")


@app.command("audit")
def audit_cmd(
    week: str | None = typer.Option(None, "--week"),
    synthetic: bool = typer.Option(False, "--synthetic"),
    config: str | None = typer.Option(None, "--config", help="study YAML path"),
) -> None:
    """Verify a week end to end (N6 (9)); exit code 1 on any problem."""
    from portstudy.pipeline.audit import audit_week

    ctx = _context(synthetic, False, False, None, week, config)
    w = _week_or_current(week, ctx)
    problems = audit_week(ctx, w)
    for line in problems:
        typer.echo(f"problem: {line}")
    typer.echo(f"audit {w}: {'ok' if not problems else f'{len(problems)} problem(s)'}")
    if problems:
        raise typer.Exit(code=1)


@app.command("canary")
def canary_cmd(
    week: str | None = typer.Option(None, "--week"),
    synthetic: bool = typer.Option(False, "--synthetic"),
    config: str | None = typer.Option(None, "--config", help="study YAML path"),
) -> None:
    """G21: gateway model list and canary prompts for the week; flags never fail the command."""
    from portstudy.llm.canary import run_canary
    from portstudy.pipeline.jobs import _week_runner

    ctx = _context(synthetic, False, False, None, week, config)
    w = _week_or_current(week, ctx)
    for row in run_canary(ctx, w, _week_runner(ctx, w)):
        flag = " MODEL_CHANGE" if row["model_change"] else ""
        typer.echo(f"{row['model']}: {row['matches']}/{row['n']} identical{flag} {row['reasons']}")


@app.command("repeat-check")
def repeat_check_cmd(
    week: str | None = typer.Option(None, "--week"),
    synthetic: bool = typer.Option(False, "--synthetic"),
    config: str | None = typer.Option(None, "--config", help="study YAML path"),
) -> None:
    """G20 (once, in the pilot after J4): re-rate 50 stocks, write pilot_repeatability.csv and
    the canary baseline."""
    from portstudy.llm.canary import repeat_check
    from portstudy.pipeline.jobs import _week_runner

    ctx = _context(synthetic, False, False, None, week, config)
    w = _week_or_current(week, ctx)
    table = repeat_check(ctx, w, _week_runner(ctx, w))
    typer.echo(table.to_string(index=False) if len(table) else "no gateway models checked")


@app.command("jobs")
def jobs_cmd() -> None:
    for name in JOBS:
        typer.echo(name)


ledger_app = typer.Typer(help="Price-ledger maintenance.", no_args_is_help=True)
app.add_typer(ledger_app, name="ledger")


@ledger_app.command("rebuild")
def ledger_rebuild_cmd(
    start: str | None = typer.Option(
        None,
        "--start",
        help="first fetch date (default: first snapshot cutoff minus history buffer; "
        "required when no snapshot exists)",
    ),
    yes: bool = typer.Option(False, "--yes", help="confirm archiving the existing ledger"),
    synthetic: bool = typer.Option(False, "--synthetic"),
    config: str | None = typer.Option(None, "--config", help="study YAML path"),
) -> None:
    """Archive the current price ledger and re-fetch it for the whole study window."""
    import datetime as _dt

    from portstudy import calendar as _cal
    from portstudy.data import ledger as _ledger
    from portstudy.data.universe import load_universe, universe_path
    from portstudy.paths import snapshots_dir

    ctx = _context(synthetic, False, True, None, None, config)

    weeks = (
        sorted(p.name for p in snapshots_dir().iterdir() if p.is_dir())
        if snapshots_dir().exists()
        else []
    )
    if start is None:
        if not weeks:
            typer.echo("--start is required: no snapshot exists to anchor the history")
            raise typer.Exit(code=1)
        first_cutoff = _cal.cutoff_day(weeks[0])
        start_date = first_cutoff - _dt.timedelta(
            days=int(ctx.cfg.data.price_history_days * 1.6) + 30
        )
    else:
        start_date = _dt.date.fromisoformat(start)
    end_date = _cal.prev_trading_day(ctx.now().date(), inclusive=True)

    universe = load_universe(universe_path(ctx.cfg, synthetic=ctx.synthetic))
    tickers = sorted(universe["ticker"].astype(str))

    if not yes:
        typer.echo(
            f"would archive data/ledger files and re-fetch {len(tickers)} tickers "
            f"(+ ETF proxies) from {start_date} to {end_date}"
        )
        raise typer.Exit(code=1)

    archive, n = _ledger.rebuild_ledger(
        ctx.cfg, ctx.providers(), tickers, start_date, end_date, ctx.log("ledger")
    )
    typer.echo(f"ledger rebuilt: {n} price rows; archive at {archive}")


@app.command("closeout")
def closeout_cmd(
    week: str | None = typer.Option(
        None, "--week", help="last live week (default: timing.live_end_date week)"
    ),
    now: str | None = typer.Option(None, "--now", help="fixed clock for tests"),
    synthetic: bool = typer.Option(False, "--synthetic"),
    config: str | None = typer.Option(None, "--config", help="study YAML path"),
) -> None:
    """End-of-study closeout: final J8 and the study_complete.json marker."""
    from portstudy.pipeline.closeout import CloseoutNotReady, run_closeout

    ctx = _context(synthetic, False, True, now, week, config)
    try:
        out = run_closeout(ctx, week, logger=ctx.log("closeout"))
    except CloseoutNotReady as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(f"study complete: {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
