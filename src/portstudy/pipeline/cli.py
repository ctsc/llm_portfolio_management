"""`portstudy` command line: run-job, run-week, status, replay, backup, eval."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from pathlib import Path

import typer

from portstudy.pipeline.context import RunContext, default_context, synthetic_clock
from portstudy.pipeline.jobs import JOBS, current_week, run_named
from portstudy.pipeline.status import JobFailed, all_statuses

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
    gateway_fallback: bool = False,
) -> RunContext:
    return default_context(
        synthetic=synthetic,
        allow_over_budget=allow_over_budget,
        force=force,
        clock=_clock(now, synthetic, week),
        config_path=Path(config) if config else None,
        gateway_fallback=gateway_fallback,
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
    gateway_fallback: bool = typer.Option(
        False, "--gateway-fallback", help="use the configured Slurm endpoint for eligible models"
    ),
) -> None:
    ctx = _context(synthetic, allow_over_budget, force, now, week, config, gateway_fallback)
    w = _week_or_current(week, ctx)
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
    gateway_fallback: bool = typer.Option(
        False, "--gateway-fallback", help="use the configured Slurm endpoint for eligible models"
    ),
) -> None:
    """Run the jobs in order for one week, stopping at the first failure."""
    ctx = _context(synthetic, allow_over_budget, force, now, week, config, gateway_fallback)
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
                + (" --gateway-fallback" if gateway_fallback else "")
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
    skip_shuffle: bool = typer.Option(False, "--skip-shuffle", help="skip the slow shuffle test"),
    config: str | None = typer.Option(None, "--config", help="study YAML path"),
) -> None:
    try:
        from portstudy.eval.report import run_eval
    except ImportError as exc:  # Phase 11 adds the evaluation engine
        typer.echo(f"evaluation engine not available: {exc}")
        raise typer.Exit(code=1) from exc
    ctx = _context(synthetic, False, True, None, None, config)
    out = run_eval(ctx, skip_shuffle=skip_shuffle)
    typer.echo(f"report written to {out}")


@app.command("jobs")
def jobs_cmd() -> None:
    for name in JOBS:
        typer.echo(name)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
