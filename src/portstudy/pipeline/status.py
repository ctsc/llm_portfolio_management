"""Per-job status files: data/outputs/<week>/jobs/<job>.status.json.

A job whose status is `done` with an unchanged `inputs_hash` is skipped unless forced.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from portstudy.hashing import sha256_file, stable_hash
from portstudy.io import read_json, write_json_atomic
from portstudy.logging_setup import now_et
from portstudy.paths import week_output_dir

JobState = Literal["pending", "running", "done", "failed", "skipped"]
JOB_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9"]


@dataclass
class JobStatus:
    job: str
    week_id: str
    state: JobState = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    inputs_hash: str | None = None
    outputs_hash: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobFailed(RuntimeError):
    def __init__(self, job: str, week_id: str, message: str) -> None:
        super().__init__(f"{job} failed for {week_id}: {message}")
        self.job = job
        self.week_id = week_id


def jobs_dir(week_id: str) -> Path:
    path = week_output_dir(week_id) / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def status_path(week_id: str, job: str) -> Path:
    return jobs_dir(week_id) / f"{job}.status.json"


def read_status(week_id: str, job: str) -> JobStatus:
    path = status_path(week_id, job)
    if not path.exists():
        return JobStatus(job=job, week_id=week_id)
    return JobStatus(**read_json(path))


def write_status(status: JobStatus) -> None:
    write_json_atomic(status.to_dict(), status_path(status.week_id, status.job))


def all_statuses(week_id: str) -> dict[str, JobStatus]:
    return {job: read_status(week_id, job) for job in JOB_NAMES}


def hash_paths(paths: list[Path]) -> str:
    """Stable hash of the contents of existing files (missing files hash as absent)."""
    return stable_hash({str(p.name): (sha256_file(p) if p.exists() else None) for p in paths})


def run_job(
    week_id: str,
    job: str,
    inputs_hash: str,
    fn: Callable[[], dict[str, Any] | None],
    force: bool = False,
    outputs: Callable[[], list[Path]] | None = None,
    pending_ok: bool = False,
) -> JobStatus:
    """Run `fn` under the status protocol. Returns the final status.

    If `fn` returns a dict containing `{"pending": True}` and `pending_ok` is set, the job is
    left in state `pending` (for example a batch that has not returned yet) rather than `done`.
    """
    current = read_status(week_id, job)
    if current.state == "done" and current.inputs_hash == inputs_hash and not force:
        current.state = "skipped"
        return current
    status = JobStatus(
        job=job,
        week_id=week_id,
        state="running",
        started_at=now_et().isoformat(),
        inputs_hash=inputs_hash,
    )
    write_status(status)
    try:
        result = fn() or {}
    except Exception as exc:  # the job's own error is recorded and re-raised as JobFailed
        status.state = "failed"
        status.finished_at = now_et().isoformat()
        status.error = f"{type(exc).__name__}: {exc}"
        status.result = {"traceback": traceback.format_exc()[-4000:]}
        write_status(status)
        raise JobFailed(job, week_id, status.error) from exc
    status.finished_at = now_et().isoformat()
    status.result = {k: v for k, v in result.items() if k != "traceback"}
    status.state = "pending" if (pending_ok and result.get("pending")) else "done"
    if outputs is not None and status.state == "done":
        status.outputs_hash = hash_paths(outputs())
    write_status(status)
    return status


def strategy_status_summary(week_id: str, ctx: Any) -> list[dict[str, Any]]:
    from portstudy.llm.runner import read_run_status
    from portstudy.optimize import read_weights_meta, weights_dir
    from portstudy.pipeline import fallback as fb
    from portstudy.types import PortfolioId

    run_status = read_run_status(week_id)
    plan = fb.read_plan(week_id)
    rows: list[dict[str, Any]] = []
    groups = [(name, False) for name in ctx.cfg.strategies.enabled] + [
        (name, True) for name in ctx.cfg.strategies.controls
    ]
    for name, is_control in groups:
        keys = (
            [name]
            if is_control
            else [PortfolioId.q0(name).key]
            + [PortfolioId.steered(name, source).key for source in ctx.sources()]
        )
        metas = {}
        for key in keys:
            path = weights_dir(week_id) / f"{key}.meta.json"
            if path.exists():
                metas[key] = read_weights_meta(week_id, key)
        shifts = [
            float(meta.get("view_shift", 0.0))
            for key, meta in metas.items()
            if key != name and meta.get("view_shift") is not None
        ]
        solver_failures = sum(bool(meta.get("fallback_flag")) for meta in metas.values())
        sources = []
        if not is_control:
            for source in ctx.sources():
                status = run_status.get(source.key)
                reason = plan.reason_for(source.key) if plan is not None else ""
                sources.append(
                    {
                        "source": source.key,
                        "endpoint": status.endpoint if status else "",
                        "valid_rate": 1.0 - status.invalid_rate if status else None,
                        "fallback_reason": reason or (status.reason if status else ""),
                    }
                )
        rows.append(
            {
                "strategy": name,
                "solver_failures": solver_failures,
                "mean_view_shift": sum(shifts) / len(shifts) if shifts else None,
                "sources": sources,
            }
        )
    return rows
