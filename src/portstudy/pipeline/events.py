"""Append-only event timeline, one JSON line per event in data/logs/runs/<run_id>.jsonl (N6 (2)).

Run start and end, job start, end, skip and failure, preflight result, retries, fallbacks,
warnings, flags, interruptions and resumes all go through `emit`, so the timeline of a weekend
can be read in one file. Lines are flushed and fsynced as they are written.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from portstudy.io import append_line_durable
from portstudy.logging_setup import current_run_id, now_et
from portstudy.paths import logs_dir

NO_RUN = "no-run-id"
# Random-kill test (N6): with PORTSTUDY_TEST_KILL_AT_EVENT=N the process dies without cleanup
# right after writing its N-th event, the same effect as SIGKILL at that point.
_KILL_AT = int(os.environ.get("PORTSTUDY_TEST_KILL_AT_EVENT", "0") or 0)
_N_EVENTS = 0


def timeline_path(run_id: str | None = None) -> Path:
    return logs_dir() / "runs" / f"{run_id or current_run_id() or NO_RUN}.jsonl"


def emit(kind: str, **fields: Any) -> None:
    """Record one event for the current run."""
    run_id = current_run_id() or NO_RUN
    line = {"ts": now_et().isoformat(), "run_id": run_id, "kind": kind, **fields}
    append_line_durable(timeline_path(run_id), json.dumps(line, sort_keys=True, default=str))
    global _N_EVENTS
    _N_EVENTS += 1
    if _KILL_AT and _N_EVENTS >= _KILL_AT:
        os._exit(137)


def read_events(run_id: str | None = None) -> list[dict[str, Any]]:
    """Events of one run (the current run by default); every run's events with `run_id="*"`."""
    paths = (
        sorted((logs_dir() / "runs").glob("*.jsonl")) if run_id == "*" else [timeline_path(run_id)]
    )
    out: list[dict[str, Any]] = []
    for path in paths:
        if path.exists():
            out.extend(json.loads(ln) for ln in path.read_text("utf-8").splitlines() if ln.strip())
    return out
