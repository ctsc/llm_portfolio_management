"""Progress of the long jobs, data/outputs/<week>/progress.json (Phase 3A).

J2 (Kronos chunks), J4 (gateway requests) and J7 (portfolios) report done and total; the file
keeps one entry per job with the elapsed time and a projected finish, which `run_weekend.py`
prints and compares with the Monday lock margin.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from typing import Any

from portstudy.io import read_json, write_json_atomic
from portstudy.logging_setup import now_et
from portstudy.paths import week_output_dir

_LOCK = threading.Lock()
_STARTED: dict[tuple[str, str], float] = {}
_LAST_WRITE: dict[tuple[str, str], float] = {}
MIN_INTERVAL_S = 2.0


def progress_path(week_id: str) -> Any:
    return week_output_dir(week_id) / "progress.json"


def read_progress(week_id: str) -> dict[str, dict[str, Any]]:
    path = progress_path(week_id)
    return dict(read_json(path)) if path.exists() else {}


def start(week_id: str, job: str) -> None:
    with _LOCK:
        _STARTED[(week_id, job)] = time.monotonic()
        _LAST_WRITE.pop((week_id, job), None)


def update(week_id: str, job: str, done: int, total: int, force: bool = False) -> None:
    """Record progress; throttled to one write every MIN_INTERVAL_S unless done or forced."""
    key = (week_id, job)
    with _LOCK:
        now = time.monotonic()
        started = _STARTED.setdefault(key, now)
        finished = total > 0 and done >= total
        if not (force or finished) and now - _LAST_WRITE.get(key, 0.0) < MIN_INTERVAL_S:
            return
        _LAST_WRITE[key] = now
        elapsed = now - started
        projected: str | None = None
        if done > 0 and total > 0:
            remaining = elapsed / done * (total - done)
            projected = (now_et() + dt.timedelta(seconds=remaining)).isoformat()
        data = read_progress(week_id)
        data[job] = {
            "done": int(done),
            "total": int(total),
            "elapsed_s": round(elapsed, 3),
            "projected_finish": projected,
        }
        write_json_atomic(data, progress_path(week_id))
