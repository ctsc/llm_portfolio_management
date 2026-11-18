"""JSON-lines logging to data/logs/<job>/<date>.jsonl, with timestamps in US Eastern."""

from __future__ import annotations

import json
import logging
import re
import secrets
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from portstudy.io import append_line_durable
from portstudy.paths import logs_dir

ET = ZoneInfo("America/New_York")


def now_et() -> datetime:
    return datetime.now(tz=ET)


# N6 (1): one run_id per run_weekend.py invocation (or standalone `portstudy run-job`), attached
# to every log line, status file, LLM record, weight meta and study-log row written in the run.
_RUN_ID = ""


def new_run_id() -> str:
    """Timestamp plus a short random suffix, for example 20261120T221502-3f9a1c."""
    return now_et().strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(3)


def set_run_id(run_id: str) -> None:
    global _RUN_ID
    _RUN_ID = run_id


def current_run_id() -> str:
    return _RUN_ID


_SECRET_KEYS = ("authorization", "api_key", "apikey", "api-key", "token", "secret", "password")
_SECRET_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)(authorization\s*[:=]\s*)\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
)
REDACTED = "[REDACTED]"


def redact(value: Any) -> Any:
    """Review fix M9: drop credentials from anything that reaches the log files. Keys that name
    a secret lose their value; strings lose bearer tokens, Authorization headers and sk- keys."""
    if isinstance(value, dict):
        return {
            k: (REDACTED if str(k).lower() in _SECRET_KEYS else redact(v)) for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact(v) for v in value]
    if isinstance(value, str):
        for pattern in _SECRET_PATTERNS:
            value = pattern.sub(lambda m: (m.group(1) if m.lastindex else "") + REDACTED, value)
        return value
    return value


class JsonLinesHandler(logging.Handler):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=ET).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info and "traceback" not in payload:
            # N6 (3): a caught exception logged with exc_info keeps its full traceback.
            payload["traceback"] = logging.Formatter().formatException(record.exc_info)
        payload.setdefault("run_id", current_run_id())
        payload = redact(json.loads(json.dumps(payload, default=str)))
        append_line_durable(self.path, json.dumps(payload, sort_keys=True))


LIBRARY_LOGGERS = ("py.warnings", "cvxpy", "httpx", "urllib3")
_LIBRARIES_LOCK = threading.Lock()
_LIBRARIES_DAY = ""


def route_library_logs(day: str | None = None) -> None:
    """N6 (4): Python warnings and the cvxpy, httpx and urllib3 loggers (WARNING and above) go to
    data/logs/libraries/<date>.jsonl. Idempotent per day."""
    global _LIBRARIES_DAY
    day = day or now_et().date().isoformat()
    with _LIBRARIES_LOCK:
        path = logs_dir() / "libraries" / f"{day}.jsonl"
        if _LIBRARIES_DAY == str(path):
            return
        logging.captureWarnings(True)
        for name in LIBRARY_LOGGERS:
            lib = logging.getLogger(name)
            lib.handlers = [h for h in lib.handlers if not isinstance(h, JsonLinesHandler)]
            handler = JsonLinesHandler(path)
            handler.setLevel(logging.WARNING)
            lib.addHandler(handler)
            if lib.level == logging.NOTSET or lib.level > logging.WARNING:
                lib.setLevel(logging.WARNING)
        _LIBRARIES_DAY = str(path)


def get_logger(job: str, date: str | None = None) -> logging.Logger:
    """Return a logger named after the job that appends to data/logs/<job>/<date>.jsonl."""
    day = date or now_et().date().isoformat()
    route_library_logs(day)
    logger = logging.getLogger(f"portstudy.{job}")
    logger.setLevel(logging.INFO)
    path = logs_dir() / job / f"{day}.jsonl"
    # Avoid duplicate handlers when the logger is requested more than once.
    for handler in logger.handlers:
        if isinstance(handler, JsonLinesHandler) and handler.path == path:
            return logger
    logger.handlers = [h for h in logger.handlers if not isinstance(h, JsonLinesHandler)]
    logger.addHandler(JsonLinesHandler(path))
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, msg, extra={"extra_fields": fields})
