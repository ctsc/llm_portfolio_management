"""Data folder locations, rooted at PORTSTUDY_DATA_DIR (default ./data under the repo root)."""

from __future__ import annotations

import os
from pathlib import Path

from portstudy.config import repo_root


def data_dir() -> Path:
    env = os.environ.get("PORTSTUDY_DATA_DIR")
    return Path(env).resolve() if env else repo_root() / "data"


def _sub(name: str) -> Path:
    path = data_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def ledger_dir() -> Path:
    return _sub("ledger")


def snapshots_dir() -> Path:
    return _sub("snapshots")


def outputs_dir() -> Path:
    return _sub("outputs")


def logs_dir() -> Path:
    return _sub("logs")


def portfolios_dir() -> Path:
    return _sub("portfolios")


def backups_dir() -> Path:
    return _sub("backups")


def exports_dir() -> Path:
    return _sub("exports")


def cache_dir() -> Path:
    return _sub("cache")


def synthetic_dir() -> Path:
    return _sub("synthetic")


def week_output_dir(week_id: str) -> Path:
    path = outputs_dir() / week_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def snapshot_dir(week_id: str, version: int = 1) -> Path:
    name = week_id if version == 1 else f"{week_id}.v{version}"
    return snapshots_dir() / name


def report_dir() -> Path:
    path = outputs_dir() / "report"
    path.mkdir(parents=True, exist_ok=True)
    return path


def prompts_dir() -> Path:
    return repo_root() / "prompts"


def config_dir() -> Path:
    return repo_root() / "config"
