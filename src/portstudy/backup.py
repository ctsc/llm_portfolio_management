"""Dated CSV backups (DECISION(N3)). Completed in Phase 13; J9 calls `run_backup`."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from portstudy.config import StudyConfig
from portstudy.hashing import sha256_file
from portstudy.io import write_json_atomic
from portstudy.logging_setup import now_et
from portstudy.paths import backups_dir, data_dir


def backup_sources(week_id: str) -> list[Path]:
    root = data_dir()
    return [
        root / "ledger",
        root / "snapshots" / week_id,
        root / "outputs" / week_id,
        root / "outputs" / "ic_history.csv",
        root / "portfolios",
        root / "logs",
    ]


def run_backup(
    week_id: str, cfg: StudyConfig, dry_run: bool = False, logger: logging.Logger | None = None
) -> Path:
    log = logger or logging.getLogger("portstudy.backup")
    stamp = now_et().strftime("%Y%m%dT%H%M%S")
    target = backups_dir() / f"{week_id}_{stamp}"
    checksums: dict[str, str] = {}
    root = data_dir()
    for src in backup_sources(week_id):
        if not src.exists():
            continue
        rel = src.relative_to(root)
        dst = target / rel
        if dry_run:
            log.info("backup_dry_run", extra={"extra_fields": {"path": str(rel)}})
            continue
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            for f in dst.rglob("*"):
                if f.is_file():
                    checksums[str(f.relative_to(target))] = sha256_file(f)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            checksums[str(dst.relative_to(target))] = sha256_file(dst)
    if not dry_run:
        write_json_atomic(checksums, target / "checksums.json")
        log.info(
            "backup_written",
            extra={"extra_fields": {"path": str(target), "n_files": len(checksums)}},
        )
    return target
