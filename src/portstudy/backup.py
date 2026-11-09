"""Dated CSV backups Carter downloads each week (DECISION(N3)).

`run_backup(week_id)` writes `data/backups/<week_id>_<timestamp>/` holding the full price
ledger, the week's snapshot CSVs, the week's output CSVs, logs, prompt files and manifest,
portfolio and trade CSVs, configs, and run metadata (git commit, package versions), plus a
`checksums.json` and one zip of the whole folder. It never overwrites an earlier backup. It
also rebuilds `data/exports/master/`: one running CSV per table across all weeks.
`restore_backup` and `verify_backup` complete the round trip.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import zipfile
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from portstudy.config import StudyConfig, repo_root
from portstudy.hashing import sha256_file
from portstudy.io import read_csv, read_json, write_csv_atomic, write_json_atomic
from portstudy.logging_setup import now_et
from portstudy.paths import backups_dir, data_dir, exports_dir, outputs_dir, portfolios_dir

MASTER_TABLES = ["llm_scores", "trades", "daily_values", "weekly_ic", "fallback_log", "cost_log"]
PACKAGES = ["pandas", "numpy", "scipy", "scikit-learn", "statsmodels", "cvxpy", "pydantic", "typer"]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def run_metadata(week_id: str) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "week_id": week_id,
        "created_at": now_et().isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "packages": versions,
    }


def backup_sources(week_id: str) -> list[tuple[Path, Path]]:
    """(source, destination relative to the backup folder) pairs; missing sources are skipped."""
    root = data_dir()
    repo = repo_root()
    pairs = [
        (root / "ledger", Path("data/ledger")),
        (root / "snapshots" / week_id, Path("data/snapshots") / week_id),
        (root / "outputs" / week_id, Path("data/outputs") / week_id),
        (root / "outputs" / "ic_history.csv", Path("data/outputs/ic_history.csv")),
        (root / "portfolios", Path("data/portfolios")),
        (root / "logs", Path("data/logs")),
        (repo / "prompts", Path("repo/prompts")),
        (repo / "config", Path("repo/config")),
    ]
    return pairs


def _copy(
    src: Path, dst: Path, log: logging.Logger, dry_run: bool, checksums: dict[str, str], base: Path
) -> None:
    if src.is_dir():
        files = sorted(p for p in src.rglob("*") if p.is_file())
    else:
        files = [src]
    for f in files:
        rel = dst / f.relative_to(src) if src.is_dir() else dst
        if dry_run:
            log.info("backup_would_copy", extra={"extra_fields": {"file": str(rel)}})
            print(f"  would copy {rel}")
            continue
        target = base / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        checksums[rel.as_posix()] = sha256_file(target)
        log.info("backup_copied", extra={"extra_fields": {"file": str(rel)}})


def run_backup(
    week_id: str, cfg: StudyConfig, dry_run: bool = False, logger: logging.Logger | None = None
) -> Path:
    log = logger or logging.getLogger("portstudy.backup")
    stamp = now_et().strftime("%Y%m%dT%H%M%S")
    target = backups_dir() / f"{week_id}_{stamp}"
    while target.exists():  # never overwrite an earlier backup
        stamp += "x"
        target = backups_dir() / f"{week_id}_{stamp}"
    checksums: dict[str, str] = {}
    for src, rel in backup_sources(week_id):
        if src.exists():
            _copy(src, rel, log, dry_run, checksums, target)
    if not dry_run:
        rebuild_master_exports(cfg)
        _copy(exports_dir() / "master", Path("data/exports/master"), log, False, checksums, target)
        write_json_atomic(run_metadata(week_id), target / "run_metadata.json")
        checksums["run_metadata.json"] = sha256_file(target / "run_metadata.json")
        write_json_atomic(checksums, target / "checksums.json")
        zip_path = target.with_suffix(".zip") if cfg.backup.keep_zips else None
        if zip_path is not None:
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for f in sorted(target.rglob("*")):
                    if f.is_file():
                        zf.write(f, f.relative_to(target).as_posix())
        print(f"backup folder: {target}")
        if zip_path is not None:
            print(f"backup zip:    {zip_path}")
        if cfg.backup.rsync_target:
            rsync_backup(target, zip_path, cfg.backup.rsync_target, log)
        log.info(
            "backup_written",
            extra={
                "extra_fields": {
                    "path": str(target),
                    "n_files": len(checksums),
                    "zip": str(zip_path) if zip_path else None,
                }
            },
        )
    else:
        print(f"dry run: backup folder would be {target}")
    return target


def rsync_command(target: Path, zip_path: Path | None, destination: str) -> list[str]:
    sources = [f"{target}/"] if zip_path is None else [f"{target}/", str(zip_path)]
    return ["rsync", "-a", "--partial", *sources, destination.rstrip("/") + f"/{target.name}/"]


def rsync_backup(
    target: Path, zip_path: Path | None, destination: str, log: logging.Logger
) -> bool:
    """Copy the dated folder (and zip) to `backup.rsync_target`; a failure is reported, not
    raised, because the local backup is already complete."""
    cmd = rsync_command(target, zip_path, destination)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        print(f"backup rsync FAILED ({destination}): {detail.strip()}")
        log.warning("backup_rsync_failed", extra={"extra_fields": {"error": detail.strip()}})
        return False
    print(f"backup rsync:  {destination}")
    return True


def verify_backup(path: Path) -> list[str]:
    """Recompute every hash in checksums.json. Returns a list of problems (empty means ok)."""
    manifest = path / "checksums.json"
    if not manifest.exists():
        return ["checksums.json missing"]
    expected: dict[str, str] = read_json(manifest)
    problems: list[str] = []
    for rel, digest in expected.items():
        f = path / rel
        if not f.exists():
            problems.append(f"missing: {rel}")
        elif sha256_file(f) != digest:
            problems.append(f"hash mismatch: {rel}")
    for f in path.rglob("*"):
        if f.is_file():
            rel = f.relative_to(path).as_posix()
            if rel != "checksums.json" and rel not in expected:
                problems.append(f"unexpected file: {rel}")
    return problems


def restore_backup(
    path: Path, target_data_dir: Path, target_repo_dir: Path | None = None, verify: bool = True
) -> list[str]:
    """Copy the backed-up data folder (and optionally prompts and config) into `target_data_dir`."""
    if verify:
        problems = verify_backup(path)
        if problems:
            raise ValueError("backup failed verification: " + "; ".join(problems[:5]))
    restored: list[str] = []
    src_data = path / "data"
    for f in sorted(p for p in src_data.rglob("*") if p.is_file()):
        rel = f.relative_to(src_data)
        dst = target_data_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        restored.append(rel.as_posix())
    if target_repo_dir is not None and (path / "repo").exists():
        for f in sorted(p for p in (path / "repo").rglob("*") if p.is_file()):
            rel = f.relative_to(path / "repo")
            dst = target_repo_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            restored.append(("repo" / rel).as_posix())
    return restored


# Master exports ---------------------------------------------------------------------------------


def _concat(frames: list[pd.DataFrame], columns: list[str] | None = None) -> pd.DataFrame:
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=columns or [])
    return pd.concat(frames, ignore_index=True)


def master_tables() -> dict[str, pd.DataFrame]:
    out_root = outputs_dir()
    weeks = sorted(p.name for p in out_root.iterdir() if p.is_dir() and p.name.count("-W") == 1)
    scores, costs, fallbacks = [], [], []
    for w in weeks:
        s = out_root / w / "llm_scores.csv"
        if s.exists():
            df = read_csv(s, keep_default_na=False, na_values=[""])
            df.insert(0, "week_id", w)
            scores.append(df)
        c = out_root / w / "cost_log.csv"
        if c.exists():
            costs.append(read_csv(c))
        f = out_root / w / "fallback_log.json"
        if f.exists():
            for e in read_json(f).get("entries", []):
                fallbacks.append({"week_id": w, **e})
    trades, daily = [], []
    if portfolios_dir().exists():
        for folder in sorted(p for p in portfolios_dir().iterdir() if p.is_dir()):
            t, d = folder / "trades.csv", folder / "daily.csv"
            if t.exists():
                trades.append(read_csv(t, keep_default_na=False, na_values=[""]))
            if d.exists():
                daily.append(read_csv(d, keep_default_na=False, na_values=[""]))
    ic_path = out_root / "ic_history.csv"
    weekly_ic = (
        read_csv(ic_path, keep_default_na=False, na_values=[""])
        if ic_path.exists()
        else pd.DataFrame()
    )
    return {
        "llm_scores": _concat(scores),
        "trades": _concat(trades),
        "daily_values": _concat(daily),
        "weekly_ic": weekly_ic,
        "fallback_log": pd.DataFrame(
            fallbacks, columns=["week_id", "portfolio", "status", "reason", "decision_source"]
        )
        if fallbacks
        else pd.DataFrame(columns=["week_id", "portfolio", "status", "reason", "decision_source"]),
        "cost_log": _concat(costs),
    }


def rebuild_master_exports(cfg: StudyConfig) -> dict[str, Path]:
    """Rebuild data/exports/master/<table>_all.csv from the weekly files (always consistent)."""
    folder = exports_dir() / "master"
    folder.mkdir(parents=True, exist_ok=True)
    tables = master_tables()
    written: dict[str, Path] = {}
    for name in cfg.backup.master_tables:
        if name not in tables:
            continue
        written[name] = write_csv_atomic(tables[name], folder / f"{name}_all.csv")
    return written


def latest_backup(week_id: str | None = None) -> Path | None:
    folders = sorted(
        p
        for p in backups_dir().iterdir()
        if p.is_dir() and (week_id is None or p.name.startswith(f"{week_id}_"))
    )
    return folders[-1] if folders else None


def backup_summary(path: Path) -> dict[str, Any]:
    meta = read_json(path / "run_metadata.json") if (path / "run_metadata.json").exists() else {}
    checks = read_json(path / "checksums.json") if (path / "checksums.json").exists() else {}
    return {
        "path": str(path),
        "n_files": len(checks),
        **{k: meta.get(k) for k in ("week_id", "git_commit")},
    }


def dump_summary(path: Path) -> str:
    return json.dumps(backup_summary(path), indent=2, default=str)
