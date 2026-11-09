"""Phase 13 tests: backup, verify, restore round trip; corruption detection; master exports."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from portstudy import calendar as cal
from portstudy.backup import (
    master_tables,
    rebuild_master_exports,
    restore_backup,
    run_backup,
    verify_backup,
)
from portstudy.config import StudyConfig
from portstudy.hashing import sha256_file
from portstudy.io import read_csv, read_json
from tests.conftest import PIPELINE_WEEKS


def _hashes(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_backup_restore_round_trip(
    pipeline_dir: Path,
    study_config: StudyConfig,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    week = cal.week_range(study_config.synthetic.first_week, PIPELINE_WEEKS)[-1]
    before = {
        k: v
        for k, v in _hashes(pipeline_dir).items()
        if k.startswith(("ledger/", f"snapshots/{week}/", f"outputs/{week}/", "portfolios/"))
        or k == "outputs/ic_history.csv"
    }
    target = run_backup(week, study_config)
    out = capsys.readouterr().out
    assert "backup folder:" in out and "backup zip:" in out
    assert target.exists() and target.with_suffix(".zip").exists()
    assert verify_backup(target) == []
    checks = read_json(target / "checksums.json")
    assert any(k.startswith("repo/prompts/") for k in checks) and "repo/config/study.yaml" in checks
    meta = read_json(target / "run_metadata.json")
    assert meta["week_id"] == week and "pandas" in meta["packages"]
    with zipfile.ZipFile(target.with_suffix(".zip")) as zf:
        assert "checksums.json" in zf.namelist() and "data/ledger/prices_raw.csv" in zf.namelist()
    # A second backup for the same week never overwrites the first.
    again = run_backup(week, study_config)
    assert again != target and again.exists() and target.exists()
    # Delete the originals, restore, and compare hashes.
    for sub in ("ledger", "snapshots", "outputs", "portfolios"):
        shutil.rmtree(pipeline_dir / sub)
    restored_root = tmp_path / "restored"
    restored = restore_backup(target, restored_root)
    assert restored
    after = {k: v for k, v in _hashes(restored_root).items() if k in before}
    assert after == before


def test_verify_detects_corruption_and_missing_files(
    pipeline_dir: Path, study_config: StudyConfig
) -> None:
    week = cal.week_range(study_config.synthetic.first_week, PIPELINE_WEEKS)[-1]
    target = run_backup(week, study_config)
    ledger = target / "data" / "ledger" / "prices_raw.csv"
    text = ledger.read_text(encoding="utf-8")
    ledger.write_text(text.replace("SYN000", "SYN999", 1), encoding="utf-8")
    problems = verify_backup(target)
    assert problems == ["hash mismatch: data/ledger/prices_raw.csv"]
    (target / "data" / "outputs" / week / "baseline.csv").unlink()
    (target / "stray.txt").write_text("x", encoding="utf-8")
    problems = verify_backup(target)
    assert f"missing: data/outputs/{week}/baseline.csv" in problems
    assert "unexpected file: stray.txt" in problems
    with pytest.raises(ValueError):
        restore_backup(target, pipeline_dir / "x")


def test_master_exports_equal_concatenation_of_weekly_files(
    pipeline_dir: Path, study_config: StudyConfig
) -> None:
    written = rebuild_master_exports(study_config)
    assert set(written) == set(study_config.backup.master_tables)
    weeks = cal.week_range(study_config.synthetic.first_week, PIPELINE_WEEKS)
    scores = pd.concat(
        [
            read_csv(pipeline_dir / "outputs" / w / "llm_scores.csv").assign(week_id=w)
            for w in weeks
        ],
        ignore_index=True,
    )
    master = read_csv(written["llm_scores"])
    assert len(master) == len(scores)
    assert master["week_id"].value_counts().to_dict() == scores["week_id"].value_counts().to_dict()
    trades = pd.concat(
        [read_csv(p) for p in sorted((pipeline_dir / "portfolios").glob("*/trades.csv"))],
        ignore_index=True,
    )
    assert len(read_csv(written["trades"])) == len(trades)
    daily = pd.concat(
        [read_csv(p) for p in sorted((pipeline_dir / "portfolios").glob("*/daily.csv"))],
        ignore_index=True,
    )
    assert len(read_csv(written["daily_values"])) == len(daily)
    ic = read_csv(pipeline_dir / "outputs" / "ic_history.csv")
    assert len(read_csv(written["weekly_ic"])) == len(ic)
    costs = pd.concat(
        [read_csv(pipeline_dir / "outputs" / w / "cost_log.csv") for w in weeks], ignore_index=True
    )
    assert len(read_csv(written["cost_log"])) == len(costs)
    tables = master_tables()
    assert list(tables["fallback_log"].columns) == [
        "week_id",
        "portfolio",
        "status",
        "reason",
        "decision_source",
    ]


def test_dry_run_writes_nothing(
    pipeline_dir: Path, study_config: StudyConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    week = cal.week_range(study_config.synthetic.first_week, PIPELINE_WEEKS)[-1]
    before = (
        sorted(p.name for p in (pipeline_dir / "backups").iterdir())
        if (pipeline_dir / "backups").exists()
        else []
    )
    target = run_backup(week, study_config, dry_run=True)
    out = capsys.readouterr().out
    assert "would copy" in out and "dry run" in out
    assert not target.exists()
    after = sorted(p.name for p in (pipeline_dir / "backups").iterdir())
    assert after == before
