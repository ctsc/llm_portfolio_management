"""Phase 16 checklist tests: UNVERIFIED markers are logged, DECISION tags are mirrored,
and the git history has one commit per phase."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _py_files() -> list[Path]:
    out = []
    for base in ("src", "scripts"):
        out.extend(p for p in (REPO_ROOT / base).rglob("*.py") if "__pycache__" not in p.parts)
    return out


def test_every_unverified_marker_is_listed_in_progress() -> None:
    progress = (REPO_ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    files_with_marker = {p for p in _py_files() if "UNVERIFIED" in p.read_text(encoding="utf-8")}
    assert files_with_marker, "expected UNVERIFIED markers on the real-service adapters"
    missing = [p.name for p in files_with_marker if p.name not in progress]
    assert missing == [], f"UNVERIFIED files not listed in PROGRESS.md: {missing}"
    # models.yaml placeholders are listed too.
    assert "config/models.yaml" in progress


def test_every_decision_tag_appears_in_open_questions() -> None:
    doc = (REPO_ROOT / "docs" / "open_questions.md").read_text(encoding="utf-8")
    ids: set[str] = set()
    for p in _py_files() + [REPO_ROOT / "config" / "study.yaml"]:
        ids.update(re.findall(r"DECISION\(([A-Z][0-9]+)\)", p.read_text(encoding="utf-8")))
    assert ids
    missing = sorted(i for i in ids if f"| {i} |" not in doc)
    assert missing == [], missing
    # Every decision in study.yaml is mirrored as well.
    study = (REPO_ROOT / "config" / "study.yaml").read_text(encoding="utf-8")
    for did in re.findall(r"- id: ([A-Z][0-9]+)", study):
        assert f"| {did} |" in doc, did


def test_git_history_has_one_commit_per_phase() -> None:
    log = subprocess.run(["git", "log", "--format=%s"], cwd=REPO_ROOT, capture_output=True, text=True)
    if log.returncode != 0:
        return  # not a git checkout (for example an extracted archive)
    subjects = [s for s in log.stdout.splitlines() if s.startswith("phase ")]
    phases = [int(re.match(r"phase (\d+):", s).group(1)) for s in subjects]  # type: ignore[union-attr]
    assert len(phases) == len(set(phases)), subjects
    assert set(phases) >= set(range(1, 16))
