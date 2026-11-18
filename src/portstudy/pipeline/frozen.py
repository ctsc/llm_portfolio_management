"""Frozen-config check (PLAN4 Phase 3).

Chosen over narrowing the job input hash: at the pilot week the live run saves a copy of
study.yaml, models.yaml and prompts_manifest.yaml under `data/frozen/`; the preflight fails
when a current file differs from the frozen copy, until `accept_config_change` records the
intentional change with a note.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from portstudy.hashing import sha256_file
from portstudy.io import read_json, write_json_atomic
from portstudy.logging_setup import now_et
from portstudy.paths import data_dir

FROZEN_FILES = [
    "config/study.yaml",
    "config/models.yaml",
    "config/prompts_manifest.yaml",
    "config/preregistration.yaml",
]


def frozen_dir() -> Path:
    return data_dir() / "frozen"


def _hashes(root: Path) -> dict[str, str]:
    return {Path(rel).name: sha256_file(root / rel) for rel in FROZEN_FILES}


def _write(root: Path, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for rel in FROZEN_FILES:
        shutil.copy2(root / rel, folder / Path(rel).name)
    write_json_atomic(
        {"saved_at": now_et().isoformat(), "files": _hashes(root)}, folder / "frozen.json"
    )


def save_frozen_config(root: Path) -> Path:
    """Copy the frozen files and their hashes; a no-op once the frozen copy exists."""
    folder = frozen_dir()
    if (folder / "frozen.json").exists():
        return folder
    _write(root, folder)
    return folder


def frozen_config_diff(root: Path) -> list[str]:
    """Frozen file names whose current contents differ; empty without a frozen copy."""
    meta_path = frozen_dir() / "frozen.json"
    if not meta_path.exists():
        return []
    saved = read_json(meta_path).get("files", {})
    return sorted(
        Path(name).name
        for name in FROZEN_FILES
        if saved.get(Path(name).name) != sha256_file(root / name)
    )


def accept_config_change(root: Path, note: str) -> None:
    """Record an intentional config change: refresh the frozen copy and append to
    changes.jsonl. `note` is required so the change is never anonymous."""
    if not note or not note.strip():
        raise ValueError("accept_config_change needs a non-empty note")
    changed = frozen_config_diff(root)
    _write(root, frozen_dir())
    line = {"accepted_at": now_et().isoformat(), "note": note, "files": changed}
    with (frozen_dir() / "changes.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, sort_keys=True) + "\n")
