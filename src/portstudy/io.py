"""Atomic CSV and JSON readers and writers.

All CSV output is UTF-8 with a header row, ISO dates, and floats written with
float_format="%.12g" so that reruns are byte-identical.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd

FLOAT_FORMAT = "%.12g"


def _fsync_dir(folder: Path) -> None:
    """Flush a folder entry to disk after a rename (POSIX only; Windows has no folder fsync)."""
    if os.name == "nt":
        return
    fd = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_replace(tmp_path: Path, final_path: Path) -> None:
    # N6 (6): the temp file reaches the disk before the rename, and the folder after it.
    with tmp_path.open("r+b") as fh:  # Windows refuses fsync on a read-only handle
        os.fsync(fh.fileno())
    # os.replace is atomic on POSIX and on Windows for same-volume paths. Windows refuses the
    # replace while another thread or process has the target open, so retry briefly there.
    for attempt in range(20):
        try:
            os.replace(tmp_path, final_path)
            break
        except PermissionError:
            if os.name != "nt" or attempt == 19:
                raise
            time.sleep(0.05)
    _fsync_dir(final_path.parent)


def append_line_durable(path: Path, line: str) -> None:
    """Append one line to a JSON-lines file, then flush and fsync it (N6 (6))."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = line if line.endswith("\n") else line + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())


def _tmp_for(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    return Path(name)


def write_csv_atomic(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """Write a DataFrame to CSV via a temp file and rename. Returns the final path."""
    path = Path(path)
    tmp = _tmp_for(path)
    try:
        df.to_csv(
            tmp, index=index, float_format=FLOAT_FORMAT, lineterminator="\n", encoding="utf-8"
        )
        _atomic_replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(Path(path), encoding="utf-8", **kwargs)
    return df


def write_json_atomic(obj: Any, path: Path) -> Path:
    path = Path(path)
    tmp = _tmp_for(path)
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True, default=_json_default)
            fh.write("\n")
        _atomic_replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_text_atomic(text: str, path: Path) -> Path:
    path = Path(path)
    tmp = _tmp_for(path)
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        _atomic_replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")
