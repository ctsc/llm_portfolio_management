"""Hash helpers: file, text, and canonical-JSON object hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(obj: Any) -> Any:
    # Convert common non-JSON types to stable JSON-compatible forms.
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, list | tuple):
        return [_canonical(v) for v in obj]
    if isinstance(obj, set | frozenset):
        return sorted(_canonical(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item") and not isinstance(obj, str | bytes):
        return obj.item()
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(_canonical(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(obj: Any) -> str:
    return sha256_text(canonical_json(obj))
