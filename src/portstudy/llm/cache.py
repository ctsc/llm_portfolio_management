"""On-disk cache of raw LLM results.

Key: sha256(model_version, prompt_hash, ticker, week_id, settings). A hit costs nothing and
is logged by the runner. Files live under data/cache/llm/<key[:2]>/<key>.json.
"""

from __future__ import annotations

from pathlib import Path

from portstudy.config import LlmSettings
from portstudy.hashing import stable_hash
from portstudy.io import read_json, write_json_atomic
from portstudy.llm.providers.base import RawResult
from portstudy.paths import cache_dir


def cache_key(
    model_version: str,
    prompt_hash: str,
    ticker: str,
    week_id: str,
    settings: LlmSettings,
    model_id: str = "",
) -> str:
    # The model id is included so two models that report the same version string never share
    # an entry.
    return stable_hash(
        {
            "model_id": model_id,
            "model_version": model_version,
            "prompt_hash": prompt_hash,
            "ticker": ticker,
            "week_id": week_id,
            "settings": settings.model_dump(),
        }
    )


class LlmCache:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (cache_dir() / "llm")

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> RawResult | None:
        path = self._path(key)
        if not path.exists():
            return None
        result = RawResult.from_dict(read_json(path))
        result.from_cache = True
        return result

    def put(self, key: str, result: RawResult) -> None:
        if result.error:
            return  # never cache a failed call
        write_json_atomic(result.to_dict(), self._path(key))

    def __len__(self) -> int:
        return sum(1 for _ in self.root.rglob("*.json")) if self.root.exists() else 0
