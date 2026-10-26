"""Prompt manifest and assembly.

Order is fixed: shared prefix first (base instructions plus the configuration's file), then
the stock part (ticker, company name, sector, news; DECISION(C1)). The shared prefix is
byte-identical across stocks so provider prompt caching works. No price statistics appear
anywhere in the prompt.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from portstudy.config import StudyConfig, repo_root
from portstudy.hashing import sha256_file, sha256_text

PromptKind = Literal["bare", "skills", "placebo"]

BASE_INSTRUCTIONS_FILE = "prompts/base_instructions.md"


class PromptConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    kind: PromptKind
    file: str | None = None
    version: str = "v0"
    sha256: str | None = None
    hashed_version: str | None = None  # the version the recorded sha256 belongs to
    enabled: bool = True
    matches: str | None = None  # for placebo: the skills configuration it is length-matched to


class PromptManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_instructions: str = BASE_INSTRUCTIONS_FILE
    base_sha256: str | None = None
    configurations: list[PromptConfiguration] = Field(default_factory=list)

    def get(self, config_id: str) -> PromptConfiguration:
        for c in self.configurations:
            if c.id == config_id:
                return c
        raise KeyError(f"unknown prompt configuration {config_id}")

    def enabled(self) -> list[PromptConfiguration]:
        return [c for c in self.configurations if c.enabled]


def manifest_path() -> Path:
    return repo_root() / "config" / "prompts_manifest.yaml"


def load_manifest(path: Path | None = None) -> PromptManifest:
    with (path or manifest_path()).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return PromptManifest.model_validate(raw)


@lru_cache(maxsize=256)
def _read_cached(path: str, mtime_ns: int, size: int) -> tuple[str, str]:
    text = Path(path).read_text(encoding="utf-8")
    return text, sha256_file(Path(path))


def _text_and_hash(rel: str) -> tuple[str, str]:
    """Prompt file text and hash, cached on (path, mtime, size) so edits are always seen."""
    path = repo_root() / rel
    st = path.stat()
    return _read_cached(str(path), st.st_mtime_ns, st.st_size)


def read_prompt_file(rel: str) -> str:
    return _text_and_hash(rel)[0]


def file_hash(rel: str) -> str:
    return _text_and_hash(rel)[1]


# Assembly -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AssembledPrompt:
    config_id: str
    shared_prefix: str
    stock_part: str
    full_text: str
    prefix_hash: str
    prompt_hash: str
    prompt_file_sha256: str | None


def shared_prefix(config: PromptConfiguration, manifest: PromptManifest) -> str:
    base = read_prompt_file(manifest.base_instructions).strip()
    if config.kind == "bare" or config.file is None:
        return base + "\n"
    extra = read_prompt_file(config.file).strip()
    return base + "\n\n" + extra + "\n"


def stock_part(stock_row: Mapping[str, Any], news_text: str, cfg: StudyConfig) -> str:
    """DECISION(C1) fields; DECISION(C2) no-news sentence when the window held no news."""
    news = news_text.strip() if isinstance(news_text, str) else ""
    if not news:
        news = cfg.llm.no_news_sentence
    lines = [
        "Stock to rate:",
        f"Ticker: {stock_row['ticker']}",
        f"Company: {stock_row['name']}",
        f"Sector: {stock_row['sector']}",
        "",
        "News from the past week:",
        news,
        "",
        "Respond with the JSON object now.",
    ]
    return "\n".join(lines) + "\n"


def assemble_prompt(
    config_id: str,
    stock_row: Mapping[str, Any],
    news_text: str,
    cfg: StudyConfig,
    manifest: PromptManifest | None = None,
) -> AssembledPrompt:
    man = manifest or load_manifest()
    config = man.get(config_id)
    prefix = shared_prefix(config, man)
    stock = stock_part(stock_row, news_text, cfg)
    full = prefix + "\n" + stock
    return AssembledPrompt(
        config_id=config_id,
        shared_prefix=prefix,
        stock_part=stock,
        full_text=full,
        prefix_hash=sha256_text(prefix),
        prompt_hash=sha256_text(full),
        prompt_file_sha256=file_hash(config.file) if config.file else None,
    )


# Token counting -------------------------------------------------------------------------------

TokenCounter = Callable[[str], int]
_COUNTERS: dict[str, TokenCounter] = {}


def register_token_counter(model_id: str, counter: TokenCounter) -> None:
    _COUNTERS[model_id] = counter


def clear_token_counters() -> None:
    _COUNTERS.clear()


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _tiktoken_counter(model_id: str) -> TokenCounter | None:
    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        enc = tiktoken.encoding_for_model(model_id)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    return lambda text: len(enc.encode(text))


def token_count(text: str, model_id: str) -> int:
    """Registered counter, then tiktoken for OpenAI-like ids, then the len/4 fallback."""
    counter = _COUNTERS.get(model_id)
    if counter is None and model_id.startswith(("gpt", "o1", "o3", "o4")):
        counter = _tiktoken_counter(model_id)
    if counter is None:
        counter = approx_tokens
    return counter(text)
