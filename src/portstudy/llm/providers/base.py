"""LLM provider interface shared by the stub, Ollama, and the two batch API adapters."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Literal

from portstudy.config import LlmSettings
from portstudy.llm.prompts import AssembledPrompt

BatchState = Literal["pending", "running", "done", "failed", "expired"]


@dataclass(frozen=True)
class BatchItem:
    """One stock-level call. `custom_id` is stable across runs: model__config__week__ticker."""

    custom_id: str
    model: str
    config_id: str
    week_id: str
    ticker: str
    prompt: AssembledPrompt


@dataclass
class RawResult:
    custom_id: str
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0
    latency_s: float = 0.0
    model_version: str = ""
    error: str | None = None
    stop_reason: str | None = None  # "max_tokens" marks a truncated answer
    from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "custom_id": self.custom_id,
            "text": self.text,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_cached": self.tokens_cached,
            "latency_s": self.latency_s,
            "model_version": self.model_version,
            "error": self.error,
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RawResult:
        return cls(
            custom_id=str(d["custom_id"]),
            text=str(d.get("text", "")),
            tokens_in=int(d.get("tokens_in", 0)),
            tokens_out=int(d.get("tokens_out", 0)),
            tokens_cached=int(d.get("tokens_cached", 0)),
            latency_s=float(d.get("latency_s", 0.0)),
            model_version=str(d.get("model_version", "")),
            error=d.get("error"),
            stop_reason=d.get("stop_reason"),
        )


@dataclass
class BatchHandle:
    provider: str
    batch_id: str
    model: str
    config_id: str
    week_id: str
    custom_ids: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "batch_id": self.batch_id,
            "model": self.model,
            "config_id": self.config_id,
            "week_id": self.week_id,
            "custom_ids": list(self.custom_ids),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BatchHandle:
        return cls(
            provider=str(d["provider"]),
            batch_id=str(d["batch_id"]),
            model=str(d["model"]),
            config_id=str(d["config_id"]),
            week_id=str(d["week_id"]),
            custom_ids=[str(x) for x in d.get("custom_ids", [])],
            extra=dict(d.get("extra", {})),
        )


@dataclass
class BatchStatus:
    state: BatchState
    n_total: int = 0
    n_done: int = 0
    n_failed: int = 0
    message: str = ""


class LlmProviderError(RuntimeError):
    """Raised for transport failures, timeouts, and provider-side errors."""


class LlmProvider(abc.ABC):
    """One stock per call. Providers never write to prompts/ and never enable tools."""

    model_id: str
    supports_batch: bool = False

    @abc.abstractmethod
    def complete(self, item: BatchItem, settings: LlmSettings) -> RawResult:
        """Synchronous completion for one stock."""

    def submit_batch(self, items: list[BatchItem], settings: LlmSettings) -> BatchHandle:
        raise NotImplementedError(f"{self.model_id} does not support batches")

    def poll_batch(self, handle: BatchHandle) -> BatchStatus:
        raise NotImplementedError(f"{self.model_id} does not support batches")

    def collect_batch(self, handle: BatchHandle) -> list[RawResult]:
        raise NotImplementedError(f"{self.model_id} does not support batches")


def make_custom_id(model: str, config_id: str, week_id: str, ticker: str) -> str:
    return f"{model}__{config_id}__{week_id}__{ticker}"


def split_custom_id(custom_id: str) -> tuple[str, str, str, str]:
    model, config_id, week_id, ticker = custom_id.split("__", 3)
    return model, config_id, week_id, ticker
