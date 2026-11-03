"""Deterministic stub LLM used for every test and the synthetic rehearsal.

The stub reads the planted sentiment words from the news text inside the prompt (like a real
model would) and returns a low, likely, and high rating. The answer is a pure function of
(ticker, week, configuration, seed) plus configurable noise per configuration. Failure
injection covers invalid JSON, truncated JSON, timeouts, a batch that never returns, and a crash.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field

import numpy as np

from portstudy.config import LlmSettings
from portstudy.data.providers.synthetic import NEGATIVE_WORDS, POSITIVE_WORDS
from portstudy.llm.providers.base import (
    BatchHandle,
    BatchItem,
    BatchStatus,
    LlmProvider,
    LlmProviderError,
    RawResult,
)
from portstudy.llm.schema import RANGE_MAX, RANGE_MIN

# Controls share noise, while skill prompt noise decreases across the configurations.
DEFAULT_NOISE = {
    "bare": 3.0,
    "placebo": 3.0,
    "skills_1": 1.2,
    "skills_2": 1.0,
    "skills_3": 0.8,
    "skills_4": 0.6,
    "skills_5": 0.4,
}
STUB_MODEL_VERSION = "stub-2026-01-01"


class StubCrash(RuntimeError):
    """Injected crash, raised from `complete` to simulate a process dying mid-run."""


@dataclass
class StubFailures:
    invalid_rate: float = 0.0
    truncated_rate: float = 0.0
    timeout_rate: float = 0.0
    batch_never_returns: bool = False
    crash_after_calls: int | None = None
    # Make a given ticker always invalid (used by the per-stock exclusion tests).
    always_invalid_tickers: set[str] = field(default_factory=set)


_WORD = re.compile(r"[a-z]+")


def sentiment_from_text(text: str) -> float:
    """Mean per-article net count of planted positive minus negative words in the news section.

    Each article is one line of the news block; averaging over lines keeps the signal in the
    planted -2..2 range regardless of how many articles a stock had that week.
    """
    marker = "News from the past week:"
    body = text.split(marker, 1)[1] if marker in text else text
    body = body.split("Respond with the JSON object now.", 1)[0]
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    nets = []
    for line in lines:
        words = _WORD.findall(line.lower())
        pos = sum(w in POSITIVE_WORDS for w in words)
        neg = sum(w in NEGATIVE_WORDS for w in words)
        nets.append(pos - neg)
    return float(sum(nets) / len(nets))


def _rng(seed: int, *parts: str) -> np.random.Generator:
    digest = hashlib.sha256(":".join([str(seed), *parts]).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


class StubLlmProvider(LlmProvider):
    supports_batch = True

    def __init__(
        self,
        model_id: str,
        seed: int = 0,
        noise_by_config: dict[str, float] | None = None,
        model_version: str | None = None,
        failures: StubFailures | None = None,
        model_skill: float = 1.0,
    ) -> None:
        self.model_id = model_id
        self.seed = seed
        self.noise_by_config = dict(DEFAULT_NOISE, **(noise_by_config or {}))
        # Default version matches the registry convention `<model_id>-2026-01-01`.
        self.model_version = model_version or f"{model_id}-{STUB_MODEL_VERSION.split('-', 1)[1]}"
        self.failures = failures or StubFailures()
        self.model_skill = model_skill
        self.calls = 0
        self._batches: dict[str, list[BatchItem]] = {}
        self._batch_settings: dict[str, LlmSettings] = {}

    # Scoring ------------------------------------------------------------------------------

    def answer_for(self, item: BatchItem) -> tuple[int, int, int, str]:
        """Deterministic (low, likely, high, rationale) for an item."""
        rng = _rng(self.seed, self.model_id, item.config_id, item.week_id, item.ticker)
        net = sentiment_from_text(item.prompt.full_text)
        noise_sd = self.noise_by_config.get(item.config_id, DEFAULT_NOISE["bare"])
        signal = self.model_skill * 1.5 * net
        noise = float(rng.normal(0.0, noise_sd))
        likely = int(np.clip(np.rint(signal + noise), RANGE_MIN, RANGE_MAX))
        half_width = max(1, int(np.ceil(abs(noise))))
        low = max(RANGE_MIN, likely - half_width)
        high = min(RANGE_MAX, likely + half_width)
        tone = "positive" if net > 0 else "negative" if net < 0 else "neutral"
        rationale = f"News tone is {tone} for {item.ticker} this week."
        return low, likely, high, rationale

    def _text_for(self, item: BatchItem, settings: LlmSettings) -> tuple[str, str | None]:
        rng = _rng(self.seed, "fail", self.model_id, item.config_id, item.week_id, item.ticker)
        low, likely, high, rationale = self.answer_for(item)
        payload = json.dumps({"low": low, "likely": likely, "high": high, "rationale": rationale})
        if item.ticker in self.failures.always_invalid_tickers:
            return "I cannot rate this stock.", None
        u = rng.random()
        if u < self.failures.invalid_rate:
            return "Sorry, here is my view: the stock looks fine.", None
        if u < self.failures.invalid_rate + self.failures.truncated_rate:
            return payload[: max(5, len(payload) // 2)], "max_tokens"
        return payload, "end_turn"

    # Interface ----------------------------------------------------------------------------

    def complete(self, item: BatchItem, settings: LlmSettings) -> RawResult:
        self.calls += 1
        if (
            self.failures.crash_after_calls is not None
            and self.calls > self.failures.crash_after_calls
        ):
            raise StubCrash(f"injected crash after {self.failures.crash_after_calls} calls")
        rng = _rng(self.seed, "timeout", self.model_id, item.custom_id, str(self.calls))
        if rng.random() < self.failures.timeout_rate:
            raise LlmProviderError("injected timeout")
        start = time.perf_counter()
        text, stop = self._text_for(item, settings)
        return RawResult(
            custom_id=item.custom_id,
            text=text,
            tokens_in=max(1, len(item.prompt.full_text) // 4),
            tokens_out=max(1, len(text) // 4),
            tokens_cached=max(0, len(item.prompt.shared_prefix) // 4),
            latency_s=time.perf_counter() - start,
            model_version=self.model_version,
            stop_reason=stop,
        )

    def submit_batch(self, items: list[BatchItem], settings: LlmSettings) -> BatchHandle:
        if not items:
            raise ValueError("empty batch")
        first = items[0]
        batch_id = (
            "stub-" + hashlib.sha256("|".join(i.custom_id for i in items).encode()).hexdigest()[:16]
        )
        self._batches[batch_id] = list(items)
        self._batch_settings[batch_id] = settings
        return BatchHandle(
            provider="stub",
            batch_id=batch_id,
            model=first.model,
            config_id=first.config_id,
            week_id=first.week_id,
            custom_ids=[i.custom_id for i in items],
        )

    def poll_batch(self, handle: BatchHandle) -> BatchStatus:
        n = len(handle.custom_ids)
        if self.failures.batch_never_returns:
            return BatchStatus("running", n_total=n, n_done=0, message="injected: never returns")
        return BatchStatus("done", n_total=n, n_done=n)

    def collect_batch(self, handle: BatchHandle) -> list[RawResult]:
        if self.failures.batch_never_returns:
            raise LlmProviderError("batch not ready")
        items = self._batches.get(handle.batch_id)
        if items is None:
            raise LlmProviderError(
                f"unknown stub batch {handle.batch_id}; items must be resubmitted"
            )
        settings = self._batch_settings[handle.batch_id]
        return [self.complete(i, settings) for i in items]

    def rehydrate_batch(self, handle: BatchHandle, items: list[BatchItem]) -> None:
        """Re-attach items to a handle after a process restart (the stub holds no server state)."""
        self._batches[handle.batch_id] = list(items)
        self._batch_settings.setdefault(handle.batch_id, LlmSettings())
