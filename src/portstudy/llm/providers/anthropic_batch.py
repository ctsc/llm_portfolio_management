"""Claude through the Anthropic Message Batches API with prompt caching on the shared prefix.

# UNVERIFIED: needs real-key smoke test (PLAN2 phase 15 E). Request shapes follow the public docs:
#   POST /v1/messages/batches   {"requests": [{"custom_id", "params": {...}}]}
#   GET  /v1/messages/batches/{id}   -> processing_status in_progress | ended, results_url
#   GET  results_url                  -> JSONL {custom_id, result: {type, message}}
# The shared prefix is sent as a `system` block with `cache_control: {"type": "ephemeral"}`.
# Temperature 0, no tools, no web search. Key: ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import json
import os
from typing import Any

from portstudy.config import LlmSettings
from portstudy.llm.providers.base import (
    BatchHandle,
    BatchItem,
    BatchStatus,
    LlmProvider,
    LlmProviderError,
    RawResult,
)
from portstudy.llm.providers.transport import HttpxTransport, LlmTransport

API_BASE = "https://api.anthropic.com"
API_VERSION = "2023-06-01"


class AnthropicBatchProvider(LlmProvider):
    supports_batch = True

    def __init__(
        self,
        model_id: str,
        provider_model: str,
        transport: LlmTransport | None = None,
        api_key: str | None = None,
        base_url: str = API_BASE,
    ) -> None:
        self.model_id = model_id
        self.provider_model = provider_model
        self.transport = transport or HttpxTransport()
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise LlmProviderError("ANTHROPIC_API_KEY is not set")
        return {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

    def build_params(self, item: BatchItem, settings: LlmSettings) -> dict[str, Any]:
        """One request's params. Prefix first (cached system block), then the stock part."""
        return {
            "model": self.provider_model,
            "max_tokens": settings.max_output_tokens,
            "temperature": settings.temperature,
            "system": [
                {
                    "type": "text",
                    "text": item.prompt.shared_prefix,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": item.prompt.stock_part}],
        }

    def build_batch_payload(self, items: list[BatchItem], settings: LlmSettings) -> dict[str, Any]:
        return {
            "requests": [
                {"custom_id": i.custom_id, "params": self.build_params(i, settings)} for i in items
            ]
        }

    def complete(self, item: BatchItem, settings: LlmSettings) -> RawResult:
        resp = self.transport.post_json(
            f"{self.base_url}/v1/messages", self.build_params(item, settings), self._headers()
        )
        return self._result_from_message(item.custom_id, resp)

    def submit_batch(self, items: list[BatchItem], settings: LlmSettings) -> BatchHandle:
        resp = self.transport.post_json(
            f"{self.base_url}/v1/messages/batches",
            self.build_batch_payload(items, settings),
            self._headers(),
        )
        first = items[0]
        return BatchHandle(
            provider="anthropic_batch",
            batch_id=str(resp["id"]),
            model=first.model,
            config_id=first.config_id,
            week_id=first.week_id,
            custom_ids=[i.custom_id for i in items],
        )

    def poll_batch(self, handle: BatchHandle) -> BatchStatus:
        resp = self.transport.get_json(
            f"{self.base_url}/v1/messages/batches/{handle.batch_id}", self._headers()
        )
        counts = resp.get("request_counts", {})
        n_done = int(counts.get("succeeded", 0))
        n_failed = int(counts.get("errored", 0)) + int(counts.get("expired", 0))
        status = str(resp.get("processing_status", ""))
        if status == "ended":
            handle.extra["results_url"] = resp.get("results_url")
            return BatchStatus("done", len(handle.custom_ids), n_done, n_failed)
        return BatchStatus("running", len(handle.custom_ids), n_done, n_failed, status)

    def collect_batch(self, handle: BatchHandle) -> list[RawResult]:
        url = handle.extra.get("results_url")
        if not url:
            status = self.poll_batch(handle)
            url = handle.extra.get("results_url")
            if status.state != "done" or not url:
                raise LlmProviderError("batch not ready")
        text = self.transport.get_text(str(url), self._headers())
        out: list[RawResult] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cid = str(row["custom_id"])
            result = row.get("result", {})
            if result.get("type") == "succeeded":
                out.append(self._result_from_message(cid, result["message"]))
            else:
                out.append(RawResult(custom_id=cid, text="", error=f"batch_{result.get('type')}"))
        return out

    @staticmethod
    def _result_from_message(custom_id: str, message: dict[str, Any]) -> RawResult:
        content = message.get("content", [])
        text = "".join(str(b.get("text", "")) for b in content if b.get("type") == "text")
        usage = message.get("usage", {})
        stop = message.get("stop_reason")
        return RawResult(
            custom_id=custom_id,
            text=text,
            tokens_in=int(usage.get("input_tokens", 0))
            + int(usage.get("cache_read_input_tokens", 0) or 0)
            + int(usage.get("cache_creation_input_tokens", 0) or 0),
            tokens_out=int(usage.get("output_tokens", 0)),
            tokens_cached=int(usage.get("cache_read_input_tokens", 0) or 0),
            model_version=str(message.get("model", "")),
            stop_reason="max_tokens" if stop == "max_tokens" else stop,
        )
