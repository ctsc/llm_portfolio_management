"""OpenAI chat models through the Batch API.

# UNVERIFIED: needs real-key smoke test (Phase 17). Shapes follow the public docs:
#   POST /v1/files (purpose=batch, JSONL body)           -> {"id": file_id}
#   POST /v1/batches {input_file_id, endpoint, completion_window: "24h"} -> {"id", "status"}
#   GET  /v1/batches/{id}   -> status validating | in_progress | finalizing | completed |
#                              failed | expired | cancelled, output_file_id
#   GET  /v1/files/{id}/content   -> JSONL {custom_id, response: {status_code, body}, error}
# Prompt caching is automatic for identical prefixes above the provider minimum; the shared
# prefix is sent first as the system message so the cache applies. `reasoning_effort` carries
# the DECISION(D1) "reasoning none" setting; confirm the accepted value for GPT-6 Luna.
# Temperature 0, no tools, no web search. Key: OPENAI_API_KEY.
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

API_BASE = "https://api.openai.com"
CHAT_ENDPOINT = "/v1/chat/completions"
TERMINAL_FAILED = {"failed", "expired", "cancelled"}


class OpenAIBatchProvider(LlmProvider):
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
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")

    def _headers(self, json_body: bool = True) -> dict[str, str]:
        if not self.api_key:
            raise LlmProviderError("OPENAI_API_KEY is not set")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if json_body:
            headers["content-type"] = "application/json"
        return headers

    def build_body(self, item: BatchItem, settings: LlmSettings) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.provider_model,
            "messages": [
                {"role": "system", "content": item.prompt.shared_prefix},
                {"role": "user", "content": item.prompt.stock_part},
            ],
            "temperature": settings.temperature,
            "max_completion_tokens": settings.max_output_tokens,
            "seed": settings.seed,
            "response_format": {"type": "json_object"},
        }
        if settings.reasoning:
            body["reasoning_effort"] = settings.reasoning
        return body

    def build_jsonl(self, items: list[BatchItem], settings: LlmSettings) -> str:
        lines = [
            json.dumps(
                {
                    "custom_id": i.custom_id,
                    "method": "POST",
                    "url": CHAT_ENDPOINT,
                    "body": self.build_body(i, settings),
                }
            )
            for i in items
        ]
        return "\n".join(lines) + "\n"

    def complete(self, item: BatchItem, settings: LlmSettings) -> RawResult:
        resp = self.transport.post_json(
            f"{self.base_url}{CHAT_ENDPOINT}", self.build_body(item, settings), self._headers()
        )
        return self._result_from_body(item.custom_id, resp)

    def submit_batch(self, items: list[BatchItem], settings: LlmSettings) -> BatchHandle:
        content = self.build_jsonl(items, settings).encode("utf-8")
        first = items[0]
        upload = self.transport.post_file(
            f"{self.base_url}/v1/files",
            f"{first.week_id}_{first.model}_{first.config_id}.jsonl",
            content,
            {"purpose": "batch"},
            self._headers(json_body=False),
        )
        resp = self.transport.post_json(
            f"{self.base_url}/v1/batches",
            {
                "input_file_id": upload["id"],
                "endpoint": CHAT_ENDPOINT,
                "completion_window": "24h",
            },
            self._headers(),
        )
        return BatchHandle(
            provider="openai_batch",
            batch_id=str(resp["id"]),
            model=first.model,
            config_id=first.config_id,
            week_id=first.week_id,
            custom_ids=[i.custom_id for i in items],
            extra={"input_file_id": upload["id"]},
        )

    def poll_batch(self, handle: BatchHandle) -> BatchStatus:
        resp = self.transport.get_json(
            f"{self.base_url}/v1/batches/{handle.batch_id}", self._headers()
        )
        status = str(resp.get("status", ""))
        counts = resp.get("request_counts", {})
        n_done = int(counts.get("completed", 0))
        n_failed = int(counts.get("failed", 0))
        if status == "completed":
            handle.extra["output_file_id"] = resp.get("output_file_id")
            handle.extra["error_file_id"] = resp.get("error_file_id")
            return BatchStatus("done", len(handle.custom_ids), n_done, n_failed)
        if status in TERMINAL_FAILED:
            return BatchStatus("failed", len(handle.custom_ids), n_done, n_failed, status)
        return BatchStatus("running", len(handle.custom_ids), n_done, n_failed, status)

    def collect_batch(self, handle: BatchHandle) -> list[RawResult]:
        file_id = handle.extra.get("output_file_id")
        if not file_id:
            status = self.poll_batch(handle)
            file_id = handle.extra.get("output_file_id")
            if status.state != "done" or not file_id:
                raise LlmProviderError("batch not ready")
        text = self.transport.get_text(
            f"{self.base_url}/v1/files/{file_id}/content", self._headers(json_body=False)
        )
        out: list[RawResult] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cid = str(row["custom_id"])
            if row.get("error"):
                out.append(RawResult(custom_id=cid, text="", error=str(row["error"])))
                continue
            response = row.get("response", {})
            if int(response.get("status_code", 200)) != 200:
                out.append(
                    RawResult(custom_id=cid, text="", error=f"http_{response.get('status_code')}")
                )
                continue
            out.append(self._result_from_body(cid, response.get("body", {})))
        return out

    @staticmethod
    def _result_from_body(custom_id: str, body: dict[str, Any]) -> RawResult:
        choices = body.get("choices", [])
        text = str(choices[0].get("message", {}).get("content", "")) if choices else ""
        finish = choices[0].get("finish_reason") if choices else None
        usage = body.get("usage", {})
        details = usage.get("prompt_tokens_details", {}) or {}
        return RawResult(
            custom_id=custom_id,
            text=text,
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
            tokens_cached=int(details.get("cached_tokens", 0) or 0),
            model_version=str(body.get("model", "")),
            stop_reason="max_tokens" if finish == "length" else finish,
        )
