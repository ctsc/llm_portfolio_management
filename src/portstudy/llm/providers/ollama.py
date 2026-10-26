"""Local models through the Ollama HTTP server.

# UNVERIFIED: needs real smoke test (Phase 17). Endpoint `/api/chat` with `format: "json"`,
# `think: false` (Ollama's thinking toggle for Qwen-class models; confirm the parameter name in
# the current Ollama API docs), and `options.temperature`, `options.seed`, `options.num_ctx`.
Calls are sequential; the runner saves after each call so a crashed job resumes.
"""

from __future__ import annotations

import time
from typing import Any

from portstudy.config import LlmSettings
from portstudy.llm.providers.base import BatchItem, LlmProvider, LlmProviderError, RawResult
from portstudy.llm.providers.transport import HttpxTransport, LlmTransport

DEFAULT_HOST = "http://localhost:11434"


class OllamaProvider(LlmProvider):
    supports_batch = False

    def __init__(
        self,
        model_id: str,
        provider_model: str,
        transport: LlmTransport | None = None,
        host: str = DEFAULT_HOST,
        num_ctx: int = 16384,
    ) -> None:
        self.model_id = model_id
        self.provider_model = provider_model
        self.transport = transport or HttpxTransport(timeout=600.0)
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx

    def build_payload(self, item: BatchItem, settings: LlmSettings) -> dict[str, Any]:
        """Shared prefix as the system message, stock part as the user message. No tools."""
        return {
            "model": self.provider_model,
            "messages": [
                {"role": "system", "content": item.prompt.shared_prefix},
                {"role": "user", "content": item.prompt.stock_part},
            ],
            "stream": False,
            "format": "json",
            "think": bool(settings.think),
            "options": {
                "temperature": settings.temperature,
                "seed": settings.seed,
                "num_predict": settings.max_output_tokens,
                "num_ctx": self.num_ctx,
            },
        }

    def complete(self, item: BatchItem, settings: LlmSettings) -> RawResult:
        payload = self.build_payload(item, settings)
        start = time.perf_counter()
        try:
            resp = self.transport.post_json(f"{self.host}/api/chat", payload, {})
        except Exception as exc:  # transport errors become provider errors
            raise LlmProviderError(f"ollama call failed: {exc}") from exc
        latency = time.perf_counter() - start
        message = resp.get("message", {}) if isinstance(resp, dict) else {}
        text = str(message.get("content", ""))
        done_reason = resp.get("done_reason") if isinstance(resp, dict) else None
        return RawResult(
            custom_id=item.custom_id,
            text=text,
            tokens_in=int(resp.get("prompt_eval_count", 0) or 0),
            tokens_out=int(resp.get("eval_count", 0) or 0),
            tokens_cached=0,
            latency_s=latency,
            model_version=str(resp.get("model", self.provider_model)),
            stop_reason="max_tokens" if done_reason == "length" else done_reason,
        )
