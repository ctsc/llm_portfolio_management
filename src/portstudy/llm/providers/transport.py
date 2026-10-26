"""Minimal JSON transport for the LLM adapters, injectable so tests can inspect payloads."""

from __future__ import annotations

from typing import Any, Protocol

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from portstudy.data.providers.http import TransientHttpError


class LlmTransport(Protocol):
    def post_json(self, url: str, payload: Any, headers: dict[str, str]) -> Any: ...

    def get_json(self, url: str, headers: dict[str, str]) -> Any: ...

    def get_text(self, url: str, headers: dict[str, str]) -> str: ...

    def post_file(
        self, url: str, filename: str, content: bytes, data: dict[str, str], headers: dict[str, str]
    ) -> Any: ...


class HttpxTransport:
    """httpx with exponential backoff on 429 and 5xx."""

    def __init__(self, attempts: int = 3, backoff_seconds: float = 2.0, timeout: float = 120.0):
        self.attempts = attempts
        self.backoff = backoff_seconds
        self.timeout = timeout

    def _send(self, fn: Any) -> httpx.Response:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.attempts),
            wait=wait_exponential(multiplier=self.backoff, min=self.backoff, max=60),
            retry=retry_if_exception_type((TransientHttpError, httpx.TransportError)),
        )
        def call() -> httpx.Response:
            resp: httpx.Response = fn()
            if resp.status_code == 429 or resp.status_code >= 500:
                raise TransientHttpError(f"{resp.status_code} from {resp.request.url}")
            resp.raise_for_status()
            return resp

        return call()

    def post_json(self, url: str, payload: Any, headers: dict[str, str]) -> Any:
        return self._send(
            lambda: httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
        ).json()

    def get_json(self, url: str, headers: dict[str, str]) -> Any:
        return self._send(lambda: httpx.get(url, headers=headers, timeout=self.timeout)).json()

    def get_text(self, url: str, headers: dict[str, str]) -> str:
        return self._send(lambda: httpx.get(url, headers=headers, timeout=self.timeout)).text

    def post_file(
        self, url: str, filename: str, content: bytes, data: dict[str, str], headers: dict[str, str]
    ) -> Any:
        return self._send(
            lambda: httpx.post(
                url,
                files={"file": (filename, content, "application/jsonl")},
                data=data,
                headers=headers,
                timeout=self.timeout,
            )
        ).json()
