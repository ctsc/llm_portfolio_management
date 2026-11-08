"""OpenAI-compatible chat completions for ARCTIC LiteLLM endpoints."""

from __future__ import annotations

import email.utils
import logging
import os
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from portstudy.config import LlmSettings
from portstudy.llm.providers.base import BatchItem, LlmProvider, LlmProviderError, RawResult

logger = logging.getLogger(__name__)


class LlmAuthenticationError(LlmProviderError):
    retryable = False


class _RetryableResponse(RuntimeError):
    def __init__(self, status_code: int, retry_after: float | None):
        super().__init__(f"HTTP {status_code}")
        self.retry_after = retry_after


class _UnsupportedParameter(RuntimeError):
    def __init__(self, field: str):
        super().__init__(field)
        self.field = field


class _RetryWait(wait_exponential):
    def __call__(self, retry_state: Any) -> float:
        exception = retry_state.outcome.exception() if retry_state.outcome else None
        retry_after = getattr(exception, "retry_after", None)
        if retry_after is not None:
            return max(0.0, min(float(retry_after), 60.0))
        return super().__call__(retry_state)


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            date = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if date.tzinfo is None:
            date = date.replace(tzinfo=UTC)
        return max(0.0, (date - datetime.now(UTC)).total_seconds())


class LiteLLMProvider(LlmProvider):
    def __init__(
        self,
        model_id: str,
        provider_model: str,
        base_url: str,
        api_key_env: str,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 180,
        endpoint_name: str = "gateway",
    ) -> None:
        self.model_id = model_id
        self.provider_model = provider_model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.endpoint_name = endpoint_name
        self.timeout = timeout
        self._client = httpx.Client(transport=transport, timeout=timeout)
        self._feature_lock = threading.Lock()
        self._feature_state: tuple[bool, bool] | None = None
        self._call_cutoff: datetime | None = None
        self._call_clock: Callable[[], datetime] | None = None

    def set_call_cutoff(self, cutoff: datetime, clock: Callable[[], datetime]) -> None:
        self._call_cutoff = cutoff
        self._call_clock = clock

    def set_features(self, send_seed: bool, json_mode: bool) -> None:
        with self._feature_lock:
            self._feature_state = (send_seed, json_mode)

    @property
    def send_seed(self) -> bool | None:
        return self._feature_state[0] if self._feature_state is not None else None

    @property
    def json_mode(self) -> bool | None:
        return self._feature_state[1] if self._feature_state is not None else None

    def build_payload(
        self,
        item: BatchItem,
        settings: LlmSettings,
        *,
        send_seed: bool | None = None,
        json_mode: bool | None = None,
    ) -> dict[str, Any]:
        include_seed = settings.send_seed if send_seed is None else send_seed
        include_json = settings.json_mode if json_mode is None else json_mode
        payload: dict[str, Any] = {
            "model": self.provider_model,
            "messages": [
                {"role": "system", "content": item.prompt.shared_prefix},
                {"role": "user", "content": item.prompt.stock_part},
            ],
            "temperature": settings.temperature,
            "max_tokens": settings.max_output_tokens,
        }
        if include_seed:
            payload["seed"] = settings.seed
        if include_json:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _headers(self) -> dict[str, str]:
        key = os.environ.get(self.api_key_env, "")
        if not key and self.endpoint_name == "gateway":
            raise LlmAuthenticationError(
                f"{self.api_key_env} is not set; connect to the GSU VPN and configure the "
                "ARCTIC LiteLLM key"
            )
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = self._headers()

        @retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=_RetryWait(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type(
                (_RetryableResponse, httpx.TimeoutException, httpx.TransportError)
            ),
        )
        def request() -> httpx.Response:
            if (
                self._call_cutoff is not None
                and self._call_clock is not None
                and self._call_clock() >= self._call_cutoff
            ):
                raise LlmProviderError("J6 cutoff reached before LiteLLM request")
            response = self._client.post(url, json=payload, headers=headers)
            if response.status_code in (429,) or response.status_code >= 500:
                raise _RetryableResponse(
                    response.status_code, _retry_after(response.headers.get("Retry-After"))
                )
            if response.status_code in (401, 403):
                raise LlmAuthenticationError(
                    f"LiteLLM returned HTTP {response.status_code}; connect to the GSU VPN "
                    "and verify the configured API key"
                )
            if response.status_code == 400:
                body = response.text.lower()
                for field in ("response_format", "seed"):
                    if field in body:
                        raise _UnsupportedParameter(field)
            response.raise_for_status()
            return response

        try:
            data = request().json()
            if not isinstance(data, dict):
                raise LlmProviderError("LiteLLM response must be a JSON object")
            return {str(key): value for key, value in data.items()}
        except (LlmProviderError, _UnsupportedParameter):
            raise
        except (_RetryableResponse, httpx.TimeoutException, httpx.TransportError) as exc:
            raise LlmProviderError(f"LiteLLM request failed after retries: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise LlmProviderError(
                f"LiteLLM request failed with HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc

    def _complete_with_current_features(self, item: BatchItem, settings: LlmSettings) -> RawResult:
        with self._feature_lock:
            if self._feature_state is None:
                self._feature_state = (settings.send_seed, settings.json_mode)
            features = [self._feature_state[0], self._feature_state[1]]

        rejected: set[str] = set()
        while True:
            payload = self.build_payload(
                item, settings, send_seed=features[0], json_mode=features[1]
            )
            started = time.perf_counter()
            try:
                data = self._post(payload)
            except _UnsupportedParameter as exc:
                index = 1 if exc.field == "response_format" else 0
                with self._feature_lock:
                    current = list(self._feature_state or (settings.send_seed, settings.json_mode))
                    if not current[index] and features[index]:
                        features = current
                        continue
                    if not current[index] or exc.field in rejected:
                        raise LlmProviderError(
                            f"LiteLLM rejected {exc.field} after it was disabled"
                        ) from exc
                    current[index] = False
                    self._feature_state = (current[0], current[1])
                    features = current
                    rejected.add(exc.field)
                logger.warning("LiteLLM endpoint rejected %s; disabling it for this run", exc.field)
                continue
            elapsed = time.perf_counter() - started
            return self._result(item, data, elapsed)

    def complete(self, item: BatchItem, settings: LlmSettings) -> RawResult:
        return self._complete_with_current_features(item, settings)

    def _result(self, item: BatchItem, data: dict[str, Any], elapsed: float) -> RawResult:
        choices = data.get("choices") or []
        if not choices:
            return RawResult(
                custom_id=item.custom_id,
                text="",
                latency_s=elapsed,
                model_version=str(data.get("model") or self.provider_model),
                error="empty_choices",
                endpoint=self.endpoint_name,
                send_seed=self.send_seed,
                json_mode=self.json_mode,
            )
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        reasoning = message.get("reasoning_content")
        usage = data.get("usage") or {}
        returned_model = str(data.get("model") or self.provider_model)
        if returned_model != self.provider_model:
            logger.warning(
                "LiteLLM returned model %s for requested model %s",
                returned_model,
                self.provider_model,
            )
        finish_reason = choice.get("finish_reason")
        return RawResult(
            custom_id=item.custom_id,
            text=content if isinstance(content, str) else "",
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            tokens_cached=int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
            latency_s=elapsed,
            model_version=returned_model,
            stop_reason="max_tokens" if finish_reason == "length" else finish_reason,
            reasoning_text=reasoning if isinstance(reasoning, str) else None,
            endpoint=self.endpoint_name,
            send_seed=self.send_seed,
            json_mode=self.json_mode,
        )

    def close(self) -> None:
        self._client.close()
