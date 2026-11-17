"""OpenAI-compatible chat completions for ARCTIC LiteLLM endpoints."""

from __future__ import annotations

import email.utils
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from portstudy.config import LlmSettings
from portstudy.llm.providers.base import (
    BatchItem,
    LlmCallCutoff,
    LlmProvider,
    LlmProviderError,
    RawResult,
)

logger = logging.getLogger(__name__)

_GATEWAY_HEADERS = (
    "x-litellm-model-id",
    "x-litellm-version",
    "x-litellm-model-group",
    "x-litellm-attempted-fallbacks",
    "x-litellm-attempted-retries",
    "x-litellm-response-duration-ms",
    "x-litellm-response-cost",
    "x-litellm-call-id",
)
_BUDGET_MARKERS = ("ExceededBudget", "ExceededTokenBudget", "budget_exceeded")
# UNVERIFIED: real ARCTIC error text. The rejection must name the parameter as unsupported or
# unknown next to its name; an echo of the request JSON does not match.
_UNSUPPORTED_PARAMETER = re.compile(
    r"(unsupported|unknown|unrecognized|unexpected|not supported|does not support)"
    r"[^.\n]{0,80}\b(seed|response_format)\b"
    r"|\b(seed|response_format)\b[^.\n]{0,80}"
    r"(unsupported|unknown|unrecognized|not supported|not allowed|not permitted)",
    re.IGNORECASE,
)


def _gateway_headers(headers: httpx.Headers) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, value in headers.items():
        key = name.lower()
        if key in _GATEWAY_HEADERS or key.startswith("x-ratelimit-"):
            out[key] = value
    return out


class LlmAuthenticationError(LlmProviderError):
    retryable = False
    model_wide = True


class LlmBudgetError(LlmProviderError):
    """ARCTIC key or team budget is spent; retrying does not help, it needs ARCTIC."""

    retryable = False
    model_wide = True


class LlmRequestRejected(LlmProviderError):
    """A 4xx the gateway will keep rejecting; the request is written as provider_error."""

    retryable = False


def _ratelimit_wait(headers: httpx.Headers) -> float | None:
    # UNVERIFIED: header format on ARCTIC; use the smaller numeric reset when both parse as
    # seconds, capped at 60 s by the retry wait.
    waits = []
    for name in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        value = headers.get(name)
        if value is None:
            continue
        try:
            waits.append(max(0.0, float(value)))
        except ValueError:
            continue
    return min(waits) if waits else None


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
        features_frozen: bool = False,
    ) -> None:
        self.model_id = model_id
        self.provider_model = provider_model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.endpoint_name = endpoint_name
        self.timeout = timeout
        self.features_frozen = features_frozen
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

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], httpx.Headers]:
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
                raise LlmCallCutoff("J6 cutoff reached before LiteLLM request")
            response = self._client.post(url, json=payload, headers=headers)
            body = response.text
            if response.status_code >= 400 and any(marker in body for marker in _BUDGET_MARKERS):
                raise LlmBudgetError(
                    f"LiteLLM budget error (HTTP {response.status_code}): {body[:500]}"
                )
            if response.status_code in (401, 403):
                raise LlmAuthenticationError(
                    f"LiteLLM returned HTTP {response.status_code}; connect to the GSU VPN "
                    "and verify the configured API key"
                )
            if response.status_code == 429:
                raise _RetryableResponse(
                    response.status_code,
                    _retry_after(response.headers.get("Retry-After"))
                    if response.headers.get("Retry-After") is not None
                    else _ratelimit_wait(response.headers),
                )
            if response.status_code >= 500:
                raise _RetryableResponse(response.status_code, None)
            if response.status_code == 400:
                logger.warning(
                    "LiteLLM returned HTTP 400 for %s; body: %s",
                    self.provider_model,
                    body[:2000],
                )
                match = _UNSUPPORTED_PARAMETER.search(body)
                if match is not None:
                    field = next(
                        g.lower()
                        for g in match.groups()
                        if g and g.lower() in ("seed", "response_format")
                    )
                    if self.features_frozen:
                        raise LlmRequestRejected(f"gateway rejected frozen feature {field}")
                    raise _UnsupportedParameter(field)
            if 400 <= response.status_code < 500:
                raise LlmRequestRejected(
                    f"LiteLLM request failed with HTTP {response.status_code}: {body[:500]}"
                )
            response.raise_for_status()
            return response

        try:
            response = request()
            data = response.json()
            if not isinstance(data, dict):
                raise LlmProviderError("LiteLLM response must be a JSON object")
            return {str(key): value for key, value in data.items()}, response.headers
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
                data, headers = self._post(payload)
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
            return self._result(item, data, headers, elapsed)

    def complete(self, item: BatchItem, settings: LlmSettings) -> RawResult:
        return self._complete_with_current_features(item, settings)

    def _result(
        self,
        item: BatchItem,
        data: dict[str, Any],
        headers: httpx.Headers,
        elapsed: float,
    ) -> RawResult:
        gateway_headers = _gateway_headers(headers)
        deployment_id = gateway_headers.get("x-litellm-model-id", "")
        gateway_version = gateway_headers.get("x-litellm-version", "")
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
                usage=data.get("usage") if isinstance(data.get("usage"), dict) else None,
                deployment_id=deployment_id,
                gateway_version=gateway_version,
                gateway_headers=gateway_headers,
            )
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        reasoning = message.get("reasoning_content")
        usage = data.get("usage") or {}
        returned_model = str(data.get("model") or self.provider_model)
        # UNVERIFIED: header values on ARCTIC. A served-by identity that does not match the
        # requested model, or any gateway fallback attempt, is an invalid answer.
        error = None
        if returned_model != self.provider_model and not returned_model.endswith(
            "/" + self.provider_model
        ):
            error = "model_mismatch"
            logger.warning(
                "LiteLLM returned model %s for requested model %s",
                returned_model,
                self.provider_model,
            )
        attempted_fallbacks = gateway_headers.get("x-litellm-attempted-fallbacks", "")
        model_group = gateway_headers.get("x-litellm-model-group", "")
        try:
            fallback_attempts = int(attempted_fallbacks) if attempted_fallbacks else 0
        except ValueError:
            fallback_attempts = 0
        if error is None and (
            fallback_attempts > 0 or (model_group and model_group != self.provider_model)
        ):
            error = "gateway_fallback"
        finish_reason = choice.get("finish_reason")
        details = usage.get("completion_tokens_details") or {}
        tokens_reasoning = (
            int(details.get("reasoning_tokens") or 0)
            if isinstance(details, dict) and details.get("reasoning_tokens") is not None
            else int(usage.get("reasoning_tokens") or 0)
        )
        return RawResult(
            custom_id=item.custom_id,
            text=content if isinstance(content, str) else "",
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            tokens_cached=int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
            latency_s=elapsed,
            model_version=returned_model,
            error=error,
            stop_reason="max_tokens" if finish_reason == "length" else finish_reason,
            reasoning_text=reasoning if isinstance(reasoning, str) else None,
            endpoint=self.endpoint_name,
            send_seed=self.send_seed,
            json_mode=self.json_mode,
            tokens_reasoning=tokens_reasoning,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            usage=usage if isinstance(usage, dict) else None,
            deployment_id=deployment_id,
            gateway_version=gateway_version,
            gateway_headers=gateway_headers,
        )

    def close(self) -> None:
        self._client.close()
