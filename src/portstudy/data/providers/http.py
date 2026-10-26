"""HTTP layer shared by the real-service adapters: retries, rate limits, fixture replay."""

from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Protocol

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from portstudy.hashing import stable_hash
from portstudy.io import read_json, write_json_atomic


class HttpClient(Protocol):
    def get_json(
        self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None
    ) -> Any: ...

    def get_text(
        self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None
    ) -> str: ...


class RateLimiter:
    """Simple sliding-window limiter: at most `per_minute` calls in any 60 second span."""

    def __init__(self, per_minute: int, sleep: Any = time.sleep, clock: Any = time.monotonic):
        self.per_minute = max(1, per_minute)
        self._calls: deque[float] = deque()
        self._sleep = sleep
        self._clock = clock

    def wait(self) -> None:
        now = self._clock()
        while self._calls and now - self._calls[0] > 60.0:
            self._calls.popleft()
        if len(self._calls) >= self.per_minute:
            self._sleep(60.0 - (now - self._calls[0]) + 0.01)
        self._calls.append(self._clock())


class TransientHttpError(RuntimeError):
    pass


class LiveHttp:
    """httpx client with exponential backoff on 429 and 5xx and a per-minute rate limit."""

    def __init__(
        self,
        per_minute: int = 60,
        attempts: int = 3,
        backoff_seconds: float = 2.0,
        timeout: float = 30.0,
    ) -> None:
        self.limiter = RateLimiter(per_minute)
        self.attempts = attempts
        self.backoff = backoff_seconds
        self.timeout = timeout

    def _get(
        self, url: str, params: dict[str, Any] | None, headers: dict[str, str] | None
    ) -> httpx.Response:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.attempts),
            wait=wait_exponential(multiplier=self.backoff, min=self.backoff, max=60),
            retry=retry_if_exception_type((TransientHttpError, httpx.TransportError)),
        )
        def call() -> httpx.Response:
            self.limiter.wait()
            resp = httpx.get(url, params=params, headers=headers, timeout=self.timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise TransientHttpError(f"{resp.status_code} from {url}")
            resp.raise_for_status()
            return resp

        return call()

    def get_json(
        self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None
    ) -> Any:
        return self._get(url, params, headers).json()

    def get_text(
        self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None
    ) -> str:
        return self._get(url, params, headers).text


class FixtureMissing(FileNotFoundError):
    pass


class FixtureHttp:
    """Replays recorded JSON responses keyed by a hash of (url, params).

    Headers are excluded from the key so that API keys never enter fixture names.
    If `recorder` is given and a fixture is missing, the live response is fetched and saved.
    """

    def __init__(self, folder: Path, recorder: HttpClient | None = None) -> None:
        self.folder = Path(folder)
        self.recorder = recorder
        self.calls: list[dict[str, Any]] = []

    def _key(self, url: str, params: dict[str, Any] | None) -> str:
        clean = {k: v for k, v in (params or {}).items() if k not in ("api_key", "token")}
        return stable_hash({"url": url, "params": clean})[:24]

    def _load(
        self, kind: str, url: str, params: dict[str, Any] | None, headers: dict[str, str] | None
    ) -> Any:
        self.calls.append({"url": url, "params": params})
        path = self.folder / f"{self._key(url, params)}.json"
        if path.exists():
            return read_json(path)
        if self.recorder is None:
            raise FixtureMissing(f"no fixture for {url} {params} at {path}")
        payload = (
            self.recorder.get_json(url, params, headers)
            if kind == "json"
            else self.recorder.get_text(url, params, headers)
        )
        write_json_atomic(payload, path)
        return payload

    def get_json(
        self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None
    ) -> Any:
        return self._load("json", url, params, headers)

    def get_text(
        self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None
    ) -> str:
        return str(self._load("text", url, params, headers))


def env_key(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"environment variable {name} is not set (see docs/config_reference.md)")
    return value
