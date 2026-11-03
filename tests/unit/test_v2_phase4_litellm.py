from __future__ import annotations

import datetime as dt
import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
import pytest

from portstudy.config import (
    GatewayConfig,
    LlmSettings,
    ModelRegistry,
    ModelSpec,
    StudyConfig,
)
from portstudy.llm.cache import LlmCache
from portstudy.llm.prompts import load_manifest
from portstudy.llm.providers.base import BatchItem, LlmProvider, LlmProviderError, RawResult
from portstudy.llm.providers.factory import make_provider
from portstudy.llm.providers.litellm import LiteLLMProvider, LlmAuthenticationError
from portstudy.llm.providers.stub import StubLlmProvider
from portstudy.llm.runner import WeekRunner, lock_deadline, read_run_status, read_scores
from portstudy.llm.validate import parse_response
from portstudy.paths import week_output_dir
from portstudy.pipeline.context import synthetic_clock

ET = ZoneInfo("America/New_York")
RATING = {"low": -1, "likely": 0, "high": 1, "rationale": "steady"}


def _response(
    *,
    content: str | None = None,
    model: str = "provider/model",
    choices: list[dict[str, object]] | None = None,
    usage: dict[str, int] | None = None,
    finish_reason: str | None = "stop",
) -> dict[str, object]:
    if choices is None:
        choices = [
            {
                "message": {"role": "assistant", "content": content or json.dumps(RATING)},
                "finish_reason": finish_reason,
            }
        ]
    return {
        "id": "completion-id",
        "model": model,
        "choices": choices,
        "usage": usage or {"prompt_tokens": 11, "completion_tokens": 7},
    }


def _item(study_config: StudyConfig) -> BatchItem:
    manifest = load_manifest()
    prompt = manifest.configurations[0]
    from portstudy.llm.prompts import assemble_prompt

    assembled = assemble_prompt(
        prompt.id,
        {"ticker": "AAA", "name": "Alpha", "sector": "Energy"},
        "Alpha beats estimates.",
        study_config,
        manifest,
    )
    return BatchItem("m__bare__2026-W05__AAA", "m", prompt.id, "2026-W05", "AAA", assembled)


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: str = "provider/model",
) -> LiteLLMProvider:
    monkeypatch.setenv("ARCTIC_LITELLM_API_KEY", "test-key")
    return LiteLLMProvider(
        "model-id",
        model,
        "https://gateway.example/v1/",
        "ARCTIC_LITELLM_API_KEY",
        transport=httpx.MockTransport(handler),
    )


def test_litellm_records_payload_and_bearer_header(
    study_config: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_response())

    provider = _provider(handler, monkeypatch)
    result = provider.complete(_item(study_config), LlmSettings(seed=77, max_output_tokens=42))

    request = seen[0]
    payload = json.loads(request.content)
    assert str(request.url) == "https://gateway.example/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert payload["model"] == "provider/model"
    assert payload["messages"] == [
        {"role": "system", "content": _item(study_config).prompt.shared_prefix},
        {"role": "user", "content": _item(study_config).prompt.stock_part},
    ]
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 42
    assert payload["seed"] == 77
    assert payload["response_format"] == {"type": "json_object"}
    assert result.tokens_in == 11 and result.tokens_out == 7


def test_litellm_omits_disabled_seed_and_json_mode(
    study_config: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_response())

    provider = _provider(handler, monkeypatch)
    provider.complete(
        _item(study_config),
        LlmSettings(send_seed=False, json_mode=False),
    )

    assert "seed" not in seen[0]
    assert "response_format" not in seen[0]


def test_litellm_disables_rejected_response_format_for_the_run(
    study_config: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        if "response_format" in payload:
            return httpx.Response(400, json={"error": {"message": "response_format unsupported"}})
        return httpx.Response(200, json=_response())

    provider = _provider(handler, monkeypatch)
    settings = LlmSettings()
    provider.complete(_item(study_config), settings)
    provider.complete(_item(study_config), settings)

    assert "response_format" in payloads[0]
    assert "response_format" not in payloads[1]
    assert "response_format" not in payloads[2]
    assert provider.json_mode is False


def test_litellm_disables_rejected_seed(
    study_config: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        if "seed" in payload:
            return httpx.Response(400, json={"error": {"message": "seed is not supported"}})
        return httpx.Response(200, json=_response())

    provider = _provider(handler, monkeypatch)
    provider.complete(_item(study_config), LlmSettings())

    assert "seed" in payloads[0]
    assert "seed" not in payloads[1]
    assert provider.send_seed is False


def test_litellm_retries_429_using_retry_after(
    study_config: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "busy"})
        return httpx.Response(200, json=_response())

    provider = _provider(handler, monkeypatch)
    result = provider.complete(_item(study_config), LlmSettings())

    assert calls == 2
    assert result.error is None


def test_litellm_does_not_start_request_after_j6_cutoff(
    study_config: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_response())

    provider = _provider(handler, monkeypatch)
    cutoff = dt.datetime.now(dt.UTC)
    provider.set_call_cutoff(cutoff, lambda: cutoff)
    with pytest.raises(LlmProviderError, match="J6 cutoff"):
        provider.complete(_item(study_config), LlmSettings())
    assert calls == 0


def test_renamed_open_weight_stubs_keep_their_random_stream(study_config: StudyConfig) -> None:
    item = _item(study_config)
    assert StubLlmProvider("stub-ow-a").answer_for(item) == StubLlmProvider(
        "stub-local-a"
    ).answer_for(item)


def test_429_storm_leaves_open_weight_calls_for_j6(
    pipeline_dir: Path,
    tmp_path: Path,
    model_registry: ModelRegistry,
    study_config: StudyConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "busy"})

    runner = _runner(
        week="2026-W15",
        registry=_litellm_registry(model_registry),
        cfg=study_config,
        provider=LiteLLMProvider(
            "stub-ow-a",
            "provider/model",
            "https://gateway.example/v1",
            "ARCTIC_LITELLM_API_KEY",
            transport=httpx.MockTransport(handler),
        ),
        tickers=1,
        tmp_path=tmp_path,
        concurrency=1,
    )
    monkeypatch.setenv("ARCTIC_LITELLM_API_KEY", "test-key")
    summary = runner.run_open_weight("stub-ow-a")
    cutoff = lock_deadline("2026-W15", study_config) - dt.timedelta(
        minutes=study_config.pipeline.lock_margin_minutes
    )
    assessed = runner.assess_run("stub-ow-a", "bare", now=cutoff, deadline=cutoff)

    assert calls == 3
    assert summary.n_calls == 1
    assert read_run_status("2026-W15")["stub-ow-a__bare"].n_done == 0
    assert assessed.state == "failed"


def test_litellm_401_fails_with_gsu_vpn_hint(
    study_config: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": "unauthorized"})

    provider = _provider(handler, monkeypatch)
    with pytest.raises(LlmAuthenticationError, match="GSU VPN"):
        provider.complete(_item(study_config), LlmSettings())
    assert calls == 1


def test_litellm_parses_reasoning_version_usage_and_length(
    study_config: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _response(
        model="provider/returned-version",
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(RATING),
                    "reasoning_content": "analysis",
                },
                "finish_reason": "length",
            }
        ],
    )
    provider = _provider(
        lambda _: httpx.Response(200, json=payload), monkeypatch, model="provider/requested"
    )
    result = provider.complete(_item(study_config), LlmSettings())
    parsed = parse_response(result.text, reasoning_text=result.reasoning_text)

    assert result.model_version == "provider/returned-version"
    assert result.tokens_in == 11 and result.tokens_out == 7
    assert result.stop_reason == "max_tokens"
    assert result.reasoning_text == "analysis"
    assert parsed.had_thinking


def test_litellm_handles_empty_choices_and_null_content(
    study_config: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = iter(
        [
            httpx.Response(200, json=_response(choices=[])),
            httpx.Response(
                200,
                json=_response(
                    choices=[
                        {
                            "message": {"content": None, "reasoning_content": "private reasoning"},
                            "finish_reason": "stop",
                        }
                    ]
                ),
            ),
        ]
    )
    provider = _provider(lambda _: next(responses), monkeypatch)

    empty = provider.complete(_item(study_config), LlmSettings())
    reasoning_only = provider.complete(_item(study_config), LlmSettings())

    assert empty.error == "empty_choices"
    assert reasoning_only.text == ""
    assert reasoning_only.reasoning_text == "private reasoning"


def test_registry_rejects_enabled_tbd_and_prohibited_ids(model_registry: ModelRegistry) -> None:
    models = list(model_registry.models)
    models[0] = models[0].model_copy(update={"provider_model": "TBD"})
    with pytest.raises(ValueError, match="provider_model"):
        ModelRegistry(models=models, stub_models=model_registry.stub_models)

    with pytest.raises(ValueError, match="prohibited fragment"):
        ModelRegistry(
            models=[
                ModelSpec(
                    id="qwen-test",
                    display_name="test",
                    kind="open_weight",
                    provider="stub",
                    provider_model="test",
                )
            ]
        )


def test_fallback_endpoint_rejects_missing_or_stale_file(tmp_path: Path) -> None:
    spec = ModelSpec(
        id="model",
        display_name="model",
        kind="open_weight",
        provider="litellm",
        provider_model="provider/model",
        serving="slurm_fallback",
        fallback_eligible=True,
    )
    missing = GatewayConfig(
        fallback_endpoint_file=str(tmp_path / "missing.json"),
    )
    with pytest.raises(FileNotFoundError, match="fallback endpoint file is missing"):
        make_provider(
            spec, gateway=missing, week_id="2026-W05", run_started_at="2026-01-01T00:00:00+00:00"
        )

    endpoint = tmp_path / "stale.json"
    endpoint.write_text('{"base_url":"http://slurm","week_id":"2026-W05"}\n', encoding="utf-8")
    old = time.time() - 60
    os.utime(endpoint, (old, old))
    with pytest.raises(ValueError, match="fallback endpoint file is stale"):
        make_provider(
            spec,
            gateway=GatewayConfig(fallback_endpoint_file=str(endpoint)),
            week_id="2026-W05",
            run_started_at=(dt.datetime.now(dt.UTC) - dt.timedelta(seconds=30)).isoformat(),
        )


class _EndpointProvider(LlmProvider):
    def __init__(self, model_id: str, endpoint: str, fail_after: int | None = None) -> None:
        self.model_id = model_id
        self.endpoint = endpoint
        self.fail_after = fail_after
        self.calls = 0

    def complete(self, item: BatchItem, settings: LlmSettings) -> RawResult:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("injected crash")
        return RawResult(
            custom_id=item.custom_id,
            text=json.dumps(RATING),
            tokens_in=1,
            tokens_out=1,
            model_version="stub-version",
            endpoint=self.endpoint,
            send_seed=settings.send_seed,
            json_mode=settings.json_mode,
        )


def _runner(
    *,
    week: str,
    registry: ModelRegistry,
    cfg: StudyConfig,
    provider: LlmProvider,
    tickers: int,
    tmp_path: Path,
    concurrency: int,
) -> WeekRunner:
    model = "stub-ow-a"
    reference = pd.DataFrame(
        [
            {"ticker": f"T{index:03d}", "name": f"Company {index}", "sector": "Energy"}
            for index in range(tickers)
        ]
    )
    news = pd.DataFrame(
        [{"ticker": f"T{index:03d}", "news_text": "No news."} for index in range(tickers)]
    )
    snapshot = SimpleNamespace(
        reference=reference,
        news_text=news,
        included_tickers=reference["ticker"].tolist(),
    )
    manifest = load_manifest()
    manifest = manifest.model_copy(
        update={"configurations": [item for item in manifest.configurations if item.id == "bare"]}
    )
    cfg = cfg.model_copy(
        update={
            "llm": cfg.llm.model_copy(
                update={"concurrency": {model: concurrency}, "prompt_ids": ["bare"]}
            )
        }
    )
    return WeekRunner(
        week,
        cfg,
        registry,
        snapshot,  # type: ignore[arg-type]
        synthetic=True,
        manifest=manifest,
        providers={model: provider},
        cache=LlmCache(tmp_path / "cache"),
        clock=synthetic_clock(week),
    )


def _litellm_registry(registry: ModelRegistry, *, serving: str = "gateway") -> ModelRegistry:
    models = [
        model.model_copy(
            update={
                "provider": "litellm",
                "provider_model": "provider/model",
                "serving": serving,
                "fallback_eligible": serving == "slurm_fallback",
            }
        )
        if model.id == "stub-ow-a"
        else model
        for model in registry.stub_models
    ]
    return registry.model_copy(update={"stub_models": models})


def test_open_weight_concurrency_has_equivalent_sorted_records(
    pipeline_dir: Path,
    tmp_path: Path,
    model_registry: ModelRegistry,
    study_config: StudyConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARCTIC_LITELLM_API_KEY", "test-key")
    active = 0
    maximum = 0
    lock = threading.Lock()

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.005)
        with lock:
            active -= 1
        return httpx.Response(200, json=_response())

    registry = _litellm_registry(model_registry)
    fast_week, serial_week = "2026-W10", "2026-W11"
    fast = _runner(
        week=fast_week,
        registry=registry,
        cfg=study_config,
        provider=LiteLLMProvider(
            "stub-ow-a",
            "provider/model",
            "https://gateway.example/v1",
            "ARCTIC_LITELLM_API_KEY",
            transport=httpx.MockTransport(handler),
        ),
        tickers=50,
        tmp_path=tmp_path / "fast",
        concurrency=4,
    )
    serial = _runner(
        week=serial_week,
        registry=registry,
        cfg=study_config,
        provider=LiteLLMProvider(
            "stub-ow-a",
            "provider/model",
            "https://gateway.example/v1",
            "ARCTIC_LITELLM_API_KEY",
            transport=httpx.MockTransport(handler),
        ),
        tickers=50,
        tmp_path=tmp_path / "serial",
        concurrency=1,
    )
    fast_summary = fast.run_open_weight("stub-ow-a")
    serial_summary = serial.run_open_weight("stub-ow-a")

    fast_scores = read_scores(fast_week).sort_values(["model", "config_id", "ticker"])
    serial_scores = read_scores(serial_week).sort_values(["model", "config_id", "ticker"])
    pd.testing.assert_frame_equal(
        fast_scores.drop(columns=["latency_s"]),
        serial_scores.drop(columns=["latency_s"]),
        check_like=True,
    )
    assert fast_summary.n_calls == serial_summary.n_calls == 50
    assert maximum > 1


def test_open_weight_resume_skips_twenty_completed_records(
    pipeline_dir: Path,
    tmp_path: Path,
    model_registry: ModelRegistry,
    study_config: StudyConfig,
) -> None:
    registry = _litellm_registry(model_registry)
    interrupted = _EndpointProvider("stub-ow-a", "gateway", fail_after=20)
    first = _runner(
        week="2026-W12",
        registry=registry,
        cfg=study_config,
        provider=interrupted,
        tickers=50,
        tmp_path=tmp_path,
        concurrency=1,
    )
    with pytest.raises(RuntimeError, match="injected crash"):
        first.run_open_weight("stub-ow-a")
    assert len(read_scores("2026-W12")) == 20

    resumed_provider = _EndpointProvider("stub-ow-a", "gateway")
    resumed = _runner(
        week="2026-W12",
        registry=registry,
        cfg=study_config,
        provider=resumed_provider,
        tickers=50,
        tmp_path=tmp_path,
        concurrency=1,
    )
    summary = resumed.run_open_weight("stub-ow-a")

    assert resumed_provider.calls == 30
    assert summary.n_calls == 30
    assert len(read_scores("2026-W12")) == 50
    assert not read_scores("2026-W12").duplicated(["model", "config_id", "ticker"]).any()


def test_gateway_fallback_archives_and_restarts_whole_source(
    pipeline_dir: Path,
    tmp_path: Path,
    model_registry: ModelRegistry,
    study_config: StudyConfig,
) -> None:
    gateway_registry = _litellm_registry(model_registry)
    initial = _runner(
        week="2026-W13",
        registry=gateway_registry,
        cfg=study_config,
        provider=_EndpointProvider("stub-ow-a", "gateway"),
        tickers=3,
        tmp_path=tmp_path,
        concurrency=1,
    )
    initial.run_open_weight("stub-ow-a")
    assert set(read_scores("2026-W13")["endpoint"]) == {"gateway"}

    fallback_registry = _litellm_registry(model_registry, serving="slurm_fallback")
    fallback_provider = _EndpointProvider("stub-ow-a", "slurm_fallback")
    fallback = _runner(
        week="2026-W13",
        registry=fallback_registry,
        cfg=study_config,
        provider=fallback_provider,
        tickers=3,
        tmp_path=tmp_path,
        concurrency=1,
    )
    fallback.run_open_weight("stub-ow-a")

    archive = week_output_dir("2026-W13") / "llm" / "archive" / "stub-ow-a__bare.gateway.jsonl"
    assert archive.exists()
    assert len(archive.read_text(encoding="utf-8").splitlines()) == 3
    assert fallback_provider.calls == 3
    assert set(read_scores("2026-W13")["endpoint"]) == {"slurm_fallback"}
    assert read_run_status("2026-W13")["stub-ow-a__bare"].endpoint == "slurm_fallback"


def test_reasoning_only_response_is_logged_as_thinking(
    pipeline_dir: Path,
    tmp_path: Path,
    model_registry: ModelRegistry,
    study_config: StudyConfig,
) -> None:
    class ReasoningProvider(_EndpointProvider):
        def complete(self, item: BatchItem, settings: LlmSettings) -> RawResult:
            result = super().complete(item, settings)
            result.text = ""
            result.reasoning_text = "reasoning without content"
            return result

    runner = _runner(
        week="2026-W14",
        registry=model_registry,
        cfg=study_config,
        provider=ReasoningProvider("stub-ow-a", "gateway"),
        tickers=1,
        tmp_path=tmp_path,
        concurrency=1,
    )
    runner.run_open_weight("stub-ow-a")

    scores = read_scores("2026-W14")
    assert scores.iloc[0]["had_thinking"]
    assert not scores.iloc[0]["valid"]
