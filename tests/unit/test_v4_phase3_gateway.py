"""v4 phase 3b: the LiteLLM gateway sends per-model limits and timeouts, reports its own
headers and token accounting, and classifies failures as budget, authentication, retryable,
rejected or per-request provider errors (DECISION(G15), N5).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from portstudy.config import LlmSettings, ModelRegistry, StudyConfig
from portstudy.llm.cache import LlmCache
from portstudy.llm.cost import CostTable, estimate_week_cost
from portstudy.llm.providers.base import BatchItem, RawResult
from portstudy.llm.providers.litellm import (
    LiteLLMProvider,
    LlmBudgetError,
    LlmRequestRejected,
)
from portstudy.llm.runner import lock_deadline, read_scores, settings_for_model
from portstudy.llm.schema import INVALID_TRUNCATED
from portstudy.llm.validate import parse_response
from portstudy.pipeline.status import JobFailed
from tests.unit.test_v2_phase4_litellm import _litellm_registry, _runner

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_weekend  # noqa: E402
import smoke_real_services  # noqa: E402

RATING = {"low": -1, "likely": 0, "high": 1, "rationale": "steady"}
MARKER = "thought"


def _response(
    *,
    content: str | None = None,
    model: str = "provider/model",
    usage: dict[str, object] | None = None,
    finish_reason: str | None = "stop",
    reasoning: str | None = None,
) -> dict[str, object]:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "completion-id",
        "model": model,
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 11, "completion_tokens": 7},
    }


def _item(study_config: StudyConfig) -> BatchItem:
    from portstudy.llm.prompts import assemble_prompt, load_manifest

    manifest = load_manifest()
    prompt = manifest.configurations[0]
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
    **kwargs: object,
) -> LiteLLMProvider:
    monkeypatch.setenv("ARCTIC_LITELLM_API_KEY", "test-key")
    return LiteLLMProvider(
        "model-id",
        "provider/model",
        "https://gateway.example/v1/",
        "ARCTIC_LITELLM_API_KEY",
        transport=httpx.MockTransport(handler),
        **kwargs,  # type: ignore[arg-type]
    )


def _records(week: str, model: str = "stub-ow-a") -> list[dict[str, object]]:
    folder = Path(os.environ["PORTSTUDY_DATA_DIR"]) / "outputs" / week / "llm"
    rows: list[dict[str, object]] = []
    for path in sorted(folder.glob(f"{model}__*.jsonl")):
        rows.extend(json.loads(ln) for ln in path.read_text().splitlines() if ln.strip())
    return rows


def test_gemma_empty_thought_marker_before_json_parses_valid(
    study_config: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first JSON object wins: an empty thought-channel marker needs no strip rule."""
    rating = json.dumps(RATING)
    prefix = chr(60) + "|channel>thought ... <channel" + chr(124) + ">"
    text = prefix + "\n" + rating + "\n" + '{"ignore": "me"}'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_response(content=text))

    raw = _provider(handler, monkeypatch).complete(
        _item(study_config), LlmSettings(json_mode=False)
    )
    parsed = parse_response(raw.text, 300, raw.stop_reason)
    assert parsed.valid
    assert (parsed.low, parsed.likely, parsed.high) == (-1, 0, 1)


def test_minimax_reasoning_content_lands_in_the_record(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reasoning = "Let me think about Alpha. " * 12
    text = "\n\n\n" + json.dumps(RATING)
    usage = {
        "prompt_tokens": 11,
        "completion_tokens": 40,
        "completion_tokens_details": {"reasoning_tokens": 25},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_response(content=text, usage=usage, reasoning=reasoning))

    provider = _provider(handler, monkeypatch)
    runner = _runner(
        week="2026-W20",
        registry=_litellm_registry(model_registry),
        cfg=study_config,
        provider=provider,
        tickers=1,
        tmp_path=pipeline_dir,
        concurrency=1,
    )
    summary = runner.run_open_weight("stub-ow-a")
    assert summary.n_invalid == 0
    (record,) = _records("2026-W20")
    assert record["valid"] and record["fallback_flag"] is None
    assert record["reasoning_text"] == reasoning
    assert record["reasoning_chars"] == len(reasoning)
    assert record["tokens_reasoning"] == 25
    assert record["finish_reason"] == "stop"
    assert record["usage"] == usage


def test_empty_content_with_finish_reason_length_is_invalid_truncated(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_response(content="", finish_reason="length"))

    provider = _provider(handler, monkeypatch)
    runner = _runner(
        week="2026-W20",
        registry=_litellm_registry(model_registry),
        cfg=study_config,
        provider=provider,
        tickers=1,
        tmp_path=pipeline_dir,
        concurrency=1,
    )
    with caplog.at_level("WARNING"):
        runner.run_open_weight("stub-ow-a")
    (record,) = _records("2026-W20")
    assert record["valid"] is False and record["fallback_flag"] == INVALID_TRUNCATED
    warnings = [r for r in caplog.records if r.message == "truncated_empty_answer"]
    assert len(warnings) == 1
    fields = getattr(warnings[0], "extra_fields", {})
    assert fields["model"] == "stub-ow-a"
    assert (
        fields["max_output_tokens"]
        == settings_for_model(runner.cfg, model_registry.get("stub-ow-a")).max_output_tokens
    )


def test_per_model_max_tokens_and_cost_use_each_models_settings(
    study_config: StudyConfig, model_registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARCTIC_LITELLM_API_KEY", "test-key")
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_response(content=json.dumps(RATING)))

    for model_id, expected in (("gemma-4-31b", 300), ("minimax-m2.7", 2000)):
        spec = model_registry.get(model_id)
        settings = settings_for_model(study_config, spec)
        assert settings.max_output_tokens == expected
        provider = LiteLLMProvider(
            spec.id,
            spec.provider_model,
            "https://gateway.example/v1/",
            "ARCTIC_LITELLM_API_KEY",
            transport=httpx.MockTransport(handler),
            timeout=settings.request_timeout_seconds,
        )
        provider.complete(_item(study_config), settings)
        assert seen[-1]["max_tokens"] == expected
        est = estimate_week_cost([_item(study_config)], spec, settings, CostTable(model_registry))
        assert est["tokens_out"] == float(expected)


def test_per_model_timeout_reaches_the_httpx_client(
    study_config: StudyConfig, model_registry: ModelRegistry
) -> None:
    for model_id, expected in (("gemma-4-31b", 180), ("minimax-m2.7", 600)):
        spec = model_registry.get(model_id)
        timeout = settings_for_model(study_config, spec).request_timeout_seconds
        provider = LiteLLMProvider(
            spec.id,
            spec.provider_model,
            "https://gateway.example/v1/",
            "ARCTIC_LITELLM_API_KEY",
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_response())),
            timeout=timeout,
        )
        assert timeout == expected
        assert provider._client.timeout == httpx.Timeout(expected)  # noqa: SLF001


def test_429_is_retried_and_honors_retry_after(
    study_config: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "5"}, json={"error": "slow"})
        return httpx.Response(200, json=_response(content=json.dumps(RATING)))

    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))
    raw = _provider(handler, monkeypatch).complete(
        _item(study_config), LlmSettings(json_mode=False)
    )
    assert calls == 2 and raw.error is None
    assert sleeps == [5.0]


@pytest.mark.parametrize("status", [400, 429])
def test_budget_errors_never_retry(
    study_config: StudyConfig, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": {"message": "ExceededTokenBudget"}})

    with pytest.raises(LlmBudgetError):
        _provider(handler, monkeypatch).complete(_item(study_config), LlmSettings())
    assert calls == 1


def test_budget_marker_in_a_200_response_body_is_a_normal_answer(
    study_config: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The marker check only applies to non-2xx responses; rationale text may say anything."""
    rating = dict(RATING)
    rating["rationale"] = "cash budget_exceeded after buybacks"
    text = json.dumps(rating)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_response(content=text))

    raw = _provider(handler, monkeypatch).complete(
        _item(study_config), LlmSettings(json_mode=False)
    )
    parsed = parse_response(raw.text, 300, raw.stop_reason)
    assert raw.error is None and parsed.valid


def test_budget_error_aborts_the_model_with_budget_exceeded(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": {"message": "budget_exceeded: team allocation spent"}}
        )

    provider = _provider(handler, monkeypatch)
    runner = _runner(
        week="2026-W20",
        registry=_litellm_registry(model_registry),
        cfg=study_config,
        provider=provider,
        tickers=3,
        tmp_path=pipeline_dir,
        concurrency=1,
    )
    with caplog.at_level("ERROR"):
        summary = runner.run_open_weight("stub-ow-a")
    assert summary.aborted == "budget_exceeded"
    assert summary.n_provider_error == 0
    assert _records("2026-W20") == []
    assert any(r.message == "gateway_budget_exceeded" for r in caplog.records)


def test_500_is_retried_then_becomes_a_provider_error_record(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal"})

    monkeypatch.setattr("time.sleep", lambda seconds: None)
    provider = _provider(handler, monkeypatch)
    runner = _runner(
        week="2026-W20",
        registry=_litellm_registry(model_registry),
        cfg=study_config,
        provider=provider,
        tickers=1,
        tmp_path=pipeline_dir,
        concurrency=1,
    )
    summary = runner.run_open_weight("stub-ow-a")
    assert summary.n_provider_error == 1
    (record,) = _records("2026-W20")
    assert record["valid"] is False and record["fallback_flag"] == "provider_error"
    assert record["tokens_out"] == 0 and record["error"]


def test_other_4xx_does_not_retry_and_becomes_provider_error(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(422, json={"error": {"message": "weird"}})

    provider = _provider(handler, monkeypatch)
    runner = _runner(
        week="2026-W20",
        registry=_litellm_registry(model_registry),
        cfg=study_config,
        provider=provider,
        tickers=1,
        tmp_path=pipeline_dir,
        concurrency=1,
    )
    summary = runner.run_open_weight("stub-ow-a")
    assert calls == 1 and summary.n_provider_retries == 0 and summary.n_provider_error == 1
    (record,) = _records("2026-W20")
    assert record["fallback_flag"] == "provider_error" and "422" in str(record["error"])


def test_one_failing_request_leaves_a_complete_source_and_j6_is_normal(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N5: 1 failure in 10 requests is one provider_error record; the source completes and its
    10% invalid rate stays under the 15% threshold."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "T009" in json.dumps(body):
            return httpx.Response(500, json={"error": "internal"})
        return httpx.Response(200, json=_response(content=json.dumps(RATING)))

    monkeypatch.setattr("time.sleep", lambda seconds: None)
    provider = _provider(handler, monkeypatch)
    runner = _runner(
        week="2026-W20",
        registry=_litellm_registry(model_registry),
        cfg=study_config,
        provider=provider,
        tickers=10,
        tmp_path=pipeline_dir,
        concurrency=2,
    )
    summary = runner.run_open_weight("stub-ow-a")
    assert summary.n_provider_error == 1 and summary.n_invalid == 1
    records = _records("2026-W20")
    assert len(records) == 10
    assert sum(1 for r in records if r["fallback_flag"] == "provider_error") == 1
    assessed = runner.assess_run("stub-ow-a", "bare")
    assert assessed.state == "ok" and abs(assessed.invalid_rate - 0.1) < 1e-9


def test_j4_fails_while_a_request_has_no_record_and_a_rerun_completes_it(
    pipeline_root: Path,
    tmp_path: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world,
    inject_week: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run cut off before a record is written leaves the model unfinished: J4 fails naming
    it, and a rerun resumes from the saved records."""
    import shutil

    import portstudy.pipeline.jobs as jobs_mod
    from portstudy.pipeline.jobs import run_named
    from portstudy.pipeline.status import read_status
    from tests.conftest import generous_budget
    from tests.integration.test_phase10_pipeline import make_ctx, run_week

    target = tmp_path / "data_late"
    shutil.copytree(pipeline_root, target, ignore=shutil.ignore_patterns("backups"))
    monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(target))
    ctx = make_ctx(generous_budget(study_config), model_registry, world, inject_week)
    run_week(ctx, inject_week, ["J1", "J2", "J3"])

    from portstudy.llm.providers.stub import StubLlmProvider
    from portstudy.llm.runner import _CallCutoff

    class Cutoff(StubLlmProvider):
        """Simulates the lock cutoff: the runner treats _CallCutoff as an unrecorded request."""

        def complete(self, item: BatchItem, settings) -> RawResult:  # type: ignore[override]
            raise _CallCutoff("J6 cutoff reached before provider call")

    real = jobs_mod._week_runner

    def late_runner(c, w):  # type: ignore[no-untyped-def]
        runner = real(c, w)
        runner._providers["stub-ow-a"] = Cutoff("stub-ow-a")
        return runner

    monkeypatch.setattr(jobs_mod, "_week_runner", late_runner)
    with pytest.raises(JobFailed):
        run_named(ctx, "J4", inject_week)
    assert read_status(inject_week, "J4").state == "failed"

    monkeypatch.setattr(jobs_mod, "_week_runner", real)
    run_named(ctx, "J4", inject_week)
    assert read_status(inject_week, "J4").state == "done"


def test_run_weekend_continues_to_j6_when_j4_fails_at_the_lock_margin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(tmp_path / "data"))
    calls: list[str] = []

    def fake_run(ctx, job, w, s):  # type: ignore[no-untyped-def]
        calls.append(job)
        if job == "J4":
            raise JobFailed("J4", w, "open-weight run left requests without records")
        return "done"

    cfg = run_weekend.default_context(synthetic=True).cfg
    margin = lock_deadline("2026-W05", cfg) - dt.timedelta(minutes=cfg.pipeline.lock_margin_minutes)
    monkeypatch.setattr(run_weekend, "step1_catch_up", lambda ctx, w, s: None)
    monkeypatch.setattr(run_weekend, "run", fake_run)
    monkeypatch.setattr(run_weekend, "gateway_health", lambda ctx, w: None)
    monkeypatch.setattr(run_weekend, "wait_for_batches", lambda ctx, w, s, p: None)
    monkeypatch.setattr(run_weekend, "print_summary", lambda w: None)

    class Ctx:
        def __init__(self, now):
            self._now = now
            self.cfg = cfg

        def now(self):  # type: ignore[no-untyped-def]
            return self._now

    monkeypatch.setattr(
        run_weekend, "default_context", lambda **kw: Ctx(margin + dt.timedelta(minutes=1))
    )
    assert (
        run_weekend.main(["--synthetic", "--week", "2026-W05", "--from", "J4", "--until", "J6"])
        == 0
    )
    # J5's wait runs through wait_for_batches (stubbed separately), so run() sees J4 then J6.
    assert calls == ["J4", "J6"]

    calls.clear()
    monkeypatch.setattr(
        run_weekend, "default_context", lambda **kw: Ctx(margin - dt.timedelta(minutes=30))
    )
    assert (
        run_weekend.main(["--synthetic", "--week", "2026-W05", "--from", "J4", "--until", "J6"])
        == 1
    )
    assert calls == ["J4"]


def test_frozen_features_raise_on_rejection_otherwise_disable(
    study_config: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = json.dumps(
        {"error": {"message": "the parameter seed is not supported by this deployment"}}
    )
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(400, content=body)
        return httpx.Response(200, json=_response(content=json.dumps(RATING)))

    frozen = _provider(handler, monkeypatch, features_frozen=True)
    with pytest.raises(LlmRequestRejected, match="frozen feature seed"):
        frozen.complete(_item(study_config), LlmSettings())
    assert frozen.send_seed in (None, True)

    calls.clear()
    unfrozen = _provider(handler, monkeypatch)
    raw = unfrozen.complete(_item(study_config), LlmSettings())
    assert raw.error is None
    assert unfrozen.send_seed is False
    assert len(calls) == 2 and "seed" not in calls[1]


def test_unrelated_400_echoing_the_request_does_not_disable_features(
    study_config: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        echoed = json.loads(request.content)
        body = json.dumps({"error": {"message": "context length exceeded", "request": echoed}})
        return httpx.Response(400, content=body)

    provider = _provider(handler, monkeypatch)
    with pytest.raises(LlmRequestRejected):
        provider.complete(_item(study_config), LlmSettings())
    assert calls == 1
    assert provider.send_seed in (None, True) and provider.json_mode in (None, True)


def test_model_mismatch_and_gateway_fallback_mark_records_invalid(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mismatch(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_response(content=json.dumps(RATING), model="other/model"))

    provider = _provider(mismatch, monkeypatch)
    runner = _runner(
        week="2026-W20",
        registry=_litellm_registry(model_registry),
        cfg=study_config,
        provider=provider,
        tickers=1,
        tmp_path=pipeline_dir,
        concurrency=1,
    )
    runner.run_open_weight("stub-ow-a")
    (record,) = _records("2026-W20")
    assert record["valid"] is False and "model_mismatch" in str(record["error"])

    def fallback(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_response(content=json.dumps(RATING)),
            headers={"x-litellm-attempted-fallbacks": "1"},
        )

    provider = _provider(fallback, monkeypatch)
    runner = _runner(
        week="2026-W21",
        registry=_litellm_registry(model_registry),
        cfg=study_config,
        provider=provider,
        tickers=1,
        tmp_path=pipeline_dir,
        concurrency=1,
    )
    runner.run_open_weight("stub-ow-a")
    (record,) = _records("2026-W21")
    assert record["valid"] is False and "gateway_fallback" in str(record["error"])

    def group(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_response(content=json.dumps(RATING)),
            headers={"x-litellm-model-group": "another/group"},
        )

    provider = _provider(group, monkeypatch)
    runner = _runner(
        week="2026-W22",
        registry=_litellm_registry(model_registry),
        cfg=study_config,
        provider=provider,
        tickers=1,
        tmp_path=pipeline_dir,
        concurrency=1,
    )
    runner.run_open_weight("stub-ow-a")
    (record,) = _records("2026-W22")
    assert record["valid"] is False and "gateway_fallback" in str(record["error"])


def test_gateway_headers_and_usage_reach_records_and_scores(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = {
        "x-litellm-model-id": "dep-7",
        "x-litellm-version": "1.2.3",
        "x-litellm-response-duration-ms": "812",
        "x-ratelimit-remaining-requests": "99",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_response(content=json.dumps(RATING)), headers=headers)

    provider = _provider(handler, monkeypatch)
    runner = _runner(
        week="2026-W20",
        registry=_litellm_registry(model_registry),
        cfg=study_config,
        provider=provider,
        tickers=1,
        tmp_path=pipeline_dir,
        concurrency=1,
    )
    runner.run_open_weight("stub-ow-a")
    (record,) = _records("2026-W20")
    assert record["deployment_id"] == "dep-7"
    assert record["gateway_version"] == "1.2.3"
    assert record["gateway_headers"]["x-ratelimit-remaining-requests"] == "99"
    assert record["gateway_headers"]["x-litellm-response-duration-ms"] == "812"
    assert record["usage"]["prompt_tokens"] == 11

    scores = runner.write_scores()
    row = scores.iloc[0]
    for column in (
        "reasoning_chars",
        "tokens_reasoning",
        "finish_reason",
        "deployment_id",
        "gateway_version",
        "model_version",
    ):
        assert column in scores.columns
    assert row["deployment_id"] == "dep-7" and row["gateway_version"] == "1.2.3"
    on_disk = read_scores("2026-W20")
    assert "deployment_id" in on_disk.columns


def test_rate_adhoc_bypasses_cache_and_record_files(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gets: list[str] = []
    puts: list[str] = []
    cache = LlmCache(pipeline_dir / "cache")
    monkeypatch.setattr(cache, "get", lambda key: gets.append(key) or None)
    monkeypatch.setattr(cache, "put", lambda key, raw: puts.append(key))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_response(content=json.dumps(RATING)))

    provider = _provider(handler, monkeypatch)
    runner = _runner(
        week="2026-W20",
        registry=_litellm_registry(model_registry),
        cfg=study_config,
        provider=provider,
        tickers=1,
        tmp_path=pipeline_dir,
        concurrency=1,
    )
    items = [
        BatchItem(
            f"stub-ow-a__bare__adhoc__T{index}",
            "stub-ow-a",
            "bare",
            "adhoc",
            f"T{index}",
            _item(study_config).prompt,
        )
        for index in range(3)
    ]
    records = runner.rate_adhoc("stub-ow-a", items)
    assert len(records) == 3 and all(r["valid"] for r in records)
    assert gets == [] and puts == []
    assert _records("2026-W20") == []


class _TimingProvider:
    def __init__(self) -> None:
        self.send_seed = True
        self.json_mode = True
        self.calls = 0

    def complete(self, item: BatchItem, settings: LlmSettings) -> RawResult:
        self.calls += 1
        return RawResult(
            custom_id=item.custom_id,
            text=json.dumps(RATING),
            tokens_in=11,
            tokens_out=7,
            tokens_reasoning=2,
            finish_reason="stop",
            send_seed=True,
            json_mode=True,
            deployment_id="dep",
            gateway_version="v1",
            usage={"prompt_tokens": 11, "completion_tokens": 7},
            gateway_headers={
                "x-litellm-response-duration-ms": "900",
                "x-ratelimit-limit-requests": "60",
            },
        )


def test_time_gateway_writes_timing_columns(
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stub = _TimingProvider()
    providers = {"gemma-4-31b": stub, "minimax-m2.7": stub}
    out = tmp_path / "gateway_timing.csv"
    frame = smoke_real_services.time_gateway(
        study_config,
        model_registry,
        4,
        out,
        providers=providers,  # type: ignore[arg-type]
    )
    assert out.exists() and len(frame) == 8
    written = frame  # returned frame equals the CSV
    for column in (
        "model",
        "prompt_id",
        "request_index",
        "client_seconds",
        "gateway_ms",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "finish_reason",
        "valid",
        "send_seed",
        "json_mode",
        "deployment_id",
        "gateway_version",
        "ratelimit",
    ):
        assert column in written.columns
    assert set(written["model"]) == {"gemma-4-31b", "minimax-m2.7"}
    assert (written["reasoning_tokens"] == 2).all()
    stdout = capsys.readouterr().out
    assert "gemma-4-31b" in stdout and "minimax-m2.7" in stdout
