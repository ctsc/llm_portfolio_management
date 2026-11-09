"""Phase 6 tests: validation, stub runs, caching, failure injection, cost guard, payloads."""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from portstudy.config import LlmSettings, ModelRegistry, StudyConfig
from portstudy.data.snapshot import Snapshot, read_snapshot
from portstudy.hashing import sha256_file
from portstudy.llm.cache import LlmCache
from portstudy.llm.cost import CostCapExceeded, CostTable, RunningCost, check_budget
from portstudy.llm.prompts import AssembledPrompt, assemble_prompt, load_manifest
from portstudy.llm.providers.anthropic_batch import AnthropicBatchProvider
from portstudy.llm.providers.base import BatchItem, make_custom_id
from portstudy.llm.providers.openai_batch import OpenAIBatchProvider
from portstudy.llm.providers.stub import StubCrash, StubFailures, StubLlmProvider
from portstudy.llm.runner import WeekRunner, lock_deadline, read_run_status, read_scores
from portstudy.llm.validate import parse_response

ET = ZoneInfo("America/New_York")
REPO_ROOT = Path(__file__).resolve().parents[2]


# Validation ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,valid,error",
    [
        ('{"low": 1, "likely": 3, "high": 5, "rationale": "ok"}', True, None),
        ('```json\n{"low": -5, "likely": -5, "high": 0, "rationale": "x"}\n```', True, None),
        ('Sure: {"low": -1, "likely": 0, "high": 1, "rationale": "y"} done', True, None),
        ('{"low": -6, "likely": 0, "high": 1, "rationale": "x"}', False, "out_of_range"),
        ('{"low": 0, "likely": 2.5, "high": 3, "rationale": "x"}', False, "not_integer"),
        ('{"low": true, "likely": 0, "high": 1, "rationale": "x"}', False, "not_integer"),
        ('{"low": 1, "likely": 0, "high": 2, "rationale": "x"}', False, "bad_order"),
        ('{"low": 0, "likely": 1, "high": "3", "rationale": "x"}', False, "not_integer"),
        ('{"low": 0, "likely": 1, "high": 2}', False, "missing_key"),
        ('{"low": 0, "likely": 1, "high": 2, "rationale": 7}', False, "rationale_not_string"),
        ("not json at all", False, "not_json"),
        ("", False, "empty"),
    ],
)
def test_parse_response_cases(raw: str, valid: bool, error: str | None) -> None:
    r = parse_response(raw)
    assert r.valid is valid
    assert r.error == error
    if not valid:
        assert r.fallback_flag == "invalid_output"


def test_truncated_answer_flagged_separately() -> None:
    r = parse_response('{"low": 0, "likely": 1, "high": 2, "rat', stop_reason="max_tokens")
    assert not r.valid and r.fallback_flag == "invalid_truncated"


def test_rationale_truncated_to_cap() -> None:
    long = " ".join(["word"] * 400)
    r = parse_response(
        json.dumps({"low": 0, "likely": 1, "high": 2, "rationale": long}),
        rationale_token_cap=20,
    )
    assert r.valid and len(r.rationale) // 4 <= 20


# Fixtures -----------------------------------------------------------------------------------


@pytest.fixture()
def week_snapshot(work_data_dir: Path, session_weeks: list[str]) -> tuple[str, Snapshot]:
    w = session_weeks[2]
    return w, read_snapshot(w)


def _runner(
    week: str,
    snap: Snapshot,
    cfg: StudyConfig,
    registry: ModelRegistry,
    work_data_dir: Path,
    providers: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> WeekRunner:
    return WeekRunner(
        week,
        cfg,
        registry,
        snap,
        synthetic=True,
        providers=providers,
        cache=LlmCache(work_data_dir / "cache" / "llm"),
        logger=logger,
    )


def _prompts_fingerprint() -> dict[str, str]:
    return {
        str(p.relative_to(REPO_ROOT)): sha256_file(p)
        for p in sorted((REPO_ROOT / "prompts").rglob("*"))
        if p.is_file()
    }


# Runner -------------------------------------------------------------------------------------


def test_stub_run_covers_every_task_once_and_second_run_is_cached(
    work_data_dir: Path,
    week_snapshot: tuple[str, Snapshot],
    study_config: StudyConfig,
    model_registry: ModelRegistry,
) -> None:
    week, snap = week_snapshot
    before = _prompts_fingerprint()
    runner = _runner(week, snap, study_config, model_registry, work_data_dir)
    tasks = runner.plan_week()
    n_models = len(model_registry.enabled(synthetic=True))
    n_conf = len(load_manifest().enabled())
    assert len(tasks) == n_models * n_conf * len(snap.included_tickers)
    assert len({t.item.custom_id for t in tasks}) == len(tasks)

    local = [m.id for m in model_registry.enabled(synthetic=True) if m.kind == "open_weight"]
    summary = runner.run_local(local[0])
    assert summary.n_calls == n_conf * len(snap.included_tickers)
    assert summary.n_cached == 0
    scores = read_scores(week)
    per_model = scores[scores["model"] == local[0]]
    assert len(per_model) == n_conf * len(snap.included_tickers)
    assert not per_model.duplicated(["model", "config_id", "ticker"]).any()
    assert per_model["valid"].all()
    assert per_model["low"].between(-5, 5).all()
    assert per_model["low"].le(per_model["likely"]).all()
    assert per_model["likely"].le(per_model["high"]).all()
    assert (per_model["width"] == per_model["high"] - per_model["low"]).all()
    assert {"extra_keys", "had_thinking", "endpoint"} <= set(per_model.columns)
    assert set(per_model["prompt_hash"].str.len()) == {64}

    # A fresh runner on a fresh output folder but the same cache is fully cached.
    for p in (work_data_dir / "outputs" / week / "llm").glob(f"{local[0]}__*.jsonl"):
        p.unlink()
    runner2 = _runner(week, snap, study_config, model_registry, work_data_dir)
    summary2 = runner2.run_local(local[0])
    assert summary2.n_calls == 0
    assert summary2.n_cached == n_conf * len(snap.included_tickers)
    scores2 = read_scores(week)
    pd.testing.assert_frame_equal(
        scores[scores["model"] == local[0]]
        .drop(columns=["cost_usd", "latency_s"])
        .reset_index(drop=True),
        scores2[scores2["model"] == local[0]]
        .drop(columns=["cost_usd", "latency_s"])
        .reset_index(drop=True),
    )
    # Running again with records in place is a no-op and never duplicates rows.
    runner2.run_local(local[0])
    assert len(read_scores(week)) == len(scores2)
    # The runner never writes into prompts/.
    assert _prompts_fingerprint() == before


def test_invalid_output_injection_counts_rates_and_retries_once(
    work_data_dir: Path,
    week_snapshot: tuple[str, Snapshot],
    study_config: StudyConfig,
    model_registry: ModelRegistry,
) -> None:
    week, snap = week_snapshot
    model = "stub-ow-a"
    bad = set(snap.included_tickers[:20])
    provider = StubLlmProvider(model, failures=StubFailures(always_invalid_tickers=bad))
    runner = _runner(
        week, snap, study_config, model_registry, work_data_dir, providers={model: provider}
    )
    summary = runner.run_local(model)
    n_conf = len(load_manifest().enabled())
    assert summary.n_invalid == 20 * n_conf
    assert summary.n_retries == 20 * n_conf
    scores = read_scores(week)
    inv = scores[(scores["model"] == model) & ~scores["valid"]]
    assert set(inv["ticker"]) == bad
    assert (inv[["low", "likely", "high"]] == 0).all().all()
    assert inv["width"].isna().all()
    assert (inv["fallback_flag"] == "invalid_output").all()
    status = runner.assess_run(model, "bare", now=dt.datetime(2026, 1, 10, 12, tzinfo=ET))
    assert status.invalid_rate == pytest.approx(20 / len(snap.included_tickers))
    assert status.state == "failed" and "invalid_rate" in status.reason
    # Below the threshold the run is ok.
    provider_ok = StubLlmProvider(
        "stub-ow-b", failures=StubFailures(always_invalid_tickers=set(snap.included_tickers[:3]))
    )
    runner_ok = _runner(
        week,
        snap,
        study_config,
        model_registry,
        work_data_dir,
        providers={"stub-ow-b": provider_ok},
    )
    runner_ok.run_local("stub-ow-b")
    assert (
        runner_ok.assess_run("stub-ow-b", "bare", now=dt.datetime(2026, 1, 10, 12, tzinfo=ET)).state
        == "ok"
    )
    assert set(read_run_status(week)) >= {f"{model}__bare", "stub-ow-b__bare"}


def test_truncated_outputs_are_flagged(
    work_data_dir: Path,
    week_snapshot: tuple[str, Snapshot],
    study_config: StudyConfig,
    model_registry: ModelRegistry,
) -> None:
    week, snap = week_snapshot
    model = "stub-ow-a"
    provider = StubLlmProvider(model, failures=StubFailures(truncated_rate=1.0))
    runner = _runner(
        week, snap, study_config, model_registry, work_data_dir, providers={model: provider}
    )
    runner.run_local(model)
    scores = read_scores(week)
    assert (scores["fallback_flag"] == "invalid_truncated").all()


def test_crash_then_resume_does_not_repeat_calls(
    work_data_dir: Path,
    week_snapshot: tuple[str, Snapshot],
    study_config: StudyConfig,
    model_registry: ModelRegistry,
) -> None:
    week, snap = week_snapshot
    model = "stub-ow-a"
    crashing = StubLlmProvider(model, failures=StubFailures(crash_after_calls=37))
    runner = _runner(
        week, snap, study_config, model_registry, work_data_dir, providers={model: crashing}
    )
    with pytest.raises(StubCrash):
        runner.run_local(model)
    # Records are saved after each call; the CSV is compiled from them.
    partial = runner.write_scores()
    assert len(partial) == 37
    healthy = StubLlmProvider(model)
    runner2 = _runner(
        week, snap, study_config, model_registry, work_data_dir, providers={model: healthy}
    )
    summary = runner2.run_local(model)
    total = len(load_manifest().enabled()) * len(snap.included_tickers)
    assert healthy.calls == total - 37
    assert summary.n_calls == total - 37
    assert len(read_scores(week)) == total
    # The resumed run equals an uninterrupted run (stub is deterministic).
    clean_dir = work_data_dir / "clean"
    runner3 = WeekRunner(
        week,
        study_config,
        model_registry,
        snap,
        synthetic=True,
        providers={model: StubLlmProvider(model)},
        cache=LlmCache(clean_dir / "cache"),
    )
    import os

    os.environ["PORTSTUDY_DATA_DIR"] = str(clean_dir)
    try:
        runner3.folder = clean_dir / "outputs" / week / "llm"
        runner3.folder.mkdir(parents=True)
        runner3.run_local(model)
        clean = read_scores(week)
    finally:
        os.environ["PORTSTUDY_DATA_DIR"] = str(work_data_dir)
    merged = read_scores(week)
    cols = ["model", "config_id", "ticker", "low", "likely", "high", "width", "valid"]
    pd.testing.assert_frame_equal(
        merged[cols].reset_index(drop=True), clean[cols].reset_index(drop=True)
    )


def test_api_submit_collect_and_batch_never_returns(
    work_data_dir: Path,
    week_snapshot: tuple[str, Snapshot],
    study_config: StudyConfig,
    model_registry: ModelRegistry,
) -> None:
    week, snap = week_snapshot
    model = "stub-api-a"
    cfg = study_config.model_copy(
        update={
            "llm": study_config.llm.model_copy(update={"cost_goal_usd": 1e9, "cost_warn_usd": 1e9})
        }
    )
    provider = StubLlmProvider(model)
    runner = _runner(week, snap, cfg, model_registry, work_data_dir, providers={model: provider})
    handles = runner.submit_api(model)
    assert len(handles) == len(load_manifest().enabled())
    assert all(
        (work_data_dir / "outputs" / week / "llm" / f"batch_{model}__{h.config_id}.json").exists()
        for h in handles
    )
    # Submitting again reuses the stored handles instead of creating new batches.
    again = runner.submit_api(model)
    assert [h.batch_id for h in again] == [h.batch_id for h in handles]
    assert (
        runner.assess_run(model, "bare", now=dt.datetime(2026, 1, 10, 12, tzinfo=ET)).state
        == "incomplete"
    )
    summary = runner.collect_api(model)
    assert summary.n_calls == len(load_manifest().enabled()) * len(snap.included_tickers)
    assert summary.cost_usd > 0
    assert (
        runner.assess_run(model, "bare", now=dt.datetime(2026, 1, 10, 12, tzinfo=ET)).state == "ok"
    )
    cost = RunningCost(week).read()
    assert set(cost["model"]) == {model}
    assert cost["cost_usd"].sum() == pytest.approx(summary.cost_usd)

    # A batch that never returns fails at the lock deadline (N1).
    model_b = "stub-api-b"
    stuck = StubLlmProvider(model_b, failures=StubFailures(batch_never_returns=True))
    runner_b = _runner(week, snap, cfg, model_registry, work_data_dir, providers={model_b: stuck})
    runner_b.submit_api(model_b)
    runner_b.collect_api(model_b)
    before = lock_deadline(week, cfg) - dt.timedelta(hours=1)
    after = lock_deadline(week, cfg) + dt.timedelta(minutes=1)
    assert runner_b.assess_run(model_b, "bare", now=before).state == "incomplete"
    status = runner_b.assess_run(model_b, "bare", now=after)
    assert status.state == "failed" and status.reason == "batch_not_ready_by_lock"


def test_cost_guard_blocks_and_allow_over_budget_logs(
    work_data_dir: Path,
    week_snapshot: tuple[str, Snapshot],
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    week, snap = week_snapshot
    model = "stub-api-a"
    tight = study_config.model_copy(
        update={
            "llm": study_config.llm.model_copy(
                update={"cost_goal_usd": 0.01, "cost_warn_usd": 0.005}
            )
        }
    )
    logger = logging.getLogger("test.cost")
    runner = _runner(
        week,
        snap,
        tight,
        model_registry,
        work_data_dir,
        providers={model: StubLlmProvider(model)},
        logger=logger,
    )
    with pytest.raises(CostCapExceeded):
        runner.submit_api(model)
    assert not list((work_data_dir / "outputs" / week / "llm").glob("batch_*.json"))
    with caplog.at_level(logging.WARNING, logger="test.cost"):
        handles = runner.submit_api(model, allow_over_budget=True)
    assert handles
    assert any(r.msg == "over_budget_allowed" for r in caplog.records)
    assert any(r.msg == "cost_warning" for r in caplog.records)
    runner.collect_api(model)
    cost = RunningCost(week).read()
    assert cost["over_budget_allowed"].astype(str).str.lower().eq("true").all()


def test_check_budget_projection_uses_remaining_weeks(
    work_data_dir: Path, study_config: StudyConfig
) -> None:
    cfg = study_config.model_copy(
        update={
            "timing": study_config.timing.model_copy(update={"live_end_date": "2026-03-27"}),
            "llm": study_config.llm.model_copy(
                update={"cost_goal_usd": 100.0, "cost_warn_usd": 100.0}
            ),
        }
    )
    logger = logging.getLogger("test.budget")
    # 2026-W05 through 2026-W13 is 9 weeks; 10 USD a week projects to 90, under the goal.
    assert check_budget("2026-W05", 10.0, cfg, False, logger) == pytest.approx(90.0)
    with pytest.raises(CostCapExceeded):
        check_budget("2026-W05", 12.0, cfg, False, logger)


def test_cost_table_math(model_registry: ModelRegistry) -> None:
    table = CostTable(model_registry)
    # stub-api-a: 1.00 in, 0.10 cached, 5.00 out, 50 percent batch discount.
    usd = table.cost_usd(
        "stub-api-a", tokens_in=1_000_000, tokens_cached=500_000, tokens_out=100_000, batch=True
    )
    assert usd == pytest.approx((500_000 * 1.0 + 500_000 * 0.10 + 100_000 * 5.0) / 1e6 * 0.5)
    assert table.cost_usd("stub-ow-a", 10, 0, 10, False) == 0.0


# Provider payloads (mock transports) -----------------------------------------------------------


class FakeTransport:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def post_json(self, url: str, payload: Any, headers: dict[str, str]) -> Any:
        self.calls.append({"kind": "post_json", "url": url, "payload": payload, "headers": headers})
        return self.responses.pop(0)

    def get_json(self, url: str, headers: dict[str, str]) -> Any:
        self.calls.append({"kind": "get_json", "url": url, "headers": headers})
        return self.responses.pop(0)

    def get_text(self, url: str, headers: dict[str, str]) -> str:
        self.calls.append({"kind": "get_text", "url": url, "headers": headers})
        return str(self.responses.pop(0))

    def post_file(
        self, url: str, filename: str, content: bytes, data: dict[str, str], headers: dict[str, str]
    ) -> Any:
        self.calls.append(
            {
                "kind": "post_file",
                "url": url,
                "filename": filename,
                "content": content,
                "data": data,
                "headers": headers,
            }
        )
        return self.responses.pop(0)


def _item(study_config: StudyConfig, model: str = "m", config_id: str = "skills_1") -> BatchItem:
    prompt: AssembledPrompt = assemble_prompt(
        config_id,
        {"ticker": "AAA", "name": "Alpha", "sector": "Energy"},
        "Alpha beats estimates.",
        study_config,
    )
    return BatchItem(
        make_custom_id(model, config_id, "2026-W05", "AAA"),
        model,
        config_id,
        "2026-W05",
        "AAA",
        prompt,
    )


def test_stub_ranges_are_deterministic_and_reflect_prompt_noise(study_config: StudyConfig) -> None:
    provider = StubLlmProvider("stub", seed=29)
    bare = _item(study_config, config_id="bare")
    skills = _item(study_config, config_id="skills_1")
    bare_widths = []
    skill_widths = []
    for index in range(500):
        ticker = f"T{index}"
        bare_item = replace(bare, custom_id=f"bare-{ticker}", ticker=ticker)
        skills_item = replace(skills, custom_id=f"skills-{ticker}", ticker=ticker)
        bare_answer = provider.answer_for(bare_item)
        assert provider.answer_for(bare_item) == bare_answer
        assert bare_answer[0] <= bare_answer[1] <= bare_answer[2]
        skill_answer = provider.answer_for(skills_item)
        assert provider.answer_for(skills_item) == skill_answer
        assert skill_answer[0] <= skill_answer[1] <= skill_answer[2]
        bare_widths.append(bare_answer[2] - bare_answer[0])
        skill_widths.append(skill_answer[2] - skill_answer[0])
    assert pd.Series(skill_widths).mean() < pd.Series(bare_widths).mean()


def _no_tools(payload: Any) -> None:
    text = json.dumps(payload).lower()
    for banned in ('"tools"', "web_search", "tool_choice", "function"):
        assert banned not in text, banned


def test_anthropic_batch_payload_and_collect(study_config: StudyConfig) -> None:
    item = _item(study_config)
    results_jsonl = (
        json.dumps(
            {
                "custom_id": item.custom_id,
                "result": {
                    "type": "succeeded",
                    "message": {
                        "model": "claude-haiku-5-5-20260101",
                        "stop_reason": "end_turn",
                        "content": [
                            {
                                "type": "text",
                                "text": '{"low": 1, "likely": 2, "high": 4, "rationale": "r"}',
                            }
                        ],
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 30,
                            "cache_read_input_tokens": 400,
                            "cache_creation_input_tokens": 0,
                        },
                    },
                },
            }
        )
        + "\n"
    )
    transport = FakeTransport(
        [
            {"id": "msgbatch_1", "processing_status": "in_progress"},
            {
                "id": "msgbatch_1",
                "processing_status": "in_progress",
                "request_counts": {"succeeded": 0},
            },
            {
                "id": "msgbatch_1",
                "processing_status": "ended",
                "results_url": "https://x/results",
                "request_counts": {"succeeded": 1, "errored": 0, "expired": 0},
            },
            results_jsonl,
        ]
    )
    provider = AnthropicBatchProvider(
        "claude-haiku-5.5", "claude-haiku-5-5", transport=transport, api_key="k"
    )
    settings = LlmSettings(temperature=0.0, max_output_tokens=300)
    handle = provider.submit_batch([item], settings)
    payload = transport.calls[0]["payload"]
    _no_tools(payload)
    req = payload["requests"][0]
    assert req["custom_id"] == item.custom_id
    assert req["params"]["temperature"] == 0.0 and req["params"]["max_tokens"] == 300
    assert req["params"]["system"][0]["text"] == item.prompt.shared_prefix
    assert req["params"]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["params"]["messages"][0]["content"] == item.prompt.stock_part
    assert transport.calls[0]["headers"]["x-api-key"] == "k"
    assert provider.poll_batch(handle).state == "running"
    assert provider.poll_batch(handle).state == "done"
    out = provider.collect_batch(handle)
    assert (
        out[0].custom_id == item.custom_id
        and out[0].tokens_cached == 400
        and out[0].tokens_in == 500
    )
    assert out[0].model_version == "claude-haiku-5-5-20260101"


def test_openai_batch_payload_and_collect(study_config: StudyConfig) -> None:
    item = _item(study_config)
    output_jsonl = (
        json.dumps(
            {
                "custom_id": item.custom_id,
                "response": {
                    "status_code": 200,
                    "body": {
                        "model": "gpt-6-luna-2026-01-01",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": '{"low": -2, "likely": -1, "high": 0, "rationale": "r"}'
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 600,
                            "completion_tokens": 25,
                            "prompt_tokens_details": {"cached_tokens": 512},
                        },
                    },
                },
                "error": None,
            }
        )
        + "\n"
    )
    transport = FakeTransport(
        [
            {"id": "file_1"},
            {"id": "batch_1", "status": "validating"},
            {
                "id": "batch_1",
                "status": "in_progress",
                "request_counts": {"completed": 0, "failed": 0},
            },
            {
                "id": "batch_1",
                "status": "completed",
                "output_file_id": "file_out",
                "request_counts": {"completed": 1, "failed": 0},
            },
            output_jsonl,
        ]
    )
    provider = OpenAIBatchProvider("gpt-6-luna", "gpt-6-luna", transport=transport, api_key="k")
    settings = LlmSettings(temperature=0.0, max_output_tokens=300, reasoning="none", seed=1234)
    handle = provider.submit_batch([item], settings)
    upload = transport.calls[0]
    assert upload["kind"] == "post_file" and upload["data"] == {"purpose": "batch"}
    line = json.loads(upload["content"].decode("utf-8").splitlines()[0])
    _no_tools(line)
    assert line["custom_id"] == item.custom_id and line["url"] == "/v1/chat/completions"
    body = line["body"]
    assert body["temperature"] == 0.0 and body["reasoning_effort"] == "none"
    assert body["messages"][0]["content"] == item.prompt.shared_prefix
    assert body["messages"][1]["content"] == item.prompt.stock_part
    assert transport.calls[1]["payload"]["input_file_id"] == "file_1"
    assert provider.poll_batch(handle).state == "running"
    assert provider.poll_batch(handle).state == "done"
    out = provider.collect_batch(handle)
    assert (
        out[0].tokens_cached == 512
        and out[0].tokens_in == 600
        and out[0].model_version == "gpt-6-luna-2026-01-01"
    )


def test_full_synthetic_week_all_models(
    work_data_dir: Path,
    week_snapshot: tuple[str, Snapshot],
    study_config: StudyConfig,
    model_registry: ModelRegistry,
) -> None:
    week, snap = week_snapshot
    cfg = study_config.model_copy(
        update={
            "llm": study_config.llm.model_copy(update={"cost_goal_usd": 1e9, "cost_warn_usd": 1e9})
        }
    )
    runner = _runner(week, snap, cfg, model_registry, work_data_dir)
    for spec in model_registry.enabled(synthetic=True):
        if spec.kind == "open_weight":
            runner.run_local(spec.id)
        else:
            runner.submit_api(spec.id)
            runner.collect_api(spec.id)
        for conf in load_manifest().enabled():
            assert (
                runner.assess_run(
                    spec.id, conf.id, now=dt.datetime(2026, 1, 10, 12, tzinfo=ET)
                ).state
                == "ok"
            )
    scores = read_scores(week)
    n_models = len(model_registry.enabled(synthetic=True))
    assert len(scores) == n_models * len(load_manifest().enabled()) * len(snap.included_tickers)
    out = work_data_dir / "outputs" / week
    assert (out / "llm_scores.csv").exists() and (out / "run_status.json").exists()
    assert (out / "cost_log.csv").exists()
    assert len(read_run_status(week)) == n_models * len(load_manifest().enabled())
    cost = RunningCost(week).read()
    assert len(cost) == n_models * len(load_manifest().enabled())
    # Skills configuration recovers the planted effect at least as well as bare on average.
    from portstudy.llm.providers.stub import sentiment_from_text

    tasks = {t.item.custom_id: t for t in runner.plan_week()}
    scores["net"] = [
        sentiment_from_text(tasks[make_custom_id(m, c, week, t)].item.prompt.full_text)
        for m, c, t in zip(scores["model"], scores["config_id"], scores["ticker"])
    ]
    corr = scores.groupby("config_id").apply(lambda g: g["likely"].corr(g["net"]))
    assert corr["skills_1"] > corr["bare"] > 0.3
