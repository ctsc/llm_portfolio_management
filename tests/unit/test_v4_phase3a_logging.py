"""v4 phase 3A: logging, fallback ledger and durable progress (N6), ARCTIC usage (N7), the study
log, progress, interrupted runs and the audit, on the shared synthetic pipeline run."""

from __future__ import annotations

import ast
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from portstudy import calendar as cal
from portstudy.config import LlmSettings, ModelRegistry, StudyConfig
from portstudy.data.providers.synthetic import SyntheticWorld, synthetic_bundle
from portstudy.io import append_line_durable, write_csv_atomic
from portstudy.llm import usage
from portstudy.llm.prompts import load_manifest
from portstudy.llm.providers.base import BatchItem, LlmProvider, RawResult
from portstudy.llm.providers.litellm import LiteLLMProvider
from portstudy.logging_setup import get_logger, route_library_logs
from portstudy.optimize import read_weights_meta
from portstudy.paths import logs_dir, outputs_dir, week_output_dir
from portstudy.pipeline import fallback_ledger as fl
from portstudy.pipeline import status as status_mod
from portstudy.pipeline.audit import audit_week
from portstudy.pipeline.context import RunContext, synthetic_clock
from portstudy.pipeline.events import read_events
from portstudy.pipeline.jobs import run_named
from portstudy.pipeline.status import JobFailed, read_status, run_job, write_status
from portstudy.pipeline.study_log import read_study_log
from tests.conftest import generous_budget
from tests.unit.test_v4_phase2_gateway import FailingProvider, _ctx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_weekend  # noqa: E402


class CrashingProvider(LlmProvider):
    """Answers the health probe, then raises a non-provider error: the J4 worker crashes (not a
    provider_error)."""

    supports_batch = False

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, item: BatchItem, settings: LlmSettings) -> RawResult:
        self.calls += 1
        if self.calls == 1:
            text = '{"low": -1, "likely": 0, "high": 1, "rationale": "probe"}'
            return RawResult(custom_id=item.custom_id, text=text, endpoint="gateway")
        raise RuntimeError("boom inside the model worker")


def _run(ctx: RunContext, week: str, monkeypatch: pytest.MonkeyPatch, *extra: str) -> int:
    monkeypatch.setattr(run_weekend, "default_context", lambda **kw: ctx)
    return int(run_weekend.main(["--synthetic", "--week", week, *extra]))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text("utf-8").splitlines() if ln.strip()]


def _run_id_of_last_run() -> str:
    starts = [e for e in read_events("*") if e["kind"] == "run_start"]
    return str(sorted(starts, key=lambda e: e["ts"])[-1]["run_id"])


def test_run_id_on_every_line_status_record_and_meta(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ctx(generous_budget(study_config), model_registry, world, inject_week)
    assert _run(ctx, inject_week, monkeypatch) == 0
    run_id = _run_id_of_last_run()
    start = next(e["ts"] for e in read_events(run_id) if e["kind"] == "run_start")
    n_lines = 0
    for path in logs_dir().glob("*/*.jsonl"):
        if path.parent.name == "runs":
            continue
        for line in _jsonl(path):
            if line["ts"] >= start:
                assert line.get("run_id") == run_id, (path, line)
                n_lines += 1
    assert n_lines > 10
    for job in ("J1", "J2", "J3", "J4", "J6", "J7"):
        assert read_status(inject_week, job).run_id == run_id, job
    records = _jsonl(sorted((week_output_dir(inject_week) / "llm").glob("*.jsonl"))[0])
    assert {r["run_id"] for r in records} == {run_id}
    assert read_weights_meta(inject_week, "mvo__Q0")["run_id"] == run_id
    kinds = {e["kind"] for e in read_events(run_id)}
    assert {"run_start", "job_start", "job_end", "run_end"} <= kinds
    assert read_study_log().set_index("week_id").loc[inject_week, "run_id"] == run_id


def test_crash_leaves_traceback_and_run_summary(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = generous_budget(study_config)
    ctx = _ctx(cfg, model_registry, world, inject_week, {"stub-ow-a": CrashingProvider()})
    assert _run(ctx, inject_week, monkeypatch) == 1
    j4 = read_status(inject_week, "J4")
    assert j4.state == "failed" and j4.traceback and "boom inside the model worker" in j4.traceback
    crash_lines = [
        line
        for path in (logs_dir() / "llm").glob("*.jsonl")
        for line in _jsonl(path)
        if line["msg"] == "model_crashed"
    ]
    assert crash_lines and "boom inside the model worker" in crash_lines[-1]["traceback"]
    summary = (week_output_dir(inject_week) / "run_summary.md").read_text("utf-8")
    assert "stopped at: J4" in summary and "J4 failed" in summary
    run_id = _run_id_of_last_run()
    assert any(e["kind"] == "job_fail" and e["job"] == "J4" for e in read_events(run_id))
    assert read_study_log().set_index("week_id").loc[inject_week, "failed_job"] == "J4"


def test_solver_and_provider_errors_log_their_traceback(
    data_dir: Path, study_config: StudyConfig, caplog: pytest.LogCaptureFixture
) -> None:
    import numpy as np

    from portstudy.optimize.mean_variance import solve_mean_variance

    tickers = [f"T{i:02d}" for i in range(25)]
    sigma = pd.DataFrame(np.eye(25) * 0.04, index=tickers, columns=tickers)
    mu = pd.Series(np.linspace(0.005, 0.02, 25), index=tickers)
    good = study_config.optimizer.solvers[0]
    with caplog.at_level(logging.WARNING):
        solve_mean_variance(
            mu, sigma, 2.5, study_config.optimizer, solvers=["NO_SUCH_SOLVER", good]
        )
    hits = [r for r in caplog.records if r.getMessage() == "solver_error_next_solver"]
    assert hits and hits[0].exc_info is not None


def test_python_warnings_reach_the_library_log(data_dir: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        logging.captureWarnings(False)
        route_library_logs()
        logging.captureWarnings(True)
        get_logger("pipeline").info("start")
        warnings.warn("library warning for N6 (4)", UserWarning, stacklevel=1)
        logging.captureWarnings(False)
    lines = [line for path in (logs_dir() / "libraries").glob("*.jsonl") for line in _jsonl(path)]
    assert any("library warning for N6 (4)" in line["msg"] for line in lines)


def test_planted_fallback_appears_once_in_the_ledger(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = generous_budget(study_config)
    ctx = _ctx(cfg, model_registry, world, inject_week, {"stub-ow-a": FailingProvider()})
    assert _run(ctx, inject_week, monkeypatch) == 0
    ledger = fl.read(inject_week)
    j6 = ledger[(ledger["kind"] == "fallback_q0") & (ledger["job"] == "J6")]
    assert set(j6["source"]) == {f"stub-ow-a__{p}" for p in ctx.prompt_ids()}
    assert not j6.duplicated(["source"]).any()
    j7 = ledger[(ledger["kind"] == "fallback_q0") & (ledger["job"] == "J7")]
    assert len(j7) == len(cfg.strategies.enabled) * len(ctx.prompt_ids())
    before = len(ledger)
    forced = _ctx(cfg, model_registry, world, inject_week, {"stub-ow-a": FailingProvider()})
    forced.force = True
    run_named(forced, "J6", inject_week)
    run_named(forced, "J7", inject_week)
    assert len(fl.read(inject_week)) == before  # a rerun upserts, it never duplicates
    assert audit_week(ctx, inject_week) == []


def test_audit_passes_clean_and_fails_on_deleted_files(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ctx(generous_budget(study_config), model_registry, world, inject_week)
    assert _run(ctx, inject_week, monkeypatch) == 0
    assert audit_week(ctx, inject_week) == []
    record_file = sorted((week_output_dir(inject_week) / "llm").glob("*.jsonl"))[0]
    lines = record_file.read_text("utf-8").splitlines(keepends=True)
    record_file.write_text("".join(lines[1:]), encoding="utf-8")
    assert any("have no record" in p for p in audit_week(ctx, inject_week))
    (week_output_dir(inject_week) / "weights" / "mvo__Q0.csv").unlink()
    problems = audit_week(ctx, inject_week)
    assert any("mvo__Q0.csv missing" in p for p in problems)
    monkeypatch.setattr(sys, "argv", ["portstudy"])
    from typer.testing import CliRunner

    from portstudy.pipeline import cli

    monkeypatch.setattr(cli, "_context", lambda *a, **k: ctx)
    result = CliRunner().invoke(cli.app, ["audit", "--week", inject_week, "--synthetic"])
    assert result.exit_code == 1 and "problem" in result.output


def test_atomic_writes_and_jsonl_appends_fsync(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import portstudy.io as io_mod

    calls: list[int] = []
    real = io_mod.os.fsync
    monkeypatch.setattr(io_mod.os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    write_csv_atomic(pd.DataFrame({"a": [1]}), data_dir / "x.csv")
    n_csv = len(calls)
    append_line_durable(data_dir / "x.jsonl", '{"a": 1}')
    assert n_csv >= 1 and len(calls) > n_csv
    get_logger("pipeline").info("durable line")
    assert len(calls) > n_csv + 1


def test_disk_check_refuses_a_job(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(status_mod, "free_disk_gb", lambda path: 1.0)
    with pytest.raises(JobFailed, match="GB free"):
        run_job("2026-W10", "J1", "h", lambda: {"ok": True})
    st = read_status("2026-W10", "J1")
    assert st.state == "failed" and "at least 2 GB" in (st.error or "")


def test_interrupted_job_is_reported_and_resumed(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ctx = _ctx(generous_budget(study_config), model_registry, world, inject_week)
    assert _run(ctx, inject_week, monkeypatch, "--until", "J2") == 0
    st = read_status(inject_week, "J2")
    st.state = "running"
    write_status(st)
    assert _run(ctx, inject_week, monkeypatch) == 0
    assert "interrupted during J2" in capsys.readouterr().out
    run_id = _run_id_of_last_run()
    events = read_events(run_id)
    assert any(e["kind"] == "resume" and e["interrupted"] == ["J2"] for e in events)
    assert any(e["kind"] == "job_interrupted" and e["job"] == "J2" for e in events)
    assert read_status(inject_week, "J2").state == "done"


def test_study_log_one_row_per_week_survives_reruns(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = generous_budget(study_config)
    second = cal.next_week_id(inject_week)
    for week in (inject_week, second, inject_week):
        assert _run(_ctx(cfg, model_registry, world, week), week, monkeypatch) == 0
    log = read_study_log()
    assert list(log["week_id"]) == [inject_week, second]
    row = log.set_index("week_id").loc[second]
    assert row["J7_state"] == "done" and row["audit_ok"] == "True"
    assert row["config_hash"] and row["universe_size"]
    assert json.loads(row["sources"])  # per-source status and invalid rate
    progress = json.loads((week_output_dir(second) / "progress.json").read_text("utf-8"))
    assert progress["J7"]["done"] == progress["J7"]["total"] > 0


# N7 ------------------------------------------------------------------------------------------


def _arctic(cfg: StudyConfig, **update: Any) -> StudyConfig:
    return cfg.model_copy(update={"arctic": cfg.arctic.model_copy(update=update)})


def test_gpu_hour_projection_warns_and_keeps_running(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    end = cal.window_end_date(cal.week_range(inject_week, 20)[-1]).isoformat()
    cfg = _arctic(
        generous_budget(study_config),
        gpu_seconds_per_request={"stub-ow-a": 3.0, "stub-ow-b": 3.0},
        gpu_hours_allocated=20.0,
    )
    cfg = cfg.model_copy(update={"timing": cfg.timing.model_copy(update={"live_end_date": end})})
    ctx = _ctx(cfg, model_registry, world, inject_week)
    assert _run(ctx, inject_week, monkeypatch) == 0
    out = capsys.readouterr().out
    assert "projected ARCTIC GPU-hours" in out and ("warning:" in out or "ALERT:" in out)
    row = read_study_log().set_index("week_id").loc[inject_week]
    assert float(row["arctic_gpu_hours_est"]) > 0
    assert float(row["arctic_gpu_hours_projected"]) > float(row["arctic_gpu_hours_total"])


def test_cost_header_from_a_stub_transport_is_summed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = {
            "model": "google/gemma-4-31B-it",
            "choices": [
                {
                    "message": {"content": '{"low":0,"likely":1,"high":2,"rationale":"x"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        return httpx.Response(200, json=body, headers={"x-litellm-response-cost": "0.0125"})

    from portstudy.config import load_study_config
    from portstudy.llm.prompts import assemble_prompt

    cfg = load_study_config(REPO_ROOT / "config" / "study.yaml")
    provider = LiteLLMProvider(
        "gemma-4-31b",
        "google/gemma-4-31B-it",
        "http://gateway.test/v1",
        "PORTSTUDY_TEST_KEY",
        transport=httpx.MockTransport(handler),
    )
    prompt = assemble_prompt(
        "bare", {"ticker": "AAA", "name": "A Co", "sector": "Energy"}, "", cfg, load_manifest()
    )
    item = BatchItem(
        "gemma-4-31b__bare__2026-W10__AAA", "gemma-4-31b", "bare", "2026-W10", "AAA", prompt
    )
    import os

    os.environ["PORTSTUDY_TEST_KEY"] = "k"
    try:
        raw = provider.complete(item, LlmSettings())
    finally:
        os.environ.pop("PORTSTUDY_TEST_KEY", None)
    assert usage._cost({"gateway_headers": raw.gateway_headers}) == pytest.approx(0.0125)


def _plant_costs(week: str, n: int, cost: float) -> None:
    path = week_output_dir(week) / "llm" / "zz_cost_fixture.jsonl"
    for i in range(n):
        append_line_durable(
            path,
            json.dumps(
                {
                    "custom_id": f"c{i}",
                    "model": "stub-ow-a",
                    "endpoint": "gateway",
                    "from_cache": False,
                    "tokens_in": 10,
                    "gateway_headers": {"x-litellm-response-cost": str(cost)},
                }
            ),
        )


def test_litellm_budget_stops_before_j4_unless_allowed(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    previous = cal.prev_week_id(inject_week)
    _plant_costs(previous, 50, 0.01)  # spent 0.50; about 0.01 per request
    cfg = _arctic(generous_budget(study_config), litellm_budget_usd=5.0)
    strict = RunContext(
        cfg=cfg,
        registry=model_registry,
        manifest=load_manifest(),
        synthetic=True,
        allow_over_budget=False,
        data_providers=synthetic_bundle(world),
        clock=synthetic_clock(inject_week),
    )
    assert _run(strict, inject_week, monkeypatch) == 1
    out = capsys.readouterr().out
    assert "ask ARCTIC to raise the LiteLLM limit" in out
    assert read_status(inject_week, "J4").state == "pending"  # never started
    assert not list((week_output_dir(inject_week) / "llm").glob("stub-ow-*.jsonl"))
    allowed = _ctx(cfg, model_registry, world, inject_week)  # allow_over_budget=True
    assert _run(allowed, inject_week, monkeypatch) == 0
    assert "--allow-over-budget" in capsys.readouterr().out


def test_missing_cost_header_warns_and_skips(
    pipeline_dir: Path, study_config: StudyConfig, model_registry: ModelRegistry, inject_week: str
) -> None:
    check = usage.check_dollar_budget(study_config, model_registry, inject_week, 800)
    assert check.ok and "skipped" in check.message and check.estimate_usd is None
    assert not (outputs_dir() / "gateway_timing.csv").exists()


# Silent exception handlers -------------------------------------------------------------------

_LOG_NAMES = ("log", "logger", "warning", "exception", "error", "emit", "record", "say", "print")
# Silent handlers per file at the time N6 was built: they translate an error into a return value
# (a parse that falls back to None, an optional dependency, a value that is checked later). A
# new silent handler, or one more in any file, fails this test.
ALLOWED_SILENT = {
    "__init__.py": 1,
    "backup.py": 3,
    "calendar.py": 1,
    "data/snapshot.py": 1,
    "eval/data.py": 1,
    "eval/report.py": 1,
    "llm/cost.py": 1,
    "llm/prompts.py": 2,
    "llm/providers/litellm.py": 4,
    "llm/runner.py": 2,
    "llm/usage.py": 1,
    "llm/validate.py": 2,
    "optimize/strategies/cvar.py": 2,
    "optimize/strategies/mvo.py": 1,
    "optimize/strategies/resampled.py": 3,
    "optimize/strategies/robust.py": 2,
    "pipeline/fallback.py": 4,
    "pipeline/jobs.py": 2,
    "pipeline/keep_awake.py": 1,
    "pipeline/preflight.py": 6,
    "pipeline/study_log.py": 1,
}


def _silent_handlers() -> dict[str, int]:
    root = REPO_ROOT / "src" / "portstudy"
    out: dict[str, int] = {}
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text("utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            body = ast.Module(body=node.body, type_ignores=[])
            raises = any(isinstance(n, ast.Raise) for n in ast.walk(body))
            logs = any(
                isinstance(n, ast.Call)
                and any(name in ast.unparse(n.func).lower() for name in _LOG_NAMES)
                for n in ast.walk(body)
            )
            if not (raises or logs):
                key = path.relative_to(root).as_posix()
                out[key] = out.get(key, 0) + 1
    return out


def test_no_new_silent_exception_handlers() -> None:
    found = _silent_handlers()
    new = {k: v for k, v in found.items() if v > ALLOWED_SILENT.get(k, 0)}
    assert new == {}, f"except blocks without a log line or re-raise: {new}"
