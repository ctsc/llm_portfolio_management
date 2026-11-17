"""v4 phase 2: the ARCTIC layer is only an HTTP endpoint. A failed gateway health check marks
that model's sources failed and the week continues on Q0 (DECISION(D2)); there is no
retry-flag, batch-submission or second-endpoint path. The pipeline tests moved here from the
deleted v2/v3 ARCTIC test files keep their behavior unchanged."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from portstudy.backup import rsync_backup, rsync_command
from portstudy.config import GatewayConfig, LlmSettings, ModelRegistry, ModelSpec, StudyConfig
from portstudy.data.providers.synthetic import SyntheticWorld, synthetic_bundle
from portstudy.llm import health
from portstudy.llm.prompts import load_manifest
from portstudy.llm.providers.base import BatchItem, LlmProvider, RawResult
from portstudy.llm.runner import read_run_status, read_scores
from portstudy.optimize import read_weights_meta
from portstudy.paths import week_output_dir
from portstudy.pipeline import fallback as fb
from portstudy.pipeline.context import RunContext, synthetic_clock
from portstudy.pipeline.jobs import _check_j7_prerequisites, run_named
from portstudy.pipeline.status import JobFailed, read_status, status_path
from tests.conftest import generous_budget
from tests.integration.test_phase10_pipeline import make_ctx, run_week

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_weekend  # noqa: E402


def _parallel(cfg: StudyConfig, flag: bool) -> StudyConfig:
    return cfg.model_copy(update={"llm": cfg.llm.model_copy(update={"parallel_models": flag})})


def _records(week: str) -> dict[str, list[str]]:
    folder = Path(os.environ["PORTSTUDY_DATA_DIR"]) / "outputs" / week / "llm"
    out: dict[str, list[str]] = {}
    for path in sorted(folder.glob("*.jsonl")):
        lines = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        out[path.name] = sorted(json.dumps(r, sort_keys=True) for r in lines)
    return out


def test_j4_runs_while_j2_is_pending_and_j7_waits(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
) -> None:
    cfg = generous_budget(study_config)
    ctx = make_ctx(cfg, model_registry, world, inject_week)
    states = run_week(ctx, inject_week, ["J1", "J3", "J4"])
    assert states["J4"] == "done" and read_status(inject_week, "J2").state != "done"
    assert len(read_scores(inject_week)) > 0
    # J6 plans the hold because J2 never ran; J7 then writes it (review fix H2 path).
    assert run_named(ctx, "J6", inject_week).state == "done"
    plan = fb.read_plan(inject_week)
    assert plan is not None and plan.no_rebalance
    # Without the hold, J7 refuses while J2 is not done.
    live = fb.FallbackPlan(week_id=inject_week, statuses=dict(plan.statuses))
    with pytest.raises(RuntimeError, match="J2 .* not done; J7 waits"):
        _check_j7_prerequisites(ctx, inject_week, live)
    assert run_named(ctx, "J7", inject_week).state == "done"


def test_j7_refuses_without_j6_or_without_j4_marks(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
) -> None:
    cfg = generous_budget(study_config)
    ctx = make_ctx(cfg, model_registry, world, inject_week)
    run_week(ctx, inject_week, ["J1", "J2", "J3"])
    plan = fb.FallbackPlan(week_id=inject_week)
    with pytest.raises(RuntimeError, match="J6 .* not done"):
        _check_j7_prerequisites(ctx, inject_week, plan)
    fb.write_plan(plan)
    with pytest.raises(JobFailed):  # J7 itself refuses: J6 has no status yet
        run_named(ctx, "J7", inject_week)
    run_week(ctx, inject_week, ["J4", "J5", "J6"])
    done_plan = fb.read_plan(inject_week)
    assert done_plan is not None
    _check_j7_prerequisites(ctx, inject_week, done_plan)  # J2, J4, J6 done: no error
    # J4 not done but every open-weight source marked fallback: allowed.
    marked = fb.FallbackPlan(
        week_id=inject_week,
        statuses={
            f"{m}__{p}": "fallback" for m in ctx.open_weight_models() for p in ctx.prompt_ids()
        },
    )
    status_path(inject_week, "J4").unlink()
    _check_j7_prerequisites(ctx, inject_week, marked)
    with pytest.raises(RuntimeError, match="J4 .* not done"):
        _check_j7_prerequisites(ctx, inject_week, done_plan)


def test_parallel_j4_matches_sequential_and_a_crash_does_not_stop_the_other_model(
    pipeline_root: Path,
    tmp_path: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shutil

    import portstudy.pipeline.jobs as jobs_mod

    results: dict[bool, dict[str, Any]] = {}
    for flag in (False, True):
        target = tmp_path / f"data_{int(flag)}"
        shutil.copytree(pipeline_root, target, ignore=shutil.ignore_patterns("backups"))
        monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(target))
        cfg = _parallel(generous_budget(study_config), flag)
        ctx = make_ctx(cfg, model_registry, world, inject_week)
        states = run_week(ctx, inject_week, ["J1", "J2", "J3", "J4"])
        assert states["J4"] == "done"
        scores = read_scores(inject_week).sort_values(["model", "config_id", "ticker"])
        results[flag] = {
            "records": _records(inject_week),
            "scores": scores.drop(
                columns=[c for c in scores.columns if "time" in c or "latency" in c]
            )
            .fillna(-1.0)
            .to_dict("list"),
            "status": {k: (v.state, v.n_done) for k, v in read_run_status(inject_week).items()},
        }
    assert set(results[True]["records"]) == set(results[False]["records"])
    assert len(results[True]["records"]) >= 2
    assert results[True]["scores"] == results[False]["scores"]
    assert results[True]["status"] == results[False]["status"]

    # One model's worker raising: the other model's sources still finish and keep their records;
    # J4 then fails naming the crashed model, and a rerun resumes it from the cache.
    target = tmp_path / "data_crash"
    shutil.copytree(pipeline_root, target, ignore=shutil.ignore_patterns("backups"))
    monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(target))
    cfg = _parallel(generous_budget(study_config), True)
    ctx = make_ctx(cfg, model_registry, world, inject_week)
    real_runner = jobs_mod._week_runner

    class Crashing:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def run_open_weight(self, model: str) -> Any:
            if model == "stub-ow-a":
                raise RuntimeError("worker died")
            return self._inner.run_open_weight(model)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    monkeypatch.setattr(jobs_mod, "_week_runner", lambda c, w: Crashing(real_runner(c, w)))
    run_week(ctx, inject_week, ["J1", "J2", "J3"])
    with pytest.raises(JobFailed):
        run_named(ctx, "J4", inject_week)
    failed = read_status(inject_week, "J4")
    assert failed.state == "failed" and "stub-ow-a: RuntimeError: worker died" in (
        failed.error or ""
    )
    statuses = read_run_status(inject_week)
    for prompt_id in ctx.prompt_ids():
        assert f"stub-ow-a__{prompt_id}" not in statuses
        assert (
            statuses[f"stub-ow-b__{prompt_id}"].n_done
            == statuses[f"stub-ow-b__{prompt_id}"].n_tasks
        )
    monkeypatch.setattr(jobs_mod, "_week_runner", real_runner)
    assert run_named(ctx, "J4", inject_week).state == "done"
    run_week(ctx, inject_week, ["J5", "J6", "J7"])
    for strategy in cfg.strategies.enabled:
        assert fb.week_status_for(inject_week, f"{strategy}__stub-ow-a__bare") == "normal"
        assert fb.week_status_for(inject_week, f"{strategy}__stub-ow-b__bare") == "normal"


class FailingProvider(LlmProvider):
    supports_batch = False

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, item: BatchItem, settings: LlmSettings) -> RawResult:
        self.calls += 1
        raise ConnectionError("gateway unreachable")


def _ctx(
    cfg: StudyConfig,
    registry: ModelRegistry,
    world: SyntheticWorld,
    week: str,
    providers: dict[str, LlmProvider] | None = None,
) -> RunContext:
    return RunContext(
        cfg=cfg,
        registry=registry,
        manifest=load_manifest(),
        synthetic=True,
        allow_over_budget=True,
        data_providers=synthetic_bundle(world),
        llm_providers=providers or {},
        clock=synthetic_clock(week),
    )


def test_from_and_until_run_only_the_requested_jobs(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import portstudy.pipeline.jobs as jobs_mod

    cfg = generous_budget(study_config)
    ctx = _ctx(cfg, model_registry, world, inject_week)
    monkeypatch.setattr(run_weekend, "default_context", lambda **kw: ctx)
    run_named(ctx, "J1", inject_week)
    run_named(ctx, "J2", inject_week)
    called: list[str] = []
    real = dict(jobs_mod.JOBS)
    for name in real:
        monkeypatch.setitem(
            jobs_mod.JOBS, name, lambda c, w, _n=name: (called.append(_n), real[_n](c, w))[1]
        )
    assert (
        run_weekend.main(["--synthetic", "--week", inject_week, "--from", "J3", "--until", "J5"])
        == 0
    )
    assert called == ["J3", "J4", "J5"]
    out = capsys.readouterr().out
    assert "jobs this run: J3 J4 J5" in out and "gateway health check ok" in out
    called.clear()
    assert run_weekend.main(["--synthetic", "--week", inject_week, "--from", "J6"]) == 0
    assert called == ["J6", "J7", "J9"]
    for job in ("J1", "J2", "J3", "J4", "J5", "J6", "J7", "J9"):
        assert read_status(inject_week, job).state == "done", job
    assert run_weekend.jobs_between("J3", "J9") == ["J3", "J4", "J5", "J6", "J7", "J8", "J9"]
    with pytest.raises(SystemExit):
        run_weekend.jobs_between("J7", "J3")


def test_health_failure_marks_sources_and_the_week_continues(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = generous_budget(study_config)
    failing = FailingProvider()
    providers: dict[str, LlmProvider] = {"stub-ow-a": failing}
    ctx = _ctx(cfg, model_registry, world, inject_week, providers)
    monkeypatch.setattr(run_weekend, "default_context", lambda **kw: ctx)
    assert run_weekend.main(["--synthetic", "--week", inject_week]) == 0
    out = capsys.readouterr().out
    assert "gateway health check failed for stub-ow-a" in out and "--from J4" in out
    assert failing.calls == 1  # the health probe only; J4 skipped the model
    report = health.read_health(inject_week)
    assert report is not None and set(report.failed) == {"stub-ow-a"}
    assert "stub-ow-b" in report.skipped
    statuses = read_run_status(inject_week)
    for prompt_id in ctx.prompt_ids():
        assert statuses[f"stub-ow-a__{prompt_id}"].state == "failed"
        assert statuses[f"stub-ow-a__{prompt_id}"].reason == health.HEALTH_REASON
        assert statuses[f"stub-ow-b__{prompt_id}"].state == "ok"
    j4 = read_status(inject_week, "J4")
    assert j4.state == "done" and j4.result["stub-ow-a"] == {"skipped": health.HEALTH_REASON}
    for strategy in cfg.strategies.enabled:
        assert fb.week_status_for(inject_week, f"{strategy}__stub-ow-a__bare") == "fallback"
        assert fb.week_status_for(inject_week, f"{strategy}__stub-ow-b__bare") == "normal"


def test_gateway_outage_marks_j6_fallback_and_j7_serves_q0(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DECISION(D2): one down gateway model is marked failed at the health check, J4 skips it,
    J6 plans only its sources as fallback with reason gateway_health_check_failed, J7 writes the
    strategy's Q0 for those portfolios, and the other model's sources run normally."""
    cfg = generous_budget(study_config)
    ctx = _ctx(cfg, model_registry, world, inject_week, {"stub-ow-a": FailingProvider()})
    monkeypatch.setattr(run_weekend, "default_context", lambda **kw: ctx)
    assert run_weekend.main(["--synthetic", "--week", inject_week]) == 0
    j4 = read_status(inject_week, "J4")
    assert j4.state == "done" and j4.result["stub-ow-a"] == {"skipped": health.HEALTH_REASON}
    assert j4.result["stub-ow-b"].get("n_calls")
    plan = fb.read_plan(inject_week)
    assert plan is not None
    for strategy in cfg.strategies.enabled:
        q0 = pd.read_csv(
            week_output_dir(inject_week) / "weights" / f"{strategy}__Q0.csv"
        ).set_index("ticker")["weight"]
        for prompt_id in ctx.prompt_ids():
            key = f"{strategy}__stub-ow-a__{prompt_id}"
            assert fb.week_status_for(inject_week, key) == "fallback"
            meta = read_weights_meta(inject_week, key)
            assert meta["decision_source"] == "fallback_q0"
            assert meta["reason"] == health.HEALTH_REASON
            mine = pd.read_csv(week_output_dir(inject_week) / "weights" / f"{key}.csv").set_index(
                "ticker"
            )["weight"]
            pd.testing.assert_series_equal(mine, q0, check_names=False)
            other = f"{strategy}__stub-ow-b__{prompt_id}"
            assert fb.week_status_for(inject_week, other) == "normal"
            assert read_weights_meta(inject_week, other)["decision_source"] == "llm"
    # The gateway comes back before the lock: `--from J4` runs J4 again and the model recovers.
    ctx.llm_providers.pop("stub-ow-a")
    assert run_weekend.main(["--synthetic", "--week", inject_week, "--from", "J4"]) == 0
    j4 = read_status(inject_week, "J4")
    assert j4.state == "done" and j4.result["stub-ow-a"].get("n_calls")
    for strategy in cfg.strategies.enabled:
        for prompt_id in ctx.prompt_ids():
            key = f"{strategy}__stub-ow-a__{prompt_id}"
            assert fb.week_status_for(inject_week, key) == "normal"
            assert read_weights_meta(inject_week, key)["decision_source"] == "llm"


def test_rsync_command_and_failure_does_not_raise(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "2026-W10_20260301T000000"
    target.mkdir()
    zip_path = target.with_suffix(".zip")
    cmd = rsync_command(target, zip_path, "user@host:/backups/")
    assert (
        cmd[:3] == ["rsync", "-a", "--partial"] and cmd[-1] == f"user@host:/backups/{target.name}/"
    )
    assert str(zip_path) in cmd and f"{target}/" in cmd
    assert rsync_command(target, None, "dest") == [
        "rsync",
        "-a",
        "--partial",
        f"{target}/",
        f"dest/{target.name}/",
    ]
    ok = rsync_backup(target, None, "/nonexistent-rsync-destination/x", logging.getLogger("t"))
    assert ok is False and "backup rsync FAILED" in capsys.readouterr().out


def test_stub_models_are_skipped_by_the_health_check_without_injected_providers(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
) -> None:
    from portstudy.llm.providers.stub import StubLlmProvider
    from portstudy.pipeline.jobs import _week_runner

    ctx = _ctx(generous_budget(study_config), model_registry, world, inject_week)
    run_named(ctx, "J1", inject_week)
    report = health.gateway_health_check(ctx, inject_week, _week_runner(ctx, inject_week))
    assert report.healthy and set(report.skipped) == set(ctx.open_weight_models())
    ctx2 = _ctx(
        generous_budget(study_config),
        model_registry,
        world,
        inject_week,
        {"stub-ow-b": StubLlmProvider("stub-ow-b")},
    )
    report2 = health.gateway_health_check(ctx2, inject_week, _week_runner(ctx2, inject_week))
    assert report2.ok == ["stub-ow-b"] and report2.skipped == ["stub-ow-a"]


def test_removed_run_weekend_flags_are_rejected() -> None:
    for flag in (
        "--gateway" + "-fallback",
        "--check" + "-wall" + "time",
        "--poll-once",
    ):
        with pytest.raises(SystemExit):
            run_weekend.main(["--synthetic", "--week", "2026-W05", flag])


def test_removed_model_and_gateway_fields_are_rejected() -> None:
    base = {
        "id": "m",
        "display_name": "m",
        "kind": "open_weight",
        "provider": "stub",
        "provider_model": "stub",
    }
    for extra in (
        {"role": "primary"},
        {"fallback" + "_eligible": True},
        {"serving": "slu" + "rm" + "_fallback"},
    ):
        with pytest.raises(ValidationError):
            ModelSpec.model_validate({**base, **extra})
    with pytest.raises(ValidationError):
        GatewayConfig.model_validate({"fallback" + "_endpoint_file": "x.json"})


def test_removed_script_folder_is_gone() -> None:
    assert not (REPO_ROOT / "scripts" / "arctic").exists()
