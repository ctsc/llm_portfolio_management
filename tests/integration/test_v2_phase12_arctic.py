"""v2 phase 12 tests: shellcheck on the ARCTIC scripts, `run_weekend.py --from/--until`, the
gateway health check (marking without the flag, switching with it), a stale endpoint file, the
walltime check, and the rsync backup option."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from portstudy.backup import rsync_backup, rsync_command
from portstudy.config import GatewayConfig, LlmSettings, ModelRegistry, StudyConfig
from portstudy.data.providers.synthetic import SyntheticWorld, synthetic_bundle
from portstudy.llm import health
from portstudy.llm.prompts import load_manifest
from portstudy.llm.providers.base import BatchItem, LlmProvider, RawResult
from portstudy.llm.providers.factory import _fallback_base_url
from portstudy.llm.providers.stub import StubLlmProvider
from portstudy.llm.runner import read_run_status
from portstudy.pipeline import fallback as fb
from portstudy.pipeline.context import RunContext, synthetic_clock
from portstudy.pipeline.jobs import run_named
from portstudy.pipeline.status import read_status
from tests.conftest import generous_budget

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_weekend  # noqa: E402

SCRIPTS = sorted((REPO_ROOT / "scripts" / "arctic").glob("*.sh")) + sorted(
    (REPO_ROOT / "scripts" / "arctic").glob("*.sbatch")
)


def _shellcheck() -> str | None:
    found = shutil.which("shellcheck") or shutil.which(
        "shellcheck", path=str(Path.home() / ".local" / "bin")
    )
    if found:
        return found
    try:
        import shellcheck_py

        return str(shellcheck_py.SHELLCHECK_PATH)
    except ImportError:
        return None


def test_shellcheck_every_arctic_script() -> None:
    assert len(SCRIPTS) == 6, [p.name for p in SCRIPTS]
    tool = _shellcheck()
    if tool is None:
        pytest.skip("shellcheck is not installed (PROGRESS note)")
    result = subprocess.run(
        [tool, "-x", "-P", "SCRIPTDIR", *map(str, SCRIPTS)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_env_example_is_all_tbd_and_common_refuses_it(tmp_path: Path) -> None:
    example = REPO_ROOT / "scripts" / "arctic" / "arctic.env.example"
    values = [
        line.split("=", 1)[1].split("#")[0].strip()
        for line in example.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert values and all(v == "TBD" for v in values)
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "arctic" / "submit_weekend.sh"), "--dry-run"],
        capture_output=True,
        text=True,
        env={**os.environ, "ARCTIC_ENV": str(example)},
    )
    assert result.returncode == 2 and "docs/arctic_setup.md" in result.stderr


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
    gateway_fallback: bool = False,
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
        gateway_fallback=gateway_fallback,
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


def test_health_failure_without_flag_marks_sources_and_continues(
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
    assert "D2 fallback" in out and "--from J4 --gateway-fallback" in out
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


def test_fallback_flag_switches_eligible_models_only(
    tmp_path: Path, study_config: StudyConfig, model_registry: ModelRegistry, world: SyntheticWorld
) -> None:
    stubs = [
        m.model_copy(update={"fallback_eligible": m.id == "stub-ow-a"})
        for m in model_registry.stub_models
    ]
    registry = model_registry.model_copy(update={"stub_models": stubs})
    week = study_config.synthetic.first_week
    plain = _ctx(study_config, registry, world, week).run_registry()
    switched = _ctx(study_config, registry, world, week, gateway_fallback=True).run_registry()
    assert {m.id: m.serving for m in plain.stub_models if m.id.startswith("stub-ow")} == {
        "stub-ow-a": "gateway",
        "stub-ow-b": "gateway",
    }
    assert switched.get("stub-ow-a").serving == "slurm_fallback"
    assert switched.get("stub-ow-b").serving == "gateway"
    # A fresh endpoint file for this week is accepted; a stale one or another week is rejected.
    endpoint = tmp_path / "fallback_endpoint.json"
    gateway = GatewayConfig(base_url="http://gateway", fallback_endpoint_file=str(endpoint))
    started = dt.datetime.now(dt.UTC) - dt.timedelta(minutes=1)
    endpoint.write_text(json.dumps({"base_url": "http://node1:8000/v1", "week_id": week}))
    assert _fallback_base_url(gateway, week, started.isoformat()) == "http://node1:8000/v1"
    with pytest.raises(ValueError, match="not current week"):
        _fallback_base_url(gateway, "2030-W01", started.isoformat())
    old = time.time() - 3600
    os.utime(endpoint, (old, old))
    with pytest.raises(ValueError, match="stale"):
        _fallback_base_url(gateway, week, started.isoformat())


def test_walltime_parsing_and_warning(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert health.parse_walltime("12:00:00") == dt.timedelta(hours=12)
    assert health.parse_walltime("1-02:30") == dt.timedelta(days=1, hours=2, minutes=30)
    assert health.parse_walltime("45") == dt.timedelta(minutes=45)
    ctx = _ctx(generous_budget(study_config), model_registry, world, inject_week)
    monkeypatch.setattr(run_weekend, "default_context", lambda **kw: ctx)
    args = ["--synthetic", "--week", inject_week, "--dry-run"]
    assert run_weekend.main([*args, "--check-walltime", "7-00:00:00"]) == 0
    assert "warning: walltime 7-00:00:00 ends" in capsys.readouterr().out
    assert run_weekend.main([*args, "--check-walltime", "00:10:00"]) == 0
    assert "before the lock" in capsys.readouterr().out


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
