"""v3 phase 3 (PLAN3): J4 without the J2 gate, parallel models, J7 prerequisites, `--poll-once`,
the finish job's resubmission, and the walltime guard in common.sh."""

from __future__ import annotations

import datetime as dt
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from portstudy.config import ModelRegistry, StudyConfig
from portstudy.data.providers.synthetic import SyntheticWorld
from portstudy.llm.providers.stub import StubFailures, StubLlmProvider
from portstudy.llm.runner import lock_deadline, read_run_status, read_scores
from portstudy.pipeline import fallback as fb
from portstudy.pipeline.jobs import _check_j7_prerequisites, run_named
from portstudy.pipeline.status import JobFailed, read_status, status_path
from tests.conftest import generous_budget
from tests.integration.test_phase10_pipeline import make_ctx, run_week, with_api_models

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCTIC = REPO_ROOT / "scripts" / "arctic"
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


def test_poll_once_exit_codes(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = with_api_models(generous_budget(study_config))
    stuck = StubLlmProvider("stub-api-a", failures=StubFailures(batch_never_returns=True))
    ctx = make_ctx(cfg, model_registry, world, inject_week, llm_providers={"stub-api-a": stuck})
    monkeypatch.setattr(run_weekend, "default_context", lambda **kw: ctx)
    run_week(ctx, inject_week, ["J1", "J2", "J3", "J4"])
    args = ["--synthetic", "--week", inject_week, "--from", "J5", "--poll-once"]
    assert run_weekend.main([*args, "--until", "J5"]) == run_weekend.POLL_AGAIN == 3
    assert read_status(inject_week, "J5").state == "pending"
    # At the lock margin the run goes on: J6 labels the pending batch as a fallback (N1).
    margin = lock_deadline(inject_week, cfg) - dt.timedelta(
        minutes=cfg.pipeline.lock_margin_minutes - 1
    )
    late = make_ctx(
        cfg,
        model_registry,
        world,
        inject_week,
        llm_providers={"stub-api-a": stuck},
        clock=lambda: margin,
    )
    monkeypatch.setattr(run_weekend, "default_context", lambda **kw: late)
    assert run_weekend.main([*args, "--until", "J7"]) == 0
    assert read_status(inject_week, "J7").state == "done"
    for strategy in cfg.strategies.enabled:
        assert fb.week_status_for(inject_week, f"{strategy}__stub-api-a__bare") == "fallback"
    # Everything collected: exit 0.
    ok = make_ctx(cfg, model_registry, world, inject_week)
    monkeypatch.setattr(run_weekend, "default_context", lambda **kw: ok)
    monkeypatch.setattr(run_weekend, "run", lambda c, j, w, s: "done")
    monkeypatch.setattr(run_weekend, "step1_catch_up", lambda c, w, s: None)
    assert run_weekend.main([*args, "--until", "J5"]) == 0


def test_check_walltime_limits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["--synthetic", "--week", "2026-W05", "--dry-run", "--check-walltime"]
    assert run_weekend.main([*base, "16:00:00"]) == 2
    assert "exceeds the 15 hour limit" in capsys.readouterr().out
    assert run_weekend.main([*base, "13:00:00"]) == 0
    assert "above the 12 hour target" in capsys.readouterr().out
    assert run_weekend.main([*base, "03:00:00"]) == 0
    assert "hour" not in capsys.readouterr().out.split("walltime")[0]


def _env_file(tmp_path: Path, **overrides: str) -> Path:
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True, exist_ok=True)
    (venv / "activate").write_text("# stub venv\n")
    values = {
        "SLURM_ACCOUNT": "acct",
        "GPU_PARTITION": "gpu",
        "CPU_PARTITION": "cpu",
        "GPU_GRES": "gpu:1",
        "GPU_MEM": "32G",
        "KRONOS_WALLTIME": "01:00:00",
        "LLM_WALLTIME": "06:00:00",
        "FINISH_WALLTIME": "03:00:00",
        "POLL_INTERVAL_MINUTES": "7",
        "SHUFFLE_WALLTIME": "02:00:00",
        "MAX_JOB_HOURS": "15",
        "PYTHON_MODULE": "python/3.11",
        "VENV_PATH": str(tmp_path / "venv"),
        "DATA_ROOT": str(tmp_path / "data"),
        "HF_HOME": str(tmp_path / "hf"),
        "ARCTIC_NETWORK_MODE": "login",
        "SHUFFLE_MAX_PARALLEL": "10",
        "LLM_FALLBACK_WALLTIME": "10:00:00",
    }
    values.update(overrides)
    path = tmp_path / "arctic.env"
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    return path


def _stub_bin(tmp_path: Path, stubs: dict[str, str]) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in stubs.items():
        path = bin_dir / name
        path.write_text(f"#!/usr/bin/env bash\n{body}\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _run(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60, check=False)


def test_common_sh_walltime_guard(tmp_path: Path) -> None:
    env_file = _env_file(tmp_path, LLM_WALLTIME="16:00:00", FINISH_WALLTIME="13:00:00")
    env = {**os.environ, "ARCTIC_ENV": str(env_file)}
    script = f'source "{ARCTIC / "common.sh"}"; check_walltime "$1"'
    rejected = _run(["bash", "-c", script, "_", "LLM_WALLTIME"], env)
    assert rejected.returncode == 2 and "exceeds the 15 hour limit" in rejected.stderr
    warned = _run(["bash", "-c", script, "_", "FINISH_WALLTIME"], env)
    assert warned.returncode == 0 and "above the 12 hour target" in warned.stderr
    fine = _run(["bash", "-c", script, "_", "KRONOS_WALLTIME"], env)
    assert fine.returncode == 0 and fine.stderr == ""
    days = _run(["bash", "-c", script.replace('"$1"', "X"), "_"], {**env, "X": "1-00:00:00"})
    assert days.returncode == 2


def test_submit_weekend_three_job_layout_in_both_modes(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    bin_dir = _stub_bin(
        tmp_path,
        {
            "sbatch": 'echo "sbatch $*" >> "$CALLS"; n=$(grep -c sbatch "$CALLS"); echo "$((4000 + n))"',
            "portstudy": 'echo "portstudy $*" >> "$CALLS"',
            "python": 'echo "python $*" >> "$CALLS"',
        },
    )
    for mode in ("login", "compute"):
        log.write_text("")
        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "ARCTIC_ENV": str(_env_file(tmp_path, ARCTIC_NETWORK_MODE=mode)),
            "CALLS": str(log),
        }
        result = _run(["bash", str(ARCTIC / "submit_weekend.sh"), "--week", "2026-W05"], env)
        assert result.returncode == 0, result.stdout + result.stderr
        calls = log.read_text().splitlines()
        assert calls[0].startswith(
            "python scripts/run_weekend.py --dry-run --check-walltime 06:00:00"
        )
        if mode == "login":
            assert calls[1] == "portstudy run-job J1 --week 2026-W05"
            assert calls[2] == "python scripts/run_weekend.py --from J3 --until J3 --week 2026-W05"
            sbatches = calls[3:]
        else:
            sbatches = calls[1:]
        assert len(sbatches) == 3
        assert sbatches[0].endswith("kronos.sbatch") and "--mem=32G" in sbatches[0]
        assert "--partition=gpu" in sbatches[0] and "--time=01:00:00" in sbatches[0]
        assert sbatches[1].endswith("llm.sbatch")
        assert "--cpus-per-task=2 --mem-per-cpu=3G --time=06:00:00" in sbatches[1]
        assert "--dependency" not in sbatches[1]
        assert sbatches[2].endswith("finish.sbatch")
        assert "--cpus-per-task=1 --mem-per-cpu=3G --time=03:00:00" in sbatches[2]
        assert "--dependency=afterany:4001:4002" in sbatches[2]
        assert "afterany" in result.stdout and "sacct -j 4003" in result.stdout
        # Dry run prints the same layout without calling anything.
        dry = _run(
            ["bash", str(ARCTIC / "submit_weekend.sh"), "--week", "2026-W05", "--dry-run"], env
        )
        assert dry.returncode == 0 and dry.stdout.count("+ sbatch") == 0
        assert dry.stderr.count("+ sbatch") == 3 and "--mem-per-cpu=3G" in dry.stderr
    assert not (ARCTIC / "weekend.sbatch").exists()


def test_finish_sbatch_resubmits_on_pending_and_continues_on_collected(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    bin_dir = _stub_bin(
        tmp_path,
        {
            "sbatch": 'echo "sbatch $*" >> "$CALLS"; echo 5151',
            "python": 'echo "python $*" >> "$CALLS"; if [[ "$*" == *--poll-once* ]]; then exit "${POLL_EXIT:-0}"; fi',
        },
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "ARCTIC_ENV": str(_env_file(tmp_path)),
        "CALLS": str(log),
        "PORTSTUDY_WEEK": "2026-W05",
        "PORTSTUDY_FINISH_FLAGS": "--account=acct --partition=cpu --cpus-per-task=1 --mem-per-cpu=3G --time=03:00:00",
    }
    pending = _run(["bash", str(ARCTIC / "finish.sbatch")], {**env, "POLL_EXIT": "3"})
    assert pending.returncode == 0, pending.stdout + pending.stderr
    calls = log.read_text().splitlines()
    assert calls == [
        "python scripts/run_weekend.py --from J5 --until J5 --poll-once --week 2026-W05",
        "sbatch --account=acct --partition=cpu --cpus-per-task=1 --mem-per-cpu=3G --time=03:00:00 "
        f"--begin=now+7minutes --export=ALL {ARCTIC / 'finish.sbatch'}",
    ]
    log.write_text("")
    collected = _run(["bash", str(ARCTIC / "finish.sbatch")], {**env, "POLL_EXIT": "0"})
    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert log.read_text().splitlines() == [
        "python scripts/run_weekend.py --from J5 --until J5 --poll-once --week 2026-W05",
        "python scripts/run_weekend.py --from J6 --until J9 --week 2026-W05",
    ]
    log.write_text("")
    failed = _run(["bash", str(ARCTIC / "finish.sbatch")], {**env, "POLL_EXIT": "1"})
    assert failed.returncode == 1 and "sbatch" not in log.read_text()
