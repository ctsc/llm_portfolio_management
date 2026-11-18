"""v4 phase 3A: canaries (G21), the pilot repeat check (G20), keep-awake, network outage
tolerance, and resume after a kill in J2, J4 and J7."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from portstudy.config import LlmSettings, ModelRegistry, StudyConfig
from portstudy.data.providers.synthetic import SyntheticWorld
from portstudy.hashing import sha256_file
from portstudy.llm import canary
from portstudy.llm.cache import LlmCache
from portstudy.llm.providers.base import BatchItem, LlmConnectionError, LlmProvider, RawResult
from portstudy.llm.providers.stub import StubLlmProvider
from portstudy.paths import outputs_dir, week_output_dir
from portstudy.pipeline import jobs as jobs_mod
from portstudy.pipeline import keep_awake
from portstudy.pipeline.jobs import _week_runner, run_named
from tests.conftest import generous_budget
from tests.integration.test_phase10_pipeline import make_ctx, run_week
from tests.unit.test_v4_phase3_desktop import _DeterministicPredictor, _forecaster, _ohlcv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_weekend  # noqa: E402


class Wrapped(LlmProvider):
    """Delegates to a stub and edits the result (shifted scores, identity fields)."""

    supports_batch = False

    def __init__(self, model: str, **edit: Any) -> None:
        self.inner = StubLlmProvider(model)
        self.edit = edit
        self.calls = 0

    def complete(self, item: BatchItem, settings: LlmSettings) -> RawResult:
        self.calls += 1
        raw = self.inner.complete(item, settings)
        shift = self.edit.get("shift", 0)
        if shift:
            obj = json.loads(raw.text[raw.text.index("{") : raw.text.rindex("}") + 1])
            for k in ("low", "likely", "high"):
                obj[k] = max(-5, min(5, int(obj[k]) + shift))
            obj["low"] = min(obj["low"], obj["likely"])
            obj["high"] = max(obj["high"], obj["likely"])
            raw = replace(raw, text=json.dumps(obj))
        fields = {k: v for k, v in self.edit.items() if k != "shift"}
        return replace(raw, **fields) if fields else raw


def _fetch(updated: str = "2026-01-01"):  # type: ignore[no-untyped-def]
    def get(path: str) -> dict[str, Any]:
        if path == "/models":
            return {"data": [{"id": "stub-ow-a"}]}
        return {"data": [{"model_name": "stub-ow-a", "model_info": {"updated_at": updated}}]}

    return get


def _canary_ctx(cfg, registry, world, week, provider):  # type: ignore[no-untyped-def]
    ctx = make_ctx(cfg, registry, world, week, llm_providers={"stub-ow-a": provider})
    return ctx


def test_canary_and_repeat_check(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = generous_budget(study_config)
    base = Wrapped("stub-ow-a", model_version="v1", deployment_id="d1", gateway_version="g1")
    ctx = _canary_ctx(cfg, model_registry, world, inject_week, base)
    run_week(ctx, inject_week, ["J1", "J2", "J3", "J4"])
    puts: list[str] = []
    monkeypatch.setattr(LlmCache, "put", lambda self, key, raw: puts.append(key))
    monkeypatch.setattr(LlmCache, "get", lambda self, key: pytest.fail("canary read the cache"))

    # G20: identical answers give agreement 1.0 and write the canary baseline.
    table = canary.repeat_check(ctx, inject_week, _week_runner(ctx, inject_week), _fetch())
    assert len(table) and (table["share_identical"] == 1.0).all()
    assert (outputs_dir() / "canary_baseline.csv").exists()
    assert not puts

    def check(provider: LlmProvider, updated: str = "2026-01-01") -> dict[str, Any]:
        c = _canary_ctx(cfg, model_registry, world, inject_week, provider)
        rows = canary.run_canary(c, inject_week, _week_runner(c, inject_week), _fetch(updated))
        return next(r for r in rows if r["model"] == "stub-ow-a")

    same = check(Wrapped("stub-ow-a", model_version="v1", deployment_id="d1", gateway_version="g1"))
    assert same["matches"] == same["n"] == 10 and not same["model_change"]
    moved = check(
        Wrapped("stub-ow-a", model_version="v2", deployment_id="d1", gateway_version="g1")
    )
    assert moved["model_change"] and "model_version" in moved["reasons"]
    redeployed = check(
        Wrapped("stub-ow-a", model_version="v1", deployment_id="d9", gateway_version="g1")
    )
    assert redeployed["model_change"] and "deployment_id" in redeployed["reasons"]
    gateway = check(
        Wrapped("stub-ow-a", model_version="v1", deployment_id="d1", gateway_version="g2"),
        updated="2026-02-02",
    )
    assert gateway["gateway_changed"] and not gateway["model_change"]
    noisy = check(
        Wrapped("stub-ow-a", shift=2, model_version="v1", deployment_id="d1", gateway_version="g1")
    )
    assert noisy["matches"] < noisy["min_matches"] and noisy["model_change"]
    assert not puts  # the canary never reaches the response cache
    checks = canary.read_checks(inject_week)
    assert set(checks["model"]) == {"stub-ow-a"}
    assert (week_output_dir(inject_week) / "canary" / "stub-ow-a.jsonl").exists()


def test_repeat_check_with_noise_lowers_agreement(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
) -> None:
    cfg = generous_budget(study_config)
    ctx = _canary_ctx(cfg, model_registry, world, inject_week, Wrapped("stub-ow-a"))
    run_week(ctx, inject_week, ["J1", "J2", "J3", "J4"])
    noisy = _canary_ctx(cfg, model_registry, world, inject_week, Wrapped("stub-ow-a", shift=1))
    table = canary.repeat_check(noisy, inject_week, _week_runner(noisy, inject_week), _fetch())
    assert (table["share_identical"] < 1.0).all()


def test_binomial_bound() -> None:
    assert canary.min_matches(10, 1.0) == 10
    assert canary.min_matches(10, 0.8) < 8
    assert canary.min_matches(0, 0.8) == 0


# Keep-awake ----------------------------------------------------------------------------------


def test_keep_awake_only_under_wsl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    version = tmp_path / "version"
    version.write_text("Linux 5.15 generic", encoding="utf-8")
    assert not keep_awake.is_wsl(version)
    version.write_text("Linux 5.15.153.1-microsoft-standard-WSL2", encoding="utf-8")
    assert keep_awake.is_wsl(version)
    monkeypatch.setattr(keep_awake, "is_wsl", lambda: False)
    assert keep_awake.start_keep_awake() is None


def test_keep_awake_is_stopped_after_a_failed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.test_v4_phase3_preflight import _stub_weekend

    calls = _stub_weekend(run_weekend, monkeypatch, synthetic=True)
    stopped: list[object] = []
    handle = object()
    monkeypatch.setattr(run_weekend, "start_keep_awake", lambda: handle)
    monkeypatch.setattr(run_weekend, "stop_keep_awake", lambda h: stopped.append(h))

    def boom(*a: object) -> str:
        raise RuntimeError("unexpected crash")

    monkeypatch.setattr(run_weekend, "run", boom)
    with pytest.raises(RuntimeError):
        run_weekend.main(["--synthetic", "--week", "2026-W05", "--from", "J3"])
    assert stopped == [handle] and calls[0] == "catch_up"


# Network outage tolerance --------------------------------------------------------------------


class Flaky(LlmProvider):
    supports_batch = False

    def __init__(self, model: str, n_fail: int) -> None:
        self.inner = StubLlmProvider(model)
        self.n_fail = n_fail
        self.calls = 0

    def complete(self, item: BatchItem, settings: LlmSettings) -> RawResult:
        self.calls += 1
        if self.calls <= self.n_fail:
            raise LlmConnectionError("LiteLLM endpoint unreachable: [Errno 101]")
        return self.inner.complete(item, settings)


def _outage_run(ctx, week: str, step_s: float):  # type: ignore[no-untyped-def]
    runner = _week_runner(ctx, week)
    clock = {"t": 0.0}
    runner._monotonic = lambda: clock["t"]
    runner._sleep = lambda s: clock.__setitem__("t", clock["t"] + step_s)
    runner.run_open_weight("stub-ow-a")
    return runner._read_records("stub-ow-a", ctx.prompt_ids()[0])


def test_short_outage_recovers_without_provider_errors(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
) -> None:
    cfg = generous_budget(study_config)
    ctx = make_ctx(cfg, model_registry, world, inject_week, {"stub-ow-a": Flaky("stub-ow-a", 6)})
    run_week(ctx, inject_week, ["J1"])
    records = _outage_run(ctx, inject_week, step_s=30.0)  # six waits, 3 minutes in total
    flags = [r["fallback_flag"] for r in records.values()]
    assert records and "provider_error" not in flags


def test_long_outage_marks_the_calls_failed(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
) -> None:
    cfg = generous_budget(study_config)
    always = Flaky("stub-ow-a", 10**9)
    ctx = make_ctx(cfg, model_registry, world, inject_week, {"stub-ow-a": always})
    run_week(ctx, inject_week, ["J1"])
    records = _outage_run(ctx, inject_week, step_s=3600.0)  # past the 60-minute window at once
    assert records and all(r["fallback_flag"] == "provider_error" for r in records.values())


# Kill and resume -----------------------------------------------------------------------------


class _DiesOnce(_DeterministicPredictor):
    def __init__(self, die_at_call: int) -> None:
        self.die_at = die_at_call
        self.calls = 0

    def predict_batch(self, frames, x_timestamps, y_timestamps, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == self.die_at:
            raise KeyboardInterrupt("killed mid-chunk")
        return super().predict_batch(frames, x_timestamps, y_timestamps, **kwargs)


def test_j2_resumes_from_the_last_finished_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tickers = [f"T{i}" for i in range(6)]
    data = _ohlcv(tickers)
    expected = _forecaster(_DeterministicPredictor(), 2, monkeypatch).forecast(
        data, horizon=2, n_samples=3, seed=7
    )
    dying = _DiesOnce(die_at_call=5)  # three samples per chunk: dies in the second chunk
    first = _forecaster(dying, 2, monkeypatch)
    first.checkpoint_dir = tmp_path / "chunks"
    with pytest.raises(KeyboardInterrupt):
        first.forecast(data, horizon=2, n_samples=3, seed=7)
    assert len(list((tmp_path / "chunks").glob("chunk_*.csv"))) == 1
    counter = _DiesOnce(die_at_call=-1)
    again = _forecaster(counter, 2, monkeypatch)
    again.checkpoint_dir = tmp_path / "chunks"
    result = again.forecast(data, horizon=2, n_samples=3, seed=7)
    pd.testing.assert_frame_equal(result, expected)
    assert counter.calls == 2 * 3  # chunk 0 is not recomputed


class _KilledAfter(LlmProvider):
    supports_batch = False

    def __init__(self, model: str, kill_after: int) -> None:
        self.inner = StubLlmProvider(model)
        self.kill_after = kill_after
        self.calls = 0

    def complete(self, item: BatchItem, settings: LlmSettings) -> RawResult:
        self.calls += 1
        if self.calls > self.kill_after:
            raise KeyboardInterrupt("killed mid-J4")
        return self.inner.complete(item, settings)


def test_j4_resumes_without_repeating_finished_calls(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
) -> None:
    cfg = generous_budget(study_config)
    cfg = cfg.model_copy(update={"llm": cfg.llm.model_copy(update={"parallel_models": False})})
    killer = _KilledAfter("stub-ow-a", 30)
    ctx = make_ctx(cfg, model_registry, world, inject_week, {"stub-ow-a": killer})
    run_week(ctx, inject_week, ["J1", "J2", "J3"])
    with pytest.raises(KeyboardInterrupt):
        run_named(ctx, "J4", inject_week)
    runner = _week_runner(ctx, inject_week)
    n_tasks = len(runner.plan_week("stub-ow-a"))
    left = runner.n_unrecorded("stub-ow-a")
    assert 0 < left < n_tasks
    counter = _KilledAfter("stub-ow-a", 10**9)
    resumed = make_ctx(cfg, model_registry, world, inject_week, {"stub-ow-a": counter})
    assert run_named(resumed, "J4", inject_week).state == "done"
    assert counter.calls == left  # no finished call is repeated
    path = week_output_dir(inject_week) / "llm"
    for f in path.glob("stub-ow-a__*.jsonl"):
        ids = [json.loads(ln)["custom_id"] for ln in f.read_text("utf-8").splitlines() if ln]
        assert len(ids) == len(set(ids))


def test_j7_keeps_finished_portfolios_after_a_kill(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = generous_budget(study_config)
    ctx = make_ctx(cfg, model_registry, world, inject_week)
    run_week(ctx, inject_week, ["J1", "J2", "J3", "J4", "J5", "J6"])
    real = jobs_mod.build_weights
    calls = {"n": 0}

    def dying(*a: Any, **k: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 6:
            raise KeyboardInterrupt("killed mid-J7")
        return real(*a, **k)

    monkeypatch.setattr(jobs_mod, "build_weights", dying)
    with pytest.raises(KeyboardInterrupt):
        run_named(ctx, "J7", inject_week)
    counted = {"n": 0}

    def counting(*a: Any, **k: Any) -> Any:
        counted["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(jobs_mod, "build_weights", counting)
    assert run_named(ctx, "J7", inject_week).state == "done"
    resumed_calls = counted["n"]
    folder = week_output_dir(inject_week) / "weights"
    resumed = {p.name: sha256_file(p) for p in folder.glob("*.csv")}
    counted["n"] = 0
    forced = make_ctx(cfg, model_registry, world, inject_week, force=True)
    assert run_named(forced, "J7", inject_week).state == "done"
    assert resumed_calls == counted["n"] - 5  # the five finished portfolios were kept
    assert resumed == {p.name: sha256_file(p) for p in folder.glob("*.csv")}


# Random kill (N6) ----------------------------------------------------------------------------


def _csv_hashes(root: Path, week: str) -> dict[str, str]:
    out = {}
    for path in sorted((root / "outputs" / week).rglob("*.csv")):
        out[path.relative_to(root).as_posix()] = sha256_file(path)
    for path in sorted((root / "portfolios").rglob("*.csv")):
        out[path.relative_to(root).as_posix()] = sha256_file(path)
    return out


def _weekend(root: Path, week: str, kill_at: int = 0) -> int:
    env = {**os.environ, "PORTSTUDY_DATA_DIR": str(root)}
    if kill_at:
        env["PORTSTUDY_TEST_KILL_AT_EVENT"] = str(kill_at)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_weekend.py"),
        "--synthetic",
        "--week",
        week,
        "--config",
        str(REPO_ROOT / "config" / "study_test.yaml"),
        "--no-keep-awake",
    ]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900).returncode


@pytest.mark.slow
def test_random_kills_equal_an_uninterrupted_run(tmp_path: Path, study_config: StudyConfig) -> None:
    week = study_config.synthetic.first_week
    world_cmd = [sys.executable, str(REPO_ROOT / "scripts" / "make_synthetic_world.py")]
    roots = {}
    for name in ("clean", "killed"):
        root = tmp_path / name
        root.mkdir()
        subprocess.run(world_cmd, env={**os.environ, "PORTSTUDY_DATA_DIR": str(root)}, check=True)
        roots[name] = root
    assert _weekend(roots["clean"], week) == 0
    n_events = sum(
        len(p.read_text("utf-8").splitlines()) for p in (roots["clean"] / "logs" / "runs").glob("*")
    )
    assert n_events >= 12, n_events
    # 20 kill points spread over the weekend's events (every event is a possible point).
    points = sorted(random.Random(20261117).sample(range(2, n_events), min(20, n_events - 2)))
    for point in points:
        code = _weekend(roots["killed"], week, kill_at=point)
        assert code in (0, 1, 137) or code < 0
    assert _weekend(roots["killed"], week) == 0
    assert _csv_hashes(roots["killed"], week) == _csv_hashes(roots["clean"], week)
    for f in (roots["killed"] / "outputs" / week / "llm").glob("*.jsonl"):
        ids = [json.loads(ln)["custom_id"] for ln in f.read_text("utf-8").splitlines() if ln]
        assert len(ids) == len(set(ids)), f.name
    events = [
        json.loads(ln)
        for p in (roots["killed"] / "logs" / "runs").glob("*.jsonl")
        for ln in p.read_text("utf-8").splitlines()
        if ln
    ]
    assert any(e["kind"] in ("job_interrupted", "resume") for e in events)
