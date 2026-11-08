"""Regression tests for the issues found in the code review (results.md), one per fix."""

from __future__ import annotations

from pathlib import Path

import pytest

from portstudy import calendar as cal
from portstudy.config import StudyConfig
from portstudy.data import ledger
from portstudy.data.providers.synthetic import PLANTED, SyntheticWorld, synthetic_bundle
from portstudy.data.snapshot import make_snapshot, read_snapshot
from portstudy.data.universe import build_universe


def test_h1_ledger_holds_only_provider_rows_after_forward_fill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    study_config: StudyConfig,
    world: SyntheticWorld,
) -> None:
    """H1: forward-filled rows go to the snapshot for the models, never to the raw ledger."""
    monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(tmp_path / "data"))
    ticker, _, _ = PLANTED["halt"]
    raw = world.prices[world.prices["ticker"] == ticker]
    all_days = sorted(world.prices["date"].unique())
    missing = [d for d in all_days if d not in set(raw["date"])]
    assert missing, "the synthetic world plants a halt"
    week = next(
        w
        for w in cal.week_range(study_config.synthetic.first_week, 60)
        if cal.cutoff_day(w) > max(missing)
    )
    bundle = synthetic_bundle(world)
    universe = build_universe(cal.cutoff_day(week), study_config, bundle, print_table=False)
    make_snapshot(week, study_config, bundle, universe)
    snap = read_snapshot(week)
    filled = snap.prices[(snap.prices["ticker"] == ticker) & snap.prices["date"].isin(missing)]
    assert len(filled) == len(missing) and (filled["volume"] == 0).all()  # models see a full matrix
    led = ledger.read_ledger()
    in_ledger = led[(led["ticker"] == ticker) & led["date"].isin(missing)]
    assert in_ledger.empty, "fabricated rows must never reach the append-only ledger"
    assert not (led["volume"] == 0).any()


def test_h4_catch_up_runs_for_every_start_job_and_from_j8_runs_j8(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """H4: `--from J3`/`--from J6` (the ARCTIC flows) still run the J8/J9 catch-up, and
    `--from J8` really runs J8."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import run_weekend

    monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(tmp_path / "data"))
    calls: list[str] = []
    monkeypatch.setattr(run_weekend, "step1_catch_up", lambda ctx, w, s: calls.append("catch_up"))
    monkeypatch.setattr(run_weekend, "run", lambda ctx, job, w, s: calls.append(job) or "done")
    monkeypatch.setattr(run_weekend, "gateway_health", lambda ctx, w: calls.append("health"))
    monkeypatch.setattr(run_weekend, "wait_for_batches", lambda ctx, w, s, p: calls.append("J5"))
    monkeypatch.setattr(run_weekend, "print_summary", lambda w: None)

    class Ctx:
        cfg = run_weekend.default_context(synthetic=True).cfg
        gateway_fallback = False

        def now(self):  # type: ignore[no-untyped-def]
            return run_weekend.synthetic_clock("2026-W05")()

    monkeypatch.setattr(run_weekend, "default_context", lambda **kw: Ctx())
    assert run_weekend.main(["--synthetic", "--week", "2026-W05", "--from", "J6"]) == 0
    assert calls == ["catch_up", "J6", "J7", "J9"]
    calls.clear()
    assert run_weekend.main(["--synthetic", "--week", "2026-W05", "--from", "J3"]) == 0
    assert calls == ["catch_up", "J3", "health", "J4", "J5", "J6", "J7", "J9"]
    calls.clear()
    assert (
        run_weekend.main(["--synthetic", "--week", "2026-W05", "--from", "J8", "--until", "J8"])
        == 0
    )
    assert calls == ["catch_up", "J8"]
    calls.clear()
    assert run_weekend.main(["--synthetic", "--week", "2026-W05"]) == 0
    assert "J8" not in calls and calls[0] == "catch_up"


def test_h2_replay_weeks_fill_the_gap_and_a_j1_failure_writes_the_hold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """H2: a calendar week without a lock is still replayed (as no_rebalance), and a J1 failure in
    run_weekend.py still runs J6 and J7 so the hold is written."""
    import sys

    from portstudy.pipeline import jobs

    monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(jobs, "weeks_with_weights", lambda: ["2026-W10", "2026-W12"])
    assert jobs.replay_weeks("2026-W12") == ["2026-W10", "2026-W11", "2026-W12"]
    assert jobs.replay_weeks("2026-W10") == ["2026-W10"]
    monkeypatch.setattr(jobs, "weeks_with_weights", lambda: [])
    assert jobs.replay_weeks("2026-W12") == []

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import run_weekend
    from portstudy.pipeline.status import JobFailed

    calls: list[str] = []

    def fake_run(ctx, job, w, s):  # type: ignore[no-untyped-def]
        calls.append(job)
        if job == "J1":
            raise JobFailed("J1", w, "provider down")
        return "done"

    monkeypatch.setattr(run_weekend, "step1_catch_up", lambda ctx, w, s: None)
    monkeypatch.setattr(run_weekend, "run", fake_run)

    class Ctx:
        cfg = run_weekend.default_context(synthetic=True).cfg
        gateway_fallback = False

        def now(self):  # type: ignore[no-untyped-def]
            return run_weekend.synthetic_clock("2026-W05")()

    monkeypatch.setattr(run_weekend, "default_context", lambda **kw: Ctx())
    assert run_weekend.main(["--synthetic", "--week", "2026-W05"]) == 1
    assert calls == ["J1", "J6", "J7"]


def test_h3_login_mode_waits_for_kronos_before_j4(tmp_path: Path) -> None:
    """H3: in login mode the script runs J3, waits for the Kronos job to leave the queue, then
    J4 to J5, then submits the weekend job. Slurm and python are stubbed on PATH."""
    import os
    import stat
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    stubs = {
        "sbatch": 'echo "sbatch $*" >> "$CALLS"; echo 4242',
        "squeue": 'echo "squeue $*" >> "$CALLS"; n=$(grep -c squeue "$CALLS"); if [ "$n" -lt 3 ]; then echo "4242 R"; fi',
        "portstudy": 'echo "portstudy $*" >> "$CALLS"',
        "python": 'echo "python $*" >> "$CALLS"',
    }
    for name, body in stubs.items():
        path = bin_dir / name
        path.write_text(f"#!/usr/bin/env bash\n{body}\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "activate").write_text("# stub venv\n")
    env_file = tmp_path / "arctic.env"
    env_file.write_text(
        "\n".join(
            [
                "SLURM_ACCOUNT=acct",
                "GPU_PARTITION=gpu",
                "CPU_PARTITION=cpu",
                "GPU_GRES=gpu:1",
                "KRONOS_WALLTIME=01:00:00",
                "WEEKEND_WALLTIME=08:00:00",
                "SHUFFLE_WALLTIME=04:00:00",
                "LLM_FALLBACK_WALLTIME=12:00:00",
                "PYTHON_MODULE=python/3.11",
                f"VENV_PATH={tmp_path / 'venv'}",
                f"DATA_ROOT={tmp_path / 'data'}",
                f"HF_HOME={tmp_path / 'hf'}",
                "ARCTIC_NETWORK_MODE=login",
                "SHUFFLE_MAX_PARALLEL=10",
                "",
            ]
        )
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "ARCTIC_ENV": str(env_file),
        "CALLS": str(log),
        "WAIT_POLL_SECONDS": "0",
    }
    result = subprocess.run(
        ["bash", str(repo / "scripts" / "arctic" / "submit_weekend.sh"), "--week", "2026-W05"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [line for line in log.read_text().splitlines() if not line.startswith("squeue")]
    n_polls = sum(1 for line in log.read_text().splitlines() if line.startswith("squeue"))
    assert calls[0].startswith("python scripts/run_weekend.py --dry-run --check-walltime")
    assert calls[1] == "portstudy run-job J1 --week 2026-W05"
    assert calls[2].startswith("sbatch --parsable") and calls[2].endswith("kronos.sbatch")
    assert calls[3] == "python scripts/run_weekend.py --from J3 --until J3 --week 2026-W05"
    assert calls[4] == "python scripts/run_weekend.py --from J4 --until J5 --week 2026-W05"
    assert "--dependency=afterok:4242" in calls[5] and calls[5].endswith("weekend.sbatch")
    assert n_polls >= 3, "the script polled squeue until the Kronos job left the queue"
    # The J4 run happens only after the last poll.
    order = log.read_text().splitlines()
    assert order.index(calls[4]) > max(
        i for i, line in enumerate(order) if line.startswith("squeue")
    )


def test_m4_m10_current_week_is_dst_aware_and_rolls_at_the_close() -> None:
    """M4: ET is read with DST (a summer Friday 23:30 ET is still Friday); M10: the week rolls
    only at the cutoff close (16:00 ET, 13:00 on an early close)."""
    import datetime as dt
    from zoneinfo import ZoneInfo

    from portstudy.pipeline.jobs import current_week

    et = ZoneInfo("America/New_York")
    utc = dt.UTC
    # 2026-07-10 is a Friday (EDT, UTC-4). 23:30 EDT = 03:30 UTC Saturday.
    assert current_week(dt.datetime(2026, 7, 11, 3, 30, tzinfo=utc)) == "2026-W28"
    assert current_week(dt.datetime(2026, 7, 10, 23, 30, tzinfo=et)) == "2026-W28"
    # Before the close on the cutoff Friday the previous week is still current.
    assert current_week(dt.datetime(2026, 7, 10, 15, 59, tzinfo=et)) == "2026-W27"
    assert current_week(dt.datetime(2026, 7, 10, 16, 0, tzinfo=et)) == "2026-W28"
    assert current_week(dt.datetime(2026, 7, 11, 9, 0, tzinfo=et)) == "2026-W28"  # Saturday
    assert current_week(dt.datetime(2026, 7, 9, 18, 0, tzinfo=et)) == "2026-W27"  # Thursday
    # Early close 2026-11-27 (day after Thanksgiving): the week rolls at 13:00.
    assert current_week(dt.datetime(2026, 11, 27, 12, 59, tzinfo=et)) == "2026-W47"
    assert current_week(dt.datetime(2026, 11, 27, 13, 0, tzinfo=et)) == "2026-W48"
    # Good Friday 2026-04-03 is closed: the cutoff is Thursday, and Friday any time is the week.
    assert current_week(dt.datetime(2026, 4, 3, 9, 0, tzinfo=et)) == "2026-W14"
    # Naive datetimes are taken as ET.
    assert current_week(dt.datetime(2026, 7, 10, 15, 0)) == "2026-W27"


def test_m5_done_job_reruns_when_its_outputs_are_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M5: a `done` status with matching inputs is skipped only while the recorded outputs are
    intact; deleting or changing an output makes the job run again."""
    from portstudy.pipeline.status import run_job

    monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(tmp_path / "data"))
    out = tmp_path / "out.txt"
    runs: list[int] = []

    def fn() -> dict[str, int]:
        runs.append(1)
        out.write_text(f"run {len(runs)}")
        return {"n": len(runs)}

    first = run_job("2026-W05", "JX", "h1", fn, outputs=lambda: [out])
    assert first.state == "done" and first.outputs_hash
    assert run_job("2026-W05", "JX", "h1", fn, outputs=lambda: [out]).state == "skipped"
    out.unlink()
    rerun = run_job("2026-W05", "JX", "h1", fn, outputs=lambda: [out])
    assert rerun.state == "done" and len(runs) == 2 and out.exists()
    out.write_text("edited by hand")
    assert run_job("2026-W05", "JX", "h1", fn, outputs=lambda: [out]).state == "done"
    assert len(runs) == 3
    assert run_job("2026-W05", "JX", "h1", fn, outputs=lambda: [out]).state == "skipped"
    # Without an outputs callable the inputs hash alone still decides.
    assert run_job("2026-W05", "JY", "h1", lambda: None).state == "done"
    assert run_job("2026-W05", "JY", "h1", lambda: None).state == "skipped"


def test_m1_m2_provider_errors_are_requeued_and_auth_failures_abort_the_model(
    pipeline_dir: Path, tmp_path: Path, model_registry, study_config: StudyConfig
) -> None:
    """M1: a task whose provider call fails is queued again (up to MAX_PROVIDER_ERRORS) instead
    of being dropped; M2: a non-retryable authentication error stops the model's run."""
    from portstudy.llm.providers.base import BatchItem, LlmProviderError, RawResult
    from portstudy.llm.providers.litellm import LlmAuthenticationError
    from portstudy.llm.providers.stub import StubLlmProvider
    from portstudy.llm.runner import MAX_PROVIDER_ERRORS, read_run_status
    from tests.unit.test_v2_phase4_litellm import _runner

    class Flaky(StubLlmProvider):
        def __init__(self, failures_per_task: int, exc: type[LlmProviderError]) -> None:
            super().__init__("stub-ow-a")
            self.failures_per_task = failures_per_task
            self.exc = exc
            self.attempts: dict[str, int] = {}

        def complete(self, item: BatchItem, settings) -> RawResult:  # type: ignore[override]
            n = self.attempts.get(item.custom_id, 0) + 1
            self.attempts[item.custom_id] = n
            if n <= self.failures_per_task:
                raise self.exc("transport failure")
            return super().complete(item, settings)

    # Two failures per task, then success: every task completes with no drops.
    flaky = Flaky(MAX_PROVIDER_ERRORS, LlmProviderError)
    runner = _runner(
        week="2026-W20",
        registry=model_registry,
        cfg=study_config,
        provider=flaky,
        tickers=3,
        tmp_path=tmp_path,
        concurrency=2,
    )
    summary = runner.run_open_weight("stub-ow-a")
    assert summary.n_provider_retries == 3 * MAX_PROVIDER_ERRORS and summary.n_dropped == 0
    assert summary.aborted is None
    assert read_run_status("2026-W20")["stub-ow-a__bare"].n_done == 3
    assert all(n == MAX_PROVIDER_ERRORS + 1 for n in flaky.attempts.values())

    # Persistent failure: each task is dropped after MAX_PROVIDER_ERRORS requeues.
    broken = Flaky(10**6, LlmProviderError)
    runner = _runner(
        week="2026-W21",
        registry=model_registry,
        cfg=study_config,
        provider=broken,
        tickers=3,
        tmp_path=tmp_path,
        concurrency=2,
    )
    summary = runner.run_open_weight("stub-ow-a")
    assert summary.n_dropped == 3 and summary.n_provider_retries == 3 * MAX_PROVIDER_ERRORS
    assert all(n == MAX_PROVIDER_ERRORS + 1 for n in broken.attempts.values())
    assert "stub-ow-a__bare" not in read_run_status("2026-W21") or (
        read_run_status("2026-W21")["stub-ow-a__bare"].n_done == 0
    )

    # Authentication failure: no requeue, no further submissions, summary.aborted is set.
    unauth = Flaky(10**6, LlmAuthenticationError)
    runner = _runner(
        week="2026-W22",
        registry=model_registry,
        cfg=study_config,
        provider=unauth,
        tickers=6,
        tmp_path=tmp_path,
        concurrency=1,
    )
    summary = runner.run_open_weight("stub-ow-a")
    assert summary.aborted == "authentication_failed"
    assert summary.n_provider_retries == 0
    assert sum(unauth.attempts.values()) == 1, "the first 401 stops the model's run"


def test_m3_gateway_fallback_marks_non_eligible_failed_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M3: under --gateway-fallback only fallback-eligible models switch endpoints; the other
    failed gateway models are still marked failed (N1)."""
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import run_weekend
    from portstudy.llm import health

    report = health.HealthReport(
        week_id="2026-W05",
        checked_at="x",
        failed={"big-model": "HTTP 503", "small-model": "HTTP 503"},
    )
    marked: list[tuple[list[str], str]] = []
    monkeypatch.setattr(health, "gateway_health_check", lambda ctx, w, r: report)
    monkeypatch.setattr(
        health,
        "mark_sources_failed",
        lambda ctx, w, models, reason: marked.append((models, reason)) or models,
    )
    monkeypatch.setattr("portstudy.pipeline.jobs._week_runner", lambda ctx, w: None)
    specs = {
        "big-model": SimpleNamespace(fallback_eligible=False),
        "small-model": SimpleNamespace(fallback_eligible=True),
    }
    ctx = SimpleNamespace(gateway_fallback=True, registry=SimpleNamespace(get=lambda m: specs[m]))
    run_weekend.gateway_health(ctx, "2026-W05")  # type: ignore[arg-type]
    assert marked == [(["big-model"], health.HEALTH_REASON)]

    marked.clear()
    ctx = SimpleNamespace(gateway_fallback=False, registry=SimpleNamespace(get=lambda m: specs[m]))
    run_weekend.gateway_health(ctx, "2026-W05")  # type: ignore[arg-type]
    assert marked == [(["big-model", "small-model"], health.HEALTH_REASON)]


def test_m6_view_shift_is_measured_on_fallback_weights_too() -> None:
    """M6: a solver-fallback allocation that differs from Q0 reports its real view shift."""
    import pandas as pd

    from portstudy.optimize import _shift_from_q0

    q0 = pd.Series({"A": 0.5, "B": 0.5})
    written = pd.Series({"A": 0.2, "B": 0.5, "C": 0.3})
    assert _shift_from_q0(written, q0, q0_only=False, is_control=False) == pytest.approx(0.3)
    assert _shift_from_q0(written, q0, q0_only=True, is_control=False) == 0.0
    assert _shift_from_q0(written, q0, q0_only=False, is_control=True) == 0.0
    assert _shift_from_q0(written, None, q0_only=False, is_control=False) == 0.0


def test_m8_j7_inputs_hash_changes_with_the_ic_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M8: J7's inputs hash covers data/outputs/ic_history.csv, so a J8 update invalidates it."""
    from types import SimpleNamespace

    from portstudy.pipeline import jobs
    from portstudy.views.ic import ICStore

    monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(jobs, "_config_hash", lambda ctx: "cfg")
    ctx = SimpleNamespace()
    before = jobs.j7_inputs_hash(ctx, "2026-W05")  # type: ignore[arg-type]
    path = ICStore().path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("week_id,ic\n2026-W04,0.1\n")
    after = jobs.j7_inputs_hash(ctx, "2026-W05")  # type: ignore[arg-type]
    assert before != after
    path.write_text("week_id,ic\n2026-W04,0.1\n2026-W05,0.2\n")
    assert jobs.j7_inputs_hash(ctx, "2026-W05") != after  # type: ignore[arg-type]
    # Once the week is locked the hash is pinned to the IC history used at lock time, so J8's
    # later updates never make a locked J7 run again.
    from portstudy.io import write_json_atomic

    locked = jobs.j7_inputs_hash(ctx, "2026-W05")  # type: ignore[arg-type]
    write_json_atomic(
        {"week_id": "2026-W05", "files": {}, "ic_history_sha256": jobs._ic_history_sha256()},
        jobs.weights_dir("2026-W05") / "locked.json",
    )
    assert jobs.j7_inputs_hash(ctx, "2026-W05") == locked  # type: ignore[arg-type]
    path.write_text("week_id,ic\n2026-W04,0.1\n2026-W05,0.2\n2026-W06,0.3\n")
    assert jobs.j7_inputs_hash(ctx, "2026-W05") == locked  # type: ignore[arg-type]


def test_m9_log_files_redact_gateway_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M9: Authorization headers, bearer tokens and sk- keys never reach the JSON log files."""
    import json
    import logging

    from portstudy.logging_setup import REDACTED, get_logger, redact

    monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(tmp_path / "data"))
    secret = "sk-" + "live-ABCDEFGHIJKLMNOP"  # built at runtime so the repo secret scan stays clean
    log = get_logger("redaction_test", date="2026-01-01")
    log.info(
        f"request failed: Authorization: Bearer {secret}",
        extra={
            "extra_fields": {
                "headers": {"Authorization": f"Bearer {secret}", "X-Trace": "ok"},
                "api_key": secret,
                "detail": f"used key {secret} for provider/model",
                "nested": [{"token": secret}, "plain text"],
            }
        },
    )
    for handler in log.handlers:
        handler.flush()
    path = tmp_path / "data" / "logs" / "redaction_test" / "2026-01-01.jsonl"
    text = path.read_text()
    assert secret not in text
    record = json.loads(text.splitlines()[-1])
    assert record["headers"] == {"Authorization": REDACTED, "X-Trace": "ok"}
    assert record["api_key"] == REDACTED
    assert record["nested"][0]["token"] == REDACTED and record["nested"][1] == "plain text"
    assert "provider/model" in record["detail"]
    assert redact({"week_id": "2026-W05", "n": 3}) == {"week_id": "2026-W05", "n": 3}
    logging.getLogger("portstudy.redaction_test").handlers.clear()


def test_m11_real_metrics_start_from_the_value_entering_the_selected_weeks(
    monkeypatch: pytest.MonkeyPatch, study_config: StudyConfig
) -> None:
    """M11: with a pilot week excluded, the real total return is measured over the selected weeks
    from the value carried into them, not from `initial_value`."""
    import pandas as pd

    from portstudy.eval import shuffle

    v0 = study_config.simulator.initial_value
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-02-02", "2026-02-03", "2026-02-09", "2026-02-10"]),
            "week_id": ["2026-W05", "2026-W05", "2026-W06", "2026-W06"],
            "value": [v0 * 1.05, v0 * 1.10, v0 * 1.00, v0 * 0.99],
        }
    )
    monkeypatch.setattr(shuffle, "read_daily", lambda portfolio: daily)
    both = shuffle.real_metrics(study_config, "p", ["2026-W05", "2026-W06"])
    assert both["total_return"] == pytest.approx(-0.01)
    only_second = shuffle.real_metrics(study_config, "p", ["2026-W06"])
    assert only_second["total_return"] == pytest.approx(0.99 / 1.10 - 1.0)
    assert shuffle.real_metrics(study_config, "p", ["2026-W07"])["total_return"] != pytest.approx(
        -0.01
    )


def test_m7_dropped_view_branch_reuses_the_written_q0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M7: `existing_q0_weights` reads the Q0 J7 wrote (csv + meta) so the dropped-view branch
    does not solve Q0 again and rewrite its files."""
    import json

    import pandas as pd

    from portstudy.optimize import Weights, existing_q0_weights, weights_dir
    from portstudy.types import PortfolioId

    monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(tmp_path / "data"))
    assert existing_q0_weights("2026-W05", "mvo") is None
    key = PortfolioId.q0("mvo").key
    idx = pd.Index(["AAA", "BBB", "CCC"], name="ticker")
    q0 = Weights(
        portfolio=key,
        week_id="2026-W05",
        weights=pd.Series([0.5, 0.3, 0.2], index=idx),
        mu_bl=pd.Series([0.01, 0.02, 0.03], index=idx),
        tilt=pd.Series([0.0, 0.001, -0.001], index=idx),
        use_llm_view=False,
        ic_b=0.03,
        ic_l=0.0,
        solver="CLARABEL",
        fallback_flag=None,
        notes=["q0"],
        strategy="mvo",
        decision_source="q0",
        strategy_diagnostics={"objective": 1.5},
        cov_mu_sha256="abc",
    )
    folder = weights_dir("2026-W05")
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ticker": idx,
            "weight": q0.weights.values,
            "mu_bl": q0.mu_bl.values,
            "tilt": q0.tilt.values,
        }
    ).to_csv(folder / f"{key}.csv", index=False)
    (folder / f"{key}.meta.json").write_text(json.dumps(q0.meta()))
    loaded = existing_q0_weights("2026-W05", "mvo")
    assert loaded is not None
    pd.testing.assert_series_equal(loaded.weights, q0.weights, check_names=False)
    pd.testing.assert_series_equal(loaded.mu_bl, q0.mu_bl, check_names=False)
    pd.testing.assert_series_equal(loaded.tilt, q0.tilt, check_names=False)
    assert loaded.meta() == json.loads(json.dumps(q0.meta()))


def test_l3_litellm_reads_cached_prompt_tokens(
    study_config: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L3: `usage.prompt_tokens_details.cached_tokens` fills `RawResult.tokens_cached`."""
    import httpx

    from portstudy.config import LlmSettings
    from tests.unit.test_v2_phase4_litellm import _item, _provider, _response

    payload = _response(usage={"prompt_tokens": 120, "completion_tokens": 9})
    payload["usage"]["prompt_tokens_details"] = {"cached_tokens": 100}  # type: ignore[index]
    provider = _provider(lambda _: httpx.Response(200, json=payload), monkeypatch)
    result = provider.complete(_item(study_config), LlmSettings())
    assert (result.tokens_in, result.tokens_cached, result.tokens_out) == (120, 100, 9)
    plain = _provider(lambda _: httpx.Response(200, json=_response()), monkeypatch)
    assert plain.complete(_item(study_config), LlmSettings()).tokens_cached == 0
