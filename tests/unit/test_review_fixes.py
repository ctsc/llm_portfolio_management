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

        def now(self):  # type: ignore[no-untyped-def]
            return run_weekend.synthetic_clock("2026-W05")()

    monkeypatch.setattr(run_weekend, "default_context", lambda **kw: Ctx())
    assert run_weekend.main(["--synthetic", "--week", "2026-W05"]) == 1
    assert calls == ["J1", "J6", "J7"]


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
