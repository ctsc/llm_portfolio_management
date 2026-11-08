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
