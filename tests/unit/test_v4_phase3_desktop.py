"""v4 phase 3a: Kronos ticker chunking (K2), Kronos failure fallback to momentum (K3), the
kronos smoke script, the backup writability probe, and small review fixes (current_week on a
non-Friday cutoff, solver_status in the weight meta, HRP's configured linkage name)."""

from __future__ import annotations

import datetime as dt
import importlib.util
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from portstudy.backup import probe_rsync_target, run_backup
from portstudy.baseline.combine import BASELINE_COLUMNS, build_baseline, read_baseline_meta
from portstudy.baseline.common import rank_u
from portstudy.baseline.kronos import ForecastProvider, KronosForecaster, chunk_seed
from portstudy.config import KronosConfig, ModelRegistry, StudyConfig
from portstudy.data.providers.synthetic import SyntheticWorld
from portstudy.data.snapshot import read_snapshot
from portstudy.optimize import read_weights_meta, weights_dir
from portstudy.optimize.mean_variance import MVResult
from portstudy.optimize.strategies import hrp, mvo
from portstudy.optimize.strategies.common import StrategyInputs
from portstudy.paths import week_output_dir
from portstudy.pipeline.jobs import current_week, run_named
from portstudy.pipeline.status import read_status
from tests.conftest import generous_budget
from tests.integration.test_phase10_pipeline import make_ctx, run_week

REPO_ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")


# K2: ticker chunking -----------------------------------------------------------------------------


def _ohlcv(tickers: list[str], days: int = 6) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2025-01-01", periods=days)
    out = {}
    for i, ticker in enumerate(tickers):
        close = 100.0 + i + np.linspace(0.0, 1.0, days)
        out[ticker] = pd.DataFrame(
            {
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": np.ones(days),
            },
            index=dates,
        )
    return out


def _fake_torch() -> SimpleNamespace:
    return SimpleNamespace(
        manual_seed=lambda seed: None,
        cuda=SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
            reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 0,
        ),
    )


def _forecaster(
    predictor: object, batch_size: int, monkeypatch: pytest.MonkeyPatch
) -> KronosForecaster:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    forecaster = KronosForecaster(
        KronosConfig(context_candles=6, batch_size=batch_size), device="cpu"
    )
    forecaster._predictor = predictor
    return forecaster


class _DeterministicPredictor:
    """Each ticker's path depends only on its own input frame."""

    def predict_batch(self, frames, x_timestamps, y_timestamps, **kwargs):
        out = []
        for frame in frames:
            ret = float(frame["close"].iloc[-1] / frame["close"].iloc[0] - 1.0)
            out.append(pd.DataFrame({"close": [frame["close"].iloc[-1] * (1 + ret)]}))
        return out


class _NoisyPredictor:
    """Draws sampling noise from numpy's global RNG."""

    def predict_batch(self, frames, x_timestamps, y_timestamps, **kwargs):
        out = []
        for frame in frames:
            noise = float(np.random.normal(0.0, 0.01))
            out.append(pd.DataFrame({"close": [frame["close"].iloc[-1] * (1 + noise)]}))
        return out


@pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
def test_kronos_chunk_sizes_give_identical_results(
    monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    tickers = [f"T{i}" for i in range(5)]
    result = _forecaster(_DeterministicPredictor(), batch_size, monkeypatch).forecast(
        _ohlcv(tickers), horizon=2, n_samples=3, seed=7
    )
    expected = _forecaster(_DeterministicPredictor(), 5, monkeypatch).forecast(
        _ohlcv(tickers), horizon=2, n_samples=3, seed=7
    )
    pd.testing.assert_frame_equal(result, expected)


def test_kronos_chunk_seeding_is_deterministic_and_varies_per_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tickers = [f"T{i}" for i in range(4)]
    data = _ohlcv(tickers)
    first = _forecaster(_NoisyPredictor(), 2, monkeypatch).forecast(
        data, horizon=2, n_samples=4, seed=7
    )
    second = _forecaster(_NoisyPredictor(), 2, monkeypatch).forecast(
        data, horizon=2, n_samples=4, seed=7
    )
    pd.testing.assert_frame_equal(first, second)
    # T0 and T2 sit at the same position in their chunks; one seed per sample reused across
    # chunks would give them identical noise.
    assert first.loc["T0", "k"] != first.loc["T2", "k"]
    assert first.loc["T1", "k"] != first.loc["T3", "k"]


def test_chunk_seed_is_stable_and_distinct() -> None:
    assert chunk_seed(7, 0, 0) == chunk_seed(7, 0, 0)
    assert 0 <= chunk_seed(7, 0, 0) < 2**32
    seen = {chunk_seed(7, sample, chunk) for sample in range(4) for chunk in range(4)}
    assert len(seen) == 16


def test_kronos_batch_size_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        KronosConfig(batch_size=0)


# K3: failure fallback ----------------------------------------------------------------------------


class _RaisingForecaster(ForecastProvider):
    name = "raising"

    def forecast(self, ohlcv_by_ticker, horizon=5, n_samples=20, seed=0) -> pd.DataFrame:
        raise RuntimeError("model exploded")


def _momentum_only_baseline(df: pd.DataFrame) -> None:
    assert list(df.columns) == BASELINE_COLUMNS
    assert df["k"].isna().all() and df["k_sd"].isna().all()
    z = df["z_b"]
    assert z.notna().all() and float(z.std(ddof=1)) == pytest.approx(1.0)


def _momentum_only_baseline_in_memory(df: pd.DataFrame) -> None:
    _momentum_only_baseline(df)
    pd.testing.assert_series_equal(
        df.set_index("ticker")["b"], rank_u(df.set_index("ticker")["m"]), check_names=False
    )


def test_kronos_failure_falls_back_to_momentum_and_writes_meta(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
) -> None:
    cfg = generous_budget(study_config)
    ctx = make_ctx(cfg, model_registry, world, inject_week)
    run_named(ctx, "J1", inject_week)
    df = build_baseline(
        inject_week, cfg, read_snapshot(inject_week), forecaster=_RaisingForecaster()
    )
    _momentum_only_baseline_in_memory(df)
    meta = read_baseline_meta(inject_week)
    assert meta["kronos_fallback"] is True and "model exploded" in meta["reason"]
    assert meta["forecaster"] == "stub" and meta["n_stocks"] == len(df)


def test_kronos_failure_with_hold_reraises(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
) -> None:
    cfg = generous_budget(study_config).model_copy(
        update={"baseline": study_config.baseline.model_copy(update={"on_kronos_failure": "hold"})}
    )
    ctx = make_ctx(cfg, model_registry, world, inject_week)
    run_named(ctx, "J1", inject_week)
    with pytest.raises(RuntimeError, match="model exploded"):
        build_baseline(
            inject_week, cfg, read_snapshot(inject_week), forecaster=_RaisingForecaster()
        )


def test_full_week_completes_on_kronos_fallback(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import portstudy.baseline.combine as combine_mod

    cfg = generous_budget(study_config)
    ctx = make_ctx(cfg, model_registry, world, inject_week)
    monkeypatch.setattr(combine_mod, "make_forecaster", lambda cfg: _RaisingForecaster())
    states = run_week(ctx, inject_week, ["J1", "J2", "J3", "J4", "J5", "J6", "J7"])
    assert all(state == "done" for state in states.values()), states
    j2 = read_status(inject_week, "J2")
    assert j2.result["kronos_fallback"] is True
    baseline = pd.read_csv(week_output_dir(inject_week) / "baseline.csv")
    _momentum_only_baseline(baseline)
    # The CSV round-trips m at limited precision, so near-ties can swap adjacent ranks; the
    # momentum ordering must still match (b = rank_u(m) up to rounding).
    assert baseline["b"].corr(baseline["m"], method="spearman") > 0.999
    folder = weights_dir(inject_week)
    for key in ctx.portfolio_keys():
        assert (folder / f"{key}.csv").exists(), key


# kronos_smoke -----------------------------------------------------------------------------------


def test_kronos_smoke_exits_cleanly_without_torch() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed here")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "kronos_smoke.py"), "--n-stocks", "5"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and "torch is not installed" in result.stdout


@pytest.mark.slow
def test_kronos_smoke_runs_on_cuda() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "kronos_smoke.py"),
            "--n-stocks",
            "10",
            "--n-samples",
            "2",
            "--batch-size",
            "4",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0 and "chunk 1/" in result.stdout


# Backup probe ------------------------------------------------------------------------------------


def test_probe_rsync_target_local_remote_and_unwritable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "new" / "backups"
    assert probe_rsync_target(str(folder)) == (True, "")
    assert folder.is_dir() and not list(folder.glob(".portstudy_probe_*"))
    ok, reason = probe_rsync_target("user@host:/backups")
    assert ok and reason == "remote target not probed"
    monkeypatch.setattr(
        Path, "write_text", lambda self, *a, **k: (_ for _ in ()).throw(OSError("denied"))
    )
    ok, reason = probe_rsync_target(str(folder))
    assert not ok and "denied" in reason


def test_run_backup_warns_on_unwritable_target_and_continues(
    data_dir: Path,
    study_config: StudyConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = str(data_dir / "backups_target")
    cfg = study_config.model_copy(
        update={"backup": study_config.backup.model_copy(update={"rsync_target": target})}
    )
    monkeypatch.setattr(
        Path, "write_text", lambda self, *a, **k: (_ for _ in ()).throw(OSError("denied"))
    )
    run_backup("2026-W02", cfg, dry_run=True)
    out = capsys.readouterr().out
    assert f"backup rsync target not writable ({target})" in out
    assert "dry run: backup folder would be" in out


# Small review fixes ------------------------------------------------------------------------------


def test_current_week_rolls_forward_at_a_non_friday_cutoff_close() -> None:
    # 2026-12-25 is a Friday holiday, so W52's cutoff is Thursday 2026-12-24 (early close 13:00).
    before = dt.datetime(2026, 12, 24, 12, 0, tzinfo=ET)
    after = dt.datetime(2026, 12, 24, 13, 30, tzinfo=ET)
    assert current_week(before) == "2026-W51"
    assert current_week(after) == "2026-W52"


def test_solver_status_lands_in_the_weight_meta(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = generous_budget(study_config)
    ctx = make_ctx(cfg, model_registry, world, inject_week)
    run_week(ctx, inject_week, ["J1", "J2", "J4"])
    real_solve = mvo.solve_mean_variance

    def inaccurate(*args, **kwargs) -> MVResult:
        result = real_solve(*args, **kwargs)
        result.solver_status = "optimal_inaccurate"
        return result

    monkeypatch.setattr(mvo, "solve_mean_variance", inaccurate)
    from portstudy.optimize import build_q0_weights

    with caplog.at_level(logging.WARNING, logger="portstudy.optimize"):
        build_q0_weights(inject_week, cfg, ic_b=0.02)
    meta = read_weights_meta(inject_week, "mvo__Q0")
    assert meta["solver_status"] == "optimal_inaccurate"
    assert any(record.message == "solver_inaccurate" for record in caplog.records)


def test_hrp_reports_the_configured_linkage() -> None:
    tickers = [f"T{i:02d}" for i in range(20)]
    rng = np.random.default_rng(0)
    sigma = pd.DataFrame(np.eye(20) * 0.01, index=tickers, columns=tickers)
    inputs = StrategyInputs(
        week_id="2026-W10",
        tickers=tickers,
        mu=pd.Series(rng.normal(0.005, 0.01, 20), index=tickers),
        sigma=sigma,
        cov_mu=sigma * 0.1,
        delta=2.5,
    )
    cfg = StudyConfig()
    assert hrp.run(inputs, cfg).solver == "single_linkage"
    configured = cfg.model_copy(
        update={
            "strategies": cfg.strategies.model_copy(
                update={"hrp": cfg.strategies.hrp.model_copy(update={"linkage": "average"})}
            )
        }
    )
    assert hrp.run(inputs, configured).solver == "average_linkage"
