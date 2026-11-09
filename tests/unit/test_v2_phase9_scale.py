"""v2 phase 9 tests: thin sectors fill from leftovers and are logged, cached and uncached MVO
agree, the cache key includes the tickers, and the benchmark script writes its table."""

from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portstudy import calendar as cal
from portstudy.config import StudyConfig, load_study_config
from portstudy.data.providers.base import (
    MembersProvider,
    PriceProvider,
    ProviderBundle,
)
from portstudy.data.universe import build_universe
from portstudy.optimize import problem_cache
from portstudy.optimize.mean_variance import solve_mean_variance

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import benchmark_pipeline  # noqa: E402

SECTORS = [
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
]
THIN = "Utilities"


class _Members(MembersProvider):
    name = "fake"

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def get_members(self, as_of: dt.date) -> pd.DataFrame:
        return self.frame.copy()


class _Prices(PriceProvider):
    name = "fake"

    def __init__(self, volume: pd.Series) -> None:
        self.volume = volume

    def get_prices(self, tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
        days = cal.trading_days(start, end)
        rows = []
        for ticker in tickers:
            for day in days:
                rows.append(
                    {
                        "ticker": ticker,
                        "date": day,
                        "close": 100.0,
                        "volume": float(self.volume[ticker]),
                    }
                )
        return pd.DataFrame(rows)

    def get_corporate_actions(
        self, tickers: list[str], start: dt.date, end: dt.date
    ) -> pd.DataFrame:
        return pd.DataFrame(columns=["ticker", "date", "kind", "value"])


def _thin_world(per_full_sector: int, thin_count: int) -> ProviderBundle:
    rows = []
    for idx, sector in enumerate(SECTORS):
        count = thin_count if sector == THIN else per_full_sector
        for i in range(count):
            rows.append(
                {"ticker": f"S{idx:02d}T{i:03d}", "name": f"{sector} {i}", "sector": sector}
            )
    members = pd.DataFrame(rows)
    volume = pd.Series(np.linspace(1e6, 5e7, len(members)), index=members["ticker"])
    bundle = ProviderBundle.__new__(ProviderBundle)
    bundle.members = _Members(members)
    bundle.prices = _Prices(volume)
    return bundle


def test_thin_sector_fills_to_size_and_is_logged(
    study_config: StudyConfig, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = study_config.model_copy(
        update={
            "universe": study_config.universe.model_copy(
                update={"size": 300, "min_history_days": 60}
            )
        }
    )
    bundle = _thin_world(per_full_sector=40, thin_count=15)
    with caplog.at_level(logging.INFO, logger="portstudy.universe"):
        out = build_universe(dt.date(2026, 1, 9), cfg, bundle, print_table=False)
    assert len(out) == 300 and out["ticker"].is_unique
    counts = out.groupby("sector").size()
    assert counts[THIN] == 15
    assert counts.drop(THIN).min() >= 27  # floor(300 / 11) per sector, then leftovers
    thin = [r for r in caplog.records if r.getMessage() == "universe_thin_sector"]
    assert len(thin) == 1
    fields = thin[0].extra_fields  # type: ignore[attr-defined]
    assert fields == {"sector": THIN, "quota": 27, "available": 15, "short": 12}
    left = [r for r in caplog.records if r.getMessage() == "universe_leftover_slots"]
    assert len(left) == 1
    assert left[0].extra_fields == {"slots": 300 - (10 * 27 + 15), "filled": 15}  # type: ignore[attr-defined]


def _random_problem(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    tickers = [f"S{i:03d}" for i in range(n)]
    loadings = rng.normal(0.0, 0.3, size=(n, 2))
    sigma = loadings @ loadings.T * 4e-4 + np.diag(rng.uniform(4e-4, 1e-3, size=n))
    return pd.DataFrame(sigma, index=tickers, columns=tickers), pd.Series(
        rng.normal(0.001, 0.004, size=n), index=tickers
    )


def test_cached_mvo_matches_uncached_within_1e9(study_config: StudyConfig) -> None:
    sigma, _ = _random_problem(60, 1)
    rng = np.random.default_rng(2)
    problem_cache.clear_problem_cache()
    for _ in range(20):
        mu = pd.Series(rng.normal(0.001, 0.004, size=60), index=sigma.index)
        cached = solve_mean_variance(mu, sigma, study_config.risk.delta, study_config.optimizer)
        fresh = solve_mean_variance(
            mu, sigma, study_config.risk.delta, study_config.optimizer, use_cache=False
        )
        assert cached.solver == fresh.solver
        np.testing.assert_allclose(cached.weights.to_numpy(), fresh.weights.to_numpy(), atol=1e-9)
    stats = problem_cache.problem_cache_stats()
    assert stats["misses"] == 1 and stats["hits"] == 19 and stats["size"] == 1


def test_cache_key_includes_tickers_and_sigma(study_config: StudyConfig) -> None:
    problem_cache.clear_problem_cache()
    sigma, mu = _random_problem(40, 3)
    solve_mean_variance(mu, sigma, study_config.risk.delta, study_config.optimizer)
    renamed = sigma.copy()
    renamed.index = renamed.columns = [f"{t}X" for t in sigma.index]
    solve_mean_variance(
        mu.set_axis(renamed.index), renamed, study_config.risk.delta, study_config.optimizer
    )
    scaled = sigma * 1.01
    solve_mean_variance(mu, scaled, study_config.risk.delta, study_config.optimizer)
    assert problem_cache.problem_cache_stats() == {"size": 3, "hits": 0, "misses": 3}


def test_projection_math() -> None:
    times = {"mvo": [0.1, 0.1], "cvar": [0.3, 0.3], "EW": [0.01]}
    proj = benchmark_pipeline.projection(
        times,
        ["mvo", "cvar"],
        ["EW"],
        n_sources=1,
        projected_sources=51,
        weeks=52,
        n_shuffles=500,
    )
    assert proj["j7_s_per_week"] == pytest.approx(2 * 0.4 + 0.01)
    assert proj["j7_projected_s_per_week"] == pytest.approx(52 * 0.4 + 0.01)
    # PLAN3 phase 2: each shuffled strategy's 51 steered portfolios plus its Q0, at its own time.
    assert proj["steered_portfolios"] == 104
    assert proj["shuffle_cpu_hours"] == pytest.approx(52 * 500 * 52 * (0.1 + 0.3) / 3600)
    only_mvo = benchmark_pipeline.projection(
        times, ["mvo", "cvar"], ["EW"], 1, 51, 52, 500, shuffle_strategies=["mvo"]
    )
    assert only_mvo["steered_portfolios"] == 52
    assert only_mvo["shuffle_cpu_hours"] == pytest.approx(52 * 500 * 52 * 0.1 / 3600)


def test_benchmark_writes_table_small(tmp_path: Path) -> None:
    code = benchmark_pipeline.main(
        [
            "--config",
            str(REPO_ROOT / "config" / "study_test.yaml"),
            "--n-stocks",
            "40",
            "--sources",
            "1",
            "--weeks",
            "2",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    files = list(tmp_path.glob("*.csv"))
    assert len(files) == 1
    table = pd.read_csv(files[0])
    assert list(table.columns) == benchmark_pipeline.BENCHMARK_COLUMNS
    cfg = load_study_config(REPO_ROOT / "config" / "study_test.yaml")
    assert list(table["strategy"]) == [*cfg.strategies.enabled, *cfg.strategies.controls]
    assert (table["mean_solve_s"] > 0).all() and table["shuffle_cpu_hours"].nunique() == 1


@pytest.mark.slow
def test_benchmark_300_stocks_all_strategies(tmp_path: Path) -> None:
    code = benchmark_pipeline.main(
        [
            "--config",
            str(REPO_ROOT / "config" / "study.yaml"),
            "--n-stocks",
            "300",
            "--sources",
            "3",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    table = pd.read_csv(next(tmp_path.glob("*.csv")))
    assert set(table["strategy"]) >= {"mvo", "cvar", "robust", "resampled", "topk"}
