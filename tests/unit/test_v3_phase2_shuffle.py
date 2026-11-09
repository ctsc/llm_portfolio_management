"""v3 phase 2: shuffle scope under DECISION(G6) and the 24-week CPU-hour projection."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from portstudy.eval.shuffle import plan_shards, shuffle_portfolios

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import benchmark_pipeline  # noqa: E402

KEYS = [
    f"{s}__{src}"
    for s in ("mvo", "cvar", "robust", "resampled", "topk")
    for src in ("Q0", "stub-ow-a__bare", "stub-ow-b__skills_1")
] + ["HRP", "EW"]


def test_shuffle_portfolios_keeps_only_the_g6_strategies() -> None:
    ordered = shuffle_portfolios(KEYS)
    assert ordered[:3] == ["mvo__Q0", "robust__Q0", "topk__Q0"]
    assert len(ordered) == 9
    assert not [k for k in ordered if k.startswith(("cvar", "resampled", "HRP", "EW"))]
    assert shuffle_portfolios(KEYS, ["cvar"]) == [
        "cvar__Q0",
        "cvar__stub-ow-a__bare",
        "cvar__stub-ow-b__skills_1",
    ]


def test_plan_shards_count_matches_the_shuffled_portfolios() -> None:
    shards = plan_shards(KEYS, 500, 25)
    assert len(shards) == 9 * 20
    assert {s.portfolio.split("__")[0] for s in shards} == {"mvo", "robust", "topk"}


def test_projection_uses_24_weeks_and_the_shuffled_strategies_only() -> None:
    times = {
        "mvo": [1.0],
        "cvar": [100.0],
        "robust": [2.0],
        "resampled": [50.0],
        "topk": [0.5],
        "HRP": [0.1],
        "EW": [0.1],
    }
    proj = benchmark_pipeline.projection(
        times,
        ["mvo", "cvar", "robust", "resampled", "topk"],
        ["HRP", "EW"],
        3,
        48,
        24,
        500,
        ["mvo", "robust", "topk"],
        replay_seconds=None,
    )
    portfolios = 49  # 48 sources plus the strategy's Q0
    expected = 24 * 500 * portfolios * (1.0 + 2.0 + 0.5) / 3600.0
    assert proj["shuffle_cpu_hours"] == pytest.approx(expected)
    assert proj["shuffle_hours_mvo"] == pytest.approx(24 * 500 * portfolios * 1.0 / 3600.0)
    assert "shuffle_hours_cvar" not in proj
    assert proj["steered_portfolios"] == 3 * portfolios
    assert math.isnan(proj["shuffle_cpu_hours_with_replay"])
    with_replay = benchmark_pipeline.projection(
        times,
        ["mvo", "cvar", "robust", "resampled", "topk"],
        ["HRP", "EW"],
        3,
        48,
        24,
        500,
        ["mvo", "robust", "topk"],
        replay_seconds=2.0,
    )
    assert with_replay["shuffle_cpu_hours_with_replay"] == pytest.approx(
        expected + 24 * 500 * 3 * portfolios * 2.0 / 3600.0
    )
