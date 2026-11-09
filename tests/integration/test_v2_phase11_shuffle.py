"""v2 phase 11 tests: shards in any order, twice, or in separate processes equal one serial
run; a missing shard fails collect with its range; each shuffle keeps the multiset of range
triples per week; the identity permutation equals the real portfolio for every strategy; a shard
file without its completion marker is rerun."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portstudy.config import ModelRegistry, StudyConfig
from portstudy.data.providers.synthetic import SyntheticWorld, synthetic_bundle
from portstudy.eval.data import load_eval_data
from portstudy.eval.shuffle import (
    RANGE_COLUMNS,
    MissingShards,
    Shard,
    _load_states,
    collect_shards,
    identity_weights,
    plan_shards,
    read_shard_plan,
    run_shard,
    run_shards_local,
    shard_is_complete,
    shuffle_portfolios,
    shuffle_test,
    week_rng,
    write_shard_plan,
)
from portstudy.llm.prompts import load_manifest
from portstudy.optimize import read_weights
from portstudy.pipeline.context import RunContext
from portstudy.types import PortfolioId
from portstudy.views.residualize import valid_scores
from tests.conftest import generous_budget

N_SHUFFLES = 20
SHARD_SIZE = 5


def _ctx(cfg: StudyConfig, registry: ModelRegistry, world: SyntheticWorld) -> RunContext:
    return RunContext(
        cfg=cfg,
        registry=registry,
        manifest=load_manifest(),
        synthetic=True,
        data_providers=synthetic_bundle(world),
    )


def _fast(cfg: StudyConfig) -> StudyConfig:
    shuffle = cfg.eval.shuffle.model_copy(
        update={"n_shuffles": N_SHUFFLES, "shard_size": SHARD_SIZE}
    )
    return cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"shuffle": shuffle})})


def test_shards_in_any_order_twice_and_in_processes_match_serial(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
) -> None:
    cfg = _fast(generous_budget(study_config))
    ctx = _ctx(cfg, model_registry, world)
    data = load_eval_data(cfg, ctx.portfolio_keys())
    portfolio = "mvo__stub-ow-a__skills_1"
    # The plan also covers the strategy's Q0; this test follows one steered portfolio.
    shards = [
        s for s in plan_shards([portfolio], N_SHUFFLES, SHARD_SIZE) if s.portfolio == portfolio
    ]
    assert [(s.start, s.stop) for s in shards] == [(0, 5), (5, 10), (10, 15), (15, 20)]
    # Serial, in order.
    for shard in shards:
        run_shard(cfg, shard, data.weeks)
    serial, _ = collect_shards(shards)
    assert list(serial["shuffle"]) == list(range(N_SHUFFLES))
    # Reverse order into a fresh data folder, then twice more, then four separate processes.
    for shard in shards:
        shard.path.unlink()
    for shard in reversed(shards):
        run_shard(cfg, shard, data.weeks)
    reordered, _ = collect_shards(shards)
    pd.testing.assert_frame_equal(serial, reordered)
    mtimes = {s.path: s.path.stat().st_mtime_ns for s in shards}
    run_shards_local(cfg, shards, data.weeks)  # complete shards are skipped, files untouched
    assert {s.path: s.path.stat().st_mtime_ns for s in shards} == mtimes
    for shard in shards:
        shard.path.unlink()
    run_shards_local(cfg, shards, data.weeks, processes=4)
    parallel, _ = collect_shards(shards)
    pd.testing.assert_frame_equal(serial, parallel)
    # Done when: 20 shuffles in 4 shards match the serial result; the table carries p-values.
    table, dists = shuffle_test(cfg, [portfolio], data.weeks, shards=shards, require_complete=True)
    assert len(dists) == N_SHUFFLES and table["n_shuffles"].iloc[0] == N_SHUFFLES
    assert table["p_total_return"].between(0, 1).all()


def test_missing_shard_fails_collect_with_its_range(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
) -> None:
    cfg = _fast(generous_budget(study_config))
    ctx = _ctx(cfg, model_registry, world)
    data = load_eval_data(cfg, ctx.portfolio_keys())
    shards = [
        s for s in plan_shards(["topk__stub-ow-b__bare"], 10, 5) if not s.portfolio.endswith("Q0")
    ]
    assert len(shards) == 2
    run_shard(cfg, shards[0], data.weeks)
    with pytest.raises(MissingShards) as info:
        collect_shards(shards)
    assert "topk__stub-ow-b__bare [5, 10)" in str(info.value)
    with pytest.raises(MissingShards):
        shuffle_test(
            cfg, ["topk__stub-ow-b__bare"], data.weeks, shards=shards, require_complete=True
        )
    # Without require_complete the missing shard is run first.
    table, dists = shuffle_test(cfg, ["topk__stub-ow-b__bare"], data.weeks, shards=shards)
    assert len(dists) == 10 and shard_is_complete(shards[1])


def test_shard_without_marker_is_rerun(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
) -> None:
    cfg = _fast(generous_budget(study_config))
    ctx = _ctx(cfg, model_registry, world)
    data = load_eval_data(cfg, ctx.portfolio_keys())
    shard = Shard("mvo__Q0", 0, 3)
    run_shard(cfg, shard, data.weeks)
    good = json.loads(shard.path.read_text(encoding="utf-8"))
    assert good["complete"] is True and len(good["results"]) == 3
    half = {k: v for k, v in good.items() if k != "complete"}
    half["results"] = half["results"][:1]
    shard.path.write_text(json.dumps(half), encoding="utf-8")
    assert not shard_is_complete(shard)
    run_shard(cfg, shard, data.weeks)
    again = json.loads(shard.path.read_text(encoding="utf-8"))
    assert again == good
    shard.path.write_text("{not json", encoding="utf-8")
    assert not shard_is_complete(shard)


def test_each_shuffle_keeps_the_multiset_of_triples_per_week(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
) -> None:
    cfg = _fast(generous_budget(study_config))
    ctx = _ctx(cfg, model_registry, world)
    data = load_eval_data(cfg, ctx.portfolio_keys())
    portfolio = "cvar__stub-ow-a__bare"
    states = [s for s in _load_states(cfg, portfolio, data.weeks) if s.status == "normal"]
    assert states
    for st in states:
        valid = valid_scores(st.scores)
        real = sorted(map(tuple, valid[RANGE_COLUMNS].to_numpy().tolist()))
        for index in (0, 7):
            rng = week_rng(cfg.eval.shuffle.seed, portfolio, index, st.week_id)
            perm = rng.permutation(len(valid))
            shuffled = valid[RANGE_COLUMNS].to_numpy()[perm]
            assert sorted(map(tuple, shuffled.tolist())) == real
            assert (shuffled != valid[RANGE_COLUMNS].to_numpy()).any()
    # Different weeks and indices draw different permutations; the same key repeats.
    a = week_rng(1, portfolio, 0, states[0].week_id).permutation(50)
    b = week_rng(1, portfolio, 0, states[0].week_id).permutation(50)
    c = week_rng(1, portfolio, 1, states[0].week_id).permutation(50)
    assert (a == b).all() and not (a == c).all()


def test_identity_permutation_equals_the_real_portfolio_for_every_strategy(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
) -> None:
    cfg = generous_budget(study_config)
    ctx = _ctx(cfg, model_registry, world)
    data = load_eval_data(cfg, ctx.portfolio_keys())
    source = ctx.sources()[0]
    for strategy in cfg.strategies.enabled:
        for key in (PortfolioId.q0(strategy).key, PortfolioId.steered(strategy, source).key):
            weights = identity_weights(cfg, key, data.weeks)
            assert weights, key
            for week, w in weights.items():
                stored = read_weights(week, key).set_index("ticker")["weight"].astype(float)
                np.testing.assert_allclose(
                    w.reindex(stored.index).to_numpy(), stored.to_numpy(), atol=1e-6, err_msg=key
                )


def test_plan_covers_q0_and_steered_only_and_round_trips(
    tmp_path: Path, study_config: StudyConfig, model_registry: ModelRegistry, world: SyntheticWorld
) -> None:
    cfg = generous_budget(study_config)
    ctx = _ctx(cfg, model_registry, world)
    keys = ctx.portfolio_keys()
    shuffled = list(cfg.eval.shuffle.strategies)  # DECISION(G6): mvo, robust, topk only
    ordered = shuffle_portfolios(keys, shuffled)
    assert ordered[: len(shuffled)] == [f"{s}__Q0" for s in shuffled]
    assert not set(ordered) & set(cfg.strategies.controls)
    assert not [k for k in ordered if k.split("__")[0] in ("cvar", "resampled")]
    assert len(ordered) == len(shuffled) * (len(ctx.sources()) + 1)
    shards = plan_shards(keys, 500, 25, shuffled)
    assert len(shards) == len(ordered) * 20
    table = write_shard_plan(shards, tmp_path / "shards.csv")
    assert list(table.columns) == ["shard", "portfolio", "start", "stop"]
    assert read_shard_plan(tmp_path / "shards.csv") == shards
    with pytest.raises(ValueError):
        plan_shards(keys, 10, 0)
