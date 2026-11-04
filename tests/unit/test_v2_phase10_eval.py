"""v2 phase 10 tests on a synthetic EvalData: Holm within each strategy, H2 pairs with the
strategy's own Q0, H3 covers every skills kind, H3b detects a planted width effect and skips
all-equal weeks, H4 ranks strategies in the planted order, and controls sit in no family."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portstudy.config import StudyConfig
from portstudy.eval.data import EvalData, likely_column, width_column
from portstudy.eval.hypotheses import (
    h1_llm_raw_vs_baseline,
    h2_combined_vs_baseline,
    h2_portfolio_vs_q0,
    h3_skills_vs_controls,
    h3b_width,
    h4_strategy_ranking,
    portfolio_metrics,
    skills_prompt_ids,
)
from portstudy.eval.tests import holm

MODELS = ["m-a", "m-b"]
PROMPTS = ["bare", "placebo", "skills_1", "skills_2"]
STRATEGIES = ["mvo", "cvar"]
CONTROLS = ["HRP", "EW"]
N_WEEKS = 30


def _weeks() -> list[str]:
    return [f"2026-W{w:02d}" for w in range(2, 2 + N_WEEKS)]


def _sources() -> list[str]:
    return [f"{m}__{p}" for m in MODELS for p in PROMPTS]


def _keys() -> list[str]:
    return (
        [f"{s}__Q0" for s in STRATEGIES]
        + CONTROLS
        + [f"{s}__{src}" for s in STRATEGIES for src in _sources()]
    )


def _fast(cfg: StudyConfig) -> StudyConfig:
    return cfg.model_copy(
        update={
            "eval": cfg.eval.model_copy(
                update={"permutation_samples": 400, "bootstrap_samples": 200}
            ),
            "strategies": cfg.strategies.model_copy(
                update={"enabled": STRATEGIES, "controls": CONTROLS}
            ),
        }
    )


def _build(
    cfg: StudyConfig,
    seed: int = 0,
    strategy_edge: dict[str, float] | None = None,
    width_effect: float = 0.0,
    equal_width_weeks: int = 0,
) -> EvalData:
    rng = np.random.default_rng(seed)
    weeks = _weeks()
    keys = _keys()
    sources = _sources()
    edge = strategy_edge or {}
    returns = pd.DataFrame(index=weeks, columns=keys, dtype=float)
    for strategy in STRATEGIES:
        q0 = rng.normal(0.001, 0.02, size=N_WEEKS)
        returns[f"{strategy}__Q0"] = q0
        for src in sources:
            returns[f"{strategy}__{src}"] = (
                q0 + edge.get(strategy, 0.0) + rng.normal(0.0, 0.002, size=N_WEEKS)
            )
    for control in CONTROLS:
        returns[control] = rng.normal(0.001, 0.02, size=N_WEEKS)
    statuses = pd.DataFrame("normal", index=weeks, columns=keys)
    base_ic = pd.Series(rng.normal(0.02, 0.05, size=N_WEEKS), index=weeks, name="baseline")
    ic = {"baseline": base_ic.to_frame("baseline")}
    for signal in ("llm_raw", "llm_resid", "combined"):
        ic[signal] = pd.DataFrame(
            {src: base_ic + rng.normal(0.0, 0.05, size=N_WEEKS) for src in sources}, index=weeks
        )
    # Stock-week table with a planted width effect: narrow ranges carry the signal.
    frames = []
    n_stocks = 60
    for i, w in enumerate(weeks):
        ret = rng.normal(0.0, 0.03, size=n_stocks)
        frame = pd.DataFrame({"ticker": [f"T{j:02d}" for j in range(n_stocks)], "ret": ret})
        frame["week_id"] = w
        frame["earnings_in_window"] = False
        for src in sources:
            if i < equal_width_weeks:
                width = np.full(n_stocks, 4.0)
            else:
                width = rng.integers(0, 9, size=n_stocks).astype(float)
            narrow = width <= np.median(width)
            signal = np.where(narrow, width_effect, 0.0) * ret + rng.normal(0, 0.03, n_stocks)
            frame[likely_column(src)] = np.clip(np.round(signal / 0.01), -5, 5)
            frame[width_column(src)] = width
        frames.append(frame)
    stock_week = pd.concat(frames, ignore_index=True)
    daily = {k: pd.DataFrame(columns=["date", "value", "week_id"]) for k in keys}
    trades = {k: pd.DataFrame(columns=["trade_value", "cost_usd", "reason"]) for k in keys}
    return EvalData(
        cfg=cfg,
        portfolios=[k for k in keys if "__" in k and not k.endswith("__Q0")],
        sources=sources,
        portfolio_keys=keys,
        weeks=weeks,
        pilot_week=None,
        ic=ic,
        weekly_returns=returns,
        daily_values=daily,
        trades=trades,
        statuses=statuses,
        spy_weekly=pd.Series(dtype=float),
        spy_vol20=pd.Series(dtype=float),
        stock_week=stock_week,
        solver_failed=pd.DataFrame(False, index=weeks, columns=keys),
        view_shift=pd.DataFrame(0.05, index=weeks, columns=keys),
    )


def test_h2_holm_adjusts_within_each_strategy_only(study_config: StudyConfig) -> None:
    cfg = _fast(study_config)
    data = _build(cfg, seed=1, strategy_edge={"mvo": 0.004, "cvar": 0.0})
    df = h2_portfolio_vs_q0(data)
    assert set(df["strategy"]) == set(STRATEGIES)
    for strategy in STRATEGIES:
        sub = df[df["strategy"] == strategy]
        assert len(sub) == len(_sources())
        np.testing.assert_allclose(sub["p_holm"].to_numpy(), holm(sub["p_perm"].to_numpy()))
    # Adjusting across both strategies at once would give larger values for the planted edge.
    mvo = df[df["strategy"] == "mvo"]
    assert mvo["significant"].all()
    joint = holm(df["p_perm"].to_numpy())
    assert (joint >= df["p_holm"].to_numpy() - 1e-12).all()


def test_h2_pairs_each_portfolio_with_its_own_q0(study_config: StudyConfig) -> None:
    cfg = _fast(study_config)
    data = _build(cfg, seed=2, strategy_edge={"mvo": 0.003, "cvar": -0.003})
    df = h2_portfolio_vs_q0(data).set_index("portfolio")
    for src in _sources():
        assert df.loc[f"mvo__{src}", "mean_gap"] > 0 > df.loc[f"cvar__{src}", "mean_gap"]
        assert df.loc[f"mvo__{src}", "source"] == src
    assert not any(k in df.index for k in CONTROLS)


def test_h3_covers_every_skills_kind_per_model(study_config: StudyConfig) -> None:
    cfg = _fast(study_config)
    data = _build(cfg, seed=3)
    df = h3_skills_vs_controls(data)
    skills = [p for p in PROMPTS if p in set(skills_prompt_ids())]
    assert skills == ["skills_1", "skills_2"]
    expected = {f"{m}: {p} vs {c}" for m in MODELS for p in skills for c in ("bare", "placebo")}
    assert set(df["portfolio"]) == expected
    assert set(df["model"]) == set(MODELS)
    for model in MODELS:
        sub = df[df["model"] == model]
        np.testing.assert_allclose(sub["p_holm"].to_numpy(), holm(sub["p_perm"].to_numpy()))


def test_h3b_detects_planted_width_effect_and_skips_equal_weeks(study_config: StudyConfig) -> None:
    cfg = _fast(study_config)
    data = _build(cfg, seed=4, width_effect=3.0, equal_width_weeks=3)
    df = h3b_width(data)
    assert set(df["source"]) == set(_sources())
    assert (df["mean_gap"] > 0).all() and (df["mean_ic_narrow"] > df["mean_ic_wide"]).all()
    assert (df["skipped_equal_width"] == 3).all()
    assert (
        df["usable_weeks"]
        + df["skipped_small_group"]
        + df["skipped_equal_width"]
        + df["skipped_no_ic"]
        == N_WEEKS
    ).all()
    assert df["significant"].all()
    np.testing.assert_allclose(df["p_holm"].to_numpy(), holm(df["p_perm"].to_numpy()))
    flat = h3b_width(_build(cfg, seed=5, width_effect=0.0))
    assert flat["p_perm"].min() > 0.001


def test_h4_ranking_matches_planted_order(study_config: StudyConfig) -> None:
    cfg = _fast(study_config)
    data = _build(cfg, seed=6, strategy_edge={"mvo": -0.002, "cvar": 0.004})
    df = h4_strategy_ranking(data)
    assert list(df["strategy"]) == ["cvar", "mvo"] and list(df["rank"]) == [1, 2]
    assert (df["ci_low"] <= df["mean_gap"]).all() and (df["mean_gap"] <= df["ci_high"]).all()
    assert (df["dropped_weeks"] == 0).all() and (df["n_weeks"] == N_WEEKS).all()
    assert "p_holm" not in df.columns
    # Weeks where no source is clean are dropped and counted.
    data.statuses.loc[data.weeks[:2], [k for k in data.portfolios if k.startswith("cvar__")]] = (
        "fallback"
    )
    again = h4_strategy_ranking(data).set_index("strategy")
    assert again.loc["cvar", "dropped_weeks"] == 2 and again.loc["cvar", "n_weeks"] == N_WEEKS - 2


def test_controls_in_metrics_but_in_no_family(study_config: StudyConfig) -> None:
    cfg = _fast(study_config)
    data = _build(cfg, seed=7)
    metrics = portfolio_metrics(data, clean_only=False)
    assert set(CONTROLS) <= set(metrics["portfolio"])
    assert set(metrics.loc[metrics["portfolio"].isin(CONTROLS), "strategy"]) == set(CONTROLS)
    for table in (
        h1_llm_raw_vs_baseline(data),
        h2_combined_vs_baseline(data),
        h2_portfolio_vs_q0(data),
        h3_skills_vs_controls(data),
        h3b_width(data),
    ):
        assert not table["portfolio"].isin(CONTROLS).any()
    assert not h4_strategy_ranking(data)["strategy"].isin(CONTROLS).any()
    assert set(h1_llm_raw_vs_baseline(data)["source"]) == set(_sources())


def test_eval_config_has_h3b_and_per_strategy_families(study_config: StudyConfig) -> None:
    assert study_config.eval.h3b.split == "median" and study_config.eval.h3b.min_group_size == 10
    from pathlib import Path

    from portstudy.config import load_study_config

    full = load_study_config(Path(__file__).resolve().parents[2] / "config" / "study.yaml")
    assert full.eval.holm_families["H2_portfolio_vs_q0"] == "per_strategy"
    assert full.eval.holm_families["H3b_width"] == "all_sources"
    assert full.eval.holm_families["H5_shuffle"] == "per_strategy"
    with pytest.raises(ValueError):
        study_config.eval.h3b.model_copy(update={"split": "mean"}).model_validate({"split": "mean"})
