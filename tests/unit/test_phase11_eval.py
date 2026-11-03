"""Phase 11 tests: statistics by hand, Holm properties, planted-effect detection, false positives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from portstudy import calendar as cal
from portstudy.config import StudyConfig
from portstudy.data.providers.synthetic import SyntheticWorld
from portstudy.eval.rank_ic import ic_tables, long_table
from portstudy.eval.tests import (
    default_maxlags,
    holm,
    min_detectable_gap,
    newey_west_mean,
    paired_summary,
    sign_flip_permutation,
    sign_test,
)
from portstudy.llm.prompts import assemble_prompt, load_manifest
from portstudy.llm.providers.base import BatchItem, make_custom_id
from portstudy.llm.providers.stub import StubLlmProvider, sentiment_from_text
from portstudy.views.ic import ICStore, rank_ic

X = pd.Series([0.02, -0.01, 0.03, 0.00, 0.05, -0.02, 0.04, 0.01])


def test_newey_west_matches_hand_calculation_with_zero_lags() -> None:
    r = newey_west_mean(X, maxlags=0)
    n = len(X)
    mean = X.mean()
    # HAC with zero lags is White's estimator: se = sqrt(sum(e^2)) / n.
    se = np.sqrt(((X - mean) ** 2).sum()) / n
    assert r.mean == pytest.approx(mean)
    assert r.se == pytest.approx(se, rel=1e-9)
    assert r.t == pytest.approx(mean / se)
    assert r.p == pytest.approx(2 * (1 - stats.norm.cdf(abs(mean / se))), rel=1e-6)
    assert r.n == n and r.maxlags == 0
    assert default_maxlags(100) == 4 and default_maxlags(8) == 2 and default_maxlags(30) == 3 and default_maxlags(1) == 1


def test_sign_test_hand_value() -> None:
    k, n, p = sign_test(X)
    assert (k, n) == (5, 7)  # one zero is dropped
    assert p == pytest.approx(stats.binomtest(5, 7, 0.5).pvalue)


def test_sign_flip_permutation_is_seeded_and_sane() -> None:
    p1 = sign_flip_permutation(X, n=2000, seed=1)
    p2 = sign_flip_permutation(X, n=2000, seed=1)
    assert p1 == p2
    # All-positive vector of length 8: exact two-sided p = 2 / 2^8 = 0.0078.
    allpos = pd.Series([1.0] * 8)
    assert sign_flip_permutation(allpos, n=20000, seed=2) == pytest.approx(2 / 256, abs=0.004)


def test_holm_never_below_raw_and_monotone() -> None:
    raw = np.array([0.01, 0.04, 0.03, 0.20, np.nan, 0.001])
    adj = holm(raw)
    mask = ~np.isnan(raw)
    assert np.all(adj[mask] >= raw[mask])
    order = np.argsort(raw[mask])
    assert np.all(np.diff(adj[mask][order]) >= -1e-12)
    assert np.isnan(adj[4])
    # Hand check: sorted raw 0.001, 0.01, 0.03, 0.04, 0.20 with m = 5.
    assert adj[5] == pytest.approx(min(1, 0.001 * 5))
    assert adj[0] == pytest.approx(max(0.005, 0.01 * 4))


def test_min_detectable_gap_formula() -> None:
    assert min_detectable_gap(0.1, 25) == pytest.approx(2.8 * 0.1 / 5)
    assert np.isnan(min_detectable_gap(0.1, 0))


def test_paired_summary_counts() -> None:
    s = paired_summary(X, tie_band=0.01, seed=0, n_perm=500, n_boot=200)
    assert (s["wins"], s["ties"], s["losses"]) == (4, 3, 1)
    assert s["n_weeks"] == 8 and s["ci_low"] <= s["mean_gap"] <= s["ci_high"]


def test_ic_tables_exclude_fallback_weeks(tmp_path: object) -> None:
    store = ICStore(tmp_path / "ic.csv")  # type: ignore[operator]
    weeks = cal.week_range("2026-W05", 4)
    rows = [{"week_id": w, "signal": "baseline", "model": "", "config_id": "", "rank_ic": 0.1, "n_stocks": 100}
            for w in weeks]
    rows += [{"week_id": w, "signal": "llm_raw", "model": "m", "config_id": "skills_1", "rank_ic": 0.2, "n_stocks": 100}
             for w in weeks]
    store.upsert(rows)
    tables = ic_tables(store, ["m__skills_1"], clean_weeks=lambda p: [weeks[0], weeks[2], weeks[3]])
    col = tables["llm_raw"]["m__skills_1"]
    assert np.isnan(col[weeks[1]]) and col[weeks[0]] == 0.2
    assert tables["baseline"]["baseline"].notna().all()
    long = long_table(tables)
    assert set(long["signal"]) == {"baseline", "llm_raw"} and len(long) == 2 * 4


# Planted effect ----------------------------------------------------------------------------------


def _weekly_ics(study_config: StudyConfig, world: SyntheticWorld, weeks: list[str], seed: int,
                configs: list[str], noise: dict[str, float] | None = None) -> pd.DataFrame:
    """Rank IC per week and configuration from the real stub, scored on the snapshot news text."""
    from portstudy.data.snapshot import read_snapshot

    adj = world.prices[world.prices["ticker"].isin(world.stock_tickers)]
    wide = adj.drop_duplicates(["ticker", "date"]).pivot(index="date", columns="ticker", values="adj_close")
    man = load_manifest()
    stub = StubLlmProvider("m", seed=seed, noise_by_config=noise)
    out: dict[str, dict[str, float]] = {c: {} for c in configs}
    for w in weeks:
        t0, t1 = cal.monday_open_day(w), cal.window_end_date(w)
        if t0 not in wide.index or t1 not in wide.index:
            continue
        ret = wide.loc[t1] / wide.loc[t0] - 1
        snap = read_snapshot(w)
        ref = snap.reference.set_index("ticker")
        text = snap.news_text.set_index("ticker")["news_text"]
        for c in configs:
            scores = {}
            for tk in snap.included_tickers:
                row = {"ticker": tk, "name": ref.loc[tk, "name"], "sector": ref.loc[tk, "sector"]}
                prompt = assemble_prompt(c, row, str(text.get(tk, "")), study_config, man)
                item = BatchItem(make_custom_id("m", c, w, tk), "m", c, w, tk, prompt)
                scores[tk] = stub.answer_for(item)[1]
            out[c][w] = rank_ic(pd.Series(scores, dtype=float), ret)
    return pd.DataFrame(out)


def test_h1_and_h3_detect_the_planted_skills_edge(snapshot_data_dir: object, study_config: StudyConfig,
                                                  world: SyntheticWorld, session_weeks: list[str]) -> None:
    # A known edge: skills reads the planted words almost cleanly, the controls barely do.
    noise = {"bare": 6.0, "skills_1": 0.5, "placebo": 6.0}
    ics = _weekly_ics(study_config, world, session_weeks, seed=0, configs=["bare", "skills_1", "placebo"],
                      noise=noise)
    assert len(ics) >= 25
    # H1: the skills configuration has a clearly positive IC against the realized returns.
    assert newey_west_mean(ics["skills_1"]).p < 0.01 and ics["skills_1"].mean() > 0.05
    assert sign_flip_permutation(ics["skills_1"], n=5000, seed=0) < 0.01
    # H3: skills beats bare and placebo after Holm across the family.
    gap_bare = ics["skills_1"] - ics["bare"]
    gap_placebo = ics["skills_1"] - ics["placebo"]
    p = holm([sign_flip_permutation(gap_bare, n=5000, seed=0), sign_flip_permutation(gap_placebo, n=5000, seed=0)])
    assert gap_bare.mean() > 0 and gap_placebo.mean() > 0
    assert (p < study_config.eval.significance).all(), (gap_bare.mean(), gap_placebo.mean(), p)
    # The two controls do not differ in expectation.
    assert sign_flip_permutation(ics["placebo"] - ics["bare"], n=5000, seed=0) > study_config.eval.significance
    sample = "News from the past week:" + chr(10) + "Alpha beats and surges. misses"
    assert sentiment_from_text(sample) == 1.0


def test_placebo_vs_bare_false_positive_rate(study_config: StudyConfig, world: SyntheticWorld,
                                             session_weeks: list[str]) -> None:
    """With no planted edge (placebo equals bare), the paired test rejects at about the nominal rate."""
    # Vectorized copy of the stub's score formula on the planted sentiment, so 200 seeds are cheap.
    sent = world.sentiment[world.sentiment["week_id"].isin(session_weeks)]
    adj = world.prices[world.prices["ticker"].isin(world.stock_tickers)]
    wide = adj.drop_duplicates(["ticker", "date"]).pivot(index="date", columns="ticker", values="adj_close")
    rets = {}
    for w in session_weeks:
        t0, t1 = cal.monday_open_day(w), cal.window_end_date(w)
        if t0 in wide.index and t1 in wide.index:
            rets[w] = wide.loc[t1] / wide.loc[t0] - 1
    weeks = [w for w in session_weeks if w in rets]
    rejections = 0
    n_seeds = 200
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        gaps = []
        for w in weeks:
            s = sent[sent["week_id"] == w].set_index("ticker")["sentiment"].reindex(rets[w].index).fillna(0)
            # bare and placebo share the noise level (1.6); only the draws differ.
            a = np.clip(np.rint(1.5 * s.to_numpy() + rng.normal(0, 1.6, len(s))), -5, 5)
            b = np.clip(np.rint(1.5 * s.to_numpy() + rng.normal(0, 1.6, len(s))), -5, 5)
            ic_a = rank_ic(pd.Series(a, index=s.index), rets[w])
            ic_b = rank_ic(pd.Series(b, index=s.index), rets[w])
            gaps.append(ic_a - ic_b)
        if sign_flip_permutation(pd.Series(gaps), n=2000, seed=seed) < study_config.eval.significance:
            rejections += 1
    rate = rejections / n_seeds
    # The sign-flip test is exact under the null: nominal 5 percent, binomial sd about 1.5 percent.
    assert rate <= 0.10, rate
