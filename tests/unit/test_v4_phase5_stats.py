"""v4 phase 5: statistics primitives and portfolio metrics, each against a known answer."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from portstudy.eval import metrics as m
from portstudy.eval.multiple import assign_families, fdr_bh
from portstudy.eval.tests import (
    block_bootstrap_ci,
    deflated_sharpe,
    expected_max_sharpe,
    holm,
    jobson_korkie_memmel,
    ledoit_wolf_sharpe_test,
    mean_test,
    paired_summary,
    probabilistic_sharpe,
    sign_flip_p,
    t_critical,
)


def test_exact_sign_flip_matches_a_hand_count() -> None:
    # Mean 2; of the 8 sign patterns only +1+2+3 and -1-2-3 reach |mean| >= 2.
    p, method = sign_flip_p(np.array([1.0, 2.0, 3.0]), exact_max_t=24)
    assert method == "exact" and p == pytest.approx(2 / 8)
    p0, _ = sign_flip_p(np.array([1.0, -1.0]), exact_max_t=24)
    assert p0 == 1.0


def test_exact_and_monte_carlo_agree_at_t20() -> None:
    x = np.random.default_rng(5).normal(0.3, 1.0, 20)
    exact, m1 = sign_flip_p(x, exact_max_t=24)
    mc, m2 = sign_flip_p(x, exact_max_t=10, n=40000, seed=1)
    assert (m1, m2) == ("exact", "monte_carlo") and abs(exact - mc) < 0.01


def test_student_t_p_values_use_t_minus_one_df() -> None:
    x = pd.Series([0.01, -0.02, 0.03, 0.015, 0.004, -0.001, 0.02, 0.012])
    res = mean_test(x, maxlags=0)
    se_plain = x.std(ddof=1) / math.sqrt(len(x))
    assert res["se"] >= se_plain - 1e-15
    assert res["p"] == pytest.approx(2 * stats.t.sf(abs(res["mean"] / res["se"]), len(x) - 1))
    assert t_critical(23) == pytest.approx(stats.t.ppf(0.975, 22))


def test_holm_and_bh_match_statsmodels() -> None:
    p = np.array([0.001, 0.02, 0.03, 0.04, 0.2, 0.5])
    assert np.allclose(holm(p), multipletests(p, method="holm")[1])
    assert np.allclose(fdr_bh(p), multipletests(p, method="fdr_bh")[1])
    assert np.isnan(fdr_bh([np.nan, 0.01])[0])


def test_assign_families() -> None:
    df = pd.DataFrame(
        {
            "comparison_id": ["rq1", "rq2", "", "", ""],
            "hypothesis": ["H1", "H2", "H1", "H3", "H4"],
            "p": [0.01, 0.04, 0.01, 0.03, 0.02],
        }
    )
    out = assign_families(df, {"rq1", "rq2"})
    assert list(out["family"]) == ["primary", "primary", "secondary", "secondary", "exploratory"]
    assert np.allclose(out["p_adj"][:2], holm(np.array([0.01, 0.04])))
    assert np.allclose(out["p_adj"][2:4], fdr_bh(np.array([0.01, 0.03])))
    assert out["p_adj"].iloc[4] == 0.02


def test_ledoit_wolf_sharpe_test() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.002, 0.02, 120)
    same = ledoit_wolf_sharpe_test(x, x, n_boot=200, seed=1)
    assert same["p"] == pytest.approx(1.0)
    better = x + 0.01
    res = ledoit_wolf_sharpe_test(better, x, n_boot=500, seed=1)
    assert res["sharpe_diff"] > 0 and res["p"] < 0.05
    assert res == ledoit_wolf_sharpe_test(better, x, n_boot=500, seed=1)
    jk = jobson_korkie_memmel(better, x)
    assert jk["z"] > 0 and jk["p"] < 0.05


def test_psr_and_dsr_follow_the_published_formulas() -> None:
    r = np.random.default_rng(3).normal(0.003, 0.02, 60)
    sr = r.mean() / r.std(ddof=1)
    skew = stats.skew(r, bias=False)
    kurt = stats.kurtosis(r, fisher=False, bias=False)
    expected = stats.norm.cdf(
        sr * math.sqrt(59) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    )
    assert probabilistic_sharpe(r) == pytest.approx(expected)
    n, v = 20, 0.01
    g = 0.5772156649015329
    e_max = math.sqrt(v) * (
        (1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * math.e))
    )
    assert expected_max_sharpe(n, v) == pytest.approx(e_max)
    assert deflated_sharpe(r, n, v) == pytest.approx(probabilistic_sharpe(r, e_max))
    assert deflated_sharpe(r, n, v) < probabilistic_sharpe(r)


def test_hand_made_portfolio_metrics() -> None:
    daily = pd.Series([0.01, -0.02, 0.005, -0.01, 0.03, -0.04, 0.0, 0.02, -0.005, 0.01])
    down = math.sqrt(np.mean(np.minimum(daily, 0) ** 2))
    assert m.sortino(daily) == pytest.approx(daily.mean() / down * math.sqrt(252))
    assert m.cvar(daily, level=0.9) == pytest.approx(0.04)  # worst 10% of 10 days: one day
    values = pd.Series([100.0, 110.0, 99.0, 105.0, 88.0, 95.0])
    assert m.max_drawdown(values) == pytest.approx(88 / 110 - 1)
    active = pd.Series([0.001, -0.002, 0.003, 0.0, 0.002])
    te = active.std(ddof=1) * math.sqrt(52)
    assert m.tracking_error(active) == pytest.approx(te)
    assert m.information_ratio(active) == pytest.approx(active.mean() * 52 / te)
    weekly = pd.Series([0.01, 0.02, -0.01, 0.005])
    assert m.certainty_equivalent(weekly, 5) == pytest.approx(
        (weekly.mean() - 2.5 * weekly.var(ddof=1)) * 52
    )
    rate, p = m.hit_rate(pd.Series([1.0, 1.0, -1.0, 0.0, 1.0]))
    assert rate == 0.75 and p == pytest.approx(stats.binomtest(3, 4, 0.5).pvalue)
    be, reason = m.break_even_cost(0.01, 0.002, 0.5)
    assert be == pytest.approx(0.024) and reason == ""
    assert np.isnan(m.break_even_cost(0.01, 0.0, 0.0)[0])


def test_block_bootstrap_ci_covers_the_mean_and_paired_summary_columns() -> None:
    x = pd.Series(np.random.default_rng(9).normal(0.002, 0.01, 23))
    lo, hi = block_bootstrap_ci(x, n=500, seed=2)
    assert lo < x.mean() < hi
    s = paired_summary(x, tie_band=0.001, seed=0, n_perm=1000, n_boot=300)
    for col in (
        "n",
        "mean_gap",
        "se_nw",
        "t",
        "p",
        "p_signflip",
        "signflip_method",
        "ci_low",
        "ci_high",
        "wins",
        "ties",
        "losses",
        "mdg",
    ):
        assert col in s
    assert s["signflip_method"] == "exact" and s["n"] == 23


def test_arch_wrappers_run_when_installed() -> None:
    if not m.arch_available():
        pytest.skip("arch not installed")
    rng = np.random.default_rng(1)
    bench = pd.Series(rng.normal(1.0, 0.1, 80))
    models = pd.DataFrame({"a": rng.normal(1.0, 0.1, 80), "b": rng.normal(0.8, 0.1, 80)})
    res = m.spa_test(bench, models, seed=1)
    assert res is not None and res["p_consistent"] < 0.05
    assert "b" in (m.stepm(bench, models, seed=1) or [])
    assert m.model_confidence_set(models, seed=1) == ["b"]
