"""Phase 7 tests: rank IC, IC store and trailing IC rules, residualization, views."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from portstudy import calendar as cal
from portstudy.config import ModelRegistry, StudyConfig, ViewsConfig
from portstudy.views.build import build_views, range_certainty, views_frame
from portstudy.views.ic import ICStore, rank_ic, trailing_ic, weekly_returns
from portstudy.views.residualize import residualize, valid_scores


def test_rank_ic_matches_scipy_spearman() -> None:
    rng = np.random.default_rng(1)
    s = pd.Series(rng.integers(-5, 6, 60).astype(float), index=[f"T{i}" for i in range(60)])
    r = pd.Series(rng.normal(size=60), index=s.index)
    expected = stats.spearmanr(s.to_numpy(), r.to_numpy()).statistic
    assert rank_ic(s, r) == pytest.approx(expected)
    assert np.isnan(rank_ic(s.iloc[:9], r.iloc[:9]))
    assert np.isnan(rank_ic(pd.Series(1.0, index=s.index), r))


def test_weekly_returns_use_opens_dividends_and_splits() -> None:
    w = "2026-W10"
    t0, t1 = cal.monday_open_day(w), cal.window_end_date(w)
    ledger = pd.DataFrame({
        "date": [t0, t1, t0, t1, t0, t1],
        "ticker": ["A", "A", "B", "B", "C", "C"],
        "open": [100.0, 110.0, 50.0, 26.0, 20.0, 21.0],
        "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0,
    })
    actions = pd.DataFrame({"date": [t0 + dt.timedelta(days=2), t0 + dt.timedelta(days=3)],
                            "ticker": ["A", "B"], "action": ["dividend", "split"], "value": [1.0, 2.0]})
    out = weekly_returns(ledger, w, actions).set_index("ticker")["ret"]
    assert out["A"] == pytest.approx((110 + 1.0) / 100 - 1)
    assert out["B"] == pytest.approx(26 * 2 / 50 - 1)
    assert out["C"] == pytest.approx(0.05)


def test_ic_store_upsert_is_idempotent(tmp_path: Path) -> None:
    store = ICStore(tmp_path / "ic.csv")
    store.upsert([{"week_id": "2026-W05", "signal": "baseline", "model": "", "config_id": "",
                   "rank_ic": 0.1, "n_stocks": 100}])
    store.upsert([{"week_id": "2026-W05", "signal": "baseline", "model": "", "config_id": "",
                   "rank_ic": 0.2, "n_stocks": 100}])
    df = store.read()
    assert len(df) == 1 and df.loc[0, "rank_ic"] == 0.2
    assert df.loc[0, "trade_date"] == cal.monday_open_day("2026-W05").isoformat()


def _store_with(tmp_path: Path, weeks: list[str], ics: list[float]) -> ICStore:
    store = ICStore(tmp_path / "ic.csv")
    store.upsert([{"week_id": w, "signal": "llm_resid", "model": "m", "config_id": "skills",
                   "rank_ic": ic, "n_stocks": 100} for w, ic in zip(weeks, ics)])
    return store


def test_cold_start_rule(tmp_path: Path) -> None:
    cfg = ViewsConfig()
    live = "2026-W10"
    weeks = cal.week_range(live, 20)
    store = _store_with(tmp_path, weeks, [0.1] * 20)
    # Weeks 1 to 5 use the fixed cold-start IC regardless of history.
    for w in weeks[:5]:
        assert trailing_ic(store, "llm_resid", w, cfg, live, "m", "skills") == 0.02
    # Week 6 has only weeks 1..4 closed before its cutoff (one-week lag): 4 clean weeks >= 3.
    assert trailing_ic(store, "llm_resid", weeks[5], cfg, live, "m", "skills") == pytest.approx(0.1)
    # Fewer than 3 clean weeks keeps the cold-start value.
    store2 = _store_with(tmp_path / "b", weeks[:2], [0.3, 0.3])
    assert trailing_ic(store2, "llm_resid", weeks[7], cfg, live, "m", "skills") == 0.02
    # Floor at zero.
    store3 = _store_with(tmp_path / "c", weeks, [-0.2] * 20)
    assert trailing_ic(store3, "llm_resid", weeks[10], cfg, live, "m", "skills") == 0.0
    # Window of 8: weeks 1..18 are closed for week 20, the mean uses the last 8 only.
    ics = [0.5] * 10 + [0.0] * 10
    store4 = _store_with(tmp_path / "d", weeks, ics)
    assert trailing_ic(store4, "llm_resid", weeks[19], cfg, live, "m", "skills") == pytest.approx(0.0)


def test_fallback_weeks_excluded_from_trailing_ic(tmp_path: Path) -> None:
    cfg = ViewsConfig()
    live = "2026-W10"
    weeks = cal.week_range(live, 12)
    ics = [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    store = _store_with(tmp_path, weeks, ics)
    with_fb = trailing_ic(store, "llm_resid", weeks[9], cfg, live, "m", "skills", fallback_weeks={weeks[3]})
    without = trailing_ic(store, "llm_resid", weeks[9], cfg, live, "m", "skills")
    assert with_fb == pytest.approx(0.1)
    assert without > with_fb


def test_look_ahead_guard(tmp_path: Path) -> None:
    cfg = ViewsConfig()
    live = "2026-W10"
    weeks = cal.week_range(live, 12)
    # Only week 8 (index 7) has a huge IC; its window closes after week 9's cutoff.
    ics = [0.1] * 7 + [5.0] + [0.1] * 4
    store = _store_with(tmp_path, weeks, ics)
    assert trailing_ic(store, "llm_resid", weeks[8], cfg, live, "m", "skills") == pytest.approx(0.1)
    # Two weeks later the window has closed and the value enters the mean.
    assert trailing_ic(store, "llm_resid", weeks[9], cfg, live, "m", "skills") > 0.5


def test_residual_is_uncorrelated_with_zb_and_skips_invalid() -> None:
    rng = np.random.default_rng(3)
    tickers = [f"T{i}" for i in range(80)]
    z_b = pd.Series(rng.normal(size=80), index=tickers)
    likely = np.rint(2 * z_b.to_numpy() + rng.normal(size=80)).clip(-5, 5).astype(int)
    low = np.maximum(-5, likely - 1)
    high = np.minimum(5, likely + 1)
    scores = pd.DataFrame({
        "ticker": tickers,
        "low": low,
        "likely": likely,
        "high": high,
        "width": high - low,
        "valid": True,
        "fallback_flag": [None] * 80,
    })
    scores.loc[:9, "valid"] = False
    scores.loc[:9, "fallback_flag"] = "invalid_output"
    scores.loc[:9, ["low", "likely", "high", "width"]] = 0
    v = valid_scores(scores)
    assert list(v.columns) == ["ticker", "low", "likely", "high", "width"]
    assert len(v) == 70 and not set(v["ticker"]) & set(tickers[:10])
    z_l = residualize(v, z_b)
    assert len(z_l) == 70
    assert abs(np.corrcoef(z_l.to_numpy(), z_b.loc[z_l.index].to_numpy())[0, 1]) < 1e-8
    assert z_l.mean() == pytest.approx(0.0, abs=1e-12) and z_l.std(ddof=1) == pytest.approx(1.0)
    # Constant residual gives zeros.
    const = pd.DataFrame({"ticker": tickers, "likely": 2 * z_b.to_numpy()})
    assert (residualize(const, z_b) == 0).all()


def test_range_certainty_decreases_with_width() -> None:
    width = pd.Series([0, 5, 10], index=["narrow", "middle", "wide"])
    certainty = range_certainty(width, ViewsConfig())
    np.testing.assert_allclose(certainty.to_numpy(), [1.0, 0.5, 0.1])


def _toy_inputs(n: int = 30) -> tuple[pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(5)
    tickers = [f"T{i}" for i in range(n)]
    pi = pd.Series(rng.uniform(0.001, 0.004, n), index=tickers)
    sigma = pd.Series(rng.uniform(0.02, 0.06, n), index=tickers)
    z_b = pd.Series(rng.normal(size=n), index=tickers)
    return pi, sigma, z_b


def test_zero_ic_gives_views_equal_to_pi_and_drops_rows() -> None:
    pi, sigma, z_b = _toy_inputs()
    cfg = ViewsConfig()
    z_l = pd.Series(np.random.default_rng(6).normal(size=30), index=pi.index)
    conf = pd.Series(0.7, index=pi.index)
    v = build_views(pi, sigma, z_b, ic_b=0.0, cfg=cfg, z_l=z_l, confidence=conf, ic_l=0.0)
    pd.testing.assert_series_equal(v.q_b, pi, check_names=False)
    assert v.use_llm_view is False and v.use_baseline_view is False
    assert "llm_view_dropped:ic_zero" in v.notes and "baseline_view_dropped" in v.notes
    assert len(v.q_l) == 0 and v.llm_tickers == []


def test_views_formulas_and_per_stock_rule() -> None:
    pi, sigma, z_b = _toy_inputs()
    cfg = ViewsConfig()
    valid = pi.index[5:]
    z_l = pd.Series(np.random.default_rng(7).normal(size=len(valid)), index=valid)
    conf = pd.Series(np.linspace(0.0, 1.2, len(valid)), index=valid)
    v = build_views(pi, sigma, z_b, ic_b=0.03, cfg=cfg, z_l=z_l, confidence=conf, ic_l=0.05)
    np.testing.assert_allclose(v.q_b.to_numpy(), (pi + 0.03 * sigma * z_b).to_numpy())
    np.testing.assert_allclose(v.omega_b.to_numpy(), (cfg.tau * sigma**2).to_numpy())
    assert v.use_llm_view and v.llm_tickers == list(valid)
    np.testing.assert_allclose(v.q_l.to_numpy(), (pi.loc[valid] + 0.05 * sigma.loc[valid] * z_l).to_numpy())
    clipped = conf.clip(0.1, 1.0)
    np.testing.assert_allclose(v.omega_l.to_numpy(), (cfg.tau * sigma.loc[valid] ** 2 / clipped).to_numpy())
    frame = views_frame(v)
    assert frame["has_llm_view"].sum() == len(valid)
    assert frame.loc[~frame["has_llm_view"], "q_L"].isna().all()
    # Every stock invalid: no LLM view at all.
    v2 = build_views(pi, sigma, z_b, ic_b=0.03, cfg=cfg, z_l=pd.Series(dtype=float), confidence=None, ic_l=0.05)
    assert v2.use_llm_view is False and "llm_view_dropped:no_valid_outputs" in v2.notes


def test_views_build_for_every_portfolio_over_synthetic_weeks(work_data_dir: Path, study_config: StudyConfig,
                                                              model_registry: ModelRegistry,
                                                              session_weeks: list[str]) -> None:
    from portstudy.baseline.combine import build_baseline
    from portstudy.data.snapshot import read_snapshot
    from portstudy.llm.cache import LlmCache
    from portstudy.llm.prompts import load_manifest
    from portstudy.llm.runner import WeekRunner, read_scores
    from portstudy.risk.covariance import build_risk_model
    from portstudy.risk.prior import build_prior

    cfg = study_config.model_copy(update={"llm": study_config.llm.model_copy(update={"cost_goal_usd": 1e9,
                                                                                  "cost_warn_usd": 1e9})})
    for w in session_weeks[:2]:
        snap = read_snapshot(w)
        base = build_baseline(w, cfg, snap).set_index("ticker")
        risk = build_risk_model(w, cfg.risk, snap, tickers=list(base.index))
        prior = build_prior(w, cfg.risk, risk, snap.reference).set_index("ticker")
        runner = WeekRunner(w, cfg, model_registry, snap, synthetic=True, cache=LlmCache(work_data_dir / "cache"))
        for spec in model_registry.enabled(synthetic=True):
            if spec.kind == "local":
                runner.run_local(spec.id)
            else:
                runner.submit_api(spec.id)
                runner.collect_api(spec.id)
        scores = read_scores(w)
        n_built = 0
        for spec in model_registry.enabled(synthetic=True):
            for conf in load_manifest().enabled():
                sub = scores[(scores["model"] == spec.id) & (scores["config_id"] == conf.id)]
                v = valid_scores(sub)
                z_l = residualize(v, base["z_b"])
                certainty = range_certainty(v.set_index("ticker")["width"], cfg.views)
                views = build_views(
                    prior["pi"],
                    prior["sigma"],
                    base["z_b"],
                    ic_b=0.02,
                    cfg=cfg.views,
                    z_l=z_l,
                    confidence=certainty,
                    ic_l=0.02,
                    label=f"{spec.id}__{conf.id}",
                )
                assert views.use_llm_view and views.use_baseline_view
                assert len(views.q_l) == len(z_l) and views.q_b.notna().all()
                n_built += 1
        assert n_built == len(model_registry.enabled(synthetic=True)) * len(load_manifest().enabled())
