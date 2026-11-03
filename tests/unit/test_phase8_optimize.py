"""Phase 8 tests: Black-Litterman posterior, mean-variance weights, Q0 equivalence rules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portstudy.config import ModelRegistry, OptimizerConfig, StudyConfig, ViewsConfig
from portstudy.optimize import WeekInputs, build_q0_weights, build_weights, read_weights
from portstudy.optimize.black_litterman import black_litterman, black_litterman_q0
from portstudy.optimize.mean_variance import OptimizerError, solve_mean_variance
from portstudy.views.build import build_views


def _case(n: int = 40, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(n)]
    f = rng.normal(size=(252, 3))
    b = rng.normal(size=(n, 3))
    rets = f @ b.T * 0.01 + rng.normal(size=(252, n)) * 0.01
    sigma = pd.DataFrame(5 * np.cov(rets.T), index=tickers, columns=tickers)
    w_mkt = rng.uniform(0.5, 2.0, n)
    w_mkt /= w_mkt.sum()
    pi = pd.Series(2.5 * sigma.to_numpy() @ w_mkt, index=tickers)
    sigma_vec = pd.Series(np.sqrt(np.diag(sigma.to_numpy())), index=tickers)
    z_b = pd.Series(rng.normal(size=n), index=tickers)
    return sigma, pi, sigma_vec, z_b


def test_no_views_gives_pi() -> None:
    sigma, pi, sv, z_b = _case()
    v = build_views(pi, sv, z_b, ic_b=0.0, cfg=ViewsConfig())
    post = black_litterman(v, sigma)
    np.testing.assert_allclose(post.mu_bl.to_numpy(), pi.to_numpy())
    assert post.n_views == 0 and np.allclose(post.tilt.to_numpy(), 0.0)


def test_confident_llm_view_moves_mu_toward_q_l() -> None:
    sigma, pi, sv, z_b = _case()
    cfg = ViewsConfig()
    target = "T7"
    z_l = pd.Series(0.0, index=pi.index)
    z_l[target] = 3.0
    conf = pd.Series(0.1, index=pi.index)
    conf[target] = 1.0
    v = build_views(pi, sv, z_b, ic_b=0.0, cfg=cfg, z_l=z_l, confidence=conf, ic_l=0.1)
    post = black_litterman(v, sigma)
    q_l_target = v.q_l[target]
    assert pi[target] < post.mu_bl[target] <= q_l_target + 1e-12
    # Higher confidence pulls closer to q_L.
    conf_low = conf.copy()
    conf_low[target] = 0.1
    v_low = build_views(pi, sv, z_b, ic_b=0.0, cfg=cfg, z_l=z_l, confidence=conf_low, ic_l=0.1)
    post_low = black_litterman(v_low, sigma)
    assert abs(post.mu_bl[target] - q_l_target) < abs(post_low.mu_bl[target] - q_l_target)


def test_posterior_matches_textbook_formula() -> None:
    sigma, pi, sv, z_b = _case(n=25, seed=3)
    cfg = ViewsConfig()
    rng = np.random.default_rng(4)
    valid = pi.index[::2]
    z_l = pd.Series(rng.normal(size=len(valid)), index=valid)
    conf = pd.Series(rng.uniform(0.2, 1.0, len(valid)), index=valid)
    v = build_views(pi, sv, z_b, ic_b=0.04, cfg=cfg, z_l=z_l, confidence=conf, ic_l=0.03)
    post = black_litterman(v, sigma)
    # Textbook: explicit inverses with P = [I; I_L].
    n = len(pi)
    idx = [list(pi.index).index(t) for t in valid]
    p_l = np.zeros((len(valid), n))
    p_l[np.arange(len(valid)), idx] = 1.0
    p = np.vstack([np.eye(n), p_l])
    q = np.concatenate([v.q_b.to_numpy(), v.q_l.to_numpy()])
    omega = np.diag(np.concatenate([v.omega_b.to_numpy(), v.omega_l.to_numpy()]))
    ts_inv = np.linalg.inv(cfg.tau * sigma.to_numpy())
    a = ts_inv + p.T @ np.linalg.inv(omega) @ p
    b = ts_inv @ pi.to_numpy() + p.T @ np.linalg.inv(omega) @ q
    expected = np.linalg.solve(a, b)
    np.testing.assert_allclose(post.mu_bl.to_numpy(), expected, rtol=1e-8, atol=1e-12)
    assert post.n_views == n + len(valid)


def test_q0_equals_bl_without_llm_rows() -> None:
    sigma, pi, sv, z_b = _case()
    cfg = ViewsConfig()
    z_l = pd.Series(np.random.default_rng(1).normal(size=len(pi)), index=pi.index)
    v = build_views(pi, sv, z_b, ic_b=0.03, cfg=cfg, z_l=z_l, confidence=None, ic_l=0.05)
    only_b = build_views(pi, sv, z_b, ic_b=0.03, cfg=cfg)
    np.testing.assert_allclose(black_litterman_q0(v, sigma).mu_bl.to_numpy(),
                               black_litterman(only_b, sigma).mu_bl.to_numpy())


def test_weights_sum_to_one_capped_and_deterministic() -> None:
    sigma, pi, sv, z_b = _case(n=60, seed=9)
    mu = pi + 0.01 * z_b
    cfg = OptimizerConfig()
    r1 = solve_mean_variance(mu, sigma, 2.5, cfg)
    r2 = solve_mean_variance(mu, sigma, 2.5, cfg)
    assert r1.weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert (r1.weights >= -1e-12).all() and (r1.weights <= cfg.max_weight + 1e-9).all()
    pd.testing.assert_series_equal(r1.weights, r2.weights)
    assert r1.fallback_flag is None and r1.solver in cfg.solvers
    # CLARABEL agrees with OSQP to solver tolerance.
    r3 = solve_mean_variance(mu, sigma, 2.5, cfg, solvers=["CLARABEL"])
    assert np.abs(r1.weights.to_numpy() - r3.weights.to_numpy()).max() < 1e-3


def test_min_stocks_guard_and_solver_failure_fallback() -> None:
    sigma, pi, sv, z_b = _case(n=15)
    with pytest.raises(OptimizerError):
        solve_mean_variance(pi, sigma, 2.5, OptimizerConfig())
    sigma, pi, sv, z_b = _case(n=30, seed=2)
    prev = pd.Series(1.0 / 30, index=pi.index)
    r = solve_mean_variance(pi, sigma, 2.5, OptimizerConfig(), previous=prev, solvers=["NO_SUCH_SOLVER"])
    assert r.fallback_flag == "solver_failed" and r.solver == "none"
    pd.testing.assert_series_equal(r.weights, prev.rename("weight"))
    with pytest.raises(OptimizerError):
        solve_mean_variance(pi, sigma, 2.5, OptimizerConfig(), solvers=["NO_SUCH_SOLVER"])


@pytest.fixture()
def week_inputs(work_data_dir: Path, study_config: StudyConfig, model_registry: ModelRegistry,
                session_weeks: list[str]) -> tuple[str, WeekInputs, StudyConfig]:
    from portstudy.baseline.combine import build_baseline
    from portstudy.data.snapshot import read_snapshot
    from portstudy.llm.cache import LlmCache
    from portstudy.llm.providers.stub import StubFailures, StubLlmProvider
    from portstudy.llm.runner import WeekRunner
    from portstudy.risk.covariance import build_risk_model
    from portstudy.risk.prior import build_prior

    cfg = study_config.model_copy(update={"llm": study_config.llm.model_copy(update={"cost_goal_usd": 1e9,
                                                                                  "cost_warn_usd": 1e9})})
    w = session_weeks[1]
    snap = read_snapshot(w)
    base = build_baseline(w, cfg, snap)
    risk = build_risk_model(w, cfg.risk, snap, tickers=list(base["ticker"]))
    build_prior(w, cfg.risk, risk, snap.reference)
    # stub-local-b answers nothing valid, to exercise the all-invalid rule.
    providers = {"stub-local-b": StubLlmProvider("stub-local-b", failures=StubFailures(
        always_invalid_tickers=set(snap.included_tickers)))}
    runner = WeekRunner(w, cfg, model_registry, snap, synthetic=True, providers=providers,
                        cache=LlmCache(work_data_dir / "cache"))
    runner.run_local("stub-local-a")
    runner.run_local("stub-local-b")
    return w, WeekInputs.load(w), cfg


def test_build_weights_every_portfolio_and_q0_rules(week_inputs: tuple[str, WeekInputs, StudyConfig]) -> None:
    w, inputs, cfg = week_inputs
    q0 = build_q0_weights(w, cfg, ic_b=0.02, inputs=inputs)
    assert q0.weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert read_weights(w, "mvo__Q0")["weight"].sum() == pytest.approx(1.0, abs=1e-8)
    assert not q0.use_llm_view
    # Normal portfolio with an LLM view differs from Q0.
    p = build_weights(w, "stub-local-a", "skills", cfg, ic_b=0.02, ic_l=0.03, inputs=inputs)
    assert p.use_llm_view and p.n_llm_view == len(inputs.tickers)
    assert not np.allclose(p.weights.to_numpy(), q0.weights.to_numpy())
    assert (p.weights <= cfg.optimizer.max_weight + 1e-9).all()
    saved_ok = read_weights(w, "mvo__stub-local-a__skills")
    assert saved_ok["has_llm_view"].all() and saved_ok["z_L"].notna().all()
    # IC_L == 0: use_llm_view False and the weights equal Q0's exactly.
    z = build_weights(w, "stub-local-a", "skills", cfg, ic_b=0.02, ic_l=0.0, inputs=inputs)
    assert not z.use_llm_view and "weights_equal_q0" in z.notes
    pd.testing.assert_series_equal(z.weights, q0.weights)
    pd.testing.assert_series_equal(z.mu_bl, q0.mu_bl)
    # Every stock invalid: no LLM view rows, weights equal Q0's.
    inv = build_weights(w, "stub-local-b", "bare", cfg, ic_b=0.02, ic_l=0.03, inputs=inputs)
    assert inv.n_invalid == len(inputs.scores_for("stub-local-b", "bare"))
    assert not inv.use_llm_view
    pd.testing.assert_series_equal(inv.weights, q0.weights)
    saved = read_weights(w, "mvo__stub-local-b__bare")
    assert saved["z_L"].isna().all() and not saved["has_llm_view"].any()
    # Deterministic.
    again = build_weights(w, "stub-local-a", "skills", cfg, ic_b=0.02, ic_l=0.03, inputs=inputs)
    pd.testing.assert_series_equal(again.weights, p.weights)
