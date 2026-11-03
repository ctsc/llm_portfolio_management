"""v2 phase 6 tests for view uncertainty and the BL posterior covariance."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from portstudy.config import ViewsConfig
from portstudy.optimize.black_litterman import (
    black_litterman,
    black_litterman_q0,
)
from portstudy.optimize.mean_variance import OptimizerError
from portstudy.views.build import build_views, views_frame, width_to_c


def _case(n: int = 30, seed: int = 52) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(n)]
    raw = rng.normal(size=(n, n))
    sigma_values = raw @ raw.T / (10 * n) + 0.01 * np.eye(n)
    sigma = pd.DataFrame(sigma_values, index=tickers, columns=tickers)
    sigma_vec = pd.Series(np.sqrt(np.diag(sigma_values)), index=tickers)
    pi = pd.Series(rng.normal(0.002, 0.001, n), index=tickers)
    z_b = pd.Series(rng.normal(size=n), index=tickers)
    return sigma, pi, sigma_vec, z_b


def test_equal_spreads_keep_the_v1_q0_posterior() -> None:
    sigma, pi, sigma_vec, z_b = _case()
    cfg = ViewsConfig()
    k_sd = pd.Series(0.02, index=pi.index)
    views = build_views(pi, sigma_vec, z_b, 0.04, cfg, k_sd=k_sd)

    np.testing.assert_allclose(views.ratio.to_numpy(), 1.0)
    np.testing.assert_allclose(
        views.omega_b.to_numpy(), cfg.tau * np.diag(sigma.to_numpy()), rtol=1e-14
    )

    posterior = black_litterman_q0(views, sigma)
    prior_precision = np.linalg.inv(cfg.tau * sigma.to_numpy())
    omega_precision = np.diag(1.0 / views.omega_b.to_numpy())
    a = prior_precision + omega_precision
    b = prior_precision @ pi.to_numpy() + omega_precision @ views.q_b.to_numpy()
    expected = np.linalg.solve(a, b)
    np.testing.assert_allclose(posterior.mu_bl.to_numpy(), expected, rtol=1e-12, atol=1e-12)


def test_kronos_spread_ratio_uses_median_and_clips_both_ends() -> None:
    sigma, pi, sigma_vec, z_b = _case(n=5)
    cfg = ViewsConfig()
    k_sd = pd.Series([1.0, 1.0, 1.0, 2.0, 3.0], index=pi.index)
    views = build_views(pi, sigma_vec, z_b, 0.04, cfg, k_sd=k_sd)
    assert views.ratio.loc["T3"] == pytest.approx(2.0)
    assert views.omega_b.loc["T3"] == pytest.approx(2 * cfg.tau * sigma_vec.loc["T3"] ** 2)

    clipped = build_views(
        pi,
        sigma_vec,
        z_b,
        0.04,
        cfg,
        k_sd=pd.Series([0.01, 1.0, 1.0, 1.0, 100.0], index=pi.index),
    )
    np.testing.assert_allclose(clipped.ratio.to_numpy(), [0.25, 1.0, 1.0, 1.0, 4.0])


def test_missing_and_degenerate_kronos_spreads_default_ratio_to_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sigma, pi, sigma_vec, z_b = _case(n=4)
    cfg = ViewsConfig()
    with caplog.at_level(logging.INFO):
        all_missing = build_views(
            pi, sigma_vec, z_b, 0.04, cfg, k_sd=pd.Series(np.nan, index=pi.index)
        )
        zeros = build_views(pi, sigma_vec, z_b, 0.04, cfg, k_sd=pd.Series(0.0, index=pi.index))
        one_missing = build_views(
            pi,
            sigma_vec,
            z_b,
            0.04,
            cfg,
            k_sd=pd.Series([0.01, np.nan, 0.01, 0.02], index=pi.index),
        )

    np.testing.assert_allclose(all_missing.ratio.to_numpy(), 1.0)
    np.testing.assert_allclose(zeros.ratio.to_numpy(), 1.0)
    assert one_missing.ratio.loc["T1"] == 1.0
    assert "baseline_spread_unavailable" in caplog.text
    assert "baseline_spread_degenerate" in caplog.text
    assert "baseline_spread_missing" in caplog.text


def test_width_maps_to_c_and_rejects_unvalidated_values() -> None:
    width = pd.Series([0.0, 4.0, 10.0], index=["narrow", "middle", "wide"])
    c = width_to_c(width, ViewsConfig())
    np.testing.assert_allclose(c.to_numpy(), [1.0, 0.6, 0.1])
    with pytest.raises(AssertionError, match=r"\[0, 10\]"):
        width_to_c(pd.Series([-1.0]), ViewsConfig())
    with pytest.raises(AssertionError, match=r"\[0, 10\]"):
        width_to_c(pd.Series([11.0]), ViewsConfig())


def test_views_frame_includes_uncertainty_inputs() -> None:
    _, pi, sigma_vec, z_b = _case(n=4)
    width = pd.Series([0.0, 4.0], index=pi.index[:2])
    views = build_views(
        pi,
        sigma_vec,
        z_b,
        0.04,
        ViewsConfig(),
        k_sd=pd.Series(0.02, index=pi.index),
        z_l=pd.Series([1.0, -1.0], index=pi.index[:2]),
        ic_l=0.03,
        width=width,
    )
    frame = views_frame(views).set_index("ticker")
    assert {"k_sd", "ratio", "width", "c"} <= set(frame.columns)
    assert frame.loc["T0", "c"] == pytest.approx(1.0)
    assert frame.loc["T1", "c"] == pytest.approx(0.6)
    assert pd.isna(frame.loc["T2", "width"]) and pd.isna(frame.loc["T2", "c"])


def test_width_and_c_are_available_when_llm_views_are_disabled() -> None:
    _, pi, sigma_vec, z_b = _case(n=4)
    width = pd.Series([0.0, 4.0, 10.0, 5.0], index=pi.index)
    z_l = pd.Series(0.5, index=pi.index)
    views = build_views(
        pi,
        sigma_vec,
        z_b,
        0.04,
        ViewsConfig(),
        k_sd=pd.Series(0.02, index=pi.index),
        z_l=z_l,
        ic_l=0.0,
        width=width,
    )
    assert not views.use_llm_view
    pd.testing.assert_series_equal(views.width, width)
    pd.testing.assert_series_equal(views.c, width_to_c(width, ViewsConfig()))
    assert views_frame(views)["q_L"].isna().all()


def test_posterior_covariance_is_symmetric_positive_definite_and_prior_without_views() -> None:
    sigma, pi, sigma_vec, z_b = _case()
    cfg = ViewsConfig()
    views = build_views(pi, sigma_vec, z_b, 0.0, cfg, k_sd=pd.Series(0.02, index=pi.index))
    posterior = black_litterman(views, sigma)

    np.testing.assert_allclose(posterior.mu_bl.to_numpy(), pi.to_numpy(), atol=1e-12)
    np.testing.assert_allclose(posterior.cov_mu.to_numpy(), cfg.tau * sigma.to_numpy(), atol=1e-12)
    np.testing.assert_allclose(
        posterior.cov_mu.to_numpy(), posterior.cov_mu.to_numpy().T, atol=1e-14
    )
    assert np.linalg.eigvalsh(posterior.cov_mu.to_numpy()).min() > 0


def test_adding_views_does_not_increase_posterior_variance_diagonal() -> None:
    sigma, pi, sigma_vec, z_b = _case()
    cfg = ViewsConfig()
    k_sd = pd.Series(np.linspace(0.01, 0.05, len(pi)), index=pi.index)
    prior = build_views(pi, sigma_vec, z_b, 0.0, cfg, k_sd=k_sd)
    with_views = build_views(
        pi,
        sigma_vec,
        z_b,
        0.03,
        cfg,
        k_sd=k_sd,
        z_l=pd.Series(np.linspace(-1.0, 1.0, len(pi)), index=pi.index),
        ic_l=0.02,
        width=pd.Series(np.linspace(0.0, 10.0, len(pi)), index=pi.index),
    )
    prior_cov = black_litterman(prior, sigma).cov_mu
    posterior_cov = black_litterman(with_views, sigma).cov_mu
    assert (np.diag(posterior_cov) <= np.diag(prior_cov) + 1e-12).all()


def test_posterior_mean_and_covariance_match_textbook_formula() -> None:
    sigma, pi, sigma_vec, z_b = _case(n=30, seed=103)
    cfg = ViewsConfig()
    rng = np.random.default_rng(104)
    valid = pi.index[::2]
    z_l = pd.Series(rng.normal(size=len(valid)), index=valid)
    width = pd.Series(rng.uniform(0.0, 10.0, size=len(valid)), index=valid)
    k_sd = pd.Series(rng.lognormal(mean=-4.0, sigma=0.5, size=len(pi)), index=pi.index)
    views = build_views(
        pi,
        sigma_vec,
        z_b,
        0.04,
        cfg,
        k_sd=k_sd,
        z_l=z_l,
        ic_l=0.03,
        width=width,
    )
    posterior = black_litterman(views, sigma)

    n = len(pi)
    idx = [list(pi.index).index(ticker) for ticker in valid]
    p_l = np.zeros((len(valid), n))
    p_l[np.arange(len(valid)), idx] = 1.0
    p = np.vstack([np.eye(n), p_l])
    q = np.concatenate([views.q_b.to_numpy(), views.q_l.to_numpy()])
    omega = np.diag(np.concatenate([views.omega_b.to_numpy(), views.omega_l.to_numpy()]))
    tau_sigma_inv = np.linalg.inv(cfg.tau * sigma.to_numpy())
    a = tau_sigma_inv + p.T @ np.linalg.inv(omega) @ p
    b = tau_sigma_inv @ pi.to_numpy() + p.T @ np.linalg.inv(omega) @ q
    expected_mu = np.linalg.solve(a, b)
    expected_cov = np.linalg.inv(a)

    np.testing.assert_allclose(posterior.mu_bl.to_numpy(), expected_mu, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(posterior.cov_mu.to_numpy(), expected_cov, rtol=1e-10, atol=1e-10)


def test_cholesky_failure_retries_once_with_jitter_and_then_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sigma, pi, sigma_vec, z_b = _case(n=5)
    views = build_views(
        pi, sigma_vec, z_b, 0.0, ViewsConfig(), k_sd=pd.Series(0.02, index=pi.index)
    )
    original = np.linalg.cholesky
    calls = 0

    def fail_once(matrix: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise np.linalg.LinAlgError("forced initial failure")
        return original(matrix)

    monkeypatch.setattr(np.linalg, "cholesky", fail_once)
    with caplog.at_level(logging.WARNING):
        black_litterman(views, sigma)
    assert calls == 2
    assert "black_litterman_cholesky_jitter" in caplog.text

    def always_fail(_matrix: np.ndarray) -> np.ndarray:
        raise np.linalg.LinAlgError("forced factorization failure")

    monkeypatch.setattr(np.linalg, "cholesky", always_fail)
    with pytest.raises(OptimizerError, match="not positive definite"):
        black_litterman(views, sigma)
