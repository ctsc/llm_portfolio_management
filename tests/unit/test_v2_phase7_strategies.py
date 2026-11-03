"""v2 phase 7 strategy package tests."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from portstudy.config import OptimizerConfig, StudyConfig
from portstudy.optimize.mean_variance import OptimizerError, solve_mean_variance
from portstudy.optimize.strategies import (
    CONTROLS,
    STRATEGIES,
    cvar,
    equal_weight,
    hrp,
    mvo,
    resampled,
    robust,
    run_control,
    run_strategy,
    topk,
)
from portstudy.optimize.strategies.common import (
    StrategyInputs,
    cap_and_renormalize,
    check_weights,
    clean_weights,
    fallback_chain,
    solve_with,
    standard_normals,
    view_shift,
)


def _inputs(n: int = 20, seed: int = 0) -> StrategyInputs:
    tickers = [f"T{i:02d}" for i in range(n)]
    rng = np.random.default_rng(seed)
    sigma = pd.DataFrame(np.eye(n) * 0.01, index=tickers, columns=tickers)
    return StrategyInputs(
        week_id="2026-W10",
        tickers=tickers,
        mu=pd.Series(rng.normal(0.005, 0.01, n), index=tickers),
        sigma=sigma,
        cov_mu=sigma * 0.1,
        delta=2.5,
    )


def _config(
    *,
    scenarios: int = 64,
    draws: int = 4,
    lam: float = 1.0,
    kappa: float = 1.0,
    optimizer_solvers: list[str] | None = None,
) -> StudyConfig:
    cfg = StudyConfig()
    strategies = cfg.strategies.model_copy(
        update={
            "cvar": cfg.strategies.cvar.model_copy(update={"n_scenarios": scenarios, "lam": lam}),
            "resampled": cfg.strategies.resampled.model_copy(update={"n_draws": draws}),
            "robust": cfg.strategies.robust.model_copy(update={"kappa": kappa}),
        }
    )
    optimizer = cfg.optimizer
    if optimizer_solvers is not None:
        optimizer = optimizer.model_copy(update={"solvers": optimizer_solvers})
    return cfg.model_copy(update={"strategies": strategies, "optimizer": optimizer})


def test_cap_and_renormalize_is_water_filling_and_preserves_order() -> None:
    raw = pd.Series([0.4, 0.3, 0.2, 0.1], index=list("abcd"))

    capped = cap_and_renormalize(raw, 0.35)

    assert isinstance(capped, pd.Series)
    assert capped.sum() == pytest.approx(1.0)
    assert capped.max() <= 0.35 + 1e-12
    assert (capped.diff().dropna() <= 0.0).all()


def test_clean_weights_and_check_weights_enforce_guardrails() -> None:
    cfg = OptimizerConfig(max_weight=0.1, min_stocks=10)
    weights = pd.Series([0.1] * 11, index=[f"T{i}" for i in range(11)])
    weights.iloc[0] = -1e-10
    cleaned = clean_weights(weights, cfg)
    check_weights(cleaned, cfg)
    assert cleaned.sum() == pytest.approx(1.0)
    assert cleaned.min() >= 0.0
    with pytest.raises(OptimizerError, match="position cap"):
        check_weights(pd.Series([0.11] + [0.1] * 8 + [0.09], index=weights.index[:10]), cfg)


def test_fallback_chain_uses_previous_then_q0_then_equal_weight() -> None:
    inputs = _inputs()
    cfg = StudyConfig()
    inputs.previous = pd.Series({"T00": 0.4, "T01": 0.4, "DROPPED": 0.2})
    inputs.q0_weights = pd.Series(1.0 / 20, index=inputs.tickers)

    previous = fallback_chain(inputs, cfg)
    assert previous.fallback_flag == "solver_failed_prev"
    check_weights(previous.weights, cfg.optimizer)
    assert "DROPPED" not in previous.weights.index

    inputs.previous = pd.Series({"DROPPED": 1.0})
    q0 = fallback_chain(inputs, cfg)
    assert q0.fallback_flag == "solver_failed_q0"
    pd.testing.assert_series_equal(q0.weights, inputs.q0_weights.rename("weight"))

    inputs.q0_weights = None
    equal = fallback_chain(inputs, cfg)
    assert equal.fallback_flag == "solver_failed_ew"
    np.testing.assert_allclose(equal.weights.to_numpy(), 1.0 / 20)


def test_solver_failure_walks_fallbacks_from_previous_to_q0_to_equal() -> None:
    inputs = _inputs(20)
    inputs.previous = pd.Series({"T00": 0.4, "T01": 0.4, "DROPPED": 0.2})
    inputs.q0_weights = pd.Series(1.0 / 20, index=inputs.tickers)
    cfg = _config(optimizer_solvers=["NOT_INSTALLED"])

    previous = mvo.run(inputs, cfg)
    assert previous.fallback_flag == "solver_failed_prev"
    assert "DROPPED" not in previous.weights.index

    inputs.previous = None
    q0 = mvo.run(inputs, cfg)
    assert q0.fallback_flag == "solver_failed_q0"

    inputs.q0_weights = None
    ew = mvo.run(inputs, cfg)
    assert ew.fallback_flag == "solver_failed_ew"
    check_weights(ew.weights, cfg.optimizer)


def test_solve_with_uses_first_optimal_installed_solver() -> None:
    value = cp.Variable()
    problem = cp.Problem(cp.Maximize(value), [value <= 1.0])

    solver = solve_with(problem, ["CLARABEL"])

    assert solver == "CLARABEL"
    assert problem.value == pytest.approx(1.0)


def test_standard_normals_are_shared_and_view_shift_is_total_variation() -> None:
    first = standard_normals("2026-W10", "cvar", (4, 3), 7)
    second = standard_normals("2026-W10", "cvar", (4, 3), 7)
    other = standard_normals("2026-W10", "resampled", (4, 3), 7)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)

    weights = pd.Series([0.5, 0.5], index=["A", "B"])
    q0 = pd.Series([1.0, 0.0], index=["A", "B"])
    assert view_shift(weights, q0) == pytest.approx(0.5)


def test_mvo_strategy_matches_v1_optimizer() -> None:
    inputs = _inputs(25)
    cfg = _config()

    result = mvo.run(inputs, cfg)
    reference = solve_mean_variance(inputs.mu, inputs.sigma, inputs.delta, cfg.optimizer)

    check_weights(result.weights, cfg.optimizer)
    pd.testing.assert_series_equal(result.weights, reference.weights)


def test_cvar_lambda_zero_selects_highest_means_and_reports_terms() -> None:
    inputs = _inputs(25)
    inputs.mu = pd.Series(
        [0.001] * 5 + [0.02] * 20,
        index=inputs.tickers,
    )
    cfg = _config(scenarios=64, lam=0.0)

    result = cvar.run(inputs, cfg)

    check_weights(result.weights, cfg.optimizer)
    assert result.weights.loc[inputs.tickers[:5]].max() <= 1e-8
    np.testing.assert_allclose(result.weights.loc[inputs.tickers[5:]], 0.05, atol=1e-7)
    assert {"in_sample_cvar", "return_term", "cvar_term"} <= set(result.diagnostics)


def test_larger_cvar_lambda_does_not_raise_in_sample_cvar() -> None:
    inputs = _inputs(25)
    low_risk_weight = cvar.run(inputs, _config(scenarios=128, lam=0.0))
    higher_risk_weight = cvar.run(inputs, _config(scenarios=128, lam=1.0))

    assert (
        higher_risk_weight.diagnostics["in_sample_cvar"]
        <= low_risk_weight.diagnostics["in_sample_cvar"] + 1e-10
    )


def test_robust_zero_kappa_matches_mvo_and_positive_kappa_reduces_return() -> None:
    inputs = _inputs(25)
    cfg_zero = _config(kappa=0.0)
    cfg_positive = _config(kappa=1.0)

    mvo_result = mvo.run(inputs, cfg_zero)
    robust_zero = robust.run(inputs, cfg_zero)
    robust_positive = robust.run(inputs, cfg_positive)

    np.testing.assert_allclose(robust_zero.weights, mvo_result.weights, atol=1e-4)
    assert (
        robust_positive.diagnostics["return_term"] <= robust_zero.diagnostics["return_term"] + 1e-8
    )


def test_resampled_strategy_converges_to_mvo_as_mean_covariance_vanishes() -> None:
    inputs = _inputs(25)
    inputs.cov_mu = pd.DataFrame(np.eye(25) * 1e-18, index=inputs.tickers, columns=inputs.tickers)
    cfg = _config(draws=8)

    result = resampled.run(inputs, cfg)
    reference = mvo.run(inputs, cfg)

    np.testing.assert_allclose(result.weights, reference.weights, atol=2e-4)
    assert result.diagnostics["failed_draws"] == 0


def test_topk_caps_twenty_names_and_breaks_boundary_ties_by_ticker() -> None:
    inputs = _inputs(25)
    inputs.mu[:] = 0.01
    result = topk.run(inputs, _config())

    selected = result.weights[result.weights > 0.0]
    assert len(selected) == 20
    np.testing.assert_allclose(selected.to_numpy(), 0.05, atol=1e-12)
    assert selected.index.tolist() == inputs.tickers[:20]


def test_strategy_registry_contains_five_steered_strategies() -> None:
    assert set(STRATEGIES) == {"mvo", "cvar", "robust", "resampled", "topk"}
    assert set(CONTROLS) == {"HRP", "EW"}
    result = run_strategy("mvo", _inputs(25), _config())
    control = run_control("EW", _inputs(25), _config())
    check_weights(result.weights, _config().optimizer)
    check_weights(control.weights, _config().optimizer)


def test_equal_weight_control_does_not_depend_on_mu_or_sigma() -> None:
    inputs = _inputs(25)
    cfg = _config()
    first = equal_weight.run(inputs, cfg)
    inputs.mu[:] = np.nan
    inputs.sigma.iloc[:, :] = np.nan
    second = equal_weight.run(inputs, cfg)
    pd.testing.assert_series_equal(first.weights, second.weights)


def test_hrp_is_mu_independent_and_balances_equal_uncorrelated_blocks() -> None:
    inputs = _inputs(40)
    covariance = np.eye(40) * 0.01
    covariance[:20, :20] += 0.009
    covariance[20:, 20:] += 0.009
    inputs.sigma = pd.DataFrame(covariance, index=inputs.tickers, columns=inputs.tickers)
    cfg = _config()

    first = hrp.run(inputs, cfg)
    inputs.mu[:] = np.nan
    second = hrp.run(inputs, cfg)

    pd.testing.assert_series_equal(first.weights, second.weights)
    pre_cap = pd.Series(first.diagnostics["pre_cap_weights"])
    assert pre_cap.loc[inputs.tickers[:20]].sum() == pytest.approx(0.5, abs=0.05)
    assert pre_cap.loc[inputs.tickers[20:]].sum() == pytest.approx(0.5, abs=0.05)


def test_hrp_handles_perfectly_correlated_pair() -> None:
    inputs = _inputs(20)
    sigma = np.eye(20) * 0.01
    sigma[0, 1] = sigma[1, 0] = 0.01
    inputs.sigma = pd.DataFrame(sigma, index=inputs.tickers, columns=inputs.tickers)

    result = hrp.run(inputs, _config())

    check_weights(result.weights, _config().optimizer)


@pytest.mark.parametrize("n", [25, 100, 300])
@pytest.mark.parametrize("seed", range(10))
def test_all_portfolios_pass_guardrails_and_repeat_identically(n: int, seed: int) -> None:
    inputs = _inputs(n, seed)
    cfg = _config(scenarios=32, draws=2)

    for name in STRATEGIES:
        first = run_strategy(name, inputs, cfg)
        second = run_strategy(name, inputs, cfg)
        check_weights(first.weights, cfg.optimizer)
        np.testing.assert_array_equal(first.weights.to_numpy(), second.weights.to_numpy())
    for name in CONTROLS:
        first = run_control(name, inputs, cfg)
        second = run_control(name, inputs, cfg)
        check_weights(first.weights, cfg.optimizer)
        np.testing.assert_array_equal(first.weights.to_numpy(), second.weights.to_numpy())


def test_cvar_solves_when_every_scenario_is_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _inputs(25)
    monkeypatch.setattr(
        cvar,
        "standard_normals",
        lambda week_id, strategy, shape, seed: np.zeros(shape),
    )

    result = cvar.run(inputs, _config(scenarios=16))

    check_weights(result.weights, _config().optimizer)


def test_cvar_cholesky_failure_uses_the_fallback_chain() -> None:
    inputs = _inputs(20)
    inputs.cov_mu = -2.0 * inputs.sigma

    result = cvar.run(inputs, _config(scenarios=16))

    assert result.fallback_flag == "solver_failed_ew"
    check_weights(result.weights, _config().optimizer)


def test_insufficient_universe_fails_before_strategy_solving() -> None:
    with pytest.raises(OptimizerError, match="at least 20"):
        mvo.run(_inputs(19), _config())
