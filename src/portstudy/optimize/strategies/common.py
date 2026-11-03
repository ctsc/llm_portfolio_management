"""Shared inputs, constraints, solver handling, and determinism for portfolio strategies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import cvxpy as cp
import numpy as np
import pandas as pd

from portstudy.config import OptimizerConfig, StudyConfig
from portstudy.optimize.mean_variance import OptimizerError


@dataclass
class StrategyInputs:
    week_id: str
    tickers: list[str]
    mu: pd.Series
    sigma: pd.DataFrame
    cov_mu: pd.DataFrame
    delta: float
    previous: pd.Series | None = None
    q0_weights: pd.Series | None = None


@dataclass
class StrategyResult:
    weights: pd.Series
    solver: str
    objective: float
    fallback_flag: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def validate_universe(inputs: StrategyInputs, cfg: StudyConfig) -> None:
    tickers = inputs.tickers
    n = len(tickers)
    if n < cfg.optimizer.min_stocks:
        raise OptimizerError(
            f"{n} stocks available, need at least {cfg.optimizer.min_stocks} for the cap"
        )
    if cfg.optimizer.max_weight * n < 1.0:
        raise OptimizerError("max_weight times the stock count is below 1; infeasible")
    if len(set(tickers)) != n:
        raise OptimizerError("strategy ticker universe must be unique")


def validate_inputs(inputs: StrategyInputs, cfg: StudyConfig) -> None:
    validate_universe(inputs, cfg)
    tickers = inputs.tickers
    n = len(tickers)
    if not np.isfinite(inputs.delta) or inputs.delta <= 0:
        raise OptimizerError("strategy delta must be positive and finite")
    if not np.isfinite(inputs.mu.reindex(tickers).to_numpy(dtype=float)).all():
        raise OptimizerError("strategy expected returns must be finite and aligned")
    for matrix in (inputs.sigma, inputs.cov_mu):
        values = matrix.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
        if values.shape != (n, n) or not np.isfinite(values).all():
            raise OptimizerError("strategy covariance matrices must be finite and aligned")


def validate_sigma(inputs: StrategyInputs) -> None:
    values = inputs.sigma.reindex(index=inputs.tickers, columns=inputs.tickers).to_numpy(
        dtype=float
    )
    if values.shape != (len(inputs.tickers), len(inputs.tickers)) or not np.isfinite(values).all():
        raise OptimizerError("strategy covariance matrix must be finite and aligned")


def check_weights(weights: pd.Series, cfg: OptimizerConfig) -> None:
    values = weights.to_numpy(dtype=float)
    if len(weights) < cfg.min_stocks:
        raise OptimizerError(
            f"{len(weights)} stocks available, need at least {cfg.min_stocks} for the cap"
        )
    if not weights.index.is_unique:
        raise OptimizerError("strategy weights must have unique tickers")
    if not np.isfinite(values).all():
        raise OptimizerError("strategy weights must be finite")
    if abs(float(values.sum()) - 1.0) > 1e-8:
        raise OptimizerError("strategy weights must sum to one")
    if (values < 0.0).any():
        raise OptimizerError("strategy weights cannot be negative")
    if (values > cfg.max_weight + 1e-9).any():
        raise OptimizerError("strategy weights exceed the position cap")


def clean_weights(weights: pd.Series, cfg: OptimizerConfig) -> pd.Series:
    cleaned = weights.astype(float).copy()
    values = cleaned.to_numpy()
    if not np.isfinite(values).all() or (values < -1e-9).any():
        raise OptimizerError("strategy weights contain invalid negative or non-finite values")
    cleaned.loc[(cleaned < 0.0)] = 0.0
    total = float(cleaned.sum())
    if total <= 0.0:
        raise OptimizerError("strategy weights cannot be normalized")
    cleaned /= total
    check_weights(cleaned, cfg)
    return cleaned


def cap_and_renormalize(weights: pd.Series | np.ndarray, cap: float) -> pd.Series | np.ndarray:
    values = np.asarray(weights, dtype=float).copy()
    if values.ndim != 1 or len(values) == 0:
        raise OptimizerError("weights must be a non-empty vector")
    if not np.isfinite(values).all() or (values < -1e-9).any():
        raise OptimizerError("weights must be finite and non-negative")
    if cap <= 0.0 or cap * len(values) < 1.0:
        raise OptimizerError("position cap is infeasible for the weight vector")
    values = np.maximum(values, 0.0)
    total = float(values.sum())
    if total <= 0.0:
        raise OptimizerError("weights cannot be normalized")
    values /= total

    for _ in range(len(values) + 1):
        over = values > cap
        if not over.any():
            break
        excess = float((values[over] - cap).sum())
        values[over] = cap
        available = values < cap
        if not available.any():
            raise OptimizerError("position cap leaves no room for excess weight")
        available_total = float(values[available].sum())
        if available_total > 0.0:
            values[available] += excess * values[available] / available_total
        else:
            values[available] += excess / int(available.sum())
    else:
        raise OptimizerError("position cap redistribution did not converge")

    residual = 1.0 - float(values.sum())
    if abs(residual) > 1e-14:
        if residual > 0.0:
            headroom = cap - values
            available = headroom > 0.0
            if not available.any():
                raise OptimizerError("position cap leaves no room for normalization")
            values[available] += residual * headroom[available] / headroom[available].sum()
        else:
            positive = values > 0.0
            values[positive] += residual * values[positive] / values[positive].sum()

    if (values > cap + 1e-9).any() or abs(float(values.sum()) - 1.0) > 1e-8:
        raise OptimizerError("capped weights failed normalization")
    if isinstance(weights, pd.Series):
        return pd.Series(values, index=weights.index, name=weights.name)
    return cast(np.ndarray, values)


def solve_with(problem: cp.Problem, solvers: Sequence[str]) -> str:
    installed = set(cp.installed_solvers())
    errors: list[str] = []
    for solver in solvers:
        if solver not in installed:
            errors.append(f"{solver}: not installed")
            continue
        try:
            problem.solve(solver=solver, verbose=False)
        except Exception as exc:
            errors.append(f"{solver}: {exc}")
            continue
        if problem.status == cp.OPTIMAL and problem.value is not None:
            return solver
        errors.append(f"{solver}: status {problem.status}")
    detail = "; ".join(errors) or "no solvers configured"
    raise OptimizerError(f"all strategy solvers failed ({detail})")


def fallback_chain(
    inputs: StrategyInputs,
    cfg: StudyConfig,
    diagnostics: dict[str, Any] | None = None,
) -> StrategyResult:
    tickers = inputs.tickers
    attempts: list[str] = []
    candidates = (
        ("solver_failed_prev", inputs.previous),
        ("solver_failed_q0", inputs.q0_weights),
    )
    for flag, candidate in candidates:
        if candidate is None:
            continue
        aligned = candidate.reindex(tickers).astype(float).fillna(0.0)
        values = aligned.to_numpy()
        if not np.isfinite(values).all() or (values < -1e-9).any():
            attempts.append(f"{flag}:invalid")
            continue
        aligned.loc[aligned < 0.0] = 0.0
        if float(aligned.sum()) <= 0.0:
            attempts.append(f"{flag}:empty")
            continue
        aligned /= float(aligned.sum())
        try:
            capped = cap_and_renormalize(aligned, cfg.optimizer.max_weight)
            if not isinstance(capped, pd.Series):
                raise TypeError("series input must produce series output")
            check_weights(capped, cfg.optimizer)
        except (OptimizerError, TypeError):
            attempts.append(f"{flag}:invalid")
            continue
        details = dict(diagnostics or {})
        details["fallback_attempts"] = attempts
        return StrategyResult(capped.rename("weight"), "none", float("nan"), flag, details)

    equal = pd.Series(1.0 / len(tickers), index=tickers, name="weight")
    check_weights(equal, cfg.optimizer)
    details = dict(diagnostics or {})
    details["fallback_attempts"] = attempts
    return StrategyResult(equal, "none", float("nan"), "solver_failed_ew", details)


def standard_normals(week_id: str, strategy: str, shape: tuple[int, ...], seed: int) -> np.ndarray:
    payload = json.dumps([int(seed), week_id, strategy], separators=(",", ":")).encode("utf-8")
    seed_bytes = hashlib.sha256(payload).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "big"))
    return rng.standard_normal(shape)


def cholesky_with_jitter(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    symmetric = 0.5 * (matrix + matrix.T)
    try:
        return np.linalg.cholesky(symmetric), 0.0
    except np.linalg.LinAlgError:
        n = len(symmetric)
        jitter = 1e-12 * max(float(np.trace(symmetric)) / n, 1.0)
        try:
            return np.linalg.cholesky(symmetric + jitter * np.eye(n)), jitter
        except np.linalg.LinAlgError as exc:
            raise OptimizerError("strategy covariance is not positive definite") from exc


def view_shift(weights: pd.Series, q0_weights: pd.Series) -> float:
    tickers = weights.index.union(q0_weights.index, sort=False)
    values = weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
    q0_values = q0_weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
    return float(0.5 * np.abs(values - q0_values).sum())
