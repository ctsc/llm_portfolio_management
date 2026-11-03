"""Target weights for every LLM portfolio and for Q0.

`build_weights(week_id, model, config_id)` runs residualization, views, Black-Litterman, and
mean-variance for one portfolio; `build_q0_weights(week_id)` does the same with the baseline
view only. Both write `data/outputs/<week>/weights/<portfolio_key>.csv` plus a meta JSON.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from portstudy.baseline.combine import read_baseline
from portstudy.config import StudyConfig
from portstudy.io import read_csv, write_csv_atomic, write_json_atomic
from portstudy.llm.runner import read_scores
from portstudy.optimize.black_litterman import black_litterman, black_litterman_q0
from portstudy.optimize.mean_variance import solve_mean_variance
from portstudy.paths import week_output_dir
from portstudy.risk.covariance import RiskModel, read_risk_model
from portstudy.risk.prior import read_prior
from portstudy.types import PortfolioId, RatingSource, StrategyName
from portstudy.views.build import Views, build_views, views_frame
from portstudy.views.ic import ICStore, trailing_ic
from portstudy.views.residualize import residualize, valid_scores

WEIGHT_COLUMNS = [
    "ticker",
    "weight",
    "mu_bl",
    "tilt",
    "pi",
    "z_b",
    "z_L",
    "confidence",
    "has_llm_view",
]


@dataclass
class Weights:
    portfolio: str
    week_id: str
    weights: pd.Series
    mu_bl: pd.Series
    tilt: pd.Series
    use_llm_view: bool
    ic_b: float
    ic_l: float
    solver: str
    fallback_flag: str | None = None
    n_llm_view: int = 0
    n_invalid: int = 0
    notes: list[str] = field(default_factory=list)

    def meta(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("weights", "mu_bl", "tilt"):
            d.pop(k)
        d["n_stocks"] = int(len(self.weights))
        return d


@dataclass
class WeekInputs:
    """Everything the optimizer needs for one week, loaded from the week's outputs."""

    week_id: str
    baseline: pd.DataFrame
    prior: pd.DataFrame
    risk: RiskModel
    scores: pd.DataFrame

    @classmethod
    def load(cls, week_id: str) -> WeekInputs:
        return cls(
            week_id=week_id,
            baseline=read_baseline(week_id).set_index("ticker"),
            prior=read_prior(week_id).set_index("ticker"),
            risk=read_risk_model(week_id),
            scores=read_scores(week_id),
        )

    @property
    def tickers(self) -> list[str]:
        return [t for t in self.risk.tickers if t in self.baseline.index and t in self.prior.index]

    def z_b(self) -> pd.Series:
        return self.baseline.loc[self.tickers, "z_b"].astype(float)

    def pi(self) -> pd.Series:
        return self.prior.loc[self.tickers, "pi"].astype(float)

    def sigma_vec(self) -> pd.Series:
        return self.prior.loc[self.tickers, "sigma"].astype(float)

    def scores_for(self, model: str, config_id: str) -> pd.DataFrame:
        s = self.scores
        return s[(s["model"] == model) & (s["config_id"] == config_id)].reset_index(drop=True)


def weights_dir(week_id: str) -> Path:
    path = week_output_dir(week_id) / "weights"
    path.mkdir(parents=True, exist_ok=True)
    return path


def portfolio_ics(
    week_id: str,
    cfg: StudyConfig,
    store: ICStore,
    live_start_week: str,
    model: str | None = None,
    config_id: str | None = None,
    fallback_weeks: set[str] | frozenset[str] = frozenset(),
) -> tuple[float, float]:
    """(IC_b, IC_L) for the week: trailing baseline IC and the portfolio's residual-signal IC."""
    ic_b = trailing_ic(store, "baseline", week_id, cfg.views, live_start_week)
    ic_l = 0.0
    if model is not None and config_id is not None:
        ic_l = trailing_ic(
            store,
            cfg.views.ic_signal_for_llm,
            week_id,
            cfg.views,
            live_start_week,
            model=model,
            config_id=config_id,
            fallback_weeks=fallback_weeks,
        )
    return ic_b, ic_l


def _previous_weights(week_id: str, portfolio: str, previous_week: str | None) -> pd.Series | None:
    if previous_week is None:
        return None
    path = weights_dir(previous_week) / f"{portfolio}.csv"
    if not path.exists():
        return None
    df = read_csv(path)
    return df.set_index("ticker")["weight"].astype(float)


def _optimize(
    portfolio: str,
    inputs: WeekInputs,
    views: Views,
    cfg: StudyConfig,
    q0_only: bool,
    previous_week: str | None,
    extra: dict[str, Any],
    logger: logging.Logger,
    z_l: pd.Series | None = None,
    confidence: pd.Series | None = None,
) -> Weights:
    sigma_matrix = inputs.risk.sigma_matrix
    post = (
        black_litterman_q0(views, sigma_matrix) if q0_only else black_litterman(views, sigma_matrix)
    )
    previous = _previous_weights(inputs.week_id, portfolio, previous_week)
    mv = solve_mean_variance(post.mu_bl, sigma_matrix, cfg.risk.delta, cfg.optimizer, previous)
    if mv.fallback_flag:
        logger.warning(
            "solver_failed",
            extra={"extra_fields": {"week_id": inputs.week_id, "portfolio": portfolio}},
        )
    out = Weights(
        portfolio=portfolio,
        week_id=inputs.week_id,
        weights=mv.weights,
        mu_bl=post.mu_bl,
        tilt=post.tilt,
        use_llm_view=views.use_llm_view and not q0_only,
        ic_b=views.ic_b,
        ic_l=views.ic_l if not q0_only else 0.0,
        solver=mv.solver,
        fallback_flag=mv.fallback_flag,
        n_llm_view=len(views.llm_tickers) if not q0_only else 0,
        notes=list(views.notes),
    )
    for k, v in extra.items():
        setattr(out, k, v)
    _save(out, views, inputs, z_l, confidence)
    return out


def _save(
    w: Weights,
    views: Views,
    inputs: WeekInputs,
    z_l: pd.Series | None,
    confidence: pd.Series | None,
) -> None:
    frame = views_frame(views).set_index("ticker")
    idx = w.weights.index
    df = pd.DataFrame(
        {
            "ticker": idx,
            "weight": w.weights.to_numpy(),
            "mu_bl": w.mu_bl.to_numpy(),
            "tilt": w.tilt.to_numpy(),
            "pi": frame["pi"].reindex(idx).to_numpy(),
            "z_b": inputs.z_b().reindex(idx).to_numpy(),
        }
    )
    has_l = z_l is not None and len(z_l) > 0
    df["z_L"] = z_l.reindex(idx).to_numpy() if has_l and z_l is not None else float("nan")
    df["confidence"] = (
        confidence.reindex(idx).to_numpy()
        if has_l and confidence is not None and len(confidence)
        else float("nan")
    )
    df["has_llm_view"] = df["ticker"].isin(views.llm_tickers) if w.use_llm_view else False
    folder = weights_dir(w.week_id)
    write_csv_atomic(df.loc[:, WEIGHT_COLUMNS], folder / f"{w.portfolio}.csv")
    write_json_atomic(w.meta(), folder / f"{w.portfolio}.meta.json")


def build_q0_weights(
    week_id: str,
    cfg: StudyConfig,
    ic_b: float,
    inputs: WeekInputs | None = None,
    previous_week: str | None = None,
    logger: logging.Logger | None = None,
    strategy: StrategyName = "mvo",
) -> Weights:
    log = logger or logging.getLogger("portstudy.optimize")
    inp = inputs or WeekInputs.load(week_id)
    views = build_views(
        inp.pi(), inp.sigma_vec(), inp.z_b(), ic_b, cfg.views, label=f"{strategy}__Q0", logger=log
    )
    return _optimize(PortfolioId.q0(strategy).key, inp, views, cfg, True, previous_week, {}, log)


def build_weights(
    week_id: str,
    model: str,
    config_id: str,
    cfg: StudyConfig,
    ic_b: float,
    ic_l: float,
    inputs: WeekInputs | None = None,
    previous_week: str | None = None,
    logger: logging.Logger | None = None,
    strategy: StrategyName = "mvo",
) -> Weights:
    """Residualize, build views, Black-Litterman, mean-variance for one LLM portfolio."""
    log = logger or logging.getLogger("portstudy.optimize")
    inp = inputs or WeekInputs.load(week_id)
    key = PortfolioId.steered(strategy, RatingSource(model, config_id)).key
    scores = inp.scores_for(model, config_id)
    valid = valid_scores(scores)
    n_invalid = int(len(scores) - len(valid))
    z_l = residualize(valid, inp.z_b()) if len(valid) else pd.Series(dtype=float)
    conf = valid.set_index("ticker")["confidence"].astype(float) if len(valid) else None
    views = build_views(
        inp.pi(),
        inp.sigma_vec(),
        inp.z_b(),
        ic_b,
        cfg.views,
        z_l=z_l,
        confidence=conf,
        ic_l=ic_l,
        logger=log,
        label=key,
    )
    # A portfolio without an LLM view this week takes the Q0 path so it matches Q0 exactly.
    q0_only = not views.use_llm_view
    out = _optimize(
        key,
        inp,
        views,
        cfg,
        q0_only,
        previous_week,
        {"n_invalid": n_invalid},
        log,
        z_l=z_l,
        confidence=conf,
    )
    if q0_only:
        out.notes.append("weights_equal_q0")
    return out


def read_weights(week_id: str, portfolio: str) -> pd.DataFrame:
    return read_csv(weights_dir(week_id) / f"{portfolio}.csv")


def read_weights_meta(week_id: str, portfolio: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (weights_dir(week_id) / f"{portfolio}.meta.json").read_text(encoding="utf-8")
    )
    return data
