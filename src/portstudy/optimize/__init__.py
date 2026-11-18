"""Target weights for every LLM portfolio and for Q0.

`build_weights(week_id, model, config_id)` runs residualization, views, Black-Litterman, and
mean-variance for one portfolio; `build_q0_weights(week_id)` does the same with the baseline
view only. Both write `data/outputs/<week>/weights/<portfolio_key>.csv` plus a meta JSON.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portstudy.baseline.combine import read_baseline
from portstudy.config import StudyConfig
from portstudy.io import read_csv, write_csv_atomic, write_json_atomic
from portstudy.llm.runner import read_scores
from portstudy.logging_setup import current_run_id
from portstudy.optimize.black_litterman import (
    Posterior,
    black_litterman,
    black_litterman_q0,
)
from portstudy.optimize.counterfactual import conf_only_posterior, uncertainty_ratio
from portstudy.optimize.strategies import run_control, run_strategy
from portstudy.optimize.strategies.common import StrategyInputs, view_shift
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
    "k_sd",
    "ratio",
    "width",
    "c",
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
    strategy: str = ""
    source: str | None = None
    decision_source: str = "llm"
    endpoint: str = ""
    view_shift: float = 0.0
    strategy_diagnostics: dict[str, Any] = field(default_factory=dict)
    cov_mu_sha256: str = ""
    fallback: bool = False
    solver_status: str = ""
    ic_mode: str = ""
    run_id: str = field(default_factory=current_run_id)
    portfolio_kind: str = ""  # q0 | control | steered | conf_only
    expected_te_vs_q0: float | None = None  # sqrt((w - w_Q0)' Sigma (w - w_Q0))
    conf_only_uncertainty_ratio: float | None = None  # G18: trace(M_conf) / trace(M_Q0)

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

    def k_sd(self) -> pd.Series:
        return self.baseline.loc[self.tickers, "k_sd"].astype(float)

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


def posterior_dir(week_id: str) -> Path:
    path = week_output_dir(week_id) / "posterior"
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
    """(IC_b, IC_L) for the week: trailing baseline IC and the portfolio's residual-signal IC.

    DECISION(A8): in `fixed` mode every week and source gets the configured `fixed_ic` so the
    view scale does not depend on the short scored history."""
    if cfg.views.ic_mode == "fixed":
        return (
            cfg.views.fixed_ic,
            cfg.views.fixed_ic if model is not None and config_id is not None else 0.0,
        )
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


def _shift_from_q0(
    weights: pd.Series, q0_weights: pd.Series | None, *, q0_only: bool, is_control: bool
) -> float:
    """Review fix M6: the view shift is measured on the weights actually written, including a
    solver-fallback allocation (the fallback is reported separately in `fallback_flag`)."""
    if q0_only or is_control or q0_weights is None:
        return 0.0
    return view_shift(weights, q0_weights)


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
    posterior: Posterior | None = None,
    strategy: str = "mvo",
    source: RatingSource | None = None,
    endpoint: str = "",
    decision_source: str = "llm",
    q0_weights: pd.Series | None = None,
    portfolio_kind: str = "",
) -> Weights:
    sigma_matrix = inputs.risk.sigma_matrix
    is_control = strategy in ("HRP", "EW")
    if posterior is not None:
        post = posterior
    elif is_control:
        prior = inputs.pi()
        post = Posterior(
            mu_bl=prior,
            tilt=pd.Series(0.0, index=inputs.tickers),
            n_views=0,
            cov_mu=pd.DataFrame(
                np.zeros((len(inputs.tickers), len(inputs.tickers))),
                index=inputs.tickers,
                columns=inputs.tickers,
            ),
        )
    else:
        post = (
            black_litterman_q0(views, sigma_matrix, logger=logger)
            if q0_only
            else black_litterman(views, sigma_matrix, logger=logger)
        )
    previous = _previous_weights(inputs.week_id, portfolio, previous_week)
    strategy_inputs = StrategyInputs(
        week_id=inputs.week_id,
        tickers=inputs.tickers,
        mu=post.mu_bl,
        sigma=sigma_matrix,
        cov_mu=post.cov_mu,
        delta=cfg.risk.delta,
        previous=previous,
        q0_weights=q0_weights,
    )
    result = (
        run_control(strategy, strategy_inputs, cfg)
        if strategy in ("HRP", "EW")
        else run_strategy(strategy, strategy_inputs, cfg)
    )
    if result.fallback_flag:
        logger.warning(
            "solver_failed",
            extra={
                "extra_fields": {
                    "week_id": inputs.week_id,
                    "portfolio": portfolio,
                    "fallback_flag": result.fallback_flag,
                }
            },
        )
    solver_status = str(result.diagnostics.get("solver_status", ""))
    if solver_status == "optimal_inaccurate":
        logger.warning(
            "solver_inaccurate",
            extra={
                "extra_fields": {
                    "week_id": inputs.week_id,
                    "portfolio": portfolio,
                    "solver_status": solver_status,
                }
            },
        )
    shift = _shift_from_q0(result.weights, q0_weights, q0_only=q0_only, is_control=is_control)
    te = (
        expected_te(result.weights, q0_weights, sigma_matrix)
        if q0_weights is not None and not q0_only and not is_control
        else None
    )
    out = Weights(
        portfolio=portfolio,
        week_id=inputs.week_id,
        weights=result.weights,
        mu_bl=post.mu_bl if not is_control else pd.Series(np.nan, index=inputs.tickers),
        tilt=post.tilt if not is_control else pd.Series(np.nan, index=inputs.tickers),
        use_llm_view=views.use_llm_view and not q0_only and not is_control,
        ic_b=views.ic_b,
        ic_l=views.ic_l if not q0_only and not is_control else 0.0,
        solver=result.solver,
        fallback_flag=result.fallback_flag,
        n_llm_view=len(views.llm_tickers) if not q0_only and not is_control else 0,
        notes=list(views.notes) if not is_control else [],
        strategy=strategy,
        source=source.key if source else None,
        decision_source=decision_source,
        endpoint=endpoint,
        view_shift=shift,
        strategy_diagnostics=result.diagnostics,
        cov_mu_sha256=_cov_mu_sha256(post) if not is_control else "",
        fallback=result.fallback_flag is not None,
        solver_status=solver_status,
        ic_mode=cfg.views.ic_mode,
        portfolio_kind=portfolio_kind
        or ("control" if is_control else "q0" if q0_only else "steered"),
        expected_te_vs_q0=te,
    )
    for k, v in extra.items():
        setattr(out, k, v)
    _save(out, views if not is_control else None, inputs, z_l)
    return out


def expected_te(weights: pd.Series, q0_weights: pd.Series, sigma: pd.DataFrame) -> float:
    """Expected tracking error against Q0: sqrt((w - w_Q0)' Sigma (w - w_Q0))."""
    tickers = list(sigma.index)
    d = (
        weights.reindex(tickers).fillna(0.0).astype(float)
        - q0_weights.reindex(tickers).fillna(0.0).astype(float)
    ).to_numpy()
    value = float(d @ sigma.loc[tickers, tickers].to_numpy(dtype=float) @ d)
    return float(np.sqrt(max(value, 0.0)))


def _save(
    w: Weights,
    views: Views | None,
    inputs: WeekInputs,
    z_l: pd.Series | None,
) -> None:
    idx = w.weights.index
    df = pd.DataFrame({"ticker": idx, "weight": w.weights.to_numpy()})
    if views is None:
        for column in ("mu_bl", "tilt", "pi", "z_b", "z_L", "k_sd", "ratio", "width", "c"):
            df[column] = float("nan")
        df["has_llm_view"] = False
    else:
        frame = views_frame(views).set_index("ticker")
        for column, values in (
            ("mu_bl", w.mu_bl),
            ("tilt", w.tilt),
            ("pi", frame["pi"]),
            ("z_b", inputs.z_b()),
            ("k_sd", frame["k_sd"]),
            ("ratio", frame["ratio"]),
            ("width", frame["width"]),
            ("c", frame["c"]),
        ):
            df[column] = values.reindex(idx).to_numpy()
        df["z_L"] = z_l.reindex(idx).to_numpy() if z_l is not None else float("nan")
        df["has_llm_view"] = df["ticker"].isin(views.llm_tickers) if w.use_llm_view else False
    folder = weights_dir(w.week_id)
    write_csv_atomic(df.loc[:, WEIGHT_COLUMNS], folder / f"{w.portfolio}.csv")
    write_json_atomic(w.meta(), folder / f"{w.portfolio}.meta.json")


def _cov_mu_sha256(posterior: Posterior) -> str:
    matrix = posterior.cov_mu.reindex(
        index=posterior.mu_bl.index, columns=posterior.mu_bl.index
    ).to_numpy(dtype="<f8")
    digest = hashlib.sha256()
    digest.update("\n".join(map(str, posterior.mu_bl.index)).encode("utf-8"))
    digest.update(matrix.tobytes(order="C"))
    return digest.hexdigest()


def _save_posterior(
    name: str,
    inputs: WeekInputs,
    views: Views,
    posterior: Posterior,
    z_l: pd.Series | None = None,
) -> None:
    frame = views_frame(views).set_index("ticker")
    tickers = posterior.mu_bl.index
    data = pd.DataFrame(
        {
            "ticker": tickers,
            "pi": views.pi.reindex(tickers).to_numpy(),
            "mu_bl": posterior.mu_bl.reindex(tickers).to_numpy(),
            "tilt": posterior.tilt.reindex(tickers).to_numpy(),
            "z_b": inputs.z_b().reindex(tickers).to_numpy(),
            "z_L": z_l.reindex(tickers).to_numpy() if z_l is not None else float("nan"),
            "k_sd": views.k_sd.reindex(tickers).to_numpy(),
            "ratio": views.ratio.reindex(tickers).to_numpy(),
            "width": views.width.reindex(tickers).to_numpy(),
            "c": views.c.reindex(tickers).to_numpy(),
            "has_llm_view": frame["has_llm_view"].reindex(tickers).to_numpy(),
            "q_L": frame["q_L"].reindex(tickers).to_numpy(),
            "mu_Q0": views.mu_q0.reindex(tickers).to_numpy() if len(views.mu_q0) else float("nan"),
        }
    )
    data = data.loc[
        :,
        [
            "ticker",
            "pi",
            "mu_bl",
            "tilt",
            "z_b",
            "z_L",
            "k_sd",
            "ratio",
            "width",
            "c",
            "has_llm_view",
            "q_L",
            "mu_Q0",
        ],
    ]
    write_csv_atomic(data, posterior_dir(inputs.week_id) / f"{name}.csv")


def _q0_views(inputs: WeekInputs, cfg: StudyConfig, ic_b: float, logger: logging.Logger) -> Views:
    return build_views(
        inputs.pi(),
        inputs.sigma_vec(),
        inputs.z_b(),
        ic_b,
        cfg.views,
        k_sd=inputs.k_sd(),
        label="Q0",
        logger=logger,
    )


def q0_posterior(
    inputs: WeekInputs, cfg: StudyConfig, ic_b: float, logger: logging.Logger
) -> Posterior:
    """The week's Q0 posterior without writing files (its mean centers the LLM view, E7)."""
    return black_litterman_q0(_q0_views(inputs, cfg, ic_b, logger), inputs.risk.sigma_matrix)


def posterior_for_q0(
    inputs: WeekInputs,
    cfg: StudyConfig,
    ic_b: float,
    logger: logging.Logger | None = None,
) -> Posterior:
    log = logger or logging.getLogger("portstudy.optimize")
    views = _q0_views(inputs, cfg, ic_b, log)
    posterior = black_litterman_q0(views, inputs.risk.sigma_matrix, logger=log)
    _save_posterior("Q0", inputs, views, posterior)
    return posterior


def _source_views(
    inputs: WeekInputs,
    source: RatingSource,
    cfg: StudyConfig,
    ic_b: float,
    ic_l: float,
    logger: logging.Logger,
    mu_q0: pd.Series | None = None,
) -> tuple[Views, pd.Series, int]:
    scores = inputs.scores_for(source.model, source.config_id)
    valid = valid_scores(scores)
    n_invalid = int(len(scores) - len(valid))
    z_l = residualize(valid, inputs.z_b()) if len(valid) else pd.Series(dtype=float)
    width = valid.set_index("ticker")["width"].astype(float) if len(valid) else None
    views = build_views(
        inputs.pi(),
        inputs.sigma_vec(),
        inputs.z_b(),
        ic_b,
        cfg.views,
        k_sd=inputs.k_sd(),
        z_l=z_l,
        ic_l=ic_l,
        width=width,
        logger=logger,
        label=source.key,
        mu_q0=mu_q0,
    )
    return views, z_l, n_invalid


def posterior_for_source(
    inputs: WeekInputs,
    source: RatingSource,
    cfg: StudyConfig,
    ic_b: float,
    ic_l: float,
    logger: logging.Logger | None = None,
    q0_post: Posterior | None = None,
) -> Posterior | None:
    log = logger or logging.getLogger("portstudy.optimize")
    q0p = q0_post or q0_posterior(inputs, cfg, ic_b, log)
    views, z_l, _ = _source_views(inputs, source, cfg, ic_b, ic_l, log, mu_q0=q0p.mu_bl)
    posterior = (
        black_litterman(views, inputs.risk.sigma_matrix, logger=log)
        if views.use_llm_view
        else black_litterman_q0(views, inputs.risk.sigma_matrix, logger=log)
    )
    _save_posterior(source.key, inputs, views, posterior, z_l)
    return posterior if views.use_llm_view else None


def build_q0_weights(
    week_id: str,
    cfg: StudyConfig,
    ic_b: float,
    inputs: WeekInputs | None = None,
    previous_week: str | None = None,
    logger: logging.Logger | None = None,
    strategy: StrategyName = "mvo",
    posterior: Posterior | None = None,
) -> Weights:
    log = logger or logging.getLogger("portstudy.optimize")
    inp = inputs or WeekInputs.load(week_id)
    views = _q0_views(inp, cfg, ic_b, log)
    post = posterior or black_litterman_q0(views, inp.risk.sigma_matrix, logger=log)
    return _optimize(
        PortfolioId.q0(strategy).key,
        inp,
        views,
        cfg,
        True,
        previous_week,
        {},
        log,
        posterior=post,
        strategy=strategy,
        decision_source="q0",
    )


def existing_q0_weights(week_id: str, strategy: StrategyName) -> Weights | None:
    """The strategy's Q0 for the week as written by J7 (`<q0>.csv` plus `<q0>.meta.json`),
    or None when it has not been written yet."""
    portfolio = PortfolioId.q0(strategy).key
    folder = weights_dir(week_id)
    csv_path, meta_path = folder / f"{portfolio}.csv", folder / f"{portfolio}.meta.json"
    if not (csv_path.exists() and meta_path.exists()):
        return None
    df = read_csv(csv_path).set_index("ticker")
    df.index.name = None  # the solver's series carry an unnamed ticker index
    meta = read_weights_meta(week_id, portfolio)
    fields = {f.name for f in dataclasses.fields(Weights)} - {"weights", "mu_bl", "tilt"}
    return Weights(
        weights=df["weight"].astype(float),
        mu_bl=df["mu_bl"].astype(float),
        tilt=df["tilt"].astype(float),
        **{k: v for k, v in meta.items() if k in fields},
    )


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
    posterior: Posterior | None = None,
    q0_weights: pd.Series | None = None,
    q0_post: Posterior | None = None,
) -> Weights:
    """Build one source-steered allocation using a shared source posterior."""
    log = logger or logging.getLogger("portstudy.optimize")
    inp = inputs or WeekInputs.load(week_id)
    source = RatingSource(model, config_id)
    key = PortfolioId.steered(strategy, source).key
    q0p = q0_post or q0_posterior(inp, cfg, ic_b, log)
    views, z_l, n_invalid = _source_views(inp, source, cfg, ic_b, ic_l, log, mu_q0=q0p.mu_bl)
    post = posterior
    if post is None and views.use_llm_view:
        post = black_litterman(views, inp.risk.sigma_matrix, logger=log)
    if not views.use_llm_view:
        # Review fix M7: reuse the Q0 already written this week instead of solving it again
        # (and rewriting its files) from the steered portfolio's previous-week path.
        q0 = existing_q0_weights(week_id, strategy) or build_q0_weights(
            week_id,
            cfg,
            ic_b,
            inputs=inp,
            previous_week=previous_week,
            logger=log,
            strategy=strategy,
        )
        q0_path = weights_dir(week_id) / f"{q0.portfolio}.csv"
        source_path = weights_dir(week_id) / f"{key}.csv"
        shutil.copyfile(q0_path, source_path)
        out = Weights(
            portfolio=key,
            week_id=week_id,
            weights=q0.weights.copy(),
            mu_bl=q0.mu_bl.copy(),
            tilt=q0.tilt.copy(),
            use_llm_view=False,
            ic_b=q0.ic_b,
            ic_l=ic_l,
            solver=q0.solver,
            fallback_flag=q0.fallback_flag,
            n_llm_view=0,
            n_invalid=n_invalid,
            notes=[*q0.notes, "weights_equal_q0"],
            strategy=strategy,
            source=source.key,
            decision_source="llm",
            endpoint=_source_endpoint(inp, source),
            view_shift=0.0,
            strategy_diagnostics=dict(q0.strategy_diagnostics),
            cov_mu_sha256=q0.cov_mu_sha256,
            fallback=q0.fallback,
            ic_mode=cfg.views.ic_mode,
            portfolio_kind="steered",
            expected_te_vs_q0=0.0,
            conf_only_uncertainty_ratio=1.0,
        )
        write_json_atomic(out.meta(), weights_dir(week_id) / f"{key}.meta.json")
        _save_posterior(
            source.key, inp, views, black_litterman_q0(views, inp.risk.sigma_matrix), z_l
        )
        return out
    if post is not None:
        _save_posterior(source.key, inp, views, post, z_l)
    conf = conf_only_posterior(views, inp.risk.sigma_matrix, logger=log)
    out = _optimize(
        key,
        inp,
        views,
        cfg,
        False,
        previous_week,
        {
            "n_invalid": n_invalid,
            "conf_only_uncertainty_ratio": uncertainty_ratio(conf, q0p),
        },
        log,
        z_l=z_l,
        posterior=post,
        strategy=strategy,
        source=source,
        endpoint=_source_endpoint(inp, source),
        decision_source="llm",
        q0_weights=q0_weights,
        portfolio_kind="steered",
    )
    return out


def build_conf_only_weights(
    week_id: str,
    model: str,
    config_id: str,
    cfg: StudyConfig,
    ic_b: float,
    ic_l: float,
    inputs: WeekInputs | None = None,
    previous_week: str | None = None,
    logger: logging.Logger | None = None,
    strategy: StrategyName = "robust",
    q0_weights: pd.Series | None = None,
    q0_post: Posterior | None = None,
) -> Weights:
    """G19: the strategy solved with Q0's posterior mean and the source's confidence-only
    posterior covariance `M_conf` (same caps, previous weights and solver settings as the
    steered portfolio). Without a valid LLM view it copies Q0's weights, like the steered one."""
    log = logger or logging.getLogger("portstudy.optimize")
    inp = inputs or WeekInputs.load(week_id)
    source = RatingSource(model, config_id)
    key = PortfolioId.conf(strategy, source).key
    q0p = q0_post or q0_posterior(inp, cfg, ic_b, log)
    views, z_l, n_invalid = _source_views(inp, source, cfg, ic_b, ic_l, log, mu_q0=q0p.mu_bl)
    if not views.use_llm_view:
        copy_q0_as(
            week_id,
            strategy,
            key,
            {
                "source": source.key,
                "decision_source": "control",
                "portfolio_kind": "conf_only",
                "conf_only_uncertainty_ratio": 1.0,
                "expected_te_vs_q0": 0.0,
            },
        )
        q0 = existing_q0_weights(week_id, strategy)
        assert q0 is not None
        return dataclasses.replace(q0, portfolio=key, portfolio_kind="conf_only")
    conf = conf_only_posterior(views, inp.risk.sigma_matrix, logger=log)
    return _optimize(
        key,
        inp,
        views,
        cfg,
        False,
        previous_week,
        {"n_invalid": n_invalid, "conf_only_uncertainty_ratio": uncertainty_ratio(conf, q0p)},
        log,
        z_l=z_l,
        posterior=conf,
        strategy=strategy,
        source=source,
        endpoint=_source_endpoint(inp, source),
        decision_source="control",
        q0_weights=q0_weights,
        portfolio_kind="conf_only",
    )


def copy_q0_as(week_id: str, strategy: StrategyName, key: str, meta_update: dict[str, Any]) -> None:
    """Write the strategy's Q0 weights under another portfolio key with an updated meta."""
    folder = weights_dir(week_id)
    q0_key = PortfolioId.q0(strategy).key
    shutil.copyfile(folder / f"{q0_key}.csv", folder / f"{key}.csv")
    meta = read_weights_meta(week_id, q0_key)
    meta.update(
        {
            "portfolio": key,
            "strategy": strategy,
            "use_llm_view": False,
            "view_shift": 0.0,
            "run_id": current_run_id(),
            **meta_update,
        }
    )
    write_json_atomic(meta, folder / f"{key}.meta.json")


def build_control_weights(
    week_id: str,
    name: str,
    cfg: StudyConfig,
    inputs: WeekInputs | None = None,
    logger: logging.Logger | None = None,
) -> Weights:
    log = logger or logging.getLogger("portstudy.optimize")
    inp = inputs or WeekInputs.load(week_id)
    empty_views = Views(
        tickers=inp.tickers,
        pi=pd.Series(dtype=float),
        sigma=pd.Series(dtype=float),
        tau=cfg.views.tau,
        ic_b=0.0,
        ic_l=0.0,
        k_sd=pd.Series(dtype=float),
        ratio=pd.Series(dtype=float),
        q_b=pd.Series(dtype=float),
        omega_b=pd.Series(dtype=float),
        use_baseline_view=False,
        q_l=pd.Series(dtype=float),
        omega_l=pd.Series(dtype=float),
        width=pd.Series(dtype=float),
        c=pd.Series(dtype=float),
        use_llm_view=False,
    )
    return _optimize(
        name,
        inp,
        empty_views,
        cfg,
        True,
        None,
        {},
        log,
        strategy=name,
        decision_source="control",
    )


def _source_endpoint(inputs: WeekInputs, source: RatingSource) -> str:
    scores = inputs.scores_for(source.model, source.config_id)
    if "endpoint" not in scores or scores.empty:
        return ""
    endpoints = {str(value) for value in scores["endpoint"] if str(value)}
    return next(iter(endpoints)) if len(endpoints) == 1 else "mixed" if endpoints else ""


def read_weights(week_id: str, portfolio: str) -> pd.DataFrame:
    return read_csv(weights_dir(week_id) / f"{portfolio}.csv")


def read_weights_meta(week_id: str, portfolio: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (weights_dir(week_id) / f"{portfolio}.meta.json").read_text(encoding="utf-8")
    )
    return data
