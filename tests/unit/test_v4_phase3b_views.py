"""v4 phase 3B: neutral LLM view (E7), confidence-only counterfactual and controls (G18, G19),
expected tracking error, and the S8 calibration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portstudy.config import LlmSettings, ModelRegistry, StudyConfig
from portstudy.data.providers.synthetic import SyntheticWorld
from portstudy.llm.providers.base import BatchItem, LlmProvider, RawResult
from portstudy.optimize import expected_te, read_weights_meta
from portstudy.optimize.black_litterman import black_litterman, black_litterman_q0
from portstudy.optimize.counterfactual import conf_only_posterior, uncertainty_ratio
from portstudy.optimize.strategies import run_strategy
from portstudy.optimize.strategies.common import StrategyInputs
from portstudy.paths import week_output_dir
from portstudy.pipeline import fallback as fb
from portstudy.pipeline.jobs import run_named
from portstudy.sim.records import read_daily
from portstudy.types import PortfolioId, RatingSource
from portstudy.views.build import build_views
from tests.conftest import generous_budget
from tests.integration.test_phase10_pipeline import make_ctx, run_week
from tests.unit.test_v4_phase2_gateway import FailingProvider

REPO_ROOT = Path(__file__).resolve().parents[2]
N = 30


def _market(diagonal: bool = False, seed: int = 3):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}" for i in range(N)]
    vol = rng.uniform(0.15, 0.4, N) / np.sqrt(52)
    if diagonal:
        cov = np.diag(vol**2)
    else:
        b = rng.normal(0, 1, (N, 3)) * vol[:, None] * 0.5
        cov = b @ b.T + np.diag((vol * 0.8) ** 2)
    sigma = pd.DataFrame(cov, index=tickers, columns=tickers)
    pi = pd.Series(rng.normal(0.001, 0.0005, N), index=tickers)
    z_b = pd.Series(rng.normal(0, 1, N), index=tickers)
    return tickers, sigma, pi, z_b


def _views(cfg, sigma, pi, z_b, ic_b, z_l=None, ic_l=0.02, mu_q0=None):  # type: ignore[no-untyped-def]
    width = pd.Series(2.0, index=pi.index)
    return build_views(
        pi,
        pd.Series(np.sqrt(np.diag(sigma)), index=pi.index),
        z_b,
        ic_b,
        cfg.views,
        z_l=z_l,
        ic_l=ic_l,
        width=width if z_l is not None else None,
        mu_q0=mu_q0,
    )


def _solve(cfg, strategy, tickers, sigma, post):  # type: ignore[no-untyped-def]
    si = StrategyInputs("2026-W10", tickers, post.mu_bl, sigma, post.cov_mu, cfg.risk.delta)
    return run_strategy(strategy, si, cfg).weights


@pytest.mark.parametrize("ic_b", [0.02, 0.0])
def test_zero_rating_is_neutral(study_config: StudyConfig, ic_b: float) -> None:
    tickers, sigma, pi, z_b = _market()
    q0_views = _views(study_config, sigma, pi, z_b, ic_b)
    q0 = black_litterman_q0(q0_views, sigma)
    z_l = pd.Series(0.0, index=tickers)
    views = _views(study_config, sigma, pi, z_b, ic_b, z_l=z_l, mu_q0=q0.mu_bl)
    assert views.use_llm_view
    steered = black_litterman(views, sigma)
    assert np.max(np.abs(steered.mu_bl - q0.mu_bl)) < 1e-12
    for strategy in ("mvo", "topk"):
        pd.testing.assert_series_equal(
            _solve(study_config, strategy, tickers, sigma, steered),
            _solve(study_config, strategy, tickers, sigma, q0),
        )
    conf = conf_only_posterior(views, sigma)
    assert np.max(np.abs(conf.mu_bl - q0.mu_bl)) < 1e-12
    assert uncertainty_ratio(conf, q0) < 1.0  # the views add confidence even when neutral


def test_tilt_follows_z_l_on_a_diagonal_covariance(study_config: StudyConfig) -> None:
    tickers, sigma, pi, z_b = _market(diagonal=True)
    q0 = black_litterman_q0(_views(study_config, sigma, pi, z_b, 0.02), sigma)
    z_l = pd.Series(np.linspace(-2, 2, N), index=tickers)
    views = _views(study_config, sigma, pi, z_b, 0.02, z_l=z_l, mu_q0=q0.mu_bl)
    move = black_litterman(views, sigma).mu_bl - q0.mu_bl
    assert (np.sign(move[z_l != 0]) == np.sign(z_l[z_l != 0])).all()


def _resampled_te(cfg: StudyConfig, scale: float) -> float:
    s = cfg.strategies
    cfg = cfg.model_copy(
        update={
            "strategies": s.model_copy(
                update={
                    "resampled": s.resampled.model_copy(update={"draw_scale": scale, "n_draws": 40})
                }
            )
        }
    )
    tickers, sigma, pi, z_b = _market()
    q0 = black_litterman_q0(_views(cfg, sigma, pi, z_b, 0.02), sigma)
    rng = np.random.default_rng(11)
    z_l = pd.Series(rng.normal(0, 1, N), index=tickers)
    post = black_litterman(_views(cfg, sigma, pi, z_b, 0.02, z_l=z_l, mu_q0=q0.mu_bl), sigma)
    return expected_te(
        _solve(cfg, "resampled", tickers, sigma, post),
        _solve(cfg, "resampled", tickers, sigma, q0),
        sigma,
    )


def test_resampled_draw_scale(study_config: StudyConfig) -> None:
    assert study_config.strategies.resampled.draw_scale == 1.0  # default keeps today's draws
    assert _resampled_te(study_config, 0.05) > _resampled_te(study_config, 1.0)


def test_conf_only_portfolio_key_round_trips() -> None:
    pid = PortfolioId.conf("robust", RatingSource("gemma-4-31b", "bare"))
    assert pid.key == "robust__CONF__gemma-4-31b__bare"
    back = PortfolioId.parse(pid.key)
    assert back == pid and back.conf_only and back.source == RatingSource("gemma-4-31b", "bare")
    with pytest.raises(ValueError):
        RatingSource.parse("CONF__bare")


class ZeroRatings(LlmProvider):
    supports_batch = False

    def complete(self, item: BatchItem, settings: LlmSettings) -> RawResult:
        text = '{"low": -1, "likely": 0, "high": 1, "rationale": "neutral"}'
        return RawResult(custom_id=item.custom_id, text=text, endpoint="gateway")


def test_synthetic_week_metas_and_conf_only_portfolios(
    pipeline_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    inject_week: str,
) -> None:
    cfg = generous_budget(study_config)
    providers = {"stub-ow-a": ZeroRatings(), "stub-ow-b": FailingProvider()}
    ctx = make_ctx(cfg, model_registry, world, inject_week, llm_providers=providers)
    # stub-ow-b fails every call: its sources fall back for the week.
    from portstudy.llm import health
    from portstudy.llm.health import mark_sources_failed
    from portstudy.pipeline.jobs import _week_runner

    run_week(ctx, inject_week, ["J1", "J2", "J3"])
    report = health.gateway_health_check(ctx, inject_week, _week_runner(ctx, inject_week))
    assert set(report.failed) == {"stub-ow-b"}
    mark_sources_failed(ctx, inject_week, ["stub-ow-b"], "gateway_health_check_failed")
    run_week(ctx, inject_week, ["J4", "J5", "J6", "J7"])
    folder = week_output_dir(inject_week) / "weights"
    for key in ctx.steered_keys():
        meta = read_weights_meta(inject_week, key)
        assert meta["expected_te_vs_q0"] is not None and "conf_only_uncertainty_ratio" in meta, key
    conf_keys = ctx.conf_only_keys()
    assert conf_keys and all((folder / f"{k}.csv").exists() for k in conf_keys)
    for strategy in ("robust", "resampled"):
        q0 = pd.read_csv(folder / f"{strategy}__Q0.csv")["weight"]
        for prompt in ctx.prompt_ids():
            a = RatingSource("stub-ow-a", prompt)
            steered = pd.read_csv(folder / f"{PortfolioId.steered(strategy, a).key}.csv")
            conf = pd.read_csv(folder / f"{PortfolioId.conf(strategy, a).key}.csv")
            pd.testing.assert_series_equal(steered["weight"], conf["weight"])  # every rating 0
            b = PortfolioId.conf(strategy, RatingSource("stub-ow-b", prompt)).key
            pd.testing.assert_series_equal(pd.read_csv(folder / f"{b}.csv")["weight"], q0)
            assert fb.week_status_for(inject_week, b) == "fallback"
            assert read_weights_meta(inject_week, b)["decision_source"] == "fallback_q0"
    ctx.force = True
    assert run_named(ctx, "J8", inject_week).state == "done"
    assert len(read_daily(conf_keys[0])) > 0  # replayed and scored like any portfolio


def test_calibration_script_writes_csv_and_summary(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import calibrate_view_sensitivity as calib

    assert (
        calib.main(["--weeks", "2", "--config", str(REPO_ROOT / "config" / "study_test.yaml")]) == 0
    )
    out = data_dir / "outputs" / "calibration"
    table = pd.read_csv(out / "view_sensitivity.csv")
    assert set(table["input"]) == {"zero", "noise", "skilled"}
    assert set(table.loc[table["strategy"] == "cvar", "setting"]) >= {"current", "1.0", "0.01"}
    summary = (out / "summary.md").read_text("utf-8")
    assert "Mean-CVaR `lam`" in summary and "Resampled `draw_scale`" in summary
