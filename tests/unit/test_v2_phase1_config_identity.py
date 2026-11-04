import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from portstudy.config import (
    ModelRegistry,
    StudyConfig,
    load_model_registry,
    load_study_config,
)
from portstudy.llm.prompts import load_manifest
from portstudy.pipeline.cli import app
from portstudy.pipeline.context import RunContext
from portstudy.types import PortfolioId, RatingSource

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_CONFIG = REPO_ROOT / "config" / "study_test.yaml"


def test_rating_source_round_trip_accepts_dotted_and_hyphenated_ids() -> None:
    source = RatingSource("model.v2-name", "skills.1")
    assert source.key == "model.v2-name__skills.1"
    assert RatingSource.parse(source.key) == source


def test_model_registry_accepts_dotted_and_hyphenated_ids() -> None:
    raw = load_model_registry().model_dump()
    raw["models"][0]["id"] = "custom.model-v2"
    registry = ModelRegistry.model_validate(raw)
    assert registry.models[0].id == "custom.model-v2"


def test_portfolio_id_round_trips_all_three_forms() -> None:
    source = RatingSource("stub-ow-a", "bare")
    steered = PortfolioId.steered("mvo", source)
    q0 = PortfolioId.q0("mvo")
    control = PortfolioId.control("HRP")

    assert PortfolioId.parse(steered.key) == steered
    assert steered.source == source
    assert PortfolioId.parse(q0.key).is_q0
    assert PortfolioId.parse(q0.key).source is None
    assert PortfolioId.parse(control.key).is_control
    assert PortfolioId.parse(control.key).source is None


@pytest.mark.parametrize(
    "key",
    ["", "model", "model__bare__extra", "mvo__", "unknown__Q0", "mvo__Q0__bare", "HRP__bare"],
)
def test_malformed_portfolio_keys_are_rejected(key: str) -> None:
    with pytest.raises(ValueError):
        PortfolioId.parse(key)


@pytest.mark.parametrize("key", ["", "model", "model__bare__extra", "Q0__bare"])
def test_malformed_rating_source_keys_are_rejected(key: str) -> None:
    with pytest.raises(ValueError):
        RatingSource.parse(key)


@pytest.mark.parametrize("model_id", ["Q0", "HRP", "EW", "bad__model"])
def test_model_registry_rejects_reserved_and_separator_ids(model_id: str) -> None:
    raw = load_model_registry().model_dump()
    raw["models"][0]["id"] = model_id
    with pytest.raises(ValidationError):
        ModelRegistry.model_validate(raw)


def test_unknown_strategy_is_rejected_by_yaml_config() -> None:
    raw = StudyConfig().model_dump()
    raw["strategies"]["enabled"] = ["unknown"]
    with pytest.raises(ValidationError):
        StudyConfig.model_validate(raw)


def test_invalid_topk_and_optimizer_limits_are_rejected() -> None:
    raw = StudyConfig().model_dump()
    raw["strategies"]["topk"]["k"] = 10
    with pytest.raises(ValidationError):
        StudyConfig.model_validate(raw)

    raw = StudyConfig().model_dump()
    raw["optimizer"]["min_stocks"] = 19
    with pytest.raises(ValidationError):
        StudyConfig.model_validate(raw)


def test_test_config_selects_expected_sources_and_portfolio_count() -> None:
    ctx = RunContext(
        cfg=load_study_config(TEST_CONFIG),
        registry=load_model_registry(),
        manifest=load_manifest(),
        synthetic=True,
    )
    assert [source.key for source in ctx.sources()] == [
        "stub-ow-a__bare",
        "stub-ow-a__placebo",
        "stub-ow-a__skills_1",
        "stub-ow-a__skills_2",
        "stub-ow-b__bare",
        "stub-ow-b__placebo",
        "stub-ow-b__skills_1",
        "stub-ow-b__skills_2",
    ]
    enabled = ctx.cfg.strategies.enabled
    assert ctx.portfolio_keys() == [
        *[f"{strategy}__Q0" for strategy in enabled],
        *ctx.cfg.strategies.controls,
        *[f"{strategy}__{source.key}" for strategy in enabled for source in ctx.sources()],
    ]
    assert len(ctx.portfolio_keys()) == (
        len(ctx.cfg.strategies.enabled)
        + len(ctx.cfg.strategies.controls)
        + len(ctx.cfg.strategies.enabled) * len(ctx.sources())
    )


def test_portfolio_keys_group_q0_controls_and_steered_by_strategy() -> None:
    raw = StudyConfig().model_dump()
    raw["strategies"]["enabled"] = ["mvo", "cvar"]
    raw["strategies"]["controls"] = ["HRP", "EW"]
    ctx = RunContext(
        cfg=StudyConfig.model_validate(raw),
        registry=load_model_registry(),
        manifest=load_manifest(),
        synthetic=True,
    )
    sources = ctx.sources()
    assert ctx.portfolio_keys() == [
        "mvo__Q0",
        "cvar__Q0",
        "HRP",
        "EW",
        *[f"mvo__{source.key}" for source in sources],
        *[f"cvar__{source.key}" for source in sources],
    ]


def test_portstudy_config_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORTSTUDY_CONFIG", str(TEST_CONFIG))
    cfg = load_study_config()
    assert cfg.llm.model_ids == ["stub-ow-a", "stub-ow-b"]
    assert cfg.strategies.cvar.n_scenarios == 2000
    assert cfg.strategies.resampled.n_draws == 20
    assert cfg.eval.shuffle.n_shuffles == 20


def test_cli_accepts_config_option() -> None:
    result = CliRunner().invoke(
        app,
        ["status", "--synthetic", "--week", "2026-W02", "--config", str(TEST_CONFIG)],
    )
    assert result.exit_code == 0, result.output


def test_run_weekend_accepts_config_option() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_weekend.py"),
            "--dry-run",
            "--synthetic",
            "--week",
            "2026-W02",
            "--config",
            str(TEST_CONFIG),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
