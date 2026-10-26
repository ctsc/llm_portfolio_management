"""Shared pytest fixtures."""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from portstudy import calendar as cal
from portstudy.config import (
    ModelRegistry,
    StudyConfig,
    load_model_registry,
    load_study_config,
)
from portstudy.data.providers.base import ProviderBundle
from portstudy.data.providers.synthetic import SyntheticWorld, synthetic_bundle
from portstudy.data.universe import build_universe

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def study_config() -> StudyConfig:
    return load_study_config(REPO_ROOT / "config" / "study.yaml")


@pytest.fixture(scope="session")
def model_registry() -> ModelRegistry:
    return load_model_registry(REPO_ROOT / "config" / "models.yaml")


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point PORTSTUDY_DATA_DIR at a temp folder for the duration of a test."""
    target = tmp_path / "data"
    target.mkdir()
    monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(target))
    yield target
    os.environ.pop("PORTSTUDY_DATA_DIR", None)


@pytest.fixture(scope="session")
def world(study_config: StudyConfig) -> SyntheticWorld:
    """One seeded synthetic world shared by the whole test session."""
    return SyntheticWorld.generate(study_config.synthetic, first_week=study_config.synthetic.first_week)


@pytest.fixture(scope="session")
def providers(world: SyntheticWorld) -> ProviderBundle:
    return synthetic_bundle(world)


@pytest.fixture(scope="session")
def first_cutoff(study_config: StudyConfig) -> dt.date:
    return cal.cutoff_day(study_config.synthetic.first_week)


@pytest.fixture(scope="session")
def universe(study_config: StudyConfig, providers: ProviderBundle, first_cutoff: dt.date) -> pd.DataFrame:
    return build_universe(first_cutoff, study_config, providers, print_table=False)
