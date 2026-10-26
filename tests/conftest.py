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


# Number of consecutive synthetic weeks snapshotted once per session for the later phases.
SESSION_WEEKS = 30


@pytest.fixture(scope="session")
def session_weeks(study_config: StudyConfig) -> list[str]:
    return cal.week_range(study_config.synthetic.first_week, SESSION_WEEKS)


@pytest.fixture(scope="session")
def session_data_root(tmp_path_factory: pytest.TempPathFactory, study_config: StudyConfig,
                      providers: ProviderBundle, universe: pd.DataFrame,
                      session_weeks: list[str]) -> Path:
    """A data dir holding SESSION_WEEKS snapshots, built once and shared read-mostly."""
    from portstudy.data.snapshot import make_snapshot

    target = tmp_path_factory.mktemp("session_data")
    previous = os.environ.get("PORTSTUDY_DATA_DIR")
    os.environ["PORTSTUDY_DATA_DIR"] = str(target)
    try:
        for w in session_weeks:
            make_snapshot(w, study_config, providers, universe)
    finally:
        if previous is None:
            os.environ.pop("PORTSTUDY_DATA_DIR", None)
        else:
            os.environ["PORTSTUDY_DATA_DIR"] = previous
    return target


@pytest.fixture()
def snapshot_data_dir(session_data_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point PORTSTUDY_DATA_DIR at the shared snapshot folder for one test."""
    monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(session_data_root))
    return session_data_root
