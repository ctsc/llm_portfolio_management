"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from portstudy.config import ModelRegistry, StudyConfig, load_model_registry, load_study_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def study_config() -> StudyConfig:
    return load_study_config(REPO_ROOT / "config" / "study.yaml")


@pytest.fixture()
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
