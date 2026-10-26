"""Shared helpers for the end-to-end rehearsal: run run_weekend.py week after week in-process."""

from __future__ import annotations

import datetime as dt
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from portstudy import calendar as cal
from portstudy.config import ModelRegistry, StudyConfig
from portstudy.data.providers.base import ProviderBundle
from portstudy.data.providers.synthetic import SyntheticWorld, synthetic_bundle
from portstudy.llm.prompts import load_manifest
from portstudy.llm.providers.base import LlmProvider
from portstudy.pipeline.context import RunContext, synthetic_clock
from tests.conftest import generous_budget

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_weekend  # noqa: E402

E2E_WEEKS = 25


class WeekendRunner:
    """Runs run_weekend.main for one week with a shared world and injectable providers."""

    def __init__(self, cfg: StudyConfig, registry: ModelRegistry, world: SyntheticWorld) -> None:
        self.cfg = cfg
        self.registry = registry
        self.world = world
        self.llm_providers: dict[str, LlmProvider] = {}
        self.data_providers: ProviderBundle | None = None
        self.clock: Callable[[], dt.datetime] | None = None

    def context(self, week: str, **kw: object) -> RunContext:
        return RunContext(
            cfg=self.cfg, registry=self.registry, manifest=load_manifest(), synthetic=True,
            allow_over_budget=bool(kw.get("allow_over_budget", False)),
            data_providers=self.data_providers or synthetic_bundle(self.world),
            llm_providers=dict(self.llm_providers),
            clock=self.clock or synthetic_clock(week),
        )

    def run(self, week: str, monkeypatch: pytest.MonkeyPatch, extra: list[str] | None = None) -> int:
        monkeypatch.setattr(run_weekend, "default_context",
                            lambda **kw: self.context(week, allow_over_budget=kw.get("allow_over_budget", False)))
        return run_weekend.main(["--synthetic", "--week", week, *(extra or [])])


@pytest.fixture(scope="session")
def e2e_weeks(study_config: StudyConfig) -> list[str]:
    return cal.week_range(study_config.synthetic.first_week, E2E_WEEKS)


@pytest.fixture(scope="session")
def e2e_root(tmp_path_factory: pytest.TempPathFactory, study_config: StudyConfig, model_registry: ModelRegistry,
             world: SyntheticWorld, e2e_weeks: list[str]) -> Iterator[Path]:
    """A full E2E_WEEKS-week synthetic run through run_weekend.py, one call per week."""
    root = tmp_path_factory.mktemp("e2e_root")
    previous = os.environ.get("PORTSTUDY_DATA_DIR")
    os.environ["PORTSTUDY_DATA_DIR"] = str(root)
    mp = pytest.MonkeyPatch()
    try:
        runner = WeekendRunner(generous_budget(study_config), model_registry, world)
        for w in e2e_weeks:
            rc = runner.run(w, mp)
            assert rc == 0, w
        yield root
    finally:
        mp.undo()
        if previous is None:
            os.environ.pop("PORTSTUDY_DATA_DIR", None)
        else:
            os.environ["PORTSTUDY_DATA_DIR"] = previous


@pytest.fixture()
def e2e_dir(e2e_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(e2e_root))
    return e2e_root
