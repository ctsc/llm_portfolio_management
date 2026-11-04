"""Shared helpers for the end-to-end rehearsal: run run_weekend.py week after week in-process."""

from __future__ import annotations

import datetime as dt
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from portstudy import calendar as cal
from portstudy.config import LlmSettings, ModelRegistry, StudyConfig
from portstudy.data.providers.base import ProviderBundle
from portstudy.data.providers.synthetic import SyntheticWorld, synthetic_bundle
from portstudy.llm.prompts import load_manifest
from portstudy.llm.providers.base import BatchItem, LlmProvider, RawResult
from portstudy.pipeline.context import RunContext, synthetic_clock
from tests.conftest import generous_budget

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_weekend  # noqa: E402

E2E_WEEKS = 30

# v2 phase 13 rehearsal: week index -> injected failure. Every other week runs clean.
INJECTIONS = {3: "source_failure", 6: "solver_failure", 9: "lock_cutoff", 12: "gateway_health"}
LOCK_CUTOFF_AFTER_FILES = 10  # J7 has written this many weight files when the clock passes the lock


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
            cfg=self.cfg,
            registry=self.registry,
            manifest=load_manifest(),
            synthetic=True,
            allow_over_budget=bool(kw.get("allow_over_budget", False)),
            data_providers=self.data_providers or synthetic_bundle(self.world),
            llm_providers=dict(self.llm_providers),
            clock=self.clock or synthetic_clock(week),
            gateway_fallback=bool(kw.get("gateway_fallback", False)),
        )

    def run(
        self, week: str, monkeypatch: pytest.MonkeyPatch, extra: list[str] | None = None
    ) -> int:
        monkeypatch.setattr(
            run_weekend,
            "default_context",
            lambda **kw: self.context(
                week,
                allow_over_budget=kw.get("allow_over_budget", False),
                gateway_fallback=kw.get("gateway_fallback", False),
            ),
        )
        return int(run_weekend.main(["--synthetic", "--week", week, *(extra or [])]))


class GatewayDown(LlmProvider):
    """A gateway that refuses every call; the health check marks its sources failed."""

    supports_batch = False

    def complete(self, item: BatchItem, settings: LlmSettings) -> RawResult:
        raise ConnectionError("gateway unreachable")


def no_solver_config(cfg: StudyConfig) -> StudyConfig:
    """Every solver list points at a solver that does not exist: each optimizing strategy fails."""
    bad = ["NO_SUCH_SOLVER"]
    return cfg.model_copy(
        update={
            "optimizer": cfg.optimizer.model_copy(update={"solvers": bad}),
            "strategies": cfg.strategies.model_copy(
                update={
                    "cvar": cfg.strategies.cvar.model_copy(update={"solvers": bad}),
                    "robust": cfg.strategies.robust.model_copy(update={"solvers": bad}),
                }
            ),
        }
    )


def lock_cutoff_clock(week: str, cfg: StudyConfig, root: Path) -> Callable[[], dt.datetime]:
    """Normal Saturday time until J7 has written LOCK_CUTOFF_AFTER_FILES weight files, then a
    minute past the lock deadline, so the remaining portfolios get Q0 with reason lock_deadline."""
    from portstudy.llm.runner import lock_deadline

    early = synthetic_clock(week)
    late = lock_deadline(week, cfg) + dt.timedelta(minutes=1)
    folder = root / "outputs" / week / "weights"

    def now() -> dt.datetime:
        n_files = len(list(folder.glob("*.csv"))) if folder.exists() else 0
        return late if n_files >= LOCK_CUTOFF_AFTER_FILES else early()

    return now


def rehearsal_registry(registry: ModelRegistry) -> ModelRegistry:
    """stub-ow-a is fallback-eligible so --gateway-fallback switches it (D2); stub-ow-b is not."""
    stubs = [
        m.model_copy(update={"fallback_eligible": m.id == "stub-ow-a"})
        for m in registry.stub_models
    ]
    return registry.model_copy(update={"stub_models": stubs})


def run_rehearsal(
    runner: WeekendRunner, weeks: list[str], mp: pytest.MonkeyPatch, root: Path
) -> None:
    """The 30-week rehearsal: clean weeks plus one of each injected failure (INJECTIONS)."""
    from portstudy.llm.providers.stub import StubFailures, StubLlmProvider

    clean_cfg = runner.cfg
    for i, w in enumerate(weeks):
        kind = INJECTIONS.get(i)
        if kind == "source_failure":
            runner.llm_providers = {
                "stub-ow-b": StubLlmProvider("stub-ow-b", failures=StubFailures(invalid_rate=0.6))
            }
            assert runner.run(w, mp) == 0, w
        elif kind == "solver_failure":
            runner.cfg = no_solver_config(clean_cfg)
            assert runner.run(w, mp) == 0, w
            runner.cfg = clean_cfg
        elif kind == "lock_cutoff":
            runner.clock = lock_cutoff_clock(w, clean_cfg, root)
            assert runner.run(w, mp) == 0, w
            runner.clock = None
        elif kind == "gateway_health":
            runner.llm_providers = {"stub-ow-a": GatewayDown()}
            assert runner.run(w, mp) == 0, w  # D2 steps printed, stub-ow-a sources failed (N1)
            runner.llm_providers = {"stub-ow-a": StubLlmProvider("stub-ow-a")}
            assert runner.run(w, mp, ["--from", "J4", "--gateway-fallback"]) == 0, w
        else:
            assert runner.run(w, mp) == 0, w
        runner.llm_providers = {}


@pytest.fixture(scope="session")
def e2e_weeks(study_config: StudyConfig) -> list[str]:
    return cal.week_range(study_config.synthetic.first_week, E2E_WEEKS)


@pytest.fixture(scope="session")
def e2e_root(
    tmp_path_factory: pytest.TempPathFactory,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    e2e_weeks: list[str],
) -> Iterator[Path]:
    """The E2E_WEEKS-week synthetic rehearsal through run_weekend.py (see run_rehearsal)."""
    root = tmp_path_factory.mktemp("e2e_root")
    previous = os.environ.get("PORTSTUDY_DATA_DIR")
    os.environ["PORTSTUDY_DATA_DIR"] = str(root)
    mp = pytest.MonkeyPatch()
    try:
        runner = WeekendRunner(
            generous_budget(study_config), rehearsal_registry(model_registry), world
        )
        run_rehearsal(runner, e2e_weeks, mp, root)
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
