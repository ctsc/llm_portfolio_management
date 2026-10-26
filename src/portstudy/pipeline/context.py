"""Shared run context for the weekly jobs: config, registries, providers, clock, flags."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

from portstudy.config import (
    ModelRegistry,
    StudyConfig,
    load_model_registry,
    load_study_config,
)
from portstudy.data.providers.base import ProviderBundle
from portstudy.llm.prompts import PromptManifest, load_manifest
from portstudy.llm.providers.base import LlmProvider
from portstudy.logging_setup import get_logger

ET = ZoneInfo("America/New_York")


@dataclass
class RunContext:
    cfg: StudyConfig
    registry: ModelRegistry
    manifest: PromptManifest
    synthetic: bool = False
    allow_over_budget: bool = False
    force: bool = False
    data_providers: ProviderBundle | None = None
    llm_providers: dict[str, LlmProvider] = field(default_factory=dict)
    clock: Callable[[], dt.datetime] | None = None
    logger: logging.Logger | None = None
    stub_seed: int = 0

    def now(self) -> dt.datetime:
        return self.clock() if self.clock else dt.datetime.now(tz=ET)

    def log(self, job: str = "pipeline") -> logging.Logger:
        return self.logger or get_logger(job)

    @property
    def live_start_week(self) -> str:
        if self.cfg.timing.live_start_week:
            return self.cfg.timing.live_start_week
        if self.synthetic:
            return self.cfg.synthetic.first_week
        raise ValueError("timing.live_start_week is not set")

    def providers(self) -> ProviderBundle:
        if self.data_providers is None:
            from portstudy.data.providers.factory import build_providers

            self.data_providers = build_providers(self.cfg, synthetic=self.synthetic)
        return self.data_providers

    def models(self) -> list[str]:
        return [m.id for m in self.registry.enabled(synthetic=self.synthetic)]

    def local_models(self) -> list[str]:
        return [m.id for m in self.registry.enabled(synthetic=self.synthetic) if m.kind == "local"]

    def api_models(self) -> list[str]:
        return [m.id for m in self.registry.enabled(synthetic=self.synthetic) if m.kind == "api"]

    def portfolio_keys(self) -> list[str]:
        from portstudy.types import PortfolioId

        keys = [PortfolioId(m, c.id).key for m in self.models() for c in self.manifest.enabled()]
        return keys


def default_context(
    synthetic: bool = False,
    allow_over_budget: bool = False,
    force: bool = False,
    clock: Callable[[], dt.datetime] | None = None,
) -> RunContext:
    return RunContext(
        cfg=load_study_config(),
        registry=load_model_registry(),
        manifest=load_manifest(),
        synthetic=synthetic,
        allow_over_budget=allow_over_budget,
        force=force,
        clock=clock,
    )


def synthetic_clock(week_id: str) -> Callable[[], dt.datetime]:
    """A clock fixed at Friday 20:00 ET of the week's cutoff, before the Monday lock."""
    from portstudy import calendar as cal

    fixed = dt.datetime.combine(cal.cutoff_day(week_id), dt.time(20, 0), tzinfo=ET)
    return lambda: fixed
