"""Shared run context for the weekly jobs: config, registries, providers, clock, flags."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
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
from portstudy.types import RatingSource

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
        available = self.registry.enabled(synthetic=self.synthetic)
        selected = self.cfg.llm.model_ids
        if not selected:
            return [m.id for m in available]
        available_by_id = {m.id: m for m in available}
        missing = [model_id for model_id in selected if model_id not in available_by_id]
        if missing:
            raise ValueError(f"configured models are unavailable: {missing}")
        return selected

    def prompt_ids(self) -> list[str]:
        available = self.manifest.enabled()
        selected = self.cfg.llm.prompt_ids
        if not selected:
            return [prompt.id for prompt in available]
        available_by_id = {prompt.id: prompt for prompt in available}
        missing = [prompt_id for prompt_id in selected if prompt_id not in available_by_id]
        if missing:
            raise ValueError(f"configured prompt ids are unavailable: {missing}")
        return selected

    def run_registry(self) -> ModelRegistry:
        selected = set(self.models())
        return self.registry.model_copy(
            update={
                "models": [model for model in self.registry.models if model.id in selected],
                "stub_models": [
                    model for model in self.registry.stub_models if model.id in selected
                ],
            }
        )

    def run_manifest(self) -> PromptManifest:
        selected = set(self.prompt_ids())
        return self.manifest.model_copy(
            update={
                "configurations": [
                    prompt for prompt in self.manifest.configurations if prompt.id in selected
                ]
            }
        )

    def local_models(self) -> list[str]:
        return [
            model_id for model_id in self.models() if self.registry.get(model_id).kind == "local"
        ]

    def api_models(self) -> list[str]:
        return [model_id for model_id in self.models() if self.registry.get(model_id).kind == "api"]

    def sources(self) -> list[RatingSource]:
        return [
            RatingSource(model, prompt) for model in self.models() for prompt in self.prompt_ids()
        ]

    def steered_keys(self) -> list[str]:
        from portstudy.types import PortfolioId

        return [
            PortfolioId.steered(strategy, source).key
            for strategy in self.cfg.strategies.enabled
            for source in self.sources()
        ]

    def q0_keys(self) -> list[str]:
        from portstudy.types import PortfolioId

        return [PortfolioId.q0(strategy).key for strategy in self.cfg.strategies.enabled]

    def control_keys(self) -> list[str]:
        from portstudy.types import PortfolioId

        return [PortfolioId.control(name).key for name in self.cfg.strategies.controls]

    def portfolio_keys(self) -> list[str]:
        return self.q0_keys() + self.control_keys() + self.steered_keys()


def default_context(
    synthetic: bool = False,
    allow_over_budget: bool = False,
    force: bool = False,
    clock: Callable[[], dt.datetime] | None = None,
    config_path: Path | None = None,
) -> RunContext:
    return RunContext(
        cfg=load_study_config(config_path),
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
