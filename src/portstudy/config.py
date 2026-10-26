"""Typed study configuration loaded from config/study.yaml and config/models.yaml.

Unknown keys are an error. Every deferred decision from devin.md section 0.3 is listed
under `decisions` with an id, a value, and a status of confirmed or provisional.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CheckRule = Literal["skip_stock", "forward_fill", "halt"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class UniverseConfig(StrictModel):
    size: int = 100
    source: str = "sp500"
    sectors: int = 11
    per_sector: int = 9
    min_history_days: int = 300
    dollar_volume_window_days: int = 60
    extra_tickers: list[str] = Field(default_factory=list)
    universe_file: str = "config/universe.csv"
    etf_proxies: list[str] = Field(default_factory=lambda: ["SPY", "IWM", "IWD", "IWF", "QUAL"])


class TimingConfig(StrictModel):
    timezone: str = "America/New_York"
    cutoff_time_et: str = "16:00"
    lock_deadline_time_et: str = "09:00"
    pilot_week: str | None = None
    live_start_week: str | None = None
    live_end_date: str | None = None


class RateLimits(StrictModel):
    alpaca_per_minute: int = 200
    finnhub_per_minute: int = 60
    fred_per_minute: int = 120
    yfinance_per_minute: int = 60


class RetryConfig(StrictModel):
    attempts: int = 3
    backoff_seconds: float = 2.0


class CheckRules(StrictModel):
    missing_days: CheckRule = "forward_fill"
    nonpositive_price: CheckRule = "skip_stock"
    extreme_move: CheckRule = "skip_stock"
    duplicate_rows: CheckRule = "skip_stock"
    stale_price: CheckRule = "skip_stock"
    no_news: CheckRule = "forward_fill"
    news_after_cutoff: CheckRule = "forward_fill"
    short_history: CheckRule = "skip_stock"
    stopped_trading: CheckRule = "skip_stock"
    extreme_move_threshold: float = 0.40
    stale_days: int = 5

    def rule_for(self, code: str) -> CheckRule:
        value = getattr(self, code)
        if value not in ("skip_stock", "forward_fill", "halt"):
            raise KeyError(code)
        return value  # type: ignore[no-any-return]


class DataConfig(StrictModel):
    price_provider: str = "alpaca"
    news_provider: str = "alpaca"
    reference_provider: str = "yfinance"
    rate_provider: str = "fred"
    earnings_provider: str = "finnhub"
    price_history_days: int = 300
    first_snapshot_news_days: int = 7
    news_token_budget_per_stock: int = 1500
    news_tokenizer: str | None = None
    rate_limits: RateLimits = RateLimits()
    retries: RetryConfig = RetryConfig()
    fixture_dir: str = "tests/fixtures/providers"
    checks: CheckRules = CheckRules()


class SyntheticConfig(StrictModel):
    seed: int = 20260101
    n_tickers: int = 120
    history_days: int = 400
    future_weeks: int = 40
    first_week: str = "2026-W02"
    world_dir: str = "data/synthetic"


class MomentumLookbacks(StrictModel):
    long: int = 252
    skip: int = 21
    medium: int = 63
    short: int = 21


class KronosConfig(StrictModel):
    repo_commit: str = "UNPINNED"
    model_id: str = "NeoQuasar/Kronos-base"
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base"
    context_candles: int = 252
    horizon: int = 5
    n_samples: int = 20
    temperature: float = 1.0
    top_p: float = 0.9
    seed: int = 7
    input_prices: Literal["adjusted", "raw"] = "adjusted"


class BaselineConfig(StrictModel):
    forecaster: Literal["stub", "kronos"] = "stub"
    momentum_lookbacks: MomentumLookbacks = MomentumLookbacks()
    kronos: KronosConfig = KronosConfig()
    stub_seed: int = 11


class RiskConfig(StrictModel):
    cov_lookback_days: int = 252
    delta: float = 2.5
    weekly_scale: float = 5.0
    min_eigenvalue: float = 1e-10
    market_cap_weights: str = "study_only"


class LlmSettings(StrictModel):
    temperature: float = 0.0
    seed: int = 1234
    max_output_tokens: int = 300
    reasoning: str = "none"
    think: bool = False


class LlmConfig(StrictModel):
    cost_goal_usd: float = 50.0
    cost_warn_usd: float = 50.0
    temperature: float = 0.0
    no_news_sentence: str = "No relevant news is available for this week."
    prompt_fields: list[str] = Field(default_factory=lambda: ["ticker", "name", "sector", "news"])
    skills_token_cap: int = 6000
    rationale_token_cap: int = 150
    placebo_length_tolerance: float = 0.02
    invalid_rate_threshold: float = 0.15
    retry_invalid_once: bool = True
    batch_retry_count: int = 1
    api_poll_seconds: int = 300
    settings_default: LlmSettings = LlmSettings()
    settings: dict[str, LlmSettings] = Field(default_factory=dict)

    def settings_for(self, model_id: str) -> LlmSettings:
        return self.settings.get(model_id, self.settings_default)


class ViewsConfig(StrictModel):
    tau: float = 0.05
    ic_window_weeks: int = 8
    ic_floor: float = 0.0
    cold_start_weeks: int = 5
    cold_start_ic: float = 0.02
    ic_min_weeks: int = 3
    confidence_clip: tuple[float, float] = (0.1, 1.0)
    ic_signal_for_llm: Literal["llm_resid", "llm_raw"] = "llm_resid"


class OptimizerConfig(StrictModel):
    max_weight: float = 0.05
    min_stocks: int = 20
    solvers: list[str] = Field(default_factory=lambda: ["OSQP", "CLARABEL"])


class SimulatorConfig(StrictModel):
    initial_value: float = 100000.0
    cost_bps: float = 10.0
    stop_loss: float = -0.08
    take_profit: float = 0.15


class PipelineConfig(StrictModel):
    jobs: list[str] = Field(
        default_factory=lambda: ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9"]
    )
    stop_on_failure: bool = True
    # J6 declares a run failed when it is not finished this many minutes before the lock, so
    # that J7 still has time to write the weights before the deadline.
    lock_margin_minutes: int = 30


class ShuffleConfig(StrictModel):
    n_shuffles: int = 500
    seed: int = 99
    processes: int = 1


class EvalConfig(StrictModel):
    tie_band: float = 0.01
    bootstrap_samples: int = 2000
    permutation_samples: int = 10000
    significance: float = 0.05
    seed: int = 2026
    combined_signal: str = "bl_tilt"
    factor_frequency: str = "daily"
    fama_french_file: str | None = None
    holm_families: dict[str, str] = Field(default_factory=dict)
    h4_splits: list[str] = Field(
        default_factory=lambda: ["market_direction", "volatility", "earnings_week"]
    )
    min_group_size: int = 10
    shuffle: ShuffleConfig = ShuffleConfig()


class BackupConfig(StrictModel):
    keep_zips: bool = True
    master_tables: list[str] = Field(
        default_factory=lambda: [
            "llm_scores",
            "trades",
            "daily_values",
            "weekly_ic",
            "fallback_log",
            "cost_log",
        ]
    )


class Decision(StrictModel):
    id: str
    value: Any
    status: Literal["confirmed", "provisional"]


class StudyConfig(StrictModel):
    universe: UniverseConfig = UniverseConfig()
    timing: TimingConfig = TimingConfig()
    data: DataConfig = DataConfig()
    synthetic: SyntheticConfig = SyntheticConfig()
    baseline: BaselineConfig = BaselineConfig()
    risk: RiskConfig = RiskConfig()
    llm: LlmConfig = LlmConfig()
    views: ViewsConfig = ViewsConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    simulator: SimulatorConfig = SimulatorConfig()
    pipeline: PipelineConfig = PipelineConfig()
    eval: EvalConfig = EvalConfig()
    backup: BackupConfig = BackupConfig()
    decisions: list[Decision] = Field(default_factory=list)

    @field_validator("decisions")
    @classmethod
    def _unique_decision_ids(cls, value: list[Decision]) -> list[Decision]:
        ids = [d.id for d in value]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate decision ids")
        return value

    def decision(self, decision_id: str) -> Decision:
        for d in self.decisions:
            if d.id == decision_id:
                return d
        raise KeyError(f"unknown decision {decision_id}")


class ModelPrice(StrictModel):
    input_per_m: float = 0.0
    cached_input_per_m: float = 0.0
    output_per_m: float = 0.0
    batch_discount: float = 0.0


class ModelSpec(StrictModel):
    id: str
    display_name: str
    kind: Literal["local", "api"]
    provider: str
    provider_model: str
    snapshot_id: str | None = None
    enabled: bool = True
    role: Literal["primary", "fallback"] = "primary"
    settings: LlmSettings = LlmSettings()
    price: ModelPrice = ModelPrice()


class ModelRegistry(StrictModel):
    models: list[ModelSpec]
    stub_models: list[ModelSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> ModelRegistry:
        ids = [m.id for m in self.models + self.stub_models]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate model ids in registry")
        return self

    def enabled(self, synthetic: bool = False) -> list[ModelSpec]:
        pool = self.stub_models if synthetic else self.models
        return [m for m in pool if m.enabled]

    def get(self, model_id: str) -> ModelSpec:
        for m in self.models + self.stub_models:
            if m.id == model_id:
                return m
        raise KeyError(f"unknown model {model_id}")


@lru_cache(maxsize=8)
def _resolved_root(env: str | None) -> Path:
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    """Repository root: PORTSTUDY_ROOT if set, else the parent of the src folder."""
    return _resolved_root(os.environ.get("PORTSTUDY_ROOT"))


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return loaded


def load_study_config(path: Path | None = None) -> StudyConfig:
    """Load and validate study.yaml. Unknown keys raise a validation error."""
    cfg_path = path or repo_root() / "config" / "study.yaml"
    return StudyConfig.model_validate(_read_yaml(cfg_path))


def load_model_registry(path: Path | None = None) -> ModelRegistry:
    """Load and validate models.yaml."""
    cfg_path = path or repo_root() / "config" / "models.yaml"
    return ModelRegistry.model_validate(_read_yaml(cfg_path))
