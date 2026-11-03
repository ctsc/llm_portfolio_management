"""Shared typed records used across the pipeline."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

WeekId = str
Ticker = str

TradeReason = Literal["rebalance", "stop_loss", "take_profit", "forced_exit"]
DecisionSource = Literal["llm", "fallback_q0", "hold", "q0", "control"]
WeekStatus = Literal["normal", "fallback", "no_rebalance"]
StrategyName = Literal["mvo", "cvar", "robust", "resampled", "topk"]
ControlName = Literal["HRP", "EW"]


@dataclass(frozen=True)
class RatingSource:
    model: str
    config_id: str

    @property
    def key(self) -> str:
        return f"{self.model}__{self.config_id}"

    @classmethod
    def parse(cls, key: str) -> RatingSource:
        parts = key.split("__")
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"invalid rating source key: {key}")
        model, config_id = parts
        if model in ("Q0", "HRP", "EW"):
            raise ValueError(f"reserved model id in rating source key: {key}")
        return cls(model, config_id)


@dataclass(frozen=True)
class PortfolioId:
    """A portfolio is a strategy and rating source, a strategy Q0, or a control."""

    strategy: str
    model: str
    config_id: str

    @classmethod
    def steered(cls, strategy: StrategyName, source: RatingSource) -> PortfolioId:
        return cls(strategy, source.model, source.config_id)

    @classmethod
    def q0(cls, strategy: StrategyName) -> PortfolioId:
        return cls(strategy, "Q0", "")

    @classmethod
    def control(cls, name: ControlName) -> PortfolioId:
        return cls(name, "", "")

    @property
    def key(self) -> str:
        if self.is_control:
            return self.strategy
        if self.is_q0:
            return f"{self.strategy}__Q0"
        return f"{self.strategy}__{self.model}__{self.config_id}"

    @classmethod
    def parse(cls, key: str) -> PortfolioId:
        if key in ("HRP", "EW"):
            return cls.control(key)  # type: ignore[arg-type]
        parts = key.split("__")
        strategies = ("mvo", "cvar", "robust", "resampled", "topk")
        if len(parts) == 2 and parts[0] in strategies and parts[1] == "Q0":
            return cls.q0(parts[0])  # type: ignore[arg-type]
        if (
            len(parts) == 3
            and parts[0] in strategies
            and all(parts[1:])
            and parts[1] not in ("Q0", "HRP", "EW")
        ):
            source = RatingSource(parts[1], parts[2])
            return cls.steered(parts[0], source)  # type: ignore[arg-type]
        raise ValueError(f"invalid portfolio key: {key}")

    @property
    def is_q0(self) -> bool:
        return self.model == "Q0" and not self.config_id and self.strategy not in ("HRP", "EW")

    @property
    def is_control(self) -> bool:
        return self.strategy in ("HRP", "EW") and not self.model and not self.config_id

    @property
    def source(self) -> RatingSource | None:
        if self.is_q0 or self.is_control:
            return None
        return RatingSource(self.model, self.config_id)


@dataclass(frozen=True)
class SnapshotRef:
    week_id: WeekId
    cutoff_date: dt.date
    trade_date: dt.date
    window_end_date: dt.date
    path: Path
    version: int = 1


@dataclass
class LlmResponse:
    low: int
    likely: int
    high: int
    rationale: str
    valid: bool
    raw: str
    error: str | None = None
    fallback_flag: str | None = None
    extra_keys: list[str] = field(default_factory=list)
    had_thinking: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trade:
    week_id: WeekId
    date: dt.date
    portfolio_id: str
    ticker: Ticker
    side: Literal["buy", "sell"]
    shares: float
    fill_price: float
    trade_value: float
    cost_bps: float
    cost_usd: float
    reason: TradeReason
    decision_source: DecisionSource
    avg_entry_price_before: float
    avg_entry_price_after: float
    weight_before: float
    weight_after: float
    stop_trigger: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["date"] = self.date.isoformat()
        return d


@dataclass
class DailyRecord:
    date: dt.date
    portfolio_id: str
    value: float
    cash: float
    cumulative_cost: float
    n_positions: int
    week_status: WeekStatus
    week_id: WeekId = ""
    flags: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["date"] = self.date.isoformat()
        return d


@dataclass
class Issue:
    severity: Literal["info", "warning", "error"]
    ticker: str
    code: str
    message: str
    rule: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
