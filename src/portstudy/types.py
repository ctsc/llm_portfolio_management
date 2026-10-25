"""Shared typed records used across the pipeline."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

WeekId = str
Ticker = str

TradeReason = Literal["rebalance", "stop_loss", "take_profit", "forced_exit"]
DecisionSource = Literal["llm", "fallback_q0", "hold", "q0"]
WeekStatus = Literal["normal", "fallback", "no_rebalance"]


@dataclass(frozen=True)
class PortfolioId:
    """A portfolio is one (model, configuration) pair, or Q0 for the quant baseline view."""

    model: str
    config_id: str

    @property
    def key(self) -> str:
        return "Q0" if self.model == "Q0" else f"{self.model}__{self.config_id}"

    @classmethod
    def q0(cls) -> PortfolioId:
        return cls(model="Q0", config_id="q0")

    @classmethod
    def parse(cls, key: str) -> PortfolioId:
        if key == "Q0":
            return cls.q0()
        model, config_id = key.split("__", 1)
        return cls(model=model, config_id=config_id)

    @property
    def is_q0(self) -> bool:
        return self.model == "Q0"


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
    score: int
    confidence: float
    rationale: str
    valid: bool
    raw: str
    error: str | None = None
    fallback_flag: str | None = None

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
