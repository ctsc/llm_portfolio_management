"""Fallback policy (confirmed design):

- A failed (model, configuration) for the week trades the Q0 weights with
  `decision_source = fallback_q0`; the week is marked `fallback` for that portfolio.
- A data or baseline failure that leaves Q0 unbuildable holds last week's positions for every
  portfolio (`week_status = no_rebalance`).
- `data/outputs/<week>/fallback_log.json` has one entry per fallback with the reason.
- `clean_weeks(portfolio)` returns the normal weeks only, used by IC history and evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from portstudy.io import read_json, write_json_atomic
from portstudy.paths import outputs_dir, week_output_dir
from portstudy.types import PortfolioId, RatingSource, WeekStatus

PortfolioStatus = Literal["normal", "fallback", "no_rebalance"]


@dataclass
class FallbackPlan:
    week_id: str
    statuses: dict[str, PortfolioStatus] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    no_rebalance: bool = False
    no_rebalance_reason: str = ""

    def status_for(self, portfolio: str) -> PortfolioStatus:
        if self.no_rebalance:
            return "no_rebalance"
        try:
            source = PortfolioId.parse(portfolio).source
        except ValueError:
            try:
                source = RatingSource.parse(portfolio)
            except ValueError:
                source = None
        return self.statuses.get(source.key, "normal") if source is not None else "normal"

    def reason_for(self, portfolio: str) -> str:
        try:
            source = PortfolioId.parse(portfolio).source
        except ValueError:
            try:
                source = RatingSource.parse(portfolio)
            except ValueError:
                source = None
        return self.reasons.get(source.key, "") if source is not None else ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FallbackPlan:
        return cls(
            week_id=str(d["week_id"]),
            statuses=dict(d.get("statuses", {})),
            reasons=dict(d.get("reasons", {})),
            no_rebalance=bool(d.get("no_rebalance", False)),
            no_rebalance_reason=str(d.get("no_rebalance_reason", "")),
        )


def plan_path(week_id: str) -> Path:
    return week_output_dir(week_id) / "fallback_plan.json"


def log_path(week_id: str) -> Path:
    return week_output_dir(week_id) / "fallback_log.json"


def write_plan(plan: FallbackPlan) -> None:
    write_json_atomic(plan.to_dict(), plan_path(plan.week_id))


def read_plan(week_id: str) -> FallbackPlan | None:
    path = plan_path(week_id)
    return FallbackPlan.from_dict(read_json(path)) if path.exists() else None


def write_log(week_id: str, entries: list[dict[str, Any]]) -> None:
    write_json_atomic({"week_id": week_id, "entries": entries}, log_path(week_id))


def read_log(week_id: str) -> list[dict[str, Any]]:
    path = log_path(week_id)
    if not path.exists():
        return []
    data = read_json(path)
    return list(data.get("entries", []))


def week_status_for(week_id: str, portfolio: str) -> WeekStatus:
    """As-traded status for a portfolio in a week, from the fallback log written by J7."""
    plan = read_plan(week_id)
    if plan is not None:
        status = plan.status_for(portfolio)
        if status != "normal":
            return status
    for e in read_log(week_id):
        if e.get("portfolio") in (portfolio, "*"):
            kind = str(e.get("status", "normal"))
            if kind in ("fallback", "no_rebalance"):
                return kind  # type: ignore[return-value]
    return "normal"


def weeks_with_logs() -> list[str]:
    return sorted(p.parent.name for p in outputs_dir().glob("*/fallback_log.json"))


def fallback_weeks(portfolio: str) -> set[str]:
    """Weeks that are not normal for the portfolio (fallback or no_rebalance)."""
    return {w for w in weeks_with_logs() if week_status_for(w, portfolio) != "normal"}


def clean_weeks(portfolio: str, weeks: list[str] | None = None) -> list[str]:
    """DECISION(N1): only weeks traded normally count toward IC history and the tests."""
    candidates = weeks if weeks is not None else weeks_with_logs()
    return [w for w in candidates if week_status_for(w, portfolio) == "normal"]
