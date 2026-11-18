"""ARCTIC usage tracking (DECISION(N7)).

GPU-hours are estimated from the week's gateway requests times `arctic.gpu_seconds_per_request`;
gateway dollars are the sum of every gateway response's `x-litellm-response-cost` header (canary
and repeat-check requests included). Both feed the study log and the weekend summary. The run
never stops on GPU-hours; before J4 it stops when the gateway dollars spent plus the week's
estimate would pass `arctic.litellm_budget_usd`, so a budget error cannot cut a week in half.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from portstudy import calendar as cal
from portstudy.config import ModelRegistry, StudyConfig
from portstudy.io import read_csv
from portstudy.paths import outputs_dir, week_output_dir

COST_HEADER = "x-litellm-response-cost"
_LOG = logging.getLogger("portstudy.llm.usage")


@dataclass
class WeekUsage:
    week_id: str
    requests: dict[str, int] = field(default_factory=dict)  # per gateway model, cache misses
    gpu_hours_est: float = 0.0
    litellm_cost_usd: float = 0.0
    n_cost_missing: int = 0
    n_cost_priced: int = 0

    @property
    def n_requests(self) -> int:
        return sum(self.requests.values())


def _record_files(week_id: str) -> list[Path]:
    folder = week_output_dir(week_id)
    # Portfolio records plus the separately kept canary and repeat-check records (G20, G21).
    return [
        *sorted((folder / "llm").glob("*.jsonl")),
        *sorted((folder / "canary").glob("*.jsonl")),
        *sorted((folder / "repeat_check").glob("*.jsonl")),
    ]


def _cost(record: dict[str, Any]) -> float | None:
    headers = record.get("gateway_headers") or {}
    value = headers.get(COST_HEADER) if isinstance(headers, dict) else None
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def week_usage(cfg: StudyConfig, registry: ModelRegistry, week_id: str) -> WeekUsage:
    usage = WeekUsage(week_id)
    specs = [*registry.models, *registry.stub_models]
    gateway = {m.id for m in specs if m.kind == "open_weight"}
    # Only real LiteLLM responses can carry the cost header; stub models never do.
    priced = {m.id for m in specs if m.kind == "open_weight" and m.provider == "litellm"}
    for path in _record_files(week_id):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            model = str(rec.get("model", ""))
            if model not in gateway or rec.get("from_cache") or rec.get("endpoint") != "gateway":
                continue
            if rec.get("fallback_flag") == "provider_error" and not rec.get("tokens_in"):
                continue  # never answered: no gateway compute or cost
            usage.requests[model] = usage.requests.get(model, 0) + 1
            cost = _cost(rec)
            if cost is None:
                usage.n_cost_missing += model in priced
            else:
                usage.litellm_cost_usd += cost
                usage.n_cost_priced += 1
    seconds = cfg.arctic.gpu_seconds_per_request
    usage.gpu_hours_est = sum(n * seconds.get(m, 0.0) for m, n in usage.requests.items()) / 3600
    return usage


def weeks_with_records() -> list[str]:
    root = outputs_dir()
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "llm").exists())


@dataclass
class UsageSummary:
    week: WeekUsage
    gpu_hours_total: float
    gpu_hours_projected: float
    litellm_cost_total: float
    litellm_cost_projected: float
    messages: list[str]


def _weeks_remaining(cfg: StudyConfig, week_id: str) -> int:
    end = cfg.timing.live_end_date
    if not end:
        return 0
    last = cal.week_id(pd.Timestamp(end).date())
    return max(0, len(cal.weeks_between(week_id, last)) - 1) if last >= week_id else 0


def usage_summary(cfg: StudyConfig, registry: ModelRegistry, week_id: str) -> UsageSummary:
    """The week's usage, the running totals through the week and the end-of-study projection,
    with the 75% warning and 90% alert lines for GPU-hours and gateway dollars."""
    weeks = [w for w in weeks_with_records() if w <= week_id]
    per_week = [week_usage(cfg, registry, w) for w in weeks]
    this = next((u for u in per_week if u.week_id == week_id), WeekUsage(week_id))
    gpu_total = sum(u.gpu_hours_est for u in per_week)
    cost_total = sum(u.litellm_cost_usd for u in per_week)
    n = max(1, len(per_week))
    remaining = _weeks_remaining(cfg, week_id)
    gpu_proj = gpu_total + remaining * gpu_total / n
    cost_proj = cost_total + remaining * cost_total / n
    a = cfg.arctic
    messages: list[str] = []
    for label, projected, limit, unit in (
        ("ARCTIC GPU-hours", gpu_proj, a.gpu_hours_allocated, "h"),
        ("gateway dollars", cost_proj, a.litellm_budget_usd, "USD"),
    ):
        if limit <= 0:
            continue
        share = projected / limit
        if share > a.alert_share:
            messages.append(
                f"ALERT: projected {label} {projected:.2f} {unit} is {share:.0%} of {limit:g}"
            )
        elif share > a.warn_share:
            messages.append(
                f"warning: projected {label} {projected:.2f} {unit} is {share:.0%} of {limit:g}"
            )
    if this.n_cost_missing and this.n_requests:
        messages.append(
            f"warning: {this.n_cost_missing} of {this.n_requests} gateway responses had no "
            f"{COST_HEADER} header; gateway dollars are understated"
        )
    return UsageSummary(this, gpu_total, gpu_proj, cost_total, cost_proj, messages)


def _cost_per_request(cfg: StudyConfig, registry: ModelRegistry, week_id: str) -> float | None:
    """Mean gateway dollars per request: the most recent earlier week with cost data, else the
    timing run (`gateway_timing.csv`). None when neither exists."""
    for w in sorted((w for w in weeks_with_records() if w < week_id), reverse=True):
        u = week_usage(cfg, registry, w)
        if u.n_cost_priced > 0:
            return u.litellm_cost_usd / u.n_cost_priced
    timing = outputs_dir() / "gateway_timing.csv"
    if timing.exists():
        df = read_csv(timing)
        if "response_cost" in df.columns:
            costs = pd.to_numeric(df["response_cost"], errors="coerce").dropna()
            if len(costs):
                return float(costs.mean())
    return None


@dataclass
class BudgetCheck:
    ok: bool
    message: str
    spent_usd: float = 0.0
    estimate_usd: float | None = None


def check_dollar_budget(
    cfg: StudyConfig,
    registry: ModelRegistry,
    week_id: str,
    n_planned: int,
    allow_over_budget: bool = False,
) -> BudgetCheck:
    """Pre-J4 check: spent gateway dollars plus this week's estimate against the budget."""
    spent = sum(
        week_usage(cfg, registry, w).litellm_cost_usd for w in weeks_with_records() if w <= week_id
    )
    per_request = _cost_per_request(cfg, registry, week_id)
    if per_request is None:
        msg = (
            f"gateway dollar check skipped: no {COST_HEADER} values yet "
            "(no earlier week and no timing run with costs)"
        )
        _LOG.warning("litellm_budget_unchecked", extra={"extra_fields": {"week_id": week_id}})
        return BudgetCheck(True, msg, spent, None)
    estimate = per_request * n_planned
    budget = cfg.arctic.litellm_budget_usd
    if spent + estimate <= budget:
        return BudgetCheck(
            True,
            f"gateway dollars: spent {spent:.2f} + this week about {estimate:.2f} of {budget:g}",
            spent,
            estimate,
        )
    msg = (
        f"gateway dollars: spent {spent:.2f} + this week about {estimate:.2f} would pass the "
        f"{budget:g} USD LiteLLM limit; ask ARCTIC to raise the LiteLLM limit"
    )
    if allow_over_budget:
        return BudgetCheck(True, msg + " (continuing: --allow-over-budget)", spent, estimate)
    return BudgetCheck(False, msg, spent, estimate)
