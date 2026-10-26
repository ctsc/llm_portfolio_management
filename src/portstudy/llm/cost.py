"""Cost table, weekly estimates, the running cost log, and the budget guard (DECISION(D3)).

Prices are per million tokens from config/models.yaml and are confirmed by the researcher in
Phase 17. The guard raises `CostCapExceeded` before any batch is submitted when the projected
study total would pass `llm.cost_goal_usd`; `--allow-over-budget` lets the run continue and
every such week is logged.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import pandas as pd

from portstudy import calendar as cal
from portstudy.config import LlmSettings, ModelRegistry, ModelSpec, StudyConfig
from portstudy.io import read_csv, write_csv_atomic
from portstudy.llm.prompts import token_count
from portstudy.llm.providers.base import BatchItem
from portstudy.paths import outputs_dir, week_output_dir

COST_LOG_COLUMNS = [
    "week_id",
    "model",
    "config_id",
    "n_calls",
    "n_cached",
    "tokens_in",
    "tokens_cached",
    "tokens_out",
    "cost_usd",
    "projected_study_usd",
    "over_budget_allowed",
]


class CostCapExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class CostTable:
    registry: ModelRegistry

    def price(self, model_id: str) -> ModelSpec:
        return self.registry.get(model_id)

    def cost_usd(
        self, model_id: str, tokens_in: int, tokens_cached: int, tokens_out: int, batch: bool
    ) -> float:
        p = self.price(model_id).price
        fresh = max(0, tokens_in - tokens_cached)
        usd = (
            fresh * p.input_per_m
            + tokens_cached * p.cached_input_per_m
            + tokens_out * p.output_per_m
        ) / 1e6
        if batch:
            usd *= 1.0 - p.batch_discount
        return usd


def estimate_week_cost(
    items: list[BatchItem],
    spec: ModelSpec,
    settings: LlmSettings,
    table: CostTable,
    assume_cached_prefix: bool = True,
) -> dict[str, float]:
    """Projected cost of `items` for one model. Output tokens are assumed at the max."""
    if not items:
        return {"cost_usd": 0.0, "cost_usd_no_cache": 0.0, "tokens_in": 0, "tokens_out": 0}
    tokens_in = 0
    tokens_cached = 0
    seen_prefix: set[str] = set()
    for item in items:
        n_prefix = token_count(item.prompt.shared_prefix, spec.provider_model)
        n_stock = token_count(item.prompt.stock_part, spec.provider_model)
        tokens_in += n_prefix + n_stock
        if assume_cached_prefix and item.prompt.prefix_hash in seen_prefix:
            tokens_cached += n_prefix
        seen_prefix.add(item.prompt.prefix_hash)
    tokens_out = len(items) * settings.max_output_tokens
    batch = spec.kind == "api"
    return {
        "cost_usd": table.cost_usd(spec.id, tokens_in, tokens_cached, tokens_out, batch),
        "cost_usd_no_cache": table.cost_usd(spec.id, tokens_in, 0, tokens_out, batch),
        "tokens_in": float(tokens_in),
        "tokens_out": float(tokens_out),
    }


def weeks_remaining(week_id: str, cfg: StudyConfig) -> int:
    """Weeks from `week_id` through `timing.live_end_date` inclusive (at least 1)."""
    end = cfg.timing.live_end_date
    if not end:
        return 1
    last = cal.week_id(dt.date.fromisoformat(end))
    try:
        return max(1, len(cal.weeks_between(week_id, last)))
    except ValueError:
        return 1


def spent_so_far(exclude_week: str | None = None) -> float:
    """Sum of actual cost over every week's cost_log.csv (optionally excluding one week)."""
    total = 0.0
    root = outputs_dir()
    for path in sorted(root.glob("*/cost_log.csv")):
        week = path.parent.name
        if week == exclude_week:
            continue
        df = read_csv(path)
        if len(df):
            total += float(df["cost_usd"].sum())
    return total


def check_budget(
    week_id: str,
    week_estimate_usd: float,
    cfg: StudyConfig,
    allow_over_budget: bool,
    logger: logging.Logger,
) -> float:
    """Return the projected study total; raise CostCapExceeded unless allowed."""
    spent = spent_so_far(exclude_week=week_id)
    projected = spent + week_estimate_usd * weeks_remaining(week_id, cfg)
    fields = {
        "week_id": week_id,
        "spent_usd": spent,
        "week_estimate_usd": week_estimate_usd,
        "projected_study_usd": projected,
        "goal_usd": cfg.llm.cost_goal_usd,
    }
    if projected >= cfg.llm.cost_warn_usd:
        logger.warning("cost_warning", extra={"extra_fields": fields})
    if projected > cfg.llm.cost_goal_usd:
        if not allow_over_budget:
            raise CostCapExceeded(
                f"projected study cost {projected:.2f} USD exceeds goal "
                f"{cfg.llm.cost_goal_usd:.2f}; pass --allow-over-budget to continue"
            )
        logger.warning("over_budget_allowed", extra={"extra_fields": fields})
    return projected


class RunningCost:
    """Per-week cost log at data/outputs/<week>/cost_log.csv, one row per (model, config)."""

    def __init__(self, week_id: str) -> None:
        self.week_id = week_id
        self.path = week_output_dir(week_id) / "cost_log.csv"

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=COST_LOG_COLUMNS)
        return read_csv(self.path)

    def record(self, row: dict[str, object]) -> None:
        df = self.read()
        row = {"week_id": self.week_id, **row}
        mask = (df["model"] == row["model"]) & (df["config_id"] == row["config_id"])
        df = df[~mask] if len(df) else df
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df = df.loc[:, COST_LOG_COLUMNS].sort_values(["model", "config_id"]).reset_index(drop=True)
        write_csv_atomic(df, self.path)

    def total(self) -> float:
        df = self.read()
        return float(df["cost_usd"].sum()) if len(df) else 0.0
