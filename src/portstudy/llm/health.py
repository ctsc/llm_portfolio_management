"""Gateway health check before J4 (v2 phase 12). One tiny call per enabled open-weight model
that is served by the LiteLLM gateway. A failing model's sources are marked failed in
`run_status.json` (N1), so J4 skips them and J6 serves the strategy's Q0 for them (DECISION(D2));
rerunning `python scripts/run_weekend.py --from J4` retries when the gateway is back."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from portstudy.io import read_json, write_json_atomic
from portstudy.llm.prompts import assemble_prompt
from portstudy.llm.providers.base import BatchItem, make_custom_id
from portstudy.llm.runner import (
    RUN_STATUS_LOCK,
    RunStatus,
    WeekRunner,
    _settings_for,
    read_run_status,
)
from portstudy.paths import week_output_dir
from portstudy.pipeline.context import RunContext

HEALTH_REASON = "gateway_health_check_failed"
HEALTH_TICKER = "HLTH"


@dataclass
class HealthReport:
    week_id: str
    checked_at: str
    ok: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)  # not on the gateway (fallback or stub)
    previous: list[dict[str, Any]] = field(default_factory=list)  # earlier checks this week

    @property
    def healthy(self) -> bool:
        return not self.failed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def health_path(week_id: str) -> Any:
    return week_output_dir(week_id) / "gateway_health.json"


def read_health(week_id: str) -> HealthReport | None:
    path = health_path(week_id)
    if not path.exists():
        return None
    data = read_json(path)
    return HealthReport(**data)


def unhealthy_models(week_id: str) -> set[str]:
    report = read_health(week_id)
    return set(report.failed) if report else set()


def _tiny_item(ctx: RunContext, week_id: str, model_id: str, config_id: str) -> BatchItem:
    prompt = assemble_prompt(
        config_id,
        {"ticker": HEALTH_TICKER, "name": "Health Check Co", "sector": "Information Technology"},
        "Health Check Co reports a routine quarterly update.",
        ctx.cfg,
        ctx.run_manifest(),
    )
    return BatchItem(
        make_custom_id(model_id, config_id, f"{week_id}-health", HEALTH_TICKER),
        model_id,
        config_id,
        week_id,
        HEALTH_TICKER,
        prompt,
    )


def gateway_health_check(ctx: RunContext, week_id: str, runner: WeekRunner) -> HealthReport:
    """Send one tiny call per open-weight model served by the gateway and write the report."""
    registry = ctx.run_registry()
    config_id = ctx.prompt_ids()[0]
    report = HealthReport(week_id=week_id, checked_at=ctx.now().isoformat())
    for model_id in ctx.open_weight_models():
        spec = registry.get(model_id)
        if spec.serving != "gateway" or (
            spec.provider == "stub" and model_id not in ctx.llm_providers
        ):
            report.skipped.append(model_id)
            continue
        try:
            provider = runner.provider(model_id, config_id)
            result = provider.complete(
                _tiny_item(ctx, week_id, model_id, config_id), _settings_for(ctx.cfg, spec)
            )
            if result.error:
                raise RuntimeError(result.error)
            report.ok.append(model_id)
        except Exception as exc:  # any transport or provider failure counts as unhealthy
            report.failed[model_id] = f"{type(exc).__name__}: {exc}"
    earlier = read_health(week_id)
    if earlier is not None:
        report.previous = [{k: v for k, v in earlier.to_dict().items() if k != "previous"}]
        report.previous += earlier.previous
    write_json_atomic(report.to_dict(), health_path(week_id))
    return report


def mark_sources_failed(ctx: RunContext, week_id: str, models: list[str], reason: str) -> list[str]:
    """Write a failed `RunStatus` for every (model, prompt) source of the given models (N1)."""
    with RUN_STATUS_LOCK:
        return _mark_sources_failed(ctx, week_id, models, reason)


def _mark_sources_failed(
    ctx: RunContext, week_id: str, models: list[str], reason: str
) -> list[str]:
    path = week_output_dir(week_id) / "run_status.json"
    data: dict[str, Any] = read_json(path) if path.exists() else {}
    now = ctx.now().isoformat()
    keys = []
    for model_id in models:
        for config_id in ctx.prompt_ids():
            key = f"{model_id}__{config_id}"
            previous = read_run_status(week_id).get(key)
            status = RunStatus(
                model=model_id,
                config_id=config_id,
                state="failed",
                reason=reason,
                assessed_at=now,
                endpoint="gateway",
                started_at=previous.started_at if previous and previous.started_at else now,
            )
            data[key] = status.to_dict()
            keys.append(key)
    write_json_atomic(data, path)
    return keys


def outage_lines(failed: dict[str, str]) -> list[str]:
    """The DECISION(D2) message printed when the gateway is down."""
    detail = ", ".join(f"{m} ({e})" for m, e in sorted(failed.items()))
    return [
        f"gateway health check failed for {detail}",
        "D2: these models' sources are marked failed for the week and trade their strategy's "
        "Q0; check the GSU VPN and rerun `python scripts/run_weekend.py --from J4` to retry "
        "before the lock",
    ]
