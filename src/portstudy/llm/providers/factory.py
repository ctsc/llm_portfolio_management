"""Build an LlmProvider from a ModelSpec."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from portstudy.config import GatewayConfig, ModelSpec, repo_root
from portstudy.io import read_json
from portstudy.llm.providers.base import LlmProvider
from portstudy.llm.providers.litellm import LiteLLMProvider
from portstudy.llm.providers.stub import StubFailures, StubLlmProvider


def make_provider(
    spec: ModelSpec,
    stub_seed: int = 0,
    stub_failures: StubFailures | None = None,
    stub_noise: dict[str, float] | None = None,
    gateway: GatewayConfig | None = None,
    week_id: str | None = None,
    run_started_at: str | None = None,
) -> LlmProvider:
    if spec.provider == "stub":
        return StubLlmProvider(
            spec.id,
            seed=stub_seed,
            noise_by_config=stub_noise,
            model_version=spec.snapshot_id or "stub",
            failures=stub_failures,
        )
    if spec.provider == "litellm":
        gateway_config = gateway or GatewayConfig()
        endpoint = spec.serving
        if spec.serving == "gateway":
            base_url = gateway_config.base_url
            api_key_env = "ARCTIC_LITELLM_API_KEY"
        else:
            base_url = _fallback_base_url(gateway_config, week_id, run_started_at)
            api_key_env = "ARCTIC_FALLBACK_API_KEY"
        if not base_url:
            raise ValueError(f"LLM endpoint URL is not configured for {spec.id} ({endpoint})")
        return LiteLLMProvider(
            spec.id,
            spec.provider_model,
            base_url,
            api_key_env,
            endpoint_name=endpoint,
        )
    if spec.provider == "anthropic_batch":
        from portstudy.llm.providers.anthropic_batch import AnthropicBatchProvider

        return AnthropicBatchProvider(spec.id, spec.snapshot_id or spec.provider_model)
    if spec.provider == "openai_batch":
        from portstudy.llm.providers.openai_batch import OpenAIBatchProvider

        return OpenAIBatchProvider(spec.id, spec.snapshot_id or spec.provider_model)
    raise ValueError(f"unknown LLM provider {spec.provider}")


def _fallback_base_url(
    gateway: GatewayConfig, week_id: str | None, run_started_at: str | None
) -> str:
    path = Path(gateway.fallback_endpoint_file)
    if not path.is_absolute():
        path = repo_root() / path
    if not path.exists():
        raise FileNotFoundError(f"fallback endpoint file is missing: {path}")
    if not week_id or not run_started_at:
        raise ValueError("fallback endpoint requires an existing source-week run")
    data: Any = read_json(path)
    base_url = data.get("base_url", data.get("url")) if isinstance(data, dict) else None
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError(f"fallback endpoint file has no base_url: {path}")
    started = datetime.fromisoformat(run_started_at).timestamp()
    if path.stat().st_mtime < started:
        raise ValueError(f"fallback endpoint file is stale for {week_id}: {path}")
    endpoint_week = data.get("week_id")
    if endpoint_week is not None and endpoint_week != week_id:
        raise ValueError(
            f"fallback endpoint file is for {endpoint_week}, not current week {week_id}: {path}"
        )
    return base_url
