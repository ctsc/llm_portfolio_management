"""Build an LlmProvider from a ModelSpec."""

from __future__ import annotations

from portstudy.config import GatewayConfig, ModelSpec
from portstudy.llm.providers.base import LlmProvider
from portstudy.llm.providers.litellm import LiteLLMProvider
from portstudy.llm.providers.stub import StubFailures, StubLlmProvider


def make_provider(
    spec: ModelSpec,
    stub_seed: int = 0,
    stub_failures: StubFailures | None = None,
    stub_noise: dict[str, float] | None = None,
    gateway: GatewayConfig | None = None,
    timeout: float = 180.0,
    features_frozen: bool | None = None,
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
        base_url = gateway_config.base_url
        api_key_env = "ARCTIC_LITELLM_API_KEY"
        if not base_url:
            raise ValueError(f"LLM endpoint URL is not configured for {spec.id} ({endpoint})")
        return LiteLLMProvider(
            spec.id,
            spec.provider_model,
            base_url,
            api_key_env,
            endpoint_name=endpoint,
            timeout=timeout,
            features_frozen=(
                gateway_config.features_frozen if features_frozen is None else features_frozen
            ),
        )
    if spec.provider == "anthropic_batch":
        from portstudy.llm.providers.anthropic_batch import AnthropicBatchProvider

        return AnthropicBatchProvider(spec.id, spec.snapshot_id or spec.provider_model)
    if spec.provider == "openai_batch":
        from portstudy.llm.providers.openai_batch import OpenAIBatchProvider

        return OpenAIBatchProvider(spec.id, spec.snapshot_id or spec.provider_model)
    raise ValueError(f"unknown LLM provider {spec.provider}")
