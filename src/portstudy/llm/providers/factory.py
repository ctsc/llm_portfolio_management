"""Build an LlmProvider from a ModelSpec."""

from __future__ import annotations

from portstudy.config import ModelSpec
from portstudy.llm.providers.base import LlmProvider
from portstudy.llm.providers.stub import StubFailures, StubLlmProvider


def make_provider(
    spec: ModelSpec,
    stub_seed: int = 0,
    stub_failures: StubFailures | None = None,
    stub_noise: dict[str, float] | None = None,
) -> LlmProvider:
    if spec.provider == "stub":
        return StubLlmProvider(
            spec.id,
            seed=stub_seed,
            noise_by_config=stub_noise,
            model_version=spec.snapshot_id or "stub",
            failures=stub_failures,
        )
    if spec.provider == "ollama":
        from portstudy.llm.providers.ollama import OllamaProvider

        return OllamaProvider(spec.id, spec.provider_model)
    if spec.provider == "anthropic_batch":
        from portstudy.llm.providers.anthropic_batch import AnthropicBatchProvider

        return AnthropicBatchProvider(spec.id, spec.snapshot_id or spec.provider_model)
    if spec.provider == "openai_batch":
        from portstudy.llm.providers.openai_batch import OpenAIBatchProvider

        return OpenAIBatchProvider(spec.id, spec.snapshot_id or spec.provider_model)
    raise ValueError(f"unknown LLM provider {spec.provider}")
