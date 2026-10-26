"""LLM output schema: integer score -5..5, confidence 0..1, rationale with a token cap."""

from __future__ import annotations

from portstudy.types import LlmResponse

SCORE_MIN = -5
SCORE_MAX = 5

# Fallback record written when the output is invalid after one retry.
FALLBACK_SCORE = 0
FALLBACK_CONFIDENCE = 0.1
INVALID_OUTPUT = "invalid_output"
INVALID_TRUNCATED = "invalid_truncated"

__all__ = [
    "FALLBACK_CONFIDENCE",
    "FALLBACK_SCORE",
    "INVALID_OUTPUT",
    "INVALID_TRUNCATED",
    "SCORE_MAX",
    "SCORE_MIN",
    "LlmResponse",
    "fallback_response",
]


def fallback_response(raw: str, error: str, flag: str = INVALID_OUTPUT) -> LlmResponse:
    return LlmResponse(
        score=FALLBACK_SCORE,
        confidence=FALLBACK_CONFIDENCE,
        rationale="",
        valid=False,
        raw=raw,
        error=error,
        fallback_flag=flag,
    )
