"""LLM output schema: an ordered integer range from -5 to 5 and a rationale (DECISION(C3))."""

from __future__ import annotations

from portstudy.types import LlmResponse

RANGE_MIN = -5
RANGE_MAX = 5

# Fallback record written when the output is invalid after one retry.
INVALID_OUTPUT = "invalid_output"
INVALID_TRUNCATED = "invalid_truncated"

__all__ = [
    "INVALID_OUTPUT",
    "INVALID_TRUNCATED",
    "RANGE_MAX",
    "RANGE_MIN",
    "LlmResponse",
    "fallback_response",
]


def fallback_response(raw: str, error: str, flag: str = INVALID_OUTPUT) -> LlmResponse:
    return LlmResponse(
        low=0,
        likely=0,
        high=0,
        rationale="",
        valid=False,
        raw=raw,
        error=error,
        fallback_flag=flag,
    )
