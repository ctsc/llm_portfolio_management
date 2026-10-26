"""Parse and validate a raw model answer into an LlmResponse.

Valid only if JSON parses, `score` is an integer in [-5, 5], `confidence` is in [0, 1], and
`rationale` is a string (truncated to the token cap). Code fences are stripped. Anything else
is invalid. A truncated answer (hit the output token limit) is flagged `invalid_truncated`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from portstudy.llm.prompts import approx_tokens
from portstudy.llm.schema import (
    INVALID_OUTPUT,
    INVALID_TRUNCATED,
    SCORE_MAX,
    SCORE_MIN,
    LlmResponse,
)

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def strip_code_fences(text: str) -> str:
    out = _FENCE.sub("", text.strip())
    return out.strip()


def _extract_object(text: str) -> str:
    """Return the first top-level {...} block if the answer carries extra text around it."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start : end + 1]


def truncate_tokens(text: str, cap: int) -> str:
    """Trim `text` to roughly `cap` tokens using the len/4 estimate (whole words)."""
    if approx_tokens(text) <= cap:
        return text
    words = text.split()
    out: list[str] = []
    for w in words:
        candidate = " ".join(out + [w])
        if approx_tokens(candidate) > cap:
            break
        out.append(w)
    return " ".join(out)


def parse_response(
    raw: str, rationale_token_cap: int = 150, stop_reason: str | None = None
) -> LlmResponse:
    text = strip_code_fences(raw or "")
    truncated = stop_reason in ("max_tokens", "length")
    flag = INVALID_TRUNCATED if truncated else INVALID_OUTPUT

    def invalid(error: str) -> LlmResponse:
        return LlmResponse(
            score=0,
            confidence=0.0,
            rationale="",
            valid=False,
            raw=raw,
            error=error,
            fallback_flag=flag,
        )

    try:
        obj: Any = json.loads(_extract_object(text))
    except (json.JSONDecodeError, ValueError):
        return invalid("json_parse_error" if not truncated else "truncated_json")
    if not isinstance(obj, dict):
        return invalid("not_an_object")
    score = obj.get("score")
    conf = obj.get("confidence")
    rationale = obj.get("rationale", "")
    # Score must be an integer (bools and floats like 2.0 are rejected).
    if isinstance(score, bool) or not isinstance(score, int):
        return invalid("score_not_integer")
    if not SCORE_MIN <= score <= SCORE_MAX:
        return invalid("score_out_of_range")
    if isinstance(conf, bool) or not isinstance(conf, int | float):
        return invalid("confidence_not_number")
    if not 0.0 <= float(conf) <= 1.0:
        return invalid("confidence_out_of_range")
    if not isinstance(rationale, str):
        return invalid("rationale_not_string")
    return LlmResponse(
        score=int(score),
        confidence=float(conf),
        rationale=truncate_tokens(rationale.strip(), rationale_token_cap),
        valid=True,
        raw=raw,
        error=None,
        fallback_flag=None,
    )
