"""Parse and validate a range response into an LlmResponse."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from portstudy.llm.prompts import approx_tokens
from portstudy.llm.schema import (
    INVALID_OUTPUT,
    INVALID_TRUNCATED,
    RANGE_MAX,
    RANGE_MIN,
    LlmResponse,
    fallback_response,
)

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_THINK_OPEN = re.compile(r"^\s*<think>", re.IGNORECASE)
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
_LOG = logging.getLogger("portstudy.llm.validate")


def strip_code_fences(text: str) -> str:
    out = _FENCE.sub("", text.strip())
    return out.strip()


def _object_at(text: str, start: int) -> tuple[str | None, int | None]:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1], index + 1
    return None, None


def _extract_first_object(text: str) -> tuple[str | None, bool]:
    """Return the first brace-delimited object and whether a later JSON object exists."""
    start = text.find("{")
    if start < 0:
        return None, False
    first, end = _object_at(text, start)
    if first is None or end is None:
        return None, False
    for next_start in (i for i in range(end, len(text)) if text[i] == "{"):
        candidate, _ = _object_at(text, next_start)
        if candidate is None:
            continue
        try:
            if isinstance(json.loads(candidate), dict):
                return first, True
        except (json.JSONDecodeError, ValueError):
            continue
    return first, False


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
    raw: str,
    rationale_token_cap: int = 150,
    stop_reason: str | None = None,
    reasoning_text: str | None = None,
) -> LlmResponse:
    truncated = stop_reason in ("max_tokens", "length")
    flag = INVALID_TRUNCATED if truncated else INVALID_OUTPUT
    text = strip_code_fences(raw or "")
    had_thinking = bool(reasoning_text and reasoning_text.strip())

    if _THINK_OPEN.match(text):
        had_thinking = True
        if not _THINK_BLOCK.match(text):
            error = INVALID_TRUNCATED if truncated else "unclosed_thinking"
            response = fallback_response(raw, error, flag)
            response.had_thinking = True
            return response
        text = _THINK_BLOCK.sub("", text, count=1)

    def invalid(error: str) -> LlmResponse:
        if error == "not_json" and truncated:
            error = INVALID_TRUNCATED
        response = fallback_response(raw, error, flag)
        response.had_thinking = had_thinking
        return response

    if not text.strip():
        return invalid(INVALID_TRUNCATED if truncated else "empty")
    object_text, has_extra_object = _extract_first_object(text)
    if object_text is None:
        return invalid(INVALID_TRUNCATED if truncated else "not_json")
    try:
        obj: Any = json.loads(object_text)
    except (json.JSONDecodeError, ValueError):
        return invalid(INVALID_TRUNCATED if truncated else "not_json")
    if not isinstance(obj, dict):
        return invalid("not_json")
    required = {"low", "likely", "high", "rationale"}
    missing = sorted(required - obj.keys())
    if missing:
        return invalid("missing_key")
    extra_keys = sorted(obj.keys() - required)
    if extra_keys:
        _LOG.warning("extra_keys", extra={"extra_fields": {"keys": extra_keys}})
    if has_extra_object:
        _LOG.warning("multiple_json_objects")

    values: dict[str, int] = {}
    for key in ("low", "likely", "high"):
        value = obj[key]
        # Review fix L2: the schema says int; a float such as 3.0 is not accepted.
        if isinstance(value, bool) or not isinstance(value, int):
            return invalid("not_integer")
        values[key] = value
        if not RANGE_MIN <= values[key] <= RANGE_MAX:
            return invalid("out_of_range")
    if values["low"] > values["likely"] or values["likely"] > values["high"]:
        return invalid("bad_order")
    rationale = obj["rationale"]
    if not isinstance(rationale, str):
        return invalid("rationale_not_string")
    if not rationale.strip():
        _LOG.info("empty_rationale")
    return LlmResponse(
        low=values["low"],
        likely=values["likely"],
        high=values["high"],
        rationale=truncate_tokens(rationale.strip(), rationale_token_cap),
        valid=True,
        raw=raw,
        error=None,
        fallback_flag=None,
        extra_keys=extra_keys,
        had_thinking=had_thinking,
    )
