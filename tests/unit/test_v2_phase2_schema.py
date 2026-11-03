from __future__ import annotations

import itertools
import json
import logging

import pytest

from portstudy.llm.schema import RANGE_MAX, RANGE_MIN, fallback_response
from portstudy.llm.validate import parse_response


def _raw(
    low: object = -1,
    likely: object = 0,
    high: object = 1,
    rationale: object = "outlook",
    **extra: object,
) -> str:
    return json.dumps({"low": low, "likely": likely, "high": high, "rationale": rationale, **extra})


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        (_raw(low=True), "not_integer"),
        (_raw(likely=2.5), "not_integer"),
        (_raw(high="3"), "not_integer"),
        (_raw(low=-6), "out_of_range"),
        (_raw(high=6), "out_of_range"),
        (_raw(low=1, likely=0, high=2), "bad_order"),
        ('{"low": 0, "likely": 0, "high": 0}', "missing_key"),
        ("not json", "not_json"),
        ('{"low": 0, "likely": 0, "high": 0, "rationale": 4}', "rationale_not_string"),
        ("", "empty"),
        ("<think>analysis</think>", "empty"),
        ("", "invalid_truncated"),
        ("<think>some analysis", "unclosed_thinking"),
        ("<think>some analysis", "invalid_truncated"),
        ('{"low": 0,', "invalid_truncated"),
    ],
)
def test_invalid_responses_have_specific_error_codes(raw: str, error: str) -> None:
    truncated = error == "invalid_truncated"
    response = parse_response(raw, stop_reason="max_tokens" if truncated else None)
    assert not response.valid
    assert response.error == error


@pytest.mark.parametrize(
    "raw",
    [
        _raw(low=-5, likely=0, high=5),
        _raw(low=3.0, likely=3.0, high=4.0),
        '{"low": -0, "likely": 0, "high": 0, "rationale": "outlook"}',
        "```json\n" + _raw() + "\n```",
        "Rating: " + _raw() + " thanks.",
    ],
)
def test_valid_responses_parse_expected_range(raw: str) -> None:
    response = parse_response(raw)
    assert response.valid
    assert response.error is None
    assert RANGE_MIN <= response.low <= response.likely <= response.high <= RANGE_MAX


def test_fallback_response_uses_zero_range() -> None:
    response = fallback_response("bad", "not_json")
    assert (response.low, response.likely, response.high) == (0, 0, 0)
    assert not response.valid


def test_json_like_text_inside_rationale_is_not_a_second_object() -> None:
    raw = _raw(rationale='Literal text: {"low": -5, "likely": -5, "high": -5}')
    response = parse_response(raw)
    assert response.valid and response.rationale.startswith("Literal text")


def test_first_json_object_wins_and_additional_object_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    first = _raw(low=-2, likely=-1, high=0)
    second = _raw(low=3, likely=4, high=5)
    with caplog.at_level(logging.WARNING, logger="portstudy.llm.validate"):
        response = parse_response(f"{first}\n{second}")
    assert response.valid and response.likely == -1
    assert "multiple_json_objects" in caplog.text


def test_thinking_block_is_removed_before_parsing() -> None:
    hidden = _raw(low=4, likely=4, high=4)
    answer = _raw(low=-1, likely=0, high=1)
    response = parse_response(f"<think>{hidden}</think>\n{answer}")
    assert response.valid and response.likely == 0
    assert response.had_thinking


def test_separate_reasoning_text_is_recorded_but_not_parsed() -> None:
    response = parse_response(_raw(), reasoning_text='{"low": 5, "likely": 5, "high": 5}')
    assert response.valid and response.likely == 0 and response.had_thinking


def test_extra_keys_are_ignored_returned_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    raw = _raw(unrequested="value")
    with caplog.at_level(logging.WARNING, logger="portstudy.llm.validate"):
        response = parse_response(raw)
    assert response.valid and response.extra_keys == ["unrequested"]
    assert "extra_keys" in caplog.text


def test_empty_rationale_is_valid_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="portstudy.llm.validate"):
        response = parse_response(_raw(rationale=""))
    assert response.valid and response.rationale == ""
    assert "empty_rationale" in caplog.text


def test_rationale_is_truncated_to_token_cap() -> None:
    response = parse_response(_raw(rationale=" ".join(["word"] * 100)), rationale_token_cap=5)
    assert response.valid and len(response.rationale.split()) <= 20


def test_every_in_range_triple_is_valid_exactly_when_ordered() -> None:
    for low, likely, high in itertools.product(range(RANGE_MIN, RANGE_MAX + 1), repeat=3):
        response = parse_response(_raw(low=low, likely=likely, high=high))
        if low <= likely <= high:
            assert response.valid
        else:
            assert not response.valid and response.error == "bad_order"
