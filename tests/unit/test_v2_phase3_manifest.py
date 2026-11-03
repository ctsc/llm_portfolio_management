"""Phase 3 tests for kind-based prompts and length checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from portstudy.llm.prompts import (
    PromptConfiguration,
    PromptManifest,
    clear_token_counters,
    load_manifest,
    register_token_counter,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_placebo_length  # noqa: E402


def test_skills_placeholders_have_equal_length() -> None:
    files = sorted((REPO_ROOT / "prompts" / "skills").glob("skills_*_PLACEHOLDER_v0.md"))
    assert len(files) == 5
    assert len({path.stat().st_size for path in files}) == 1


def test_by_kind_order_and_disabled_skills_exclusion() -> None:
    manifest = load_manifest()
    assert [config.id for config in manifest.by_kind("skills")] == [
        "skills_1",
        "skills_2",
        "skills_3",
        "skills_4",
    ]


def _check_lengths(
    monkeypatch: pytest.MonkeyPatch,
    counts: dict[str, int],
    skill_ids: tuple[str, ...],
    matches: str | None = "skills_mean",
) -> list[str]:
    configurations = [
        PromptConfiguration(id=skill_id, kind="skills", file=f"{skill_id}.md")
        for skill_id in skill_ids
    ]
    configurations.append(
        PromptConfiguration(id="placebo", kind="placebo", file="placebo.md", matches=matches)
    )
    manifest = PromptManifest(configurations=configurations)
    monkeypatch.setattr(check_placebo_length, "read_prompt_file", lambda path: path)
    register_token_counter("phase3-test", lambda text: counts[text])
    try:
        return check_placebo_length.check_lengths(0.05, 0.02, 6000, ["phase3-test"], manifest)
    finally:
        clear_token_counters()


def test_skills_and_placebo_are_checked_against_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problems = _check_lengths(
        monkeypatch,
        {"skill-a.md": 100, "skill-b.md": 108, "placebo.md": 104},
        ("skill-a", "skill-b"),
    )
    assert problems == []


def test_skills_outside_mean_tolerance_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problems = _check_lengths(
        monkeypatch,
        {"skill-a.md": 100, "skill-b.md": 112, "placebo.md": 106},
        ("skill-a", "skill-b"),
    )
    assert any("skills skill-a" in problem for problem in problems)
    assert any("skills skill-b" in problem for problem in problems)


def test_single_enabled_skill_is_its_own_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problems = _check_lengths(monkeypatch, {"skill-a.md": 100, "placebo.md": 100}, ("skill-a",))
    assert problems == []


def test_zero_enabled_skills_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problems = _check_lengths(monkeypatch, {"placebo.md": 100}, ())
    assert any("no enabled skills configurations" in problem for problem in problems)


def test_unknown_placebo_match_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problems = _check_lengths(
        monkeypatch,
        {"skill-a.md": 100, "placebo.md": 100},
        ("skill-a",),
        matches="missing",
    )
    assert any("matches unknown skills configuration missing" in problem for problem in problems)
