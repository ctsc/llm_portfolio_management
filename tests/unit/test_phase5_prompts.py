"""Phase 5 tests: prompt assembly order, shared prefix, no-news sentence, manifest check."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import yaml

from portstudy.config import StudyConfig
from portstudy.data.snapshot import read_snapshot
from portstudy.llm.prompts import (
    assemble_prompt,
    clear_token_counters,
    load_manifest,
    register_token_counter,
    token_count,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_placebo_length  # noqa: E402
import refresh_manifest  # noqa: E402

ROW_A = {"ticker": "AAA", "name": "Alpha Co", "sector": "Energy"}
ROW_B = {"ticker": "BBB", "name": "Beta Inc", "sector": "Utilities"}


def test_shared_prefix_identical_across_stocks_and_order(study_config: StudyConfig) -> None:
    man = load_manifest()
    for cid in ("bare", "skills_1", "placebo"):
        a = assemble_prompt(cid, ROW_A, "Alpha beats estimates.", study_config, man)
        b = assemble_prompt(cid, ROW_B, "Beta misses estimates.", study_config, man)
        assert a.shared_prefix == b.shared_prefix
        assert a.prefix_hash == b.prefix_hash
        assert a.prompt_hash != b.prompt_hash
        # Order: prefix first, then the stock part; every C1 field appears in the stock part.
        assert a.full_text.startswith(a.shared_prefix)
        assert a.full_text.endswith(a.stock_part)
        assert a.full_text.index("Ticker: AAA") > len(a.shared_prefix)
        for field in (
            "Ticker: AAA",
            "Company: Alpha Co",
            "Sector: Energy",
            "Alpha beats estimates.",
        ):
            assert field in a.stock_part
    bare = assemble_prompt("bare", ROW_A, "x", study_config, man)
    skills = assemble_prompt("skills_1", ROW_A, "x", study_config, man)
    assert bare.prefix_hash != skills.prefix_hash
    assert bare.prompt_file_sha256 is None and skills.prompt_file_sha256 is not None
    assert bare.shared_prefix in skills.shared_prefix


def test_no_news_sentence_used(study_config: StudyConfig) -> None:
    p = assemble_prompt("bare", ROW_A, "", study_config)
    assert study_config.llm.no_news_sentence in p.stock_part
    p2 = assemble_prompt("bare", ROW_A, "   \n", study_config)
    assert study_config.llm.no_news_sentence in p2.stock_part


def test_prefix_hash_changes_when_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, study_config: StudyConfig
) -> None:
    # Copy the repo prompts and manifest into a temp root and edit the skills file.
    root = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "prompts", root / "prompts")
    (root / "config").mkdir()
    shutil.copy(
        REPO_ROOT / "config" / "prompts_manifest.yaml", root / "config" / "prompts_manifest.yaml"
    )
    monkeypatch.setenv("PORTSTUDY_ROOT", str(root))
    before = assemble_prompt("skills_1", ROW_A, "x", study_config, load_manifest())
    skills_file = root / "prompts" / "skills" / "skills_1_PLACEHOLDER_v0.md"
    skills_file.write_text(
        skills_file.read_text(encoding="utf-8") + "\nextra line\n", encoding="utf-8"
    )
    after = assemble_prompt("skills_1", ROW_A, "x", study_config, load_manifest())
    assert before.prefix_hash != after.prefix_hash
    assert before.prompt_file_sha256 != after.prompt_file_sha256
    # The manifest check fails on the unversioned edit, and --write refuses it.
    problems = refresh_manifest.check_manifest(root / "config" / "prompts_manifest.yaml")
    assert any("changed but version is still v0" in p for p in problems)
    assert (
        refresh_manifest.main(
            ["--write", "--manifest", str(root / "config" / "prompts_manifest.yaml")]
        )
        == 1
    )
    # After a version bump (file renamed and manifest updated) the write succeeds.
    new_file = skills_file.with_name("skills_1_PLACEHOLDER_v1.md")
    skills_file.rename(new_file)
    raw = yaml.safe_load((root / "config" / "prompts_manifest.yaml").read_text(encoding="utf-8"))
    for c in raw["configurations"]:
        if c["id"] == "skills_1":
            c["file"] = "prompts/skills/skills_1_PLACEHOLDER_v1.md"
            c["version"] = "v1"
    (root / "config" / "prompts_manifest.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert (
        refresh_manifest.main(
            ["--write", "--manifest", str(root / "config" / "prompts_manifest.yaml")]
        )
        == 0
    )
    assert refresh_manifest.check_manifest(root / "config" / "prompts_manifest.yaml") == []


def test_repo_manifest_is_current() -> None:
    assert refresh_manifest.check_manifest(REPO_ROOT / "config" / "prompts_manifest.yaml") == []


def test_placebo_length_check_with_mock_tokenizers(study_config: StudyConfig) -> None:
    man = load_manifest()
    clear_token_counters()
    try:
        register_token_counter("mock-a", lambda t: len(t.split()))
        assert token_count("one two three", "mock-a") == 3
        assert token_count("abcdefgh", "unknown-model") == 2
        problems = check_placebo_length.check_lengths(
            0.05, 0.02, 6000, ["mock-a", "fallback-model"], man
        )
        assert problems == [], problems
        # A tiny cap makes the skills file fail for every tokenizer.
        problems = check_placebo_length.check_lengths(0.05, 0.02, 10, ["mock-a"], man)
        assert any("cap is 10" in p for p in problems)
    finally:
        clear_token_counters()


def test_placeholder_files_carry_banner() -> None:
    for rel in (
        "prompts/skills/skills_1_PLACEHOLDER_v0.md",
        "prompts/placebo/placebo_PLACEHOLDER_v0.md",
    ):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert text.startswith("PLACEHOLDER")


def test_prompts_assemble_for_every_stock_and_configuration(
    snapshot_data_dir: Path, study_config: StudyConfig, session_weeks: list[str]
) -> None:
    snap = read_snapshot(session_weeks[0])
    man = load_manifest()
    ref = snap.reference.set_index("ticker")
    news = snap.news_text.set_index("ticker")["news_text"]
    hashes = set()
    for cfg in man.enabled():
        for tk in snap.included_tickers:
            row = {"ticker": tk, "name": ref.loc[tk, "name"], "sector": ref.loc[tk, "sector"]}
            p = assemble_prompt(cfg.id, row, str(news.get(tk, "")), study_config, man)
            assert p.full_text.startswith(p.shared_prefix)
            # No price statistics anywhere in the prompt.
            assert "adj_close" not in p.full_text and "return" not in p.stock_part.lower()
            hashes.add(p.prompt_hash)
    assert len(hashes) == len(man.enabled()) * len(snap.included_tickers)
