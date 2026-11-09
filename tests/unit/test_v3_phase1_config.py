"""v3 phase 1: PLAN3 config values, the shuffle-strategy validator and the wording check."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from portstudy.config import StudyConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_banned_words  # noqa: E402


def _study(path: str = "config/study.yaml") -> dict:
    with (REPO_ROOT / path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_study_config_holds_the_plan3_values() -> None:
    raw = _study()
    cfg = StudyConfig.model_validate(raw)
    assert cfg.llm.gateway.base_url == "https://litellm.rs.gsu.edu/v1"
    assert "minimax-m2.7" not in cfg.llm.settings  # the unsourced 2048 override is gone
    assert cfg.eval.shuffle.strategies == ["mvo", "robust", "topk"]
    ids = {d.id: d for d in cfg.decisions}
    assert {"G6", "W1", "M1"} <= set(ids)
    assert "Mean-CVaR and Resampled MVO are excluded" in str(ids["G6"].value)
    assert "15 hours" in str(ids["W1"].value) and "mem-per-cpu=3G" in str(ids["M1"].value)
    test_cfg = StudyConfig.model_validate(_study("config/study_test.yaml"))
    assert test_cfg.eval.shuffle.strategies == ["mvo", "robust", "topk"]
    assert test_cfg.eval.shuffle.n_shuffles == 20


def test_shuffle_strategies_must_be_enabled() -> None:
    raw = _study()
    raw["strategies"]["enabled"] = ["mvo", "topk"]
    raw["eval"]["shuffle"]["strategies"] = ["mvo", "robust"]
    with pytest.raises(ValueError, match="eval.shuffle.strategies not in strategies.enabled"):
        StudyConfig.model_validate(raw)
    raw["eval"]["shuffle"]["strategies"] = ["mvo"]
    assert StudyConfig.model_validate(raw).eval.shuffle.strategies == ["mvo"]


def test_wording_check_rejects_the_researcher(tmp_path: Path) -> None:
    bad = tmp_path / "doc.md"
    bad.write_text("Set by the " + "researcher before the freeze.\n", encoding="utf-8")
    problems = check_banned_words.check([bad])
    assert problems and "names the " + "researcher" in problems[0]
    good = tmp_path / "ok.md"
    good.write_text("Set by Carter before the freeze.\n", encoding="utf-8")
    assert check_banned_words.check([good]) == []
    assert check_banned_words.check([REPO_ROOT]) == []
