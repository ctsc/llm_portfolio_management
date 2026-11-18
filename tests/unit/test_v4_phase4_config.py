"""v4 phase 4: decisions, the evaluation config and the pre-registration file."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from portstudy.config import (
    Preregistration,
    load_model_registry,
    load_preregistration,
    load_study_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _base() -> dict[str, object]:
    data = yaml.safe_load((REPO_ROOT / "config" / "preregistration.yaml").read_text("utf-8"))
    assert isinstance(data, dict)
    return data


def test_study_and_test_configs_load_their_preregistration() -> None:
    for name in ("study.yaml", "study_test.yaml"):
        cfg = load_study_config(REPO_ROOT / "config" / name)
        prereg = load_preregistration(cfg=cfg)
        assert [c.id for c in prereg.primary_comparisons] == [
            "rq1_llm_vs_baseline",
            "rq2_tilt_gap",
            "rq3_skills_vs_bare",
            "rq5_skills_vs_placebo",
        ]
        assert not prereg.frozen and prereg.view_sensitivity_min_tracking_error is None
    test_cfg = load_study_config(REPO_ROOT / "config" / "study_test.yaml")
    assert test_cfg.eval.preregistration_file.endswith("preregistration_test.yaml")
    assert test_cfg.eval.tie_band == 0.001 and test_cfg.eval.signflip_exact_max_t == 24
    assert test_cfg.eval.multiple_testing.exploratory == ["H4"]


def test_old_holm_families_key_is_rejected() -> None:
    from portstudy.config import StudyConfig

    with pytest.raises(ValidationError):
        StudyConfig.model_validate({"eval": {"holm_families": {"H1": "all"}}})


def test_unknown_control_and_too_many_entries_raise() -> None:
    data = _base()
    data["residual_ic_controls"] = ["baseline_score", "dividend_yield"]
    with pytest.raises(ValidationError, match="unknown residual_ic_controls"):
        Preregistration.model_validate(data)
    data = _base()
    entry = {"hypothesis": "H1", "kind": "pooled_source_ic_gap"}
    data["primary_comparisons"] = [{"id": f"c{i}", **entry} for i in range(11)]
    with pytest.raises(ValidationError, match="at most 10"):
        Preregistration.model_validate(data)


def test_frozen_needs_a_date_and_the_g13_threshold() -> None:
    data = {**_base(), "frozen": True}
    with pytest.raises(ValidationError, match="frozen_on"):
        Preregistration.model_validate(data)
    data["frozen_on"] = "2026-11-20"
    with pytest.raises(ValidationError, match="view_sensitivity_min_tracking_error"):
        Preregistration.model_validate(data)
    data["view_sensitivity_min_tracking_error"] = 0.0005
    assert Preregistration.model_validate(data).frozen


def test_selectors_must_name_enabled_models_prompts_and_strategies() -> None:
    cfg = load_study_config(REPO_ROOT / "config" / "study_test.yaml")
    data = _base()
    data["primary_comparisons"] = [
        {
            "id": "x",
            "hypothesis": "H1",
            "kind": "pooled_source_ic_gap",
            "models": ["no-such-model"],
        },
        {
            "id": "y",
            "hypothesis": "H3",
            "kind": "pooled_prompt_ic_gap",
            "prompts": ["skills_9"],
            "control_prompt": "bare",
        },
        {"id": "z", "hypothesis": "H2", "kind": "pooled_tilt_ic_gap", "strategies": ["nope"]},
    ]
    problems = Preregistration.model_validate(data).check_against(
        cfg, load_model_registry(), list(cfg.llm.prompt_ids)
    )
    text = " ".join(problems)
    assert "no-such-model" in text and "skills_9" in text and "nope" in text


def test_prompt_comparison_needs_a_control_prompt() -> None:
    data = _base()
    data["primary_comparisons"] = [
        {"id": "x", "hypothesis": "H3", "kind": "pooled_prompt_ic_gap", "prompts": ["skills_1"]}
    ]
    with pytest.raises(ValidationError, match="control_prompt"):
        Preregistration.model_validate(data)


def test_new_decisions_are_recorded() -> None:
    cfg = load_study_config(REPO_ROOT / "config" / "study.yaml")
    ids = {d.id for d in cfg.decisions}
    expected = {
        "L1",
        "K2",
        "K3",
        "E7",
        "S8",
        "G1b",
        "N4",
        "N5",
        "N6",
        "N7",
        "C8",
        "A7",
        "A8",
        "A9",
        "U1",
        "U2",
        "D5",
        "G22",
        *(f"G{i}" for i in range(8, 22)),
    }
    assert expected <= ids, sorted(expected - ids)
    assert cfg.decision("B4").value == 100
    assert not {"G6", "W1", "M1"} & ids
