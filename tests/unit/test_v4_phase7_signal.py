"""v4 phase 7: signal-level tests on planted panels and on the synthetic pipeline run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from portstudy.config import StudyConfig, load_preregistration
from portstudy.eval.signal_tests import fama_macbeth, pooled_regression, weekly_gap

CONTROLS = ["baseline_score", "momentum_12_1", "sector"]


def _panel(kind: str, n_weeks: int = 20, n: int = 100, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for w in range(n_weeks):
        style = rng.normal(0, 1, n)
        specific = rng.normal(0, 1, n)
        ret = 0.4 * style + rng.normal(0, 1, n)
        if kind == "specific":
            ret = ret + 0.4 * specific
            score = specific
        else:
            score = style + rng.normal(0, 0.5, n)  # a style tilt: the score is a control plus noise
        frames.append(
            pd.DataFrame(
                {
                    "week_id": f"2026-W{w + 1:02d}",
                    "ticker": [f"T{i}" for i in range(n)],
                    "ret": ret,
                    "score": score,
                    "baseline_score_z": rng.normal(0, 1, n),
                    "momentum_12_1_z": style,
                    "sector": rng.choice(["A", "B", "C"], n),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_residual_ic_keeps_a_specific_effect_and_removes_a_style_tilt() -> None:
    specific = fama_macbeth(_panel("specific"), "score", CONTROLS)
    assert specific.mean() > 0.2 and specific.mean() / specific.std(ddof=1) * np.sqrt(20) > 5
    style = _panel("style")
    raw = weekly_gap(style.assign(zero=0.0 * style["score"] + style.index % 7), "score", "zero")
    assert raw["ic_a"].mean() > 0.2  # the raw IC is clearly positive
    resid = fama_macbeth(style, "score", CONTROLS)
    t_stat = resid.mean() / resid.std(ddof=1) * np.sqrt(len(resid))
    assert abs(t_stat) < 2.5 and abs(resid.mean()) < 0.15  # near zero, not significant


def test_pooled_regression_recovers_a_planted_prompt_difference() -> None:
    rng = np.random.default_rng(3)
    rows = []
    for w in range(24):
        week_effect = rng.normal(0, 0.05)
        for model, m_eff in (("a", 0.01), ("b", -0.01)):
            for prompt in ("placebo", "skills_1", "skills_2"):
                ic = week_effect + m_eff + (0.03 if prompt != "placebo" else 0.0)
                rows.append(
                    {"week_id": w, "model": model, "prompt": prompt, "y": ic + rng.normal(0, 0.01)}
                )
    df = pd.DataFrame(rows).assign(skills=lambda d: (d["prompt"] != "placebo").astype(float))
    res = pooled_regression(df, "y ~ skills + C(model) + C(week_id)", "skills")
    assert abs(res["estimate"] - 0.03) < 0.006 and res["p"] < 1e-6 and res["n_weeks"] == 24


def test_signal_tests_on_the_synthetic_run(pipeline_dir: Path, study_config: StudyConfig) -> None:
    from portstudy.config import load_model_registry
    from portstudy.eval.data import load_eval_data
    from portstudy.eval.signal_tests import signal_tests
    from portstudy.llm.prompts import load_manifest
    from portstudy.pipeline.context import RunContext

    ctx = RunContext(
        cfg=study_config, registry=load_model_registry(), manifest=load_manifest(), synthetic=True
    )
    data = load_eval_data(study_config, ctx.portfolio_keys(), synthetic=True)
    prereg = load_preregistration(cfg=study_config)
    table = signal_tests(data, prereg)
    assert {"H1", "H2", "H3", "H3b"} <= set(table["hypothesis"])
    primary = table[table["family"] == "primary"]
    assert set(primary["comparison_id"]) == {c.id for c in prereg.primary_comparisons}
    assert set(table["family"]) <= {"primary", "secondary", "exploratory"}
    # One scored week in the shared run: p-values need at least two weeks and stay NaN here
    # (the Phase 11 synthetic e2e run checks the pooled tests over many weeks).
    assert table.loc[table["p_adj"].notna(), "n"].ge(2).all()
    # The stub's skills prompts are less noisy than the controls (synthetic world).
    ic = table[table["comparison"] == "LLM rank IC above zero"].set_index("unit")["estimate"]
    for model in ("stub-ow-a", "stub-ow-b"):
        assert ic[f"{model}__skills_1"] > ic[f"{model}__placebo"]
    h2 = table[table["comparison"].str.startswith("steered tilt minus Q0")]
    assert len(h2) and h2["mean_conf_only_uncertainty_ratio"].notna().any()
