"""Check placebo length matching and the skills token cap (DECISION(C4)).

For every enabled model's tokenizer (via `token_count`), every placebo file must be within
`llm.placebo_length_tolerance` (default 2 percent) of the skills file it matches, and every
skills file must be under `llm.skills_token_cap`. Exit code 1 on any failure.
"""

from __future__ import annotations

import sys

from portstudy.config import load_model_registry, load_study_config
from portstudy.llm.prompts import PromptManifest, load_manifest, read_prompt_file, token_count


def check_lengths(
    cfg_tolerance: float, cap: int, model_ids: list[str], manifest: PromptManifest
) -> list[str]:
    problems: list[str] = []
    skills = {c.id: c for c in manifest.enabled() if c.kind == "skills" and c.file}
    placebos = [c for c in manifest.enabled() if c.kind == "placebo" and c.file]
    for model_id in model_ids:
        for sid, s in skills.items():
            n = token_count(read_prompt_file(str(s.file)), model_id)
            if n > cap:
                problems.append(f"[{model_id}] skills {sid} has {n} tokens, cap is {cap}")
        for p in placebos:
            target = p.matches or next(iter(skills), None)
            if target is None or target not in skills:
                problems.append(f"placebo {p.id} matches unknown skills configuration {target}")
                continue
            n_p = token_count(read_prompt_file(str(p.file)), model_id)
            n_s = token_count(read_prompt_file(str(skills[target].file)), model_id)
            gap = abs(n_p - n_s) / max(1, n_s)
            if gap > cfg_tolerance:
                problems.append(
                    f"[{model_id}] placebo {p.id} is {n_p} tokens vs skills {target} {n_s} "
                    f"({gap:.1%} apart, tolerance {cfg_tolerance:.0%})"
                )
    return problems


def main() -> int:
    cfg = load_study_config()
    registry = load_model_registry()
    model_ids = [m.id for m in registry.enabled()]
    problems = check_lengths(
        cfg.llm.placebo_length_tolerance, cfg.llm.skills_token_cap, model_ids, load_manifest()
    )
    if problems:
        print("placebo length check failed:")
        for p in problems:
            print("  " + p)
        return 1
    print("placebo length check ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
