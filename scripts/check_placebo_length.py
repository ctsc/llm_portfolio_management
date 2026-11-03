"""Check skills and placebo lengths for every enabled model tokenizer.

Each enabled skills file must be within `llm.skills_length_tolerance` of the mean enabled
skills length. Placebo files must be within `llm.placebo_length_tolerance` of their match.
Every enabled skills and placebo file must be under the C4 token cap.
"""

from __future__ import annotations

import sys
from statistics import mean

from portstudy.config import load_model_registry, load_study_config
from portstudy.llm.prompts import PromptManifest, load_manifest, read_prompt_file, token_count


def check_lengths(
    skills_tolerance: float,
    placebo_tolerance: float,
    cap: int,
    model_ids: list[str],
    manifest: PromptManifest,
) -> list[str]:
    problems: list[str] = []
    skills = manifest.by_kind("skills")
    placebos = manifest.by_kind("placebo")
    skill_files = [c for c in skills if c.file]
    placebo_files = [c for c in placebos if c.file]
    for config in skills:
        if not config.file:
            problems.append(f"skills {config.id} has no prompt file")
    for config in placebos:
        if not config.file:
            problems.append(f"placebo {config.id} has no prompt file")
    if not skill_files:
        problems.append("no enabled skills configurations")

    for model_id in model_ids:
        lengths: dict[str, int] = {}
        for config in skill_files + placebo_files:
            n = token_count(read_prompt_file(str(config.file)), model_id)
            lengths[config.id] = n
            if n > cap:
                problems.append(
                    f"[{model_id}] {config.kind} {config.id} has {n} tokens, cap is {cap}"
                )
        if not skill_files:
            continue

        target_mean = mean(lengths[config.id] for config in skill_files)
        for config in skill_files:
            n = lengths[config.id]
            gap = abs(n - target_mean) / max(1, target_mean)
            if gap > skills_tolerance:
                problems.append(
                    f"[{model_id}] skills {config.id} is {n} tokens vs mean skills length "
                    f"{target_mean:.1f} ({gap:.1%} apart, tolerance {skills_tolerance:.0%})"
                )

        skill_lengths = {config.id: lengths[config.id] for config in skill_files}
        for config in placebo_files:
            target = config.matches or next(iter(skill_lengths), None)
            if target == "skills_mean":
                target_length = target_mean
                target_name = "mean skills length"
            elif target in skill_lengths:
                target_length = skill_lengths[target]
                target_name = f"skills {target}"
            else:
                problems.append(
                    f"placebo {config.id} matches unknown skills configuration {target}"
                )
                continue
            n = lengths[config.id]
            gap = abs(n - target_length) / max(1, target_length)
            if gap > placebo_tolerance:
                problems.append(
                    f"[{model_id}] placebo {config.id} is {n} tokens vs {target_name} "
                    f"{target_length:.1f} ({gap:.1%} apart, tolerance {placebo_tolerance:.0%})"
                )
    return problems


def main() -> int:
    cfg = load_study_config()
    registry = load_model_registry()
    model_ids = [m.id for m in registry.enabled()]
    problems = check_lengths(
        cfg.llm.skills_length_tolerance,
        cfg.llm.placebo_length_tolerance,
        cfg.llm.skills_token_cap,
        model_ids,
        load_manifest(),
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
