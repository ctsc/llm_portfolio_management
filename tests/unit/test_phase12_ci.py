"""Phase 12 tests: the CI file validates and every command it calls exists locally."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_banned_words  # noqa: E402
import check_secrets  # noqa: E402


def _ci() -> dict[str, object]:
    with (REPO_ROOT / ".gitlab-ci.yml").open("r", encoding="utf-8") as fh:
        loaded: dict[str, object] = yaml.safe_load(fh)
    return loaded


def test_ci_yaml_passes_yamllint() -> None:
    cmd = [sys.executable, "-m", "yamllint", "-c", ".yamllint", ".gitlab-ci.yml", ".github/workflows/ci.yml",
           "config/study.yaml", "config/models.yaml", "config/prompts_manifest.yaml"]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ci_stages_and_jobs() -> None:
    ci = _ci()
    assert ci["stages"] == ["lint", "test", "checks"]
    assert ci["image"] == "python:3.11"
    jobs = {k: v for k, v in ci.items() if isinstance(v, dict) and "stage" in v}
    assert {"lint", "test", "e2e", "checks"} <= set(jobs)
    e2e = jobs["e2e"]
    assert isinstance(e2e, dict) and "rules" in e2e  # default branch only
    assert "cache" in ci


def test_every_ci_command_exists() -> None:
    ci = _ci()
    scripts_called: list[str] = []
    for name, job in ci.items():
        if not isinstance(job, dict) or "script" not in job:
            continue
        for line in job["script"]:
            parts = shlex.split(str(line))
            exe = parts[0]
            if exe == "python":
                assert parts[1] == "-m" or (REPO_ROOT / parts[1]).exists(), (name, line)
                if parts[1] != "-m":
                    scripts_called.append(parts[1])
            elif exe == "pip":
                continue
            else:
                assert shutil.which(exe) is not None or (REPO_ROOT / ".venv" / "Scripts" / f"{exe}.exe").exists(), exe
    assert {"scripts/refresh_manifest.py", "scripts/check_placebo_length.py", "scripts/check_banned_words.py",
            "scripts/check_secrets.py"} <= set(scripts_called)


def test_repo_passes_banned_word_and_secret_checks() -> None:
    assert check_banned_words.check([REPO_ROOT]) == []
    assert check_secrets.check([REPO_ROOT]) == []


def test_checks_catch_planted_problems(tmp_path: Path) -> None:
    bad = tmp_path / "bad.md"
    bad.write_text("an em dash \u2014 here, and the treatment " + "a" + "rm" + " name\n", encoding="utf-8")
    problems = check_banned_words.check([bad])
    assert any("em dash" in p for p in problems) and any("banned condition word" in p for p in problems)
    key = tmp_path / "leak.py"
    key.write_text('API_KEY = "' + "sk-ant-" + "a" * 24 + '"\n', encoding="utf-8")
    found = check_secrets.check([key])
    assert found and all("anthropic_key" in p or "assigned_key" in p or "openai_key" in p for p in found)


def _github_ci() -> dict[str, object]:
    with (REPO_ROOT / ".github" / "workflows" / "ci.yml").open("r", encoding="utf-8") as fh:
        loaded: dict[str, object] = yaml.safe_load(fh)
    return loaded


def test_github_workflow_mirrors_gitlab_ci() -> None:
    """GitHub is the primary host: its workflow runs the same jobs and commands as .gitlab-ci.yml."""
    gh = _github_ci()
    jobs = gh["jobs"]
    assert isinstance(jobs, dict)
    assert {"lint", "test", "integration", "e2e", "slow", "checks"} <= set(jobs)
    for name in ("integration", "e2e"):
        assert "default_branch" in str(jobs[name]["if"]), name
    assert jobs["slow"]["if"].startswith("github.event_name == 'workflow_dispatch'")

    def run_lines(job: dict[str, object]) -> list[str]:
        out: list[str] = []
        for step in job["steps"]:  # type: ignore[union-attr]
            if isinstance(step, dict) and "run" in step:
                out.append(" ".join(str(step["run"]).split()))
            if isinstance(step, dict) and step.get("uses", "").startswith("actions/setup-python"):
                assert step["with"]["python-version"] == "3.11"
        return out

    gl = _ci()
    for name in ("lint", "test", "integration", "e2e", "slow", "checks"):
        gl_lines = [" ".join(str(x).split()) for x in gl[name]["script"]]  # type: ignore[index]
        gh_lines = run_lines(jobs[name])
        assert all(line in gh_lines for line in gl_lines), (name, gl_lines, gh_lines)
    for name in ("test", "integration", "e2e"):
        checkout = next(s for s in jobs[name]["steps"] if str(s.get("uses", "")).startswith("actions/checkout"))
        assert checkout["with"]["fetch-depth"] == 0, name  # full history for the phase checklist test
