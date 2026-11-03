"""Fail if anything that looks like an API key or private key is committed.

Usage: python scripts/check_secrets.py [paths...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".venv",
    "data",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "vendor",
    "htmlcov",
    "portstudy.egg-info",
}
SKIP_FILES = {Path(__file__).name}

PATTERNS = {
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{10,}"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bghp_[A-Za-z0-9]{30,}"),
    "gitlab_token": re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA|EC|OPENSSH|DSA|PGP) PRIVATE KEY"),
    "assigned_key": re.compile(
        r"(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"][A-Za-z0-9/+_-]{16,}['\"]",
        re.IGNORECASE,
    ),
    "arctic_litellm_key": re.compile(
        r"\bARCTIC_LITELLM_API_KEY\s*[:=]\s*['\"]?[A-Za-z0-9/+_-]{16,}"
    ),
    "arctic_fallback_key": re.compile(
        r"\bARCTIC_FALLBACK_API_KEY\s*[:=]\s*['\"]?[A-Za-z0-9/+_-]{16,}"
    ),
}


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def iter_files(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for base in paths:
        if base.is_file():
            out.append(base)
            continue
        for p in base.rglob("*"):
            if any(part in SKIP_DIRS for part in p.relative_to(base).parts):
                continue
            if (
                p.is_file()
                and p.name not in SKIP_FILES
                and p.suffix not in {".png", ".pyc", ".zip"}
            ):
                out.append(p)
    return out


def check(paths: list[Path]) -> list[str]:
    problems: list[str] = []
    for p in iter_files(paths):
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for name, pat in PATTERNS.items():
                if pat.search(line):
                    problems.append(f"{_rel(p)}:{i}: {name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    args = [Path(a) for a in (argv if argv is not None else sys.argv[1:])] or [ROOT]
    problems = check(args)
    for p in problems:
        print(p)
    print("secret check ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
