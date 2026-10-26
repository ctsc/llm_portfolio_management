"""Fail if the repository contains an em dash or the banned word for experimental conditions.

Scans tracked text files (code, docs, config, prompts). `devin.md` states the rules themselves
and is excluded; so is this script, whose patterns would otherwise match.
Usage: python scripts/check_banned_words.py [paths...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".cfg", ".ini", ".json", ".csv"}
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
SKIP_FILES = {"devin.md", Path(__file__).name}

EM_DASH = "\u2014"
# The banned condition word, built from parts so this file does not contain it literally.
BANNED_WORD = re.compile(r"\b" + "a" + "rm" + r"s?\b", re.IGNORECASE)


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
            if p.is_file() and p.suffix in TEXT_SUFFIXES and p.name not in SKIP_FILES:
                out.append(p)
    return out


def check(paths: list[Path]) -> list[str]:
    problems: list[str] = []
    for p in iter_files(paths):
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if EM_DASH in line:
                problems.append(f"{_rel(p)}:{i}: em dash")
            if BANNED_WORD.search(line):
                problems.append(f"{_rel(p)}:{i}: banned condition word")
    return problems


def main(argv: list[str] | None = None) -> int:
    args = [Path(a) for a in (argv if argv is not None else sys.argv[1:])] or [ROOT]
    problems = check(args)
    for p in problems:
        print(p)
    print("banned word check ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
