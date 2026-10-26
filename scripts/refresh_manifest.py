"""Recompute prompt file hashes in config/prompts_manifest.yaml.

Usage:
  python scripts/refresh_manifest.py --check   fail (exit 1) if any file changed without a
                                               version bump, or any hash is stale
  python scripts/refresh_manifest.py --write   write the current hashes into the manifest
                                               (only allowed when the version was bumped)

A file "changed without a version bump" when its hash differs from the manifest but the
manifest's `version` for that configuration is unchanged. The researcher bumps `version`
in the manifest (and the file name) before running `--write`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from portstudy.llm.prompts import file_hash, load_manifest, manifest_path


def check_manifest(path: Path) -> list[str]:
    """Return a list of problems; empty means the manifest matches the files on disk."""
    man = load_manifest(path)
    problems: list[str] = []
    base_hash = file_hash(man.base_instructions)
    if man.base_sha256 is None:
        problems.append("base_instructions has no sha256 recorded")
    elif man.base_sha256 != base_hash:
        problems.append("base_instructions changed; bump every configuration version and rewrite")
    for c in man.configurations:
        if c.file is None:
            continue
        current = file_hash(c.file)
        if c.sha256 is None:
            problems.append(f"{c.id}: no sha256 recorded (run --write)")
        elif c.sha256 != current:
            if c.hashed_version == c.version:
                problems.append(f"{c.id}: {c.file} changed but version is still {c.version}")
            else:
                problems.append(
                    f"{c.id}: version bumped to {c.version}, hash is stale (run --write)"
                )
        elif c.hashed_version != c.version:
            problems.append(f"{c.id}: hashed_version {c.hashed_version} != version {c.version}")
        if c.version not in c.file:
            problems.append(f"{c.id}: file name {c.file} does not carry version {c.version}")
    return problems


def write_manifest(path: Path) -> None:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["base_sha256"] = file_hash(raw.get("base_instructions", "prompts/base_instructions.md"))
    for c in raw["configurations"]:
        if c.get("file"):
            c["sha256"] = file_hash(c["file"])
            c["hashed_version"] = c.get("version", "v0")
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False, allow_unicode=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="with --write: accept a changed file without a version bump "
        "(first authoring of a file only)",
    )
    args = parser.parse_args(argv)
    path = args.manifest or manifest_path()
    if args.write:
        # Refuse to overwrite a hash when the version did not change.
        stale = [p for p in check_manifest(path) if "changed but version" in p]
        if stale and not args.force:
            print("refusing to write: bump the version first")
            for p in stale:
                print("  " + p)
            return 1
        write_manifest(path)
        print(f"manifest written: {path}")
        return 0
    problems = check_manifest(path)
    if problems:
        print("manifest check failed:")
        for p in problems:
            print("  " + p)
        return 1
    print("manifest ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
