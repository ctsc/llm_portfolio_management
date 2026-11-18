"""Live-readiness preflight (PLAN4 Phase 3C). Thin CLI over portstudy.pipeline.preflight.

Usage:
    python scripts/check_live_ready.py [--config PATH] [--skip-network]
                                       [--accept-config-change NOTE]

Prints every failure, every warning and the marker counts (TBD / unverified / unpinned).
Exit 1 when any failure exists, else 0. `--accept-config-change NOTE` records an intended
config change and clears the frozen-config diff before the checks run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

from portstudy.config import load_model_registry, load_study_config, repo_root
from portstudy.llm.prompts import load_manifest
from portstudy.pipeline.frozen import accept_config_change
from portstudy.pipeline.preflight import (
    NEEDS_VERIFICATION_TAG,
    live_ready,
    unverified_config_lines,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="study YAML path")
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="omit the gateway probe (gateway reachability is not checked)",
    )
    parser.add_argument(
        "--accept-config-change",
        metavar="NOTE",
        default=None,
        help="accept the current config as the new frozen baseline; NOTE is required",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    if args.accept_config_change is not None:
        try:
            accept_config_change(root, args.accept_config_change)
        except ValueError as exc:
            print(f"error: {exc}")
            return 1
        print(f"config change accepted: {args.accept_config_change}")

    cfg = load_study_config(Path(args.config) if args.config else None)
    registry = load_model_registry()
    manifest = load_manifest()

    skip = args.skip_network
    if skip:
        print("gateway probe skipped (--skip-network)")
    report = live_ready(
        cfg,
        registry,
        manifest,
        root=root,
        env=os.environ,
        now=dt.datetime.now(),
        gateway_probe=(lambda _url: None) if skip else None,
        clock_skew_s=(lambda _url: None) if skip else None,
    )

    for line in report.failures:
        print(f"FAIL {line}")
    for line in report.warnings:
        print(f"WARN {line}")
    print(
        "markers: " + " ".join(f"{name}={count}" for name, count in sorted(report.markers.items()))
    )
    for line in unverified_config_lines(root):
        print(f"{NEEDS_VERIFICATION_TAG} config line: {line}")
    if report.failures:
        print(f"live readiness: {len(report.failures)} failure(s)")
        return 1
    print("live readiness: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
