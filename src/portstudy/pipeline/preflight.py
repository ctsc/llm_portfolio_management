"""Live-readiness preflight (PLAN4 Phase 3C).

`live_ready` collects every reason a live weekend run must not start. All checks are offline
except the injected gateway probe; `run_weekend` calls it for non-synthetic runs and
`scripts/check_live_ready.py` is the thin CLI. Phase 3B calibration values and Phase 4
preregistration plug in through `EXTRA_CHECKS`.
"""

from __future__ import annotations

import datetime as dt
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from portstudy.config import ModelRegistry, StudyConfig
from portstudy.llm.prompts import PromptManifest
from portstudy.paths import data_dir, outputs_dir, snapshots_dir
from portstudy.pipeline.frozen import frozen_config_diff

WEEK_FOLDER = re.compile(r"^(\d{4}-W\d{2})(\.v\d+)?$")
MARKER_DIRS = ("src", "scripts", "config")
# Built from pieces so the PROGRESS.md marker scanner (test_phase16_checklist) does not flag
# this file, which only counts the markers.
NEEDS_VERIFICATION_TAG = "UN" + "VERIFIED"
MARKERS = ("TBD", NEEDS_VERIFICATION_TAG, "UNPINNED")

# Phase 3B (calibration values) and Phase 4 (preregistration) append checks here; each returns
# failure lines, which are collected like every other failure.
EXTRA_CHECKS: list[Callable[..., list[str]]] = []


@dataclass
class PreflightReport:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    markers: dict[str, int] = field(default_factory=dict)


def _gateway_probe(base_url: str) -> str | None:
    """GET <base>/health/liveliness with a 10 s timeout; None when it answers, else the error."""
    import httpx

    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    try:
        response = httpx.get(f"{url}/health/liveliness", timeout=10.0)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    if response.status_code >= 400:
        return f"HTTP {response.status_code}"
    return None


def _disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def _gpu_free_gb() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return 0.0
    free, _total = torch.cuda.mem_get_info()
    return float(free) / 1e9


def _marker_counts(root: Path) -> dict[str, int]:
    counts = {marker: 0 for marker in MARKERS}
    for folder in MARKER_DIRS:
        base = root / folder
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for marker in MARKERS:
                counts[marker] += text.count(marker)
    return counts


def unverified_config_lines(root: Path) -> list[str]:
    """Config file lines still carrying the unverified marker (for example `json_mode`)."""
    out: list[str] = []
    config_dir = root / "config"
    if not config_dir.exists():
        return out
    for path in sorted(config_dir.iterdir()):
        if not path.is_file():
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
        ):
            if NEEDS_VERIFICATION_TAG in line:
                out.append(f"{path.name}:{lineno}: {line.strip()}")
    return out


def _kronos_head(root: Path) -> str | None:
    """HEAD of vendor/kronos, or None when git cannot answer (reported separately)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root / "vendor" / "kronos"), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def live_ready(
    cfg: StudyConfig,
    registry: ModelRegistry,
    manifest: PromptManifest,
    *,
    root: Path,
    env: Mapping[str, str],
    now: dt.datetime,
    week_id: str | None = None,
    gateway_probe: Callable[[str], str | None] | None = None,
    disk_free_gb: Callable[[Path], float] | None = None,
    gpu_free_gb: Callable[[], float | None] | None = None,
) -> PreflightReport:
    """Collect every reason a live run must not start; never stops at the first failure."""
    probe = gateway_probe or _gateway_probe
    disk = disk_free_gb or _disk_free_gb
    gpu = gpu_free_gb or _gpu_free_gb
    report = PreflightReport()
    failures = report.failures
    warnings = report.warnings

    # Prompts and pinned inputs.
    for conf in manifest.enabled():
        if conf.file and "PLACEHOLDER" in Path(conf.file).name:
            failures.append(f"prompt configuration {conf.id} uses placeholder file {conf.file}")
    if cfg.baseline.kronos.repo_commit == "UNPINNED":
        failures.append("baseline.kronos.repo_commit is UNPINNED")
    if cfg.baseline.forecaster == "stub":
        failures.append("baseline.forecaster is stub")

    # Models.
    enabled = registry.enabled()
    for spec in enabled:
        if spec.provider_model == "TBD":
            failures.append(f"model {spec.id} has provider_model TBD")
        if spec.kind == "api":
            if not spec.price.confirmed:
                failures.append(
                    f"api model {spec.id} prices are placeholders (price.confirmed false, D5)"
                )
            if spec.snapshot_id is None:
                failures.append(f"api model {spec.id} has snapshot_id null (D5)")

    # Timing.
    if cfg.timing.live_end_date is None:
        failures.append("timing.live_end_date is null")
    if cfg.timing.pilot_week is None:
        failures.append("timing.pilot_week is null")

    # Environment keys for every enabled provider.
    providers = {spec.provider for spec in enabled}
    required: list[tuple[str, str]] = []
    if "litellm" in providers:
        required.append(("ARCTIC_LITELLM_API_KEY", "litellm gateway models"))
    if "anthropic_batch" in providers:
        required.append(("ANTHROPIC_API_KEY", "anthropic_batch models"))
    if "openai_batch" in providers:
        required.append(("OPENAI_API_KEY", "openai_batch models"))
    if cfg.data.price_provider == "alpaca" or cfg.data.news_provider == "alpaca":
        required.append(("ALPACA_API_KEY_ID", "alpaca price/news provider"))
        required.append(("ALPACA_API_SECRET_KEY", "alpaca price/news provider"))
    if cfg.data.earnings_provider == "finnhub":
        required.append(("FINNHUB_API_KEY", "finnhub earnings provider"))
    if cfg.data.rate_provider == "fred":
        required.append(("FRED_API_KEY", "fred rate provider"))
    for key, why in required:
        if not env.get(key):
            failures.append(f"environment key {key} missing ({why})")

    # Kronos checkout.
    vendor = root / "vendor" / "kronos"
    pinned = cfg.baseline.kronos.repo_commit
    if not vendor.exists():
        failures.append("vendor/kronos is missing")
    elif pinned != "UNPINNED":
        head = _kronos_head(root)
        if head is None:
            warnings.append(
                "vendor/kronos HEAD not checked (git rev-parse failed); pin comparison skipped"
            )
        elif not head.startswith(pinned):
            failures.append(
                f"vendor/kronos HEAD {head[:12]} does not start with pinned commit {pinned}"
            )

    # Disk and GPU.
    free_gb = disk(data_dir())
    if free_gb < 5.0:
        failures.append(f"less than 5 GB free at the data dir ({free_gb:.1f} GB)")
    gpu_free = gpu()
    needed = max(2.0, 4.5 * cfg.baseline.kronos.batch_size / 300)
    if gpu_free is None:
        warnings.append("GPU memory not checked (torch not installed)")
    elif gpu_free < needed:
        warnings.append(
            f"GPU free memory {gpu_free:.1f} GB below the {needed:.1f} GB "
            f"batch_size {cfg.baseline.kronos.batch_size} needs"
        )

    # Gateway reachability (skipped only via --skip-network, which passes a no-op probe).
    error = probe(cfg.llm.gateway.base_url)
    if error:
        failures.append(f"gateway URL does not answer; is the GSU VPN on? ({error})")

    # Universe file (C8).
    u_path = root / cfg.universe.universe_file
    if not u_path.exists():
        failures.append(f"universe file {u_path} is missing")
    else:
        try:
            uni = pd.read_csv(u_path, dtype=str, keep_default_na=False)
            missing = [
                str(row["ticker"])
                for _, row in uni.iterrows()
                if not str(row.get("name", "")).strip() or not str(row.get("sector", "")).strip()
            ]
        except Exception as exc:
            failures.append(f"universe file {u_path} cannot be read ({exc})")
        else:
            if missing:
                failures.append(
                    f"universe file has blank name or sector for {len(missing)} tickers "
                    f"({', '.join(missing[:5])})"
                )

    # A7: no week folders earlier than the pilot under the live data dir.
    pilot = cfg.timing.pilot_week
    if pilot:
        for base in (outputs_dir(), snapshots_dir()):
            if not base.exists():
                continue
            for entry in sorted(base.iterdir()):
                match = WEEK_FOLDER.match(entry.name)
                if match and match.group(1) < pilot:
                    failures.append(
                        f"week folder {entry.name} under {base.name}/ "
                        f"predates pilot_week {pilot} (A7)"
                    )

    # Feature freeze after the pilot.
    effective_week = week_id
    if effective_week is None:
        from portstudy.pipeline.jobs import current_week

        effective_week = current_week(now)
    if pilot and effective_week > pilot and not cfg.llm.gateway.features_frozen:
        failures.append(
            f"llm.gateway.features_frozen is false after pilot_week {pilot} "
            f"(current week {effective_week})"
        )

    # Frozen config.
    diff = frozen_config_diff(root)
    if diff:
        failures.append(
            "config changed since the pilot: "
            + ", ".join(diff)
            + '; run check_live_ready.py --accept-config-change "<note>" if intended'
        )

    for check in EXTRA_CHECKS:
        failures.extend(
            check(cfg=cfg, registry=registry, manifest=manifest, root=root, env=env, now=now)
        )

    report.markers = _marker_counts(root)
    return report
