"""One row per week in data/outputs/study_log.csv (Phase 3A).

`run_weekend.py` upserts the week's row at the end of every run, including failed runs, so the
log shows each week's jobs, data, sources, costs, ARCTIC usage, backup and audit results in one
place. Columns that a later step fills (the G21 canary columns) stay empty until it runs.
"""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from portstudy.config import repo_root
from portstudy.data.snapshot import read_snapshot, snapshot_exists
from portstudy.hashing import sha256_file, stable_hash
from portstudy.io import read_csv, read_json, write_csv_atomic
from portstudy.logging_setup import current_run_id, now_et
from portstudy.paths import data_dir, outputs_dir, week_output_dir
from portstudy.pipeline.context import RunContext
from portstudy.pipeline.status import JOB_NAMES, read_status

CONFIG_FILES = ("config/study.yaml", "config/models.yaml", "config/prompts_manifest.yaml")
PREREG_FILE = "config/preregistration.yaml"

STUDY_LOG_COLUMNS = [
    "week_id",
    "run_id",
    "started_at",
    "finished_at",
    *[f"{j}_{field}" for j in JOB_NAMES for field in ("state", "s")],
    "failed_job",
    "git_commit",
    "config_hash",
    "preregistration_hash",
    "universe_size",
    "n_removed",
    "removed",
    "n_large_move",
    "n_market_cap_carried",
    "price_provider",
    "price_feed",
    "providers",
    "kronos_fallback",
    "kronos_peak_gpu_mb",
    "sources",
    "n_fallback_rows",
    "api_cost_usd",
    "gateway_models",
    "gateway_stats",
    "canary_matches",
    "model_versions",
    "model_change",
    "arctic_gpu_hours_est",
    "arctic_gpu_hours_total",
    "arctic_gpu_hours_projected",
    "litellm_cost_usd",
    "litellm_cost_total",
    "litellm_cost_projected",
    "backup_path",
    "backup_verified",
    "audit_ok",
    "audit_problems",
    "notes",
]


def study_log_path() -> Path:
    return outputs_dir() / "study_log.csv"


def read_study_log() -> pd.DataFrame:
    path = study_log_path()
    if not path.exists():
        return pd.DataFrame(columns=STUDY_LOG_COLUMNS)
    return read_csv(path, keep_default_na=False, dtype=str)


def config_hash(root: Path | None = None) -> str:
    base = root or repo_root()
    return stable_hash({name: sha256_file(base / name) for name in CONFIG_FILES})


def git_commit(root: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root or repo_root(),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _snapshot_fields(ctx: RunContext, week_id: str) -> dict[str, Any]:
    if not snapshot_exists(week_id):
        return {}
    snap = read_snapshot(week_id)
    manifest = snap.manifest
    issues = snap.issues
    raw_removed = manifest.get("removed") or []
    removed = [str(t) for t in raw_removed] if isinstance(raw_removed, list) else []
    return {
        "universe_size": len(snap.included_tickers),
        "n_removed": len(removed),
        "removed": ";".join(removed),
        "n_large_move": int((issues["code"] == "large_move").sum()) if len(issues) else 0,
        "n_market_cap_carried": manifest.get("n_market_cap_carried", ""),
        "price_provider": ctx.cfg.data.price_provider,
        "price_feed": ctx.cfg.data.alpaca_feed if ctx.cfg.data.price_provider == "alpaca" else "",
        "providers": _json(manifest.get("providers", {})),
    }


def _source_fields(ctx: RunContext, week_id: str) -> dict[str, Any]:
    from portstudy.llm.runner import read_run_status, read_scores
    from portstudy.pipeline import fallback as fb

    plan = fb.read_plan(week_id)
    statuses = read_run_status(week_id)
    sources: dict[str, dict[str, Any]] = {}
    for source in ctx.sources():
        st = statuses.get(source.key)
        sources[source.key] = {
            "status": plan.status_for(source.key) if plan else "",
            "reason": plan.reason_for(source.key) if plan else "",
            "invalid_rate": round(st.invalid_rate, 4) if st else None,
        }
    scores = read_scores(week_id)
    stats: dict[str, dict[str, Any]] = {}
    if len(scores):
        for model in ctx.open_weight_models():
            sub = scores[scores["model"] == model]
            if not len(sub):
                continue
            finish = sub["finish_reason"].astype(str) if "finish_reason" in sub else pd.Series()
            stats[model] = {
                "mean_seconds": round(float(sub["latency_s"].astype(float).mean()), 3),
                "mean_completion_tokens": round(float(sub["tokens_out"].astype(float).mean()), 1),
                "length_share": round(float((finish == "length").mean()), 4) if len(finish) else 0,
            }
    cost_path = week_output_dir(week_id) / "cost_log.csv"
    api_cost = 0.0
    if cost_path.exists():
        cost = read_csv(cost_path)
        api_models = set(ctx.api_models())
        api_cost = float(cost.loc[cost["model"].isin(api_models), "cost_usd"].sum())
    return {"sources": _json(sources), "gateway_stats": _json(stats), "api_cost_usd": api_cost}


def _backup_fields(week_id: str) -> dict[str, Any]:
    status = read_status(week_id, "J9")
    path = str((status.result or {}).get("path", "")) if status.state == "done" else ""
    verified: bool | str = ""
    if path:
        zip_path = Path(path).with_suffix(".zip")
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                verified = zf.testzip() is None
        else:
            verified = Path(path).is_dir()
    return {"backup_path": path, "backup_verified": verified}


def week_row(
    ctx: RunContext,
    week_id: str,
    started_at: str,
    failed_job: str = "",
    audit_problems: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect every column for the week from the files the run wrote."""
    from portstudy.baseline.combine import read_baseline_meta
    from portstudy.llm.usage import usage_summary
    from portstudy.pipeline import fallback_ledger as fl

    row: dict[str, Any] = {c: "" for c in STUDY_LOG_COLUMNS}
    row.update(
        week_id=week_id,
        run_id=current_run_id(),
        started_at=started_at,
        finished_at=now_et().isoformat(),
        failed_job=failed_job,
        git_commit=git_commit(),
        config_hash=config_hash(),
        preregistration_hash=(
            sha256_file(repo_root() / PREREG_FILE) if (repo_root() / PREREG_FILE).exists() else ""
        ),
    )
    for job in JOB_NAMES:
        st = read_status(week_id, job)
        row[f"{job}_state"] = st.state
        row[f"{job}_s"] = round(st.duration_s, 3) if st.duration_s is not None else ""
    row.update(_snapshot_fields(ctx, week_id))
    meta = read_baseline_meta(week_id)
    row["kronos_fallback"] = bool(meta.get("kronos_fallback", False))
    row["kronos_peak_gpu_mb"] = meta.get("kronos_peak_gpu_mb") or ""
    row.update(_source_fields(ctx, week_id))
    row["n_fallback_rows"] = len(fl.read(week_id))
    usage = usage_summary(ctx.cfg, ctx.registry, week_id)
    row.update(
        arctic_gpu_hours_est=round(usage.week.gpu_hours_est, 4),
        arctic_gpu_hours_total=round(usage.gpu_hours_total, 4),
        arctic_gpu_hours_projected=round(usage.gpu_hours_projected, 4),
        litellm_cost_usd=round(usage.week.litellm_cost_usd, 6),
        litellm_cost_total=round(usage.litellm_cost_total, 6),
        litellm_cost_projected=round(usage.litellm_cost_projected, 6),
    )
    models_file = data_dir() / "runs" / week_id / "gateway_models.json"
    if models_file.exists():
        row["gateway_models"] = _json(read_json(models_file).get("models", []))
    from portstudy.llm.canary import read_checks

    checks = read_checks(week_id)
    if len(checks):
        by_model = checks.set_index("model")
        row["canary_matches"] = _json(
            {m: f"{r['matches']}/{r['n']}" for m, r in by_model.iterrows()}
        )
        row["model_versions"] = _json({m: r["model_version"] for m, r in by_model.iterrows()})
        row["model_change"] = _json({m: r["model_change"] for m, r in by_model.iterrows()})
    row.update(_backup_fields(week_id))
    if audit_problems is not None:
        row["audit_ok"] = not audit_problems
        row["audit_problems"] = " | ".join(audit_problems)
    row.update(extra or {})
    return row


def upsert(row: dict[str, Any]) -> pd.DataFrame:
    """Insert or replace the week's row; the file keeps one row per week."""
    new = pd.DataFrame([{c: row.get(c, "") for c in STUDY_LOG_COLUMNS}]).astype(str)
    old = read_study_log()
    if len(old):
        old = old[old["week_id"] != str(row["week_id"])]
    merged = pd.concat([old, new], ignore_index=True) if len(old) else new
    merged = merged.reindex(columns=STUDY_LOG_COLUMNS).sort_values("week_id", kind="stable")
    write_csv_atomic(merged.reset_index(drop=True), study_log_path())
    return merged
