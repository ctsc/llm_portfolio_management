"""data/outputs/<week>/run_summary.md, written at the end of every run (N6 (8)).

Lists every job with state and duration, counts of records, invalid outputs, fallbacks,
warnings and errors for the run, and a "needs attention" list that the run script prints.
"""

from __future__ import annotations

import json
from pathlib import Path

from portstudy.io import write_text_atomic
from portstudy.logging_setup import current_run_id
from portstudy.paths import logs_dir, week_output_dir
from portstudy.pipeline import fallback_ledger as fl
from portstudy.pipeline.status import JOB_NAMES, read_status

ATTENTION_KINDS = {
    "kronos_fallback",
    "fallback_q0",
    "no_rebalance",
    "provider_error",
    "solver_fallback",
    "optimal_inaccurate",
    "removed_stock",
    "no_price_liquidation",
    "feature_disabled",
    "model_change",
}


def _log_counts(run_id: str) -> tuple[int, int]:
    """Warnings and errors written by this run across the JSON-lines job logs."""
    warnings = errors = 0
    if not run_id:
        return 0, 0
    for path in logs_dir().glob("*/*.jsonl"):
        if path.parent.name == "runs":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if f'"run_id": "{run_id}"' not in line:
                continue
            level = json.loads(line).get("level")
            warnings += level == "WARNING"
            errors += level in ("ERROR", "CRITICAL")
    return warnings, errors


def write_run_summary(
    week_id: str,
    failed_job: str = "",
    messages: list[str] | None = None,
    audit_problems: list[str] | None = None,
) -> tuple[Path, list[str]]:
    """Write the summary and return its path and the needs-attention lines."""
    from portstudy.llm.runner import read_scores

    run_id = current_run_id()
    lines = [f"# Run summary {week_id}", "", f"- run_id: `{run_id}`"]
    if failed_job:
        lines.append(f"- stopped at: {failed_job}")
    lines += ["", "| job | state | seconds | error |", "|---|---|---|---|"]
    attention: list[str] = []
    for job in JOB_NAMES:
        st = read_status(week_id, job)
        secs = f"{st.duration_s:.1f}" if st.duration_s is not None else ""
        lines.append(f"| {job} | {st.state} | {secs} | {st.error or ''} |")
        if st.state == "failed":
            attention.append(f"{job} failed: {st.error}")
    scores = read_scores(week_id)
    n_records = len(scores)
    n_invalid = int((~scores["valid"].astype(bool)).sum()) if n_records else 0
    ledger = fl.read(week_id)
    warnings, errors = _log_counts(run_id)
    lines += [
        "",
        f"- LLM records: {n_records}; invalid: {n_invalid}",
        f"- fallback ledger rows: {len(ledger)}",
        f"- warnings: {warnings}; errors: {errors}",
    ]
    for row in ledger.to_dict("records") if len(ledger) else []:
        if row["kind"] in ATTENTION_KINDS:
            target = row["portfolio"] or row["source"] or row["ticker"] or "*"
            attention.append(f"{row['kind']} ({row['job']}) {target}: {row['reason']}")
    attention += list(messages or [])
    if audit_problems:
        attention += [f"audit: {p}" for p in audit_problems]
    lines += ["", "## Needs attention", ""]
    lines += [f"- {a}" for a in attention] or ["- nothing"]
    path = week_output_dir(week_id) / "run_summary.md"
    write_text_atomic("\n".join(lines) + "\n", path)
    return path, attention
