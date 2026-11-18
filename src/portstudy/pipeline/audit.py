"""`portstudy audit --week W`: verify a week end to end (N6 (9)).

Every expected file exists, file hashes match what the snapshot manifest and the weights lock
recorded, every planned LLM request has a record, every portfolio has weights, and the fallback
ledger matches the weight metadata. Returns the list of problems; empty means the week passed.
"""

from __future__ import annotations

from portstudy.data.snapshot import read_snapshot, snapshot_exists
from portstudy.hashing import sha256_file
from portstudy.paths import week_output_dir
from portstudy.pipeline import fallback as fb
from portstudy.pipeline import fallback_ledger as fl
from portstudy.pipeline.context import RunContext
from portstudy.pipeline.status import read_status

REQUIRED_JOBS = ("J1", "J2", "J6", "J7")


def audit_week(ctx: RunContext, week_id: str) -> list[str]:
    from portstudy.llm.runner import read_run_status
    from portstudy.optimize import read_weights_meta, weights_dir
    from portstudy.pipeline.jobs import _week_runner, read_weights_lock

    problems: list[str] = []
    for job in REQUIRED_JOBS:
        state = read_status(week_id, job).state
        if state != "done":
            problems.append(f"{job} is {state}, not done")

    if not snapshot_exists(week_id):
        problems.append("snapshot missing")
    else:
        snap = read_snapshot(week_id)
        files = snap.manifest.get("files", {})
        for name, digest in dict(files).items():  # type: ignore[call-overload]
            path = snap.path / name
            if not path.exists():
                problems.append(f"snapshot file {name} missing")
            elif sha256_file(path) != digest:
                problems.append(f"snapshot file {name} changed since it was written")

    plan = fb.read_plan(week_id)
    lock = read_weights_lock(week_id)
    if lock is None:
        problems.append("weights/locked.json missing")
        return problems
    if lock.get("no_rebalance"):
        return problems
    for name, digest in dict(lock.get("files", {})).items():
        path = weights_dir(week_id) / name
        if not path.exists():
            problems.append(f"weight file {name} missing")
        elif sha256_file(path) != digest:
            problems.append(f"weight file {name} changed after the lock")
    for key in ctx.portfolio_keys():
        if not (weights_dir(week_id) / f"{key}.csv").exists():
            problems.append(f"portfolio {key} has no weights")

    statuses = read_run_status(week_id)
    runner = _week_runner(ctx, week_id) if snapshot_exists(week_id) else None
    for source in ctx.sources():
        if plan is not None and plan.statuses.get(source.key) == "fallback":
            continue
        st = statuses.get(source.key)
        if st is None or runner is None:
            problems.append(f"source {source.key} has no run status")
            continue
        recorded = runner._read_records(source.model, source.config_id)
        planned = [t for t in runner.plan_week(source.model) if t.config_id == source.config_id]
        missing = [t.item.custom_id for t in planned if t.item.custom_id not in recorded]
        if missing:
            problems.append(f"source {source.key}: {len(missing)} planned requests have no record")

    ledger = fl.read(week_id)
    served = set(ledger.loc[ledger["kind"] == "fallback_q0", "portfolio"]) if len(ledger) else set()
    for key in ctx.portfolio_keys():
        path = weights_dir(week_id) / f"{key}.meta.json"
        if not path.exists():
            continue
        is_fallback = read_weights_meta(week_id, key).get("decision_source") == "fallback_q0"
        if is_fallback and key not in served:
            problems.append(f"{key} traded a Q0 fallback that is not in fallbacks.csv")
        if not is_fallback and key in served:
            problems.append(f"fallbacks.csv lists {key} but its weights are not a Q0 fallback")
    if not (week_output_dir(week_id) / "llm_scores.csv").exists() and ctx.sources():
        problems.append("llm_scores.csv missing")
    return problems
