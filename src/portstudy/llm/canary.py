"""Weekly model checks and canaries (G21) and the pilot repeat check (G20).

Before J4 each weekend: save the gateway's `/v1/models` and `/v1/model/info` answers, send the
frozen canary prompts (`config/canary/canary_inputs.csv`, 10 stocks with fixed inputs) to every
gateway model with the response cache bypassed, and compare the answers with the pilot's
(`data/outputs/canary_baseline.csv`). A changed `model_version` or deployment ID, or fewer
identical `likely` scores than the binomial bound allows, flags `model_change`; a changed
gateway version or model `updated_at` is logged as `gateway_changed` (information only).
Results go to `data/outputs/model_checks.csv`, the study log and the fallback ledger. A flag
never stops the run. Canary and repeat-check records are kept apart from the portfolio records.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats

from portstudy.config import ModelSpec, repo_root
from portstudy.io import read_csv, write_csv_atomic, write_json_atomic, write_text_atomic
from portstudy.llm.prompts import assemble_prompt
from portstudy.llm.providers.base import BatchItem, make_custom_id
from portstudy.paths import data_dir, outputs_dir, week_output_dir

CANARY_FILE = "config/canary/canary_inputs.csv"
CANARY_COLUMNS = ["ticker", "name", "sector", "news_text"]
BASELINE_COLUMNS = [
    "model",
    "ticker",
    "likely",
    "valid",
    "model_version",
    "deployment_id",
    "gateway_version",
    "updated_at",
]
CHECK_COLUMNS = [
    "week_id",
    "model",
    "n",
    "matches",
    "min_matches",
    "agreement_share",
    "model_version",
    "deployment_id",
    "gateway_version",
    "updated_at",
    "model_change",
    "gateway_changed",
    "reasons",
]
DEFAULT_AGREEMENT = 0.8  # used until the pilot repeat check has measured it
FLAG_ALPHA = 0.01
REPEAT_TICKERS = 50
CANARY_WEEK = "canary"
_LOG = logging.getLogger("portstudy.llm.canary")

ModelFetch = Callable[[str], dict[str, Any] | None]


def gateway_models_path(week_id: str) -> Path:
    return data_dir() / "runs" / week_id / "gateway_models.json"


def load_canary_inputs(root: Path | None = None) -> pd.DataFrame:
    df = read_csv((root or repo_root()) / CANARY_FILE, keep_default_na=False)
    missing = [c for c in CANARY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{CANARY_FILE} is missing columns {missing}")
    return df.loc[:, CANARY_COLUMNS]


def _fetch_json(base_url: str, path: str) -> dict[str, Any] | None:
    """GET <base_url><path> with the gateway key; None when refused or unreachable."""
    import httpx

    key = os.environ.get("ARCTIC_LITELLM_API_KEY", "")
    try:
        response = httpx.get(
            base_url.rstrip("/") + path,
            headers={"Authorization": f"Bearer {key}"} if key else {},
            timeout=20.0,
        )
    except httpx.HTTPError as exc:
        _LOG.warning("gateway_models_unreachable", extra={"extra_fields": {"error": str(exc)}})
        return None
    if response.status_code >= 400:
        _LOG.warning(
            "gateway_models_refused",
            extra={"extra_fields": {"path": path, "status": response.status_code}},
        )
        return None
    data = response.json()
    return data if isinstance(data, dict) else {"data": data}


def save_gateway_models(
    week_id: str, base_url: str, fetch: ModelFetch | None = None
) -> dict[str, Any]:
    """Save `/v1/models` and `/v1/model/info` (every field) for the week."""
    get = fetch or (lambda path: _fetch_json(base_url, path))
    models = get("/models")
    info = get("/model/info")
    if info is None:
        _LOG.info("model_info_unavailable", extra={"extra_fields": {"week_id": week_id}})
    out = {
        "models": [m.get("id") for m in (models or {}).get("data", []) if isinstance(m, dict)],
        "models_raw": models,
        "model_info": info,
    }
    write_json_atomic(out, gateway_models_path(week_id))
    return out


def updated_at_by_model(info: dict[str, Any] | None) -> dict[str, str]:
    """`updated_at` per model name from `/v1/model/info` (LiteLLM's `data` list)."""
    out: dict[str, str] = {}
    for entry in (info or {}).get("data", []) or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("model_name") or "")
        meta = entry.get("model_info") or {}
        stamp = meta.get("updated_at") if isinstance(meta, dict) else None
        if name and stamp:
            out[name] = str(stamp)
    return out


def canary_items(model: str, config_id: str, week_id: str, ctx: Any) -> list[BatchItem]:
    rows = load_canary_inputs()
    items = []
    for row in rows.to_dict("records"):
        prompt = assemble_prompt(
            config_id,
            {"ticker": row["ticker"], "name": row["name"], "sector": row["sector"]},
            str(row["news_text"]),
            ctx.cfg,
            ctx.run_manifest(),
        )
        # The canary inputs come from one frozen snapshot, so the request identity is the same
        # every week (not the current week); records are kept per week in the canary folder.
        items.append(
            BatchItem(
                make_custom_id(model, config_id, CANARY_WEEK, str(row["ticker"])),
                model,
                config_id,
                CANARY_WEEK,
                str(row["ticker"]),
                prompt,
            )
        )
    return items


def checked_models(ctx: Any) -> list[str]:
    """Gateway open-weight models; stub models only when a provider is injected (tests)."""
    registry = ctx.run_registry()
    out = []
    for model_id in ctx.open_weight_models():
        spec: ModelSpec = registry.get(model_id)
        if spec.serving != "gateway":
            continue
        if spec.provider == "stub" and model_id not in ctx.llm_providers:
            continue
        out.append(model_id)
    return out


def _write_records(folder: Path, model: str, records: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in records)
    write_text_atomic(text, folder / f"{model}.jsonl")


def rate_canaries(
    ctx: Any, week_id: str, runner: Any, folder: Path, models: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    config_id = ctx.prompt_ids()[0]
    out: dict[str, pd.DataFrame] = {}
    for model in models if models is not None else checked_models(ctx):
        records = runner.rate_adhoc(model, canary_items(model, config_id, week_id, ctx))
        _write_records(folder, model, records)
        out[model] = pd.DataFrame(records)
    return out


def min_matches(n: int, share: float, alpha: float = FLAG_ALPHA) -> int:
    """Smallest identical-answer count that is not suspicious at `alpha`: fewer flags."""
    if n <= 0:
        return 0
    share = min(max(share, 0.0), 1.0)
    for k in range(n + 1):
        if stats.binom.cdf(k, n, share) >= alpha:
            return k
    return n


def agreement_share(model: str) -> float:
    path = outputs_dir() / "pilot_repeatability.csv"
    if not path.exists():
        return DEFAULT_AGREEMENT
    df = read_csv(path)
    sub = df[df["model"] == model]
    if not len(sub):
        return DEFAULT_AGREEMENT
    return float(sub["share_identical"].astype(float).mean())


def baseline_path() -> Path:
    return outputs_dir() / "canary_baseline.csv"


def checks_path() -> Path:
    return outputs_dir() / "model_checks.csv"


def _first(df: pd.DataFrame, column: str) -> str:
    values = [str(v) for v in df.get(column, pd.Series(dtype=str)) if str(v) not in ("", "nan")]
    return values[0] if values else ""


def compare(
    week_id: str,
    model: str,
    answers: pd.DataFrame,
    baseline: pd.DataFrame,
    updated_at: str,
) -> dict[str, Any]:
    """Compare one model's canary answers with the pilot baseline."""
    base = baseline[baseline["model"] == model].set_index("ticker")
    mine = answers.set_index("ticker") if len(answers) else pd.DataFrame()
    common = [t for t in base.index if t in mine.index]
    matches = sum(
        bool(mine.loc[t, "valid"]) and int(mine.loc[t, "likely"]) == int(base.loc[t, "likely"])
        for t in common
    )
    share = agreement_share(model)
    bound = min_matches(len(common), share)
    version, deployment = _first(answers, "model_version"), _first(answers, "deployment_id")
    gateway_version = _first(answers, "gateway_version")
    reasons: list[str] = []
    flag = False
    if len(base):
        b_version, b_deploy = (
            _first(base.reset_index(), "model_version"),
            _first(base.reset_index(), "deployment_id"),
        )
        if b_version and version and version != b_version:
            flag = True
            reasons.append(f"model_version {b_version} -> {version}")
        if b_deploy and deployment and deployment != b_deploy:
            flag = True
            reasons.append(f"deployment_id {b_deploy} -> {deployment}")
        if matches < bound:
            flag = True
            reasons.append(f"{matches} of {len(common)} identical, below the bound {bound}")
    else:
        reasons.append("no pilot baseline yet")
    changed = False
    if len(base):
        b_gateway = _first(base.reset_index(), "gateway_version")
        b_updated = _first(base.reset_index(), "updated_at")
        if b_gateway and gateway_version and gateway_version != b_gateway:
            changed = True
            reasons.append(f"gateway_changed: version {b_gateway} -> {gateway_version}")
        if b_updated and updated_at and updated_at != b_updated:
            changed = True
            reasons.append(f"gateway_changed: updated_at {b_updated} -> {updated_at}")
    return {
        "week_id": week_id,
        "model": model,
        "n": len(common),
        "matches": matches,
        "min_matches": bound,
        "agreement_share": round(share, 4),
        "model_version": version,
        "deployment_id": deployment,
        "gateway_version": gateway_version,
        "updated_at": updated_at,
        "model_change": flag,
        "gateway_changed": changed,
        "reasons": "; ".join(reasons),
    }


def _upsert_checks(rows: list[dict[str, Any]]) -> None:
    path = checks_path()
    old = read_csv(path, keep_default_na=False, dtype=str) if path.exists() else pd.DataFrame()
    new = pd.DataFrame(rows, columns=CHECK_COLUMNS).astype(str)
    if len(old):
        key = old["week_id"] + "|" + old["model"]
        old = old[~key.isin(set(new["week_id"] + "|" + new["model"]))]
    merged = pd.concat([old, new], ignore_index=True) if len(old) else new
    merged = merged.sort_values(["week_id", "model"], kind="stable").reset_index(drop=True)
    write_csv_atomic(merged.loc[:, CHECK_COLUMNS], path)


def read_checks(week_id: str) -> pd.DataFrame:
    path = checks_path()
    if not path.exists():
        return pd.DataFrame(columns=CHECK_COLUMNS)
    df = read_csv(path, keep_default_na=False, dtype=str)
    return df[df["week_id"] == week_id]


def run_canary(
    ctx: Any, week_id: str, runner: Any, fetch: ModelFetch | None = None
) -> list[dict[str, Any]]:
    """G21 for one week; returns one check row per model."""
    from portstudy.llm.health import unhealthy_models
    from portstudy.pipeline import fallback_ledger as fl

    down = unhealthy_models(week_id)  # a model the health check marked failed is not probed again
    models = [m for m in checked_models(ctx) if m not in down]
    if not models:
        return []
    saved = save_gateway_models(week_id, ctx.cfg.llm.gateway.base_url, fetch)
    stamps = updated_at_by_model(saved.get("model_info"))
    answers = rate_canaries(ctx, week_id, runner, week_output_dir(week_id) / "canary", models)
    baseline = (
        read_csv(baseline_path(), keep_default_na=False)
        if baseline_path().exists()
        else pd.DataFrame(columns=BASELINE_COLUMNS)
    )
    rows = []
    for model in models:
        spec = ctx.registry.get(model)
        row = compare(week_id, model, answers[model], baseline, stamps.get(spec.provider_model, ""))
        rows.append(row)
        if row["model_change"]:
            _LOG.warning("model_change", extra={"extra_fields": row})
            fl.record(week_id, "model_change", row["reasons"], "J4", source=model)
        elif row["gateway_changed"]:
            _LOG.info("gateway_changed", extra={"extra_fields": row})
    _upsert_checks(rows)
    return rows


def write_canary_baseline(
    ctx: Any, week_id: str, runner: Any, fetch: ModelFetch | None = None
) -> Path:
    """The pilot's canary answers, the reference for every later week (G20)."""
    saved = save_gateway_models(week_id, ctx.cfg.llm.gateway.base_url, fetch)
    stamps = updated_at_by_model(saved.get("model_info"))
    folder = week_output_dir(week_id) / "repeat_check"
    answers = rate_canaries(ctx, week_id, runner, folder)
    frames = []
    for model, df in answers.items():
        spec = ctx.registry.get(model)
        part = pd.DataFrame(
            {
                "model": model,
                "ticker": df["ticker"],
                "likely": df["likely"],
                "valid": df["valid"],
                "model_version": df.get("model_version", ""),
                "deployment_id": df.get("deployment_id", ""),
                "gateway_version": df.get("gateway_version", ""),
                "updated_at": stamps.get(spec.provider_model, ""),
            }
        )
        frames.append(part)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BASELINE_COLUMNS)
    write_csv_atomic(out.loc[:, BASELINE_COLUMNS], baseline_path())
    return baseline_path()


def repeat_check(
    ctx: Any, week_id: str, runner: Any, fetch: ModelFetch | None = None
) -> pd.DataFrame:
    """G20: rate the first 50 universe stocks again under every prompt type and compare with
    the week's J4 records; writes pilot_repeatability.csv and the canary baseline."""
    rows = []
    folder = week_output_dir(week_id) / "repeat_check"
    for model in checked_models(ctx):
        tasks = runner.plan_week(model)
        tickers = sorted({t.ticker for t in tasks})[:REPEAT_TICKERS]
        chosen = [t for t in tasks if t.ticker in set(tickers)]
        again = runner.rate_adhoc(model, [t.item for t in chosen])
        _write_records(folder, f"{model}__repeat", again)
        by_id = {r["custom_id"]: r for r in again}
        for config_id in sorted({t.config_id for t in chosen}):
            first = runner._read_records(model, config_id)
            pairs = []
            invalid = 0
            for t in chosen:
                if t.config_id != config_id:
                    continue
                a, b = first.get(t.item.custom_id), by_id.get(t.item.custom_id)
                if a is None or b is None:
                    continue
                if not (a["valid"] and b["valid"]):
                    invalid += 1
                    continue
                pairs.append((int(a["likely"]), int(b["likely"])))
            df = pd.DataFrame(pairs, columns=["first", "again"])
            n = len(df)
            rho = (
                float(stats.spearmanr(df["first"], df["again"]).statistic)
                if n >= 3 and df["first"].nunique() > 1 and df["again"].nunique() > 1
                else float("nan")
            )
            rows.append(
                {
                    "model": model,
                    "config_id": config_id,
                    "n": n,
                    "share_identical": float((df["first"] == df["again"]).mean())
                    if n
                    else float("nan"),
                    "share_within_1": float(((df["first"] - df["again"]).abs() <= 1).mean())
                    if n
                    else float("nan"),
                    "rank_correlation": rho,
                    "invalid": invalid,
                }
            )
    table = pd.DataFrame(
        rows,
        columns=[
            "model",
            "config_id",
            "n",
            "share_identical",
            "share_within_1",
            "rank_correlation",
            "invalid",
        ],
    )
    write_csv_atomic(table, outputs_dir() / "pilot_repeatability.csv")
    write_canary_baseline(ctx, week_id, runner, fetch)
    return table
