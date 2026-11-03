"""Project the study cost per model and configuration from logged runs (DECISION(D3)).

Reads real token counts from every data/outputs/<week>/llm_scores.csv written so far (the
Phase 17 prompt test runs), averages tokens per call, and prints the projected cost for the
remaining weeks with and without prompt caching.

Usage: python scripts/estimate_cost.py [--weeks N] [--stocks 100]
"""

from __future__ import annotations

import argparse

import pandas as pd

from portstudy.config import load_model_registry, load_study_config
from portstudy.llm.cost import CostTable
from portstudy.llm.prompts import load_manifest
from portstudy.paths import outputs_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weeks", type=int, default=20, help="remaining live weeks to project")
    parser.add_argument(
        "--stocks", type=int, default=None, help="stocks per week (default: observed)"
    )
    args = parser.parse_args()
    cfg = load_study_config()
    registry = load_model_registry()
    table = CostTable(registry)
    manifest = load_manifest()

    frames = []
    for path in sorted(outputs_dir().glob("*/llm_scores.csv")):
        df = pd.read_csv(path)
        if len(df):
            df["week_id"] = path.parent.name
            frames.append(df)
    if not frames:
        print("no llm_scores.csv found under", outputs_dir())
        n_stocks = args.stocks or 100
        planned_calls = len(registry.enabled()) * len(manifest.enabled()) * n_stocks
        print(
            f"planned calls per week: {len(registry.enabled())} models x "
            f"{len(manifest.enabled())} prompt types x {n_stocks} stocks = {planned_calls}; "
            f"{planned_calls * args.weeks} over {args.weeks} weeks"
        )
        return
    scores = pd.concat(frames, ignore_index=True)
    rows = []
    for (model, config_id), g in scores.groupby(["model", "config_id"]):
        try:
            spec = registry.get(str(model))
        except KeyError:
            continue
        n_stocks = args.stocks or int(g.groupby("week_id").size().mean())
        tin = float(g["tokens_in"].mean())
        tcached = float(g["tokens_cached"].mean())
        tout = float(g["tokens_out"].mean())
        batch = spec.kind == "api"
        per_call_cached = table.cost_usd(spec.id, int(tin), int(tcached), int(tout), batch)
        per_call_fresh = table.cost_usd(spec.id, int(tin), 0, int(tout), batch)
        rows.append(
            {
                "model": model,
                "config_id": config_id,
                "weeks_logged": g["week_id"].nunique(),
                "stocks_per_week": n_stocks,
                "calls_per_week": n_stocks,
                "projected_calls": n_stocks * args.weeks,
                "tokens_in": round(tin),
                "tokens_cached": round(tcached),
                "tokens_out": round(tout),
                "usd_per_week_cached": per_call_cached * n_stocks,
                "usd_per_week_no_cache": per_call_fresh * n_stocks,
                "usd_projected_cached": per_call_cached * n_stocks * args.weeks,
                "usd_projected_no_cache": per_call_fresh * n_stocks * args.weeks,
            }
        )
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()
    print(f"configurations: {[c.id for c in manifest.enabled()]}")
    n_stocks = args.stocks or int(
        scores.groupby(["week_id", "model", "config_id"]).size().median()
    )
    planned_calls = len(registry.enabled()) * len(manifest.enabled()) * n_stocks
    print(
        f"projected total over {args.weeks} weeks: "
        f"{out['usd_projected_cached'].sum():.2f} USD with caching, "
        f"{out['usd_projected_no_cache'].sum():.2f} USD without; "
        f"goal {cfg.llm.cost_goal_usd:.2f} USD; "
        f"planned calls per week {len(registry.enabled())} models x "
        f"{len(manifest.enabled())} prompt types x {n_stocks} stocks = {planned_calls}, "
        f"{planned_calls * args.weeks} over {args.weeks} weeks"
    )


if __name__ == "__main__":
    main()
