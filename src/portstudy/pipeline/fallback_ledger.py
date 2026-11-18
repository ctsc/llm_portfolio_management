"""Weekly fallback ledger, data/outputs/<week>/fallbacks.csv (N6 (5)).

One row per degradation: Q0 fallbacks, `no_rebalance`, Kronos fallback, `provider_error`,
invalid outputs, solver fallbacks, `optimal_inaccurate`, carried market caps, forward-filled
prices, skipped and removed stocks, no-price liquidations, disabled gateway features and
`model_change` flags. Rows are keyed by (kind, job, portfolio, source, ticker), so a rerun
replaces a row instead of adding a second one; every recorded degradation also goes to the
event timeline (which carries the run_id and time; the CSV stays byte-identical on reruns).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd

from portstudy.io import read_csv, write_csv_atomic
from portstudy.paths import week_output_dir

LEDGER_COLUMNS = [
    "week_id",
    "kind",
    "job",
    "portfolio",
    "source",
    "ticker",
    "reason",
]
KEY = ["kind", "job", "portfolio", "source", "ticker"]
_LOCK = threading.Lock()


def ledger_path(week_id: str) -> Path:
    return week_output_dir(week_id) / "fallbacks.csv"


def read(week_id: str) -> pd.DataFrame:
    path = ledger_path(week_id)
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    return read_csv(path, keep_default_na=False, dtype=str)


def record_many(week_id: str, rows: list[dict[str, str]]) -> None:
    """Upsert several rows at once (one file rewrite)."""
    if not rows:
        return
    from portstudy.pipeline.events import emit

    new = pd.DataFrame(
        [
            {
                "week_id": week_id,
                "kind": r["kind"],
                "job": r.get("job", ""),
                "portfolio": r.get("portfolio", ""),
                "source": r.get("source", ""),
                "ticker": r.get("ticker", ""),
                "reason": r.get("reason", ""),
            }
            for r in rows
        ],
        columns=LEDGER_COLUMNS,
    ).drop_duplicates(KEY, keep="last")
    with _LOCK:
        old = read(week_id)
        if len(old):
            keep = ~old.set_index(KEY).index.isin(new.set_index(KEY).index)
            old = old[keep]
        merged = pd.concat([old, new], ignore_index=True) if len(old) else new
        merged = merged.sort_values(KEY, kind="stable").reset_index(drop=True)
        write_csv_atomic(merged.loc[:, LEDGER_COLUMNS], ledger_path(week_id))
    for r in new.to_dict("records"):
        emit("fallback", **{("fallback_kind" if k == "kind" else str(k)): v for k, v in r.items()})


def record(
    week_id: str,
    kind: str,
    reason: str,
    job: str,
    portfolio: str = "",
    source: str = "",
    ticker: str = "",
) -> None:
    """Upsert one degradation for the week."""
    record_many(
        week_id,
        [
            {
                "kind": kind,
                "job": job,
                "portfolio": portfolio,
                "source": source,
                "ticker": ticker,
                "reason": reason,
            }
        ],
    )
