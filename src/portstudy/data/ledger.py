"""Append-only price ledger: raw daily OHLC and corporate actions, written once, never rewritten.

The simulator reads only the ledger, so past fills, P&L, and stops can never change when a
provider later re-adjusts its history.
"""

from __future__ import annotations

import logging

import pandas as pd

from portstudy.io import read_csv, write_csv_atomic
from portstudy.paths import ledger_dir

LEDGER_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]
ACTION_COLUMNS = ["date", "ticker", "action", "value"]


def _prices_path() -> str:
    return str(ledger_dir() / "prices_raw.csv")


def _actions_path() -> str:
    return str(ledger_dir() / "corporate_actions.csv")


def read_ledger() -> pd.DataFrame:
    path = ledger_dir() / "prices_raw.csv"
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    df = read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def read_actions() -> pd.DataFrame:
    path = ledger_dir() / "corporate_actions.csv"
    if not path.exists():
        return pd.DataFrame(columns=ACTION_COLUMNS)
    df = read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def append_prices(new_rows: pd.DataFrame, logger: logging.Logger | None = None) -> int:
    """Append rows whose (date, ticker) is not yet in the ledger. Returns the count added.

    Rows already present are never rewritten; a differing provider value logs
    `ledger_mismatch` and the ledger value is kept.
    """
    if len(new_rows) == 0:
        return 0
    incoming = new_rows.loc[:, LEDGER_COLUMNS].copy()
    incoming["date"] = pd.to_datetime(incoming["date"]).dt.date
    incoming = incoming.drop_duplicates(["date", "ticker"], keep="first")
    existing = read_ledger()
    if len(existing):
        key_existing = existing.set_index(["date", "ticker"])
        key_incoming = incoming.set_index(["date", "ticker"])
        overlap = key_incoming.index.intersection(key_existing.index)
        if len(overlap) and logger is not None:
            old = key_existing.loc[overlap, ["open", "high", "low", "close"]].astype(float)
            new = key_incoming.loc[overlap, ["open", "high", "low", "close"]].astype(float)
            diff = (old - new).abs().max(axis=1) > 1e-9
            for (d, tk) in old.index[diff.to_numpy()]:
                logger.warning(
                    "ledger_mismatch",
                    extra={"extra_fields": {"date": str(d), "ticker": tk,
                                            "ledger_close": float(old.loc[(d, tk), "close"]),
                                            "provider_close": float(new.loc[(d, tk), "close"])}},
                )
        fresh = key_incoming.drop(overlap).reset_index()
    else:
        fresh = incoming
    if len(fresh) == 0:
        return 0
    merged = pd.concat([existing, fresh], ignore_index=True)
    merged = merged.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
    write_csv_atomic(merged, ledger_dir() / "prices_raw.csv")
    return len(fresh)


def append_actions(new_rows: pd.DataFrame) -> int:
    if len(new_rows) == 0:
        return 0
    incoming = new_rows.loc[:, ACTION_COLUMNS].copy()
    incoming["date"] = pd.to_datetime(incoming["date"]).dt.date
    existing = read_actions()
    if len(existing):
        key = set(zip(existing["date"], existing["ticker"], existing["action"]))
        mask = [
            (d, t, a) not in key
            for d, t, a in zip(incoming["date"], incoming["ticker"], incoming["action"])
        ]
        incoming = incoming[mask]
    if len(incoming) == 0:
        return 0
    merged = pd.concat([existing, incoming], ignore_index=True)
    merged = merged.sort_values(["date", "ticker", "action"], kind="stable").reset_index(drop=True)
    write_csv_atomic(merged, ledger_dir() / "corporate_actions.csv")
    return len(incoming)
