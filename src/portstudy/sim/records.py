"""Trade and daily record storage under data/portfolios/<portfolio>/."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from portstudy.hashing import sha256_file
from portstudy.io import read_csv, write_csv_atomic
from portstudy.paths import portfolios_dir
from portstudy.types import DailyRecord, Trade

TRADE_COLUMNS = [
    "week_id",
    "date",
    "portfolio_id",
    "ticker",
    "side",
    "shares",
    "fill_price",
    "trade_value",
    "cost_bps",
    "cost_usd",
    "reason",
    "decision_source",
    "avg_entry_price_before",
    "avg_entry_price_after",
    "weight_before",
    "weight_after",
    "stop_trigger",
]
DAILY_COLUMNS = [
    "date",
    "portfolio_id",
    "value",
    "cash",
    "cumulative_cost",
    "n_positions",
    "week_status",
    "week_id",
    "flags",
]


def portfolio_dir(portfolio_id: str) -> Path:
    path = portfolios_dir() / portfolio_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def trades_frame(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.DataFrame([t.to_dict() for t in trades]).loc[:, TRADE_COLUMNS]


def daily_frame(records: list[DailyRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    return pd.DataFrame([r.to_dict() for r in records]).loc[:, DAILY_COLUMNS]


def write_records(
    portfolio_id: str, trades: list[Trade], records: list[DailyRecord]
) -> tuple[Path, Path]:
    folder = portfolio_dir(portfolio_id)
    t_path = write_csv_atomic(trades_frame(trades), folder / "trades.csv")
    d_path = write_csv_atomic(daily_frame(records), folder / "daily.csv")
    return t_path, d_path


def read_trades(portfolio_id: str) -> pd.DataFrame:
    path = portfolios_dir() / portfolio_id / "trades.csv"
    if not path.exists():
        return pd.DataFrame(columns=TRADE_COLUMNS)
    df = read_csv(path, keep_default_na=False, na_values=[""])
    if len(df):
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def read_daily(portfolio_id: str) -> pd.DataFrame:
    path = portfolios_dir() / portfolio_id / "daily.csv"
    if not path.exists():
        return pd.DataFrame(columns=DAILY_COLUMNS)
    df = read_csv(path, keep_default_na=False, na_values=[""])
    if len(df):
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["flags"] = df["flags"].fillna("").astype(str)
    return df


def records_hash(portfolio_id: str) -> dict[str, str]:
    folder = portfolios_dir() / portfolio_id
    return {
        name: sha256_file(folder / name)
        for name in ("trades.csv", "daily.csv")
        if (folder / name).exists()
    }


def as_date(value: object) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])
