"""Weekly rank IC, the IC history store, and the trailing IC used to scale views.

Rank IC: Spearman(score at the Friday cutoff, total return from the next Monday open to the
following Monday open). Trailing IC: mean of the last `ic_window_weeks` scored weeks whose
return window closed before the cutoff, floored at `ic_floor`; fixed `cold_start_ic` for the
first `cold_start_weeks` live weeks and whenever fewer than `ic_min_weeks` clean weeks exist.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from portstudy import calendar as cal
from portstudy.config import ViewsConfig
from portstudy.io import read_csv, write_csv_atomic
from portstudy.paths import outputs_dir

IC_COLUMNS = [
    "week_id",
    "trade_date",
    "window_end_date",
    "signal",
    "model",
    "config_id",
    "rank_ic",
    "n_stocks",
]
SIGNALS = ("baseline", "llm_raw", "llm_resid", "combined")
MIN_STOCKS_FOR_IC = 10


def weekly_returns(
    ledger: pd.DataFrame, week_id: str, actions: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Total return per stock from the trade-date open to the window-end open (raw prices).

    Dividends with an ex-date inside (trade_date, window_end_date] are added back; splits
    inside the window rescale the end price so the return is in start-date share terms.
    Returns a frame `ticker, week_id, trade_date, window_end_date, ret` for stocks that have
    both opens.
    """
    trade_date = cal.monday_open_day(week_id)
    end_date = cal.window_end_date(week_id)
    led = ledger
    start = led[led["date"] == trade_date].set_index("ticker")["open"].astype(float)
    end = led[led["date"] == end_date].set_index("ticker")["open"].astype(float)
    common = start.index.intersection(end.index)
    adj_end = end.loc[common].copy()
    extra = pd.Series(0.0, index=common)
    if actions is not None and len(actions):
        inside = actions[(actions["date"] > trade_date) & (actions["date"] <= end_date)]
        for tk in common:
            ratio = 1.0
            rows = inside[inside["ticker"] == tk].sort_values("date")
            for action, value in zip(rows["action"], rows["value"]):
                if action == "split":
                    ratio *= float(value)
                elif action == "dividend":
                    extra[tk] += float(value) * ratio
            adj_end[tk] = end[tk] * ratio
    ret = (adj_end + extra) / start.loc[common] - 1.0
    out = pd.DataFrame(
        {
            "ticker": common,
            "week_id": week_id,
            "trade_date": trade_date,
            "window_end_date": end_date,
            "ret": ret.to_numpy(),
        }
    )
    return out.sort_values("ticker").reset_index(drop=True)


def rank_ic(scores: pd.Series, returns: pd.Series) -> float:
    """Spearman correlation with average ranks for ties. NaN if < 10 stocks or zero variance."""
    joined = pd.concat([scores.rename("s"), returns.rename("r")], axis=1).dropna()
    if len(joined) < MIN_STOCKS_FOR_IC:
        return float("nan")
    rs = joined["s"].rank(method="average")
    rr = joined["r"].rank(method="average")
    if rs.std(ddof=1) == 0 or rr.std(ddof=1) == 0:
        return float("nan")
    return float(np.corrcoef(rs.to_numpy(), rr.to_numpy())[0, 1])


class ICStore:
    """CSV table of weekly rank ICs at data/outputs/ic_history.csv."""

    def __init__(self, path: object | None = None) -> None:
        from pathlib import Path

        self.path = Path(str(path)) if path is not None else outputs_dir() / "ic_history.csv"

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=IC_COLUMNS)
        df = read_csv(self.path, keep_default_na=False, na_values=[""])
        df["model"] = df["model"].fillna("").astype(str)
        df["config_id"] = df["config_id"].fillna("").astype(str)
        return df

    def upsert(self, rows: Iterable[dict[str, object]]) -> pd.DataFrame:
        """Insert or replace rows keyed by (week_id, signal, model, config_id). Idempotent."""
        new = pd.DataFrame(list(rows))
        if len(new) == 0:
            return self.read()
        for col in ("model", "config_id"):
            new[col] = new[col].fillna("").astype(str)
        new["trade_date"] = [cal.monday_open_day(w).isoformat() for w in new["week_id"]]
        new["window_end_date"] = [cal.window_end_date(w).isoformat() for w in new["week_id"]]
        old = self.read()
        key = ["week_id", "signal", "model", "config_id"]
        if len(old):
            drop = old.set_index(key).index.isin(new.set_index(key).index)
            old = old[~drop]
        merged = pd.concat([old, new.loc[:, IC_COLUMNS]], ignore_index=True)
        merged = merged.sort_values(key, kind="stable").reset_index(drop=True)
        write_csv_atomic(merged, self.path)
        return merged

    def series(self, signal: str, model: str = "", config_id: str = "") -> pd.DataFrame:
        df = self.read()
        mask = (df["signal"] == signal) & (df["model"] == model) & (df["config_id"] == config_id)
        return df[mask].sort_values("week_id").reset_index(drop=True)


def live_week_number(as_of_week: str, live_start_week: str) -> int:
    """1-based position of `as_of_week` in the live run (0 if before the live start)."""
    if as_of_week < live_start_week:
        return 0
    return len(cal.weeks_between(live_start_week, as_of_week))


def trailing_ic(
    store: ICStore,
    signal: str,
    as_of_week: str,
    cfg: ViewsConfig,
    live_start_week: str,
    model: str = "",
    config_id: str = "",
    fallback_weeks: set[str] | frozenset[str] = frozenset(),
) -> float:
    """DECISION(A6) cold start, look-ahead guard, fallback-week exclusion, floor at ic_floor."""
    if live_week_number(as_of_week, live_start_week) <= cfg.cold_start_weeks:
        return cfg.cold_start_ic
    cutoff = cal.cutoff_day(as_of_week)
    hist = store.series(signal, model, config_id)
    if len(hist) == 0:
        return cfg.cold_start_ic
    closed = pd.to_datetime(hist["window_end_date"]).dt.date <= cutoff
    clean = ~hist["week_id"].isin(set(fallback_weeks))
    before = hist["week_id"] < as_of_week
    usable = hist[closed & clean & before & hist["rank_ic"].notna()]
    usable = usable.tail(cfg.ic_window_weeks)
    if len(usable) < cfg.ic_min_weeks:
        return cfg.cold_start_ic
    mean = float(usable["rank_ic"].astype(float).mean())
    if math.isnan(mean):
        return cfg.cold_start_ic
    return max(cfg.ic_floor, mean)


def window_closed_by(week_id: str, as_of: dt.date) -> bool:
    return cal.window_end_date(week_id) <= as_of
