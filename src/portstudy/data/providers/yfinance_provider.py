"""yfinance adapters: prices (fallback), market caps and sectors (reference).

yfinance has no HTTP layer to swap, so the provider takes injectable fetch callables. In
fixture mode the callables read recorded JSON. The yfinance import is optional.
# UNVERIFIED: needs real smoke test (Phase 17). No key required.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from portstudy.data.providers.base import (
    PRICE_COLUMNS,
    REFERENCE_COLUMNS,
    PriceProvider,
    ReferenceProvider,
    empty_frame,
)
from portstudy.hashing import stable_hash
from portstudy.io import read_json, write_json_atomic

HistoryFetch = Callable[[str, dt.date, dt.date], list[dict[str, Any]]]
InfoFetch = Callable[[str], dict[str, Any]]


def _yf_history(ticker: str, start: dt.date, end: dt.date) -> list[dict[str, Any]]:
    import yfinance as yf  # type: ignore[import-not-found]

    hist = yf.Ticker(ticker).history(
        start=start.isoformat(),
        end=(end + dt.timedelta(days=1)).isoformat(),
        auto_adjust=False,
        actions=True,
    )
    rows = []
    for idx, r in hist.iterrows():
        rows.append(
            {
                "date": idx.date().isoformat(),
                "open": float(r["Open"]),
                "high": float(r["High"]),
                "low": float(r["Low"]),
                "close": float(r["Close"]),
                "adj_close": float(r.get("Adj Close", r["Close"])),
                "volume": float(r["Volume"]),
                "dividends": float(r.get("Dividends", 0.0)),
                "splits": float(r.get("Stock Splits", 0.0)),
            }
        )
    return rows


def _yf_info(ticker: str) -> dict[str, Any]:
    import yfinance as yf  # type: ignore[import-not-found]

    info = yf.Ticker(ticker).info or {}
    return {
        "name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "market_cap": info.get("marketCap"),
    }


class FixtureFetch:
    """Replays recorded yfinance payloads from a folder; records them when `live` is set."""

    def __init__(self, folder: Path, live: bool = False) -> None:
        self.folder = Path(folder)
        self.live = live

    def _get(self, key: dict[str, Any], producer: Callable[[], Any]) -> Any:
        path = self.folder / f"yf_{stable_hash(key)[:24]}.json"
        if path.exists():
            return read_json(path)
        if not self.live:
            raise FileNotFoundError(f"no yfinance fixture for {key}")
        payload = producer()
        write_json_atomic(payload, path)
        return payload

    def history(self, ticker: str, start: dt.date, end: dt.date) -> list[dict[str, Any]]:
        key = {
            "kind": "history",
            "ticker": ticker,
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        return list(self._get(key, lambda: _yf_history(ticker, start, end)))

    def info(self, ticker: str) -> dict[str, Any]:
        return dict(self._get({"kind": "info", "ticker": ticker}, lambda: _yf_info(ticker)))


class YFinancePriceProvider(PriceProvider):
    name = "yfinance"

    def __init__(self, history: HistoryFetch = _yf_history) -> None:
        self.history = history

    def _rows(self, tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
        frames = []
        for tk in tickers:
            rows = self.history(tk, start, end)
            if rows:
                df = pd.DataFrame(rows)
                df["ticker"] = tk
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"]).dt.date
        return out

    def get_prices(self, tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
        df = self._rows(tickers, start, end)
        if len(df) == 0:
            return empty_frame(PRICE_COLUMNS)
        return df.loc[:, PRICE_COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)

    def get_corporate_actions(
        self, tickers: list[str], start: dt.date, end: dt.date
    ) -> pd.DataFrame:
        df = self._rows(tickers, start, end)
        rows = []
        if len(df):
            for rec in df.to_dict("records"):
                div = float(rec.get("dividends", 0.0) or 0.0)
                split = float(rec.get("splits", 0.0) or 0.0)
                if div > 0:
                    rows.append(
                        {
                            "date": rec["date"],
                            "ticker": rec["ticker"],
                            "action": "dividend",
                            "value": div,
                        }
                    )
                if split > 0:
                    rows.append(
                        {
                            "date": rec["date"],
                            "ticker": rec["ticker"],
                            "action": "split",
                            "value": split,
                        }
                    )
        return pd.DataFrame(rows, columns=["date", "ticker", "action", "value"])


class YFinanceReferenceProvider(ReferenceProvider):
    """Names, sectors, and market caps from yfinance; earnings dates from an optional provider."""

    name = "yfinance"

    def __init__(
        self, info: InfoFetch = _yf_info, earnings: Any = None, members: pd.DataFrame | None = None
    ) -> None:
        self.info = info
        self.earnings = earnings
        self.members = members

    def get_reference(self, tickers: list[str], as_of: dt.date) -> pd.DataFrame:
        rows = []
        member_sectors = (
            dict(zip(self.members["ticker"], self.members["sector"]))
            if self.members is not None
            else {}
        )
        member_names = (
            dict(zip(self.members["ticker"], self.members["name"]))
            if self.members is not None
            else {}
        )
        nxt = (
            self.earnings.next_earnings(tickers, as_of)
            if self.earnings is not None
            else pd.Series(index=tickers, dtype="object")
        )
        for tk in tickers:
            info = self.info(tk)
            rows.append(
                {
                    "ticker": tk,
                    "name": member_names.get(tk) or info.get("name") or tk,
                    "sector": member_sectors.get(tk) or info.get("sector"),
                    "market_cap": info.get("market_cap"),
                    "next_earnings_date": nxt.get(tk) if nxt is not None else None,
                }
            )
        return pd.DataFrame(rows, columns=REFERENCE_COLUMNS)
