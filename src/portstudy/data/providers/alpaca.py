"""Alpaca Market Data adapters (daily bars, corporate actions, news) over plain HTTP.

Written against the Alpaca Market Data API v2 docs (https://docs.alpaca.markets/reference).
Every endpoint here is marked UNVERIFIED until the PLAN2 phase 15 E smoke test with a real key.
Keys: ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from portstudy.data.providers.base import (
    ACTION_COLUMNS,
    NEWS_COLUMNS,
    PRICE_COLUMNS,
    NewsProvider,
    PriceProvider,
    empty_frame,
)
from portstudy.data.providers.http import HttpClient, env_key

DATA_URL = "https://data.alpaca.markets"


def _headers(key_id: str | None, secret: str | None) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": key_id or env_key("ALPACA_API_KEY_ID"),
        "APCA-API-SECRET-KEY": secret or env_key("ALPACA_API_SECRET_KEY"),
        "accept": "application/json",
    }


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


class AlpacaPriceProvider(PriceProvider):
    """# UNVERIFIED: needs real-key smoke test (PLAN2 phase 15 E)."""

    name = "alpaca"

    def __init__(
        self,
        http: HttpClient,
        key_id: str | None = None,
        secret: str | None = None,
        feed: str = "iex",
        symbols_per_call: int = 100,
    ) -> None:
        self.http = http
        self.headers = _headers(key_id, secret)
        self.feed = feed
        self.symbols_per_call = symbols_per_call

    def _bars(
        self, tickers: list[str], start: dt.date, end: dt.date, adjustment: str
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for chunk in _chunks(tickers, self.symbols_per_call):
            page: str | None = None
            while True:
                params: dict[str, Any] = {
                    "symbols": ",".join(chunk),
                    "timeframe": "1Day",
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "adjustment": adjustment,
                    "feed": self.feed,
                    "limit": 10000,
                    "sort": "asc",
                }
                if page:
                    params["page_token"] = page
                payload = self.http.get_json(f"{DATA_URL}/v2/stocks/bars", params, self.headers)
                for symbol, bars in (payload.get("bars") or {}).items():
                    for b in bars or []:
                        rows.append(
                            {
                                "date": dt.datetime.fromisoformat(
                                    b["t"].replace("Z", "+00:00")
                                ).date(),
                                "ticker": symbol,
                                "open": float(b["o"]),
                                "high": float(b["h"]),
                                "low": float(b["l"]),
                                "close": float(b["c"]),
                                "volume": float(b["v"]),
                            }
                        )
                page = payload.get("next_page_token")
                if not page:
                    break
        return pd.DataFrame(
            rows, columns=["date", "ticker", "open", "high", "low", "close", "volume"]
        )

    def get_prices(self, tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
        # Raw bars for the ledger and the simulator; fully adjusted close for the models.
        raw = self._bars(tickers, start, end, adjustment="raw")
        adj = self._bars(tickers, start, end, adjustment="all")
        if len(raw) == 0:
            return empty_frame(PRICE_COLUMNS)
        merged = raw.merge(
            adj[["date", "ticker", "close"]].rename(columns={"close": "adj_close"}),
            on=["date", "ticker"],
            how="left",
        )
        merged["adj_close"] = merged["adj_close"].fillna(merged["close"])
        return merged.loc[:, PRICE_COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)

    def get_corporate_actions(
        self, tickers: list[str], start: dt.date, end: dt.date
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for chunk in _chunks(tickers, self.symbols_per_call):
            params = {
                "symbols": ",".join(chunk),
                "types": "forward_split,reverse_split,cash_dividend",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": 1000,
            }
            payload = self.http.get_json(f"{DATA_URL}/v1/corporate-actions", params, self.headers)
            actions = payload.get("corporate_actions") or {}
            for s in actions.get("forward_splits", []) + actions.get("reverse_splits", []):
                ratio = float(s["new_rate"]) / float(s["old_rate"])
                rows.append(
                    {
                        "date": dt.date.fromisoformat(s["ex_date"]),
                        "ticker": s["symbol"],
                        "action": "split",
                        "value": ratio,
                    }
                )
            for d in actions.get("cash_dividends", []):
                rows.append(
                    {
                        "date": dt.date.fromisoformat(d["ex_date"]),
                        "ticker": d["symbol"],
                        "action": "dividend",
                        "value": float(d["rate"]),
                    }
                )
        return pd.DataFrame(rows, columns=ACTION_COLUMNS)


class AlpacaNewsProvider(NewsProvider):
    """# UNVERIFIED: needs real-key smoke test (PLAN2 phase 15 E)."""

    name = "alpaca"

    def __init__(
        self,
        http: HttpClient,
        key_id: str | None = None,
        secret: str | None = None,
        symbols_per_call: int = 50,
    ) -> None:
        self.http = http
        self.headers = _headers(key_id, secret)
        self.symbols_per_call = symbols_per_call

    def get_news(
        self, tickers: list[str], start_utc: dt.datetime, end_utc: dt.datetime
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        wanted = set(tickers)
        for chunk in _chunks(tickers, self.symbols_per_call):
            page: str | None = None
            while True:
                params: dict[str, Any] = {
                    "symbols": ",".join(chunk),
                    "start": start_utc.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end": end_utc.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "limit": 50,
                    "sort": "desc",
                    "include_content": "false",
                }
                if page:
                    params["page_token"] = page
                payload = self.http.get_json(f"{DATA_URL}/v1beta1/news", params, self.headers)
                for item in payload.get("news") or []:
                    published = dt.datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
                    for symbol in item.get("symbols") or []:
                        if symbol in wanted:
                            rows.append(
                                {
                                    "ticker": symbol,
                                    "published_at": published,
                                    "headline": item.get("headline", ""),
                                    "summary": item.get("summary", "") or "",
                                    "source": item.get("source", "alpaca"),
                                }
                            )
                page = payload.get("next_page_token")
                if not page:
                    break
        return pd.DataFrame(rows, columns=NEWS_COLUMNS)
