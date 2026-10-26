"""Finnhub adapter for earnings calendar dates.

Written against https://finnhub.io/docs/api/earnings-calendar
# UNVERIFIED: needs real-key smoke test (Phase 17). Key: FINNHUB_API_KEY.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from portstudy.data.providers.http import HttpClient, env_key

FINNHUB_URL = "https://finnhub.io/api/v1/calendar/earnings"


class FinnhubEarningsProvider:
    """Returns a frame `ticker, date` of scheduled or reported earnings dates."""

    name = "finnhub"

    def __init__(self, http: HttpClient, api_key: str | None = None) -> None:
        self.http = http
        self.api_key = api_key or env_key("FINNHUB_API_KEY")

    def get_earnings(self, tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for tk in tickers:
            params = {
                "from": start.isoformat(),
                "to": end.isoformat(),
                "symbol": tk,
                "token": self.api_key,
            }
            payload = self.http.get_json(FINNHUB_URL, params)
            for item in payload.get("earningsCalendar", []):
                rows.append(
                    {"ticker": item.get("symbol", tk), "date": dt.date.fromisoformat(item["date"])}
                )
        return pd.DataFrame(rows, columns=["ticker", "date"])

    def next_earnings(
        self, tickers: list[str], as_of: dt.date, horizon_days: int = 120
    ) -> pd.Series:
        frame = self.get_earnings(tickers, as_of, as_of + dt.timedelta(days=horizon_days))
        if len(frame) == 0:
            return pd.Series(index=tickers, dtype="object")
        frame = frame[frame["date"] > as_of]
        nxt = frame.groupby("ticker")["date"].min()
        return nxt.reindex(tickers)
