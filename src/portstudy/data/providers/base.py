"""Provider interfaces. Every provider returns plain DataFrames with fixed column names.

prices:    date, ticker, open, high, low, close, adj_close, volume
actions:   date, ticker, action (split | dividend), value (ratio or cash per share)
news:      ticker, published_at (UTC), headline, summary, source
reference: ticker, name, sector, market_cap, next_earnings_date
rates:     date, rate
members:   ticker, name, sector
"""

from __future__ import annotations

import abc
import datetime as dt
from dataclasses import dataclass

import pandas as pd

PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
ACTION_COLUMNS = ["date", "ticker", "action", "value"]
NEWS_COLUMNS = ["ticker", "published_at", "headline", "summary", "source"]
REFERENCE_COLUMNS = ["ticker", "name", "sector", "market_cap", "next_earnings_date"]
RATE_COLUMNS = ["date", "rate"]
MEMBER_COLUMNS = ["ticker", "name", "sector"]


def empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


class PriceProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def get_prices(self, tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
        """Daily bars in [start, end] with PRICE_COLUMNS."""

    def get_corporate_actions(
        self, tickers: list[str], start: dt.date, end: dt.date
    ) -> pd.DataFrame:
        """Splits and dividends in [start, end] with ACTION_COLUMNS. Default: none."""
        return empty_frame(ACTION_COLUMNS)


class NewsProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def get_news(
        self, tickers: list[str], start_utc: dt.datetime, end_utc: dt.datetime
    ) -> pd.DataFrame:
        """Articles published in (start_utc, end_utc] with NEWS_COLUMNS."""


class ReferenceProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def get_reference(self, tickers: list[str], as_of: dt.date) -> pd.DataFrame:
        """Sectors, names, market caps, next earnings dates with REFERENCE_COLUMNS."""


class RateProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def get_rates(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Daily T-bill rate (annualized, percent) with RATE_COLUMNS."""


class MembersProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def get_members(self, as_of: dt.date) -> pd.DataFrame:
        """Current index members with MEMBER_COLUMNS."""


@dataclass
class ProviderBundle:
    prices: PriceProvider
    news: NewsProvider
    reference: ReferenceProvider
    rates: RateProvider
    members: MembersProvider

    def names(self) -> dict[str, str]:
        return {
            "prices": self.prices.name,
            "news": self.news.name,
            "reference": self.reference.name,
            "rates": self.rates.name,
            "members": self.members.name,
        }


def ensure_columns(df: pd.DataFrame, columns: list[str], what: str) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{what} frame is missing columns {missing}")
    return df.loc[:, columns].copy()
