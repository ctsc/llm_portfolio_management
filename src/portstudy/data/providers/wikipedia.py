"""S&P 500 member list from the Wikipedia constituents table (ticker, name, GICS sector).

Parsed with the standard library HTML parser so no extra dependency is needed.
# UNVERIFIED: needs real smoke test (PLAN2 phase 15 E). No key required.
"""

from __future__ import annotations

import datetime as dt
from html.parser import HTMLParser

import pandas as pd

from portstudy.data.providers.base import MEMBER_COLUMNS, MembersProvider
from portstudy.data.providers.http import HttpClient

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


class _TableParser(HTMLParser):
    """Collects rows of the table whose id is `constituents`."""

    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.depth = 0
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table" and dict(attrs).get("id") == "constituents":
            self.in_table = True
            self.depth = 0
        if not self.in_table:
            return
        if tag == "table":
            self.depth += 1
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if not self.in_table:
            return
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None
        elif tag == "table":
            self.depth -= 1
            if self.depth <= 0:
                self.in_table = False

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def parse_constituents(html: str) -> pd.DataFrame:
    parser = _TableParser()
    parser.feed(html)
    if not parser.rows:
        raise ValueError("constituents table not found")
    header = [h.lower() for h in parser.rows[0]]
    sym_i = next(i for i, h in enumerate(header) if h.startswith("symbol"))
    name_i = next(i for i, h in enumerate(header) if h.startswith("security"))
    sector_i = next(i for i, h in enumerate(header) if "gics sector" in h)
    rows = []
    for r in parser.rows[1:]:
        if len(r) <= max(sym_i, name_i, sector_i):
            continue
        # Wikipedia uses dots for share classes (BRK.B); Alpaca and yfinance differ, so keep
        # the Wikipedia form and let the price provider map it if needed.
        rows.append(
            {"ticker": r[sym_i].strip(), "name": r[name_i].strip(), "sector": r[sector_i].strip()}
        )
    return pd.DataFrame(rows, columns=MEMBER_COLUMNS)


class WikipediaMembersProvider(MembersProvider):
    name = "wikipedia"

    def __init__(self, http: HttpClient) -> None:
        self.http = http

    def get_members(self, as_of: dt.date) -> pd.DataFrame:
        html = self.http.get_text(WIKI_URL, headers={"User-Agent": "portstudy/0.1 (research)"})
        return parse_constituents(html)
