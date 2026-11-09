"""FRED adapter for the 3-month T-bill series DTB3.

Written against https://fred.stlouisfed.org/docs/api/fred/series_observations.html
# UNVERIFIED: needs real-key smoke test (PLAN2 phase 15 E). Key: FRED_API_KEY.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from portstudy.data.providers.base import RATE_COLUMNS, RateProvider
from portstudy.data.providers.http import HttpClient, env_key

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


class FredRateProvider(RateProvider):
    name = "fred"

    def __init__(
        self, http: HttpClient, api_key: str | None = None, series_id: str = "DTB3"
    ) -> None:
        self.http = http
        self.api_key = api_key or env_key("FRED_API_KEY")
        self.series_id = series_id

    def get_rates(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        params: dict[str, Any] = {
            "series_id": self.series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
        }
        payload = self.http.get_json(FRED_URL, params)
        rows = []
        for obs in payload.get("observations", []):
            value = obs.get("value", ".")
            if value in (".", "", None):
                continue
            rows.append({"date": dt.date.fromisoformat(obs["date"]), "rate": float(value)})
        return pd.DataFrame(rows, columns=RATE_COLUMNS)
