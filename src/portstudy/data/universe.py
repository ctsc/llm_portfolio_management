"""Sector balanced stock list drawn from the S&P 500. DECISION(B2), DECISION(B4)."""

from __future__ import annotations

import datetime as dt
import logging
import math
from pathlib import Path

import pandas as pd

from portstudy import calendar as cal
from portstudy.config import StudyConfig, repo_root
from portstudy.data.providers.base import ProviderBundle
from portstudy.io import read_csv, write_csv_atomic
from portstudy.paths import synthetic_dir

UNIVERSE_COLUMNS = ["ticker", "name", "sector", "rank_in_sector", "avg_dollar_volume", "version"]


def universe_path(cfg: StudyConfig, synthetic: bool = False) -> Path:
    """The frozen universe file: config/universe.csv, or the synthetic copy under data/."""
    if synthetic:
        return synthetic_dir() / "universe.csv"
    return repo_root() / cfg.universe.universe_file


def load_universe(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    missing = [c for c in UNIVERSE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"universe file {path} is missing columns {missing}")
    return df


def build_universe(
    as_of: dt.date,
    cfg: StudyConfig,
    providers: ProviderBundle,
    out_path: Path | None = None,
    version: str = "v1",
    print_table: bool = True,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Rank members within each GICS sector by 60-day average dollar volume and pick the top
    floor(size / sectors) per sector, then the highest-volume stocks left over.

    Stocks with fewer than `min_history_days` trading days before `as_of` are dropped first.
    `universe.extra_tickers` each replace the lowest-volume pick in their sector. Sectors that
    cannot fill their quota and the leftover slots used to reach `size` are logged.
    """
    log = logger or logging.getLogger("portstudy.universe")
    ucfg = cfg.universe
    members = providers.members.get_members(as_of)
    tickers = sorted(members["ticker"].astype(str))

    # Pull enough history to count trading days and compute dollar volume.
    start = as_of - dt.timedelta(days=int(ucfg.min_history_days * 1.6) + 30)
    prices = providers.prices.get_prices(tickers, start, as_of)
    prices = prices.drop_duplicates(["ticker", "date"])
    counts = prices.groupby("ticker").size()
    eligible = set(counts[counts >= ucfg.min_history_days].index)

    window_days = cal.trading_days(start, as_of)[-ucfg.dollar_volume_window_days :]
    recent = prices[prices["date"].isin(set(window_days)) & prices["ticker"].isin(eligible)]
    dollar = (recent["close"] * recent["volume"]).groupby(recent["ticker"]).mean()

    table = members[members["ticker"].isin(dollar.index)].copy()
    table["avg_dollar_volume"] = table["ticker"].map(dollar)
    table = table.sort_values(
        ["sector", "avg_dollar_volume", "ticker"], ascending=[True, False, True], kind="stable"
    )
    table["rank_in_sector"] = table.groupby("sector").cumcount() + 1

    per_sector = math.floor(ucfg.size / ucfg.sectors)
    available = table.groupby("sector").size()
    for sector in sorted(set(members["sector"].astype(str))):
        count = int(available.get(sector, 0))
        if count < per_sector:
            log.info(
                "universe_thin_sector",
                extra={
                    "extra_fields": {
                        "sector": sector,
                        "quota": per_sector,
                        "available": count,
                        "short": per_sector - count,
                    }
                },
            )
    picked = table[table["rank_in_sector"] <= per_sector].copy()
    leftover = table[table["rank_in_sector"] > per_sector]
    n_extra = ucfg.size - len(picked)
    if n_extra > 0 and len(leftover):
        top_left = leftover.sort_values(
            ["avg_dollar_volume", "ticker"], ascending=[False, True]
        ).head(n_extra)
        picked = pd.concat([picked, top_left])
        log.info(
            "universe_leftover_slots",
            extra={"extra_fields": {"slots": n_extra, "filled": len(top_left)}},
        )
    if len(picked) < ucfg.size:
        log.warning(
            "universe_short",
            extra={"extra_fields": {"size": ucfg.size, "picked": len(picked)}},
        )

    # Researcher additions replace the lowest-volume pick in the same sector.
    if ucfg.extra_tickers:
        ref = providers.reference.get_reference(list(ucfg.extra_tickers), as_of)
        for row in ref.itertuples(index=False):
            same = picked[picked["sector"] == row.sector]
            if len(same):
                drop_idx = same.sort_values("avg_dollar_volume").index[0]
                picked = picked.drop(index=drop_idx)
            picked = pd.concat(
                [
                    picked,
                    pd.DataFrame(
                        [
                            {
                                "ticker": row.ticker,
                                "name": row.name,
                                "sector": row.sector,
                                "avg_dollar_volume": float("nan"),
                                "rank_in_sector": 0,
                            }
                        ]
                    ),
                ]
            )

    picked = picked.sort_values(["sector", "rank_in_sector", "ticker"], kind="stable")
    picked["version"] = version
    out = picked.loc[:, UNIVERSE_COLUMNS].reset_index(drop=True)

    if print_table:
        counts_by_sector = out.groupby("sector").size().rename("count")
        print(counts_by_sector.to_string())
        print(f"total: {len(out)}")

    if out_path is not None:
        write_csv_atomic(out, out_path)
    return out
