"""Phase 2 done-when: 30 consecutive weekly snapshots from the synthetic world."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from portstudy import calendar as cal
from portstudy.config import StudyConfig
from portstudy.data import ledger
from portstudy.data.providers.base import ProviderBundle
from portstudy.data.providers.synthetic import PLANTED
from portstudy.data.snapshot import make_snapshot, read_snapshot

# Codes the planted world is expected to raise. Anything else is unexpected.
EXPECTED_CODES = {
    "missing_days", "nonpositive_price", "extreme_move", "duplicate_rows", "stale_price",
    "no_news", "news_after_cutoff", "short_history", "stopped_trading", "no_prices",
}
PLANTED_TICKERS = {v[0] for v in PLANTED.values()}


def test_thirty_consecutive_snapshots(data_dir: Path, study_config: StudyConfig,
                                      providers: ProviderBundle, universe: pd.DataFrame) -> None:
    weeks = cal.week_range(study_config.synthetic.first_week, 30)
    for w in weeks:
        ref = make_snapshot(w, study_config, providers, universe)
        snap = read_snapshot(w)
        assert snap.prices["date"].max() == ref.cutoff_date
        issues = snap.issues
        unexpected = issues[~issues["code"].isin(EXPECTED_CODES)]
        assert unexpected.empty, unexpected
        # Only planted tickers may be skipped; no_news is info only and never skips.
        skipped = set(issues.loc[issues["rule"] == "skip_stock", "ticker"])
        assert skipped <= PLANTED_TICKERS, skipped - PLANTED_TICKERS
        assert len(snap.included_tickers) >= 95
    # The ledger holds one row per (date, ticker) and never duplicates across weeks.
    led = ledger.read_ledger()
    assert not led.duplicated(["date", "ticker"]).any()
    assert led["date"].max() == cal.cutoff_day(weeks[-1])
