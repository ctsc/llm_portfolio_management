"""Phase 3A data tests: U1 large-move flag, crash removal, unexplained splits, the removals
file, snapshot integration, and the ledger rebuild."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from portstudy import calendar as cal
from portstudy.config import StudyConfig
from portstudy.data import ledger
from portstudy.data.checks import check_prices, tickers_to_skip
from portstudy.data.providers.base import ProviderBundle
from portstudy.data.snapshot import make_snapshot, read_snapshot
from portstudy.data.universe import (
    read_removals,
    record_removal,
    removed_tickers,
)


def _frame(
    days: list[dt.date],
    adj_closes: list[float],
    raw_closes: list[float] | None = None,
    ticker: str = "X",
) -> pd.DataFrame:
    raw = raw_closes if raw_closes is not None else adj_closes
    return pd.DataFrame(
        {
            "date": days,
            "ticker": ticker,
            "open": raw,
            "high": raw,
            "low": raw,
            "close": raw,
            "adj_close": adj_closes,
            "volume": 1000.0,
        }
    )


def _check(frame: pd.DataFrame, cfg: StudyConfig, since: dt.date, cutoff: dt.date):
    days = [d for d in cal.trading_days(since, cutoff)]
    return check_prices(
        frame,
        days,
        cfg.data.checks,
        pd.DataFrame(columns=["date", "ticker", "action", "value"]),
        min_history_days=1,
        since=since,
    )


def _move_issues(issues):
    return [i for i in issues if i.code in ("large_move", "crash_removal", "unexplained_split")]


def test_large_move_flags_and_keeps(study_config: StudyConfig) -> None:
    # +12% adjusted move over the week: informational flag only, stock kept.
    since = dt.date(2026, 3, 6)
    days = cal.trading_days(since + dt.timedelta(days=1), dt.date(2026, 3, 13))
    frame = _frame([since] + days, [100.0] + [112.0 + i * 0.01 for i in range(len(days))])
    issues = _check(frame, study_config, since, dt.date(2026, 3, 13))
    large = [i for i in issues if i.code == "large_move"]
    assert len(large) == 1 and large[0].severity == "info" and large[0].rule == "flag"
    assert large[0].details["return"] >= 0.10
    assert "X" not in tickers_to_skip(_move_issues(issues))


def test_crash_removal_removes(study_config: StudyConfig) -> None:
    since = dt.date(2026, 3, 6)
    days = cal.trading_days(since + dt.timedelta(days=1), dt.date(2026, 3, 13))
    adj = [100.0, 100.0, 55.0, 54.9, 55.1, 55.0]
    frame = _frame([since] + days, adj)
    issues = _check(frame, study_config, since, dt.date(2026, 3, 13))
    crash = [i for i in issues if i.code == "crash_removal"]
    assert len(crash) == 1 and crash[0].rule == "remove" and crash[0].severity == "error"
    assert crash[0].details["return"] <= -0.40
    assert crash[0].details["date"] in (days[1].isoformat(), days[-1].isoformat())
    assert "X" in tickers_to_skip(_move_issues(issues))


def test_unexplained_split_skips_one_week(study_config: StudyConfig) -> None:
    # Raw price doubles with no split action and flat adjusted prices.
    since = dt.date(2026, 3, 6)
    days = cal.trading_days(since + dt.timedelta(days=1), dt.date(2026, 3, 13))
    frame = _frame(
        [since] + days,
        [100.0 + i * 0.01 for i in range(6)],
        raw_closes=[100.0, 100.0, 200.0, 200.1, 199.9, 200.0],
    )
    issues = _check(frame, study_config, since, dt.date(2026, 3, 13))
    us = [i for i in issues if i.code == "unexplained_split"]
    assert len(us) == 1 and us[0].severity == "error"
    assert "X" in tickers_to_skip(
        _move_issues(issues)
    )  # skipped this week only; nothing is recorded


def test_move_older_than_since_raises_nothing(study_config: StudyConfig) -> None:
    # A 40% crash 200 days before the window raises nothing under the `since` scope.
    since = dt.date(2026, 3, 6)
    days = cal.trading_days(since + dt.timedelta(days=1), dt.date(2026, 3, 13))
    old_days = cal.trading_days(since - dt.timedelta(days=290), since)
    adj_old = [100.0, 60.0] + [60.0] * (len(old_days) - 2)
    frame = _frame(old_days + days, adj_old + [60.0 + i * 0.01 for i in range(len(days))])
    issues = _check(frame, study_config, since, dt.date(2026, 3, 13))
    assert not any(i.code in ("crash_removal", "large_move") for i in issues)


def test_real_split_raises_nothing(study_config: StudyConfig) -> None:
    # Raw jump explained by a split action and flat adjusted prices.
    since = dt.date(2026, 3, 6)
    days = cal.trading_days(since + dt.timedelta(days=1), dt.date(2026, 3, 13))
    frame = _frame(
        [since] + days,
        [100.0 + i * 0.01 for i in range(6)],
        raw_closes=[100.0, 100.0, 50.0, 50.01, 49.99, 50.0],
    )
    actions = pd.DataFrame([{"date": days[1], "ticker": "X", "action": "split", "value": 2.0}])
    issues = check_prices(
        frame,
        days,
        study_config.data.checks,
        actions,
        since=since,
    )
    assert not any(i.code in ("crash_removal", "unexplained_split") for i in issues)


# Removals file and snapshot integration ----------------------------------------------------


def test_removals_file_first_wins(data_dir: Path) -> None:
    record_removal("2026-W10", "SYN011", dt.date(2026, 3, 10), -0.45, "crash_removal")
    record_removal("2026-W11", "SYN011", dt.date(2026, 3, 17), -0.50, "crash_removal")
    rem = read_removals()
    assert len(rem) == 1 and rem.iloc[0]["week_id"] == "2026-W10"
    assert removed_tickers() == {"SYN011"}
    assert removed_tickers(through_week="2026-W10") == {"SYN011"}
    assert removed_tickers(through_week="2026-W09") == set()


def test_planted_crash_removes_stock_in_next_snapshot(
    study_config: StudyConfig,
    providers: ProviderBundle,
    universe: pd.DataFrame,
    world,
    data_dir: Path,
) -> None:
    from portstudy.data.providers.synthetic import PLANTED

    tk, off, _ = PLANTED["extreme_move"]
    if tk not in set(universe["ticker"]):
        universe = pd.concat(
            [
                universe,
                pd.DataFrame([{"ticker": tk, "name": tk, "sector": "tech", "cap_bucket": "large"}]),
            ],
            ignore_index=True,
        )
    # Week whose cutoff covers the planted crash day.
    first_cutoff = cal.cutoff_day(study_config.synthetic.first_week)
    days = cal.trading_days(
        first_cutoff - dt.timedelta(days=500), first_cutoff + dt.timedelta(days=300)
    )
    crash_day = days[days.index(first_cutoff) + off]
    crash_week = cal.week_id(crash_day)
    next_week = cal.next_week_id(crash_week)
    for w in cal.week_range(study_config.synthetic.first_week, 100):
        if w > crash_week:
            break
        make_snapshot(w, study_config, providers, universe)
    ref = make_snapshot(next_week, study_config, providers, universe)

    rem = read_removals()
    assert tk in set(rem["ticker"])
    snap_uni = pd.read_csv(ref.path / "universe.csv")
    assert not bool(snap_uni.loc[snap_uni["ticker"] == tk, "included"].iloc[0])
    # The removal week excluded it too (crash week manifest names it).
    prev = read_snapshot(crash_week)
    assert tk in prev.manifest["removed"] or tk in prev.manifest.get("skipped", [])


def test_ledger_rebuild_archives_and_refetches(
    study_config: StudyConfig, providers: ProviderBundle, universe: pd.DataFrame, data_dir: Path
) -> None:
    start = cal.cutoff_day(study_config.synthetic.first_week) - dt.timedelta(days=60)
    end = start + dt.timedelta(days=30)
    tickers = sorted(universe["ticker"].astype(str))[:5]
    prices = providers.prices.get_prices(tickers, start, end)
    ledger.append_prices(prices)
    before = ledger.read_ledger()

    archive, n = ledger.rebuild_ledger(study_config, providers, tickers, start, end)
    assert (archive / "prices_raw.csv").exists()
    fresh = ledger.read_ledger()
    # Rebuilt ledger equals a fresh fetch for the same window (plus ETF proxies).
    etfs = list(study_config.universe.etf_proxies)
    etf_prices = providers.prices.get_prices(etfs, start, end)
    expected = pd.concat([prices, etf_prices], ignore_index=True)
    assert len(fresh) == len(expected)
    assert set(fresh["ticker"]) == set(expected["ticker"])
    assert n >= len(before)
