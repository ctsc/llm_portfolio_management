"""Phase 2 tests: synthetic world, checks, snapshots, universe, ledger, adapters."""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from portstudy import calendar as cal
from portstudy.config import StudyConfig
from portstudy.data import ledger
from portstudy.data.checks import check_news, check_prices
from portstudy.data.providers.alpaca import AlpacaNewsProvider, AlpacaPriceProvider
from portstudy.data.providers.base import ProviderBundle
from portstudy.data.providers.fred import FredRateProvider
from portstudy.data.providers.synthetic import PLANTED, SyntheticWorld, synthetic_bundle
from portstudy.data.providers.wikipedia import parse_constituents
from portstudy.data.snapshot import make_snapshot, read_snapshot
from portstudy.data.universe import build_universe
from portstudy.paths import snapshot_dir

UTC = dt.UTC


# Synthetic world ---------------------------------------------------------------------------


def test_world_is_deterministic(study_config: StudyConfig, world: SyntheticWorld) -> None:
    again = SyntheticWorld.generate(study_config.synthetic, first_week=study_config.synthetic.first_week)
    pd.testing.assert_frame_equal(world.prices, again.prices)
    pd.testing.assert_frame_equal(world.news, again.news)


def test_world_round_trips_to_disk(world: SyntheticWorld, tmp_path: Path) -> None:
    world.save(tmp_path / "w")
    back = SyntheticWorld.load(tmp_path / "w")
    assert back.meta["seed"] == world.meta["seed"]
    assert len(back.prices) == len(world.prices)
    assert len(back.news) == len(world.news)
    assert back.news["published_at"].iloc[0].tzinfo is not None


def test_world_has_planted_sentiment_effect(world: SyntheticWorld) -> None:
    # Realized Monday-to-Monday returns should correlate positively with planted sentiment.
    adj = world.prices[world.prices["ticker"].isin(world.stock_tickers)]
    wide = adj.drop_duplicates(["ticker", "date"]).pivot(index="date", columns="ticker", values="adj_close")
    ics = []
    for w in world.weeks[60:100]:
        t0, t1 = cal.monday_open_day(w), cal.window_end_date(w)
        if t0 not in wide.index or t1 not in wide.index:
            continue
        ret = wide.loc[t1] / wide.loc[t0] - 1
        s = world.sentiment[world.sentiment["week_id"] == w].set_index("ticker")["sentiment"]
        joined = pd.concat([ret.rename("r"), s.rename("s")], axis=1).dropna()
        ics.append(joined["r"].rank().corr(joined["s"].rank()))
    assert sum(ics) / len(ics) > 0.05


# Checks -------------------------------------------------------------------------------------


def test_each_check_fires_on_its_planted_problem(study_config: StudyConfig, world: SyntheticWorld,
                                                 first_cutoff: dt.date) -> None:
    days = cal.trading_days(first_cutoff - dt.timedelta(days=500), first_cutoff + dt.timedelta(days=200))
    idx = days.index(first_cutoff)
    cutoff = days[idx + 70]
    window = days[idx + 70 - 299 : idx + 71]
    prices = world.prices[(world.prices["date"] >= window[0]) & (world.prices["date"] <= cutoff)]
    issues = check_prices(prices, window, study_config.data.checks, world.actions, min_history_days=300)
    codes = {(i.ticker, i.code) for i in issues}
    assert (PLANTED["delisting"][0], "stopped_trading") in codes
    assert (PLANTED["halt"][0], "missing_days") in codes
    assert (PLANTED["zero_price"][0], "nonpositive_price") in codes
    assert (PLANTED["extreme_move"][0], "extreme_move") in codes
    assert (PLANTED["stale_price"][0], "stale_price") in codes
    assert (PLANTED["duplicate_rows"][0], "duplicate_rows") in codes
    # A split is not an extreme move.
    assert (PLANTED["split"][0], "extreme_move") not in codes
    # Short history fires at the first cutoff for the late starter.
    early_window = days[idx - 299 : idx + 1]
    early = world.prices[(world.prices["date"] >= early_window[0]) & (world.prices["date"] <= first_cutoff)]
    early_codes = {(i.ticker, i.code) for i in check_prices(early, early_window, study_config.data.checks)}
    assert (PLANTED["short_history"][0], "short_history") in early_codes
    # No check fires for a clean stock.
    assert not any(t == "SYN000" for t, _ in codes)


def test_news_checks(study_config: StudyConfig, world: SyntheticWorld) -> None:
    w = world.weeks[50]
    cutoff_utc = dt.datetime.combine(cal.cutoff_day(w), dt.time(16, 0),
                                     tzinfo=dt.timezone(dt.timedelta(hours=-5))).astimezone(UTC)
    prev_utc = cutoff_utc - dt.timedelta(days=7)
    bundle = synthetic_bundle(world)
    news = bundle.news.get_news(world.stock_tickers, prev_utc, cutoff_utc + dt.timedelta(hours=12))
    issues = check_news(news, world.stock_tickers, cutoff_utc, study_config.data.checks)
    codes = {i.code for i in issues}
    assert "news_after_cutoff" in codes
    assert "no_news" in codes
    assert all(i.rule == "forward_fill" for i in issues)


# Universe -----------------------------------------------------------------------------------


def test_build_universe_is_balanced_and_deterministic(study_config: StudyConfig, providers: ProviderBundle,
                                                      first_cutoff: dt.date, universe: pd.DataFrame) -> None:
    assert len(universe) == 100
    counts = universe.groupby("sector").size()
    assert len(counts) == 11
    assert set(counts.unique()) <= {9, 10}
    assert counts.sum() == 100
    again = build_universe(first_cutoff, study_config, providers, print_table=False)
    pd.testing.assert_frame_equal(universe, again)
    # The short-history stock is never picked.
    assert PLANTED["short_history"][0] not in set(universe["ticker"])


def test_extra_ticker_replaces_lowest_in_sector(study_config: StudyConfig, providers: ProviderBundle,
                                                first_cutoff: dt.date, universe: pd.DataFrame) -> None:
    # Pick a stock that is not in the universe and ask for it as an extra.
    all_members = providers.members.get_members(first_cutoff)
    extra = sorted(set(all_members["ticker"]) - set(universe["ticker"]) - {PLANTED["short_history"][0]})[0]
    cfg = study_config.model_copy(update={"universe": study_config.universe.model_copy(
        update={"extra_tickers": [extra]})})
    out = build_universe(first_cutoff, cfg, providers, print_table=False)
    assert len(out) == 100
    assert extra in set(out["ticker"])
    assert out.groupby("sector").size().equals(universe.groupby("sector").size())


# Snapshot -----------------------------------------------------------------------------------


def test_snapshot_has_no_rows_after_cutoff(data_dir: Path, study_config: StudyConfig, providers: ProviderBundle,
                                           universe: pd.DataFrame, world: SyntheticWorld) -> None:
    w = world.weeks[85]
    ref = make_snapshot(w, study_config, providers, universe)
    snap = read_snapshot(w)
    assert snap.prices["date"].max() == ref.cutoff_date
    assert snap.etf_prices["date"].max() == ref.cutoff_date
    cutoff_dt = dt.datetime.combine(ref.cutoff_date, dt.time(16, 0),
                                    tzinfo=dt.timezone(dt.timedelta(hours=-5)))
    assert (snap.news["published_at"] <= cutoff_dt).all()
    # The raw world has late articles for this week, so the filter did real work.
    raw = world.news[(pd.to_datetime(world.news["published_at"], utc=True) > cutoff_dt)
                     & (pd.to_datetime(world.news["published_at"], utc=True) <= cutoff_dt + dt.timedelta(hours=8))]
    assert len(raw) > 0
    assert set(snap.news_text["ticker"]) == set(universe["ticker"])
    assert "earnings_in_window" in snap.reference.columns
    assert snap.manifest["trade_date"] == cal.monday_open_day(w).isoformat()
    assert snap.manifest["window_end_date"] == cal.window_end_date(w).isoformat()


def test_snapshot_rerun_is_byte_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                          study_config: StudyConfig, providers: ProviderBundle,
                                          universe: pd.DataFrame, world: SyntheticWorld) -> None:
    w = world.weeks[85]
    hashes = []
    for name in ("a", "b"):
        monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(tmp_path / name))
        make_snapshot(w, study_config, providers, universe)
        manifest = json.loads((snapshot_dir(w) / "manifest.json").read_text(encoding="utf-8"))
        hashes.append(manifest["files"])
    assert hashes[0] == hashes[1]


def test_snapshot_is_immutable_and_versioned(data_dir: Path, study_config: StudyConfig, providers: ProviderBundle,
                                             universe: pd.DataFrame, world: SyntheticWorld) -> None:
    w = world.weeks[85]
    ref1 = make_snapshot(w, study_config, providers, universe)
    ref2 = make_snapshot(w, study_config, providers, universe)
    assert ref1.path == ref2.path and ref2.version == 1
    ref3 = make_snapshot(w, study_config, providers, universe, force_new_version=True)
    assert ref3.version == 2 and ref3.path.name == f"{w}.v2"
    assert read_snapshot(w).ref.version == 2


def test_snapshot_closed_friday_uses_thursday(data_dir: Path, study_config: StudyConfig, providers: ProviderBundle,
                                              universe: pd.DataFrame) -> None:
    # Good Friday 2026-04-03 is in ISO week 2026-W14.
    w = cal.week_id(dt.date(2026, 4, 3))
    ref = make_snapshot(w, study_config, providers, universe)
    assert ref.cutoff_date == dt.date(2026, 4, 2)
    assert read_snapshot(w).prices["date"].max() == dt.date(2026, 4, 2)


def test_removed_stock_is_dropped_and_logged(data_dir: Path, study_config: StudyConfig, providers: ProviderBundle,
                                             universe: pd.DataFrame, world: SyntheticWorld,
                                             first_cutoff: dt.date,
                                             caplog: pytest.LogCaptureFixture) -> None:
    delisted = PLANTED["delisting"][0]
    uni = universe.copy()
    if delisted not in set(uni["ticker"]):
        row = uni.iloc[[0]].copy()
        row["ticker"] = delisted
        uni = pd.concat([uni.iloc[1:], row], ignore_index=True)
    # Pick a week whose cutoff is after the planted delisting day.
    days = cal.trading_days(first_cutoff, first_cutoff + dt.timedelta(days=400))
    w = cal.week_id(days[PLANTED["delisting"][1] + 15])
    logger = logging.getLogger("test.snapshot")
    with caplog.at_level(logging.WARNING, logger="test.snapshot"):
        make_snapshot(w, study_config, providers, uni, logger=logger)
    snap = read_snapshot(w)
    assert delisted not in snap.included_tickers
    assert len(snap.included_tickers) < len(uni)
    assert any(getattr(r, "extra_fields", {}).get("ticker") == delisted and r.msg == "stock_skipped"
               for r in caplog.records)
    assert ((snap.issues["ticker"] == delisted) & (snap.issues["code"] == "stopped_trading")).any()


# Ledger -------------------------------------------------------------------------------------


def test_ledger_is_append_only_and_keeps_old_values(data_dir: Path) -> None:
    logger = logging.getLogger("test.ledger")
    rows = pd.DataFrame({"date": [dt.date(2026, 1, 5), dt.date(2026, 1, 6)], "ticker": ["AAA", "AAA"],
                         "open": [1.0, 2.0], "high": [1.1, 2.1], "low": [0.9, 1.9], "close": [1.0, 2.0],
                         "volume": [10, 10]})
    assert ledger.append_prices(rows, logger) == 2
    changed = rows.copy()
    changed.loc[0, "close"] = 99.0
    changed = pd.concat([changed, pd.DataFrame({"date": [dt.date(2026, 1, 7)], "ticker": ["AAA"], "open": [3.0],
                                                "high": [3.1], "low": [2.9], "close": [3.0], "volume": [10]})])
    assert ledger.append_prices(changed, logger) == 1
    out = ledger.read_ledger()
    assert len(out) == 3
    assert out.loc[out["date"] == dt.date(2026, 1, 5), "close"].iloc[0] == 1.0


# Real adapters, mock-tested -------------------------------------------------------------------


class FakeHttp:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def get_json(self, url: str, params: dict[str, object] | None = None,
                 headers: dict[str, str] | None = None) -> object:
        self.calls.append({"url": url, "params": params, "headers": headers})
        return self.responses.pop(0)

    def get_text(self, url: str, params: dict[str, object] | None = None,
                 headers: dict[str, str] | None = None) -> str:
        return str(self.responses.pop(0))


def test_alpaca_prices_merge_raw_and_adjusted() -> None:
    raw = {"bars": {"AAA": [{"t": "2026-01-05T05:00:00Z", "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 100}]},
           "next_page_token": None}
    adj = {"bars": {"AAA": [{"t": "2026-01-05T05:00:00Z", "o": 5, "h": 5.5, "l": 4.5, "c": 5.25, "v": 200}]},
           "next_page_token": None}
    http = FakeHttp([raw, adj])
    provider = AlpacaPriceProvider(http, key_id="k", secret="s")
    out = provider.get_prices(["AAA"], dt.date(2026, 1, 1), dt.date(2026, 1, 10))
    assert list(out.columns) == ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    assert out.loc[0, "close"] == 10.5 and out.loc[0, "adj_close"] == 5.25
    assert http.calls[0]["params"]["adjustment"] == "raw"  # type: ignore[index]
    assert http.calls[1]["params"]["adjustment"] == "all"  # type: ignore[index]
    assert http.calls[0]["headers"]["APCA-API-KEY-ID"] == "k"  # type: ignore[index]


def test_alpaca_news_paginates_and_filters_symbols() -> None:
    page1 = {"news": [{"created_at": "2026-01-05T12:00:00Z", "headline": "h1", "summary": "s1",
                       "symbols": ["AAA", "ZZZ"], "source": "x"}], "next_page_token": "p2"}
    page2 = {"news": [{"created_at": "2026-01-06T12:00:00Z", "headline": "h2", "summary": "",
                       "symbols": ["BBB"], "source": "y"}], "next_page_token": None}
    http = FakeHttp([page1, page2])
    provider = AlpacaNewsProvider(http, key_id="k", secret="s")
    start = dt.datetime(2026, 1, 1, tzinfo=UTC)
    out = provider.get_news(["AAA", "BBB"], start, start + dt.timedelta(days=7))
    assert list(out["ticker"]) == ["AAA", "BBB"]
    assert out["published_at"].iloc[0].tzinfo is not None
    assert http.calls[1]["params"]["page_token"] == "p2"  # type: ignore[index]


def test_fred_skips_missing_observations() -> None:
    http = FakeHttp([{"observations": [{"date": "2026-01-02", "value": "4.25"},
                                       {"date": "2026-01-05", "value": "."}]}])
    out = FredRateProvider(http, api_key="k").get_rates(dt.date(2026, 1, 1), dt.date(2026, 1, 10))
    assert len(out) == 1 and out.loc[0, "rate"] == 4.25


def test_wikipedia_constituents_parser() -> None:
    html = """
    <table id="constituents"><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr>
    <tr><td><a href="#">MMM</a></td><td>3M</td><td>Industrials</td><td>Conglomerates</td></tr>
    <tr><td>AOS</td><td>A. O. Smith</td><td>Industrials</td><td>Building Products</td></tr></table>
    <table id="changes"><tr><td>ignored</td></tr></table>
    """
    out = parse_constituents(html)
    assert list(out["ticker"]) == ["MMM", "AOS"]
    assert list(out["sector"]) == ["Industrials", "Industrials"]
