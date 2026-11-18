"""Frozen weekly data snapshots.

`make_snapshot` pulls prices, news, reference data, and rates as of the cutoff, applies the
data checks, writes everything to data/snapshots/<week_id>/ as CSV plus a manifest, appends
new raw trading days to the price ledger, and is immutable once written.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from portstudy import calendar as cal
from portstudy.config import StudyConfig
from portstudy.data import ledger
from portstudy.data.checks import (
    check_news,
    check_prices,
    has_halt,
    issues_frame,
    tickers_to_skip,
)
from portstudy.data.providers.base import REFERENCE_COLUMNS, ProviderBundle
from portstudy.data.universe import record_removal, removed_tickers
from portstudy.hashing import sha256_file
from portstudy.io import read_csv, read_json, write_csv_atomic, write_json_atomic
from portstudy.logging_setup import now_et
from portstudy.paths import snapshot_dir, snapshots_dir
from portstudy.types import Issue, SnapshotRef

ET = ZoneInfo("America/New_York")
UTC = dt.UTC

SNAPSHOT_FILES = [
    "prices.csv",
    "etf_prices.csv",
    "actions.csv",
    "news.csv",
    "news_text.csv",
    "reference.csv",
    "rates.csv",
    "universe.csv",
    "issues.csv",
]


class SnapshotHalt(RuntimeError):
    """Raised when a check with rule `halt` fires."""


def cutoff_datetime_et(cutoff: dt.date, cfg: StudyConfig) -> dt.datetime:
    hh, mm = (int(x) for x in cfg.timing.cutoff_time_et.split(":"))
    return dt.datetime.combine(cutoff, dt.time(hh, mm), tzinfo=ET)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _normalize_headline(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _valid_cap(value: Any) -> bool:
    try:
        cap = float(value)
    except (TypeError, ValueError):
        return False
    return cap > 0 and not math.isnan(cap)


def carry_market_caps(
    reference: pd.DataFrame,
    earlier: Callable[[str], float | None],
    earlier_week: Callable[[str], str | None] | None = None,
) -> tuple[pd.DataFrame, list[Issue]]:
    """Fill missing market caps from the most recent earlier snapshot (DECISION(C8)).

    `earlier(ticker)` returns that ticker's most recent earlier cap or None; `earlier_week`
    (optional) names the snapshot it came from. Carried and missing caps are issues, never
    skipped stocks; more than 10% degraded adds a summary warning."""
    ref = reference.copy()
    issues: list[Issue] = []
    carried = missing = 0
    for idx, row in ref.iterrows():
        cap = row.get("market_cap")
        if _valid_cap(cap):
            continue
        tk = str(row["ticker"])
        prev = earlier(tk)
        if prev is not None:
            ref.at[idx, "market_cap"] = prev
            carried += 1
            details = {}
            if earlier_week is not None:
                details["from_week"] = earlier_week(tk) or ""
            issues.append(
                Issue(
                    "info",
                    tk,
                    "market_cap_carried",
                    "market cap carried from the most recent earlier snapshot",
                    "carry",
                    details,
                )
            )
        else:
            missing += 1
            issues.append(
                Issue(
                    "warning",
                    tk,
                    "market_cap_missing",
                    "no earlier snapshot has a market cap for this ticker",
                    "carry",
                )
            )
    if len(ref) and (carried + missing) > 0.10 * len(ref):
        issues.append(
            Issue(
                "warning",
                "*",
                "market_caps_degraded",
                f"market caps degraded: {carried} carried, {missing} missing of {len(ref)}",
                "carry",
            )
        )
    return ref, issues


def _earlier_cap_lookup(
    week_id: str,
) -> tuple[Callable[[str], float | None], Callable[[str], str | None]]:
    """Walk back up to 60 weeks (latest snapshot version per week) for a ticker's cap."""
    snapshots: list[tuple[str, pd.Series]] | None = None

    def load() -> list[tuple[str, pd.Series]]:
        out: list[tuple[str, pd.Series]] = []
        w = week_id
        for _ in range(60):
            w = cal.prev_week_id(w)
            versions = _existing_versions(w)
            if versions:
                df = read_csv(snapshot_dir(w, versions[-1]) / "reference.csv")
                if {"ticker", "market_cap"} <= set(df.columns):
                    out.append((w, df.set_index(df["ticker"].astype(str))["market_cap"]))
        return out

    def find(ticker: str) -> tuple[float, str] | None:
        nonlocal snapshots
        if snapshots is None:
            snapshots = load()
        for w, caps in snapshots:
            if ticker in caps.index:
                value = caps[ticker]
                if _valid_cap(value):
                    return float(value), w
        return None

    def cap(ticker: str) -> float | None:
        found = find(ticker)
        return found[0] if found else None

    def wk(ticker: str) -> str | None:
        found = find(ticker)
        return found[1] if found else None

    return cap, wk


def _existing_versions(week_id: str) -> list[int]:
    versions = []
    for p in snapshots_dir().glob(f"{week_id}*"):
        if p.name == week_id:
            versions.append(1)
        elif p.name.startswith(f"{week_id}.v"):
            versions.append(int(p.name.split(".v")[1]))
    return sorted(versions)


def _ref_for(week_id: str, path: Path, version: int) -> SnapshotRef:
    return SnapshotRef(
        week_id=week_id,
        cutoff_date=cal.cutoff_day(week_id),
        trade_date=cal.monday_open_day(week_id),
        window_end_date=cal.window_end_date(week_id),
        path=path,
        version=version,
    )


def make_snapshot(
    week_id: str,
    cfg: StudyConfig,
    providers: ProviderBundle,
    universe: pd.DataFrame,
    force_new_version: bool = False,
    logger: logging.Logger | None = None,
) -> SnapshotRef:
    """Build the frozen snapshot for `week_id`. Returns the existing one if already written."""
    log = logger or logging.getLogger("portstudy.snapshot")
    versions = _existing_versions(week_id)
    if versions and not force_new_version:
        v = versions[-1]
        return _ref_for(week_id, snapshot_dir(week_id, v), v)
    version = (versions[-1] + 1) if versions else 1
    folder = snapshot_dir(week_id, version)

    cutoff = cal.cutoff_day(week_id)
    cutoff_dt = cutoff_datetime_et(cutoff, cfg)
    cutoff_utc = cutoff_dt.astimezone(UTC)
    trade_date = cal.monday_open_day(week_id)
    window_end = cal.window_end_date(week_id)

    # Universe tickers and ETF proxies.
    tickers = sorted(universe["ticker"].astype(str))
    etfs = list(cfg.universe.etf_proxies)

    # U1: tickers removed in an earlier week are not fetched at all; they stay in the
    # snapshot's universe.csv with included = False and nothing replaces them.
    earlier_removed = sorted(removed_tickers(cal.prev_week_id(week_id)) & set(tickers))
    fetch_tickers = [t for t in tickers if t not in earlier_removed]
    removal_issues = [
        Issue(
            "info",
            tk,
            "removed_from_study",
            "removed from the study in an earlier week",
            "remove",
        )
        for tk in earlier_removed
    ]

    # Prices: at least price_history_days trading days back to the cutoff.
    hist_days = cfg.data.price_history_days
    start = cutoff - dt.timedelta(days=int(hist_days * 1.6) + 30)
    expected_days = cal.trading_days(start, cutoff)
    if len(expected_days) > hist_days:
        expected_days = expected_days[-hist_days:]
    start = expected_days[0]
    prices = providers.prices.get_prices(fetch_tickers, start, cutoff)
    prices = prices[prices["date"] <= cutoff].copy()
    etf_prices = providers.prices.get_prices(etfs, start, cutoff)
    etf_prices = etf_prices[etf_prices["date"] <= cutoff].copy()
    actions = providers.prices.get_corporate_actions(fetch_tickers, start, cutoff)

    # News window: (previous cutoff 16:00 ET, cutoff 16:00 ET]; first snapshot uses 7 days.
    prev_week = cal.prev_week_id(week_id)
    if _existing_versions(prev_week):
        news_start = cutoff_datetime_et(cal.cutoff_day(prev_week), cfg).astimezone(UTC)
    else:
        news_start = cutoff_utc - dt.timedelta(days=cfg.data.first_snapshot_news_days)
    # Pull a little past the cutoff so the leakage check can see late articles, then drop them.
    raw_news = providers.news.get_news(
        fetch_tickers, news_start, cutoff_utc + dt.timedelta(hours=12)
    )
    news_issues = check_news(raw_news, tickers, cutoff_utc, cfg.data.checks)
    if len(raw_news):
        pub = pd.to_datetime(raw_news["published_at"], utc=True)
        news = raw_news[pub <= cutoff_utc].copy()
        news["published_at"] = pd.to_datetime(news["published_at"], utc=True)
    else:
        news = raw_news.copy()
    news = _dedupe_news(news)
    news_text = _build_news_text(news, tickers, cfg)

    # DECISION(C8): name and sector come from the frozen universe file for every ticker; the
    # reference provider is used only for market_cap and next_earnings_date.
    uni = universe.copy()
    uni["ticker"] = uni["ticker"].astype(str)
    uni = uni.set_index("ticker")
    universe_issues: list[Issue] = []
    for tk in tickers:
        name = uni.at[tk, "name"] if tk in uni.index and "name" in uni.columns else None
        sector = uni.at[tk, "sector"] if tk in uni.index and "sector" in uni.columns else None
        if not (isinstance(name, str) and name.strip()) or not (
            isinstance(sector, str) and sector.strip()
        ):
            universe_issues.append(
                Issue(
                    "error",
                    tk,
                    "universe_missing_field",
                    "universe file has no name or sector for this ticker",
                    "halt",
                )
            )
    try:
        fetched = providers.reference.get_reference(fetch_tickers, cutoff)
    except Exception:
        log.exception("reference_call_failed", extra={"extra_fields": {"week_id": week_id}})
        fetched = pd.DataFrame(columns=REFERENCE_COLUMNS)
    reference = fetched.set_index("ticker").reindex(fetch_tickers).reset_index()
    for col in REFERENCE_COLUMNS:
        if col not in reference.columns:
            reference[col] = pd.NA
    names = uni["name"] if "name" in uni.columns else pd.Series(dtype=object)
    sectors = uni["sector"] if "sector" in uni.columns else pd.Series(dtype=object)
    reference["name"] = reference["ticker"].map(names)
    reference["sector"] = reference["ticker"].map(sectors)
    nxt = pd.to_datetime(reference["next_earnings_date"], errors="coerce").dt.date
    reference["earnings_in_window"] = [
        bool(d is not None and not pd.isna(d) and trade_date <= d < window_end) for d in nxt
    ]
    earlier_cap, earlier_cap_week = _earlier_cap_lookup(week_id)
    reference, cap_issues = carry_market_caps(reference, earlier_cap, earlier_cap_week)
    # J2 still fails when every cap is missing and no earlier snapshot exists (risk/prior.py
    # raises on a zero market-cap sum); the carry only smooths over single-week outages.
    n_cap_carried = sum(1 for i in cap_issues if i.code == "market_cap_carried")
    n_cap_missing = sum(1 for i in cap_issues if i.code == "market_cap_missing")
    rates = providers.rates.get_rates(start, cutoff)

    # Checks and rules.
    price_issues = check_prices(
        prices,
        expected_days,
        cfg.data.checks,
        actions,
        min_history_days=cfg.universe.min_history_days,
        since=cal.cutoff_day(prev_week),
    )
    issues: list[Issue] = price_issues + news_issues + universe_issues + cap_issues + removal_issues
    for issue in issues:
        log.log(
            logging.WARNING if issue.severity != "info" else logging.INFO,
            "check_issue",
            extra={"extra_fields": {"week_id": week_id, **issue.to_dict()}},
        )
    if has_halt(issues):
        raise SnapshotHalt(
            f"halt rule fired for {week_id}: "
            + "; ".join(f"{i.ticker}:{i.code}" for i in issues if i.rule == "halt")
        )
    skipped = tickers_to_skip(issues)
    # Tickers with no price rows at all are dropped without replacement and logged.
    present = set(prices["ticker"].astype(str))
    for tk in tickers:
        if tk not in present and tk not in skipped:
            skipped.add(tk)
            issues.append(Issue("error", tk, "no_prices", "no price rows in window", "skip_stock"))
            log.warning(
                "stock_dropped",
                extra={"extra_fields": {"week_id": week_id, "ticker": tk, "reason": "no_prices"}},
            )
    for tk in sorted(skipped):
        log.warning("stock_skipped", extra={"extra_fields": {"week_id": week_id, "ticker": tk}})
    universe_out = universe.copy()
    universe_out["ticker"] = universe_out["ticker"].astype(str)
    universe_out["included"] = ~universe_out["ticker"].isin(skipped)
    universe_out["article_count"] = (
        universe_out["ticker"]
        .map(news_text.set_index("ticker")["article_count"])
        .fillna(0)
        .astype(int)
    )

    # Forward fill missing days for included stocks (rule forward_fill), dropping duplicates.
    prices = prices.drop_duplicates(["ticker", "date"], keep="first")
    # The ledger gets the provider's rows only; the forward fill below is for the models.
    raw_prices = prices.copy()
    prices = _forward_fill(prices, expected_days)
    # Write the folder. Dates are ISO strings; published_at carries the ET offset.
    folder.mkdir(parents=True, exist_ok=False)
    news_out = news.copy()
    if len(news_out):
        news_out["published_at"] = news_out["published_at"].map(
            lambda x: x.tz_convert(ET).isoformat()
        )
    write_csv_atomic(_iso(prices, "date"), folder / "prices.csv")
    write_csv_atomic(_iso(etf_prices, "date"), folder / "etf_prices.csv")
    write_csv_atomic(_iso(actions, "date"), folder / "actions.csv")
    write_csv_atomic(news_out, folder / "news.csv")
    write_csv_atomic(news_text, folder / "news_text.csv")
    write_csv_atomic(_iso(reference, "next_earnings_date"), folder / "reference.csv")
    write_csv_atomic(_iso(rates, "date"), folder / "rates.csv")
    write_csv_atomic(universe_out, folder / "universe.csv")
    write_csv_atomic(issues_frame(issues), folder / "issues.csv")
    manifest = {
        "week_id": week_id,
        "version": version,
        "cutoff_date": cutoff.isoformat(),
        "cutoff_datetime_et": cutoff_dt.isoformat(),
        "trade_date": trade_date.isoformat(),
        "window_end_date": window_end.isoformat(),
        "news_window_start_utc": news_start.isoformat(),
        "providers": providers.names(),
        "retrieved_at": now_et().isoformat(),
        "n_tickers": len(tickers),
        "n_included": int(universe_out["included"].sum()),
        "n_market_cap_carried": n_cap_carried,
        "n_market_cap_missing": n_cap_missing,
        "n_large_move": sum(1 for i in issues if i.code == "large_move"),
        "removed": sorted(i.ticker for i in issues if i.code == "crash_removal"),
        "removed_earlier": earlier_removed,
        "skipped": sorted(skipped),
        "files": {name: sha256_file(folder / name) for name in SNAPSHOT_FILES},
    }
    write_json_atomic(manifest, folder / "manifest.json")

    # Append new trading days to the raw price ledger and the actions log.
    all_raw = pd.concat([raw_prices, etf_prices], ignore_index=True)
    ledger.append_prices(all_raw, logger=log)
    ledger.append_actions(actions)

    # U1: record this week's crash removals after the folder is written, so every
    # later snapshot excludes them (the first removal for a ticker wins).
    for issue in issues:
        if issue.code == "crash_removal":
            record_removal(
                week_id,
                issue.ticker,
                dt.date.fromisoformat(str(issue.details["date"])),
                float(issue.details["return"]),
                "crash_removal",
            )
    return _ref_for(week_id, folder, version)


def _iso(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    if col in out.columns and len(out):
        out[col] = out[col].map(lambda x: x.isoformat() if hasattr(x, "isoformat") else x)
    return out


def _forward_fill(prices: pd.DataFrame, expected_days: list[dt.date]) -> pd.DataFrame:
    """Fill missing trading days inside each stock's own span with the prior close."""
    out = []
    day_set = expected_days
    for ticker, g in prices.groupby("ticker", sort=True):
        g = g.sort_values("date").set_index("date")
        span = [d for d in day_set if g.index[0] <= d <= g.index[-1]]
        if len(span) != len(g):
            g = g.reindex(span)
            missing = g["close"].isna()
            g["ticker"] = ticker
            for col in ["open", "high", "low", "close", "adj_close"]:
                g[col] = g[col].ffill()
            g.loc[missing, "volume"] = 0.0
            for col in ["open", "high", "low"]:
                g.loc[missing, col] = g.loc[missing, "close"]
        out.append(g.reset_index().rename(columns={"index": "date"}))
    if not out:
        return prices
    merged = pd.concat(out, ignore_index=True)
    return merged.loc[:, list(prices.columns)]


def _dedupe_news(news: pd.DataFrame) -> pd.DataFrame:
    if len(news) == 0:
        return news
    out = news.copy()
    out["_norm"] = out["headline"].map(_normalize_headline)
    out = out.sort_values(["ticker", "published_at"], kind="stable")
    out = out.drop_duplicates(["ticker", "_norm"], keep="first")
    return out.drop(columns=["_norm"]).reset_index(drop=True)


def _build_news_text(news: pd.DataFrame, tickers: list[str], cfg: StudyConfig) -> pd.DataFrame:
    """Per stock: newest-first articles up to the token budget, joined into one text block."""
    budget = cfg.data.news_token_budget_per_stock
    rows = []
    by_ticker = {str(k): g for k, g in news.groupby("ticker")} if len(news) else {}
    for tk in tickers:
        g = by_ticker.get(tk)
        if g is None or len(g) == 0:
            rows.append({"ticker": tk, "article_count": 0, "news_text": ""})
            continue
        g = g.sort_values("published_at", ascending=False, kind="stable")
        parts: list[str] = []
        used = 0
        for pub, headline, summary_raw in zip(g["published_at"], g["headline"], g["summary"]):
            stamp = pd.Timestamp(pub).tz_convert(ET).strftime("%Y-%m-%d %H:%M %Z")
            summary = str(summary_raw) if isinstance(summary_raw, str) else ""
            line = f"[{stamp}] {headline}. {summary}".strip()
            cost = estimate_tokens(line)
            if parts and used + cost > budget:
                break
            parts.append(line)
            used += cost
        rows.append({"ticker": tk, "article_count": len(parts), "news_text": "\n".join(parts)})
    return pd.DataFrame(rows, columns=["ticker", "article_count", "news_text"])


# Reading ------------------------------------------------------------------------------------


@dataclass
class Snapshot:
    ref: SnapshotRef

    @property
    def path(self) -> Path:
        return self.ref.path

    @cached_property
    def manifest(self) -> dict[str, object]:
        data: dict[str, object] = read_json(self.path / "manifest.json")
        return data

    def _read_dates(self, name: str, col: str = "date") -> pd.DataFrame:
        df = read_csv(self.path / name)
        if col in df.columns and len(df):
            df[col] = pd.to_datetime(df[col]).dt.date
        return df

    @cached_property
    def prices(self) -> pd.DataFrame:
        return self._read_dates("prices.csv")

    @cached_property
    def etf_prices(self) -> pd.DataFrame:
        return self._read_dates("etf_prices.csv")

    @cached_property
    def actions(self) -> pd.DataFrame:
        return self._read_dates("actions.csv")

    @cached_property
    def news(self) -> pd.DataFrame:
        df = read_csv(self.path / "news.csv")
        if len(df):
            df["published_at"] = pd.to_datetime(df["published_at"], utc=True).dt.tz_convert(ET)
        return df

    @cached_property
    def news_text(self) -> pd.DataFrame:
        df = read_csv(self.path / "news_text.csv", keep_default_na=False)
        df["news_text"] = df["news_text"].astype(str)
        return df

    @cached_property
    def reference(self) -> pd.DataFrame:
        return self._read_dates("reference.csv", "next_earnings_date")

    @cached_property
    def rates(self) -> pd.DataFrame:
        return self._read_dates("rates.csv")

    @cached_property
    def universe(self) -> pd.DataFrame:
        return read_csv(self.path / "universe.csv")

    @cached_property
    def issues(self) -> pd.DataFrame:
        return read_csv(self.path / "issues.csv")

    @property
    def included_tickers(self) -> list[str]:
        u = self.universe
        return sorted(u.loc[u["included"], "ticker"].astype(str))

    def adj_close_wide(self) -> pd.DataFrame:
        """Adjusted close, dates by tickers, included tickers only."""
        p = self.prices[self.prices["ticker"].isin(self.included_tickers)]
        wide = p.pivot(index="date", columns="ticker", values="adj_close").sort_index()
        return wide


def read_snapshot(week_id: str, version: int | None = None) -> Snapshot:
    versions = _existing_versions(week_id)
    if not versions:
        raise FileNotFoundError(f"no snapshot for {week_id}")
    v = version or versions[-1]
    return Snapshot(_ref_for(week_id, snapshot_dir(week_id, v), v))


def snapshot_exists(week_id: str) -> bool:
    return bool(_existing_versions(week_id))
