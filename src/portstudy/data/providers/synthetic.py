"""Seeded synthetic market used for every offline test and the rehearsal run.

Prices come from a factor model (market plus sector plus idiosyncratic noise) with a weak
planted momentum effect and a weak planted "LLM-detectable" effect carried by sentiment
words in the news text. Planted data problems (split, dividend, delisting, halt, extreme
move, zero price, stale price, duplicate rows, short history, no-news weeks, news after
the cutoff) let every data check and the leakage tests fire on known cases.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from portstudy import calendar as cal
from portstudy.config import SyntheticConfig
from portstudy.data.providers.base import (
    ACTION_COLUMNS,
    MEMBER_COLUMNS,
    NEWS_COLUMNS,
    PRICE_COLUMNS,
    RATE_COLUMNS,
    REFERENCE_COLUMNS,
    MembersProvider,
    NewsProvider,
    PriceProvider,
    ProviderBundle,
    RateProvider,
    ReferenceProvider,
    empty_frame,
)
from portstudy.io import read_csv, write_csv_atomic, write_json_atomic

ET = ZoneInfo("America/New_York")
UTC = dt.UTC

GICS_SECTORS = [
    "Information Technology",
    "Health Care",
    "Financials",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",
]

ETF_TICKERS = ["SPY", "IWM", "IWD", "IWF", "QUAL"]

# Sentiment vocabulary. The stub LLM counts these words; the planted effect lives here.
POSITIVE_WORDS = ["beats", "upgrade", "record", "strong", "surges", "raises", "expands"]
NEGATIVE_WORDS = ["misses", "downgrade", "probe", "weak", "plunges", "cuts", "recall"]
NEUTRAL_WORDS = ["announces", "reports", "schedules", "hosts", "updates", "names", "files"]

# Planted events, as trading-day offsets from the first cutoff day. Tickers are fixed so
# tests can target them directly.
PLANTED = {
    "split": ("SYN005", 30, 2.0),
    "dividend": ("SYN006", 20, 0.50),
    "delisting": ("SYN007", 60, None),
    "halt": ("SYN009", 40, 3),
    "extreme_move": ("SYN011", 50, -0.45),
    "zero_price": ("SYN013", 35, None),
    "stale_price": ("SYN015", 45, 6),
    "duplicate_rows": ("SYN017", 25, None),
    "short_history": ("SYN019", -200, None),
}

# Weekly planted effect per unit of sentiment (sentiment in -2..2, applied over 5 days).
NEWS_EFFECT_WEEKLY = 0.004
MOMENTUM_EFFECT_DAILY = 0.0005
NO_NEWS_PROB = 0.12
LATE_NEWS_PROB = 0.10


def _ticker(i: int) -> str:
    return f"SYN{i:03d}"


@dataclass
class SyntheticWorld:
    prices: pd.DataFrame
    actions: pd.DataFrame
    news: pd.DataFrame
    earnings: pd.DataFrame
    rates: pd.DataFrame
    members: pd.DataFrame
    sentiment: pd.DataFrame
    base_caps: pd.DataFrame
    meta: dict[str, object]

    # Construction ---------------------------------------------------------------------------

    @classmethod
    def generate(cls, cfg: SyntheticConfig, first_week: str = "2026-W02") -> SyntheticWorld:
        rng = np.random.default_rng(cfg.seed)
        n = cfg.n_tickers
        tickers = [_ticker(i) for i in range(n)]
        sectors = [GICS_SECTORS[i % len(GICS_SECTORS)] for i in range(n)]

        # Trading day grid: history_days before the first cutoff, future_weeks after it.
        first_cutoff = cal.cutoff_day(first_week)
        end_date = cal.week_friday(first_week) + dt.timedelta(days=7 * (cfg.future_weeks + 2))
        back_start = first_cutoff - dt.timedelta(days=int(cfg.history_days * 1.6) + 30)
        all_days = cal.trading_days(back_start, end_date)
        first_idx = all_days.index(first_cutoff)
        days = all_days[first_idx - cfg.history_days :]
        first_idx = cfg.history_days
        n_days = len(days)
        day_index = pd.DatetimeIndex(pd.to_datetime(days))

        # Week labels for each day: the week whose return window contains the day.
        # A day belongs to the window of week w if monday_open(w) <= day < monday_open(w+1).
        day_week = _window_week_of_days(days)
        weeks = sorted(set(day_week))

        # Factor loadings and sentiment draws.
        beta = rng.normal(1.0, 0.25, n)
        gamma = rng.normal(0.8, 0.2, n)
        idio_vol = rng.uniform(0.010, 0.020, n)
        sector_idx = np.array([GICS_SECTORS.index(s) for s in sectors])
        sent_probs = np.array([0.10, 0.22, 0.36, 0.22, 0.10])
        sentiment = rng.choice([-2, -1, 0, 1, 2], size=(len(weeks), n), p=sent_probs)
        has_news = rng.random((len(weeks), n)) >= NO_NEWS_PROB
        sentiment = np.where(has_news, sentiment, 0)
        week_pos = {w: k for k, w in enumerate(weeks)}

        # Daily factor returns.
        f_m = rng.normal(0.0003, 0.010, n_days)
        f_s = rng.normal(0.0, 0.005, (n_days, len(GICS_SECTORS)))
        f_size = rng.normal(0.0, 0.004, n_days)
        f_value = rng.normal(0.0, 0.003, n_days)
        f_qual = rng.normal(0.0, 0.003, n_days)
        eps = rng.normal(0.0, 1.0, (n_days, n)) * idio_vol

        log_p = np.zeros((n_days, n))
        log_p[0] = np.log(rng.uniform(20, 300, n))
        for t in range(1, n_days):
            r = beta * f_m[t] + gamma * f_s[t, sector_idx] + eps[t]
            # Weak planted momentum: drift toward the sign of the 12-1 month return.
            if t > 252:
                past = log_p[t - 21] - log_p[t - 252]
                z = (past - past.mean()) / (past.std() + 1e-12)
                r = r + MOMENTUM_EFFECT_DAILY * np.clip(z, -2, 2)
            # Planted news effect: sentiment of the week whose window holds this day.
            w = day_week[t]
            if w in week_pos:
                r = r + NEWS_EFFECT_WEEKLY / 5.0 * sentiment[week_pos[w]]
            log_p[t] = log_p[t - 1] + r

        adj_close = np.exp(log_p)
        # Planted extreme move (real crash, no split) on the adjusted series too.
        em_ticker, em_off, em_ret = PLANTED["extreme_move"]
        em_i, em_t = tickers.index(em_ticker), first_idx + em_off
        adj_close[em_t:, em_i] *= 1.0 + float(em_ret)  # type: ignore[arg-type]

        # Raw close equals adjusted close until a dividend or split is planted.
        raw_close = adj_close.copy()
        actions: list[dict[str, object]] = []
        sp_ticker, sp_off, sp_ratio = PLANTED["split"]
        sp_i, sp_t = tickers.index(sp_ticker), first_idx + sp_off
        raw_close[sp_t:, sp_i] = adj_close[sp_t:, sp_i] / float(sp_ratio)  # type: ignore[arg-type]
        # Before the split the adjusted series is the raw series divided by the ratio.
        adj_close[:sp_t, sp_i] = raw_close[:sp_t, sp_i] / float(sp_ratio)  # type: ignore[arg-type]
        actions.append(
            {"date": days[sp_t], "ticker": sp_ticker, "action": "split", "value": sp_ratio}
        )
        dv_ticker, dv_off, dv_cash = PLANTED["dividend"]
        dv_i, dv_t = tickers.index(dv_ticker), first_idx + dv_off
        # Adjusted prices before the ex-date are scaled down by the dividend factor.
        factor = 1.0 - float(dv_cash) / raw_close[dv_t - 1, dv_i]  # type: ignore[arg-type]
        adj_close[:dv_t, dv_i] = raw_close[:dv_t, dv_i] * factor
        actions.append(
            {"date": days[dv_t], "ticker": dv_ticker, "action": "dividend", "value": dv_cash}
        )

        # OHLC around the close and volume.
        base_vol = np.exp(rng.normal(14.0, 0.8, n))
        rows = []
        for i, tk in enumerate(tickers):
            spread = rng.uniform(0.002, 0.012, n_days)
            gap = rng.normal(0.0, 0.004, n_days)
            close = raw_close[:, i]
            open_ = np.empty(n_days)
            open_[0] = close[0]
            open_[1:] = close[:-1] * (1 + gap[1:])
            open_[sp_t:] = open_[sp_t:] if i != sp_i else open_[sp_t:]
            if i == sp_i:
                open_[sp_t] = close[sp_t - 1] / float(sp_ratio) * (1 + gap[sp_t])  # type: ignore[arg-type]
            high = np.maximum(open_, close) * (1 + spread)
            low = np.minimum(open_, close) * (1 - spread)
            vol = base_vol[i] * np.exp(rng.normal(0, 0.3, n_days))
            ratio = adj_close[:, i] / close
            rows.append(
                pd.DataFrame(
                    {
                        "date": day_index,
                        "ticker": tk,
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "adj_close": adj_close[:, i],
                        "volume": np.round(vol),
                        "_adj_ratio": ratio,
                    }
                )
            )
        prices = pd.concat(rows, ignore_index=True)

        # ETF proxies built from the factors.
        etf_rets = {
            "SPY": f_m,
            "IWM": 1.05 * f_m + f_size,
            "IWD": 0.95 * f_m + f_value,
            "IWF": 1.05 * f_m - f_value,
            "QUAL": f_m + 0.5 * f_qual,
        }
        etf_rows = []
        for tk, r in etf_rets.items():
            lp = np.log(100.0) + np.concatenate([[0.0], np.cumsum(r[1:])])
            c = np.exp(lp)
            o = np.concatenate([[c[0]], c[:-1]])
            etf_rows.append(
                pd.DataFrame(
                    {
                        "date": day_index,
                        "ticker": tk,
                        "open": o,
                        "high": np.maximum(o, c) * 1.003,
                        "low": np.minimum(o, c) * 0.997,
                        "close": c,
                        "adj_close": c,
                        "volume": 5e7,
                        "_adj_ratio": 1.0,
                    }
                )
            )
        prices = pd.concat([prices] + etf_rows, ignore_index=True)
        prices["date"] = pd.to_datetime(prices["date"]).dt.date

        # Planted data problems.
        prices = _plant_problems(prices, tickers, days, first_idx)

        # Market caps: base cap times the adjusted price path.
        base_caps = pd.DataFrame(
            {
                "ticker": tickers,
                "base_cap": np.exp(rng.normal(np.log(4e10), 0.9, n)),
                "base_price": adj_close[0],
            }
        )

        # Earnings: quarterly cycle per stock, offset within the first quarter.
        offsets = rng.integers(0, 63, n)
        earn_rows = []
        for i, tk in enumerate(tickers):
            for t in range(int(offsets[i]), n_days, 63):
                earn_rows.append({"ticker": tk, "date": days[t]})
        earnings = pd.DataFrame(earn_rows)

        # T-bill rate in percent, slow random walk around 4.
        rate = 4.0 + np.cumsum(rng.normal(0, 0.01, n_days))
        rates = pd.DataFrame({"date": days, "rate": np.round(rate, 3)})

        # Members table and names.
        members = pd.DataFrame(
            {
                "ticker": tickers,
                "name": [f"Synthetic Company {i:03d}" for i in range(n)],
                "sector": sectors,
            }
        )

        # News, with the sentiment vocabulary and UTC timestamps.
        news, sent_rows = _make_news(rng, tickers, members, weeks, sentiment, has_news)
        sentiment_df = pd.DataFrame(sent_rows)

        meta: dict[str, object] = {
            "seed": cfg.seed,
            "first_week": first_week,
            "first_cutoff": first_cutoff.isoformat(),
            "history_days": cfg.history_days,
            "future_weeks": cfg.future_weeks,
            "n_tickers": n,
            "weeks": weeks,
            "planted": {k: [str(x) for x in v] for k, v in PLANTED.items()},
        }
        return cls(
            prices,
            pd.DataFrame(actions),
            news,
            earnings,
            rates,
            members,
            sentiment_df,
            base_caps,
            meta,
        )

    # Persistence ----------------------------------------------------------------------------

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        write_csv_atomic(
            self.prices.drop(columns=["_adj_ratio"], errors="ignore"), folder / "prices.csv"
        )
        write_csv_atomic(self.actions, folder / "actions.csv")
        news = self.news.copy()
        news["published_at"] = news["published_at"].map(lambda x: x.isoformat())
        write_csv_atomic(news, folder / "news.csv")
        write_csv_atomic(self.earnings, folder / "earnings.csv")
        write_csv_atomic(self.rates, folder / "rates.csv")
        write_csv_atomic(self.members, folder / "members.csv")
        write_csv_atomic(self.sentiment, folder / "sentiment.csv")
        write_csv_atomic(self.base_caps, folder / "base_caps.csv")
        write_json_atomic(self.meta, folder / "meta.json")

    @classmethod
    def load(cls, folder: Path) -> SyntheticWorld:
        prices = read_csv(folder / "prices.csv")
        prices["date"] = pd.to_datetime(prices["date"]).dt.date
        actions = read_csv(folder / "actions.csv")
        if len(actions):
            actions["date"] = pd.to_datetime(actions["date"]).dt.date
        news = read_csv(folder / "news.csv")
        news["published_at"] = pd.to_datetime(news["published_at"], utc=True).map(
            lambda x: x.to_pydatetime()
        )
        earnings = read_csv(folder / "earnings.csv")
        earnings["date"] = pd.to_datetime(earnings["date"]).dt.date
        rates = read_csv(folder / "rates.csv")
        rates["date"] = pd.to_datetime(rates["date"]).dt.date
        members = read_csv(folder / "members.csv")
        sentiment = read_csv(folder / "sentiment.csv")
        base_caps = read_csv(folder / "base_caps.csv")
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        return cls(prices, actions, news, earnings, rates, members, sentiment, base_caps, meta)

    @property
    def weeks(self) -> list[str]:
        weeks = self.meta["weeks"]
        assert isinstance(weeks, list)
        return [str(w) for w in weeks]

    @property
    def stock_tickers(self) -> list[str]:
        return list(self.members["ticker"])


def _window_week_of_days(days: list[dt.date]) -> list[str]:
    """For each day, the week id whose return window (Monday open to Monday open) holds it."""
    out = []
    for d in days:
        # The window of week w starts at monday_open(w), which is after cutoff(w).
        # Find w such that monday_open(w) <= d < monday_open(next(w)).
        w = cal.week_id(d)
        # Day d in ISO week X belongs to the window of week X-1 (scores made at X-1's cutoff).
        w_prev = cal.prev_week_id(w)
        if d < cal.monday_open_day(w_prev):
            w_prev = cal.prev_week_id(w_prev)
        elif d >= cal.monday_open_day(w):
            w_prev = w
        out.append(w_prev)
    return out


def _plant_problems(
    prices: pd.DataFrame, tickers: list[str], days: list[dt.date], first_idx: int
) -> pd.DataFrame:
    df = prices
    # Delisting: no rows after the offset.
    tk, off, _ = PLANTED["delisting"]
    cut = days[first_idx + int(off)]  # type: ignore[arg-type]
    df = df[~((df["ticker"] == tk) & (df["date"] > cut))]
    # Halt: missing days.
    tk, off, n_missing = PLANTED["halt"]
    gone = set(days[first_idx + int(off) : first_idx + int(off) + int(n_missing)])  # type: ignore[arg-type]
    df = df[~((df["ticker"] == tk) & (df["date"].isin(gone)))]
    # Zero price on one day.
    tk, off, _ = PLANTED["zero_price"]
    d0 = days[first_idx + int(off)]  # type: ignore[arg-type]
    mask = (df["ticker"] == tk) & (df["date"] == d0)
    df.loc[mask, ["open", "high", "low", "close", "adj_close"]] = 0.0
    # Stale: identical closes for several days.
    tk, off, n_stale = PLANTED["stale_price"]
    stale_days = set(days[first_idx + int(off) : first_idx + int(off) + int(n_stale)])  # type: ignore[arg-type]
    mask = (df["ticker"] == tk) & (df["date"].isin(stale_days))
    first_val = df.loc[mask, "close"].iloc[0]
    first_adj = df.loc[mask, "adj_close"].iloc[0]
    df.loc[mask, "close"] = first_val
    df.loc[mask, "adj_close"] = first_adj
    # Duplicate rows on one day.
    tk, off, _ = PLANTED["duplicate_rows"]
    d1 = days[first_idx + int(off)]  # type: ignore[arg-type]
    dup = df[(df["ticker"] == tk) & (df["date"] == d1)]
    df = pd.concat([df, dup], ignore_index=True)
    # Short history: drop rows before the (negative) offset.
    tk, off, _ = PLANTED["short_history"]
    start = days[first_idx + int(off)]  # type: ignore[arg-type]
    df = df[~((df["ticker"] == tk) & (df["date"] < start))]
    return df.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)


def _make_news(
    rng: np.random.Generator,
    tickers: list[str],
    members: pd.DataFrame,
    weeks: list[str],
    sentiment: np.ndarray,
    has_news: np.ndarray,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    names = dict(zip(members["ticker"], members["name"]))
    sources = ["SynWire", "Synthetic Times", "FakeStreet Journal"]
    rows: list[dict[str, object]] = []
    sent_rows: list[dict[str, object]] = []
    for k, w in enumerate(weeks):
        cutoff = cal.cutoff_day(w)
        prev_cutoff = cal.cutoff_day(cal.prev_week_id(w))
        start_utc = dt.datetime.combine(prev_cutoff, dt.time(16, 0), tzinfo=ET).astimezone(UTC)
        end_utc = dt.datetime.combine(cutoff, dt.time(16, 0), tzinfo=ET).astimezone(UTC)
        span = (end_utc - start_utc).total_seconds()
        for i, tk in enumerate(tickers):
            s = int(sentiment[k, i])
            sent_rows.append(
                {"week_id": w, "ticker": tk, "sentiment": s, "has_news": bool(has_news[k, i])}
            )
            if not has_news[k, i]:
                continue
            n_articles = int(rng.integers(1, 4))
            for _ in range(n_articles):
                offset = float(rng.uniform(1, span - 1))
                published = start_utc + dt.timedelta(seconds=offset)
                headline, summary = _article_text(rng, names[tk], s)
                rows.append(
                    {
                        "ticker": tk,
                        "published_at": published,
                        "headline": headline,
                        "summary": summary,
                        "source": sources[int(rng.integers(0, len(sources)))],
                    }
                )
            # Occasionally duplicate an article with a trivially different headline.
            if rng.random() < 0.15:
                last = rows[-1]
                dup = dict(last)
                dup["headline"] = str(last["headline"]).upper()
                dup["published_at"] = last["published_at"] + dt.timedelta(minutes=5)  # type: ignore[operator]
                rows.append(dup)
        # Late Friday articles after the cutoff carry next week's sentiment and must be
        # excluded from this week's snapshot (leakage test).
        if k + 1 < len(weeks):
            for i, tk in enumerate(tickers):
                if rng.random() < LATE_NEWS_PROB and has_news[k + 1, i]:
                    s_next = int(sentiment[k + 1, i])
                    headline, summary = _article_text(rng, names[tk], s_next)
                    published = end_utc + dt.timedelta(minutes=float(rng.uniform(30, 360)))
                    rows.append(
                        {
                            "ticker": tk,
                            "published_at": published,
                            "headline": headline,
                            "summary": summary,
                            "source": sources[int(rng.integers(0, len(sources)))],
                        }
                    )
    news = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    return news.sort_values(["ticker", "published_at"], kind="stable").reset_index(
        drop=True
    ), sent_rows


def _article_text(rng: np.random.Generator, name: str, s: int) -> tuple[str, str]:
    n_pos = max(0, s) + (1 if rng.random() < 0.15 else 0)
    n_neg = max(0, -s) + (1 if rng.random() < 0.15 else 0)
    words = (
        list(rng.choice(POSITIVE_WORDS, n_pos))
        + list(rng.choice(NEGATIVE_WORDS, n_neg))
        + list(rng.choice(NEUTRAL_WORDS, 1))
    )
    rng.shuffle(words)
    headline = f"{name} {' '.join(words)} in quarterly update"
    summary = f"{name} {' and '.join(words)} according to the company statement."
    return headline, summary


# Providers ----------------------------------------------------------------------------------


class SyntheticPriceProvider(PriceProvider):
    name = "synthetic"

    def __init__(self, world: SyntheticWorld) -> None:
        self.world = world

    def get_prices(self, tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
        p = self.world.prices
        mask = p["ticker"].isin(tickers) & (p["date"] >= start) & (p["date"] <= end)
        out = p.loc[mask, PRICE_COLUMNS].copy()
        return out.reset_index(drop=True)

    def get_corporate_actions(
        self, tickers: list[str], start: dt.date, end: dt.date
    ) -> pd.DataFrame:
        a = self.world.actions
        if len(a) == 0:
            return empty_frame(ACTION_COLUMNS)
        mask = a["ticker"].isin(tickers) & (a["date"] >= start) & (a["date"] <= end)
        return a.loc[mask, ACTION_COLUMNS].copy().reset_index(drop=True)


class SyntheticNewsProvider(NewsProvider):
    name = "synthetic"

    def __init__(self, world: SyntheticWorld) -> None:
        self.world = world

    def get_news(
        self, tickers: list[str], start_utc: dt.datetime, end_utc: dt.datetime
    ) -> pd.DataFrame:
        n = self.world.news
        if len(n) == 0:
            return empty_frame(NEWS_COLUMNS)
        pub = pd.to_datetime(n["published_at"], utc=True)
        mask = n["ticker"].isin(tickers) & (pub > start_utc) & (pub <= end_utc)
        return n.loc[mask, NEWS_COLUMNS].copy().reset_index(drop=True)


class SyntheticReferenceProvider(ReferenceProvider):
    name = "synthetic"

    def __init__(self, world: SyntheticWorld) -> None:
        self.world = world

    def get_reference(self, tickers: list[str], as_of: dt.date) -> pd.DataFrame:
        w = self.world
        members = w.members.set_index("ticker")
        caps = w.base_caps.set_index("ticker")
        # Latest adjusted close at or before as_of gives the market cap path.
        p = w.prices[(w.prices["date"] <= as_of) & w.prices["ticker"].isin(tickers)]
        last = p.sort_values("date").groupby("ticker")["adj_close"].last()
        rows = []
        for tk in tickers:
            if tk not in members.index:
                continue
            cap = float("nan")
            if tk in last.index and tk in caps.index:
                cap = float(caps.loc[tk, "base_cap"] * last[tk] / caps.loc[tk, "base_price"])
            e = w.earnings[(w.earnings["ticker"] == tk) & (w.earnings["date"] > as_of)]
            nxt = e["date"].min() if len(e) else None
            rows.append(
                {
                    "ticker": tk,
                    "name": members.loc[tk, "name"],
                    "sector": members.loc[tk, "sector"],
                    "market_cap": cap,
                    "next_earnings_date": nxt,
                }
            )
        return pd.DataFrame(rows, columns=REFERENCE_COLUMNS)


class SyntheticRateProvider(RateProvider):
    name = "synthetic"

    def __init__(self, world: SyntheticWorld) -> None:
        self.world = world

    def get_rates(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        r = self.world.rates
        mask = (r["date"] >= start) & (r["date"] <= end)
        return r.loc[mask, RATE_COLUMNS].copy().reset_index(drop=True)


class SyntheticMembersProvider(MembersProvider):
    name = "synthetic"

    def __init__(self, world: SyntheticWorld) -> None:
        self.world = world

    def get_members(self, as_of: dt.date) -> pd.DataFrame:
        return self.world.members.loc[:, MEMBER_COLUMNS].copy()


def synthetic_bundle(world: SyntheticWorld) -> ProviderBundle:
    return ProviderBundle(
        prices=SyntheticPriceProvider(world),
        news=SyntheticNewsProvider(world),
        reference=SyntheticReferenceProvider(world),
        rates=SyntheticRateProvider(world),
        members=SyntheticMembersProvider(world),
    )


def load_or_generate_world(
    cfg: SyntheticConfig, folder: Path, first_week: str = "2026-W02"
) -> SyntheticWorld:
    """Load the saved world if it exists and matches the seed, otherwise generate and save it."""
    meta = folder / "meta.json"
    if meta.exists():
        stored = json.loads(meta.read_text(encoding="utf-8"))
        if (
            stored.get("seed") == cfg.seed
            and stored.get("first_week") == first_week
            and stored.get("n_tickers") == cfg.n_tickers
            and stored.get("history_days") == cfg.history_days
            and stored.get("future_weeks") == cfg.future_weeks
        ):
            return SyntheticWorld.load(folder)
    world = SyntheticWorld.generate(cfg, first_week=first_week)
    world.save(folder)
    return world
