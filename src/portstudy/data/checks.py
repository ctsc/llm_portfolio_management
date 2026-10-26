"""Data quality checks. Each returns a list of Issue records; each code maps to a config rule."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from portstudy.config import CheckRules
from portstudy.types import Issue

CHECK_CODES = [
    "missing_days",
    "nonpositive_price",
    "extreme_move",
    "duplicate_rows",
    "stale_price",
    "no_news",
    "news_after_cutoff",
    "short_history",
    "stopped_trading",
]


def check_prices(
    prices: pd.DataFrame,
    expected_days: list[dt.date],
    rules: CheckRules,
    actions: pd.DataFrame | None = None,
    min_history_days: int = 300,
) -> list[Issue]:
    """Run every price check on a long prices frame (date, ticker, open, ..., adj_close)."""
    issues: list[Issue] = []
    split_days: set[tuple[str, dt.date]] = set()
    if actions is not None and len(actions):
        splits = actions[actions["action"] == "split"]
        split_days = {(str(t), d) for t, d in zip(splits["ticker"], splits["date"])}
    expected = set(expected_days)
    for ticker, g in prices.groupby("ticker", sort=True):
        tk = str(ticker)
        g = g.sort_values("date", kind="stable")
        dates = list(g["date"])

        # Duplicate rows for the same date.
        dup = g["date"].duplicated()
        if dup.any():
            issues.append(
                Issue(
                    "error",
                    tk,
                    "duplicate_rows",
                    f"{int(dup.sum())} duplicate date rows",
                    rules.rule_for("duplicate_rows"),
                )
            )
        g = g.drop_duplicates("date", keep="first")

        # Short history: fewer than the minimum number of days.
        if len(g) < min_history_days:
            issues.append(
                Issue(
                    "error",
                    tk,
                    "short_history",
                    f"{len(g)} days of history, need {min_history_days}",
                    rules.rule_for("short_history"),
                    {"n_days": len(g)},
                )
            )

        # Stopped trading: no row on the last expected day (delisted or suspended).
        if dates and expected_days and dates[-1] < expected_days[-1]:
            issues.append(
                Issue(
                    "error",
                    tk,
                    "stopped_trading",
                    f"last price {dates[-1].isoformat()} is before the cutoff",
                    rules.rule_for("stopped_trading"),
                    {"last_date": dates[-1].isoformat()},
                )
            )

        # Missing days within the stock's own span.
        if dates:
            span = {d for d in expected if dates[0] <= d <= dates[-1]}
            missing = sorted(span - set(dates))
            if missing:
                issues.append(
                    Issue(
                        "warning",
                        tk,
                        "missing_days",
                        f"{len(missing)} missing trading days",
                        rules.rule_for("missing_days"),
                        {"dates": [d.isoformat() for d in missing[:10]]},
                    )
                )

        # Zero or negative prices.
        price_cols = ["open", "high", "low", "close", "adj_close"]
        bad = (g[price_cols] <= 0).any(axis=1)
        if bad.any():
            issues.append(
                Issue(
                    "error",
                    tk,
                    "nonpositive_price",
                    f"{int(bad.sum())} rows with zero or negative prices",
                    rules.rule_for("nonpositive_price"),
                    {"dates": [d.isoformat() for d in g.loc[bad, "date"].head(10)]},
                )
            )

        # Extreme daily move on the raw close without a split flag on that day.
        good = g[~bad]
        if len(good) > 1:
            ret = good["close"].pct_change().abs()
            hit_dates = good.loc[ret > rules.extreme_move_threshold, "date"]
            hits = [d for d in hit_dates if (tk, d) not in split_days]
            if hits:
                issues.append(
                    Issue(
                        "error",
                        tk,
                        "extreme_move",
                        f"{len(hits)} daily moves above {rules.extreme_move_threshold:.0%} "
                        "without a split",
                        rules.rule_for("extreme_move"),
                        {"dates": [d.isoformat() for d in hits[:10]]},
                    )
                )

        # Stale prices: close unchanged for stale_days consecutive days.
        if len(good) >= rules.stale_days:
            same = good["close"].diff().eq(0).to_numpy()
            longest, run = 0, 0
            for flag in same:
                run = run + 1 if flag else 0
                longest = max(longest, run)
            if longest >= rules.stale_days - 1:
                issues.append(
                    Issue(
                        "warning",
                        tk,
                        "stale_price",
                        f"close unchanged for {rules.stale_days} or more days",
                        rules.rule_for("stale_price"),
                    )
                )
    return issues


def check_news(
    news: pd.DataFrame,
    tickers: list[str],
    cutoff_utc: dt.datetime,
    rules: CheckRules,
) -> list[Issue]:
    """News checks: articles after the cutoff and stocks with no news."""
    issues: list[Issue] = []
    if len(news):
        pub = pd.to_datetime(news["published_at"], utc=True)
        late = news[pub > cutoff_utc]
        for ticker, g in late.groupby("ticker"):
            issues.append(
                Issue(
                    "error",
                    str(ticker),
                    "news_after_cutoff",
                    f"{len(g)} articles published after the cutoff",
                    rules.rule_for("news_after_cutoff"),
                )
            )
        with_news = set(news.loc[pub <= cutoff_utc, "ticker"])
    else:
        with_news = set()
    for tk in sorted(set(tickers) - with_news):
        issues.append(
            Issue("info", tk, "no_news", "no news in the window", rules.rule_for("no_news"))
        )
    return issues


def tickers_to_skip(issues: list[Issue]) -> set[str]:
    return {i.ticker for i in issues if i.rule == "skip_stock"}


def has_halt(issues: list[Issue]) -> bool:
    return any(i.rule == "halt" for i in issues)


def issues_frame(issues: list[Issue]) -> pd.DataFrame:
    cols = ["severity", "ticker", "code", "message", "rule"]
    if not issues:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([{c: getattr(i, c) for c in cols} for i in issues])
