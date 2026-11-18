"""Data quality checks. Each returns a list of Issue records; each code maps to a config rule."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from portstudy.config import CheckRules
from portstudy.types import Issue

CHECK_CODES = [
    "missing_days",
    "nonpositive_price",
    "large_move",
    "crash_removal",
    "unexplained_split",
    "duplicate_rows",
    "stale_price",
    "no_news",
    "news_after_cutoff",
    "short_history",
    "stopped_trading",
]


def _week_move_issues(
    tk: str,
    g: pd.DataFrame,
    since: dt.date | None,
    cutoff: dt.date,
    rules: CheckRules,
    split_days: set[tuple[str, dt.date]],
) -> list[Issue]:
    """U1: large-move flag, crash removal and unexplained-split checks, looking only
    at the days in (since, cutoff] with the last row at or before `since` as the base."""
    if since is None:
        return []
    issues: list[Issue] = []
    win = g[(g["date"] > since) & (g["date"] <= cutoff)].sort_values("date", kind="stable")
    base_rows = g[g["date"] <= since]
    if len(win) == 0 or len(base_rows) == 0:
        return issues
    base = base_rows.iloc[-1]
    last = win.iloc[-1]
    week_move = float(last["adj_close"]) / float(base["adj_close"]) - 1.0

    if abs(week_move) >= rules.large_move_flag:
        issues.append(
            Issue(
                "info",
                tk,
                "large_move",
                f"week move {week_move:+.1%} since {since.isoformat()}",
                "flag",
                {"return": week_move},
            )
        )

    # Single-day returns inside the window (raw and adjusted); the first window day is
    # measured against the base row at `since`.
    adj_vals = [float(base["adj_close"])] + [float(v) for v in win["adj_close"]]
    raw_vals = [float(base["close"])] + [float(v) for v in win["close"]]
    win_days = [d for d in win["date"]]
    adj_rets = [adj_vals[i] / adj_vals[i - 1] - 1.0 for i in range(1, len(adj_vals))]
    raw_rets = [raw_vals[i] / raw_vals[i - 1] - 1.0 for i in range(1, len(raw_vals))]

    worst = week_move
    worst_day = cutoff
    for day, r in zip(win_days, adj_rets):
        if r <= rules.crash_removal and r < worst:
            worst, worst_day = r, day
    if worst <= rules.crash_removal:
        issues.append(
            Issue(
                "error",
                tk,
                "crash_removal",
                f"adjusted return {worst:+.1%} at or below the crash limit",
                "remove",
                {"return": worst, "date": worst_day.isoformat()},
            )
        )
        return issues

    # Raw jump without an adjusted move and without a split action: a data error, skipped for
    # this week only (the stock is not removed from the study).
    for day, raw_r, adj_r in zip(win_days, raw_rets, adj_rets):
        if abs(raw_r) >= abs(rules.crash_removal) and (tk, day) not in split_days:
            if abs(adj_r) < abs(rules.crash_removal) / 2:
                issues.append(
                    Issue(
                        "error",
                        tk,
                        "unexplained_split",
                        f"raw move {raw_r:+.1%} without an adjusted move or a split action",
                        rules.unexplained_split,
                        {"return": raw_r, "date": day.isoformat()},
                    )
                )
    return issues


def check_prices(
    prices: pd.DataFrame,
    expected_days: list[dt.date],
    rules: CheckRules,
    actions: pd.DataFrame | None = None,
    min_history_days: int = 300,
    since: dt.date | None = None,
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

        # U1: week move, crash removal and unexplained splits inside (since, cutoff].
        good = g[~bad]
        if len(good) > 1 and expected_days:
            issues.extend(_week_move_issues(tk, good, since, expected_days[-1], rules, split_days))

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
    # "remove" also excludes the ticker from this week's snapshot (weight 0 at the Monday
    # rebalance, position sold at the open); it never comes back once recorded.
    return {i.ticker for i in issues if i.rule in ("skip_stock", "remove")}


def has_halt(issues: list[Issue]) -> bool:
    return any(i.rule == "halt" for i in issues)


def issues_frame(issues: list[Issue]) -> pd.DataFrame:
    cols = ["severity", "ticker", "code", "message", "rule"]
    if not issues:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([{c: getattr(i, c) for c in cols} for i in issues])
