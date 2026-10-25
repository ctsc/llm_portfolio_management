"""NYSE trading calendar helpers and ISO week labels.

Uses pandas_market_calendars when installed, otherwise a built-in holiday table for
2024 through 2027 (full closures and early closes).

Week labels: `week_id` is the ISO week of the cutoff Friday. Scores made at that cutoff
trade at the next Monday open and the return window runs to the following Monday open.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from functools import lru_cache

import pandas as pd

# Full-day closures, by year. Weekend holidays are listed on their observed weekday.
_HOLIDAYS: dict[int, list[str]] = {
    2024: [
        "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19",
        "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    ],
    2025: [
        "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
        "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    ],
    2026: [
        "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
        "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    ],
    2027: [
        "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
        "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
    ],
}

# Early closes (13:00 ET). The day still counts as a trading day.
_EARLY_CLOSES: dict[int, list[str]] = {
    2024: ["2024-07-03", "2024-11-29", "2024-12-24"],
    2025: ["2025-07-03", "2025-11-28", "2025-12-24"],
    2026: ["2026-11-27", "2026-12-24"],
    2027: ["2027-11-26"],
}

SUPPORTED_YEARS = sorted(_HOLIDAYS)


def _to_date(value: dt.date | str | pd.Timestamp) -> dt.date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


@lru_cache(maxsize=1)
def _holiday_set() -> frozenset[dt.date]:
    return frozenset(dt.date.fromisoformat(d) for days in _HOLIDAYS.values() for d in days)


@lru_cache(maxsize=1)
def _early_close_set() -> frozenset[dt.date]:
    return frozenset(dt.date.fromisoformat(d) for days in _EARLY_CLOSES.values() for d in days)


def _builtin_is_trading_day(day: dt.date) -> bool:
    if day.year not in _HOLIDAYS:
        raise ValueError(
            f"built-in NYSE calendar covers {SUPPORTED_YEARS[0]} to {SUPPORTED_YEARS[-1]}, "
            f"got {day.isoformat()}; install pandas_market_calendars for other years"
        )
    return day.weekday() < 5 and day not in _holiday_set()


def _pmc_days(start: dt.date, end: dt.date) -> list[dt.date] | None:
    try:
        import pandas_market_calendars as pmc  # type: ignore[import-not-found]
    except ImportError:
        return None
    nyse = pmc.get_calendar("NYSE")
    sched = nyse.schedule(start_date=start.isoformat(), end_date=end.isoformat())
    return [ts.date() for ts in sched.index]


def trading_days(start: dt.date | str, end: dt.date | str) -> list[dt.date]:
    """Trading days in [start, end], inclusive."""
    s, e = _to_date(start), _to_date(end)
    if e < s:
        return []
    pmc_days = _pmc_days(s, e)
    if pmc_days is not None:
        return pmc_days
    days: list[dt.date] = []
    day = s
    while day <= e:
        if _builtin_is_trading_day(day):
            days.append(day)
        day += dt.timedelta(days=1)
    return days


def is_trading_day(day: dt.date | str) -> bool:
    d = _to_date(day)
    return len(trading_days(d, d)) == 1


def is_early_close(day: dt.date | str) -> bool:
    return _to_date(day) in _early_close_set()


def prev_trading_day(day: dt.date | str, inclusive: bool = False) -> dt.date:
    d = _to_date(day)
    if not inclusive:
        d -= dt.timedelta(days=1)
    while not is_trading_day(d):
        d -= dt.timedelta(days=1)
    return d


def next_trading_day(day: dt.date | str, inclusive: bool = False) -> dt.date:
    d = _to_date(day)
    if not inclusive:
        d += dt.timedelta(days=1)
    while not is_trading_day(d):
        d += dt.timedelta(days=1)
    return d


def week_id(day: dt.date | str) -> str:
    """ISO week label like 2026-W46 for the week containing `day`."""
    iso = _to_date(day).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def week_friday(week: str) -> dt.date:
    """The calendar Friday of an ISO week label."""
    year_s, week_s = week.split("-W")
    return dt.date.fromisocalendar(int(year_s), int(week_s), 5)


def cutoff_day(week_friday_or_id: dt.date | str) -> dt.date:
    """DECISION(A4): the Friday, or if closed the last prior trading day."""
    friday = (
        week_friday(week_friday_or_id)
        if isinstance(week_friday_or_id, str) and "-W" in week_friday_or_id
        else _to_date(week_friday_or_id)
    )
    return prev_trading_day(friday, inclusive=True)


def monday_open_day(week: str) -> dt.date:
    """The Monday after the week's cutoff Friday, or the next trading day if closed."""
    monday = week_friday(week) + dt.timedelta(days=3)
    return next_trading_day(monday, inclusive=True)


def next_week_id(week: str) -> str:
    return week_id(week_friday(week) + dt.timedelta(days=7))


def prev_week_id(week: str) -> str:
    return week_id(week_friday(week) - dt.timedelta(days=7))


def window_end_date(week: str) -> dt.date:
    """The Monday open that closes the return window for scores made at `week`'s cutoff."""
    return monday_open_day(next_week_id(week))


def week_range(first_week: str, n_weeks: int) -> list[str]:
    weeks = [first_week]
    for _ in range(n_weeks - 1):
        weeks.append(next_week_id(weeks[-1]))
    return weeks


def weeks_between(first_week: str, last_week: str) -> list[str]:
    weeks = [first_week]
    while weeks[-1] != last_week:
        weeks.append(next_week_id(weeks[-1]))
        if len(weeks) > 1000:
            raise ValueError("week range too long or last_week before first_week")
    return weeks


def week_alignment(week: str) -> dict[str, str]:
    """Explicit week alignment: cutoff, trade date, and window end date."""
    return {
        "week_id": week,
        "cutoff_date": cutoff_day(week).isoformat(),
        "trade_date": monday_open_day(week).isoformat(),
        "window_end_date": window_end_date(week).isoformat(),
    }


def to_dates(values: Iterable[dt.date | str]) -> list[dt.date]:
    return [_to_date(v) for v in values]
