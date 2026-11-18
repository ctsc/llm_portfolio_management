"""End-of-study closeout: final ledger refresh, final J8, and the study_complete marker.

Runs once `timing.live_end_date` (or an explicit `--week`) has fully closed: the ledger must
already hold the Monday open after the last live week so every portfolio's final
open-to-open return exists (N4).
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

from portstudy import calendar as cal
from portstudy.data import ledger
from portstudy.io import write_json_atomic
from portstudy.paths import outputs_dir, portfolios_dir
from portstudy.pipeline.context import RunContext
from portstudy.pipeline.jobs import J8_replay_score, refresh_ledger
from portstudy.sim.records import read_week_opens


def run_closeout(
    ctx: RunContext,
    week: str | None,
    logger: logging.Logger | None = None,
) -> Path:
    """Run the closeout sequence; returns the path to study_complete.json.

    Raises CloseoutNotReady (shown by the CLI, exit code 1) when the ledger does not yet hold
    the Monday open after the last week, or when J8 or a closing entry is missing.
    """
    log = logger or logging.getLogger("portstudy.closeout")
    last_week = week
    if last_week is None:
        if not ctx.cfg.timing.live_end_date:
            raise CloseoutNotReady("no --week given and timing.live_end_date is not set")
        last_week = cal.week_id(dt.date.fromisoformat(str(ctx.cfg.timing.live_end_date)))

    now_date = ctx.now().date()
    through = cal.prev_trading_day(now_date, inclusive=True)
    refresh_ledger(ctx, through)

    required = cal.window_end_date(last_week)
    book_dates = ledger.read_ledger()
    last_date = book_dates["date"].max() if len(book_dates) else None
    if last_date is None or last_date < required:
        raise CloseoutNotReady(f"closeout can run on or after {required.isoformat()}")

    status = J8_replay_score(ctx, last_week)
    if status.state != "done":
        raise CloseoutNotReady(f"J8 did not complete: {status.state} {status.error or ''}")

    # Every portfolio must carry the closing entry for the week after the last week.
    next_week = cal.next_week_id(last_week)
    portfolios = sorted(p.name for p in portfolios_dir().iterdir() if p.is_dir())
    missing = [
        p for p in portfolios if next_week not in set(read_week_opens(p)["week_id"].astype(str))
    ]
    if missing:
        raise CloseoutNotReady(
            f"week_opens.csv missing closing entry {next_week} for: {', '.join(missing)}"
        )

    marker = {
        "completed_at": ctx.now().isoformat(),
        "last_week": last_week,
        "ledger_last_date": last_date.isoformat(),
        "n_portfolios": len(portfolios),
    }
    out = outputs_dir() / "study_complete.json"
    write_json_atomic(marker, out)
    log.info("study_complete", extra={"extra_fields": marker})
    return out


class CloseoutNotReady(Exception):
    """The closeout ran before its inputs were complete; the message is shown to the user."""
