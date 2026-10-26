"""Simulator entry points: build the price book from the ledger and replay portfolios."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from portstudy.config import StudyConfig
from portstudy.data import ledger
from portstudy.sim.records import records_hash, write_records
from portstudy.sim.simulator import PortfolioSim, PriceBook, WeekPlan, replay

__all__ = ["PortfolioSim", "PriceBook", "WeekPlan", "ledger_book", "replay", "replay_all"]


def ledger_book() -> PriceBook:
    return PriceBook(ledger.read_ledger(), ledger.read_actions())


def replay_all(
    cfg: StudyConfig,
    plans_by_portfolio: Mapping[str, Sequence[WeekPlan]],
    book: PriceBook | None = None,
) -> dict[str, dict[str, str]]:
    """Replay every portfolio from its stored weekly plans and write trades.csv and daily.csv.

    Safe to re-run: each portfolio is rebuilt from scratch, deterministically, from the ledger.
    Returns the file hashes per portfolio.
    """
    book = book or ledger_book()
    hashes: dict[str, dict[str, str]] = {}
    for portfolio_id, plans in plans_by_portfolio.items():
        sim = replay(portfolio_id, plans, book, cfg.simulator)
        write_records(portfolio_id, sim.trades, sim.daily)
        hashes[portfolio_id] = records_hash(portfolio_id)
    return hashes
