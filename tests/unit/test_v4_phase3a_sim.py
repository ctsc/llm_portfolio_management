"""Phase 3A tests: open-to-open weekly returns (N4), stop reset on rebalance day (G1b),
no-price liquidation (U2), and removed-ticker liquidation (U1)."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from portstudy.config import SimulatorConfig
from portstudy.eval.data import weekly_close_returns, weekly_open_returns
from portstudy.sim.simulator import (
    PortfolioSim,
    PriceBook,
    WeekPlan,
    replay,
    simulate_week,
)

CFG = SimulatorConfig(
    initial_value=100000.0,
    cost_bps=10.0,
    stop_loss=-0.08,
    take_profit=0.15,
    no_price_liquidation_days=5,
)


def _ledger(
    rows: dict[str, list[tuple[float | None, float | None]]], days: list[dt.date]
) -> pd.DataFrame:
    out = []
    for tk, ocs in rows.items():
        for d, (o, c) in zip(days, ocs):
            if o is None:
                continue
            out.append(
                {
                    "date": d,
                    "ticker": tk,
                    "open": o,
                    "high": max(o, c),
                    "low": min(o, c),
                    "close": c,
                    "volume": 1000.0,
                }
            )
    return pd.DataFrame(out)


# Three consecutive weeks: trade Mondays 3/9, 3/16, 3/23; closing Monday 3/30.
# Stock A has a large Friday-close to Monday-open gap each weekend.
W10, W11, W12 = "2026-W10", "2026-W11", "2026-W12"
W13 = "2026-W13"
D10 = [dt.date(2026, 3, d) for d in (9, 10, 11, 12, 13)]
D11 = [dt.date(2026, 3, d) for d in (16, 17, 18, 19, 20)]
D12 = [dt.date(2026, 3, d) for d in (23, 24, 25, 26, 27)]
D13 = [dt.date(2026, 3, d) for d in (30, 31)] + [dt.date(2026, 4, d) for d in (1, 2, 3)]


def _gap_book(with_w13: bool = True) -> tuple[PriceBook, list[dt.date]]:
    # A: flat within each week at 100, but each Monday opens 50% above the prior Friday close
    # so the weekend gap is huge while the open-to-open holding return is 0%.
    a10 = [(100.0, 100.0)] * 5
    a11 = [(150.0, 150.0)] * 5
    a12 = [(225.0, 225.0)] * 5
    a13 = [(337.5, 337.5)] * 5
    b = [(50.0, 50.0)] * 5
    rows = {"A": a10 + a11 + a12 + (a13 if with_w13 else []), "B": b * (4 if with_w13 else 3)}
    days = D10 + D11 + D12 + (D13 if with_w13 else [])
    return PriceBook(_ledger(rows, days)), days


def _plans(n: int = 3) -> list[WeekPlan]:
    w = pd.Series({"A": 0.5, "B": 0.5})
    return [WeekPlan(x, w, "llm", "normal") for x in (W10, W11, W12)][:n]


def test_open_to_open_ignores_weekend_gap() -> None:
    book, _ = _gap_book()
    sim = replay("p", _plans(), book, CFG)
    assert len(sim.week_opens) == 4  # three weeks plus the closing entry
    closes = pd.DataFrame([r.to_dict() for r in sim.daily])
    close_ret = weekly_close_returns(closes, CFG.initial_value)
    # The first holding week's close-to-close return is moved by the +50% weekend gap
    # (A is half the book, so ~+25% vs ~0 without the gap).
    assert close_ret[W11] > 0.15
    open_ret = weekly_open_returns(
        pd.DataFrame({"week_id": w, "date": d, "value_open": v} for w, d, v in sim.week_opens)
    )
    # Open-to-open: the gap is inside the week boundary; W11 return is ~0 net of rebalance cost.
    assert abs(open_ret[W11]) < 0.01


def test_open_to_open_matches_hand_calculation_with_cost() -> None:
    book, _ = _gap_book()
    sim = replay("p", _plans(1), book, CFG)
    v_open = sim.week_opens[0][2]
    assert v_open == pytest.approx(100000.0)
    closing = sim.week_opens[-1]
    assert closing[0] == W11
    # Hand calc: each 50k buy pays 10 bps cost. A holds 499.5 sh that gap to 150 at the
    # closing open -> 499.5*150 + 999*50 - leftover cash 0.
    a_sh = 50000 / (100.0 * 1.001)  # fill price includes the 10 bps cost
    b_sh = 50000 / (50.0 * 1.001)
    expected = a_sh * 150.0 + b_sh * 50.0
    assert closing[2] == pytest.approx(expected, rel=1e-6)
    ret = weekly_open_returns(
        pd.DataFrame({"week_id": w, "date": d, "value_open": v} for w, d, v in sim.week_opens)
    )
    assert ret[W10] == pytest.approx(expected / 100000.0 - 1.0, rel=1e-6)


def test_no_rebalance_week_still_gets_entry() -> None:
    book, _ = _gap_book()
    sim = replay(
        "p",
        [
            WeekPlan(W10, pd.Series({"A": 0.5, "B": 0.5}), "llm", "normal"),
            WeekPlan(W11, None, "hold", "no_rebalance"),
            WeekPlan(W12, None, "hold", "no_rebalance"),
        ],
        book,
        CFG,
    )
    ids = [w for w, _, _ in sim.week_opens]
    assert ids == [W10, W11, W12, W13]


def test_closing_entry_only_when_next_monday_in_book() -> None:
    book, _ = _gap_book(with_w13=False)
    sim = replay("p", _plans(), book, CFG)
    assert [w for w, _, _ in sim.week_opens] == [W10, W11, W12]


def test_daily_value_open_only_on_trade_day() -> None:
    book, _ = _gap_book()
    sim = replay("p", _plans(1), book, CFG)
    opens = [r.value_open for r in sim.daily]
    assert opens[0] is not None and all(v is None for v in opens[1:])


# G1b: the review's two-stock scenario.


def _rise_book(rise: float, flat_after_days: int | None = None) -> PriceBook:
    # W47 trades Mon 2026-11-23, W48 trades Mon 2026-11-30; A rises `rise` per day for the
    # first `flat_after_days` book days (or forever), B flat.
    days = [dt.date(2026, 11, 23 + i) for i in range(5)] + [
        dt.date(2026, 11, 30) + dt.timedelta(days=i) for i in range(5)
    ]
    a_price = 100.0
    a_rows = []
    for i in range(10):
        if flat_after_days is not None and i >= flat_after_days:
            a_rows.append((a_price, a_price))
        else:
            a_rows.append((a_price, a_price * (1 + rise)))
            a_price = a_price * (1 + rise)
    rows = {"A": a_rows, "B": [(50.0, 50.0)] * 10}
    return PriceBook(_ledger(rows, days))


def test_stop_on_rebalance_day_not_sold_tuesday() -> None:
    # Review C1: A rises 3% a day for six days then flat; at the W47 rebalance A has crossed
    # the take-profit (15% cumulative), the rebalance trims it to target, and the reset
    # avg_cost must keep the stop from firing again so nothing is sold Tuesday and no cash
    # sits idle for the rest of the week.
    book = _rise_book(0.03, flat_after_days=6)
    sim = PortfolioSim("p", CFG)
    weights = pd.Series({"A": 0.5, "B": 0.5})
    simulate_week(sim, WeekPlan("2026-W47", weights, "llm", "normal"), book)
    # Force A to have run up: after week 1 A has risen ~3%/day x5 = ~16% > take_profit, so a
    # take_profit pending exists going into the W48 rebalance.
    w48_first_trade_date = dt.date(2026, 11, 30)
    n_trades_before = len(sim.trades)
    simulate_week(sim, WeekPlan("2026-W48", weights, "llm", "normal"), book)
    new_trades = sim.trades[n_trades_before:]
    # All week-2 trades happen on the Monday; A is flat after day 6 so no Tuesday sale.
    assert all(t.date == w48_first_trade_date for t in new_trades)
    # No idle cash: at the last close the portfolio is fully invested.
    assert sim.daily[-1].cash == pytest.approx(0.0, abs=1.0)


def test_further_rise_still_triggers_take_profit() -> None:
    # After the reset, a stock that rises another 15% from Monday's open still triggers.
    book = _rise_book(0.20)  # 20%/day forever
    sim = PortfolioSim("p", CFG)
    weights = pd.Series({"A": 0.5, "B": 0.5})
    simulate_week(sim, WeekPlan("2026-W47", weights, "llm", "normal"), book)
    n0 = len(sim.trades)
    simulate_week(sim, WeekPlan("2026-W48", weights, "llm", "normal"), book)
    # A keeps rising 20%/day from the new avg_cost: a take_profit pending fires again and the
    # stock is sold on Tuesday (pending sale executes the next day with a price).
    later = [t for t in sim.trades[n0:] if t.date > dt.date(2026, 11, 30)]
    assert any(t.reason in ("take_profit", "rebalance") for t in later)


# U2 / U1 liquidation.


def test_no_price_liquidation_on_fifth_missing_day() -> None:
    # B appears only on Monday of week 1 then disappears from the ledger.
    days = D10 + D11 + D12
    rows = {
        "A": [(100.0, 100.0)] * 15,
        "B": [(50.0, 50.0)] + [(None, None)] * 14,
    }
    book = PriceBook(_ledger(rows, days))
    sim = PortfolioSim("p", CFG)
    simulate_week(sim, WeekPlan(W10, pd.Series({"A": 0.5, "B": 0.5}), "llm", "normal"), book)
    simulate_week(sim, WeekPlan(W11, pd.Series({"A": 1.0}), "llm", "normal"), book)
    simulate_week(sim, WeekPlan(W12, pd.Series({"A": 1.0}), "llm", "normal"), book)
    liq = [t for t in sim.trades if t.reason == "no_price_liquidation"]
    assert len(liq) == 1 and liq[0].ticker == "B"
    assert liq[0].fill_price == pytest.approx(50.0)  # last close
    # The liquidation happens on the 5th missing day (Tue-Fri of week 1 = 4, Mon of week 2 = 5),
    # after that week's rebalance; its cash is invested at the next rebalance.
    assert liq[0].date == dt.date(2026, 3, 16)
    buys = [t for t in sim.trades if t.ticker == "A" and t.shares > 0]
    assert buys[-1].date == dt.date(2026, 3, 23)


def test_short_outage_not_liquidated_exits_via_forced_exit() -> None:
    # B missing for 2 days then priced again: no liquidation; the forced-exit pending sells it
    # at the next priced open.
    days = D10
    rows = {
        "A": [(100.0, 100.0)] * 5,
        "B": [(50.0, 50.0), (None, None), (None, None), (48.0, 48.0), (48.0, 48.0)],
    }
    book = PriceBook(_ledger(rows, days))
    sim = PortfolioSim("p", CFG)
    simulate_week(sim, WeekPlan(W10, pd.Series({"A": 0.5, "B": 0.5}), "llm", "normal"), book)
    assert not any(t.reason == "no_price_liquidation" for t in sim.trades)
    assert any(
        t.ticker == "B" and t.date == dt.date(2026, 3, 12) and t.reason == "forced_exit"
        for t in sim.trades
    )


def test_removed_ticker_without_monday_open_liquidated_at_rebalance() -> None:
    days = D10 + D11
    rows = {
        "A": [(100.0, 100.0)] * 10,
        "B": [(50.0, 50.0)] * 5 + [(None, None)] * 5,  # gone in week 2
    }
    book = PriceBook(_ledger(rows, days))
    sim = PortfolioSim("p", CFG)
    simulate_week(sim, WeekPlan(W10, pd.Series({"A": 0.5, "B": 0.5}), "llm", "normal"), book)
    simulate_week(
        sim,
        WeekPlan(W11, pd.Series({"A": 1.0}), "llm", "normal", removed=frozenset({"B"})),
        book,
    )
    liq = [t for t in sim.trades if t.reason == "no_price_liquidation"]
    assert len(liq) == 1 and liq[0].ticker == "B"
    assert liq[0].date == dt.date(2026, 3, 16)  # liquidated at the rebalance, not day 5
