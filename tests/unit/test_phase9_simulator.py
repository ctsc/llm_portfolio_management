"""Phase 9 tests: hand-calculated simulator example, stops, costs, actions, ledger isolation."""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portstudy import calendar as cal
from portstudy.config import SimulatorConfig, StudyConfig
from portstudy.data import ledger
from portstudy.sim import ledger_book, replay_all
from portstudy.sim.records import read_daily, read_trades, records_hash
from portstudy.sim.simulator import PortfolioSim, PriceBook, WeekPlan, replay, simulate_week
from portstudy.views.ic import weekly_returns

CFG = SimulatorConfig(initial_value=100000.0, cost_bps=10.0, stop_loss=-0.08, take_profit=0.15)
W10 = "2026-W10"  # cutoff Fri 2026-03-06, trade Mon 2026-03-09, window end Mon 2026-03-16
DAYS = [dt.date(2026, 3, d) for d in (9, 10, 11, 12, 13)]


def _ledger(rows: dict[str, list[tuple[float, float]]], days: list[dt.date] = DAYS) -> pd.DataFrame:
    """rows: ticker -> [(open, close), ...] per day."""
    out = []
    for tk, ocs in rows.items():
        for d, (o, c) in zip(days, ocs):
            if o is None:
                continue
            out.append({"date": d, "ticker": tk, "open": o, "high": max(o, c), "low": min(o, c),
                        "close": c, "volume": 1000.0})
    return pd.DataFrame(out)


HAND_ROWS = {
    "A": [(100, 102), (101, 101), (102, 103), (103, 92), (93, 95)],
    "B": [(50, 49), (49, 45.5), (45, 46), (46, 47), (47, 47)],
    "C": [(20, 21), (21, 23.1), (23, 23), (23, 23), (23, 23)],
}
HAND_WEIGHTS = pd.Series({"A": 0.5, "B": 0.3, "C": 0.19})


def test_hand_calculated_three_stock_five_day_example() -> None:
    book = PriceBook(_ledger(HAND_ROWS))
    sim = PortfolioSim("P", CFG)
    trades, daily = simulate_week(sim, WeekPlan(W10, HAND_WEIGHTS, "llm", "normal"), book)
    # Monday buys: 500 A at 100, 600 B at 50, 950 C at 20; cost 0.001 * 99000 = 99.
    buys = [t for t in trades if t.date == DAYS[0]]
    assert [(t.ticker, t.side, round(t.shares, 6), t.fill_price) for t in buys] == [
        ("A", "buy", 500.0, 100), ("B", "buy", 600.0, 50), ("C", "buy", 950.0, 20)]
    assert sum(t.cost_usd for t in buys) == pytest.approx(99.0)
    assert all(t.reason == "rebalance" and t.decision_source == "llm" for t in buys)
    assert [round(t.weight_after, 6) for t in buys] == [0.5, 0.3, 0.19]
    values = [round(r.value, 2) for r in daily]
    # Mon close: 901 + 51000 + 29400 + 19950
    # Tue close: 901 + 50500 + 27300 + 21945; B stop (45.5 <= 46) and C take-profit (23.1 >= 23)
    # Wed open: sell B 600 at 45 (cost 27), sell C 950 at 23 (cost 21.85); close A 103
    # Thu close: A 92 is exactly the 0.92 * 100 stop threshold
    # Fri open: sell A 500 at 93 (cost 46.5)
    assert values == [101251.00, 100646.00, 101202.15, 95702.15, 96155.65]
    sells = [t for t in trades if t.side == "sell"]
    assert [(t.date, t.ticker, t.reason, t.fill_price, round(t.cost_usd, 2)) for t in sells] == [
        (DAYS[2], "B", "stop_loss", 45, 27.0), (DAYS[2], "C", "take_profit", 23, 21.85),
        (DAYS[4], "A", "stop_loss", 93, 46.5)]
    assert round(sim.cumulative_cost, 2) == 194.35
    assert round(sim.cash, 2) == 96155.65 and not sim.positions
    assert daily[-1].n_positions == 0 and daily[1].n_positions == 3
    # Average cost is unchanged by sales and set by the buy price.
    assert sells[0].avg_entry_price_before == 50 and sells[0].avg_entry_price_after == 0.0


def test_fractional_shares_and_weights_match_targets() -> None:
    rows = {"A": [(33.33, 33.33)] * 5, "B": [(7.77, 7.77)] * 5, "C": [(101.01, 101.01)] * 5}
    book = PriceBook(_ledger(rows))
    sim = PortfolioSim("P", CFG)
    w = pd.Series({"A": 0.4, "B": 0.35, "C": 0.25})
    simulate_week(sim, WeekPlan(W10, w, "llm", "normal"), book)
    opens = book.open[DAYS[0]]
    # Weights sum to 1 so the cash rule scales every buy by the same factor (1 / 1.001).
    for t, target in w.items():
        assert sim.positions[t].shares * opens[t] / CFG.initial_value == pytest.approx(target / 1.001)
    assert sim.positions["A"].shares != round(sim.positions["A"].shares)
    assert sim.cash == pytest.approx(0.0, abs=1e-6)


def test_cash_shortfall_scales_buys_proportionally() -> None:
    rows = {"A": [(100, 100)] * 5, "B": [(100, 100)] * 5}
    book = PriceBook(_ledger(rows))
    sim = PortfolioSim("P", CFG)
    simulate_week(sim, WeekPlan(W10, pd.Series({"A": 0.6, "B": 0.4}), "llm", "normal"), book)
    scale = 1.0 / 1.001
    assert sim.positions["A"].shares == pytest.approx(600 * scale)
    assert sim.positions["B"].shares == pytest.approx(400 * scale)
    assert sim.cash == pytest.approx(0.0, abs=1e-8)


def test_holiday_monday_trades_on_tuesday() -> None:
    w = "2026-W07"  # Monday 2026-02-16 is Presidents Day
    assert cal.monday_open_day(w) == dt.date(2026, 2, 17)
    days = [dt.date(2026, 2, d) for d in (17, 18, 19, 20)]
    book = PriceBook(_ledger({"A": [(10, 10)] * 4}, days))
    sim = PortfolioSim("P", CFG)
    trades, daily = simulate_week(sim, WeekPlan(w, pd.Series({"A": 0.5}), "llm", "normal"), book)
    assert trades[0].date == dt.date(2026, 2, 17) and len(daily) == 4


def test_no_rebalance_week_holds_but_stops_still_run() -> None:
    book = PriceBook(_ledger(HAND_ROWS))
    sim = PortfolioSim("P", CFG)
    simulate_week(sim, WeekPlan(W10, HAND_WEIGHTS, "llm", "normal"), book)
    # Next week: hold, but a stock that breaches the stop at a close is still sold.
    w11 = "2026-W11"
    days = [dt.date(2026, 3, d) for d in (16, 17, 18, 19, 20)]
    book2 = PriceBook(_ledger({"A": [(95, 80), (80, 80), (80, 80), (80, 80), (80, 80)]}, days))
    sim2 = PortfolioSim("P", CFG)
    sim2.cash = 1000.0
    from portstudy.sim.simulator import Position
    sim2.positions["A"] = Position(100.0, 100.0, 95.0)
    trades, daily = simulate_week(sim2, WeekPlan(w11, None, "hold", "no_rebalance"), book2)
    assert len(trades) == 1 and trades[0].reason == "stop_loss" and trades[0].date == days[1]
    assert all(r.week_status == "no_rebalance" for r in daily)


def test_dividend_credits_cash_and_split_keeps_value() -> None:
    rows = {"A": [(100, 100), (100, 100), (50, 50), (50, 50), (50, 50)]}
    actions = pd.DataFrame({"date": [DAYS[1], DAYS[2]], "ticker": ["A", "A"],
                            "action": ["dividend", "split"], "value": [0.5, 2.0]})
    book = PriceBook(_ledger(rows), actions)
    sim = PortfolioSim("P", CFG)
    trades, daily = simulate_week(sim, WeekPlan(W10, pd.Series({"A": 0.5}), "llm", "normal"), book)
    # 500 shares bought at 100 (cost 50): cash 100000 - 50000 - 50 = 49950.
    assert daily[0].cash == pytest.approx(49950.0)
    # Ex-date Tuesday: 500 * 0.5 = 250 credited.
    assert daily[1].cash == pytest.approx(50200.0)
    # Split Wednesday: 1000 shares at avg cost 50; value unchanged from Tuesday's close.
    assert sim.positions["A"].shares == pytest.approx(1000.0)
    assert sim.positions["A"].avg_cost == pytest.approx(50.0)
    assert daily[2].value == pytest.approx(daily[1].value)
    # No stop fired: the split-adjusted average cost keeps the threshold right.
    assert len(trades) == 1


def test_stop_on_friday_close_before_rebalance_yields_one_net_trade() -> None:
    rows = {"A": [(100, 100), (100, 100), (100, 100), (100, 100), (100, 90)],
            "B": [(50, 50)] * 5}
    book = PriceBook(_ledger(rows))
    sim = PortfolioSim("P", CFG)
    simulate_week(sim, WeekPlan(W10, pd.Series({"A": 0.5, "B": 0.4}), "llm", "normal"), book)
    assert sim.pending_sales == {"A": "stop_loss"}
    days = [dt.date(2026, 3, d) for d in (16, 17, 18, 19, 20)]
    book2 = PriceBook(_ledger({"A": [(91, 91)] * 5, "B": [(50, 50)] * 5}, days))
    # Optimizer keeps A at a lower weight: exactly one trade for A at Monday's open.
    trades, _ = simulate_week(sim, WeekPlan("2026-W11", pd.Series({"A": 0.2, "B": 0.4}), "llm", "normal"), book2)
    a_trades = [t for t in trades if t.ticker == "A" and t.date == days[0]]
    assert len(a_trades) == 1
    t = a_trades[0]
    assert t.reason == "rebalance" and t.stop_trigger == "stop_loss" and t.side == "sell"
    assert t.date == days[0] and t.cost_usd == pytest.approx(0.001 * t.trade_value)
    # The average cost is unchanged by the partial sale, so Monday's close at 91 is a new stop.
    later = [x for x in trades if x.ticker == "A" and x.date > days[0]]
    assert len(later) == 1 and later[0].reason == "stop_loss" and later[0].date == days[1]
    # If the optimizer drops A entirely, the single trade is a full sale.
    sim_b = PortfolioSim("P", CFG)
    simulate_week(sim_b, WeekPlan(W10, pd.Series({"A": 0.5, "B": 0.4}), "llm", "normal"), book)
    trades_b, _ = simulate_week(sim_b, WeekPlan("2026-W11", pd.Series({"B": 0.4}), "llm", "normal"), book2)
    a_b = [t for t in trades_b if t.ticker == "A"]
    assert len(a_b) == 1 and a_b[0].stop_trigger == "stop_loss" and "A" not in sim_b.positions
    assert a_b[0].date == days[0] and a_b[0].reason == "rebalance"


def test_missing_price_carries_last_close_and_forces_exit() -> None:
    rows = {"A": [(100, 100), (None, None), (None, None), (120, 120), (120, 120)],
            "B": [(10, 10)] * 5}
    book = PriceBook(_ledger(rows))
    sim = PortfolioSim("P", CFG)
    trades, daily = simulate_week(sim, WeekPlan(W10, pd.Series({"A": 0.5, "B": 0.1}), "llm", "normal"), book)
    assert "carried:A" in daily[1].flags and daily[1].value == pytest.approx(daily[0].value)
    forced = [t for t in trades if t.reason == "forced_exit"]
    assert len(forced) == 1 and forced[0].date == DAYS[3] and forced[0].fill_price == 120


def test_reconciliation_and_replay_determinism(data_dir: Path) -> None:
    book = PriceBook(_ledger(HAND_ROWS))
    plans = [WeekPlan(W10, HAND_WEIGHTS, "llm", "normal")]
    h1 = replay_all(StudyConfig(simulator=CFG), {"P": plans}, book)
    h2 = replay_all(StudyConfig(simulator=CFG), {"P": plans}, book)
    assert h1 == h2 and set(h1["P"]) == {"trades.csv", "daily.csv"}
    trades, daily = read_trades("P"), read_daily("P")
    buys = trades[trades["side"] == "buy"]["trade_value"].sum()
    sells = trades[trades["side"] == "sell"]["trade_value"].sum()
    costs = trades["cost_usd"].sum()
    assert daily["cumulative_cost"].iloc[-1] == pytest.approx(costs)
    # Cash reconciles with the trade log, and the final value equals cash (no positions).
    assert daily["cash"].iloc[-1] == pytest.approx(CFG.initial_value - buys + sells - costs)
    assert daily["value"].iloc[-1] == pytest.approx(daily["cash"].iloc[-1])
    # Daily returns compound to the total return.
    r = daily["value"].pct_change().dropna()
    assert float(np.prod(1 + r)) == pytest.approx(daily["value"].iloc[-1] / daily["value"].iloc[0])


def test_provider_rewrite_does_not_change_replay(data_dir: Path) -> None:
    import logging

    led = _ledger(HAND_ROWS)
    ledger.append_prices(led, logging.getLogger("t"))
    plans = {"P": [WeekPlan(W10, HAND_WEIGHTS, "llm", "normal")]}
    h1 = replay_all(StudyConfig(simulator=CFG), plans, ledger_book())
    # The provider now reports different prices for a past day; the ledger keeps its values.
    changed = led.copy()
    changed.loc[changed["date"] == DAYS[3], "close"] = 500.0
    assert ledger.append_prices(changed, logging.getLogger("t")) == 0
    h2 = replay_all(StudyConfig(simulator=CFG), plans, ledger_book())
    assert h1 == h2
    assert records_hash("P") == h2["P"]


def test_week_alignment_scores_pair_with_the_following_window() -> None:
    w = "2026-W10"
    cutoff = cal.cutoff_day(w)
    assert cutoff == dt.date(2026, 3, 6)
    t0, t1 = cal.monday_open_day(w), cal.window_end_date(w)
    assert t0 == dt.date(2026, 3, 9) and t1 == dt.date(2026, 3, 16) and t0 > cutoff
    earlier_monday = dt.date(2026, 3, 2)
    led = pd.DataFrame({
        "date": [earlier_monday, cutoff, t0, t1], "ticker": "A",
        "open": [50.0, 80.0, 100.0, 110.0], "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0})
    out = weekly_returns(led, w)
    assert out.loc[0, "trade_date"] == t0 and out.loc[0, "window_end_date"] == t1
    assert out.loc[0, "ret"] == pytest.approx(0.10)  # never 80 -> 100 or 50 -> 80


def test_thirty_weeks_thirteen_portfolios_under_a_minute(snapshot_data_dir: Path, study_config: StudyConfig,
                                                         session_weeks: list[str]) -> None:
    book = ledger_book()
    led = ledger.read_ledger()
    tickers = sorted(t for t in led["ticker"].unique() if str(t).startswith("SYN"))[:100]
    rng = np.random.default_rng(0)
    plans: dict[str, list[WeekPlan]] = {}
    for p in range(13):
        weekly = []
        for w in session_weeks:
            raw = rng.random(len(tickers))
            raw = np.minimum(raw / raw.sum(), 0.05)
            raw = raw / raw.sum()
            weekly.append(WeekPlan(w, pd.Series(raw, index=tickers), "llm" if p else "q0", "normal"))
        plans[f"P{p}"] = weekly
    start = time.perf_counter()
    sims = {k: replay(k, v, book, study_config.simulator) for k, v in plans.items()}
    elapsed = time.perf_counter() - start
    assert elapsed < 60.0, elapsed
    for sim in sims.values():
        assert len(sim.daily) > 100 and all(r.value > 0 for r in sim.daily)
