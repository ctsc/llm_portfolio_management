"""Trade simulator with stop-loss and take-profit guardrails, driven only by the price ledger.

Rules (all from PLAN.md Phase 9):
- Start at `initial_value` cash on the same start date for every portfolio.
- Rebalance at the Monday open: value positions at the open, target_value_i = w_i * V_open,
  fractional shares filled at the open, cost = cost_bps / 10000 * |trade value| paid from cash.
  Sells settle first; if cash is short for the buys after costs, every buy is scaled by the
  same factor so that buys plus their costs equal the available cash.
- Daily mark at each close (raw ledger close); stop-loss and take-profit are checked at every
  close and queue a full sale at the next open, cost applies, proceeds stay in cash.
- A sale queued for the same open as a rebalance becomes one net trade to the new target,
  with `reason = rebalance` and `stop_trigger` recording the cause; cost is paid once.
- Average cost is the share-weighted average on buys and unchanged by sales.
- Dividends credit cash on the ex-date; splits multiply shares and divide the average cost.
- A stock with no price on a day carries its last close, is flagged, and is sold at the first
  open where a price exists. After `no_price_liquidation_days` consecutive missing days it is
  sold at its last close instead (`no_price_liquidation`, DECISION(U2)).
- Each week's trade date records the portfolio value at the open (after that day's corporate
  actions, before any rebalance or pending sale) as `week_opens` for open-to-open weekly
  returns ((N4)); `replay` appends a closing entry at the following Monday open when
  the book has it.
- A stop pending on a rebalance day whose position is still held afterwards resets its
  `avg_cost` to the rebalance open, so the same move does not fire the stop twice (G1b).
"""

from __future__ import annotations

import copy
import datetime as dt
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import pandas as pd

from portstudy import calendar as cal
from portstudy.config import SimulatorConfig
from portstudy.types import DailyRecord, DecisionSource, Trade, WeekStatus

EPS_SHARES = 1e-12


class PriceBook:
    """Raw open and close by date from the ledger, plus corporate actions by date."""

    def __init__(self, ledger: pd.DataFrame, actions: pd.DataFrame | None = None) -> None:
        self.open: dict[dt.date, dict[str, float]] = {}
        self.close: dict[dt.date, dict[str, float]] = {}
        led = ledger
        for d, g in led.groupby("date", sort=True):
            day = _as_date(d)
            self.open[day] = dict(zip(g["ticker"].astype(str), g["open"].astype(float)))
            self.close[day] = dict(zip(g["ticker"].astype(str), g["close"].astype(float)))
        self.dates: list[dt.date] = sorted(self.open)
        self.actions: dict[dt.date, list[tuple[str, str, float]]] = {}
        if actions is not None and len(actions):
            for d, tk, action, value in zip(
                actions["date"], actions["ticker"], actions["action"], actions["value"]
            ):
                self.actions.setdefault(_as_date(d), []).append(
                    (str(tk), str(action), float(value))
                )

    @property
    def last_date(self) -> dt.date | None:
        return self.dates[-1] if self.dates else None

    def has_day(self, day: dt.date) -> bool:
        return day in self.open

    def open_price(self, day: dt.date, ticker: str) -> float | None:
        return self.open.get(day, {}).get(ticker)

    def close_price(self, day: dt.date, ticker: str) -> float | None:
        return self.close.get(day, {}).get(ticker)


def _as_date(value: object) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.Timestamp(str(value)).date()


@dataclass
class Position:
    shares: float
    avg_cost: float
    last_close: float
    no_price_days: int = 0  # consecutive ledger days without a positive close (U2)


@dataclass
class WeekPlan:
    """What the simulator trades for one week."""

    week_id: str
    weights: pd.Series | None  # None means hold (no_rebalance)
    decision_source: DecisionSource = "llm"
    week_status: WeekStatus = "normal"
    removed: frozenset[str] = frozenset()  # tickers removed from the study (U1)


@dataclass
class PortfolioSim:
    portfolio_id: str
    cfg: SimulatorConfig
    cash: float = field(default=0.0)
    positions: dict[str, Position] = field(default_factory=dict)
    cumulative_cost: float = 0.0
    pending_sales: dict[str, str] = field(default_factory=dict)  # ticker -> reason
    trades: list[Trade] = field(default_factory=list)
    daily: list[DailyRecord] = field(default_factory=list)
    # (week_id, trade date, value at the open) per week, plus one closing entry (N4).
    week_opens: list[tuple[str, dt.date, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cash == 0.0:
            self.cash = float(self.cfg.initial_value)

    # Valuation ------------------------------------------------------------------------------

    def _price_or_carry(self, prices: dict[str, float], ticker: str) -> tuple[float, bool]:
        p = prices.get(ticker)
        if p is None or p <= 0:
            return self.positions[ticker].last_close, True
        return p, False

    def value_at(self, prices: dict[str, float]) -> tuple[float, list[str]]:
        total = self.cash
        carried: list[str] = []
        for tk, pos in self.positions.items():
            p, flag = self._price_or_carry(prices, tk)
            total += pos.shares * p
            if flag:
                carried.append(tk)
        return total, carried

    # Corporate actions ----------------------------------------------------------------------

    def apply_actions(self, actions: Iterable[tuple[str, str, float]]) -> None:
        for tk, action, value in actions:
            pos = self.positions.get(tk)
            if pos is None or pos.shares <= EPS_SHARES:
                continue
            if action == "dividend":
                self.cash += pos.shares * value
            elif action == "split" and value > 0:
                pos.shares *= value
                pos.avg_cost /= value
                pos.last_close /= value

    # Trading --------------------------------------------------------------------------------

    def _trade(
        self,
        week_id: str,
        day: dt.date,
        ticker: str,
        shares: float,
        price: float,
        reason: str,
        decision_source: DecisionSource,
        v_open: float,
        stop_trigger: str | None = None,
    ) -> None:
        """Execute a signed share trade at `price` (positive buys, negative sells)."""
        if abs(shares) <= EPS_SHARES:
            return
        pos = self.positions.get(ticker)
        before_shares = pos.shares if pos else 0.0
        before_cost = pos.avg_cost if pos else 0.0
        value = shares * price
        cost = self.cfg.cost_bps / 10000.0 * abs(value)
        self.cash -= value + cost
        self.cumulative_cost += cost
        new_shares = before_shares + shares
        if shares > 0:
            new_cost = (before_shares * before_cost + shares * price) / new_shares
        else:
            new_cost = before_cost
        if new_shares <= EPS_SHARES:
            self.positions.pop(ticker, None)
            new_shares, new_cost = 0.0, 0.0
        else:
            self.positions[ticker] = Position(new_shares, new_cost, price)
        self.trades.append(
            Trade(
                week_id=week_id,
                date=day,
                portfolio_id=self.portfolio_id,
                ticker=ticker,
                side="buy" if shares > 0 else "sell",
                shares=abs(shares),
                fill_price=price,
                trade_value=abs(value),
                cost_bps=self.cfg.cost_bps,
                cost_usd=cost,
                reason=reason,  # type: ignore[arg-type]
                decision_source=decision_source,
                avg_entry_price_before=before_cost,
                avg_entry_price_after=new_cost,
                weight_before=before_shares * price / v_open if v_open else 0.0,
                weight_after=new_shares * price / v_open if v_open else 0.0,
                stop_trigger=stop_trigger,
            )
        )

    def _liquidate_at_last_close(
        self,
        week_id: str,
        day: dt.date,
        ticker: str,
        prices: dict[str, float],
        decision_source: DecisionSource,
    ) -> bool:
        """Sell the whole position at `last_close` with the usual cost (U2/U1)."""
        pos = self.positions.get(ticker)
        if pos is None or pos.shares <= EPS_SHARES or pos.last_close <= 0:
            return False
        v_open, _ = self.value_at(prices)
        self._trade(
            week_id,
            day,
            ticker,
            -pos.shares,
            pos.last_close,
            "no_price_liquidation",
            decision_source,
            v_open,
        )
        self.pending_sales.pop(ticker, None)
        return True

    def rebalance(
        self,
        week_id: str,
        day: dt.date,
        weights: pd.Series,
        opens: dict[str, float],
        decision_source: DecisionSource,
        removed: frozenset[str] = frozenset(),
    ) -> None:
        # U1: a held ticker removed from the study with no positive open today is liquidated at
        # its last close; with a positive open it is simply sold at the open (weight 0 below).
        for tk in sorted(set(self.positions) & set(removed)):
            if not (opens.get(tk) or 0) > 0:
                self._liquidate_at_last_close(week_id, day, tk, opens, decision_source)
        stopped = {
            tk
            for tk, reason in self.pending_sales.items()
            if reason in ("stop_loss", "take_profit")
        }
        v_open, _ = self.value_at(opens)
        targets = {str(t): float(w) for t, w in weights.items() if float(w) > 0}
        tickers = sorted(set(targets) | set(self.positions))
        plan: list[tuple[str, float, float]] = []  # ticker, trade value, price
        for tk in tickers:
            price = opens.get(tk)
            if price is None or price <= 0:
                continue  # no price today: cannot trade, handled by the carry rule
            current = self.positions[tk].shares * price if tk in self.positions else 0.0
            target = targets.get(tk, 0.0) * v_open
            diff = target - current
            if abs(diff) > 1e-9:
                plan.append((tk, diff, price))
        # Sells first so their proceeds fund the buys.
        c = self.cfg.cost_bps / 10000.0
        sells = [p for p in plan if p[1] < 0]
        buys = [p for p in plan if p[1] > 0]
        for tk, diff, price in sells:
            trig = self.pending_sales.pop(tk, None)
            self._trade(
                week_id,
                day,
                tk,
                diff / price,
                price,
                "rebalance",
                decision_source,
                v_open,
                stop_trigger=trig,
            )
        total_buy = sum(d for _, d, _ in buys)
        scale = 1.0
        if total_buy > 0 and total_buy * (1 + c) > self.cash:
            scale = max(0.0, self.cash / (total_buy * (1 + c)))
        for tk, diff, price in buys:
            trig = self.pending_sales.pop(tk, None)
            self._trade(
                week_id,
                day,
                tk,
                diff * scale / price,
                price,
                "rebalance",
                decision_source,
                v_open,
                stop_trigger=trig,
            )
        # A pending sale for a stock that already sits at target still counts as handled.
        for tk in list(self.pending_sales):
            if tk in opens and tk in targets:
                self.pending_sales.pop(tk)
        # G1b: a stop-loss or take-profit pending on the rebalance day has been handled here;
        # if the position is still held and priced, reset avg_cost to today's open so the same
        # move does not fire the stop again at the close. forced_exit pendings are not reset.
        for tk in stopped:
            pos = self.positions.get(tk)
            price = opens.get(tk)
            if pos is not None and pos.shares > EPS_SHARES and price is not None and price > 0:
                pos.avg_cost = price
                self.pending_sales.pop(tk, None)
        # Float rounding can leave cash a hair below zero after a fully scaled buy.
        if abs(self.cash) < 1e-6:
            self.cash = 0.0

    def execute_pending(
        self, week_id: str, day: dt.date, opens: dict[str, float], decision_source: DecisionSource
    ) -> None:
        v_open, _ = self.value_at(opens)
        for tk in sorted(self.pending_sales):
            pos = self.positions.get(tk)
            price = opens.get(tk)
            if pos is None:
                self.pending_sales.pop(tk)
                continue
            if price is None or price <= 0:
                continue  # still no price: keep waiting for the first available open
            reason = self.pending_sales.pop(tk)
            self._trade(week_id, day, tk, -pos.shares, price, reason, decision_source, v_open)

    def check_stops(self, closes: dict[str, float]) -> None:
        for tk, pos in self.positions.items():
            price = closes.get(tk)
            if price is None or price <= 0 or pos.avg_cost <= 0:
                continue
            # A relative tolerance makes "exactly at the threshold" robust to float rounding.
            tol = 1e-9 * pos.avg_cost
            if price <= (1.0 + self.cfg.stop_loss) * pos.avg_cost + tol:
                self.pending_sales.setdefault(tk, "stop_loss")
            elif price >= (1.0 + self.cfg.take_profit) * pos.avg_cost - tol:
                self.pending_sales.setdefault(tk, "take_profit")

    def mark(
        self,
        week_id: str,
        day: dt.date,
        closes: dict[str, float],
        week_status: WeekStatus,
        decision_source: DecisionSource = "hold",
    ) -> None:
        flags: list[str] = []
        for tk, pos in list(self.positions.items()):  # a liquidation removes the entry
            p = closes.get(tk)
            if p is None or p <= 0:
                pos.no_price_days += 1
                if pos.no_price_days >= self.cfg.no_price_liquidation_days:
                    # U2: too long without a price; sell at the last close today so the daily
                    # value already reflects the sale, then forget the waiting pending sale.
                    if self._liquidate_at_last_close(week_id, day, tk, closes, decision_source):
                        flags.append(f"liquidated:{tk}")
                        continue
                flags.append(f"carried:{tk}")
                self.pending_sales.setdefault(tk, "forced_exit")
            else:
                pos.no_price_days = 0
                pos.last_close = p
        value, _ = self.value_at(closes)
        self.daily.append(
            DailyRecord(
                date=day,
                portfolio_id=self.portfolio_id,
                value=value,
                cash=self.cash,
                cumulative_cost=self.cumulative_cost,
                n_positions=sum(1 for p in self.positions.values() if p.shares > EPS_SHARES),
                week_status=week_status,
                week_id=week_id,
                flags=";".join(flags),
            )
        )


def week_days(week_id: str, book: PriceBook) -> list[dt.date]:
    """Ledger days in [trade_date, window_end_date) for the week."""
    start, end = cal.monday_open_day(week_id), cal.window_end_date(week_id)
    return [d for d in book.dates if start <= d < end]


def simulate_week(
    sim: PortfolioSim, plan: WeekPlan, book: PriceBook
) -> tuple[list[Trade], list[DailyRecord]]:
    """Advance `sim` through one week. Returns the trades and daily records added."""
    n_t, n_d = len(sim.trades), len(sim.daily)
    trade_date = cal.monday_open_day(plan.week_id)
    for day in week_days(plan.week_id, book):
        sim.apply_actions(book.actions.get(day, []))
        opens = book.open.get(day, {})
        open_value: float | None = None
        if day == trade_date:
            # N4: value at the week's open, after that day's corporate actions and before any
            # rebalance or pending sale; recorded for every week status.
            open_value, _ = sim.value_at(opens)
            sim.week_opens.append((plan.week_id, day, open_value))
        if day == trade_date and plan.week_status != "no_rebalance" and plan.weights is not None:
            sim.rebalance(
                plan.week_id,
                day,
                plan.weights,
                opens,
                plan.decision_source,
                removed=plan.removed,
            )
            # Pending sales for stocks the optimizer dropped were sold inside the rebalance;
            # anything still pending has no price today and waits.
        else:
            sim.execute_pending(plan.week_id, day, opens, plan.decision_source)
        closes = book.close.get(day, {})
        sim.mark(plan.week_id, day, closes, plan.week_status, plan.decision_source)
        if day == trade_date and sim.daily:
            sim.daily[-1].value_open = open_value
        sim.check_stops(closes)
    return sim.trades[n_t:], sim.daily[n_d:]


def replay(
    portfolio_id: str, plans: Sequence[WeekPlan], book: PriceBook, cfg: SimulatorConfig
) -> PortfolioSim:
    """Rebuild a portfolio from scratch over `plans`, reading only the ledger."""
    sim = PortfolioSim(portfolio_id, cfg)
    for plan in plans:
        simulate_week(sim, plan, book)
    if plans:
        # N4: the last week's open-to-open return needs the following Monday open. When the
        # book has it, append a closing entry for the next week id, valued on a deep copy so
        # that day's corporate actions apply without trading.
        last_week = plans[-1].week_id
        end_day = cal.window_end_date(last_week)
        if book.has_day(end_day):
            clone = copy.deepcopy(sim)
            clone.apply_actions(book.actions.get(end_day, []))
            value, _ = clone.value_at(book.open[end_day])
            sim.week_opens.append((cal.next_week_id(last_week), end_day, value))
    return sim
