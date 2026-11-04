"""Load everything the evaluation needs from the data folder into one object."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from portstudy import calendar as cal
from portstudy.config import StudyConfig
from portstudy.data import ledger
from portstudy.data.snapshot import read_snapshot, snapshot_exists
from portstudy.eval.rank_ic import ic_tables
from portstudy.llm.runner import read_scores
from portstudy.optimize import read_weights_meta, weights_dir
from portstudy.pipeline import fallback as fb
from portstudy.pipeline.jobs import weeks_with_weights
from portstudy.sim.records import read_daily, read_trades
from portstudy.types import PortfolioId, RatingSource
from portstudy.views.ic import ICStore, weekly_returns


def likely_column(source: str) -> str:
    return f"likely:{source}"


def width_column(source: str) -> str:
    return f"width:{source}"


def weekly_portfolio_returns(daily: pd.DataFrame, initial_value: float) -> pd.Series:
    """Return per week from the last close of the previous week to the last close of this week."""
    if len(daily) == 0:
        return pd.Series(dtype=float)
    last = daily.groupby("week_id")["value"].last().sort_index()
    prev = last.shift(1).fillna(initial_value)
    return (last / prev - 1.0).rename("ret")


def max_drawdown(values: pd.Series) -> float:
    if len(values) == 0:
        return float("nan")
    peak = values.cummax()
    return float((values / peak - 1.0).min())


def sharpe(weekly: pd.Series, periods: int = 52) -> float:
    r = weekly.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))


@dataclass
class EvalData:
    """Everything the evaluation reads. `portfolios` are the steered (strategy, source) keys;
    `sources` are rating-source keys; `portfolio_keys` also hold every Q0 and control."""

    cfg: StudyConfig
    portfolios: list[str]
    sources: list[str]
    portfolio_keys: list[str]
    weeks: list[str]
    pilot_week: str | None
    ic: dict[str, pd.DataFrame]
    weekly_returns: pd.DataFrame  # weeks by every portfolio (steered, Q0, controls), as traded
    daily_values: dict[str, pd.DataFrame]
    trades: dict[str, pd.DataFrame]
    statuses: pd.DataFrame  # weeks by portfolios: normal | fallback | no_rebalance
    spy_weekly: pd.Series
    spy_vol20: pd.Series
    stock_week: pd.DataFrame = field(default_factory=pd.DataFrame)  # per stock-week rows
    strategy_of: dict[str, str] = field(default_factory=dict)  # steered or Q0 key -> strategy
    source_of: dict[str, str] = field(default_factory=dict)  # steered key -> source key
    controls: list[str] = field(default_factory=list)
    solver_failed: pd.DataFrame = field(default_factory=pd.DataFrame)  # weeks by portfolios
    view_shift: pd.DataFrame = field(default_factory=pd.DataFrame)  # weeks by portfolios

    def __post_init__(self) -> None:
        for key in self.portfolio_keys:
            pid = PortfolioId.parse(key)
            if pid.is_control:
                if key not in self.controls:
                    self.controls.append(key)
                continue
            self.strategy_of.setdefault(key, str(pid.strategy))
            if pid.source is not None:
                self.source_of.setdefault(key, pid.source.key)

    def strategies(self) -> list[str]:
        """Strategies in config order that have a portfolio in the data."""
        present = set(self.strategy_of.values())
        ordered = [s for s in self.cfg.strategies.enabled if s in present]
        return ordered + sorted(present - set(ordered))

    def q0_key(self, strategy: str) -> str:
        return f"{strategy}__Q0"

    def steered_for(self, strategy: str) -> list[str]:
        return [key for key in self.portfolios if self.strategy_of.get(key) == strategy]

    def clean_mask(self, portfolio: str) -> pd.Series:
        """Weeks traded normally. A steered portfolio inherits its source's fallback weeks."""
        return (
            self.statuses[portfolio].eq("normal")
            if portfolio in self.statuses
            else pd.Series(dtype=bool)
        )

    def source_clean_mask(self, source: str) -> pd.Series:
        for key, src in self.source_of.items():
            if src == source and key in self.statuses:
                return self.clean_mask(key)
        return pd.Series(True, index=self.weeks)

    def fallback_counts(self) -> pd.DataFrame:
        rows = []
        for p in self.statuses.columns:
            s = self.statuses[p]
            rows.append(
                {
                    "portfolio": p,
                    "weeks": int(len(s)),
                    "normal": int((s == "normal").sum()),
                    "fallback": int((s == "fallback").sum()),
                    "no_rebalance": int((s == "no_rebalance").sum()),
                }
            )
        return pd.DataFrame(rows)


def _spy_series(weeks: list[str]) -> tuple[pd.Series, pd.Series]:
    led = ledger.read_ledger()
    spy = led[led["ticker"] == "SPY"].sort_values("date")
    weekly: dict[str, float] = {}
    vol: dict[str, float] = {}
    if len(spy):
        opens = spy.set_index("date")["open"].astype(float)
        closes = spy.set_index("date")["close"].astype(float)
        rets = closes.pct_change()
        for w in weeks:
            t0, t1 = cal.monday_open_day(w), cal.window_end_date(w)
            if t0 in opens.index and t1 in opens.index:
                weekly[w] = float(opens[t1] / opens[t0] - 1.0)
            cutoff = cal.cutoff_day(w)
            window = rets[rets.index <= cutoff].tail(20)
            if len(window) >= 10:
                vol[w] = float(window.std(ddof=1) * np.sqrt(252))
    return pd.Series(weekly, name="spy_ret"), pd.Series(vol, name="spy_vol20")


def _stock_week_table(weeks: list[str], sources: list[str]) -> pd.DataFrame:
    """Per (week, stock): realized return, earnings flag, and each source's valid likely rating
    and range width (`likely:<source>`, `width:<source>`)."""
    led = ledger.read_ledger()
    actions = ledger.read_actions()
    last = max(led["date"]) if len(led) else None
    frames = []
    for w in weeks:
        if last is None or cal.window_end_date(w) > last or not snapshot_exists(w):
            continue
        rets = weekly_returns(led, w, actions).set_index("ticker")["ret"]
        snap = read_snapshot(w)
        ref = snap.reference.set_index("ticker")
        base = pd.DataFrame({"ret": rets})
        base["week_id"] = w
        flags = ref["earnings_in_window"].reindex(base.index)
        base["earnings_in_window"] = [bool(v) if pd.notna(v) else False for v in flags]
        scores = read_scores(w)
        columns: dict[str, pd.Series] = {}
        for key in sources:
            source = RatingSource.parse(key)
            sub = scores[
                (scores["model"] == source.model)
                & (scores["config_id"] == source.config_id)
                & scores["valid"]
            ].set_index("ticker")
            columns[likely_column(key)] = sub["likely"].reindex(base.index).astype(float)
            if "width" in sub.columns:
                columns[width_column(key)] = sub["width"].reindex(base.index).astype(float)
            else:
                columns[width_column(key)] = pd.Series(np.nan, index=base.index)
        if columns:
            base = pd.concat([base, pd.DataFrame(columns)], axis=1)
        frames.append(base.reset_index())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _meta_tables(weeks: list[str], keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weeks by portfolios: solver-failed flag and view shift, from the weight meta files."""
    failed: dict[str, dict[str, bool]] = {}
    shift: dict[str, dict[str, float]] = {}
    for key in keys:
        failed[key], shift[key] = {}, {}
        for w in weeks:
            if not (weights_dir(w) / f"{key}.meta.json").exists():
                continue
            meta = read_weights_meta(w, key)
            failed[key][w] = bool(meta.get("fallback_flag"))
            value = meta.get("view_shift")
            shift[key][w] = float(value) if value is not None else float("nan")
    failed_df = pd.DataFrame(failed).reindex(weeks).fillna(False).astype(bool)
    shift_df = pd.DataFrame(shift).reindex(weeks).astype(float)
    return failed_df, shift_df


def load_eval_data(
    cfg: StudyConfig, portfolios: list[str], weeks: list[str] | None = None
) -> EvalData:
    all_weeks = weeks or weeks_with_weights()
    pilot = cfg.timing.pilot_week
    # The pilot week never counts toward results.
    weeks_used = [w for w in all_weeks if w != pilot]
    keys = list(dict.fromkeys(portfolios))
    source_portfolios = [key for key in keys if PortfolioId.parse(key).source is not None]
    portfolio_by_source: dict[str, str] = {}
    for key in source_portfolios:
        source = PortfolioId.parse(key).source
        if source is not None:
            portfolio_by_source.setdefault(source.key, key)
    store = ICStore()
    ic = ic_tables(
        store,
        list(portfolio_by_source),
        clean_weeks=lambda source: fb.clean_weeks(portfolio_by_source[source], weeks_used),
    )
    for signal in ic:
        ic[signal] = ic[signal][ic[signal].index.isin(weeks_used)]
    daily = {k: read_daily(k) for k in keys}
    trades = {k: read_trades(k) for k in keys}
    weekly = pd.DataFrame(
        {k: weekly_portfolio_returns(daily[k], cfg.simulator.initial_value) for k in keys}
    )
    weekly = weekly[weekly.index.isin(weeks_used)]
    statuses = pd.DataFrame({k: {w: fb.week_status_for(w, k) for w in weeks_used} for k in keys})
    spy_weekly, spy_vol = _spy_series(weeks_used)
    solver_failed, view_shift = _meta_tables(weeks_used, keys)
    return EvalData(
        cfg=cfg,
        portfolios=source_portfolios,
        sources=list(portfolio_by_source),
        portfolio_keys=keys,
        weeks=weeks_used,
        pilot_week=pilot,
        ic=ic,
        weekly_returns=weekly,
        daily_values=daily,
        trades=trades,
        statuses=statuses,
        spy_weekly=spy_weekly,
        spy_vol20=spy_vol,
        stock_week=_stock_week_table(weeks_used, list(portfolio_by_source)),
        solver_failed=solver_failed,
        view_shift=view_shift,
    )
