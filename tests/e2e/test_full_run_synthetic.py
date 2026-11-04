"""Phase 14 end-to-end rehearsal on synthetic data."""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import pandas as pd
import pytest

from portstudy import calendar as cal
from portstudy.config import ModelRegistry, StudyConfig
from portstudy.data.providers.base import ProviderBundle
from portstudy.data.providers.synthetic import SyntheticWorld, synthetic_bundle
from portstudy.data.snapshot import read_snapshot
from portstudy.eval.data import load_eval_data
from portstudy.eval.report import run_eval
from portstudy.hashing import sha256_file
from portstudy.io import read_csv, read_json
from portstudy.llm.cost import CostTable
from portstudy.llm.providers.stub import StubFailures, StubLlmProvider
from portstudy.llm.runner import lock_deadline, read_scores
from portstudy.pipeline import fallback as fb
from portstudy.pipeline.jobs import run_named
from portstudy.pipeline.status import read_status
from portstudy.sim.records import read_daily, read_trades
from tests.conftest import generous_budget
from tests.e2e.conftest import INJECTIONS, REPO_ROOT, WeekendRunner, rehearsal_registry

pytestmark = pytest.mark.e2e


def _prompt_hashes() -> dict[str, str]:
    return {
        str(p.relative_to(REPO_ROOT)): sha256_file(p)
        for p in sorted((REPO_ROOT / "prompts").rglob("*"))
        if p.is_file()
    }


def _csv_hashes(root: Path) -> dict[str, str]:
    """Hashes of every CSV under outputs and portfolios; the latency column is dropped from scores."""
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*.csv")):
        rel = p.relative_to(root).as_posix()
        if rel.startswith(("backups/", "exports/", "snapshots/", "ledger/", "synthetic/")):
            continue
        if p.name == "llm_scores.csv":
            df = read_csv(p, keep_default_na=False).drop(columns=["latency_s"])
            out[rel] = pd.util.hash_pandas_object(df, index=False).sum().item().__str__()
        else:
            out[rel] = sha256_file(p)
    return out


def test_full_run_synthetic(
    e2e_dir: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    e2e_weeks: list[str],
) -> None:
    cfg = generous_budget(study_config)
    model_registry = rehearsal_registry(model_registry)
    injected = {
        e2e_weeks[i] for i in INJECTIONS
    }  # v2 phase 13: failure weeks are checked elsewhere
    before_prompts = _prompt_hashes()
    context = WeekendRunner(cfg, model_registry, world).context(e2e_weeks[0])
    keys = context.portfolio_keys()
    n_sources = len(context.sources())
    assert len(keys) == len(cfg.strategies.enabled) * (n_sources + 1) + len(cfg.strategies.controls)
    replayed_keys = keys
    for w in e2e_weeks:
        snap = read_snapshot(w)
        # No look-ahead: nothing in a snapshot is dated after its cutoff.
        assert snap.prices["date"].max() <= cal.cutoff_day(w)
        assert (
            snap.news["published_at"]
            <= pd.Timestamp(cal.cutoff_day(w), tz="America/New_York") + pd.Timedelta(hours=16)
        ).all()
        lock = read_json(e2e_dir / "outputs" / w / "weights" / "locked.json")
        if w not in injected:  # the lock-cutoff week locks after the deadline on purpose
            assert dt.datetime.fromisoformat(lock["locked_at"]) <= lock_deadline(w, cfg)
        for key in replayed_keys:
            wdf = read_csv(e2e_dir / "outputs" / w / "weights" / f"{key}.csv")
            assert wdf["weight"].sum() == pytest.approx(1.0, abs=1e-8)
            assert (wdf["weight"] >= -1e-12).all() and (
                wdf["weight"] <= cfg.optimizer.max_weight + 1e-9
            ).all()
            if w not in injected:
                assert fb.week_status_for(w, key) == "normal"
        scores = read_scores(w)
        if w not in injected:
            assert len(scores) == n_sources * len(snap.included_tickers)
    for key in replayed_keys:
        trades, daily = read_trades(key), read_daily(key)
        assert (
            trades["cost_usd"] - trades["trade_value"] * cfg.simulator.cost_bps / 1e4
        ).abs().max() < 1e-6
        buys = trades.loc[trades["side"] == "buy", "trade_value"].sum()
        sells = trades.loc[trades["side"] == "sell", "trade_value"].sum()
        divs = 0.0
        last = daily.iloc[-1]
        # Cash identity up to dividends credited (synthetic world pays one dividend).
        implied = cfg.simulator.initial_value - buys + sells - trades["cost_usd"].sum() + divs
        assert last["cash"] >= implied - 1e-6
        assert (daily.loc[~daily["week_id"].isin(injected), "week_status"] == "normal").all()
        assert daily["value"].min() > 0
    ic = read_csv(e2e_dir / "outputs" / "ic_history.csv")
    closed = [w for w in e2e_weeks if cal.window_end_date(w) <= cal.cutoff_day(e2e_weeks[-1])]
    assert set(ic.loc[ic["signal"] == "baseline", "week_id"]) == set(closed)
    assert _prompt_hashes() == before_prompts
    ctx = WeekendRunner(cfg, model_registry, world).context(e2e_weeks[-1])
    folder = run_eval(ctx, skip_shuffle=True, folder=e2e_dir / "outputs" / "report")
    assert (folder / "report.md").exists()
    clean_weeks = [w for w in e2e_weeks if w not in injected]
    data = load_eval_data(cfg, ctx.portfolio_keys(), weeks=clean_weeks)
    assert data.fallback_counts()["fallback"].sum() == 0


def test_failure_week(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    e2e_weeks: list[str],
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(root))
    cfg = generous_budget(study_config)
    # The test config lists the open-weight stubs; week 4 needs an API model for its stuck batch.
    api_ids = [*cfg.llm.model_ids, "stub-api-a", "stub-api-b"]
    cfg = cfg.model_copy(update={"llm": cfg.llm.model_copy(update={"model_ids": api_ids})})
    runner = WeekendRunner(cfg, model_registry, world)
    weeks = e2e_weeks[:6]
    assert runner.run(weeks[0], monkeypatch) == 0
    # Week 2: a local crash, then a clean resume.
    runner.llm_providers = {
        "stub-ow-a": StubLlmProvider("stub-ow-a", failures=StubFailures(crash_after_calls=40))
    }
    assert runner.run(weeks[1], monkeypatch) == 1
    runner.llm_providers = {}
    assert runner.run(weeks[1], monkeypatch) == 0
    # Week 3: invalid-output storm for one model.
    runner.llm_providers = {
        "stub-ow-b": StubLlmProvider("stub-ow-b", failures=StubFailures(invalid_rate=0.6))
    }
    assert runner.run(weeks[2], monkeypatch) == 0
    runner.llm_providers = {}
    # Week 4: a batch that never returns; the clock sits at the lock margin so J6 can decide.
    runner.llm_providers = {
        "stub-api-a": StubLlmProvider("stub-api-a", failures=StubFailures(batch_never_returns=True))
    }
    margin = lock_deadline(weeks[3], cfg) - dt.timedelta(
        minutes=cfg.pipeline.lock_margin_minutes - 1
    )
    runner.clock = lambda: margin
    assert runner.run(weeks[3], monkeypatch) == 0
    runner.clock = None
    runner.llm_providers = {}
    # Week 5: data failure, every portfolio holds.
    bundle = synthetic_bundle(world)

    class Broken(type(bundle.prices)):  # type: ignore[misc]
        def get_prices(self, tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
            raise ConnectionError("provider down")

    runner.data_providers = ProviderBundle(
        prices=Broken(world),
        news=bundle.news,
        reference=bundle.reference,
        rates=bundle.rates,
        members=bundle.members,
    )
    assert runner.run(weeks[4], monkeypatch) == 1
    ctx = runner.context(weeks[4])
    for job in ("J6", "J7"):
        assert run_named(ctx, job, weeks[4]).state == "done"
    runner.data_providers = None
    # Week 6: normal again; its catch-up step replays everything.
    assert runner.run(weeks[5], monkeypatch) == 0

    for conf_id in cfg.llm.prompt_ids:
        assert fb.week_status_for(weeks[1], f"mvo__stub-ow-a__{conf_id}") == "normal"
        assert fb.week_status_for(weeks[2], f"mvo__stub-ow-b__{conf_id}") == "fallback"
        assert fb.week_status_for(weeks[3], f"mvo__stub-api-a__{conf_id}") == "fallback"
        assert fb.week_status_for(weeks[3], f"mvo__stub-api-b__{conf_id}") == "normal"
    assert fb.week_status_for(weeks[4], "mvo__Q0") == "no_rebalance"
    daily = read_daily("mvo__stub-ow-b__skills_1")
    assert set(daily.loc[daily["week_id"] == weeks[2], "week_status"]) == {"fallback"}
    assert set(daily.loc[daily["week_id"] == weeks[4], "week_status"]) == {"no_rebalance"}
    trades = read_trades("mvo__stub-ow-b__skills_1")
    assert (trades.loc[trades["week_id"] == weeks[2], "decision_source"] == "fallback_q0").all()
    assert not ((trades["week_id"] == weeks[4]) & (trades["reason"] == "rebalance")).any()
    data = load_eval_data(cfg, runner.context(weeks[5]).portfolio_keys(), weeks=weeks)
    counts = data.fallback_counts().set_index("portfolio")
    assert (
        counts.loc["mvo__stub-ow-b__skills_1", "fallback"] == 1
        and counts.loc["mvo__stub-ow-b__skills_1", "no_rebalance"] == 1
    )
    assert counts.loc["mvo__stub-api-b__bare", "fallback"] == 0
    assert fb.clean_weeks("mvo__stub-ow-b__skills_1", weeks) == [
        weeks[0],
        weeks[1],
        weeks[3],
        weeks[5],
    ]


def test_resume_equals_uninterrupted_run(
    e2e_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    e2e_weeks: list[str],
) -> None:
    reference = _csv_hashes(e2e_dir)
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(root))
    cfg = generous_budget(study_config)
    runner = WeekendRunner(cfg, rehearsal_registry(model_registry), world)
    week = e2e_weeks[0]
    runner.llm_providers = {
        "stub-ow-a": StubLlmProvider("stub-ow-a", failures=StubFailures(crash_after_calls=33))
    }
    assert runner.run(week, monkeypatch) == 1
    assert read_status(week, "J4").state == "failed"
    runner.llm_providers = {}
    assert runner.run(week, monkeypatch) == 0
    mine = _csv_hashes(root)
    for rel, digest in mine.items():
        if rel.startswith(f"outputs/{week}/") or rel.startswith("portfolios/"):
            if rel.startswith("portfolios/"):
                continue  # the reference portfolios span 25 weeks
            assert reference[rel] == digest, rel


def test_determinism_two_runs_identical(
    e2e_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    e2e_weeks: list[str],
) -> None:
    reference = _csv_hashes(e2e_dir)
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(root))
    runner = WeekendRunner(generous_budget(study_config), rehearsal_registry(model_registry), world)
    weeks = e2e_weeks[:3]  # the first injected week is index 3
    for w in weeks:
        assert runner.run(w, monkeypatch) == 0
    mine = _csv_hashes(root)
    compared = 0
    for rel, digest in mine.items():
        if any(rel.startswith(f"outputs/{w}/") for w in weeks):
            assert reference[rel] == digest, rel
            compared += 1
    assert compared > 3 * 15
    # The IC history and weekly returns for the shared weeks agree too.
    ref_ic = read_csv(e2e_dir / "outputs" / "ic_history.csv")
    my_ic = read_csv(root / "outputs" / "ic_history.csv")
    merged = my_ic.merge(
        ref_ic, on=["week_id", "signal", "model", "config_id"], suffixes=("_a", "_b")
    )
    assert (
        len(merged) == len(my_ic)
        and (merged["rank_ic_a"] - merged["rank_ic_b"]).abs().max() < 1e-12
    )


def test_no_future_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    world: SyntheticWorld,
    e2e_weeks: list[str],
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(root))
    cfg = generous_budget(study_config)
    week = e2e_weeks[1]
    cutoff = cal.cutoff_day(week)
    bundle = synthetic_bundle(world)
    asked: list[tuple[str, object]] = []

    class Guarded(type(bundle.prices)):  # type: ignore[misc]
        def get_prices(self, tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
            asked.append(("prices", end))
            if end > cutoff:
                raise AssertionError(f"asked for prices through {end}, after the cutoff {cutoff}")
            return super().get_prices(tickers, start, end)

    class GuardedNews(type(bundle.news)):  # type: ignore[misc]
        def get_news(
            self, tickers: list[str], start_utc: dt.datetime, end_utc: dt.datetime
        ) -> pd.DataFrame:
            # The snapshot pulls a few hours past the cutoff on purpose (leakage check) and drops them.
            limit = dt.datetime.combine(
                cutoff, dt.time(16), tzinfo=dt.timezone(dt.timedelta(hours=-5))
            )
            assert end_utc <= limit + dt.timedelta(hours=12)
            return super().get_news(tickers, start_utc, end_utc)

    runner = WeekendRunner(cfg, model_registry, world)
    runner.data_providers = ProviderBundle(
        prices=Guarded(world),
        news=GuardedNews(world),
        reference=bundle.reference,
        rates=bundle.rates,
        members=bundle.members,
    )
    ctx = runner.context(week)
    for job in ("J1", "J2"):
        assert run_named(ctx, job, week).state == "done"
    assert asked and all(end <= cutoff for _, end in asked)
    snap = read_snapshot(week)
    limit = pd.Timestamp(cutoff, tz="America/New_York") + pd.Timedelta(hours=16)
    assert (snap.news["published_at"] <= limit).all()
    assert snap.prices["date"].max() <= cutoff


def test_cost_log(
    e2e_dir: Path, study_config: StudyConfig, model_registry: ModelRegistry, e2e_weeks: list[str]
) -> None:
    table = CostTable(model_registry)
    total_log, total_scores = 0.0, 0.0
    for w in e2e_weeks:
        if w == e2e_weeks[12]:  # gateway week: the first run logs the failed sources at zero
            continue
        cost = read_csv(e2e_dir / "outputs" / w / "cost_log.csv")
        scores = read_scores(w)
        total_log += float(cost["cost_usd"].sum())
        total_scores += float(scores["cost_usd"].sum())
        for _, row in scores.iterrows():
            spec = model_registry.get(str(row["model"]))
            expected = table.cost_usd(
                spec.id,
                int(row["tokens_in"]),
                int(row["tokens_cached"]),
                int(row["tokens_out"]),
                spec.kind == "api",
            )
            assert (
                float(row["cost_usd"]) == pytest.approx(expected, abs=1e-9)
                or float(row["cost_usd"]) == 0.0
            )
    assert total_log == pytest.approx(total_scores)
    # Only API models carry a price; the test config runs the two open-weight stubs.
    api_models = [m for m in study_config.llm.model_ids if model_registry.get(m).kind == "api"]
    assert (total_log > 0) == bool(api_models)
    assert (
        read_csv(e2e_dir / "outputs" / e2e_weeks[0] / "cost_log.csv")["projected_study_usd"] >= 0
    ).all()


def test_backups_exist_for_every_week(e2e_dir: Path, e2e_weeks: list[str]) -> None:
    folders = [p for p in (e2e_dir / "backups").iterdir() if p.is_dir()]
    weeks_backed = {p.name.split("_")[0] for p in folders}
    assert set(e2e_weeks) <= weeks_backed
    shutil.rmtree(folders[0])  # removing one old backup never affects the others
    assert len([p for p in (e2e_dir / "backups").iterdir() if p.is_dir()]) == len(folders) - 1
