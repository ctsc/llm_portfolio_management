"""PLAN4 Phase 3C: Alpaca feed/cutoff timestamp, smoke check, C8 snapshot identity and
market-cap carry, frozen config, live-readiness preflight, run_weekend wiring."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from portstudy import calendar as cal
from portstudy.config import (
    ModelRegistry,
    StudyConfig,
    load_model_registry,
    load_study_config,
)
from portstudy.data.providers.alpaca import AlpacaPriceProvider
from portstudy.data.providers.base import ProviderBundle
from portstudy.data.snapshot import (
    SnapshotHalt,
    carry_market_caps,
    make_snapshot,
    read_snapshot,
)
from portstudy.data.universe import build_universe
from portstudy.llm.prompts import PromptConfiguration, PromptManifest, load_manifest
from portstudy.pipeline.frozen import (
    FROZEN_FILES,
    accept_config_change,
    frozen_config_diff,
    save_frozen_config,
)
from portstudy.pipeline.preflight import PreflightReport, live_ready
from portstudy.risk.covariance import build_risk_model
from portstudy.risk.prior import build_prior

REPO_ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")


class FakeHttp:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def get_json(self, url, params=None, headers=None):  # type: ignore[no-untyped-def]
        self.calls.append({"url": url, "params": params or {}})
        return self.responses.pop(0)

    def get_text(self, url, params=None, headers=None):  # type: ignore[no-untyped-def]
        return str(self.responses.pop(0))


def _bar(day: str, volume: float = 100.0) -> dict[str, object]:
    return {"t": f"{day}T05:00:00Z", "o": 10, "h": 11, "l": 9, "c": 10.5, "v": volume}


def _bars_payload(ticker: str, days: list[str]) -> dict[str, object]:
    return {
        "bars": {ticker: [_bar(d) for d in days]},
        "next_page_token": None,
    }


# G16 + C11: feed parameter and cutoff-day end timestamp -----------------------------------------


def test_alpaca_bars_send_feed_and_rfc3339_end() -> None:
    """The configured feed goes out on the bars request and `end` is the cutoff day at
    23:59:59 ET when that is safely in the past (C11)."""
    http = FakeHttp([_bars_payload("AAA", ["2026-01-09"]), _bars_payload("AAA", ["2026-01-09"])])
    clock = lambda: dt.datetime(2026, 1, 20, 12, 0, tzinfo=dt.UTC)  # noqa: E731
    provider = AlpacaPriceProvider(http, key_id="k", secret="s", feed="sip", clock=clock)
    provider.get_prices(["AAA"], dt.date(2026, 1, 1), dt.date(2026, 1, 9))
    for call in http.calls:
        params = call["params"]
        assert params["feed"] == "sip"
        assert params["end"] == "2026-01-09T23:59:59-05:00"


def test_alpaca_end_never_later_than_now_minus_16() -> None:
    """On the cutoff day itself the end timestamp is capped at now minus 16 minutes."""
    now_et = dt.datetime(2026, 1, 9, 12, 0, tzinfo=ET)
    http = FakeHttp([_bars_payload("AAA", ["2026-01-09"]), _bars_payload("AAA", ["2026-01-09"])])
    provider = AlpacaPriceProvider(http, key_id="k", secret="s", feed="sip", clock=lambda: now_et)
    provider.get_prices(["AAA"], dt.date(2026, 1, 1), dt.date(2026, 1, 9))
    expected = (now_et - dt.timedelta(minutes=16)).isoformat()
    for call in http.calls:
        assert call["params"]["end"] == expected


def test_alpaca_iex_feed_is_sent_when_configured() -> None:
    http = FakeHttp([_bars_payload("AAA", ["2026-01-09"]), _bars_payload("AAA", ["2026-01-09"])])
    clock = lambda: dt.datetime(2026, 1, 20, tzinfo=dt.UTC)  # noqa: E731
    provider = AlpacaPriceProvider(http, key_id="k", secret="s", feed="iex", clock=clock)
    provider.get_prices(["AAA"], dt.date(2026, 1, 1), dt.date(2026, 1, 9))
    assert http.calls[0]["params"]["feed"] == "iex"


# alpaca_check in scripts/smoke_real_services.py -------------------------------------------------


def _smoke():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import smoke_real_services

    return smoke_real_services


def test_alpaca_check_passes_when_last_bar_is_last_trading_day(
    study_config: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALPACA_API_KEY_ID", "k")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "s")
    smoke = _smoke()
    days = cal.trading_days(dt.date(2026, 1, 5), dt.date(2026, 1, 9))
    assert days[-1] == dt.date(2026, 1, 9)
    http = FakeHttp(
        [
            _bars_payload("AAA", [d.isoformat() for d in days]),
            _bars_payload("AAA", [d.isoformat() for d in days]),
            {"news": [{"created_at": "2026-01-08T12:00:00Z", "symbols": ["AAA"]}]},
        ]
    )
    out = smoke.alpaca_check(study_config, http, "AAA", dt.date(2026, 1, 5), dt.date(2026, 1, 9))
    assert "feed=sip" in out and "last bar 2026-01-09" in out and "1 articles" in out


def test_alpaca_check_fails_when_cutoff_day_bar_missing(
    study_config: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALPACA_API_KEY_ID", "k")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "s")
    smoke = _smoke()
    days = [d for d in cal.trading_days(dt.date(2026, 1, 5), dt.date(2026, 1, 9))][:-1]
    http = FakeHttp(
        [
            _bars_payload("AAA", [d.isoformat() for d in days]),
            _bars_payload("AAA", [d.isoformat() for d in days]),
            {"news": []},
        ]
    )
    with pytest.raises(AssertionError, match="last trading day"):
        smoke.alpaca_check(study_config, http, "AAA", dt.date(2026, 1, 5), dt.date(2026, 1, 9))


# C8: name/sector from the frozen universe and market-cap carry ----------------------------------


class _CustomReference:
    """Reference provider that returns a wrong name and a missing sector; used to prove the
    snapshot ignores provider identity fields."""

    name = "custom"

    def get_reference(self, tickers: list[str], as_of: dt.date) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": tickers,
                "name": ["WRONG"] * len(tickers),
                "sector": [None] * len(tickers),
                "market_cap": [1e9] * len(tickers),
                "next_earnings_date": [None] * len(tickers),
            }
        )


class _RaisingReference:
    name = "raising"

    def get_reference(self, tickers: list[str], as_of: dt.date) -> pd.DataFrame:
        raise RuntimeError("reference provider down")


def _bundle_with_reference(providers: ProviderBundle, reference) -> ProviderBundle:
    return ProviderBundle(
        prices=providers.prices,
        news=providers.news,
        reference=reference,
        rates=providers.rates,
        members=providers.members,
    )


def test_carry_market_caps_pure() -> None:
    ref = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "market_cap": [1e9, float("nan"), -5.0],
        }
    )
    out, issues = carry_market_caps(ref, lambda tk: 5e8 if tk == "BBB" else None)
    assert out.set_index("ticker").at["BBB", "market_cap"] == pytest.approx(5e8)
    carried = [i for i in issues if i.code == "market_cap_carried"]
    assert len(carried) == 1 and carried[0].severity == "info" and carried[0].rule == "carry"
    missing = [i for i in issues if i.code == "market_cap_missing"]
    assert [i.ticker for i in missing] == ["CCC"]
    assert missing[0].severity == "warning" and missing[0].rule == "carry"
    assert any(i.code == "market_caps_degraded" and i.ticker == "*" for i in issues)


def test_carry_market_caps_details_name_the_source_week() -> None:
    ref = pd.DataFrame({"ticker": ["AAA"], "market_cap": [float("nan")]})
    out, issues = carry_market_caps(ref, lambda tk: 5e8, lambda tk: "2026-W01")
    assert issues[0].details["from_week"] == "2026-W01"
    assert out.at[0, "market_cap"] == pytest.approx(5e8)


def test_snapshot_name_sector_come_from_universe_not_provider(
    data_dir: Path,
    study_config: StudyConfig,
    providers: ProviderBundle,
    universe: pd.DataFrame,
    session_weeks: list[str],
) -> None:
    bundle = _bundle_with_reference(providers, _CustomReference())
    week = session_weeks[0]
    make_snapshot(week, study_config, bundle, universe)
    snap = read_snapshot(week)
    uni = universe.set_index(universe["ticker"].astype(str))
    merged = snap.reference.set_index("ticker")
    for tk in merged.index:
        if tk in uni.index:
            assert merged.at[tk, "name"] == uni.at[tk, "name"]
            assert merged.at[tk, "sector"] == uni.at[tk, "sector"]


def test_reference_outage_carries_caps_and_j2_prior_works(
    data_dir: Path,
    study_config: StudyConfig,
    providers: ProviderBundle,
    universe: pd.DataFrame,
    session_weeks: list[str],
) -> None:
    """Week 2 with the reference provider down carries week 1's caps; J2's market prior still
    computes on the carried values."""
    week1, week2 = session_weeks[0], session_weeks[1]
    make_snapshot(week1, study_config, providers, universe)
    caps_w1 = read_snapshot(week1).reference.set_index("ticker")["market_cap"]

    cutoff2 = cal.cutoff_day(week2)
    uni2 = build_universe(cutoff2, study_config, providers, print_table=False)
    bundle = _bundle_with_reference(providers, _RaisingReference())
    make_snapshot(week2, study_config, bundle, uni2)
    snap2 = read_snapshot(week2)
    ref2 = snap2.reference.set_index("ticker")

    issues = snap2.issues
    carried = issues[issues["code"] == "market_cap_carried"]
    assert len(carried), "expected carried caps when the reference call fails"
    assert (carried["severity"] == "info").all() and (carried["rule"] == "carry").all()
    for tk, cap in caps_w1.items():
        if tk in ref2.index:
            assert ref2.at[tk, "market_cap"] == pytest.approx(cap)
    degraded = issues[(issues["code"] == "market_caps_degraded") & (issues["ticker"] == "*")]
    assert len(degraded) == 1
    assert snap2.manifest["n_market_cap_carried"] == len(carried)

    # J2's prior path still computes on the carried caps.
    model = build_risk_model(week2, study_config.risk, snap2)
    prior = build_prior(week2, study_config.risk, model, snap2.reference)
    assert prior["w_mkt"].sum() == pytest.approx(1.0)


def test_blank_universe_sector_halts_before_writing(
    data_dir: Path,
    study_config: StudyConfig,
    providers: ProviderBundle,
    universe: pd.DataFrame,
    session_weeks: list[str],
) -> None:
    uni = universe.copy()
    uni.loc[uni.index[0], "sector"] = " "
    week = session_weeks[0]
    with pytest.raises(SnapshotHalt):
        make_snapshot(week, study_config, providers, uni)
    assert not list((data_dir / "snapshots").glob(f"{week}*"))


# Frozen config ----------------------------------------------------------------------------------


def _frozen_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel in FROZEN_FILES:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"original {rel}\n", encoding="utf-8")
    return root


def test_frozen_config_save_diff_accept(data_dir: Path, tmp_path: Path) -> None:
    root = _frozen_root(tmp_path)
    folder = save_frozen_config(root)
    assert (folder / "frozen.json").exists()
    assert frozen_config_diff(root) == []

    (root / "config" / "study.yaml").write_text("changed\n", encoding="utf-8")
    assert frozen_config_diff(root) == ["study.yaml"]

    with pytest.raises(ValueError):
        accept_config_change(root, "   ")
    accept_config_change(root, "raised the cost goal")
    assert frozen_config_diff(root) == []
    lines = (folder / "changes.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and "raised the cost goal" in lines[0]


# live_ready -------------------------------------------------------------------------------------


def _probes() -> dict[str, object]:
    return {
        "gateway_probe": lambda _url: None,
        "disk_free_gb": lambda _p: 100.0,
        "gpu_free_gb": lambda: 8.0,
        "clock_skew_s": lambda _url: 3.0,
    }


def test_live_ready_real_repo_lists_every_known_failure(data_dir: Path) -> None:
    cfg = load_study_config()
    registry = load_model_registry()
    manifest = load_manifest()
    report = live_ready(
        cfg,
        registry,
        manifest,
        root=REPO_ROOT,
        env={},
        now=dt.datetime(2026, 1, 10, tzinfo=dt.UTC),
        gateway_probe=lambda _u: "timed out",
        disk_free_gb=lambda _p: 100.0,
        gpu_free_gb=lambda: 8.0,
        clock_skew_s=lambda _u: None,
    )
    text = "\n".join(report.failures)
    assert "PLACEHOLDER" in text
    assert "UNPINNED" in text
    assert "forecaster is stub" in text
    assert "live_end_date" in text and "pilot_week" in text
    for key in (
        "ARCTIC_LITELLM_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "ALPACA_API_KEY_ID",
        "ALPACA_API_SECRET_KEY",
        "FINNHUB_API_KEY",
        "FRED_API_KEY",
    ):
        assert key in text, key
    assert "snapshot_id null" in text
    assert "price.confirmed false" in text
    assert "vendor/kronos" in text
    assert "universe file" in text
    assert "GSU VPN" in text
    assert set(report.markers) == {"TBD", "UNVERIFIED", "UNPINNED"}


@pytest.fixture()
def ready_setup(
    data_dir: Path,
    tmp_path: Path,
    study_config: StudyConfig,
    model_registry: ModelRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    """A complete fixture config: pinned kronos, dated timing, confirmed API prices with
    snapshot ids, a tmp root with a universe file and a fake vendor/kronos."""
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    (root / "vendor" / "kronos").mkdir(parents=True)
    (root / "config" / "universe.csv").write_text(
        "ticker,name,sector\nAAA,A Corp,Tech\nBBB,B Corp,Health\n", encoding="utf-8"
    )
    # v4 phase 4: a frozen copy of the test pre-registration (the preflight refuses an unfrozen one).
    prereg = (REPO_ROOT / "config" / "preregistration_test.yaml").read_text(encoding="utf-8")
    prereg = prereg.replace("frozen: false", "frozen: true").replace(
        "frozen_on: null", 'frozen_on: "2026-11-20"'
    )
    prereg = prereg.replace(
        "view_sensitivity_min_tracking_error: null", "view_sensitivity_min_tracking_error: 0.0005"
    )
    (root / "config" / "preregistration_test.yaml").write_text(prereg, encoding="utf-8")

    pinned = "abc123def456"
    kronos = study_config.baseline.kronos.model_copy(update={"repo_commit": pinned})
    baseline = study_config.baseline.model_copy(update={"forecaster": "kronos", "kronos": kronos})
    timing = study_config.timing.model_copy(
        update={"pilot_week": "2026-W10", "live_end_date": "2030-12-31"}
    )
    cfg = study_config.model_copy(update={"baseline": baseline, "timing": timing})

    models = []
    for spec in model_registry.models:
        if spec.kind == "api":
            spec = spec.model_copy(
                update={
                    "snapshot_id": f"{spec.id}-2026-01-01",
                    "price": spec.price.model_copy(update={"confirmed": True}),
                }
            )
        models.append(spec)
    registry = ModelRegistry(models=models, stub_models=model_registry.stub_models)

    manifest = PromptManifest(
        configurations=[PromptConfiguration(id="p1", kind="bare", enabled=True)]
    )
    env = {
        "ARCTIC_LITELLM_API_KEY": "k",
        "ANTHROPIC_API_KEY": "k",
        "OPENAI_API_KEY": "k",
        "ALPACA_API_KEY_ID": "k",
        "ALPACA_API_SECRET_KEY": "k",
        "FINNHUB_API_KEY": "k",
        "FRED_API_KEY": "k",
    }
    monkeypatch.setattr("portstudy.pipeline.preflight._kronos_head", lambda _root: pinned)
    return {
        "cfg": cfg,
        "registry": registry,
        "manifest": manifest,
        "root": root,
        "env": env,
    }


def _call_ready(setup: dict[str, object], **overrides: object) -> PreflightReport:
    kwargs = {
        "root": setup["root"],
        "env": setup["env"],
        "now": dt.datetime(2026, 1, 10, tzinfo=dt.UTC),
        "week_id": "2026-W10",
        **_probes(),
        **overrides,
    }
    return live_ready(setup["cfg"], setup["registry"], setup["manifest"], **kwargs)  # type: ignore[arg-type]


def test_live_ready_fixture_config_passes(ready_setup: dict[str, object]) -> None:
    report = _call_ready(ready_setup)
    assert report.failures == [], report.failures


def test_live_ready_pre_pilot_week_folder_fails(
    ready_setup: dict[str, object], data_dir: Path
) -> None:
    (data_dir / "outputs" / "2026-W05").mkdir(parents=True)
    (data_dir / "snapshots" / "2026-W04.v3").mkdir(parents=True)
    report = _call_ready(ready_setup)
    assert any("2026-W05" in f and "pilot_week" in f for f in report.failures)
    assert any("2026-W04.v3" in f for f in report.failures)


def test_live_ready_features_frozen_after_pilot(ready_setup: dict[str, object]) -> None:
    report = _call_ready(ready_setup, week_id="2026-W11")
    assert any("features_frozen" in f for f in report.failures)
    report = _call_ready(ready_setup, week_id="2026-W10")
    assert not any("features_frozen" in f for f in report.failures)


def test_live_ready_frozen_diff_fails(
    ready_setup: dict[str, object], data_dir: Path, tmp_path: Path
) -> None:
    root = _frozen_root(tmp_path / "frozen_repo")
    save_frozen_config(root)
    (root / "config" / "models.yaml").write_text("changed\n", encoding="utf-8")
    report = _call_ready(ready_setup, root=root)
    assert any(
        "config changed since the pilot" in f and "models.yaml" in f for f in report.failures
    )


def test_live_ready_gpu_warning_never_fails(ready_setup: dict[str, object]) -> None:
    report = _call_ready(ready_setup, gpu_free_gb=lambda: None)
    assert any("torch not installed" in w for w in report.warnings)
    report = _call_ready(ready_setup, gpu_free_gb=lambda: 0.5)
    assert any("GPU free memory" in w for w in report.warnings)
    assert report.failures == []


# run_weekend wiring -----------------------------------------------------------------------------


def _run_weekend():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_weekend

    return run_weekend


class _Ctx:
    def __init__(self, synthetic: bool) -> None:
        self.cfg = load_study_config()
        self.registry = load_model_registry()
        self.manifest = load_manifest()
        self.synthetic = synthetic
        self.clock = None

    def now(self) -> dt.datetime:
        return dt.datetime(2026, 3, 6, 20, 0, tzinfo=ET)


def _stub_weekend(rw, monkeypatch: pytest.MonkeyPatch, synthetic: bool) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(rw, "default_context", lambda **kw: _Ctx(synthetic))
    monkeypatch.setattr(rw, "step1_catch_up", lambda ctx, w, s: calls.append("catch_up"))
    monkeypatch.setattr(rw, "run", lambda ctx, job, w, s: calls.append(job) or "done")
    monkeypatch.setattr(rw, "gateway_health", lambda ctx, w: calls.append("health"))
    monkeypatch.setattr(rw, "wait_for_batches", lambda ctx, w, s, p: calls.append("J5"))
    monkeypatch.setattr(rw, "print_summary", lambda w: None)
    # Phase 3A steps that read real run files: the stub context has none.
    monkeypatch.setattr(rw, "dollar_budget_check", lambda ctx, w: None)
    monkeypatch.setattr(rw, "model_checks", lambda ctx, w: [])
    monkeypatch.setattr(rw, "report_progress", lambda ctx, w: [])
    monkeypatch.setattr(rw, "audit_week", lambda ctx, w: [])
    monkeypatch.setattr(rw, "finish_run", lambda *a, **k: None)
    monkeypatch.setattr(rw, "start_keep_awake", lambda: None)
    return calls


def test_run_weekend_synthetic_never_calls_live_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rw = _run_weekend()
    monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(tmp_path / "data"))
    _stub_weekend(rw, monkeypatch, synthetic=True)

    def _boom(*a, **k):  # type: ignore[no-untyped-def]
        raise AssertionError("live_ready must not run for synthetic")

    monkeypatch.setattr(rw, "live_ready", _boom)
    assert rw.main(["--synthetic", "--week", "2026-W05", "--from", "J3"]) == 0


def test_run_weekend_preflight_failure_returns_2_before_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rw = _run_weekend()
    monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(tmp_path / "data"))
    calls = _stub_weekend(rw, monkeypatch, synthetic=False)
    monkeypatch.setattr(rw, "live_ready", lambda *a, **k: PreflightReport(failures=["bad config"]))
    assert rw.main(["--week", "2026-W05", "--from", "J3"]) == 2
    assert calls == []


def test_run_weekend_skip_preflight_proceeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rw = _run_weekend()
    monkeypatch.setenv("PORTSTUDY_DATA_DIR", str(tmp_path / "data"))
    calls = _stub_weekend(rw, monkeypatch, synthetic=False)

    def _boom(*a, **k):  # type: ignore[no-untyped-def]
        raise AssertionError("live_ready must not run with --skip-preflight")

    monkeypatch.setattr(rw, "live_ready", _boom)
    assert rw.main(["--week", "2026-W05", "--from", "J3", "--skip-preflight"]) == 0
    assert calls[0] == "catch_up" and "J3" in calls


def test_clock_skew_beyond_two_minutes_fails(ready_setup: dict[str, object]) -> None:
    assert not any("hwclock" in f for f in _call_ready(ready_setup).failures)
    report = _call_ready(ready_setup, clock_skew_s=lambda _u: 300.0)
    assert any("sudo hwclock -s" in f and "+300 s" in f for f in report.failures)
