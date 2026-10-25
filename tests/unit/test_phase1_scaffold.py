"""Phase 1 tests: config, atomic IO, hashing, calendar."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from portstudy import calendar as cal
from portstudy import io as pio
from portstudy.config import (
    ModelRegistry,
    StudyConfig,
    load_model_registry,
    load_study_config,
)
from portstudy.hashing import sha256_file, sha256_text, stable_hash

REPO_ROOT = Path(__file__).resolve().parents[2]


# Config -------------------------------------------------------------------------------------


def test_config_loads_and_has_decisions(study_config: StudyConfig) -> None:
    assert study_config.views.tau == 0.05
    assert study_config.optimizer.max_weight == 0.05
    assert study_config.simulator.cost_bps == 10
    ids = {d.id for d in study_config.decisions}
    for required in ["D1", "D3", "D4", "B1", "B2", "B4", "A6", "C2", "E2", "E3", "G6", "N1"]:
        assert required in ids
    assert study_config.decision("D3").status == "provisional"
    assert study_config.decision("C2").value == study_config.llm.no_news_sentence


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    raw = yaml.safe_load((REPO_ROOT / "config" / "study.yaml").read_text(encoding="utf-8"))
    raw["views"]["unknown_knob"] = 1
    bad = tmp_path / "study.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_study_config(bad)


def test_config_rejects_bad_decision_status(tmp_path: Path) -> None:
    raw = yaml.safe_load((REPO_ROOT / "config" / "study.yaml").read_text(encoding="utf-8"))
    raw["decisions"][0]["status"] = "maybe"
    bad = tmp_path / "study.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_study_config(bad)


def test_model_registry_loads(model_registry: ModelRegistry) -> None:
    enabled = model_registry.enabled()
    assert {m.id for m in enabled} >= {"qwen3.5-4b", "phi4-mini", "claude-haiku-5.5", "gpt-6-luna"}
    assert model_registry.get("llama3.2-3b").enabled is False
    stubs = model_registry.enabled(synthetic=True)
    assert len(stubs) == 4
    assert all(m.provider == "stub" for m in stubs)


def test_model_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    raw = yaml.safe_load((REPO_ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    raw["models"].append(dict(raw["models"][0]))
    bad = tmp_path / "models.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_model_registry(bad)


# Atomic IO ----------------------------------------------------------------------------------


def test_write_csv_atomic_round_trip_and_byte_identical(tmp_path: Path) -> None:
    df = pd.DataFrame({"date": ["2026-01-02"], "ticker": ["AAA"], "close": [1.0 / 3.0]})
    p = tmp_path / "x.csv"
    pio.write_csv_atomic(df, p)
    first = p.read_bytes()
    pio.write_csv_atomic(df, p)
    assert p.read_bytes() == first
    assert b"0.333333333333" in first
    back = pio.read_csv(p)
    assert list(back.columns) == ["date", "ticker", "close"]
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_survives_simulated_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "x.json"
    pio.write_json_atomic({"a": 1}, p)
    original = os.replace

    def boom(src: object, dst: object) -> None:
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        pio.write_json_atomic({"a": 2}, p)
    monkeypatch.setattr(os, "replace", original)
    # The old file is intact and no temp file is left behind.
    assert pio.read_json(p) == {"a": 1}
    assert not list(tmp_path.glob("*.tmp"))


# Hashing ------------------------------------------------------------------------------------


def test_hashing_is_stable(tmp_path: Path) -> None:
    assert stable_hash({"b": 1, "a": [1, 2]}) == stable_hash({"a": [1, 2], "b": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})
    p = tmp_path / "f.txt"
    p.write_text("hello", encoding="utf-8")
    assert sha256_file(p) == sha256_text("hello")


# Calendar -----------------------------------------------------------------------------------


def test_calendar_good_friday_2027_uses_thursday() -> None:
    # 2027-03-26 is Good Friday; the cutoff moves to Thursday 2027-03-25.
    assert not cal.is_trading_day("2027-03-26")
    assert cal.cutoff_day(dt.date(2027, 3, 26)) == dt.date(2027, 3, 25)
    assert cal.cutoff_day(cal.week_id(dt.date(2027, 3, 26))) == dt.date(2027, 3, 25)


def test_calendar_christmas_and_new_year_fridays_closed() -> None:
    assert not cal.is_trading_day("2026-12-25")
    assert cal.cutoff_day(dt.date(2026, 12, 25)) == dt.date(2026, 12, 24)
    assert cal.is_early_close("2026-12-24")
    assert not cal.is_trading_day("2027-01-01")
    assert cal.cutoff_day(dt.date(2027, 1, 1)) == dt.date(2026, 12, 31)


def test_calendar_thanksgiving_half_day() -> None:
    assert not cal.is_trading_day("2026-11-26")
    assert cal.is_trading_day("2026-11-27")
    assert cal.is_early_close("2026-11-27")
    assert cal.cutoff_day(dt.date(2026, 11, 27)) == dt.date(2026, 11, 27)


def test_calendar_monday_holiday_trades_tuesday() -> None:
    # MLK day 2027-01-18 is a Monday holiday. Week 2027-W02's cutoff Friday is 2027-01-15.
    week = cal.week_id(dt.date(2027, 1, 15))
    assert week == "2027-W02"
    assert cal.monday_open_day(week) == dt.date(2027, 1, 19)
    assert cal.window_end_date(week) == dt.date(2027, 1, 25)


def test_week_id_and_alignment() -> None:
    assert cal.week_id(dt.date(2026, 11, 13)) == "2026-W46"
    assert cal.week_friday("2026-W46") == dt.date(2026, 11, 13)
    align = cal.week_alignment("2026-W46")
    assert align == {
        "week_id": "2026-W46",
        "cutoff_date": "2026-11-13",
        "trade_date": "2026-11-16",
        "window_end_date": "2026-11-23",
    }
    assert cal.next_week_id("2026-W53") == "2027-W01"
    assert cal.prev_week_id("2027-W01") == "2026-W53"


def test_trading_days_count_2026() -> None:
    days = cal.trading_days("2026-01-01", "2026-12-31")
    assert len(days) == 251
    assert all(d.weekday() < 5 for d in days)
