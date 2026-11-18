"""v4 phase 6: Kenneth French factor loader and the stock-week characteristic panel."""

from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portstudy.eval.characteristics import CHARACTERISTIC_COLUMNS, week_characteristics
from portstudy.eval.factors import (
    FRENCH_COLUMNS,
    FactorCoverageError,
    check_coverage,
    french_factors,
)

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "factors"
FF5 = FIX / "F-F_Research_Data_5_Factors_2x3_daily.csv"
MOM = FIX / "F-F_Momentum_Factor_daily.csv"


def test_loader_returns_seven_decimal_columns() -> None:
    f = french_factors(FF5, MOM)
    assert list(f.columns) == FRENCH_COLUMNS and len(f) == 5
    assert f.loc[dt.date(2026, 1, 5), "MKT_RF"] == pytest.approx(0.0052)
    assert f.loc[dt.date(2026, 1, 9), "MOM"] == pytest.approx(-0.006)
    assert f.loc[dt.date(2026, 1, 7), "RF"] == pytest.approx(0.00017)


def test_loader_reads_the_csv_inside_a_zip(tmp_path: Path) -> None:
    zipped = []
    for src in (FF5, MOM):
        z = tmp_path / f"{src.stem}_CSV.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.write(src, src.name)
        zipped.append(z)
    pd.testing.assert_frame_equal(french_factors(*zipped), french_factors(FF5, MOM))


def test_coverage_check_fails_clearly() -> None:
    f = french_factors(FF5, MOM)
    check_coverage(f, dt.date(2026, 1, 9))
    with pytest.raises(FactorCoverageError, match="before the study's last day 2026-01-16"):
        check_coverage(f, dt.date(2026, 1, 16))


def _world(n_days: int = 320, seed: int = 4):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    days = [d.date() for d in pd.bdate_range("2025-01-01", periods=n_days)]
    tickers = [f"S{i}" for i in range(6)]
    mkt = rng.normal(0.0004, 0.01, n_days)
    rets = {t: 0.8 * mkt + rng.normal(0, 0.01, n_days) for t in tickers}
    px = pd.DataFrame({t: 100 * np.cumprod(1 + r) for t, r in rets.items()}, index=days)
    spy = pd.Series(100 * np.cumprod(1 + mkt), index=days)
    factors = pd.DataFrame(
        {
            "MKT_RF": mkt,
            "SMB": rng.normal(0, 0.004, n_days),
            "HML": rng.normal(0, 0.004, n_days),
            "RMW": rng.normal(0, 0.004, n_days),
            "CMA": rng.normal(0, 0.003, n_days),
            "MOM": rng.normal(0, 0.005, n_days),
            "RF": 0.0001,
        },
        index=days,
    )
    ref = pd.DataFrame(
        {
            "ticker": tickers,
            "market_cap": np.linspace(1e9, 6e9, 6),
            "sector": ["A", "A", "B", "B", "C", "C"],
        }
    )
    base = pd.DataFrame({"ticker": tickers, "z_b": np.linspace(-1, 1, 6)})
    return days, px, spy, factors, ref, base


def test_characteristics_use_only_data_up_to_the_cutoff() -> None:
    days, px, spy, factors, ref, base = _world()
    cutoff = days[290]
    first = week_characteristics(cutoff, px, ref, base, factors, spy)
    assert set(CHARACTERISTIC_COLUMNS) <= set(first.columns)
    assert (
        first[["momentum_12_1", "beta_252d", "loading_hml", "volatility_63d"]].notna().all().all()
    )
    later = px.copy()
    later.loc[later.index > cutoff] *= 3.0  # a huge move after the cutoff
    f2 = factors.copy()
    f2.loc[f2.index > cutoff, "HML"] = 0.5
    s2 = spy.copy()
    s2.loc[s2.index > cutoff] *= 0.1
    second = week_characteristics(cutoff, later, ref, base, f2, s2)
    pd.testing.assert_frame_equal(first, second)
    assert first["log_market_cap_z"].abs().max() > 0  # cross-sectional z-scores


def test_synthetic_factor_files_load(tmp_path: Path) -> None:
    from portstudy.eval.factors import synthetic_french_files

    days = pd.bdate_range("2026-01-05", periods=30)
    rows = []
    rng = np.random.default_rng(1)
    for t in ("SPY", "IWM", "IWD", "IWF", "QUAL"):
        rows.append(
            pd.DataFrame(
                {
                    "date": [d.date() for d in days],
                    "ticker": t,
                    "close": 100 * np.cumprod(1 + rng.normal(0, 0.01, 30)),
                }
            )
        )
    ff5, mom = synthetic_french_files(pd.concat(rows), tmp_path)
    f = french_factors(ff5, mom)
    assert list(f.columns) == FRENCH_COLUMNS and len(f) == 29
