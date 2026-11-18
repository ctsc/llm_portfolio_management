"""H5 factor regressions on daily returns (DECISION(G5)).

Factor stand-ins from the ETF proxies in the ledger: market = SPY minus T-bill, size = IWM
minus SPY, value = IWD minus IWF, profitability = QUAL minus SPY; investment is unavailable
and reported as a gap. If an official Fama-French daily file is present at
`eval.fama_french_file`, MKT-RF, SMB, HML, RMW, CMA, and RF are used instead.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from portstudy.data import ledger
from portstudy.data.snapshot import read_snapshot, snapshot_exists
from portstudy.eval.tests import default_maxlags

ETF_FACTORS = ["MKT_RF", "SMB", "HML", "RMW"]
FF_FACTORS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
FRENCH_COLUMNS = [*FF_FACTORS, "MOM", "RF"]


def daily_rf_from_snapshots(weeks: list[str]) -> pd.Series:
    """Daily risk-free rate from the T-bill series stored in the snapshots (percent, annual)."""
    frames = []
    for w in weeks:
        if snapshot_exists(w):
            r = read_snapshot(w).rates
            if len(r):
                frames.append(r)
    if not frames:
        return pd.Series(dtype=float)
    rates = pd.concat(frames).drop_duplicates("date").set_index("date")["rate"].astype(float)
    return (rates / 100.0 / 252.0).sort_index().rename("rf")


def etf_factors(rf: pd.Series) -> pd.DataFrame:
    led = ledger.read_ledger()
    etfs = led[led["ticker"].isin(["SPY", "IWM", "IWD", "IWF", "QUAL"])]
    closes = etfs.pivot(index="date", columns="ticker", values="close").astype(float).sort_index()
    rets = closes.pct_change().dropna(how="all")
    rf_aligned = rf.reindex(rets.index).ffill().fillna(0.0)
    out = pd.DataFrame(index=rets.index)
    out["MKT_RF"] = rets["SPY"] - rf_aligned
    out["SMB"] = rets["IWM"] - rets["SPY"]
    out["HML"] = rets["IWD"] - rets["IWF"]
    out["RMW"] = rets["QUAL"] - rets["SPY"]
    out["RF"] = rf_aligned
    return out.dropna()


def fama_french_factors(path: Path) -> pd.DataFrame:
    """Read the Ken French daily five-factor CSV (percent units) into decimal returns."""
    raw = pd.read_csv(path, skiprows=3)
    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw = raw[raw["date"].astype(str).str.fullmatch(r"\d{8}")]
    raw["date"] = pd.to_datetime(raw["date"].astype(str), format="%Y%m%d").dt.date
    df = raw.set_index("date").astype(float) / 100.0
    df = df.rename(columns={"Mkt-RF": "MKT_RF"})
    return df.loc[:, [*FF_FACTORS, "RF"]]


def regress_portfolio(
    daily_values: pd.DataFrame, factors: pd.DataFrame, names: list[str]
) -> dict[str, float]:
    """Alpha, betas, R squared, HAC errors for one portfolio's daily excess returns."""
    values = daily_values.set_index("date")["value"].astype(float).sort_index()
    rets = values.pct_change().dropna()
    joined = pd.concat([rets.rename("ret"), factors], axis=1, join="inner").dropna()
    out: dict[str, float] = {"n_days": float(len(joined))}
    if len(joined) < 20:
        return out
    y = joined["ret"] - joined["RF"]
    x = sm.add_constant(joined[names])
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": default_maxlags(len(joined))})
    out["alpha_daily"] = float(model.params["const"])
    out["alpha_annual"] = float(model.params["const"] * 252)
    out["alpha_t"] = float(model.tvalues["const"])
    out["alpha_p"] = float(model.pvalues["const"])
    for n in names:
        out[f"beta_{n}"] = float(model.params[n])
        out[f"t_{n}"] = float(model.tvalues[n])
    out["r_squared"] = float(model.rsquared)
    return out


def factor_table(
    daily_by_portfolio: dict[str, pd.DataFrame], weeks: list[str], fama_french_file: str | None
) -> tuple[pd.DataFrame, str]:
    if fama_french_file and Path(fama_french_file).exists():
        factors, names, source = (
            fama_french_factors(Path(fama_french_file)),
            FF_FACTORS,
            "fama_french",
        )
    else:
        factors, names, source = (
            etf_factors(daily_rf_from_snapshots(weeks)),
            ETF_FACTORS,
            "etf_proxies",
        )
    rows = []
    for key, daily in daily_by_portfolio.items():
        if len(daily) == 0:
            continue
        rows.append({"portfolio": key, **regress_portfolio(daily, factors, names)})
    df = pd.DataFrame(rows)
    if source == "etf_proxies" and len(df):
        df["beta_CMA"] = np.nan  # investment factor unavailable from ETF proxies
    return df, source


# Kenneth French daily files (DECISION(G11)) ---------------------------------------------------


class FactorCoverageError(RuntimeError):
    """The factor file ends before the study's last day."""


def _french_text(path: Path) -> str:
    """The CSV text of a French Data Library file, from the zip or an extracted CSV."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            return zf.read(name).decode("utf-8", errors="replace")
    return path.read_text(encoding="utf-8", errors="replace")


def read_french_daily(path: Path) -> pd.DataFrame:
    """Daily rows (YYYYMMDD) of a French CSV in decimals; header and footer lines are skipped."""
    lines = _french_text(path).splitlines()
    header = next(
        i
        for i, ln in enumerate(lines)
        if ln.strip().startswith(",") or "Mkt-RF" in ln or ln.strip().lower().startswith(",mom")
    )
    body = [lines[header]] + [
        ln
        for ln in lines[header + 1 :]
        if ln[:8].strip().isdigit() and len(ln.split(",")[0].strip()) == 8
    ]
    raw = pd.read_csv(io.StringIO("\n".join(body)))
    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw.columns = [str(c).strip() for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"].astype(str).str.strip(), format="%Y%m%d").dt.date
    df = raw.set_index("date").astype(float) / 100.0
    return df.rename(columns={"Mkt-RF": "MKT_RF", "Mom": "MOM", "WML": "MOM"})


def french_factors(ff5_path: Path, momentum_path: Path) -> pd.DataFrame:
    """MKT_RF, SMB, HML, RMW, CMA, MOM, RF in decimals, joined on date."""
    ff5 = read_french_daily(ff5_path)
    mom = read_french_daily(momentum_path)
    mom = mom.rename(columns={mom.columns[0]: "MOM"}) if "MOM" not in mom else mom
    joined = ff5.join(mom[["MOM"]], how="inner")
    return joined.loc[:, FRENCH_COLUMNS].sort_index()


def check_coverage(factors: pd.DataFrame, last_day: dt.date) -> None:
    if not len(factors) or max(factors.index) < last_day:
        end = max(factors.index).isoformat() if len(factors) else "nothing"
        raise FactorCoverageError(
            f"French factor file ends {end}, before the study's last day {last_day.isoformat()}; "
            "run scripts/fetch_french_factors.py once the library covers the study"
        )


def synthetic_french_files(
    world_prices: pd.DataFrame, folder: Path, seed: int = 11
) -> tuple[Path, Path]:
    """Seeded French-format FF5 and momentum CSVs from the synthetic world's ETF proxies, so tests
    and the synthetic e2e run exercise the French path."""
    import numpy as np

    closes = world_prices.pivot_table(index="date", columns="ticker", values="close")
    rets = closes.pct_change().dropna(how="all")
    rng = np.random.default_rng(seed)
    out = pd.DataFrame(index=rets.index)
    out["Mkt-RF"] = rets["SPY"]
    out["SMB"] = rets["IWM"] - rets["SPY"]
    out["HML"] = rets["IWD"] - rets["IWF"]
    out["RMW"] = rets["QUAL"] - rets["SPY"]
    out["CMA"] = rng.normal(0, 0.003, len(out))
    out["RF"] = 0.0001
    mom = pd.DataFrame({"Mom": rng.normal(0, 0.006, len(out))}, index=out.index)
    folder.mkdir(parents=True, exist_ok=True)

    def write(df: pd.DataFrame, path: Path, title: str) -> Path:
        stamp = [pd.Timestamp(d).strftime("%Y%m%d") for d in df.index]
        body = "\n".join(
            ",".join([s, *(f"{v * 100:.4f}" for v in row)]) for s, row in zip(stamp, df.to_numpy())
        )
        path.write_text(
            f"{title}\n\n," + ",".join(df.columns) + "\n" + body + "\n", encoding="utf-8"
        )
        return path

    return (
        write(out, folder / "F-F_Research_Data_5_Factors_2x3_daily.csv", "synthetic FF5"),
        write(mom, folder / "F-F_Momentum_Factor_daily.csv", "synthetic momentum"),
    )


FF5_NAME = "F-F_Research_Data_5_Factors_2x3_daily"
MOM_NAME = "F-F_Momentum_Factor_daily"


def _find(folder: Path, stem: str) -> Path | None:
    for suffix in ("_CSV.zip", ".csv", ".CSV", ".zip"):
        path = folder / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def load_factors(
    cfg: Any, weeks: list[str], synthetic: bool = False, require_coverage: bool = True
) -> tuple[pd.DataFrame | None, str]:
    """The study's daily factors and their label. `french`: the configured files (default
    `data/factors/`), with a synthetic set generated from the synthetic world's ETFs in synthetic
    runs; fails clearly when the files end before the study. `etf_proxy`: the ETF method, labeled
    for monitoring only (no MOM or CMA)."""
    from portstudy import calendar as cal
    from portstudy.paths import data_dir

    if cfg.eval.factor_source == "etf_proxy":
        return etf_factors(daily_rf_from_snapshots(weeks)), "ETF proxy, monitoring only"
    folder = data_dir() / "factors"
    ff5 = Path(cfg.eval.fama_french_file) if cfg.eval.fama_french_file else _find(folder, FF5_NAME)
    mom = Path(cfg.eval.momentum_file) if cfg.eval.momentum_file else _find(folder, MOM_NAME)
    if (ff5 is None or mom is None) and synthetic:
        led = ledger.read_ledger()
        ff5, mom = synthetic_french_files(led, folder)
    if ff5 is None or mom is None:
        if not require_coverage:
            return None, "french (files missing)"
        raise FactorCoverageError(
            f"French factor files not found in {folder}; run scripts/fetch_french_factors.py"
        )
    factors = french_factors(ff5, mom)
    if require_coverage and weeks:
        check_coverage(factors, cal.cutoff_day(weeks[-1]))
    return factors, "french"
