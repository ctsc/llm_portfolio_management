"""Download the Kenneth French daily FF5 (2x3) and momentum files into data/factors/ (G11).

Records the download date and the last date covered in data/factors/factors_meta.json, and refuses
to overwrite a file that already covers a later date. Run once the library covers the study.

# UNVERIFIED: confirm the file names on the French Data Library on the first run.

Usage: python scripts/fetch_french_factors.py [--force]
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import httpx

from portstudy.eval.factors import read_french_daily
from portstudy.io import read_json, write_json_atomic
from portstudy.paths import data_dir

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "ff5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "momentum": "F-F_Momentum_Factor_daily_CSV.zip",
}


def fetch(folder: Path, force: bool = False, get=httpx.get) -> dict[str, str]:  # type: ignore[no-untyped-def]
    folder.mkdir(parents=True, exist_ok=True)
    meta_path = folder / "factors_meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    for key, name in FILES.items():
        response = get(BASE + name, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
        tmp = folder / f".{name}.part"
        tmp.write_bytes(response.content)
        last = max(read_french_daily(tmp).index).isoformat()
        old = meta.get(key, {}).get("last_date")
        if old and old > last and not force:
            tmp.unlink()
            raise SystemExit(f"{name}: existing file covers {old}, download only {last}; kept it")
        tmp.replace(folder / name)
        meta[key] = {"file": name, "downloaded": dt.date.today().isoformat(), "last_date": last}
        print(f"{name}: covers through {last}")
    write_json_atomic(meta, meta_path)
    return {k: v["last_date"] for k, v in meta.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    fetch(data_dir() / "factors", force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
