"""Write the seeded synthetic world to data/synthetic/ so tests and the demo can reuse it.

Usage: python scripts/make_synthetic_world.py [--force]
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil

from portstudy import calendar as cal
from portstudy.config import load_study_config
from portstudy.data.providers.synthetic import SyntheticWorld, synthetic_bundle
from portstudy.data.universe import build_universe, universe_path
from portstudy.paths import synthetic_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="regenerate even if present")
    args = parser.parse_args()
    cfg = load_study_config()
    folder = synthetic_dir()
    if args.force and folder.exists():
        shutil.rmtree(folder)
    world = SyntheticWorld.generate(cfg.synthetic, first_week=cfg.synthetic.first_week)
    world.save(folder)
    as_of: dt.date = cal.cutoff_day(cfg.synthetic.first_week)
    build_universe(as_of, cfg, synthetic_bundle(world), out_path=universe_path(cfg, synthetic=True))
    print(f"synthetic world written to {folder}")
    print(f"weeks: {world.weeks[0]} to {world.weeks[-1]} ({len(world.weeks)} weeks)")


if __name__ == "__main__":
    main()
