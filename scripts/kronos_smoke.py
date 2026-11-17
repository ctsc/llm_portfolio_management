"""Time one Kronos baseline week on synthetic OHLCV and report GPU memory.

Usage: python scripts/kronos_smoke.py --n-stocks 300 --device cuda
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from portstudy.baseline.kronos import KronosForecaster
from portstudy.config import KronosConfig


def synthetic_ohlcv(n_stocks: int, days: int, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-11-06", periods=days)
    out = {}
    for i in range(n_stocks):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, days)))
        open_ = close * (1 + rng.normal(0, 0.003, days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, days)))
        vol = rng.integers(1_000_000, 10_000_000, days).astype(float)
        out[f"T{i:03d}"] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-stocks", type=int, default=300)
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    import torch

    print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        torch.cuda.reset_peak_memory_stats()

    cfg = KronosConfig()
    data = synthetic_ohlcv(a.n_stocks, cfg.context_candles + 10, seed=1)
    fc = KronosForecaster(cfg, device=a.device)

    t0 = time.perf_counter()
    fc._load()
    t1 = time.perf_counter()
    out = fc.forecast(data, horizon=cfg.horizon, n_samples=a.n_samples, seed=cfg.seed)
    t2 = time.perf_counter()

    print(f"load_s={t1 - t0:.1f}  forecast_s={t2 - t1:.1f}", end="  ")
    print(f"stocks={a.n_stocks}  samples={a.n_samples}")
    print(f"rows={len(out)}  nan_k={int(out['k'].isna().sum())}")
    if torch.cuda.is_available():
        print(f"peak_gpu_mem_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}")


if __name__ == "__main__":
    main()
