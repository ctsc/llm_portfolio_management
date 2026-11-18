# Desktop setup (WSL2)

The weekly pipeline runs on Carter's Windows desktop inside WSL2 Ubuntu. Only the LLM calls for
the two open-weight models go to ARCTIC, over HTTPS to its LiteLLM gateway
(`https://litellm.rs.gsu.edu/v1`, GSU network or VPN required). API models are called through
their providers' batch APIs. Nothing runs on ARCTIC compute nodes.

Reference machine (measured 2026-11-10): AMD Ryzen 5 8400F (6 cores, 12 threads), 31.6 GB RAM,
NVIDIA RTX 3050 with 6 GB, WSL2 Ubuntu, CUDA visible inside WSL.

## Why WSL2

- The weekly run is built and timed for Linux paths and tools (`rsync`, the keep-awake helper).
  The code and offline tests also run on native Windows (peak memory in the benchmark prints `n/a`
  there unless `psutil` is installed).
- The NVIDIA GPU is usable from WSL2.
- Do not run from a OneDrive folder through `/mnt/c/...`. OneDrive-synced folders refuse WSL
  `chmod` and editable installs ("Operation not permitted"), so `pip install -e` and `git clone`
  fail there. The Windows copy of the repository now lives at `C:\dev\portfoliomanagement`, outside
  OneDrive (`manualwork.md` 0.1); the working checkout for runs is still the WSL clone in `~/pm`.

## One-time setup

In an admin PowerShell, only if Ubuntu is not installed yet:

```powershell
wsl --install -d Ubuntu
```

Open Ubuntu (`wsl -d Ubuntu` from a PowerShell tab if Windows Terminal has no Ubuntu entry yet)
and run one line at a time, waiting for the `$` prompt after each:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv venv ~/pm311 --python 3.11
source ~/pm311/bin/activate
git clone https://github.com/ctsc/llm_portfolio_management.git ~/pm
cd ~/pm
uv sync --extra ml --extra solvers --extra data --extra api --extra stats --extra dev
nvidia-smi
```

Notes:
- `uv sync` installs exactly the versions pinned in `uv.lock`. It downloads PyTorch with CUDA (a
  few GB) and can print nothing for minutes. Do not press Ctrl+C.
- `einops` and `safetensors` (needed by Kronos) are in the `ml` extra; `arch` (SPA, StepM, MCS in
  the report) is in the `stats` extra. `python scripts/kronos_smoke.py` checks Kronos on the GPU.
- The setuptools warning about `project.license` as a TOML table is harmless.
- To update the checkout later: `cd ~/pm && git pull`. Edits made in the Windows copy reach WSL only
  through git.

## Kronos

```bash
cd ~/pm
git clone https://github.com/shiyu-coder/Kronos vendor/kronos
git -C vendor/kronos checkout <pinned commit>
```

Set `baseline.kronos.repo_commit` to that commit and `baseline.forecaster: kronos` in
`config/study.yaml` (it is `stub` until then). `vendor/kronos/` is gitignored. The model weights
(`NeoQuasar/Kronos-base`, `NeoQuasar/Kronos-Tokenizer-base`) download from Hugging Face on first
use (about 409 MB).

Smoke test (synthetic prices, reports load time, forecast time and peak GPU memory):

```bash
python scripts/kronos_smoke.py --n-stocks 100
python scripts/kronos_smoke.py --n-stocks 300
```

Measured 2026-11-10: 100 stocks in 195 s at 1.79 GB peak; 300 stocks in 3,373 s at 4.49 GB peak
with other apps holding about 1.2 GB of GPU memory, which likely spilled into shared memory. Close
GPU-heavy apps before J2. Kronos forecasts in chunks of `baseline.kronos.batch_size` (default 100) and
frees GPU memory between chunks; `python scripts/kronos_smoke.py --n-stocks 300 --batch-size 100`
prints per-chunk seconds and peak GPU memory. The chunked timing has not been measured on the
desktop yet (`manualwork.md` B3).

## Keys and network

Keys live in `~/.env` inside WSL (mode 600), never in the repository:

```
ARCTIC_LITELLM_API_KEY=...
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
ALPACA_API_KEY_ID=...
ALPACA_API_SECRET_KEY=...
FINNHUB_API_KEY=...
FRED_API_KEY=...
```

ARCTIC key: the project is approved in Elpis (2026-11-16). On the GSU network or VPN, sign in once at
https://litellm.rs.gsu.edu/ui with your campus ID, ask ARCTIC (Andalib) to add you to the
project's team on the service (the wiki also lists a Hydra ticket), then create your own keys there (Virtual Keys, "+ Create New Key", Team =
the project). A key is shown only once; save it in your password manager. Use one key per tool.
See `llm_research/09_arctic_process_and_model_facts.md` for the access route and what ARCTIC logs. The gateway is an alpha service and needs the
GSU network or VPN. Check it without a key:

```bash
curl -sS https://litellm.rs.gsu.edu/health/liveliness
```

It should print "I'm alive!". MiniMax M2.7 thinks before answering and the thinking counts toward
the output limit (ARCTIC suggests about 2,000 tokens for it).

Connect the GSU VPN before J4. `python scripts/smoke_real_services.py --dry-run` lists what each
service check would call; without `--dry-run` it makes one tiny real call per service and enabled
model.

## Backups

J9 writes `data/backups/<week>_<timestamp>/` and a zip. Set `backup.rsync_target` to a folder
outside the WSL home so the backup survives a WSL problem, for example
`/mnt/c/Users/carte/portstudy_backups`, and copy it to cloud storage. Whether WSL can write into a
OneDrive-synced folder is unverified; J9 probes the target first (it writes and deletes a small file) and prints a
warning naming the target when it is not writable. The local backup is written either way.

## Benchmarks

```bash
python scripts/benchmark_pipeline.py --n-stocks 300 --sources 3 --weeks 24
python scripts/benchmark_pipeline.py --n-stocks 100 --sources 3 --weeks 24
```

Measured 2026-11-10 on the reference machine (mean seconds per solve, 300 / 100 stocks): MVO
0.079 / 0.009, Mean-CVaR 114.4 / 21.6, Robust MVO 3.60 / 0.020, Resampled MVO 11.7 / 1.1, Top-K
0.003 / 0.001. J7 per week at 3 sources: 519 s / 91 s; projected at 51 sources: 6,748 s / 1,181 s.
Peak memory 1.4 GB / 0.6 GB.

Open a second tab and run `top` to watch progress; a `python` process at 100% or more CPU is
working (300% means about three threads busy).

## Reliability settings before live weekends

- Windows sleep set to Never during runs.
- Windows Update paused for the study (Settings, Windows Update, Pause updates, up to 5 weeks at
  a time; renew it). Active hours cannot cover a weekend.
- GSU VPN connected during J4.
- GPU-heavy apps closed during J2 (Overwolf, Discord, Teams, Steam, Epic, games).
- Backup copied off the desktop after J9.

See `docs/weekend_checklist.md` for the per-weekend routine and `docs/runbook.md` for failures.
