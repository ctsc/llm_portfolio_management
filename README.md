# portstudy: LLM skills and investor mindsets study

With a quantitative portfolio optimizer and guardrails held fixed, this code tests whether an
LLM given a specialized skills file can beat or improve a quant baseline, using a weekly
pipeline simulated on live data. The build plan is `devin.md`; the agent's log is
`PROGRESS.md`; the formulas are in `docs/design.md`.

Everything here runs offline on a synthetic world with a stub LLM. Real services (Alpaca,
yfinance, Finnhub, FRED, Ollama, Anthropic, OpenAI) sit behind the same interfaces and are
switched on by the researcher in Phase 17 of `devin.md`.

## Install

Python 3.11 is required.

```
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS or Linux
pip install -e ".[dev]"
```

Optional extras: `.[data]` (alpaca-py, yfinance, finnhub-python, pandas_market_calendars),
`.[api]` (anthropic, openai, tiktoken), `.[ml]` (torch, transformers for Kronos). The core
imports and tests without them.

## Run the synthetic demo

```
python scripts/make_synthetic_world.py          # writes data/synthetic/ (optional, done on demand)
python scripts/run_weekend.py --synthetic --week 2026-W02
python scripts/run_weekend.py --synthetic --week 2026-W03
python scripts/run_weekend.py --synthetic --week 2026-W04
portstudy status --week 2026-W04 --synthetic
portstudy eval --synthetic --skip-shuffle
```

Each weekend run takes about 10 seconds. The report lands in `data/outputs/report/report.md`
with CSV tables and PNG figures beside it. Drop `--skip-shuffle` to run the shuffled-ratings
test (500 shuffles per portfolio by default; set `eval.shuffle.n_shuffles` lower to try it).

Other commands: `portstudy run-job J4 --week ... --synthetic`, `portstudy replay --synthetic`,
`portstudy backup --week ... --synthetic [--dry-run]`, `python scripts/benchmark_pipeline.py`.

## Run the tests

```
ruff check src tests scripts
mypy src scripts
pytest tests/unit tests/integration        # about 10 minutes
pytest tests/e2e                           # 25 synthetic weeks, about 7 minutes
python scripts/refresh_manifest.py --check
python scripts/check_placebo_length.py
python scripts/check_banned_words.py
python scripts/check_secrets.py
```

## Directory map

```
config/        study.yaml (every setting and decision), models.yaml, prompts_manifest.yaml
prompts/       base_instructions.md, skills/ and placebo/ (placeholders until Phase 17)
src/portstudy/
  config.py paths.py io.py hashing.py logging_setup.py calendar.py types.py
  data/        providers (synthetic, alpaca, yfinance, finnhub, fred, wikipedia), checks,
               ledger, snapshot, universe
  baseline/    momentum, kronos (stub and Kronos forecasters), combine
  risk/        covariance (Ledoit-Wolf), prior (market cap)
  llm/         prompts, schema, validate, cache, cost, runner, providers (stub, ollama,
               anthropic_batch, openai_batch)
  views/       ic (rank IC, IC store, trailing IC), residualize, build
  optimize/    black_litterman, mean_variance, build_weights
  sim/         simulator (ledger-driven, stops), records
  pipeline/    context, status, jobs J1 to J9, fallback, cli
  eval/        tests, rank_ic, data, hypotheses, factors, shuffle, report
  backup.py
scripts/       run_weekend.py, make_synthetic_world.py, refresh_manifest.py,
               check_placebo_length.py, check_banned_words.py, check_secrets.py,
               estimate_cost.py, benchmark_pipeline.py, smoke_real_services.py
tests/         unit/ integration/ e2e/
docs/          runbook.md, config_reference.md, open_questions.md, design.md,
               gitlab_setup.md, desktop_checklist.md
data/          (gitignored) ledger/ snapshots/ outputs/ logs/ portfolios/ backups/ exports/
```

## Data folder

`PORTSTUDY_DATA_DIR` (default `./data`) roots every data path. All stored data is CSV
(UTF-8, header row, ISO dates, floats as `%.12g`) plus small JSON manifests. Times are US
Eastern with an explicit offset. Week labels (`2026-W46`) are the ISO week of the cutoff
Friday; scores made at that cutoff trade at the next Monday open and their return window
runs to the following Monday open. Every table that holds returns carries `week_id`,
`trade_date`, and `window_end_date`.
