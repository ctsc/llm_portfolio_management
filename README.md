# portstudy

**Do LLM "skills files" add anything to a quant stock-picking baseline?**
A weekly, fully reproducible research pipeline that holds the optimizer and risk guardrails fixed and measures whether language-model stock views, with and without a specialized skills file, improve on a momentum-plus-forecast baseline.

Everything in this repository runs offline on a seeded synthetic market with a stub LLM. The real data and model services sit behind the same interfaces and are switched on with keys when the live study starts.

---

## Table of contents

1. [How the study works](#how-the-study-works)
2. [Pipeline at a glance](#pipeline-at-a-glance)
3. [Quick start](#quick-start)
4. [Commands](#commands)
5. [Repository layout](#repository-layout)
6. [Data conventions](#data-conventions)
7. [Testing and quality gates](#testing-and-quality-gates)
8. [Going live](#going-live)
9. [Documentation](#documentation)

---

## How the study works

Each week, for 100 sector-balanced S&P 500 stocks:

| Step | What happens |
|---|---|
| **Quant baseline** | Momentum features (12-1, 3-month, 1-month reversal) plus a 5-day price forecast are rank-averaged into `z_b`. |
| **Risk model** | Ledoit-Wolf covariance on 252 daily returns, scaled to weekly; market-cap prior `pi = delta Sigma w_mkt`. |
| **LLM views** | Every model rates every stock from its ticker, name, sector, and the week's news, under three prompt configurations: `bare`, `skills`, and a length-matched `placebo`. One stock per call, temperature 0, JSON output only. |
| **Residualize** | LLM scores are regressed on `z_b`; the standardized residual `z_L` is the part of the view the baseline did not already know. |
| **Black-Litterman** | Baseline and LLM views enter as stacked identity views, scaled by trailing rank IC and the model's stated confidence. |
| **Optimize** | Long-only mean-variance with a 5 percent cap. `Q0` is the same pipeline with the baseline view only. |
| **Simulate** | Monday-open rebalance at raw ledger prices, 10 bps costs, daily stop-loss (-8%) and take-profit (+15%), dividends and splits handled. |
| **Evaluate** | Weekly rank IC gaps (H1 to H3), pre-registered splits (H4), factor regressions and a shuffled-ratings test (H5), all with Holm correction. |

Thirteen portfolios trade side by side: 4 models x 3 configurations plus `Q0`. A null result is a valid result; the evaluation reports the minimum detectable gap and never tunes toward a positive finding.

---

## Pipeline at a glance

```
                 Friday 16:00 ET cutoff                              Monday 09:00 ET lock
                          |                                                   |
  J1 snapshot ──> J2 baseline + risk ──> J3 submit API batches ──> J4 local models
                                                 |                       |
                                          J5 collect batches ──> J6 validate (N1 failure rule)
                                                                         |
                                                              J7 weights (BL + MV, Q0, fallbacks)
                                                                         |
            next weekend: J8 replay from the price ledger + score rank IC ──> J9 dated CSV backup
```

- **One hand-run script.** `scripts/run_weekend.py` runs the jobs in order, polls batches with a countdown, and resumes where it stopped. No scheduler.
- **Every job is idempotent.** Status files record inputs hashes; a finished job with unchanged inputs is skipped.
- **Labeled fallbacks.** A model that fails the week (invalid-output rate above 15 percent, or not finished by the lock) trades the `Q0` weights, tagged `fallback_q0`, and that week is excluded from its clean-weeks statistics.
- **History cannot drift.** The simulator reads only an append-only raw price ledger, so past fills and stops never change when a provider re-adjusts its data.
- **No look-ahead.** Snapshots are frozen at the cutoff; news after 16:00 ET is dropped strictly; the IC that scales a view only uses weeks whose return window has already closed.

---

## Quick start

Python 3.11 is required.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS or Linux
pip install -e ".[dev]"
```

Run three synthetic weeks and build the report (about 30 seconds total):

```bash
python scripts/run_weekend.py --synthetic --week 2026-W02
python scripts/run_weekend.py --synthetic --week 2026-W03
python scripts/run_weekend.py --synthetic --week 2026-W04
portstudy status --week 2026-W04 --synthetic
portstudy eval --synthetic --skip-shuffle
```

Each run ends with a per-portfolio summary (status, weights file, hash). The report lands in `data/outputs/report/report.md` with CSV tables and PNG figures beside it. Drop `--skip-shuffle` to run the shuffled-ratings test (500 shuffles per portfolio by default; lower `eval.shuffle.n_shuffles` to try it quickly).

Optional extras: `.[data]` (alpaca-py, yfinance, finnhub-python, pandas_market_calendars), `.[api]` (anthropic, openai, tiktoken), `.[ml]` (torch, transformers for the Kronos forecaster). The core imports and tests without them.

---

## Commands

| Command | Purpose |
|---|---|
| `python scripts/run_weekend.py [--week W] [--synthetic] [--dry-run] [--allow-over-budget]` | The weekly run: catch-up replay and backup, J1 to J6, batch polling, J7, J9 |
| `portstudy run-job J4 --week W [--force]` | Run or rerun one job |
| `portstudy run-week --week W` | All jobs in order, stopping at the first failure |
| `portstudy status --week W` | State, timestamps, and error of every job |
| `portstudy replay` | Rebuild every portfolio from the ledger and rescore closed weeks |
| `portstudy backup --week W [--dry-run]` | Dated CSV backup folder plus zip |
| `portstudy eval [--skip-shuffle]` | Full evaluation report |
| `python scripts/make_synthetic_world.py` | Write the seeded synthetic market to `data/synthetic/` |
| `python scripts/benchmark_pipeline.py` | Seconds per job for one 100-stock week |
| `python scripts/estimate_cost.py` | Projected API cost from logged token counts |
| `python scripts/refresh_manifest.py --check\|--write` | Prompt file hashes; a prompt edited without a version bump fails |
| `python scripts/check_placebo_length.py` | Placebo within 2 percent of the skills file for every tokenizer |
| `python scripts/smoke_real_services.py` | One tiny real call per data service and model (needs keys) |

Add `--synthetic` to any pipeline command to use the synthetic world and stub LLM.

---

## Repository layout

```
config/                 study.yaml (every setting and study decision), models.yaml, prompts_manifest.yaml
prompts/                base_instructions.md, skills/ and placebo/ (placeholders until the live study)
src/portstudy/
  calendar.py           NYSE calendar, cutoff and trade dates, ISO week labels
  config.py             strict pydantic config; unknown keys are errors
  data/                 providers (synthetic, alpaca, yfinance, finnhub, fred, wikipedia),
                        checks, append-only ledger, frozen snapshots, universe builder
  baseline/             momentum, forecasters (stub and Kronos), combine into z_b
  risk/                 Ledoit-Wolf covariance, market-cap prior
  llm/                  prompt assembly, validation, cache, cost guard, runner,
                        providers (stub, ollama, anthropic_batch, openai_batch)
  views/                rank IC store and trailing IC, residualization, view construction
  optimize/             Black-Litterman, cvxpy mean-variance, weight files
  sim/                  ledger-driven simulator with stops, trade and daily records
  pipeline/             run context, job status, jobs J1 to J9, fallback policy, typer CLI
  eval/                 statistics, IC tables, hypotheses H1 to H5, factors, shuffle, report
  backup.py             dated backups, verify and restore, master CSV exports
scripts/                run_weekend.py and the utilities listed above
tests/                  unit/  integration/  e2e/
docs/                   runbook, config reference, open questions, design, GitLab setup,
                        desktop checklist
data/                   (gitignored) ledger/ snapshots/ outputs/ logs/ portfolios/ backups/ exports/
PLAN.md                 the phased build plan and study decisions
PROGRESS.md             build log, one entry per phase
```

---

## Data conventions

- `PORTSTUDY_DATA_DIR` (default `./data`) roots every data path.
- All stored data is CSV: UTF-8, header row, ISO dates, floats written as `%.12g` so reruns are byte-identical. Small JSON manifests carry hashes and metadata.
- All times are US Eastern with an explicit offset, for example `2026-11-13T16:00:00-05:00`. Provider UTC timestamps are converted on ingest.
- `week_id` (for example `2026-W46`) is the ISO week of the cutoff Friday. Scores made at that cutoff trade at the next Monday open and their return window runs to the following Monday open. Every table that holds returns stores `week_id`, `trade_date`, and `window_end_date` explicitly.
- Study decisions are config values in `config/study.yaml`, tagged `DECISION(<id>)` in code and mirrored in `docs/open_questions.md`. Code never changes them.

---

## Testing and quality gates

```bash
ruff check src tests scripts
mypy src scripts
pytest tests/unit tests/integration      # about 10 minutes
pytest tests/e2e                         # 25 synthetic weeks end to end, about 7 minutes
python scripts/refresh_manifest.py --check
python scripts/check_placebo_length.py
python scripts/check_banned_words.py
python scripts/check_secrets.py
```

Highlights of what the suite proves:

- A hand-calculated 3-stock, 5-day simulator example to the cent, exact-threshold stops, dividend and split handling, and replay determinism by hash.
- Black-Litterman equals the textbook closed form; weights sum to 1 within 1e-8 and respect the cap; portfolios with a dropped LLM view match `Q0` exactly.
- Failure injection for every job (crash and resume, batch never returns, cost cap, invalid-output storm, data outage, late lock) ends in a labeled fallback or a clean rerun.
- The planted effect in the synthetic world is recovered by H1 and H3, and the placebo-vs-bare false-positive rate stays at the nominal level over 200 seeds.
- Two full runs with the same seed produce identical outputs; no file under `prompts/` is ever written by code.

GitLab CI (`.gitlab-ci.yml`) runs lint, tests, and the repository checks on every push; the e2e job runs on the default branch.

---

## Going live

The live study needs accounts, keys, a desktop GPU, the real skills file, and a frozen stock list. The ordered checklist is Phase 17 of `PLAN.md`; `docs/desktop_checklist.md` covers each weekend; `docs/runbook.md` covers bad Saturdays. Items that cannot be verified offline are marked `UNVERIFIED` in code and listed in `PROGRESS.md`; `scripts/smoke_real_services.py` exercises each one with a real key.

---

## Documentation

| File | Contents |
|---|---|
| `PLAN.md` | Phased build plan, study decisions, formula sheet |
| `PROGRESS.md` | What was built in each phase, tests, open items |
| `docs/design.md` | One section per pipeline stage with the exact formulas |
| `docs/config_reference.md` | Every config key with default, meaning, and decision id |
| `docs/open_questions.md` | Decision table with config locations and code tags |
| `docs/runbook.md` | Status, reruns, every failure and its fallback, restore from backup |
| `docs/gitlab_setup.md` | Create the project, push, confirm CI |
| `docs/desktop_checklist.md` | Before, during, and after each weekend run |

License: MIT.
