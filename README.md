# LLM-Enhanced Quantitative Portfolio Management

**Research question: do large-language-model stock ratings carry information that a quantitative baseline does not already have, and does that information survive contact with real portfolio-construction rules?**

This repository is the complete, reproducible research system for a master's thesis in quantitative finance. It runs a weekly, pre-registered live study in which several language models rate every stock in a sector-balanced S&P 500 universe under controlled prompt conditions, the ratings are turned into Black-Litterman views alongside a momentum-plus-forecast baseline, and five allocation strategies trade those views under identical guardrails. Everything that touches the result (data snapshots, prompts, views, optimizers, the simulator, the statistics) is versioned, hashed, and testable offline on a seeded synthetic market with a stub model. The research paper written from this study reports and interprets the findings; this README describes the system, the protocol, and the choices behind them.

---

## Table of contents

1. [Research design](#research-design)
2. [Hypotheses and statistical protocol](#hypotheses-and-statistical-protocol)
3. [Methodology](#methodology)
4. [Weekly pipeline](#weekly-pipeline)
5. [Failure handling](#failure-handling)
6. [Compute and services](#compute-and-services)
7. [Reproducibility](#reproducibility)
8. [Quick start](#quick-start)
9. [Commands](#commands)
10. [Repository layout](#repository-layout)
11. [Data conventions](#data-conventions)
12. [Testing and quality gates](#testing-and-quality-gates)
13. [Project status](#project-status)
14. [Documentation](#documentation)
15. [Design choices](#design-choices)

---

## Research design

**Unit of study.** A *rating source* is one (model, prompt type) pair. A *portfolio* is one (strategy, rating source) pair, one strategy's baseline-only portfolio (`<strategy>__Q0`), or one of the two forecast-free controls (`HRP`, `EW`). Rank IC belongs to the rating source; the same ratings give the same IC under every strategy, so the strategy axis isolates allocation effects from information effects.

**Treatments.** Each model rates each stock under seven prompt configurations:

| Prompt type | Purpose |
|---|---|
| `bare` | The task instructions and the stock's data only. |
| `placebo` | `bare` plus a neutral passage length-matched (within 2 percent) to the mean skills file, so any skills effect cannot be explained by prompt length alone. |
| `skills_1` to `skills_5` | Five specialized analyst-skills documents of similar length (within 5 percent of their mean). `skills_5` is disabled by default. |

Every prompt is built from the same fields (ticker, name, sector, the week's news up to the cutoff) at temperature 0, one stock per call, JSON output only. Prompt files are content-hashed in `config/prompts_manifest.yaml`; editing one without bumping its version fails CI.

**Models.** Open-weight models (Gemma 4 31B, MiniMax M2.7, with Gemma 4 E4B, Phi-4 Mini and Llama 3.2 3B as requested deployments) served through the ARCTIC LiteLLM gateway, plus one or two commercial API models (Claude Haiku 5.5, GPT-6 Luna) through their batch endpoints. The default configuration enables 4 models and 6 prompt types: 24 rating sources, and `(24 + 1) x 5 + 2 = 127` portfolios. LLM calls per week are `models x prompt types x stocks`; strategies add portfolios, not calls.

**Output schema.** Each rating is a range, `{"low": int, "likely": int, "high": int, "rationale": str}` with integers in [-5, 5] and `low <= likely <= high`. The `likely` value is the point view; the width `high - low` is the model's own uncertainty and sets how much the optimizer trusts it. A response that fails the schema is retried once and then counted as invalid.

**Controls built into the design.**

- *Baseline-only twin for every strategy.* Each strategy's `Q0` portfolio runs the identical Black-Litterman and optimizer path with the baseline view only, so every steered portfolio has a same-week, same-strategy counterfactual.
- *Residualization.* LLM ratings are regressed on the baseline signal; only the residual enters the model, so the LLM is credited only for what the baseline did not already know.
- *Common random numbers.* Monte Carlo draws (CVaR scenarios, resampled means) are seeded by (seed, week, strategy) and shared by every source, its Q0 and every shuffle, so simulation noise cancels in every comparison.
- *Pre-registration.* All study decisions (model set, universe size, prompt types, schema, uncertainty mappings, strategy parameters, guardrails, test families) are fixed in `config/study.yaml` and `config/models.yaml` before the live run, tagged `DECISION(<id>)` in code, and never changed by code.
- *No backtest.* The study is live and forward-only: ratings are made at the Friday close and traded at the next Monday open, so no model ever sees the outcome it is rated on.

---

## Hypotheses and statistical protocol

Each hypothesis is a weekly paired comparison. Weekly outcomes are counted as win, loss or tie (a gap inside a +-0.01 tie band is a tie), tested with an exact sign-flip permutation test (10,000 permutations) and reported with a paired bootstrap interval (2,000 resamples), and corrected for multiple comparisons with the Holm step-down procedure inside the family shown.

| Hypothesis | Comparison | Holm family |
|---|---|---|
| **H1** information | rank IC of the raw LLM `likely` rating minus rank IC of the baseline score, per source | all sources |
| **H2** incremental value | (a) rank IC of the combined Black-Litterman tilt minus the baseline IC, per source; (b) each steered portfolio's weekly return minus its own strategy's Q0 return on the source's clean weeks | (a) all sources; (b) per strategy |
| **H3** prompt skills | each `skills_k` source vs `bare` and vs `placebo` for the same model | per model |
| **H3b** calibration | per source and week, rank IC of stocks whose range width is at or below the median width minus the IC of the wider half (each group at least 10 stocks) | all sources |
| **H4** strategy ranking (exploratory, no Holm) | strategies ranked by mean weekly return gap vs their own Q0 with bootstrap intervals; splits by market direction, volatility regime and earnings weeks | none |
| **H5** robustness | (a) daily factor regressions of every portfolio on the ETF proxies (SPY, IWM, IWD, IWF, QUAL); (b) a shuffled-ratings test: 500 replays per steered portfolio and per Q0 with the ratings permuted across stocks within each week and the view strength held fixed, giving an empirical null for total return and Sharpe | per strategy |

Rank IC is Spearman's correlation between the rating at the Friday cutoff and the return from the next Monday open to the following Monday open. The report states the minimum detectable weekly gap (`2.8 sd / sqrt(T)`) so that the power of each family is explicit; more strategies add portfolios, not weeks. Controls (HRP, EW) appear in the metrics and factor tables and in no hypothesis family. A week in which a source fell back to Q0 is excluded from that source's clean-week statistics.

---

## Methodology

The formula sheet in `PLAN2.md` is the single source of truth; `docs/design.md` walks through every stage. In brief, for the N stocks in the week's universe:

**Quantitative baseline.** Momentum features (`M12_1 = P[t-21]/P[t-252] - 1`, `M3 = P[t]/P[t-63] - 1`, one-month reversal `R1 = -(P[t]/P[t-21] - 1)`) are rank-averaged into `m`. Kronos, a pretrained time-series foundation model, samples 20 five-day price paths per stock on a GPU; their mean gives the forecast `k` and their standard deviation `s` the forecast dispersion. The baseline score is `b = 0.5 u(m) + 0.5 u(k)` with `u(x) = rank(x)/N`, standardized to `z_b`.

**Risk model and prior.** The Ledoit-Wolf shrinkage covariance of 252 daily returns, scaled by 5 to weekly (`Sigma`), and the market-cap equilibrium prior `pi = delta Sigma w_mkt` with `delta = 2.5`.

**Views.** The LLM `likely` rating is regressed on `z_b`; the standardized residual `z_L` is the LLM view signal. Both views are absolute views on every stock (`P = I`):

```
q_b = pi + IC_b sigma z_b          Omega_b,ii = tau Sigma_ii clip(s_i / median(s), 0.25, 4)
q_L = pi + IC_L sigma z_L          Omega_L,ii = tau Sigma_ii / c_i,   c_i = clip(1 - w_i/10, 0.1, 1)
```

with `tau = 0.05`. The view strengths `IC_b`, `IC_L` are the trailing eight-week mean rank ICs of each signal, floored at zero (a fixed 0.02 during the five-week cold start). A Kronos spread above the median widens the baseline's uncertainty on that stock; a wide rating range widens the LLM's. If `IC_L = 0`, or a source has no valid ratings, the LLM view rows are dropped and each portfolio trades exactly its strategy's Q0 weights.

**Black-Litterman.** With the stacked views `P = [I; I]`, `A = inv(tau Sigma) + P' inv(Omega) P`, the posterior mean `mu_BL = solve(A, inv(tau Sigma) pi + P' inv(Omega) q)` and the posterior covariance of the mean `M = inv(A)`. Black-Litterman runs once per source per week; every strategy reads the same `mu_BL` and `M`.

**Allocation strategies.** All are long-only with a 5 percent position cap and weights summing to one; cvxpy problems are compiled once per week and re-solved with `mu` as a parameter.

| Strategy | Objective |
|---|---|
| MVO | `max w' mu_BL - (delta/2) w' Sigma w` |
| Mean-CVaR | Rockafellar-Uryasev form on 10,000 scenarios `r_s = mu_BL + chol(Sigma + M) z_s`: `max w' mu_BL - lam [zeta + sum_s u_s / ((1-beta) S)]`, `u_s >= -r_s' w - zeta`, `u_s >= 0`, `beta = 0.95`, `lam = 1` |
| Robust MVO | `max w' mu_BL - kappa ||chol(M)' w||_2 - (delta/2) w' Sigma w`, `kappa = 1` (worst case over a one-standard-deviation ellipsoid of the posterior mean) |
| Resampled MVO | mean of 200 MVO solutions at `mu_d = mu_BL + chol(M) z_d` |
| Top-K | the 20 largest `mu_BL` (ties by ticker) at equal weight |
| HRP (control) | hierarchical risk parity on `Sigma` only: correlation distance, single linkage, recursive bisection, then the cap |
| Equal weight (control) | `w_i = 1/N` |

The `view_shift = 0.5 sum_i |w_i - w_Q0,i|` diagnostic records how far each steered portfolio moved from its Q0; the report warns when a strategy's mean shift is below 0.01, because a strategy that ignores its views cannot show an LLM effect.

**Simulator.** Weights are locked by 09:00 ET Monday and executed at the Monday open at the price recorded in an append-only raw ledger. Transaction costs are 10 bps of traded value; a stop-loss at -8 percent or take-profit at +15 percent of average cost is checked at every daily close and executed at the next open; dividends and splits are applied from the provider's corporate-actions feed. Weekly returns, daily values and every trade are stored per portfolio.

---

## Weekly pipeline

```
                 Friday 16:00 ET cutoff                              Monday 09:00 ET lock
                          |                                                   |
  J1 snapshot ──> J2 baseline + risk ──> J3 submit API batches ──> J4 open-weight models
                                                 |                       |
                                          J5 collect batches ──> J6 validate (N1 failure rule)
                                                                         |
                                                              J7 weights (BL, five strategies + Q0 each, controls)
                                                                         |
            next weekend: J8 replay from the price ledger + rank IC ──> J9 dated CSV backup
```

| Job | What it does |
|---|---|
| J1 | Freezes the week's snapshot: universe, 300 days of prices, news up to the cutoff, reference data, rates and earnings dates. Data checks (missing days, stale or non-positive prices, extreme moves, short history, delisting) either forward-fill for the models or drop the stock, as configured. Only provider-originated price rows enter the ledger. |
| J2 | Momentum, Kronos forecast and spread, Ledoit-Wolf covariance, market-cap prior. |
| J3 / J5 | Submit and collect the commercial API models' batch jobs. |
| J4 | Open-weight inference through the gateway with bounded concurrency, a response cache, provider retries and a cost guard. |
| J6 | Validates every source: invalid-output rate above 15 percent, or not finished 30 minutes before the lock, marks the source failed for the week (`fallback_plan.json`). |
| J7 | Residualization, Black-Litterman, every strategy for Q0 and for every source, both controls; writes weights, metadata and `locked.json`. |
| J8 | Replays every portfolio from the ledger for the closed weeks, scores rank IC for each source and appends it to the IC history. |
| J9 | Dated CSV backup, zip and optional rsync to a second location. |

`scripts/run_weekend.py` runs the jobs in order, polls batches with a countdown, and resumes where it stopped. Every job records an inputs hash and an outputs hash in its status file; a finished job is skipped only while its inputs are unchanged and its outputs are intact. Weights are final once `locked.json` exists: later IC updates never re-solve a locked week.

---

## Failure handling

The protocol is written so that no failure is silent and every fallback is labeled in the data.

- **Source failure (N1).** A source that fails validation or misses the lock trades its strategy's Q0 weights, tagged `fallback_q0` with the reason; that week is excluded from the source's clean-week tests.
- **Gateway outage (D2).** A health check before J4 marks the failing models' sources; with `--gateway-fallback`, the fallback-eligible small models are served from the researcher's own Slurm vLLM job instead, one endpoint per source-week, and the serving endpoint is recorded with the scores.
- **Provider errors.** Transport errors, rate limits and server errors are retried by the provider and then requeued twice before the task is dropped; an authentication failure aborts the model for the week rather than burning the remaining calls.
- **Solver failure.** Each strategy falls back in order to its previous weights, its own Q0, then equal weight, with `fallback_flag` set and `view_shift` measured on the weights actually traded.
- **Missed week.** If J1 or J2 fails, the week is written as `no_rebalance` and locked, so the replay still applies stops, dividends and daily values; the replay also fills any calendar gap between locked weeks.
- **History cannot drift.** The simulator reads only the append-only raw ledger, so past fills never change when a provider re-adjusts its data. Snapshots are frozen at the cutoff and news after 16:00 ET is dropped.
- **Secrets never reach logs.** The JSON-lines log handler redacts credential-bearing keys, bearer tokens and API keys.

`docs/runbook.md` lists every failure, its symptom in `portstudy status`, and the rerun or restore command.

---

## Compute and services

| Component | Where it runs | Notes |
|---|---|---|
| Weekly pipeline (J1 to J9), simulator, evaluation | the researcher's machine or an ARCTIC CPU node | pure Python; the same code runs the synthetic rehearsal |
| Open-weight LLM inference | ARCTIC LiteLLM gateway (OpenAI-compatible, bearer key) | `llm/providers/litellm.py`; JSON mode and seed support are confirmed at go-live |
| Commercial API models | Anthropic and OpenAI batch endpoints | `anthropic_batch`, `openai_batch`; cost guard at the D3 budget |
| Kronos forecast | ARCTIC GPU node (`scripts/arctic/kronos.sbatch`) | 20 sampled paths per stock, seeded |
| Gateway fallback | ARCTIC GPU node running vLLM (`llm_fallback.sbatch`) | writes the endpoint file only once every server answers `/v1/models` |
| Shuffled-ratings test (H5) | ARCTIC CPU array (`shuffle_array.sbatch`) | sharded 25 shuffles per task; projected at roughly 16,600 CPU hours at 300 stocks, 4,000 at 100 |
| Market data | Alpaca (prices, news, corporate actions), yfinance (reference), FRED (rates), Finnhub (earnings) | rate-limited, retried, checked on ingest |

`scripts/arctic/submit_weekend.sh` runs the weekend in two network modes: `compute` submits one Slurm job, `login` runs the network-bound jobs on the login node and waits for the Kronos job before J4. `docs/arctic_setup.md` and `docs/runbook.md` section 9 cover both.

---

## Reproducibility

- **Offline twin of the whole study.** `scripts/make_synthetic_world.py` writes a seeded 120-stock market with a planted rating effect that decreases from `skills_1` to `skills_5`, planted halts and a planted delisting; the stub LLM reproduces the full response path (ranges, invalid outputs, rationale). The 30-week end-to-end rehearsal injects a source failure, a solver failure, a lock-deadline cutoff and a gateway outage recovered with `--gateway-fallback`; two runs with the same seed give byte-identical outputs.
- **Deterministic numerics.** Floats are written as `%.12g`; every random stream is seeded from SHA-256 of a named tuple (`(seed, week_id, strategy)` for the strategy draws, `(seed, portfolio, index, week)` for shuffles), so a shard can run any subset and still match a serial run.
- **Hashes everywhere.** Prompt files, snapshots, status inputs and outputs, weight files and the lock file all carry SHA-256 digests; the covariance-of-mean matrix is recorded by its digest in every weights file.
- **Decisions are data.** `config/study.yaml` holds every pre-registered value with its decision id; the report header prints the values in force so a result can never be separated from the settings that produced it.
- **Single script, resumable.** The weekend is one hand-run script with idempotent jobs; there is no scheduler state outside the repository's data directory.

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

Each run ends with a per-portfolio summary (status, weights file, hash). The report lands in `data/outputs/report/report.md` with CSV tables and PNG figures beside it. Drop `--skip-shuffle` to run the shuffled-ratings test (500 shuffles per portfolio by default; lower `eval.shuffle.n_shuffles` to try it quickly). Select another study file with `--config config/study_test.yaml` or `PORTSTUDY_CONFIG=config/study_test.yaml`.

Optional extras: `.[data]` (alpaca-py, yfinance, finnhub-python, pandas_market_calendars), `.[api]` (anthropic, openai, tiktoken), `.[ml]` (torch, transformers for the Kronos forecaster).

---

## Commands

| Command | Purpose |
|---|---|
| `python scripts/run_weekend.py [--week W] [--synthetic] [--dry-run] [--allow-over-budget] [--gateway-fallback] [--config PATH]` | The weekly run: catch-up replay and backup, J1 to J6, batch polling, J7, J9 |
| `portstudy run-job J4 --week W [--force] [--gateway-fallback] [--config PATH]` | Run or rerun open-weight inference |
| `portstudy run-week --week W` | All jobs in order, stopping at the first failure |
| `portstudy status --week W` | State, timestamps, and error of every job |
| `portstudy replay` | Rebuild every portfolio from the ledger and rescore closed weeks |
| `portstudy backup --week W [--dry-run]` | Dated CSV backup folder plus zip |
| `portstudy eval [--skip-shuffle]` | Full evaluation report |
| `portstudy shuffle plan\|run\|collect` | H5 shuffled-ratings test sharded for a Slurm array: list shards, run one, or assemble the results |
| `python scripts/make_synthetic_world.py` | Write the seeded synthetic market to `data/synthetic/` |
| `python scripts/benchmark_pipeline.py [--n-stocks 300 --sources 3 --weeks 52]` | Seconds per solve per strategy, weekly J7 time, projected shuffle CPU hours; `--jobs` times J1 to J9 once |
| `python scripts/estimate_cost.py` | Projected API cost from logged token counts |
| `python scripts/refresh_manifest.py --check\|--write` | Prompt file hashes; a prompt edited without a version bump fails |
| `python scripts/check_placebo_length.py` | Skills within 5 percent and placebo within 2 percent of the enabled skills mean for every tokenizer |
| `python scripts/smoke_real_services.py [--dry-run]` | Tiny real calls per service and enabled model; dry-run prints gateway payloads only |

Add `--synthetic` to any pipeline command to use the synthetic world and stub LLM.

---

## Repository layout

```
config/                 study.yaml, study_test.yaml, models.yaml, prompts_manifest.yaml
prompts/                base_instructions.md, skills/ and placebo/ (placeholders until the live study)
src/portstudy/
  calendar.py           NYSE calendar, cutoff and trade dates, ISO week labels
  config.py             strict pydantic config; unknown keys are errors
  data/                 providers (synthetic, alpaca, yfinance, finnhub, fred, wikipedia),
                        checks, append-only ledger, frozen snapshots, universe builder
  baseline/             momentum, forecasters (stub and Kronos), combine into z_b and k_sd
  risk/                 Ledoit-Wolf covariance, market-cap prior
  llm/                  prompt assembly, validation, cache, cost guard, runner,
                        providers (stub, litellm, anthropic_batch, openai_batch)
  views/                rank IC store and trailing IC, residualization, view construction
  optimize/             Black-Litterman, cvxpy mean-variance, weight files
  sim/                  ledger-driven simulator with stops, trade and daily records
  pipeline/             run context, job status, jobs J1 to J9, fallback policy, typer CLI
  eval/                 statistics, IC tables, hypotheses H1 to H5, factors, shuffle, report
  backup.py             dated backups, verify and restore, master CSV exports
scripts/                run_weekend.py and the utilities listed above
tests/                  unit/  integration/  e2e/
docs/                   runbook, config reference, open questions, design, GitHub and GitLab setup,
                        desktop checklist
data/                   (gitignored) ledger/ snapshots/ outputs/ logs/ portfolios/ backups/ exports/
PLAN2.md                 the current phased build plan and study decisions
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
ruff format --check src tests scripts
mypy src scripts
pytest tests/unit tests/integration      # about 9 minutes
pytest tests/e2e                         # 30-week synthetic rehearsal with injected failures, about 14 minutes
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

GitHub Actions (`.github/workflows/ci.yml`) runs lint, unit tests, and the repository checks on every push and the integration and e2e jobs on the default branch; `.gitlab-ci.yml` mirrors it for the GitLab fallback host (see `docs/github_setup.md` and `docs/gitlab_setup.md`).

---

## Project status

The system is built and rehearsed; the live study starts after the manual steps in v2 phase 15 of `PLAN2.md`.

- **Built and verified.** v1 and v2 phases 1 to 14 are complete, one commit per phase, each recorded in `PROGRESS.md`. A read-only senior review (`results.md`) then checked every formula against the `PLAN2.md` sheet with independent reference implementations (closed-form Black-Litterman, scipy, statsmodels, scikit-learn, a from-scratch HRP) and found no mismatch; the 31 robustness findings it raised were each fixed with a regression test (`tests/unit/test_review_fixes.py`) and logged in the "Review fixes" section of `PROGRESS.md`.
- **Remaining manual work (phase 15, researcher only).** Accounts, keys and spending limits; the ARCTIC allocation, environment file and model weights; the GitLab data and backup folders; smoke tests of every real service and the 300-stock benchmark on ARCTIC; the research content (the five skills files, the placebo passage, the final universe size, the paid-API budget D3); then the freeze, a pilot week, and go.
- **Not yet verified against live services.** Gateway JSON mode, seed support and rate limits; the MiniMax reasoning format; Kronos per-path sampling; ARCTIC partitions and walltimes; the exact shuffle CPU hours. Each is marked `UNVERIFIED` in the config or code until phase 15 confirms it.

```
cp scripts/arctic/arctic.env.example scripts/arctic/arctic.env   # fill every TBD (phase 15 C)
bash scripts/arctic/submit_weekend.sh --dry-run                  # sbatch commands and the walltime check
bash scripts/arctic/submit_weekend.sh                            # the weekend; prints job ids and rerun commands
python scripts/run_weekend.py --from J4 --gateway-fallback       # D2, after sbatch scripts/arctic/llm_fallback.sbatch
```

---

## Documentation

| File | Contents |
|---|---|
| `PLAN2.md` | Current phased build plan and study decisions |
| `PROGRESS.md` | What was built in each phase, tests, open items, and every review fix |
| `results.md` | The senior code review: mathematical proofs, pipeline and connection checks, findings by severity |
| `docs/design.md` | One section per pipeline stage with the exact formulas |
| `docs/config_reference.md` | Every config key with default, meaning, and decision id |
| `docs/open_questions.md` | Decision table with config locations and code tags |
| `docs/runbook.md` | Status, reruns, every failure and its fallback, restore from backup, the ARCTIC weekend |
| `docs/arctic_setup.md` | First-person ARCTIC setup list: env file, venv, model weights, licenses, network, quotas |
| `docs/arctic_gpu_request.md` | The ARCTIC allocation request (phase 15 D decides its final form) |
| `docs/github_setup.md` | Remotes, the Actions workflow, confirm CI (primary host) |
| `docs/gitlab_setup.md` | The same for GitLab, kept as the fallback host |
| `docs/desktop_checklist.md` | Before, during, and after each weekend run |

License: MIT.

## Design choices

Every non-obvious choice made while building the system is listed here so that the paper can point at each one. The five study design choices, with value, config key, and reason (they match `config/study.yaml`):

- **K1, Kronos spread to baseline uncertainty** (`views.kronos_spread_clip`): Omega_b,ii = tau Sigma_ii clip(s_i / median(s), 0.25, 4). Scaling by the median keeps the average baseline uncertainty at the v1 level, so the spread only shifts trust between stocks. The clip stops one unusual forecast from making a view almost certain or almost ignored.
- **S4, Mean-CVaR lambda = 1.0** (`strategies.cvar.lam`): expected return and tail loss are both weekly returns, so equal weight needs no tuning and is not fit to past returns.
- **S5, Robust MVO kappa = 1.0** (`strategies.robust.kappa`): the worst case is taken over a one standard deviation ellipsoid of the posterior mean, a neutral setting not fit to past returns.
- **G3b, median width split** (`eval.h3b`): splitting at each week's median gives two groups of similar size, so neither group's rank IC is much noisier than the other; weeks with a group under 10 stocks are skipped.
- **P1, skills length tolerance 5%** (`llm.skills_length_tolerance`): keeps the skills files close enough in length that length cannot explain differences between them; the placebo stays within 2% of their mean.

D3, the paid API budget, is the only provisional decision; see `docs/open_questions.md`. Every other entry below is a build-time choice, newest first, tagged "chosen during build" when the plan left it open.

- **v2 phase 14 e2e eval without the in-process shuffle (chosen during build)**: the v1 rehearsal test ran `run_eval` with 2 shuffles over every portfolio, 10.7 minutes on its own with 77 portfolios; it now skips the shuffle, which `test_twenty_shuffles_in_four_shards` and the phase 11 integration tests cover, keeping e2e under the 15-minute bound without shrinking the config.
- **v2 phase 13 one shared rehearsal run (chosen during build)**: the e2e fixture `e2e_root` is the 30-week rehearsal itself, with the injected failures on fixed week indexes (`tests/e2e/conftest.py::INJECTIONS`: 3 source failure, 6 solver failure in every optimizing strategy, 9 lock-deadline cutoff, 12 gateway health failure recovered with `--gateway-fallback`), and the planted delisting from the synthetic world. The v1 invariants skip those four weeks; a second 30-week run would have doubled CI time.
- **v2 phase 13 solver failure by configuration (chosen during build)**: the failure week points every solver list at `NO_SUCH_SOLVER` through the study config for that week only, so each strategy's own failure path runs (previous weights, `fallback_flag = solver_failed`) without patching the optimizers.
- **v2 phase 13 lock cutoff by a file-counting clock (chosen during build)**: the rehearsal clock passes the lock deadline once J7 has written `LOCK_CUTOFF_AFTER_FILES` weight files, so completed portfolios keep their weights and the rest get Q0 with reason `lock_deadline`, deterministically.
- **v2 phase 13 D2 rerun through the inputs hash (chosen during build)**: `--gateway-fallback` changes the run registry (eligible models move to `slurm_fallback`), which is part of J4's inputs hash, so `run_weekend.py --from J4 --gateway-fallback` reruns J4 without `--force`; J6 always reruns and J7 follows the new scores.
- **v2 phase 13 CI split (plan)**: unit tests on every push; integration and e2e on the default branch; `slow` benchmarks manual.
- **v2 phase 12 health check marks sources, J4 skips them (chosen during build)**: a failed gateway health check without `--gateway-fallback` writes a failed `run_status.json` entry per source of the failing model and `gateway_health.json`; J4 skips those models so the run continues and J6 serves the strategy's Q0 (N1). With the flag J4 runs every model (eligible ones on the fallback endpoint).
- **v2 phase 12 `--until` and `--check-walltime` on `run_weekend.py` (chosen during build)**: login mode needs J3 to J5 on the login node and the plan asks for a walltime warning before submitting; both live in the script so the sbatch wrappers stay thin.
- **v2 phase 12 sbatch flags from `submit_weekend.sh` (chosen during build)**: account, partition, GRES and walltime are passed as `sbatch` flags from `arctic.env`, so no `#SBATCH` line holds a `TBD`; `common.sh` refuses any `TBD` value.
- **v2 phase 12 rsync never raises (chosen during build)**: the local backup is complete before rsync runs, so a transfer failure is printed and logged and J9 still succeeds.
- **v2 phase 11 per-week shuffle RNG (chosen during build)**: each (shuffle index, week) seeds its own RNG from SHA-256 of (seed, portfolio, index, week), so a shard can run any subset of weeks or indices and still match a serial run.
- **v2 phase 11 report runs missing shards in-process (chosen during build)**: `shuffle_test` runs missing shards itself unless `require_complete` is set; the `collect` command sets it and fails listing every missing range.
- **v2 phase 11 shard files under `data/shuffle/` (chosen during build)**: one JSON per shard with a `complete` marker and the per-index metrics, keeping the file count at shuffles / 25 per portfolio.
- **v2 phase 10 H4 splits per strategy (chosen during build)**: the v1 market direction and volatility splits run both on the source IC gaps (as in v1) and on each strategy's weekly return gap vs its Q0; the earnings split stays on source IC because it is defined per stock rating.
- **v2 phase 10 stock-week table per source (chosen during build)**: `EvalData.stock_week` holds `likely:<source>` and `width:<source>` columns once per rating source rather than once per portfolio, since five strategies share each source's ratings and H3b needs the width.
- **v2 phase 10 fallback counts in the header (chosen during build)**: "fallback weeks" counts weeks in which at least one source fell back; "fallback-served" counts steered portfolio-weeks that traded a Q0 copy.
- **v2 phase 10 header values (chosen during build)**: the report prints the K1, S4, S5, G3b, and P1 values in force from the config objects beside the `decisions` text, and says "not listed" when a config has no decisions block (the test config).
- **v2 phase 9 compiled-problem cache (chosen during build)**: cvxpy problems are keyed by their constant data (tickers, SHA-256 of Sigma, and for robust and CVaR the SHA-256 of M plus week and seed) with `mu` as a parameter, so resampled draws, every source in a week, and the shuffle test reuse one canonicalization; `warm_start=False` keeps cached and fresh solves identical.
- **v2 phase 9 benchmark on synthetic matrices (chosen during build)**: `scripts/benchmark_pipeline.py` times the optimizers on a seeded 3-factor covariance instead of a full pipeline week, so the 300-stock numbers isolate solver cost; the v1 job timing stays behind `--jobs`.
- **v2 phase 9 shuffle projection (chosen during build)**: CPU hours = weeks x shuffles x (projected sources x strategies) x mean strategy solve time, with 51 projected sources (7 models x 7 prompt types + Q0) as PLAN2 states; the ARCTIC request of 15,000 CPU hours is flagged when exceeded.
- **v2 phase 9 slow tests (chosen during build)**: pytest `addopts` deselects `-m slow` by default, so the 300-stock benchmark test runs only on request (`pytest -m slow`).
- **DECISION(G1), portfolio guardrails** (`optimizer.max_weight`): all strategies and controls use a 5 percent cap with no short positions, matching the same investability limits.
- **DECISION(S2), forecast-free controls**: HRP and equal weight do not use LLM views or Q0 weights, so they provide strategy-independent comparators.
- **DECISION(S1), enabled strategy package**: run MVO, Mean-CVaR, Robust MVO, Resampled MVO, and Top-K, with HRP and equal weight controls, to compare each planned allocation rule under shared guardrails.
- **DECISION(S3), Mean-CVaR scenarios** (`strategies.cvar.n_scenarios`, `strategies.cvar.beta`): 10,000 posterior scenarios at a 0.95 tail level define a reproducible weekly tail-risk objective.
- **DECISION(S6), Resampled MVO draws** (`strategies.resampled.n_draws`): 200 draws balance Monte Carlo stability with a bounded number of solver runs.
- **DECISION(S7), Top-K** (`strategies.topk.k`): selecting 20 names gives a fixed, interpretable equal-weight comparator.
- **DECISION(S1), strategy-aware portfolio keys**: a strategy, source, Q0, or control has an unambiguous identity, while source-level IC remains shared across strategies.
- **v2 phase 8 shared source posterior (chosen during build)**: compute one posterior and covariance-of-mean per rating source, then reuse it across strategies so strategy comparisons differ only in allocation.
- **Source-keyed fallback plan (chosen during build)**: store run status once per rating source because every strategy consumes the same LLM run.
- **Lock-deadline fallback (chosen during build)**: preserve completed portfolios and copy strategy-specific Q0 files for unfinished portfolios, making each decision traceable to its actual source.
- **v2 phase 8 tests run under `config/study_test.yaml` (chosen during build)**: the shared test fixtures load the test config (2 stub models x 4 prompt types, 2,000 CVaR scenarios, 20 resampled draws) because the full study config gave 129 portfolios per week and a 12-minute fixture setup; tests that need an API model add the two API stubs to `llm.model_ids`.
- **v2 phase 8 J7 tests live in `tests/integration/` (chosen during build)**: `test_v2_phase8_jobs.py` runs J1 to J8 on the shared pipeline fixture, so it sits with the other job-level tests; the pure fallback-plan and hash tests stay in `tests/unit/`.
- **Seeded stub spread (chosen during build)**: use a lognormal with median 1 percent and log sigma 0.5; a separate stream seeded by `stub_seed + 1` preserves the existing `k` samples.
- **Kronos sample adapter (chosen during build)**: use `predict_paths` when exposed and otherwise make repeated `sample_count=1` calls, matching the current upstream documentation that multi-sample outputs are averaged.
- **No valid Kronos paths (chosen during build)**: set `k` and `k_sd` to NaN so baseline combination excludes the ticker instead of inventing a return.
- **Config selectors for models and prompt IDs (chosen during build)**: `llm.model_ids` and `llm.prompt_ids` make the small offline test configuration reproducible without changing the default study.
- **Strategy seeds of 7 (chosen during build)**: reuses the existing Kronos seed for deterministic defaults where PLAN2 does not prescribe strategy-specific seeds.
- **Exclude `PLAN2.md` from the banned-word scan (chosen during build)**: it contains the rules and examples that the scanner is designed to enforce, and remains untouched.
- **DECISION(C3), range output**: the model reports low, likely, and high integer ratings, making uncertainty explicit without adding a separate confidence field.
- **DECISION(E1), width-based LLM view uncertainty** (`views.width_conf_clip`): `c_i = clip(1 - width_i / 10, 0.1, 1.0)` and `Omega_L,ii = tau Sigma_ii / c_i`, so wider rating ranges receive less precision.
- **Stub range width (chosen during build)**: half-width follows the absolute sampled noise with a minimum of one, so noisier synthetic ratings have wider ranges.
- **View-layer width conversion (chosen during build)**: calculate `c_i` in `build_views` so validated rating widths directly determine LLM view uncertainty.
- **Empty endpoint metadata (chosen during build)**: keep the output column stable until a provider exposes the resolved serving endpoint.
- **DECISION(B1), prompt configurations**: compare `bare`, `placebo`, and five named skills prompts, with `skills_5` disabled by default, so skills sources can be added without changing discovery code.
- **Equal-length neutral skills placeholders (chosen during build)**: use equal-length text with distinct configuration labels so each placeholder has its own prompt hash until the researcher replaces them.
- **Stub noise progression (chosen during build)**: use `3.0` for both controls and `1.2`, `1.0`, `0.8`, `0.6`, and `0.4` for `skills_1` to `skills_5` to create distinct synthetic skill effects below the controls.
- **DECISION(D1), open-weight registry**: enable Gemma 4 31B and MiniMax M2.7 through LiteLLM, keep requested unprovisioned models disabled, and retain the two API models so the registry matches the study plan.
- **DECISION(D2), fallback serving**: use the gateway by default and switch eligible sources to the configured Slurm endpoint only when explicitly requested, preserving one endpoint per source-week.
- **DECISION(D3), gateway pricing**: gateway calls have zero listed price, while API prices continue through the existing cost guard.
- **DECISION(N1), open-weight timing**: use bounded workers with a default concurrency of four and stop starting provider requests at the lock margin.
- **v2 phase 7 replay test scope (chosen during build)**: reserve HRP and EW weight-file and transaction-cost assertions for the v2 phase 8 J7 integration, because v2 phase 7 defines their registry while v2 phase 8 adds them to weekly replay.
- **Full git history in CI for the phase checklist**: set `GIT_DEPTH: "0"` on GitLab and `fetch-depth: 0` on GitHub so the one-commit-per-phase check can see the entire committed phase history.
- **Common random number seed encoding (chosen during build)**: derive the NumPy generator seed from the first eight bytes of SHA-256 over the JSON tuple `(seed, week_id, strategy)`, yielding repeatable draws shared across portfolio sources.
- **Resampled weight dispersion (chosen during build)**: report the mean per-ticker standard deviation across successful draw weights, providing one scale-consistent diagnostic.
- **HRP cluster variance (chosen during build)**: use inverse-variance allocations inside each cluster when computing cluster variance, following standard HRP bisection without using expected returns.
- **CVaR diagnostic tail average (chosen during build)**: use fractional mass at the boundary of the worst 5 percent of losses, matching empirical expected shortfall when the scenario count does not divide evenly.
- **HRP audit diagnostics (chosen during build)**: retain linkage order and pre-cap weights so the cluster allocation can be inspected independently of the shared cap.
- **Deterministic portfolio metadata (chosen during build)**: label non-solver strategies and controls as `deterministic` and use NaN for their unused objective.
- **Cholesky retry jitter (chosen during build)**: retry once with diagonal jitter `1e-12 * max(trace(M) / N, 1)` to recover only near-singular covariance inputs while keeping the adjustment bounded and deterministic.
- **Synthetic cutoff clock (chosen during build)**: derive a synthetic runner's default time from its source week so historical synthetic status is date-independent; live provider calls retain the wall-clock J6 cutoff.
- **Stable stub randomness across renamed open-weight IDs (chosen during build)**: preserve the previous synthetic output stream when `stub-local-*` IDs become `stub-ow-*`, avoiding unrelated fixture drift.
- **Stub concurrency (chosen during build)**: set test-only open-weight stubs to one worker for deterministic failure injection; the gateway models use four workers and the concurrency tests exercise four workers directly.
- **Seed and JSON mode defaults**: send both optional fields by default and disable a rejected field for the rest of the run; actual ARCTIC support is unverified.
- **ARCTIC endpoint placeholder (chosen during build)**: keep the configured gateway URL empty until the provisioned endpoint is supplied rather than inventing a hostname.
- **Fallback endpoint metadata (chosen during build)**: read `base_url` or `url` from the configured JSON file, accept an optional week ID, and reject files older than the source run start.
- **Reuse `httpx` and `tenacity` (chosen during build)**: both are already project dependencies, so v2 phase 4 adds no package dependency.
