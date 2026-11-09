# Specialized LLM Skills Across Fixed Quantitative Portfolio Optimization Strategies

MS Computer Science research project, Georgia State University. Carter Tierney, advised by Dr. Parag Tamhankar. Compute: GSU ARCTIC.

This repository is the system behind a controlled study of how large language model (LLM) views change the results of different quantitative portfolio optimization strategies. The data, the quant baseline, the Ledoit-Wolf covariance, the way views are built, and the guardrails are held fixed, so only the LLM's ratings and the strategy change. Each LLM rates every stock as a range (low, most likely, high) using a bare prompt, one of four to five specialized skills files, or a placebo (filler text the same length as a skills file). These ratings become Black-Litterman views that steer five fixed strategies, and each strategy is compared with its own quant-baseline-only version, which ranks stocks on momentum plus a pretrained forecasting model. A weekly evaluation tracks the rank information coefficient (IC) and portfolio returns over a live run to find which strategies the LLM helps or hurts the most, and why. The research proposal and the ARCTIC resource request are in `docs/proposals/`.

## Research question and hypotheses

*How can LLM views built from specialized skills improve quantitative portfolio optimization strategies, and which strategies do LLMs help or hurt the most, and why?*

- **H1** How often do the LLM's ratings alone beat the quant baseline's ratings?
- **H2** How often does adding the LLM views improve each strategy's portfolio, compared with the same strategy run on the baseline alone (Q0)?
- **H3** Do specialized skills change the result compared with bare prompts and placebo?
- **H4** Which strategies do the LLM views help or hurt the most, and in which conditions (up or down markets, high or low volatility, earnings weeks)?
- **H5** How much of the result comes from luck or general market movement rather than the LLM?

| Hypothesis | Test | Holm family (`eval.holm_families`) |
|---|---|---|
| H1 | weekly rank IC of each LLM source minus the baseline's rank IC | all sources |
| H2 (ratings) | weekly rank IC of the combined score (`bl_tilt`) minus the baseline's | all sources |
| H2 (portfolios) | each steered portfolio's weekly return minus its strategy's Q0 return | per strategy |
| H3 | each skills source's weekly rank IC minus the same model's bare and placebo rank IC | per model |
| H3b | rank IC of narrow-range ratings minus wide-range ratings, split at each source-week's median width | all sources |
| H4 | strategies ranked by their mean H2 gap; gaps compared across market direction, volatility and earnings-week groups | exploratory, no Holm |
| H5 | Fama-French five-factor alpha, plus 500 shuffled-rating replays per MVO, Robust MVO and Top-K portfolio | per strategy |

## Study design

**Rating source vs portfolio.** A rating source is one LLM with one prompt type. A portfolio is one strategy run on one source's views (a steered portfolio), on the baseline view alone (that strategy's Q0), or a forecast-free control. Rank IC is a property of the source; returns are a property of the portfolio.

**Prompt types.** `bare`, `placebo`, and the skills files `skills_1` to `skills_4` (`skills_5` is disabled by default; `config/prompts_manifest.yaml`). Every prompt is built from the same fields (ticker, name, sector, the week's news up to the cutoff) at temperature 0, one stock per call, JSON output only. Each skills file must be within 5 percent of the mean skills length and the placebo within 2 percent, so length cannot explain a difference between them.

**Models** (`config/models.yaml`). Enabled by default: Gemma 4 31B and MiniMax M2.7 through the ARCTIC LiteLLM gateway, and Claude Haiku 5.5 and GPT-6 Luna through their providers' batch APIs. Gemma 4 E4B, Phi-4 Mini (3.8B) and Llama 3.2 3B are registered but disabled until ARCTIC deploys them on the gateway. Models banned by GSU's AI tool registry are not used.

**Portfolio count.** With S sources, the pipeline writes 5 (S + 1) + 2 portfolios a week: five strategies, each on S sources plus its own Q0, plus HRP and equal weight. The default 4 models x 6 prompt types give 24 sources and 127 portfolios; the largest planned configuration (7 models x 7 prompt types) gives 49 sources and 252 portfolios.

**Output schema** (DECISION(C3)). The model returns JSON with integer `low`, `likely` and `high` ratings in [-5, 5], `low <= likely <= high`, and a short `rationale`. Anything else is invalid; one retry is allowed.

**Controls.** Each strategy's Q0 (the same strategy with only the baseline view), HRP and equal weight (no forecasts at all), the placebo prompt, residualization of the LLM score against the baseline, and common random numbers: every portfolio in a week uses the same seeded scenario and resampling draws, so two portfolios differ only through their views.

**Fixed settings.** Every study setting is fixed in `config/study.yaml`, tagged `DECISION(<id>)` in code, and frozen with the `frozen-v2` tag before the live run. Code never changes a study setting.

**Live run only.** A one-week pilot runs first, then the live run on real weekly data. There is no backtest, because the LLMs may already have seen news and prices from any past period. The universe is 100 to 300 S&P 500 stocks balanced across the eleven sectors; the number depends on the ARCTIC allocation (`universe.size`, 100 until confirmed).

## Methodology

`docs/design.md` walks through every stage and `PLAN2.md` holds the formula sheet. For the N stocks in the week's universe:

**Quant baseline.** Momentum features (`M12_1 = P[t-21]/P[t-252] - 1`, `M3 = P[t]/P[t-63] - 1`, one-month reversal `R1 = -(P[t]/P[t-21] - 1)`) are rank-averaged into `m`. Kronos, a pretrained candlestick foundation model, samples 20 five-day price paths per stock on a GPU; their mean gives the forecast `k` and their standard deviation `s` the forecast spread. The baseline score is `b = 0.5 u(m) + 0.5 u(k)` with `u(x) = rank(x)/N`, standardized to `z_b`.

**Risk model and prior.** The Ledoit-Wolf shrinkage covariance of 252 daily returns, scaled by 5 to weekly (`Sigma`), and the market-cap equilibrium prior `pi = delta Sigma w_mkt` with `delta = 2.5`.

**Views.** The LLM `likely` rating is regressed on `z_b`; the standardized residual `z_L` is the LLM view signal. Both views are absolute views on every stock (`P = I`):

```
q_b = pi + IC_b sigma z_b          Omega_b,ii = tau Sigma_ii clip(s_i / median(s), 0.25, 4)       (K1)
q_L = pi + IC_L sigma z_L          Omega_L,ii = tau Sigma_ii / c_i,  c_i = clip(1 - width_i/10, 0.1, 1)   (E1)
```

with `tau = 0.05`. The view strengths `IC_b` and `IC_L` are the trailing eight-week mean rank ICs of each signal, floored at zero (a fixed 0.02 during the five-week cold start). A Kronos spread above the median widens the baseline's uncertainty on that stock; a wide rating range (`width = high - low`) widens the LLM's. If `IC_L = 0`, or a source has no valid ratings, the LLM view rows are dropped and each portfolio trades exactly its strategy's Q0 weights.

**Black-Litterman.** With the stacked views `P = [I; I]`, `A = inv(tau Sigma) + P' inv(Omega) P`, the posterior mean is `mu_BL = solve(A, inv(tau Sigma) pi + P' inv(Omega) q)` and the posterior covariance of the mean is `M = inv(A)`. Black-Litterman runs once per source per week; every strategy reads the same `mu_BL` and `M`.

**Strategies.** All are long-only with a 5 percent position cap and weights summing to one; cvxpy problems are compiled once per week and re-solved with `mu` as a parameter.

| Strategy | Objective |
|---|---|
| MVO | `max w' mu_BL - (delta/2) w' Sigma w` |
| Mean-CVaR | Rockafellar-Uryasev form on 10,000 scenarios `r_s = mu_BL + chol(Sigma + M) z_s`: `max w' mu_BL - lam [zeta + sum_s u_s / ((1-beta) S)]`, `u_s >= -r_s' w - zeta`, `u_s >= 0`, `beta = 0.95`, `lam = 1` |
| Robust MVO | `max w' mu_BL - kappa ||chol(M)' w||_2 - (delta/2) w' Sigma w`, `kappa = 1` (worst case over a one-standard-deviation ellipsoid of the posterior mean) |
| Resampled MVO | mean of 200 MVO solutions at `mu_d = mu_BL + chol(M) z_d` |
| Top-K | the 20 largest `mu_BL` (ties by ticker) at equal weight |
| HRP (control) | hierarchical risk parity on `Sigma` only: correlation distance, single linkage, recursive bisection, then the cap |
| Equal weight (control) | `w_i = 1/N` |

`view_shift = 0.5 sum_i |w_i - w_Q0,i|` records how far each steered portfolio moved from its Q0; the report warns when a strategy's mean shift is below 0.01, because a strategy that ignores its views cannot show an LLM effect.

**Guardrails and simulator** (DECISION(G1)). Weights are locked by 09:00 ET Monday and executed at the Monday open at the price recorded in an append-only raw ledger. Trading costs are 10 basis points of traded value; a stop-loss at -8 percent or take-profit at +15 percent of average cost is checked at every daily close and executed at the next open; dividends and splits are applied from the provider's corporate-actions feed; no shorts, at most 5 percent per stock, weekly rebalance. Weekly returns, daily values and every trade are stored per portfolio.

## Statistics

- **Win, tie, loss.** Each weekly gap counts as a tie when its absolute value is below `eval.tie_band` (0.01); wins and losses are the rest.
- **Tests on each weekly gap series.** The sign test on wins and losses; the exact sign-flip permutation test on the mean gap (10,000 draws; `p_perm` is the primary p-value and the one Holm adjusts within each family); Newey-West standard errors for the mean, correcting for neighboring weeks being related; and a paired bootstrap interval (2,000 resamples) for the mean gap.
- **Minimum detectable gap.** From the observed weekly standard deviation and week count at the 0.05 level; a gap below it is inconclusive, not evidence of no effect, and the report says so.
- **Factor regressions.** The Fama-French five-factor model on daily portfolio returns when `eval.fama_french_file` is set (the live study); the ETF proxies in `universe.etf_proxies` are the offline fallback only.
- **Shuffled-ratings test** (DECISION(G6)). 500 replays per portfolio with the same scores given to random stocks, for the MVO, Robust MVO and Top-K steered portfolios and those three strategies' Q0s. Mean-CVaR and Resampled MVO are excluded: their solves are about 99 percent of the shuffle compute (22 s and 18 s per solve at 300 stocks against 0.08 s, 0.41 s and 0.003 s), and the Q0 comparison and factor regression still cover them. The report prints an explicit exclusion line for each.

## Weekly pipeline

```
                 Friday 16:00 ET cutoff                              Monday 09:00 ET lock
                          |                                                   |
  J1 snapshot ──> J2 baseline + risk        J3 submit API batches ──> J4 open-weight models
                        |                                |                       |
                        |                         J5 collect batches ──> J6 validate (N1 failure rule)
                        |                                                        |
                        └────────────────────────> J7 weights (BL, five strategies + Q0 each, controls)
                                                                                 |
            next weekend: J8 replay from the price ledger + rank IC ──> J9 dated CSV backup
```

| Job | What it does |
|---|---|
| J1 | Freezes the week's snapshot: universe, 300 days of prices, news up to the cutoff, reference data, rates and earnings dates. Data checks (missing days, stale or non-positive prices, extreme moves, short history, delisting) forward-fill for the models or drop the stock, as configured. Only provider-originated price rows enter the ledger. |
| J2 | Momentum, Kronos forecast and spread, Ledoit-Wolf covariance, market-cap prior. |
| J3 / J5 | Submit and collect the API models' batch jobs. |
| J4 | Open-weight inference through the gateway: one worker per model (`llm.parallel_models`), each with its own concurrency pool, a response cache, provider retries and a cost guard. J4 does not wait for J2. |
| J6 | Validates every source: an invalid-output rate above 15 percent, or not finished 30 minutes before the lock, marks the source failed for the week (`fallback_plan.json`). |
| J7 | Waits for J2, J4 (or every open-weight source marked) and J6, then residualization, Black-Litterman, every strategy for Q0 and for every source, both controls; writes weights, metadata and `locked.json`. |
| J8 | Replays every portfolio from the ledger for the closed weeks, scores rank IC for each source and appends it to the IC history. |
| J9 | Dated CSV backup, zip and optional rsync to a second location. |

`scripts/run_weekend.py` runs the jobs in order and resumes where it stopped; `--poll-once` makes one J5 pass and exits 3 while batches are pending. Every job records an inputs hash and an outputs hash in its status file; a finished job is skipped only while its inputs are unchanged and its outputs are intact. Weights are final once `locked.json` exists.

**ARCTIC layout** (`scripts/arctic/`, DECISION(W1)). `submit_weekend.sh` submits three Slurm jobs: `kronos.sbatch` (J1, J2 on a GPU node) and `llm.sbatch` (J3, J4 on CPU cores) at the same time with no dependency between them, then `finish.sbatch` with `--dependency=afterany` on both. The finish job runs J5 once; if batches are still pending it resubmits itself `POLL_INTERVAL_MINUTES` later and exits, otherwise it runs J6 to J9. In `login` network mode J1 and J3 run on the login node first and the LLM job runs J4 only. No job waits idle for anything outside the cluster, and `common.sh` refuses any walltime above 15 hours.

## Failure handling

No failure is silent and every fallback is labeled in the data.

- **Source failure (N1).** A source that fails validation or misses the lock trades its strategy's Q0 weights, tagged `fallback_q0` with the reason; that week is excluded from the source's clean-week tests.
- **Model failure.** When one model's J4 worker crashes, the other models finish and keep their records, then J4 fails naming the model so a rerun resumes it from the cache; an authentication error marks that model's sources failed and the week proceeds.
- **Gateway outage (D2).** A health check before J4 marks the failing models' sources; with `--gateway-fallback`, the fallback-eligible small models are served from Carter's own Slurm vLLM job (`llm_fallback.sbatch`) instead, one endpoint per source-week, recorded with the scores.
- **Provider errors.** Transport errors, rate limits and server errors are retried by the provider and then requeued twice before the task is dropped.
- **Solver failure.** Each strategy falls back in order to its previous weights, its own Q0, then equal weight, with `fallback_flag` set and `view_shift` measured on the weights actually traded.
- **Missed week.** If J1 or J2 fails, the week is written as `no_rebalance` and locked, so the replay still applies stops, dividends and daily values; the replay also fills any calendar gap between locked weeks.
- **Walltime.** Every Slurm job asks for at most 15 hours (target 12); a job that still hits its limit is rerun from the named job (`run_weekend.py --from J4`), because every job resumes from its status files.
- **History cannot drift.** The simulator reads only the append-only raw ledger. Snapshots are frozen at the cutoff and news after 16:00 ET is dropped.
- **Secrets never reach logs.** The JSON-lines log handler redacts credential-bearing keys, bearer tokens and API keys.

`docs/runbook.md` lists every failure, its symptom in `portstudy status`, and the rerun or restore command.

## Compute

| Job | Where | Cores and memory | Expected wall time (100 / 300 stocks) |
|---|---|---|---|
| `kronos.sbatch` (J1, J2) | ARCTIC GPU node | 1 GPU, `--mem=$GPU_MEM` | under 1 h / about 1 h |
| `llm.sbatch` (J3, J4) | ARCTIC CPU node | 2 cores, `--mem-per-cpu=3G` | about 1.9 h / 5.8 h at 40 s per request on the large models |
| `finish.sbatch` (J5 to J9) | ARCTIC CPU node | 1 core, `--mem-per-cpu=3G` | about 0.2 h / 0.7 h per pass |
| `llm_fallback.sbatch` (vLLM, D2 only) | ARCTIC GPU node | 1 GPU, `--mem=$GPU_MEM` | up to `LLM_FALLBACK_WALLTIME` |
| `shuffle_array.sbatch` (one shard) | ARCTIC CPU array | 1 core, `--mem-per-cpu=3G` | 0.2 min / 4.1 min of solves per 25-shuffle shard |
| API models | Anthropic and OpenAI batch endpoints | - | polled by the finish job |
| Market data | Alpaca (prices, news, corporate actions), yfinance (reference), FRED (rates), Finnhub (earnings) | - | inside J1 |

The shuffled-ratings test over 24 weeks, 51 sources and the three G6 strategies projects to 5.6 CPU hours at 100 stocks and 85 CPU hours at 300 (solve time only; `scripts/benchmark_pipeline.py`, with `--replay-seconds` to add a measured replay time). Without the cluster, `portstudy shuffle run-local --processes N` runs the same shards on any machine. The budget table with its basis is in `PROGRESS.md` (v3 phase 3).

## Quick start

Python 3.11 is required.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS or Linux
pip install -e ".[dev]"
```

Run three synthetic weeks and build the report:

```bash
python scripts/run_weekend.py --synthetic --week 2026-W02
python scripts/run_weekend.py --synthetic --week 2026-W03
python scripts/run_weekend.py --synthetic --week 2026-W04
portstudy status --week 2026-W04 --synthetic
portstudy eval --synthetic --skip-shuffle
```

Each run ends with a per-portfolio summary (status, weights file, hash). The report lands in `data/outputs/report/report.md` with CSV tables and PNG figures beside it. Drop `--skip-shuffle` to run the shuffled-ratings test (lower `eval.shuffle.n_shuffles` to try it quickly). Optional extras: `.[data]` (alpaca-py, yfinance, finnhub-python, pandas_market_calendars), `.[api]` (anthropic, openai, tiktoken), `.[ml]` (torch, transformers for Kronos).

## Commands

| Command | Purpose |
|---|---|
| `python scripts/run_weekend.py [--week W] [--synthetic] [--from J] [--until J] [--poll-once] [--dry-run] [--gateway-fallback] [--check-walltime T]` | The weekly run: catch-up replay and backup, J1 to J6, J7, J9; resumable |
| `portstudy run-job J4 --week W [--force] [--gateway-fallback]` | Run or rerun one job |
| `portstudy run-week --week W` | All jobs in order, stopping at the first failure |
| `portstudy status --week W` | State, timestamps, and error of every job |
| `portstudy replay` | Rebuild every portfolio from the ledger and rescore closed weeks |
| `portstudy backup --week W [--dry-run]` | Dated CSV backup folder plus zip |
| `portstudy eval [--skip-shuffle]` | Full evaluation report |
| `portstudy shuffle plan\|run\|run-local\|collect` | H5 shards for a Slurm array (plan prints the estimated minutes per shard), one shard, all shards locally, or assemble the results |
| `python scripts/make_synthetic_world.py` | Write the seeded synthetic market to `data/synthetic/` |
| `python scripts/benchmark_pipeline.py [--n-stocks 300 --sources 3 --weeks 24]` | Seconds per solve per strategy, weekly J7 time, shuffle CPU hours per G6 strategy |
| `python scripts/estimate_cost.py` | Projected API cost from logged token counts |
| `python scripts/refresh_manifest.py --check\|--write` | Prompt file hashes; a prompt edited without a version bump fails |
| `python scripts/check_placebo_length.py` | Skills within 5 percent and placebo within 2 percent of the enabled skills mean |
| `python scripts/smoke_real_services.py [--dry-run]` | Tiny real calls per service and enabled model |
| `bash scripts/arctic/submit_weekend.sh [--week W] [--dry-run]` | The ARCTIC weekend: three Slurm jobs, printed with their ids and rerun commands |

Add `--synthetic` to any pipeline command to use the synthetic world and stub LLM. Set `PORTSTUDY_DATA_DIR` (default `./data`) to move the data root.

## Repository layout

```
config/                 study.yaml, study_test.yaml, models.yaml, prompts_manifest.yaml
prompts/                base_instructions.md, skills/ and placebo/ (placeholders until the live study)
src/portstudy/
  calendar.py           NYSE calendar, cutoff and trade dates, ISO week labels
  config.py             strict pydantic config; unknown keys are errors
  backup.py             dated backups, verify and restore, master CSV exports
  data/                 providers/ (synthetic, alpaca, yfinance, finnhub, fred, wikipedia), checks,
                        append-only ledger, frozen snapshots, universe builder
  baseline/             momentum, forecasters (stub and Kronos), combination into z_b and k_sd
  risk/                 Ledoit-Wolf covariance, market-cap prior
  llm/                  prompt assembly, schema and validation, cache, cost guard, health check,
                        runner, providers/ (stub, litellm, anthropic_batch, openai_batch)
  views/                rank IC store and trailing IC, residualization, view construction
  optimize/             black_litterman.py, mean_variance.py, problem_cache.py,
                        strategies/ (mvo, cvar, robust, resampled, topk, hrp, equal weight, registry)
  sim/                  ledger-driven simulator with stops, trade and daily records
  pipeline/             run context, job status, jobs J1 to J9, fallback policy, typer CLI
  eval/                 statistics, rank IC, hypotheses H1 to H5, factors, shuffle, report
scripts/                run_weekend.py and the utilities above
scripts/arctic/         common.sh, submit_weekend.sh, kronos / llm / finish / llm_fallback /
                        shuffle_array .sbatch, arctic.env.example
tests/                  unit/  integration/  e2e/
docs/                   design, runbook, config reference, open questions, ARCTIC setup, weekend
                        checklist, GitHub and GitLab setup, build decisions, proposals/
data/                   (gitignored) ledger/ snapshots/ outputs/ logs/ portfolios/ shuffle/ backups/
PLAN2.md, PLAN3.md      the phased build plans and study decisions
PROGRESS.md             build log, one entry per phase
results.md              the read-only code review that preceded PLAN3
```

## Testing and quality gates

```bash
ruff check src tests scripts
ruff format --check src tests scripts
mypy src scripts
pytest                                   # unit, integration and e2e (slow benchmarks excluded), about 27 minutes
pytest tests/e2e                         # only the 30-week synthetic rehearsal with injected failures, about 15 to 20 minutes
python scripts/refresh_manifest.py --check
python scripts/check_placebo_length.py
python scripts/check_banned_words.py
python scripts/check_secrets.py
```

The suite proves, among other things: a hand-calculated simulator example to the cent and exact-threshold stops; Black-Litterman equal to the closed form and portfolios with a dropped LLM view equal to Q0; a labeled fallback or clean rerun for every injected failure (crash and resume, batch never returns, cost cap, invalid-output storm, data outage, late lock, a crashed model worker); the planted synthetic effect recovered by H1 and H3 with the placebo-vs-bare false-positive rate at the nominal level; identical outputs from two runs with the same seed; the ARCTIC scripts under shellcheck with the three-job layout checked in both network modes.

GitHub Actions (`.github/workflows/ci.yml`) runs lint, unit tests and the repository checks on every push and the integration and e2e jobs on `master`; `.gitlab-ci.yml` mirrors it for the GitLab fallback host.

## Project status

- **Built and tested.** v1, v2 phases 1 to 14, the review fixes from `results.md`, and PLAN3 phases 1 to 6, each recorded in `PROGRESS.md`.
- **Remaining manual work** (Carter, every step listed in `manualwork.md`): PLAN2 phase 15 (accounts, keys and limits; the ARCTIC allocation, `arctic.env` and model weights; smoke tests of every real service; the skills and placebo texts; the pilot week; the `frozen-v2` tag) and the PLAN2 project completion checklist.
- **Not yet verified against live services** (`UNVERIFIED` in config and code): gateway JSON mode, seed support and rate limits; the MiniMax output format and whether it needs an output-token override; Kronos per-path sampling and its real GPU time; ARCTIC partitions, memory policy and the real per-request LLM time; the shuffle replay time per shuffle.

## Documentation

| File | Contents |
|---|---|
| `docs/proposals/` | The IEEE project proposal and the ARCTIC resource request (PDF and LaTeX) |
| `PLAN2.md`, `PLAN3.md` | The build plans and study decisions (`PLAN3.md` section 0.1 for G6, W1, M1) |
| `PROGRESS.md` | What was built in each phase, tests, measurements, open items |
| `results.md` | The read-only code review: formula proofs, pipeline checks, findings by severity |
| `docs/design.md` | One section per pipeline stage with the exact formulas, plus the ARCTIC job layout |
| `docs/config_reference.md` | Every config key with default, meaning, and decision id |
| `docs/open_questions.md` | Decision table with config locations and code tags |
| `docs/runbook.md` | Status, reruns, every failure and its fallback, restore from backup, the ARCTIC weekend |
| `docs/arctic_setup.md` | ARCTIC setup: env file, venv, model weights, network, W1 and M1 |
| `docs/weekend_checklist.md` | Before, during and after each ARCTIC weekend run |
| `docs/github_setup.md` | Remotes, the Actions workflow (primary host) |
| `docs/gitlab_setup.md` | The same for GitLab, kept as the fallback host |
| `docs/build_decisions.md` | Every build-time choice, newest first |

License: MIT.

## Design choices

The five study design choices, with value, config key, and reason (they match `config/study.yaml`):

- **K1, Kronos spread to baseline uncertainty** (`views.kronos_spread_clip`): Omega_b,ii = tau Sigma_ii clip(s_i / median(s), 0.25, 4). Scaling by the median keeps the average baseline uncertainty at the v1 level, so the spread only shifts trust between stocks. The clip stops one unusual forecast from making a view almost certain or almost ignored.
- **S4, Mean-CVaR lambda = 1.0** (`strategies.cvar.lam`): expected return and tail loss are both weekly returns, so equal weight needs no tuning and is not fit to past returns.
- **S5, Robust MVO kappa = 1.0** (`strategies.robust.kappa`): the worst case is taken over a one standard deviation ellipsoid of the posterior mean, a neutral setting not fit to past returns.
- **G3b, median width split** (`eval.h3b`): splitting at each week's median gives two groups of similar size, so neither group's rank IC is much noisier than the other; weeks with a group under 10 stocks are skipped.
- **P1, skills length tolerance 5%** (`llm.skills_length_tolerance`): keeps the skills files close enough in length that length cannot explain differences between them; the placebo stays within 2% of their mean.

The PLAN3 decisions:

- **G6, shuffle scope** (`eval.shuffle.strategies`): MVO, Robust MVO and Top-K only, because Mean-CVaR and Resampled MVO were about 99 percent of the shuffle compute and the Q0 comparison and factor regression still cover them.
- **W1, walltime** (`MAX_JOB_HOURS` in `arctic.env`): every Slurm job at most 15 hours, target 12, and never idle waiting on outside work, so a stalled API batch cannot burn an allocation.
- **M1, memory** (`--mem-per-cpu=3G`): CPU jobs request 3 GB per core and only the cores they use, so service units are charged for work done.

D3, the paid API budget, is the only provisional decision; see `docs/open_questions.md`. Every other build-time choice is in `docs/build_decisions.md`.
