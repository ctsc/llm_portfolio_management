# Specialized LLM Skills Across Fixed Quantitative Portfolio Optimization Strategies

MS Computer Science research project, Georgia State University. Carter Tierney, advised by Dr. Parag Tamhankar. LLM calls: GSU ARCTIC's LiteLLM gateway. Everything else: Carter's desktop (WSL2).

This repository is the system behind a controlled study of how large language model (LLM) views change the results of different quantitative portfolio optimization strategies. The data, the quant baseline, the Ledoit-Wolf covariance, the way views are built, and the guardrails are held fixed, so only the LLM's ratings and the strategy change. Each LLM rates every stock as a range (low, most likely, high) using a bare prompt, one of four specialized skills files, or a placebo (filler text the same length as a skills file). These ratings become Black-Litterman views that steer five fixed strategies, and each strategy is compared with its own quant-baseline-only version, which ranks stocks on momentum plus a pretrained forecasting model. A weekly evaluation tracks the rank information coefficient (IC) and portfolio returns over a live run to find which strategies the LLM helps or hurts the most, and why. The research proposal and the ARCTIC resource request are in `llm_research/current_documents/`; the research notes, methodology and findings are in `llm_research/`; the build plans are in `plans/`.

## Research questions

*How can LLM views built from specialized skills improve quantitative portfolio optimization strategies, and which strategies do LLMs help or hurt the most, and why?*

- **RQ1** How often do the LLM's ratings alone beat the quant baseline's ratings?
- **RQ2** How often does adding the LLM views improve each strategy's portfolio, compared with the same strategy run on the baseline alone (Q0)?
- **RQ3** Do specialized skills change the result compared with bare prompts and placebo?
- **RQ4** Which strategies do the LLM views help or hurt the most, and in which conditions (up or down markets, high or low volatility, earnings weeks)?
- **RQ5** How much of the result comes from luck or general market movement rather than the LLM?

The proposal ends with a stated hypothesis for each research question. Code, config and the report keep the IDs H1 to H5 (Hn tests RQn).

| ID (RQ) | Test (current design; see `docs/design.md`) |
|---|---|
| H1 | weekly rank IC of each LLM source minus the baseline's, plus residual IC (Fama-MacBeth, controlling for the baseline score, momentum, size, volatility, beta, sector and factor loadings) |
| H2 (ratings) | weekly rank IC of the LLM-steered Black-Litterman tilt minus Q0's tilt, on the same stocks |
| H2 (portfolios) | each steered portfolio's active return vs its strategy's Q0: mean, information ratio, tracking error, Ledoit-Wolf Sharpe test, break-even cost; all weeks and active-view weeks |
| H3 | skills vs bare and vs placebo on rank IC and on active return, pooled across models with model and week fixed effects |
| H3b | narrow minus wide rating ranges, a score-by-confidence interaction, and a width vs error check |
| H4 | strategies ranked by mean active return, strategy-vs-strategy differences, market, volatility and earnings-week splits (exploratory) |
| H5 | daily active return regressed on Fama-French five factors plus momentum; the placebo prompt as the content control |

Four pooled main comparisons (LLM-minus-baseline rank IC gap, steered-minus-Q0 tilt IC gap, skills minus bare, skills minus placebo) are fixed in `config/preregistration.yaml` before the live run and corrected with Holm; every other comparison uses Benjamini-Hochberg. Every metric is computed for every source, portfolio, model, prompt and strategy and saved to `results/eval/all_metrics.csv`, and null results are reported (G22). The shuffled-ratings test of earlier plans is dropped. The code still implements the older evaluation until `plans/PLAN4.md` Phases 1 to 9 are built.

## Study design

**Rating source vs portfolio.** A rating source is one LLM with one prompt type. A portfolio is one strategy run on one source's views (a steered portfolio), on the baseline view alone (that strategy's Q0), or a forecast-free control. Rank IC is a property of the source; returns are a property of the portfolio.

**Prompt types.** `bare`, `placebo`, and the skills files `skills_1` to `skills_4` (`skills_5` is disabled by default; `config/prompts_manifest.yaml`). Every prompt is built from the same fields (ticker, name, sector, the week's news up to the cutoff) at temperature 0, one stock per call, JSON output only. Each skills file must be within 5 percent of the mean skills length and the placebo within 2 percent, so length cannot explain a difference between them.

**Models** (`config/models.yaml`). Gemma 4 31B and MiniMax M2.7 through the ARCTIC LiteLLM gateway (the two models ARCTIC already hosts), and Claude Haiku 5.5 and GPT-6 Luna through their providers' batch APIs. Models banned by GSU's AI tool registry are not used.

**Portfolio count.** With S sources, the pipeline writes 5 (S + 1) + 2 portfolios a week: five strategies, each on S sources plus its own Q0, plus HRP and equal weight. The default 4 models x 6 prompt types give 24 sources and 127 portfolios; with 7 prompt types, 28 sources and 147 portfolios.

**Output schema** (DECISION(C3)). The model returns JSON with integer `low`, `likely` and `high` ratings in [-5, 5], `low <= likely <= high`, and a short `rationale`. Anything else is invalid; one retry is allowed.

**Controls.** Each strategy's Q0 (the same strategy with only the baseline view), HRP and equal weight (no forecasts at all), the placebo prompt, residualization of the LLM score against the baseline, and common random numbers: every portfolio in a week uses the same seeded scenario and resampling draws, so two portfolios differ only through their views.

**Fixed settings.** Every study setting is fixed in `config/study.yaml`, tagged `DECISION(<id>)` in code, and frozen with the `frozen-v4` tag before the live run. Code never changes a study setting.

**Live run only.** A one-week pilot runs first, then the live run on real weekly data. There is no backtest, because the LLMs may already have seen news and prices from any past period. The universe is 100 S&P 500 stocks balanced across the eleven sectors (`universe.size`, decision B4), the size ARCTIC's startup allocation of 1,000 GPU-hours fits.

## Methodology

`docs/design.md` walks through every stage and `plans/PLAN2.md` holds the formula sheet. For the N stocks in the week's universe:

**Quant baseline.** Momentum features (`M12_1 = P[t-21]/P[t-252] - 1`, `M3 = P[t]/P[t-63] - 1`, one-month reversal `R1 = -(P[t]/P[t-21] - 1)`) are rank-averaged into `m`. Kronos, a pretrained candlestick foundation model, samples 20 five-day price paths per stock on the desktop GPU; their mean gives the forecast `k` and their standard deviation `s` the forecast spread. The baseline score is `b = 0.5 u(m) + 0.5 u(k)` with `u(x) = rank(x)/N`, standardized to `z_b`.

**Risk model and prior.** The Ledoit-Wolf shrinkage covariance of 252 daily returns, scaled by 5 to weekly (`Sigma`), and the market-cap equilibrium prior `pi = delta Sigma w_mkt` with `delta = 2.5`.

**Views.** The LLM `likely` rating is regressed on `z_b`; the standardized residual `z_L` is the LLM view signal. Both views are absolute views on every stock (`P = I`):

```
q_b = pi + IC_b sigma z_b          Omega_b,ii = tau Sigma_ii clip(s_i / median(s), 0.25, 4)       (K1)
q_L = mu_Q0 + IC_L sigma z_L       Omega_L,ii = tau Sigma_ii / c_i,  c_i = clip(1 - width_i/10, 0.1, 1)   (E1)
```

The LLM view is centered on the week's Q0 posterior mean `mu_Q0` (decision E7), so a rating of 0 changes no expected return and leaves MVO and Top-K equal to Q0. For Robust and Resampled MVO, whose weights also react to the lower uncertainty a view brings, confidence-only control portfolios (`<strategy>__CONF__<model>__<prompt>`, decision G19) use Q0's mean with the source's view uncertainty, separating a view's information from its added confidence. Every steered portfolio's meta records `expected_te_vs_q0` and `conf_only_uncertainty_ratio` (G18).

with `tau = 0.05`. The view strengths `IC_b` and `IC_L` are fixed at 0.02 every week (`views.ic_mode: fixed`, decision A8), so no view is switched off by a short noisy IC history; J8 still scores every weekly rank IC, and `ic_mode: trailing` keeps the older trailing-IC rule for tests. A Kronos spread above the median widens the baseline's uncertainty on that stock; a wide rating range (`width = high - low`) widens the LLM's. If `IC_L = 0`, or a source has no valid ratings, the LLM view rows are dropped and each portfolio trades exactly its strategy's Q0 weights.

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

The evaluation uses only saved ratings, weights and returns, so it adds no compute (decision G12 and the proposal's Evaluation section).

- **Paired weekly gaps.** Mean gap with Newey-West standard errors and Student-t critical values, a block-bootstrap 95% interval, a sign-flip permutation p-value (exact when there are 24 weeks or fewer), win, tie and loss counts, and the minimum detectable effect `2.8 sd / sqrt(T)`.
- **Multiple testing.** Holm on the pre-registered main comparisons; Benjamini-Hochberg on the rest; H4 exploratory.
- **Portfolio metrics.** Annualized Sharpe net of the T-bill rate (with the probabilistic and deflated Sharpe), Sortino, maximum drawdown and CVaR from daily returns, information ratio and tracking error vs Q0, certainty-equivalent return, turnover, break-even cost, stop counts, view shift.
- **Factor regressions.** Kenneth French daily FF5 plus momentum on daily active returns for the final run; the ETF proxies in `universe.etf_proxies` for live monitoring only.

Status: the code currently implements the older version (Holm everywhere, ETF-proxy regressions of total returns, and Sharpe without the risk-free rate). `plans/PLAN4.md` Phases 5 to 9 bring it to the version above.

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

`scripts/run_weekend.py` runs the jobs in order on the desktop and resumes where it stopped. Every job records an inputs hash and an outputs hash in its status file; a finished job is skipped only while its inputs are unchanged and its outputs are intact. Weights are final once `locked.json` exists.

**Where it runs** (decision L1). Carter runs `scripts/run_weekend.py` by hand in WSL2 on his desktop, Friday night or Saturday morning (decision N2). J2 runs Kronos on the local GPU; J4 sends the open-weight requests to ARCTIC's LiteLLM gateway over the GSU VPN; J3 and J5 use the providers' batch APIs; everything else runs on the desktop CPU. Nothing in the repository runs on ARCTIC compute nodes.

## Failure handling

No failure is silent and every fallback is labeled in the data.

- **Source failure (N1).** A source that fails validation or misses the lock trades its strategy's Q0 weights, tagged `fallback_q0` with the reason; that week is excluded from the source's clean-week tests.
- **Model failure.** When one model's J4 worker crashes, the other models finish and keep their records, then J4 fails naming the model so a rerun resumes it from the cache; an authentication error marks that model's sources failed and the week proceeds.
- **Gateway outage (D2).** A health check before J4 marks the failing model's sources failed for the week; they trade Q0 and that week is left out of their statistics. There is no fallback server.
- **Provider errors.** Transport errors, rate limits and server errors are retried by the provider and then requeued twice before the task is dropped.
- **Solver failure.** Each strategy falls back in order to its previous weights, its own Q0, then equal weight, with `fallback_flag` set and `view_shift` measured on the weights actually traded.
- **Missed week.** If J1 or J2 fails, the week is written as `no_rebalance` and locked, so the replay still applies stops, dividends and daily values; the replay also fills any calendar gap between locked weeks.
- **Interrupted run.** If the desktop sleeps, restarts or loses the network, rerun `run_weekend.py` (or `--from` the named job); every job resumes from its status files.
- **History cannot drift.** The simulator reads only the append-only raw ledger. Snapshots are frozen at the cutoff and news after 16:00 ET is dropped.
- **Secrets never reach logs.** The JSON-lines log handler redacts credential-bearing keys, bearer tokens and API keys.

`docs/runbook.md` lists every failure, its symptom in `portstudy status`, and the rerun or restore command.

## Compute

| Work | Where | Measured or estimated time per weekend (300 / 100 stocks) |
|---|---|---|
| J2 Kronos | desktop GPU (RTX 3050, 6 GB) | 56 min / 3 min measured 2026-11-10 (300 stocks likely slowed by GPU memory spill; about 10 min expected with chunking, `baseline.kronos.batch_size`; to be measured on the desktop) |
| J4 gateway ratings, 2 models | ARCTIC LiteLLM gateway | about 4 to 9 h / proportionally less (estimate: 30 to 60 s per request, 4 at a time per model; MiniMax reasoning could add to it; to be measured, `manualwork.md` E) |
| J3 / J5 API models | Anthropic and OpenAI batch endpoints | providers allow up to 24 h; usually back during J4 |
| J7 all strategies | desktop CPU (Ryzen 5 8400F) | about 63 min at 28 sources (estimated from the measured per-source slope), 112 min at 51 / 20 min at 51 |
| J1, J6, J8, J9 | desktop | minutes |
| Final evaluation | desktop | minutes |

ARCTIC allocation (granted 2026-11-16): the default startup allocation of 1,000 GPU and 10,000 CPU service units, so the study uses 100 stocks. ARCTIC request (revised 2026-11-12): GPU hours for the gateway LLM calls only (2,730 for 300 stocks, 910 for 100), no CPU hours or storage. The per-request figures behind it (about 40 V100-equivalent GPU-seconds per Gemma request and 90 per MiniMax request, whose reasoning counts as output) are estimates, and how gateway requests convert into GPU service units is not yet known (`llm_research/09_arctic_process_and_model_facts.md`). Desktop benchmark details: `docs/desktop_setup.md` and `llm_research/04_models_compute_arctic.md`.

## Quick start

Python 3.11 is required.

```bash
python -m venv .venv
source .venv/bin/activate       # Linux or WSL2 (the study runs in WSL2; docs/desktop_setup.md)
pip install -e ".[dev]"
```

Run three synthetic weeks and build the report:

```bash
python scripts/run_weekend.py --synthetic --week 2026-W02
python scripts/run_weekend.py --synthetic --week 2026-W03
python scripts/run_weekend.py --synthetic --week 2026-W04
portstudy status --week 2026-W04 --synthetic
portstudy eval --synthetic
```

Each run ends with a per-portfolio summary (status, weights file, hash). The report lands in `data/outputs/report/report.md` with CSV tables and PNG figures beside it. Optional extras: `.[data]` (alpaca-py, yfinance, finnhub-python, pandas_market_calendars), `.[api]` (anthropic, openai, tiktoken), `.[ml]` (torch, transformers for Kronos).

## Commands

| Command | Purpose |
|---|---|
| `python scripts/run_weekend.py [--week W] [--synthetic] [--from J] [--until J] [--dry-run] [--allow-over-budget]` | The weekly run: catch-up replay and backup, J1 to J6, J7, J9; resumable |
| `portstudy run-job J4 --week W [--force]` | Run or rerun one job |
| `portstudy run-week --week W` | All jobs in order, stopping at the first failure |
| `portstudy status --week W` | State, timestamps, and error of every job |
| `portstudy replay` | Rebuild every portfolio from the ledger and rescore closed weeks |
| `portstudy backup --week W [--dry-run]` | Dated CSV backup folder plus zip |
| `portstudy eval` | Full evaluation report |
| `python scripts/make_synthetic_world.py` | Write the seeded synthetic market to `data/synthetic/` |
| `python scripts/benchmark_pipeline.py [--n-stocks 300 --sources 3 --weeks 24]` | Seconds per solve per strategy, weekly J7 time, peak memory |
| `python scripts/estimate_cost.py` | Projected API cost from logged token counts |
| `python scripts/refresh_manifest.py --check\|--write` | Prompt file hashes; a prompt edited without a version bump fails |
| `python scripts/check_placebo_length.py` | Skills within 5 percent and placebo within 2 percent of the enabled skills mean |
| `python scripts/smoke_real_services.py [--dry-run]` | Tiny real calls per service and enabled model |
| `python scripts/kronos_smoke.py --n-stocks 300` | Kronos forecast time and peak GPU memory on synthetic prices |

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
  eval/                 statistics, rank IC, hypotheses H1 to H5, factors, report
scripts/                run_weekend.py and the utilities above
tests/                  unit/  integration/  e2e/
docs/                   code docs: design, runbook, desktop setup, weekend checklist, config
                        reference, open questions, build decisions, GitHub and GitLab setup
llm_research/           research notes, methodology, findings, references, correspondence;
                        current_documents/ holds the current proposal and ARCTIC request
data/                   (gitignored) ledger/ snapshots/ outputs/ logs/ portfolios/ backups/
plans/                  PLAN.md, PLAN2.md, PLAN3.md, PLAN4.md (the current plan), PROGRESS.md (build
                        log), results.md (the code review that preceded PLAN3)
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

The suite proves, among other things: a hand-calculated simulator example to the cent and exact-threshold stops; Black-Litterman equal to the closed form and portfolios with a dropped LLM view equal to Q0; a labeled fallback or clean rerun for every injected failure (crash and resume, batch never returns, cost cap, invalid-output storm, data outage, late lock, a crashed model worker); the planted synthetic effect recovered by H1 and H3 with the placebo-vs-bare false-positive rate at the nominal level; identical outputs from two runs with the same seed.

GitHub Actions (`.github/workflows/ci.yml`) runs lint, unit tests and the repository checks on every push and the integration and e2e jobs on `master`; `.gitlab-ci.yml` mirrors it for the GitLab fallback host.

## Project status

- **Built and tested.** v1, v2 phases 1 to 14, the review fixes from `plans/results.md`, and PLAN3 phases 1 to 6, each recorded in `plans/PROGRESS.md`.
- **In progress: `plans/PLAN4.md` Phases 1 to 11** (Phases 1 to 3B are done; Phase 3B added the neutral LLM view, the confidence-only controls and the S8 calibration; Phase 3A added open-to-open weekly returns, the large-move and crash-removal rules, the stop and no-price fixes, the study log, event timeline, fallback ledger, audit, ARCTIC usage tracking, canaries and the kill-safe resume; Phase 3 added Kronos chunking, the gateway output and error handling, the consolidated price feed and the live-readiness preflight. Phases 1 and 2: the retired G6 test, the ARCTIC job scripts, the fallback server path and the small-model entries are removed; see `plans/PROGRESS.md`). Still to build: desktop runtime (Kronos chunking, live-readiness checks, lock file), the new evaluation and report, docs recheck, and end-to-end verification.
- **Manual work: `manualwork.md`** (Carter, the only manual list): emails to Parag and Andalib, gateway key and VPN, desktop setup, API keys and budget, skills and placebo texts, stock list, preregistration, smoke tests, the pilot, the `frozen-v4` tag.
- **Not yet verified against live services** (`UNVERIFIED` in config and code): gateway JSON mode, seed support, rate limits and real seconds per request; whether MiniMax needs an output-token override; Kronos per-path sampling and its chunked GPU time.

## Documentation

| File | Contents |
|---|---|
| `llm_research/` | Research notes: overview, decisions, pipeline and methods, compute and ARCTIC, evaluation methodology and findings, dropped methods, literature, correspondence; `current_documents/` has the current proposal and ARCTIC request |
| `plans/PLAN4.md` | The current build plan (Devin) |
| `manualwork.md` | Every manual step Carter has to do, in one file |
| `plans/PLAN.md`, `PLAN2.md`, `PLAN3.md` | Earlier build plans (PLAN2 section 0.3 still holds the unchanged decision values) |
| `plans/PROGRESS.md` | What was built in each phase, tests, measurements, open items |
| `plans/results.md` | The read-only code review: formula proofs, pipeline checks, findings by severity |
| `docs/design.md` | One section per pipeline stage with the exact formulas and the current evaluation design |
| `docs/desktop_setup.md` | WSL2 setup, Kronos, keys, backups, benchmarks, reliability settings |
| `docs/weekend_checklist.md` | Before, during and after each weekend run |
| `docs/runbook.md` | Status, reruns, every failure and its fallback, restore from backup |
| `docs/config_reference.md` | Every config key with default, meaning, and decision id |
| `docs/open_questions.md` | Current and retired decisions |
| `docs/github_setup.md`, `docs/gitlab_setup.md` | Remotes and CI (GitHub primary, GitLab fallback) |
| `docs/build_decisions.md` | Every build-time choice, newest first |

License: MIT.

## Design choices

The five study design choices, with value, config key, and reason (they match `config/study.yaml`):

- **K1, Kronos spread to baseline uncertainty** (`views.kronos_spread_clip`): Omega_b,ii = tau Sigma_ii clip(s_i / median(s), 0.25, 4). Scaling by the median keeps the average baseline uncertainty at the v1 level, so the spread only shifts trust between stocks. The clip stops one unusual forecast from making a view almost certain or almost ignored.
- **S4, Mean-CVaR lambda = 1.0** (`strategies.cvar.lam`): expected return and tail loss are both weekly returns, so equal weight needs no tuning and is not fit to past returns.
- **S5, Robust MVO kappa = 1.0** (`strategies.robust.kappa`): the worst case is taken over a one standard deviation ellipsoid of the posterior mean, a neutral setting not fit to past returns.
- **G3b, median width split** (`eval.h3b`): splitting at each week's median gives two groups of similar size, so neither group's rank IC is much noisier than the other; weeks with a group under 10 stocks are skipped.
- **P1, skills length tolerance 5%** (`llm.skills_length_tolerance`): keeps the skills files close enough in length that length cannot explain differences between them; the placebo stays within 2% of their mean.

The PLAN4 decisions (`docs/open_questions.md`): everything except LLM calls runs on the desktop (L1); Kronos runs in chunks (K2); a gateway outage fails that model's week (D2); the shuffle test, Slurm walltime and memory rules are retired (G6, W1, M1); and the evaluation uses pre-registered main comparisons, intention-to-treat and active-week reporting, residual IC, risk-free-adjusted and daily downside metrics, French factors and small-sample inference (G7 to G12).

D3, the paid API budget, is the only provisional decision; see `docs/open_questions.md`. Every other build-time choice is in `docs/build_decisions.md`.
