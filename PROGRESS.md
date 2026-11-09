# PROGRESS.md: build log

Each entry: what was built, tests passing, anything skipped and why.

## Phase 1: Scaffold, config, and utilities

Built:
- `pyproject.toml` with dependency groups `ml`, `api`, `data`, `dev`; core installs without extras.
- `config/study.yaml` (every section and every decision from PLAN.md 0.3) and `config/models.yaml` (model registry with stub models for tests).
- `src/portstudy/config.py`: strict pydantic models, unknown keys rejected, decisions carry `id`, `value`, `status`.
- `paths.py`, `hashing.py`, `io.py` (atomic writes, `%.12g` floats), `logging_setup.py` (JSON lines, ET timestamps), `types.py`, `calendar.py` (built-in NYSE table 2024 to 2027 with early closes, `pandas_market_calendars` when installed).

Tests: 14 unit tests passing (`tests/unit/test_phase1_scaffold.py`). `ruff check`, `mypy src` clean.

Notes:
- Python 3.11 installed via `uv`; the venv lives at `.venv`.
- Holiday table hand-entered from the NYSE schedule; `pandas_market_calendars` (data extra) takes over when installed.

UNVERIFIED items (to be smoke-tested in Phase 17):
- `config/models.yaml`: Ollama tags for `qwen3.5:4b`, `phi4-mini`, `llama3.2:3b`; dated snapshot IDs for `claude-haiku-5.5` and `gpt-6-luna`; all prices are placeholders.

## Phase 2: Synthetic world and data layer

Built:
- `data/providers/base.py`: `PriceProvider`, `NewsProvider`, `ReferenceProvider`, `RateProvider`, `MembersProvider`, `ProviderBundle`, fixed column names.
- `data/providers/synthetic.py`: seeded factor-model world (120 tickers, 400 history days, 40 future weeks) with planted momentum, planted news sentiment effect, split, dividend, delisting, halt, extreme move, zero price, stale price, duplicate rows, short history, no-news weeks, and post-cutoff articles. ETF proxies SPY, IWM, IWD, IWF, QUAL.
- `scripts/make_synthetic_world.py` writes the world and the synthetic universe to `data/synthetic/`.
- Real adapters behind the same interfaces, mock-tested only: `alpaca.py` (bars raw plus adjusted, news), `yfinance_provider.py` (prices, info, fallback sectors), `finnhub.py` (earnings), `fred.py` (DTB3), `wikipedia.py` (S&P 500 constituents). Shared `http.py` with tenacity retries, a token-bucket rate limit from config, and `FixtureHttp` replay. `factory.py` builds the bundle from config.
- `data/checks.py`: missing days, nonpositive price, extreme move (split-aware), duplicate rows, stale price, no news, news after cutoff, short history, stopped trading; each maps to a config rule and is logged.
- `data/ledger.py`: append-only raw price ledger and corporate actions, `ledger_mismatch` warning on differing provider values.
- `data/snapshot.py`: `make_snapshot` (strict cutoff filter, dedupe, token budget, article_count, earnings_in_window, immutable versions, manifest hashes, ledger append), `read_snapshot`.
- `data/universe.py`: sector-balanced top-9-per-sector plus one, `extra_tickers` replacement, `config/universe.csv` writer with a printed sector table.

Tests: 31 unit tests plus `tests/integration/test_snapshots_30_weeks.py` (30 consecutive snapshots, only planted issues, ledger has no duplicate days). `ruff check`, `mypy src` clean.

Fixes made while finishing the phase:
- `.gitignore` used `data/`, which also ignored `src/portstudy/data/`; changed to `/data/`.
- `_forward_fill` broadcast bug on halted stocks (fired on the Good Friday test).
- Removed-stock test now picks a week after the planted delisting instead of a fixed index.

UNVERIFIED items (Phase 17 smoke test):
- `alpaca.py`: `AlpacaPriceProvider` (`/v2/stocks/bars`, adjustment raw and all, `/v1/corporate-actions`), `AlpacaNewsProvider` (`/v1beta1/news`).
- `yfinance_provider.py`: `YFinancePriceProvider`, `YFinanceReferenceProvider` (history, info fields `marketCap`, `sector`, `longName`).
- `finnhub.py`: `FinnhubEarningsProvider` (`/calendar/earnings`).
- `fred.py`: `FredRateProvider` (`/fred/series/observations`, series DTB3).
- `wikipedia.py`: `WikipediaMembersProvider` (table id `constituents`).

## Phase 3: Quant baseline

Built:
- `baseline/common.py`: `complete_tickers` and `common_universe(snapshot)` (the single ticker set shared by the baseline and the risk model), `rank_u`.
- `baseline/momentum.py`: `momentum_features` (M12_1, M3, R1, m) with incomplete-history exclusion and logging.
- `baseline/kronos.py`: `ForecastProvider` interface, seeded `StubForecaster` (drift plus noise, per-ticker seed), `KronosForecaster` wrapper (vendor/kronos, Hugging Face weights, batch predict when present, GPU memory freed), `make_forecaster` from config.
- `baseline/combine.py`: `combine` (b, z_b), `ohlcv_by_ticker` (adjusted OHLCV, DECISION(E4)), `build_baseline` writing `data/outputs/<week>/baseline.csv`, `read_baseline`.
- Shared test fixtures: `session_data_root` builds 30 consecutive synthetic snapshots once per session; `snapshot_data_dir` points a test at them.

Tests: 7 unit tests (hand-computed momentum, ties, z_b standardization, stub seeding, determinism across 30 weeks, planted momentum rank IC positive). `ruff check`, `mypy src` clean.

UNVERIFIED items (Phase 17):
- `baseline/kronos.py`: `KronosForecaster._load` and `forecast_return` (vendored API names `KronosTokenizer`, `Kronos`, `KronosPredictor`, `predict_batch`, argument names `pred_len`, `T`, `top_p`, `sample_count`).

## Phase 4: Risk model and market prior

Built:
- `risk/covariance.py`: `daily_returns`, `ledoit_wolf_covariance` (sklearn LedoitWolf, alpha stored, weekly scale 5, symmetrized, SPD assertion with `min_eigenvalue`), `build_risk_model` writing `sigma_matrix.csv`, `sigma.csv`, `risk_meta.json` per week, `read_risk_model`. Uses `complete_tickers` so the ticker set matches the baseline.
- `risk/prior.py`: `market_weights` (DECISION(E5), study tickers only, renormalized), `market_prior` (`pi = delta Sigma w_mkt`), `build_prior` writing `prior.csv`, `read_prior`.

Tests: 6 unit tests (3-stock toy equals sklearn times 5, N=300 T=252 stays positive definite, SPD guard, weights sum to 1, pi linear in delta, per-week save and reload). `ruff check`, `mypy src` clean.

## Phase 5: Prompt system

Built:
- `prompts/base_instructions.md` (shared instructions plus the exact JSON schema), `prompts/skills/skills_PLACEHOLDER_v0.md`, `prompts/placebo/placebo_PLACEHOLDER_v0.md` (both start with the PLACEHOLDER banner; no investing advice).
- `config/prompts_manifest.yaml`: configurations `bare`, `skills`, `placebo` with `id`, `kind`, `file`, `version`, `sha256`, `hashed_version`, `enabled`, `matches`. New skills files are new entries, no code change.
- `scripts/refresh_manifest.py`: `--check` fails when a file changed without a version bump (hash differs while `hashed_version == version`); `--write` refuses the same case unless `--force` (first authoring); a bump plus `--write` records the new hash.
- `llm/prompts.py`: `load_manifest`, `assemble_prompt` (prefix then stock part, DECISION(C1) fields, DECISION(C2) no-news sentence, `prefix_hash`, `prompt_hash`, `prompt_file_sha256`), `token_count` (registered counters, tiktoken for OpenAI-like ids, len/4 fallback).
- `scripts/check_placebo_length.py`: placebo within 2 percent of its skills file for every enabled model's tokenizer; skills files under the C4 cap. Passes for the placeholders under len/4, tiktoken, and a word-count tokenizer.

Tests: 7 unit tests (prefix byte-identical across stocks, hash changes with the file, no-news sentence, prefix-then-stock order, unversioned edit fails the manifest check and a bump passes, repo manifest current, prompts assemble for every stock and configuration with no price statistics). `ruff check`, `mypy src` clean.

## Phase 6: LLM runner

Built:
- `llm/schema.py` (fallback constants, `LlmResponse` re-export), `llm/validate.py` (`parse_response`: fences stripped, integer score in [-5, 5], confidence in [0, 1], rationale truncated to the cap, `invalid_truncated` when the answer hit the output limit).
- `llm/providers/base.py`: `LlmProvider`, `BatchItem`, `RawResult`, `BatchHandle`, `BatchStatus`, stable custom ids `model__config__week__ticker`. `complete` takes the `BatchItem` (which carries the `AssembledPrompt`) so the stub can read the ticker and week from it.
- `llm/providers/stub.py`: deterministic stub reading the planted sentiment words from the prompt; per-configuration noise (skills better than bare, placebo equal to bare); failure injection (invalid, truncated, timeout, batch never returns, crash after N calls, always-invalid tickers).
- `llm/providers/transport.py` (injectable JSON transport with backoff), `ollama.py`, `anthropic_batch.py`, `openai_batch.py`: no tools, no web search, temperature 0, shared prefix first (Anthropic `cache_control` system block; OpenAI automatic prefix caching), `reasoning_effort` for OpenAI, `think: false` for Ollama. Mock-tested on hand-written response shapes.
- `llm/providers/factory.py`: `make_provider` from a `ModelSpec`.
- `llm/cache.py`: `cache_key(model_version, prompt_hash, ticker, week_id, settings, model_id)` and an on-disk `LlmCache`; hits are logged and cost nothing.
- `llm/cost.py`: `CostTable` from models.yaml, `estimate_week_cost`, `check_budget` (projection = spent so far plus this week's estimate times weeks remaining to `timing.live_end_date`; raises `CostCapExceeded` unless `--allow-over-budget`, logged), `RunningCost` per-week `cost_log.csv`.
- `llm/runner.py`: `WeekRunner.plan_week`, `run_local` (resumable via per-call jsonl records, cache first, retry invalid once, provider-error retries), `submit_api` (cache hits answered locally, cost guard before submission, handles persisted and reused), `poll_api`, `collect_api` (resubmits failed items up to `batch_retry_count`), `write_scores` (`llm_scores.csv` rebuilt from records), `assess_run` (DECISION(N1): invalid rate above threshold, local not finished by lock, batch not ready by lock) writing `run_status.json`; `lock_deadline`. Model version changes mid-run are logged as `model_version_changed`.
- `scripts/estimate_cost.py`: projects study cost per model and configuration from logged token counts, with and without caching.
- Test fixture `work_data_dir`: a private data dir seeded with the first four session snapshots.
- `config/models.yaml`: stub models now carry distinct snapshot ids.

Tests: 25 unit tests covering every item in the Phase 6 test list plus the full synthetic week for all four stub models. `ruff check`, `mypy src` clean.

UNVERIFIED items (Phase 17):
- `llm/providers/ollama.py`: `OllamaProvider.complete` (`/api/chat`, `think`, `format: json`, `options.num_ctx`).
- `llm/providers/anthropic_batch.py`: `submit_batch`, `poll_batch`, `collect_batch` (Message Batches endpoints, `cache_control`, usage field names).
- `llm/providers/openai_batch.py`: `submit_batch`, `poll_batch`, `collect_batch` (`/v1/files`, `/v1/batches`, `reasoning_effort` value for GPT-6 Luna, `max_completion_tokens`).
- Adapters use the documented REST shapes over httpx rather than the vendor SDKs so the core imports without the `api` extra; swapping to the SDKs is a transport-level change.

## Phase 7: Rank IC, residualization, and views

Built:
- `views/ic.py`: `weekly_returns` (Monday open to Monday open from raw ledger prices, dividends added back, splits rescaled), `rank_ic` (Spearman with average ranks, NaN under 10 stocks or zero variance), `ICStore` (CSV `week_id, trade_date, window_end_date, signal, model, config_id, rank_ic, n_stocks`, idempotent upsert), `trailing_ic` (DECISION(A6) cold start for the first 5 live weeks, only weeks whose window closed before the cutoff, fallback weeks skipped, last 8 weeks, at least 3 clean weeks else cold-start value, floored at `ic_floor`).
- `views/residualize.py`: `valid_scores` (drops invalid outputs so a fallback 0 is never residualized), `residualize` (OLS on z_b with intercept, standardized ddof=1, zeros when the residual std is under 1e-9).
- `views/build.py`: `build_views` (q_b, q_L, Omega_b, Omega_L; per-stock rule keeps baseline-only rows for invalid stocks; zero-IC rule drops the LLM view and logs `llm_view_dropped`; zero IC_b drops the baseline view), `views_frame` for saving.

Tests: 10 unit tests (Spearman equals scipy, opens plus dividends plus splits, store idempotence, cold start, 8-week window, floor, fallback exclusion, look-ahead guard, residual orthogonal to z_b with invalid stocks excluded, zero IC gives pi, per-stock rule, all-invalid case, views build for every portfolio over two synthetic weeks). The weights-equal-Q0 assertions for the zero-IC and all-invalid cases are completed in Phase 8 once the optimizer exists. `ruff check`, `mypy src` clean.

## Phase 8: Black-Litterman and mean-variance

Built:
- `optimize/black_litterman.py`: `black_litterman` with stacked identity views (P = [I; I_L]); Omega is diagonal so the system is solved as (I + tau Sigma D) mu = pi + tau Sigma d with `np.linalg.solve`, no explicit inverse of Sigma. `black_litterman_q0` (P = I, q = q_b, Omega = Omega_b) is used by Q0 and by any portfolio whose LLM view was dropped, so those weights match Q0 exactly. Returns `mu_BL` and the tilt `mu_BL - pi`.
- `optimize/mean_variance.py`: cvxpy long-only mean-variance with the 5 percent cap via a Cholesky factor, OSQP then CLARABEL, at least 20 stocks asserted, solver failure returns the previous week's weights with `fallback_flag = "solver_failed"`.
- `optimize/__init__.py`: `WeekInputs` (baseline, prior, risk model, scores for a week), `portfolio_ics` (trailing IC_b and IC_L), `build_weights`, `build_q0_weights`, `read_weights`, `read_weights_meta`; writes `data/outputs/<week>/weights/<portfolio>.csv` (`ticker, weight, mu_bl, tilt, pi, z_b, z_L, confidence, has_llm_view`) plus a meta JSON.
- `pyproject.toml`: mypy skips following cvxpy, scipy, sklearn, statsmodels, matplotlib (partial stubs).

Tests: 7 unit tests (no views gives pi, confident view moves mu toward q_L and more so with higher confidence, textbook equivalence with explicit inverses, Q0 equals BL without LLM rows, weights sum to 1 within 1e-8 and sit in [0, 0.05] deterministically with OSQP and CLARABEL agreeing, min-stock guard and solver-failure fallback, and the Phase 7 carry-overs: IC_L = 0 and all-invalid portfolios have weights identical to Q0). `ruff check`, `mypy src` clean.

## Phase 9: Trade simulator and guardrails

Built:
- `sim/records.py`: trade and daily CSV storage under `data/portfolios/<portfolio>/` (`trades.csv`, `daily.csv`), readers, and `records_hash`.
- `sim/simulator.py`: `PriceBook` (raw open and close by date plus corporate actions, built only from the ledger), `PortfolioSim` (cash, positions with share-weighted average cost, pending sales), `simulate_week`, `replay`, `WeekPlan` (weights, decision source, week status). Rules implemented and tested: Monday-open rebalance at raw opens with 10 bps cost paid from cash, sells before buys, proportional scaling of buys when cash is short after costs (documented in the module docstring), fractional shares, daily mark at the close, stop-loss and take-profit checked at every close (relative tolerance 1e-9 so an exact threshold fires), queued sales at the next open with cost, a stop queued before a rebalance collapses into one net trade with `stop_trigger` and a single cost charge, dividends credited on the ex-date, splits rescaling shares and average cost, missing prices carried and flagged with a forced exit at the first available open, `no_rebalance` weeks hold while stops still run.
- `sim/__init__.py`: `ledger_book`, `replay_all` (rebuilds every portfolio from scratch and writes records; returns file hashes).

Tests: 12 unit tests including the hand-calculated 3-stock, 5-day example to the cent (expected values written in the test), exact-threshold stop and take-profit, cost on stop sales, holiday Monday trading Tuesday, fractional shares and target weights, cash shortfall rule, dividend and split, Friday stop before a rebalance, carried price and forced exit, reconciliation (cash equals initial minus buys plus sells minus costs; daily returns compound to the total), replay twice identical by hash, provider rewrite ignored by the ledger, week alignment with hand-set dates, and 30 synthetic weeks for 13 portfolios in well under a minute. `ruff check`, `mypy src` clean.

## Phase 10: Fallback policy and weekly jobs

Built:
- `pipeline/context.py`: `RunContext` (config, registries, synthetic flag, data and LLM provider overrides for tests, injectable clock, `live_start_week`), `default_context`, `synthetic_clock` (Friday 20:00 ET of the week).
- `pipeline/status.py`: `JobStatus` files at `data/outputs/<week>/jobs/<job>.status.json` with `state`, `started_at`, `finished_at`, `inputs_hash`, `outputs_hash`, `error`; `run_job` skips a `done` job with an unchanged inputs hash unless forced, records failures with a traceback tail, and supports a `pending` state for batches not yet returned.
- `pipeline/fallback.py`: `FallbackPlan` (J6 output), `fallback_log.json` (J7 output), `week_status_for`, `fallback_weeks`, `clean_weeks` (DECISION(N1)).
- `pipeline/jobs.py`: J1 snapshot (frozen universe built once), J2 baseline plus covariance plus prior, J3 API submit (cost guard first), J4 local inference (requires J2 `done` via the status file), J5 API collect (pending until every batch returns), J6 validate (`assess_run` against the lock deadline minus `pipeline.lock_margin_minutes`; refuses to decide while runs are incomplete before that margin; `no_rebalance` when J2 is not done), J7 weights (refuses writes after the lock deadline, trailing ICs from the IC store with fallback weeks excluded, Q0 copied for failed portfolios with `decision_source = fallback_q0`, `weights/locked.json` with `locked_at` and file hashes), J8 replay and score (refreshes the ledger through the latest close, replays every portfolio from stored weights, upserts baseline, llm_raw, llm_resid, and combined ICs for every closed window), J9 backup.
- `pipeline/cli.py` (typer): `run-job`, `run-week`, `status`, `replay`, `backup`, `eval`, `jobs`; `--synthetic`, `--week`, `--force`, `--allow-over-budget`, `--now`.
- `scripts/run_weekend.py`: the one hand-run script (DECISION(N2)); step 1 catch-up J8/J9, step 2 J1 to J6 with J5 polling and a countdown to the lock margin, step 3 J7 and J9; Saturday warning; on failure prints the failed job, the fallback that applies, and the exact rerun command; `--week`, `--dry-run`, `--synthetic`, `--now`, `--poll-seconds`; ends with the per-portfolio summary table (status, weights file, hash).
- `backup.py`: first version of `run_backup` (dated folder with checksums); completed in Phase 13.
- `config/study.yaml`: `pipeline.lock_margin_minutes: 30`.
- Runner: `assess_run` takes an explicit deadline; stub batches are rehydrated from the runner's items after a process restart; empty-frame concat warnings removed in the ledger and cost log.

Tests: `tests/integration/test_phase10_pipeline.py` (8 tests): every job alone and idempotent rerun, J4 crash then resume without repeated calls, J5 batch never returns (J6 refuses before the margin, labels the fallback at the margin, J7 trades Q0 with the fallback label, excluded from clean weeks and trailing IC, present in as-traded records), J3 cost cap then `allow_over_budget`, invalid rate above threshold, J1 data failure giving `no_rebalance` for every portfolio, J7 lock refusal, and `run_weekend.py` killed at each of J1, J2, J3, J4, J5, J6, J7, J9 in one sequential pass with every rerun resuming. Done-when: `run_weekend.py --synthetic` for 30 consecutive weeks (see the next entry for timing).

Done-when for Phase 10: `python scripts/run_weekend.py --synthetic --week <w>` for 30 consecutive synthetic weeks (2026-W02 to 2026-W31) completed with exit code 0 for every week; 23 minutes wall clock on the desktop (about 45 s per fresh week, dominated by the 1800 stub calls, the per-call cache files, and the growing backup copy). Profiling and speed-ups are noted under Phase 14.

## Phase 11: Evaluation engine

Built (`src/portstudy/eval/`):
- `tests.py`: `newey_west_mean` (HAC Bartlett via statsmodels, default maxlags floor(4 (T/100)^(2/9)), at least 1), `sign_test` (scipy binomtest), `sign_flip_permutation` (seeded, exact under the null), `holm` (statsmodels, NaN-safe), `min_detectable_gap` (2.8 sd / sqrt(T)), `bootstrap_ci`, `paired_summary` (win/tie/loss with the tie band, mean gap, NW se, bootstrap CI, sign and permutation p). The exact sign-flip permutation p is the primary p-value for Holm because the HAC t-test over-rejects at 20 to 30 weeks (observed 12.5 percent at the nominal 5 percent in the false-positive test); the NW p is reported beside it.
- `rank_ic.py`: weekly IC tables per signal (baseline, llm_raw, llm_resid, combined) with non-clean weeks blanked per portfolio; `long_table`.
- `data.py`: `EvalData` loader (IC tables, weekly as-traded returns from the daily records, trades, week statuses, SPY weekly return and 20-day realized vol at the cutoff, per stock-week table with realized return, earnings flag, and valid scores); the pilot week is excluded.
- `hypotheses.py`: H1 (llm_raw minus baseline), H2 (combined minus baseline; portfolio weekly return minus Q0 with Sharpe, max drawdown, turnover, total cost), H3 (skills vs bare and vs placebo per model, any extra skills configuration joins the family), H4 splits (market direction by SPY weekly return, volatility by median SPY vol, earnings week with the min-group rule and usable-week counts), earnings exposure (descriptive), `portfolio_metrics` as traded and clean only.
- `factors.py`: ETF factor stand-ins (MKT-RF, SMB, HML, RMW; CMA reported as a gap) or the Fama-French daily file when configured; HAC regression per portfolio on daily excess returns.
- `shuffle.py`: shuffled-ratings test for every portfolio plus Q0 (z_b shuffled for Q0), stored IC_L and `use_llm_view` held fixed, per-week inputs loaded once, replay from the ledger, cached per (seed, index) so reruns resume, optional multiprocessing, p = (1 + at or above) / (n + 1), Holm across portfolios.
- `report.py`: CSV and Markdown tables plus PNG figures (weekly IC series, gap bars, win/tie/loss bars, equity curves with Q0, shuffle histograms); as-traded and clean-only results with fallback counts; the minimum detectable gap is printed and gaps below it are called inconclusive. `run_eval` backs `portstudy eval --synthetic`.
- Synthetic world: `NEWS_EFFECT_WEEKLY` raised from 0.004 to 0.010 so the planted LLM-detectable effect is recoverable within 30 weeks (the oracle IC ceiling was about 0.10, below the 30-week minimum detectable gap). Stub noise now 3.0 (bare, placebo) vs 0.8 (skills), and the stub averages the planted word count per article so the signal no longer saturates.

Tests: 10 unit tests (hand calculations for NW with zero lags, sign test, permutation, Holm never below raw and monotone, min detectable gap, paired counts, fallback-week exclusion in IC tables, planted skills edge detected by H1 and H3 after Holm on the real stub over 30 snapshot weeks, placebo-vs-bare false-positive rate at or under 10 percent over 200 seeds) and `tests/integration/test_phase11_report.py` (report builds on the synthetic pipeline run with every table and figure, shuffle cache resumes). `portstudy eval --synthetic` produced the full report on the 30-week run (90 s with 5 shuffles per portfolio).

## Phase 12: GitLab CI

Built:
- `.gitlab-ci.yml`: image `python:3.11`, stages `lint` (ruff, mypy), `test` (pytest over `tests/unit` and `tests/integration` with coverage artifacts), `e2e` (default branch only), `checks` (manifest hash check, placebo length, banned words, secret patterns, yamllint). Pip cache per branch; no secrets.
- `.yamllint` config; `.gitattributes` forcing LF line endings (the Windows editor had produced CRLF in a few files, now renormalized).
- `scripts/check_banned_words.py` (em dash and the banned condition word; `PLAN.md`, which states the rule, and the script itself are excluded) and `scripts/check_secrets.py` (OpenAI, Anthropic, AWS, GitHub, GitLab key shapes, private key blocks, quoted key assignments).
- `docs/gitlab_setup.md`, `docs/desktop_checklist.md`.

Tests: `tests/unit/test_phase12_ci.py` (5 tests): yamllint passes, stages and jobs present with the default-branch rule on e2e, every command in the CI file exists locally, the repository passes both greps, and both greps catch planted problems.

## Phase 13: Backups

Built:
- `backup.py`: `run_backup(week_id)` writes `data/backups/<week_id>_<timestamp>/` with the full ledger, the week's snapshot and output CSVs, `ic_history.csv`, portfolio and trade CSVs, logs, prompt files and manifest, configs, `run_metadata.json` (git commit, Python and package versions), `checksums.json`, and one zip of the folder; never overwrites an earlier backup; prints the folder and zip paths; `--dry-run` prints every file it would copy and writes nothing. `rebuild_master_exports` maintains `data/exports/master/{llm_scores,trades,daily_values,weekly_ic,fallback_log,cost_log}_all.csv` from the weekly files. `verify_backup` recomputes every hash and flags missing or unexpected files; `restore_backup(path, target)` verifies first, then copies the data folder (and optionally prompts and config) back.
- `portstudy backup --week ... [--dry-run]` and J9 call it.

Tests: `tests/integration/test_phase13_backup.py` (4 tests): round trip on the synthetic pipeline run (back up, delete the originals, restore, hashes equal), corrupted and missing files detected by `verify_backup` (and restore refuses), master CSVs equal the concatenation of the weekly files, dry run writes nothing.

## Phase 14: End-to-end rehearsal on synthetic data

Speed-ups found by profiling one week (19 s to about 7 s in process, from about 45 s per fresh week in the Phase 10 run):
- Prompt file text and hashes cached on (path, mtime, size); `repo_root()` cached; `WeekRunner.plan_week` cached per model (it was rebuilt 34 times a week).
- Multithreaded OpenBLAS made every 100-by-100 `np.linalg.solve` take 0.16 s on the desktop (0.1 ms single-threaded). `portstudy/__init__.py` now pins the BLAS pool to one thread (env vars plus `threadpoolctl`).
- Stub default `model_version` follows the registry convention so an injected stub and `make_provider` agree.

Built (`tests/e2e/`):
- `conftest.py`: `WeekendRunner` drives `scripts/run_weekend.py` in process week after week with the shared world, injectable LLM and data providers, and a clock; session fixture `e2e_root` runs 25 synthetic weeks (4 stub models, 3 configurations, 100 stocks).
- `test_full_run_synthetic`: no snapshot row after its cutoff, weights locked before the deadline, sum to 1 within 1e-8 and capped, costs equal 10 bps of trade value, cash identity, every week normal, IC history covers exactly the closed windows, no file under `prompts/` changed, report builds.
- `test_failure_week`: J4 crash then resume (week 2), invalid-output storm (week 3, fallback), batch never returns at the lock margin (week 4, fallback for that model only), data failure (week 5, no_rebalance for all), normal week 6; as-traded records carry the labels and clean weeks exclude them.
- `test_resume_equals_uninterrupted_run`: a crash in J4 then a rerun gives CSVs identical to the uninterrupted run (latency column excluded).
- `test_determinism_two_runs_identical`: a second 6-week run matches the first run's outputs and IC history exactly.
- `test_no_future_data`: guarded providers raise on any price request past the cutoff; the news pull stays inside the documented 12-hour leakage window and the snapshot holds nothing after 16:00 ET.
- `test_cost_log`: every logged cost equals the cost table applied to the logged tokens; weekly cost log totals equal the score totals.
- `test_backups_exist_for_every_week`.
- `scripts/benchmark_pipeline.py`: seconds per job for one synthetic week.

`pytest tests/e2e` passes in 6 min 23 s on the desktop.

## Phase 15: Documentation

Built:
- `README.md`: install, synthetic demo, tests, directory map, data folder conventions.
- `docs/runbook.md`: where files live, `portstudy status`, rerunning one job, a table of every failure with its fallback and remedy, restore from backup, prompt or config changes mid-run.
- `docs/config_reference.md`: every key in `study.yaml`, `models.yaml`, and `prompts_manifest.yaml` with default, meaning, and decision id; environment variables and key names.
- `docs/open_questions.md`: the section 0.3 decision table with config locations and `DECISION(...)` code tags, the provisional items settled in Phase 17, and the rule that any change after the live start is logged and versioned.
- `docs/design.md`: one section per pipeline stage with the formula sheet copied from `PLAN.md`.
- `scripts/smoke_real_services.py` (prepared for Phase 17 B4): one tiny real call per data service and per enabled model, printing status, cost, and latency; loads `.env`.

Done-when: the README demo (three weekend runs, `portstudy status`, `portstudy eval --synthetic --skip-shuffle`) was followed in a fresh Python 3.11 virtual environment created from the README alone; every step succeeded. Note: the system `python` on this desktop is 3.14, so the venv must be created with the 3.11 interpreter (`uv python` install); the README states the 3.11 requirement.

## v2 phase 1: strategy dimension, identity, and config

Built:
- Added frozen rating-source and strategy-aware portfolio identities for steered portfolios, each strategy's Q0, and standalone controls.
- Added strategy config models, Top-K and optimizer validation, registry ID validation, `PORTSTUDY_CONFIG`, and `--config` for context-creating CLI commands and `run_weekend.py`.
- Added `config/study_test.yaml`, source and portfolio key methods in `RunContext`, and strategy-aware keys throughout jobs, optimizers, evaluation readers, and summaries.
- Updated stale README and design, runbook, config-reference, and decision documentation. Added the required `## Design choices` section.
- Adjusted the banned-word checker to skip the supplied `PLAN2.md`, which contains policy examples and remains untouched.

Tests:
- v1 baseline before edits: pytest had 141 passed, 2 failed, and 3 warnings in 319.08 seconds (321 seconds wrapper time). One failure could not find `ruff` because `.venv/bin` was not on PATH; the other found a banned-word example in `PLAN2.md:34`.
- Baseline Ruff check, mypy, manifest check, placebo-length check, secrets check, and yamllint passed. Ruff format check reported 27 files needing formatting. The banned-word script failed on `PLAN2.md`.
- Final full pytest: 168 passed with 3 warnings and exit code 0. The repository's double-quiet invocation did not report elapsed time.
- Final `ruff check src tests scripts` and `mypy src scripts` passed.
- Final secret, banned-word, manifest, placebo-length, and yamllint checks passed.
- `ruff format --check src tests scripts` still reports 27 files would be reformatted, matching the v1 baseline; no unrelated files were formatted.

Changed v1 tests and reasons:
- `tests/unit/test_phase8_optimize.py`: portfolio identity now includes the strategy prefix.
- `tests/integration/test_phase10_pipeline.py`, `tests/integration/test_phase11_report.py`, and `tests/e2e/test_full_run_synthetic.py`: expectations use strategy-specific Q0 and steered keys; the duplicate manual Q0 insertion was removed because `portfolio_keys()` now includes it.

UNVERIFIED:
- No new real-service behavior was added. Existing v1 items requiring real-service smoke tests remain unresolved and are listed in their prior entries.

Skipped:
- Phases 2 through 14 are outside this handoff. No real data, paid APIs, brokers, Slurm, or external services were used.

Choices made while unattended:
- CVaR and resampled strategy seeds are 7, matching the existing Kronos seed for deterministic defaults.
- Added `llm.model_ids` and `llm.prompt_ids` selectors so the test config can select two stub models and a fixed prompt subset without changing study defaults.
- Excluded `PLAN2.md` from the banned-word scan because it contains required examples of the rules being checked; the plan file remains unchanged.

## v2 phase 2: low, likely, high output schema

Built:
- Replaced the scalar response schema with ordered `low`, `likely`, and `high` integer ratings, strict validation, first-object parsing, thinking-block handling, extra-key logging, and fallback metadata.
- Updated score records and downstream readers to store `low`, `likely`, `high`, and `width`. Residualization uses `likely`; shuffle evaluation permutes the range columns together.
- Derived view certainty from width using DECISION(E1), retained the v2 phase 6 view-builder interface, and updated optimizer weight output.
- Updated the deterministic stub so likely ratings use the v1 signal plus noise and range width follows sampled noise. Updated the base prompt and refreshed its manifest hash.
- Updated README, design, and config-reference docs for the range schema and width-derived certainty.
- Updated `scripts/smoke_real_services.py` to report `likely` from the new response type. The script was not run against real services.

Tests and checks:
- Full pytest: 200 passed, 3 warnings in 369.60 seconds. The first invocation omitted `.venv/bin` from PATH and failed `test_every_ci_command_exists`; rerunning with the venv bin directory on PATH passed all tests.
- `ruff check src tests scripts`: passed. The updated smoke script also passed its targeted Ruff check.
- `mypy src scripts`: passed after updating the smoke script response field.
- Manifest, placebo-length, banned-word, secret, and yamllint checks passed. The banned-word and secret checks were rerun after this entry was added.
- Targeted `ruff format --check` reports three modified v1 tests that remain in the known 27-file formatting baseline. The other changed Python files passed; no baseline files were bulk formatted.
- `git diff --check` and LF checks passed after the final edits.

Changed v1 tests and reasons:
- `tests/unit/test_phase6_llm.py`: updated parser fixtures, runner columns, provider payloads, and stub assertions for the range response fields.
- `tests/unit/test_phase7_views.py`: updated residualization and view fixtures to use likely ratings and width, and tested width-derived certainty.
- `tests/unit/test_phase11_eval.py`: changed the stub tuple selection to use the likely rating.

UNVERIFIED:
- No new unverified items were introduced. Real-provider behavior remains unverified because this phase was kept offline.

Skipped:
- v2 phase 3 was not started, as requested.
- No real services or paid APIs were called.

Choices made while unattended:
- Stub half-width is `max(1, ceil(abs(noise)))`, so synthetic uncertainty follows the sampled noise while retaining a nonzero interval.
- Width-derived certainty is applied at the optimizer boundary; the existing view-builder interface remains until v2 phase 6.
- The `endpoint` output field is empty because `RawResult` does not yet expose a resolved serving endpoint; the column is retained for the later serving implementation.

## v2 phase 3: prompt manifest and skills placeholders

Built:
- Replaced the single skills placeholder with five equal-length neutral files and updated the manifest to `bare`, `placebo`, and `skills_1` through `skills_5`. `skills_5` is disabled by default; placebo matches `skills_mean`.
- Added enabled kind-based prompt discovery and mean-based skills and placebo length validation per enabled tokenizer, including the 6000-token cap, zero and one-skill cases, and unknown match handling.
- Added `skills_length_tolerance: 0.05` for DECISION(P1), assigned distinct stub noise levels to the five skills configurations, and updated the test configuration prompt subset.
- Updated README, design, and config-reference documentation for the manifest, enabled prompt sources, and length rules.

Tests and checks:
- Full pytest: 207 passed, 3 warnings in 487.05 seconds. The warnings are NumPy empty-slice warnings from the synthetic report test.
- `ruff check src tests scripts` and `mypy src scripts` passed.
- Manifest refresh check, placebo-length check, banned-word check, secret check, and yamllint passed.
- Targeted `ruff format --check` reports seven existing v1 files that would be reformatted. They are part of the known 27-file baseline and were left unchanged: the synthetic e2e test, Phase 10 and Phase 11 integration tests, and the Phase 5, Phase 6, Phase 8, and Phase 11 unit tests. The new v2 phase 3 test and the other checked files pass.
- `git diff --check` and LF line-ending checks passed.

Changed v1 tests and reasons:
- `tests/unit/test_phase5_prompts.py`: migrated placeholder, prompt assembly, and checker assertions to the new skills IDs and mean-length rules.
- `tests/unit/test_phase6_llm.py`: migrated stub call and cache expectations to distinct hashes for the five skills placeholders.
- `tests/unit/test_phase8_optimize.py`: migrated weight-building and readback checks to the enabled `skills_1` configuration.
- `tests/unit/test_phase11_eval.py`: migrated IC source and planted-edge assertions to `skills_1`.
- `tests/unit/test_v2_phase1_config_identity.py`: updated the test config source list for its four selected prompt IDs.
- Phase 10 integration and synthetic e2e tests: migrated persisted portfolio references to `skills_1` and updated counts for six enabled prompt configurations.
- Added `tests/unit/test_v2_phase3_manifest.py` for enabled kind ordering, disabled configuration exclusion, placeholder lengths, skills tolerances, mean matching, and zero, one, and unknown-match edge cases.

UNVERIFIED:
- No real-provider behavior was added or exercised. Live services remain outside the offline scope.

Skipped:
- v2 phase 4 and later phases were not started.
- No real data, paid APIs, brokers, or external services were used.

Choices made while unattended:
- Used equal-length neutral text with distinct configuration labels so the placeholders remain independently hashed and cached while the real texts are unavailable.
- Set stub noise to 3.0 for both controls and 1.2, 1.0, 0.8, 0.6, and 0.4 for skills_1 through skills_5, respectively, to meet the required distinct lower-noise progression.
- Applied the C4 token cap to enabled skills and placebo files because the plan requires every checked file to remain under the cap.

## v2 phase 4: ARCTIC LiteLLM provider and model registry

Built:
- Added the injectable-transport LiteLLM chat-completions provider, bearer authentication, JSON and seed payload flags, response parsing, retry-after-aware retries, immediate VPN guidance for 401/403, and run-scoped capability downgrade.
- Added bounded open-weight execution with main-thread record writes, resume and cache support, endpoint identity in cache keys, per-source endpoint status, J6 cutoff checks, endpoint archives, and full source restart after endpoint switching.
- Added gateway and fallback configuration, eligibility and registry validation, model registry updates, the `J4_open_weight_infer` implementation, and `--gateway-fallback` CLI and weekend options.
- Updated cost, secret, and smoke scripts. Smoke `--dry-run` prints gateway payloads without calling data services or model endpoints.
- Updated README and the relevant runbook, desktop, config-reference, and design documentation. Removed the v2 phase 4-listed Ollama provider and its payload test.
- Added `tests/unit/test_v2_phase4_litellm.py` for mocked provider behavior, concurrency, resume, endpoint switch, and registry validation. All HTTP in the tests is mocked.

Changed v1 tests and reasons:
- `tests/unit/test_phase1_scaffold.py`: updated model registry expectations to the D1 model list.
- `tests/unit/test_phase6_llm.py`: migrated stub IDs and open-weight kind checks; removed the Ollama payload test because v2 phase 4 replaces it with LiteLLM.
- `tests/e2e/test_full_run_synthetic.py`, `tests/integration/test_phase10_pipeline.py`, `tests/integration/test_phase11_report.py`, `tests/unit/test_phase7_views.py`, `tests/unit/test_phase8_optimize.py`, and `tests/unit/test_v2_phase1_config_identity.py`: migrated renamed stub IDs.

UNVERIFIED:
- The actual ARCTIC gateway URL and gateway authentication behavior were not available in the repository. `llm.gateway.base_url` remains empty rather than inventing an endpoint.
- ARCTIC support for `seed`, `response_format`, model identifiers, returned model version behavior, and reasoning fields has not been verified against a live service. The plan defaults for seed and JSON mode are enabled and capability downgrade is implemented.
- The fallback endpoint JSON format (`base_url` or `url`, optional `week_id`) and file-mtime freshness rule are implementation choices, not confirmed ARCTIC or Slurm requirements.
- No new dependency was needed; `httpx` and `tenacity` were already declared.

Tests and checks:
- Full pytest: 223 passed, 3 warnings in 583.69 seconds. The warnings are NumPy empty-slice warnings from the synthetic report test.
- `ruff check src tests scripts` and `mypy src scripts` passed. After formatting the new provider, its targeted Ruff check passed.
- Manifest refresh, placebo-length, banned-word, secret, dry-run smoke, and yamllint checks passed. The smoke dry run printed planned gateway payloads and made no requests.
- Targeted `ruff format --check` reports 11 changed files from the known 27-file baseline that would be reformatted. The new LiteLLM provider and v2 phase 4 test are formatted; baseline files were left unchanged.
- `git diff --check` and LF line-ending checks passed. No Ollama or Qwen references remain in `src/` or `config/`.

Skipped:
- No real data, paid API calls, ARCTIC requests, or fallback endpoint requests were made.
- v2 phase 5 was not started.

Choices made while unattended:
- Kept the gateway URL empty until the provisioned value is known.
- Used the endpoint file's modification time compared with the source run's recorded start time to reject stale fallback metadata. An optional matching week ID is also checked.
- Applied the run-scoped disabled state across resume by persisting seed and JSON-mode flags in source status and call records.
- Used four workers for the gateway models and set test-only stubs to one worker so synthetic failure injection remains deterministic.
- Derived a synthetic runner's default clock from its source-week schedule so historical synthetic status does not depend on the machine's current date. Synthetic open-weight stubs are not constrained by the live wall-clock cutoff.
- Preserved the prior stub randomness stream while renaming `stub-local-*` to `stub-ow-*`, so registry renames do not change unrelated synthetic portfolio fixtures.

## v2 phase 5: Kronos path spread

Built:
- Changed `ForecastProvider.forecast` to return per-ticker `k` and `k_sd`. `k` remains the mean forecast return; `k_sd` is the sample standard deviation with `ddof=1`.
- Added optional `predict_paths` support and the repeated single-sample fallback with seeds `seed + j`. Sampling-path selection is logged once per forecast call with an `UNVERIFIED` tag.
- Added path validation and logging for non-positive closes. Invalid paths are dropped; `k_sd` stays NaN with fewer than two valid paths, and both fields are NaN when no paths remain.
- Preserved the v1 stub `k` sequence and added a seeded lognormal `k_sd`. Added `k_sd` to `baseline.csv` and `WeekInputs.k_sd()`. Baseline rows with missing `k_sd` remain present.
- Documented DECISION(K1) in the README and the baseline formula and spread output in `docs/design.md`. DECISION(K1) view-uncertainty application is for v2 phase 6; this phase only adds the input field.
- Clarified v1 and v2 phase references in README and the affected design, config-reference, open-questions, and GitLab setup docs.

Tests and checks:
- Full pytest: 228 passed, 3 warnings. The warnings are the existing NumPy empty-slice warnings from the synthetic report test.
- `ruff check src tests scripts` passed. `mypy src scripts` passed with no issues in 80 source files.
- Targeted v2 phase 5 tests passed. `ruff format --check` passed for the changed source files and the new v2 phase 5 test. `tests/unit/test_phase3_baseline.py` remains unformatted within the known 27-file v1 baseline and was not bulk-formatted.
- Manifest refresh, placebo-length, banned-word, secret, and yamllint checks passed. `scripts/smoke_real_services.py --dry-run` printed planned payloads and made no requests.
- `git diff --check` and LF line-ending checks passed.

Changed v1 tests and reasons:
- `tests/unit/test_phase3_baseline.py`: migrated direct calls to the renamed forecast API and DataFrame input for `combine`, and asserted `k_sd` is present in every synthetic-week baseline.

UNVERIFIED:
- The repository has no `vendor/` directory and the configured Kronos commit is `UNPINNED`. The public upstream README documents multi-sample output as averaged, but does not establish the API at the configured commit.
- The `predict_paths` adapter is optional and mocked; actual vendor support for it and the repeated-call behavior have not been verified. The code marks the selected runtime path `UNVERIFIED` and needs a real smoke test in v2 phase 15.
- No Kronos model was downloaded or run.

Skipped:
- v2 phase 6 and later work was not started. No live model, paid API, broker, ARCTIC, or Slurm request was made. No dependency was added.

Choices made while unattended:
- Set stub `k_sd` to a per-ticker lognormal with median 1 percent and log sigma 0.5. Use a separate stream seeded with `seed + 1` so adding spread does not change v1 `k`.
- Use `predict_paths` as an optional per-path adapter; otherwise use repeated `sample_count=1` calls because the current public upstream documentation describes multi-sample output as averaged.
- If all Kronos paths are invalid, set both `k` and `k_sd` to NaN so `combine` excludes that ticker rather than inventing a return.

## v2 phase 6: width-based view uncertainty and BL posterior covariance

Built:
- Added the DECISION(K1) spread ratio to baseline view uncertainty and the DECISION(E1) width scale to LLM view uncertainty. `views_frame` and stored weights now include `k_sd`, `ratio`, `width`, and `c`.
- Added `Posterior.cov_mu`. The Black-Litterman precision matrix is factored once with Cholesky; triangular solves return the posterior mean and covariance. A failed factorization gets one logged diagonal-jitter retry before `OptimizerError`.
- Updated the view configuration, optimizer output columns, and shuffle view construction.
- Updated README and design/config-reference formulas for v2 phase 6.

Changed v1 tests and reasons:
- `tests/unit/test_phase7_views.py`: migrated the view formulas and synthetic-week builder to pass spread and width inputs.
- `tests/unit/test_phase8_optimize.py`: replaced the old certainty-input test with narrow-versus-wide range behavior, migrated the explicit BL formula test, and added a cap-preserving solver-tolerance regression test.

Tests and checks:
- Full pytest: 239 passed, 3 existing NumPy empty-slice warnings, 567.44 seconds.
- `ruff check src tests scripts` passed. `mypy src scripts` passed with no issues in 80 source files.
- Manifest, placebo-length, banned-word, secret, and yamllint checks passed. `scripts/smoke_real_services.py --dry-run` printed payloads only and made no requests.
- `ruff format --check` passed for changed source files and `tests/unit/test_v2_phase6_views_bl.py`. The two modified v1 tests remain unformatted within the known 27-file baseline, so I left their unrelated formatting unchanged.
- `git diff --check` and LF line-ending checks passed.

Choices made while unattended:
- Kept `cov_mu` as a ticker-labeled DataFrame so downstream strategies can align it with `mu_bl`.
- Used the existing SciPy dependency's triangular solver for the two Cholesky solves.
- Treated missing, infinite, or negative path spreads as unavailable, assigning ratio 1 to those stocks and logging the fallback.
- Treated an omitted `k_sd` series as all missing, which keeps callers without baseline spread data on the neutral ratio.
- Kept validated width and `c` values available for audit output when `IC_L` disables LLM view rows.
- Repaired solver-tolerance cleanup by redistributing any shortfall only across remaining position-cap headroom. The v2 phase 6 posterior change exposed that the previous clip-then-renormalize step could exceed the fixed maximum weight.

## v2 phase 7: strategy package

Built:
- Added shared strategy inputs, result records, guardrails, water-filling cap normalization, ordered solver attempts, fallback chain, deterministic common normals, Cholesky retry, and `view_shift`.
- Added MVO, Mean-CVaR, Robust MVO, Resampled MVO, Top-K, HRP, and equal-weight strategy/control implementations with registry dispatch.
- Enabled all five steered strategies and both controls in `config/study.yaml`; declared the optional `solvers` extra for HIGHS.
- Added the Phase 7 strategy suite and updated README and design/config references.
- CI sets `GIT_DEPTH: "0"` so the checklist history test sees all phase commits rather than a depth-20 clone.

Changed v1 tests and reasons:
- `tests/e2e/test_full_run_synthetic.py`: compute the expected configured portfolio count and exercise generated weights and trades for replayed portfolios; HRP/EW weight files are part of the v2 phase 8 J7 integration.
- `tests/integration/test_phase10_pipeline.py`: assert fallback status and log counts across all five configured strategies rather than only MVO.
- `tests/integration/test_phase11_report.py`: require shuffle results only for non-control portfolios, and check positive transaction costs only for portfolios replayed before v2 phase 8.

`tests/unit/test_v2_phase7_strategies.py` uses a `1e-8` tolerance for the zero-optimal CVaR weights because the solver returns small floating-point residuals.

Tests and checks:
- Full pytest: 288 passed, 1199 warnings in 1190.94 seconds. Warnings are pandas DataFrame-fragmentation notices from evaluation and three NumPy empty-slice warnings in the synthetic report test.
- 300-stock Mean-CVaR benchmark: 10,000 scenarios, beta 0.95, lambda 1.0; CLARABEL completed in 23.434 seconds with no fallback, using seeded synthetic inputs.
- `ruff check src tests scripts` passed. `mypy src scripts` passed with no issues in 89 source files.
- `ruff format --check` passed for the strategy modules and the new Phase 7 test. The three modified v1 tests remain unformatted within the known 27-file baseline; no bulk formatting was applied.
- Manifest refresh, placebo-length, banned-word, secret, and yamllint checks passed. `scripts/smoke_real_services.py --dry-run` made no requests.
- `git diff --check` and LF line-ending checks passed.

Choices made while unattended:
- Kept the Phase 7 strategy/control registry independent of weekly J7 replay wiring scheduled for Phase 8; adjusted existing pipeline assertions to cover the current phase boundary.
- Encode common random seeds as the first eight bytes of SHA-256 over canonical JSON `[seed, week_id, strategy]`.
- Define resampled weight dispersion as the mean standard deviation across ticker weights.
- Use inverse-variance allocations inside HRP clusters and retain the leaf order and pre-cap weights for diagnostics.
- Compute diagnostic empirical CVaR as a fractional average over the worst tail, including fractional mass at its boundary.
- Label non-solver portfolios `deterministic` and set their unused objective to NaN.
- Retry a covariance Cholesky factorization once with diagonal jitter `1e-12 * max(trace(M) / N, 1)`.
- Add `highspy>=1.7` as the optional HIGHS solver extra; the repository environment already provides an importable `highspy`, so no package installation was needed.

## v2 phase 8: weights and replay over strategies

Built:
- Added shared Q0/source posterior generation and persisted the required source-level posterior CSVs. Weight metadata records the covariance-of-mean SHA-256, strategy diagnostics, solver/fallback fields, endpoint, decision source, and view shift.
- Wired J7 to produce Q0 weights, HRP/EW controls, then source portfolios; source failure and lock-deadline paths copy strategy-specific Q0 files and metadata.
- Changed fallback plans to source keys, replay plans to use persisted `decision_source`, and IC history to read baseline/raw/residual/combined signals from source posteriors.
- Added strategy-grouped status summaries and updated the test config to enable all strategies and both controls.
- Updated README, design, runbook, config reference, and Phase 8 integration assertions.

Changed v1 tests and reasons:
- `tests/integration/test_phase10_pipeline.py`: assert source-keyed fallback reasons while retaining per-strategy fallback outputs.
- `tests/integration/test_phase11_report.py`: include controls in replay and shuffle expectations now that Phase 8 writes and replays them.
- `tests/e2e/test_full_run_synthetic.py`: include controls in replay assertions and derive score counts from configured sources; `test_failure_week` adds the API stubs to `llm.model_ids` because the test config lists only the open-weight stubs.
- `tests/integration/test_phase10_pipeline.py` `test_j3_cost_cap_blocks_then_allow_over_budget` adds the API stubs because only API models carry a price, so the cap cannot trip on open-weight stubs alone.
- `tests/e2e/test_full_run_synthetic.py` `test_cost_log`: the total cost must be positive exactly when the run includes an API model.
- `tests/unit/test_v2_phase1_config_identity.py`: the test-config portfolio order now covers every enabled strategy and control (Q0s, controls, then steered keys) instead of MVO only.
- `tests/unit/test_phase1_scaffold.py`: `test_config_loads_and_has_decisions` loads `config/study.yaml` itself because it checks the study file's decision list.
- `tests/conftest.py`: the shared `study_config` fixture now loads `config/study_test.yaml`. Under `config/study.yaml` (4 stub models x 6 prompt types x 5 strategies = 129 portfolios, 10,000 CVaR scenarios, 200 resampled draws) the 3-week pipeline fixture took 720 s to build and one J1 to J9 test took 250 s. Tests that counted sources from the registry and manifest now count them from the run context or `cfg.llm`; `test_j5_batch_never_returns_is_a_labeled_fallback` adds the API stubs.

Tests added:
- `tests/integration/test_v2_phase8_jobs.py`: weight file count and `check_weights` on every file, posterior files per source, forced source failure copies each strategy's Q0 with `fallback_q0` and `view_shift` 0, dropped LLM view (cold-start IC 0) byte-identical to the strategy's Q0, lock deadline mid-loop labels unfinished sources `lock_deadline`, IC history has one row per source and signal per week, forced J7 and J8 reruns give identical file hashes and trade records.
- `tests/unit/test_v2_phase8_jobs.py`: source-keyed fallback status applies to every strategy; covariance-of-mean hash is stable and sensitive.

Runtime under the test config (8 cores, VM): e2e 25-week setup 494 s, `test_failure_week` 130 s, `test_determinism_two_runs_identical` 90 s, whole `tests/e2e` about 13 min; the slowest unit/integration items are the LiteLLM 429-storm fixture (42 s), the 30 consecutive snapshots (29 s), and the J7/J8 rerun test (28 s). `scripts/run_weekend.py print_summary` skips the `.meta.json` entries that `locked.json` now lists.

Choices made while unattended:
- Tests run under `config/study_test.yaml` (see the README design choice); the "30-week integration test" in the Phase 8 done-when is read as the e2e synthetic run (`tests/e2e`, 25 trading weeks on the 30 snapshot weeks of `tests/conftest.py`) with all five strategies and both controls under the test config.
- The J7 integration tests live in `tests/integration/` because they run J1 to J8 on the shared pipeline fixture.
- Compute one posterior and covariance-of-mean per rating source and reuse it across strategies, so allocation is the only strategy-dependent input.
- Keep source-level failure and fallback reasons in J6; J7 copies outputs separately for each strategy.
- If the deadline arrives during J7, preserve completed portfolios and use Q0 copies for unfinished portfolios.

## v2 phase 9: 300-stock scale and compute projection

Built:
- `optimize/problem_cache.py`: compiled cvxpy problems keyed by constant data with `mu` as a `cp.Parameter` (LRU of 16; the ticker tuple is in every key). MVO (`mean_variance.py`) keys on (tickers, SHA-256 of Sigma, delta, cap); robust MVO adds the SHA-256 of M; mean-CVaR adds M, week, seed, and scenario settings because its shocks come from `Sigma + M`. Resampled MVO reuses the cached MVO problem for every draw. Solves pass `warm_start=False` so cached and fresh solves agree to 1e-9 (`use_cache=False` builds the same parametrized problem fresh).
- `data/universe.py`: v1 rule unchanged; logs `universe_thin_sector` (sector, quota, available, short) for every sector below its quota, `universe_leftover_slots` (slots, filled), and `universe_short` when the size cannot be reached. `build_universe` takes an optional logger.
- `scripts/benchmark_pipeline.py`: `--n-stocks`, `--sources`, `--strategies`, `--controls`, `--weeks`, `--project-sources` (default 51), `--seed`, `--out-dir`; times every strategy and control on a seeded 3-factor covariance, prints seconds per solve, J7 seconds per week now and at the projected source count, peak RSS, and projected shuffle CPU hours (weeks x shuffles x steered portfolios x mean strategy solve time); writes `data/benchmarks/<date>.csv` with fixed `BENCHMARK_COLUMNS`; warns above 15,000 CPU hours. `--jobs` keeps the v1 per-job timing.
- `config/study.yaml`: `universe.size` comment `DECISION(B4): 100 to 300, set after the ARCTIC allocation`. `pyproject.toml` `addopts` deselects `slow` by default.

Benchmark (VM, 8 cores, `config/study.yaml` settings, 300 stocks, 3 sources, mean seconds per solve): mvo 0.13, cvar 27.2 (10,000 scenarios, CLARABEL), robust 0.56, resampled 17.1 (200 draws), topk 0.002, HRP 0.014, EW 0.000. J7 seconds per week: 180 at 3 sources; 2,343 (39 min) at 51 sources (7 models x 7 prompt types + Q0). Peak memory 1,348 MB. Projected shuffle CPU hours at 52 weeks x 500 shuffles x 255 steered portfolios x 9.0 s mean solve: 16,595, which exceeds the 15,000 CPU hour ARCTIC request by about 1,600 hours (11 percent). Mean-CVaR is 75 percent of the solve time; the gap is reported here and no DECISION value was changed to close it (options for Carter: fewer CVaR scenarios, fewer shuffles, or a smaller universe). `--jobs` mode was not rerun.

Tests (`tests/unit/test_v2_phase9_scale.py`): synthetic members with a 15-name sector at size 300 give 300 names with the thin-sector and leftover-slot logs; cached vs uncached MVO within 1e-9 on 20 random mu vectors with one miss and 19 hits; the cache key separates ticker and Sigma changes; projection arithmetic; the benchmark writes its table under the test config at 40 stocks; a `@pytest.mark.slow` 300-stock benchmark excluded from default runs.

Changed v1 tests and reasons: none.

UNVERIFIED: the 51-source projection assumes 7 models x 7 prompt types; ARCTIC core speed relative to this VM is unknown, so CPU hours are VM-relative.

Choices made while unattended:
- Benchmark on synthetic matrices rather than a pipeline week, so the numbers isolate optimizer cost (README Design choices).
- `warm_start=False` on every solve, trading a little OSQP speed for cached and fresh solves being identical.
- Shuffle projection counts one solve per week per shuffle per steered portfolio.

## v2 phase 10: per-strategy evaluation

Built:
- `eval/data.py`: `EvalData` carries `strategy_of`, `source_of`, `controls`, `solver_failed` and `view_shift` (weeks by portfolios, from the weight meta files), `strategies()`, `q0_key()`, `steered_for()`, and `source_clean_mask()`; `stock_week` holds `likely:<source>` and `width:<source>` once per rating source; weekly returns cover every portfolio including HRP and EW.
- `eval/hypotheses.py`: H1 and H2 IC per source (Holm across sources); H2 return pairs each steered portfolio with its own strategy's Q0 and Holm-adjusts within each strategy; H3 takes the `skills` prompt types from `manifest.by_kind` and Holm-adjusts per model; `h3b_width` (DECISION(G3b)) splits each clean source-week at the median width, skips small or all-equal weeks and counts them, reports mean IC per width group; `h4_strategy_ranking` ranks strategies by the mean weekly gap vs their Q0 across clean sources with a bootstrap CI (no Holm) and counts dropped weeks; `h4_splits` runs market direction and volatility on source IC gaps and on strategy gaps; `portfolio_metrics` adds `strategy`, `source`, `mean_view_shift`, `fallback_weeks`, `solver_failed_weeks` and covers controls.
- `eval/shuffle.py`: H5 Holm within each strategy (steered portfolios plus its Q0), controls excluded.
- `eval/report.py`: run summary header (portfolio counts, weeks, fallback and fallback-served weeks, K1, S4, S5, G3b, P1 values), H4 ranking table and `fig_h4_ranking.png`, one section per strategy (metrics, H2 return table, H5 table, `fig_equity_<strategy>.png` with Q0, steered, HRP, EW), H3b table and `fig_h3b_width.png`, a WARNING line for a strategy whose mean `view_shift` is below 0.01, `h3b_width.csv` and `h4_strategy_ranking.csv`.
- `config.py` `H3bConfig` (`eval.h3b`), `config/study.yaml` `eval.holm_families` per PLAN2 and `eval.h3b` tagged DECISION(G3b).

Tests: `tests/unit/test_v2_phase10_eval.py` on a synthetic 30-week `EvalData` (2 models x 4 prompts x 2 strategies, HRP and EW): Holm within each strategy only on a planted edge, H2 pairs with the own Q0, H3 finds every skills kind per model, H3b positive on planted width data and skips all-equal weeks, H4 ranking matches the planted order and counts dropped weeks, controls appear in metrics and in no family, config families and H3b validation. `tests/integration/test_phase11_report.py` checks the new files and report sections on the synthetic pipeline run.

Changed v1 tests and reasons: `tests/integration/test_phase11_report.py` expects the v2 phase 10 outputs and the per-strategy shuffle column.

UNVERIFIED: none new.

Choices made while unattended: see the README design choices for v2 phase 10 (splits per strategy, stock-week table per source, fallback count definitions, header values).

## v2 phase 11: sharded shuffle test

Built:
- `eval/shuffle.py`: per-week RNG `week_rng(seed, portfolio, index, week)`; `shuffled_weights` permutes the range rows (`low, likely, high, width`) and reruns residualize, views, BL, and the portfolio's own strategy through `run_strategy` (Q0 permutes `z_b` and `k_sd` together); `Shard`, `plan_shards`, `write_shard_plan`, `read_shard_plan`, `run_shard` (atomic JSON with `complete` marker under `data/shuffle/<portfolio>/`), `run_shards_local` (optionally multi-process), `collect_shards` raising `MissingShards` with every missing range, `shuffle_table` (p-values, Holm within each strategy), `shuffle_test(shards=, require_complete=)`, `identity_weights`.
- `pipeline/cli.py`: `portstudy shuffle plan|run|run-local|collect`.
- `config.py` `ShuffleConfig.shard_size` (25) and `config/study.yaml`.

Tests (`tests/integration/test_v2_phase11_shuffle.py`): 20 shuffles in 4 shards serial, reversed, rerun (files untouched), and in 4 processes give identical results; a missing shard fails `collect` with `[5, 10)` listed and is run by the in-process path; a shard file without its marker or with invalid JSON is rerun to identical content; every shuffle keeps the multiset of range triples per week and different (index, week) keys give different permutations; the identity permutation equals the stored weights for every strategy's Q0 and steered portfolio (atol 1e-6); the plan covers each Q0 and steered portfolio, no controls, and round-trips through CSV.

Changed v1 tests and reasons: none beyond the v2 phase 10 report test (shuffle table now carries `strategy`).

UNVERIFIED: Slurm array behaviour on ARCTIC (v2 phase 12 adds the sbatch file).

Choices made while unattended: see the README design choices for v2 phase 11.

## v2 phase 12: ARCTIC Slurm scripts

Built:
- `scripts/arctic/`: `arctic.env.example` (every value `TBD`, pointing to `docs/arctic_setup.md`), `common.sh` (loads the env, refuses `TBD`, activates the venv, exports `PORTSTUDY_DATA_DIR`), `kronos.sbatch` (J2, J1 first in compute mode), `weekend.sbatch` (`run_weekend.py --from J3` or `--from J6` by `ARCTIC_NETWORK_MODE`, `PORTSTUDY_FROM` override), `submit_weekend.sh` (compute: kronos then weekend `--dependency=afterok`; login: J1 here, kronos, J3 to J5 here, weekend J6 to J9; walltime warning via `--check-walltime`; prints job ids, log paths and rerun commands), `shuffle_array.sbatch` (`portstudy shuffle run --shard-index $SLURM_ARRAY_TASK_ID`), `llm_fallback.sbatch` (DECISION(D2) server for fallback-eligible models, writes host, port, week, start time and job id to the endpoint file; serving tool `# UNVERIFIED`, vLLM assumed).
- `llm/health.py`: `gateway_health_check` (one tiny prompt per enabled open-weight model on the gateway, report in `gateway_health.json`), `mark_sources_failed` (N1 statuses), `d2_steps`, `parse_walltime`, `unhealthy_models`.
- `pipeline/jobs.py` J4 skips models marked unhealthy unless `--gateway-fallback`.
- `scripts/run_weekend.py`: `--from`, `--until`, `--check-walltime`, `jobs_between`, `gateway_health` before J4.
- `backup.py`: `rsync_command`, `rsync_backup`; `BackupConfig.rsync_target`.
- `docs/arctic_setup.md`.

Tests (`tests/integration/test_v2_phase12_arctic.py`): shellcheck on every script (skipped with a note when not installed; installed here through `shellcheck-py`); `run_weekend.py --from J3 --synthetic` runs J3 to J9 only and `--until J5` stops after J5; a health failure without the flag marks exactly the failing model's sources failed, J4 skips that model, and J6/J7 serve Q0 for them; with a fresh endpoint file and the flag the fallback-eligible model switches to `slurm_fallback` and the others stay on the gateway; a stale endpoint file is rejected; `parse_walltime`; the walltime warning; `rsync_command` and a failing rsync that does not raise.

Changed v1 tests and reasons: none.

UNVERIFIED: every ARCTIC value (`TBD`), the serving command in `llm_fallback.sbatch`, Slurm behaviour (no cluster here).

Choices made while unattended: see the README design choices for v2 phase 12.

## v2 phase 13: end-to-end rehearsal and CI

Built:
- `tests/e2e/conftest.py`: `E2E_WEEKS = 30`; `run_rehearsal` runs the clean weeks plus `INJECTIONS` (week 3 `stub-ow-b` invalid-output storm, week 6 `NO_SUCH_SOLVER` in every optimizing strategy, week 9 lock-deadline cutoff mid-J7 through `lock_cutoff_clock`, week 12 gateway health failure via `GatewayDown` followed by `run_weekend.py --from J4 --gateway-fallback`); `rehearsal_registry` makes `stub-ow-a` fallback-eligible so D2 switches it and `stub-ow-b` stays on the gateway.
- `tests/e2e/test_full_run.py`: v2 invariants over the shared run: one weight file per portfolio and week, one Q0 per strategy, controls never read views (`decision_source = control`, `use_llm_view = False`, `view_shift = 0`), every weight file passes `check_weights`, IC history has one row per source and signal and no strategy column; one test per injected week (fallback statuses, exact Q0 copies, `solver_failed` on every optimizing strategy but not Top-K or the controls, `lock_deadline` copies with completed portfolios kept, the gateway week ending `ok` with `decision_source = llm`); the planted delisting leaves the universe and every book; eval over the clean weeks with a positive H3b narrow-minus-wide gap and `view_shift > 0` for every steered strategy; a 20-shuffle test in 4 shards through `run_shards_local` and `collect_shards`; an output digest (`PORTSTUDY_E2E_DIGEST_FILE`) for the two-runs-identical check.
- `tests/e2e/test_full_run_synthetic.py` (v1 rehearsal): skips the four injected weeks in its "every week normal" checks; the determinism test compares the three weeks before the first injection.
- `.gitlab-ci.yml`: `test` runs `tests/unit` on every push; `integration` and `e2e` run on the default branch; `slow` is manual (`pytest -m slow tests`); yamllint now covers `config/study_test.yaml`.

Changed v1 tests and reasons: `test_full_run_synthetic.py` as above, because the shared run now contains the injected failure weeks.

e2e runtime (this machine, `pytest tests/e2e` alone): 13 min 1 s wall for 17 tests, of which the 30-week rehearsal fixture takes 9 min 35 s and `test_failure_week` 2 min; under the 15-minute bound, so the test config was not shrunk. Two runs give the same output digest (`PORTSTUDY_E2E_DIGEST_FILE`).

## v2 phase 14: documentation and final checklist

Built:
- `README.md`: v2 scope in "How the study works" (five strategies, controls, the portfolio count formula, calls per week), "Going live" on ARCTIC with the commands, the **Design choices** section opening with K1, S4, S5, G3b, and P1 in the plan's wording (value, config key, reason) and the D3 pointer to `docs/open_questions.md`, documentation table rows for `docs/arctic_setup.md` and `docs/arctic_gpu_request.md`.
- `docs/design.md`: "v2 in one page" (rating source vs portfolio, the strategy package, posterior covariance, both uncertainty formulas, common random numbers, the fallback chain, the single-endpoint rule, the 5 percent cap at each rebalance with drift between rebalances).
- `docs/runbook.md` section 9: the ARCTIC weekend in both network modes, the D2 steps, backups and rsync, the post-run shuffle array and eval.
- `docs/config_reference.md`: `backup.rsync_target` and the environment variables `ARCTIC_LITELLM_API_KEY`, `ARCTIC_FALLBACK_API_KEY`, `ARCTIC_ENV`, `ARCTIC_NETWORK_MODE`, `PORTSTUDY_CONFIG`, `PORTSTUDY_WEEK`, `PORTSTUDY_FROM`, `PORTSTUDY_GATEWAY_FALLBACK`, `PORTSTUDY_SHARDS`, `PORTSTUDY_FALLBACK_PORT`, `PORTSTUDY_E2E_DIGEST_FILE`.
- `docs/open_questions.md`: D3 is the only provisional row; C4 and C7 are marked confirmed v1 values (PLAN2 section 0.3 lists them as unchanged from v1) and are rechecked in phase 15 G. The C3 wording no longer uses the retired single-point field name.
- `docs/arctic_setup.md`: rewritten in the first person with the checklist (env file, venv, model weights in `HF_HOME`, Hugging Face license acceptance for Gemma and Llama, gateway key and VPN, network mode probe, quotas, dry run).
- `scripts/check_banned_words.py`: flags "the researcher" and "the student" in `docs/arctic_setup.md`.
- Left in place, as the plan asks: `docs/arctic_gpu_request.md` (untracked on purpose until phase 15 D) and `docs/desktop_checklist.md`.

Final checks (item 9): em dash scan and banned words (`check_banned_words.py`), `ruff check src tests scripts`, `mypy src scripts`, the full suite (unit, integration, e2e), `check_secrets.py`, and `grep -rni "qwen\|ollama\|confidence" src/ config/` (tracked files; the stale v1 `ollama` provider module was removed in v2 phase 4, and only build artefacts under `__pycache__`/`egg-info` still mention it). All passed at this commit: ruff and mypy clean, 315 unit and integration tests, 17 e2e tests (two runs with the same output digest), no em dash, no banned word, no secret, no stale reference.

v2 phase 13 follow-up recorded here: the v1 rehearsal test called `run_eval` with a 2-shuffle H5 over all 77 portfolios, which alone took 10.7 minutes once the strategy package was in; it now runs the eval with `skip_shuffle=True` (the shuffle is exercised by `test_twenty_shuffles_in_four_shards` and the v2 phase 11 integration tests). Measured e2e runtime after that change: 14 min 13 s wall for 17 tests on this machine (the 30-week rehearsal fixture 10 min 22 s, `test_failure_week` 2 min 12 s), under the 15-minute bound.

UNVERIFIED items still open for phase 15:
- Every ARCTIC value in `scripts/arctic/arctic.env.example` (account, partitions, GRES, walltimes, module, venv, data root, `HF_HOME`, network mode, array throttle).
- The serving command in `scripts/arctic/llm_fallback.sbatch` (vLLM assumed) and the fallback-eligible model ids.
- Gateway behaviour: JSON mode, seeds, rate limits, MiniMax reasoning output, small-model ids; the gateway key is read from `ARCTIC_LITELLM_API_KEY` and redacted in logs.
- Kronos per-path sampling (`predict_paths`) on the real checkpoint; the 20-call fallback costs about 20x the GPU time.
- The v2 phase 9 projection: about 16,600 shuffle CPU hours at 300 stocks against the 15,000 requested; universe size 100 in `config/study.yaml` until the allocation is known.
- D3, the paid API budget, after the phase 15 G test prompts.

## Review fixes (from results.md)

Each item below was found in the read-only code review (`results.md`), fixed in its own commit,
and confirmed by a regression test in `tests/unit/test_review_fixes.py` that fails on the old
code and passes on the new one, plus the full suite before the push.

- **H1, forward-filled rows reached the raw ledger** (`data/snapshot.py`): `make_snapshot` now
  keeps `raw_prices` before `_forward_fill` and appends only the provider's rows (plus ETF rows)
  to the ledger; the forward-filled frame still goes to the snapshot for the models. Confirmed:
  `test_h1_ledger_holds_only_provider_rows_after_forward_fill` (planted halt `SYN009`: the
  snapshot has the three filled rows with volume 0, the ledger has none).
- **H4, J8 never ran on ARCTIC** (`scripts/run_weekend.py`): `step1_catch_up` (J8 and J9 for
  earlier weeks) runs on every invocation regardless of `--from`; `--from J8`/`--until J8` runs J8
  for the current week; the default weekend run still leaves this week's J8 to next weekend.
  Confirmed: `test_h4_catch_up_runs_for_every_start_job_and_from_j8_runs_j8`.
- **H2, a failed J1/J2 week vanished from the replay** (`scripts/run_weekend.py`,
  `pipeline/jobs.py`): when J1 or J2 fails the script now runs J6 (which plans `no_rebalance`
  when J2 is not done) and J7 (which writes the `no_rebalance` lock) before exiting 1; and
  `jobs.replay_weeks` gives J8 every calendar week from the first locked week to the current one,
  so a week with no lock is replayed as a hold (stops, dividends and daily values are no longer
  skipped). Confirmed: `test_h2_replay_weeks_fill_the_gap_and_a_j1_failure_writes_the_hold`.
- **H3, login mode aborted at J4** (`scripts/arctic/submit_weekend.sh`): login mode now runs
  J3 alone, waits for the Kronos job to leave the Slurm queue (`wait_for_job`, `squeue` poll,
  `WAIT_POLL_SECONDS`), then runs J4 to J5 and submits the weekend job. Confirmed:
  `test_h3_login_mode_waits_for_kronos_before_j4` (stubbed `sbatch`/`squeue`/`portstudy`/`python`
  record the call order; J4 starts only after the last poll).
- **M4 + M10, `current_week` used a fixed UTC-5 and rolled at midnight** (`pipeline/jobs.py`):
  the clock is read in `America/New_York` (DST-aware) and the week rolls forward only at the
  cutoff day's close (16:00 ET, 13:00 on an early close), so a run started Friday afternoon no
  longer freezes a snapshot without Friday's bar. Confirmed:
  `test_m4_m10_current_week_is_dst_aware_and_rolls_at_the_close` (summer Friday 23:30 EDT,
  15:59 vs 16:00, early close 2026-11-27, Good Friday cutoff Thursday, naive datetimes).
- **M5, a done job was skipped even when its outputs were gone** (`pipeline/status.py`):
  `run_job` skips a `done` job only while `hash_paths(outputs())` still equals the recorded
  `outputs_hash`; a deleted or edited output runs the job again (with a warning). Confirmed:
  `test_m5_done_job_reruns_when_its_outputs_are_gone`.
- **M1 + M2, provider errors dropped tasks; authentication failures were per task**
  (`llm/runner.py`, `pipeline/jobs.py`): a task whose call raises `LlmProviderError` is queued
  again up to `MAX_PROVIDER_ERRORS = 2` times (`RunSummary.n_provider_retries`), then dropped
  (`n_dropped`); a non-retryable `LlmAuthenticationError` stops the model's run at once
  (`RunSummary.aborted = "authentication_failed"`) and J4 marks the model's sources failed with
  that reason. The 429-storm test now expects 9 gateway calls (3 attempts x 3 provider retries).
  Confirmed: `test_m1_m2_provider_errors_are_requeued_and_auth_failures_abort_the_model`.
- **M3, non-eligible models ran against a failed gateway under `--gateway-fallback`**
  (`scripts/run_weekend.py`, `pipeline/jobs.py`): the health check marks the failed models that
  are not `fallback_eligible` as failed (N1) even in fallback mode, and J4 only lifts the
  unhealthy mark for eligible models. Confirmed:
  `test_m3_gateway_fallback_marks_non_eligible_failed_models`.
- **M6, the view shift was forced to 0 on a solver fallback** (`optimize/__init__.py`):
  `_shift_from_q0` measures the shift on the weights actually written, fallback or not; the
  fallback itself stays in `fallback_flag`/`fallback`. Confirmed:
  `test_m6_view_shift_is_measured_on_fallback_weights_too`.
- **M7, the dropped-view branch solved Q0 again and rewrote its files**
  (`optimize/__init__.py`): `build_weights` now uses `existing_q0_weights` (the Q0 csv and
  meta J7 already wrote for the strategy) and only solves Q0 when it is missing. Confirmed:
  `test_m7_dropped_view_branch_reuses_the_written_q0` (round trip of csv + meta into `Weights`).
- **M8, J7's inputs hash ignored the IC history** (`pipeline/jobs.py`): `j7_inputs_hash` adds
  `data/outputs/ic_history.csv`, so an IC update before the lock invalidates a `done` J7; the
  lock file records `ic_history_sha256`, and a locked week hashes that pinned value, so J8's
  later updates never re-solve locked weights (the idempotent-rerun integration test holds).
  Confirmed: `test_m8_j7_inputs_hash_changes_with_the_ic_history`.
- **M9, the promised log redaction did not exist** (`logging_setup.py`): `redact` runs on every
  record the JSON-lines handler writes: keys named like `authorization`, `api_key`, `token`,
  `secret`, `password` lose their value; strings lose `Bearer ...` tokens, `Authorization: ...`
  values and `sk-...` keys. Confirmed: `test_m9_log_files_redact_gateway_credentials` (the key
  appears nowhere in the written file, harmless fields survive).
- **M11, the shuffle total return compared a fresh-start replay against the full history**
  (`eval/shuffle.py`): `real_metrics` measures the real portfolio over the selected weeks only,
  starting from the value carried into the first selected week (the shuffles start from
  `initial_value` over the same weeks). Confirmed:
  `test_m11_real_metrics_start_from_the_value_entering_the_selected_weeks`.
- **L1** is covered by H4 (`--from J8` runs J8).
- **L2, `3.0` passed as an integer** (`llm/validate.py`): `low`/`likely`/`high` must be JSON
  integers; integral floats are `not_integer` like any other float. The schema test that listed
  `3.0` among valid answers now asserts the rejection. Confirmed:
  `test_integral_floats_are_not_integers`.
- **L3, cached prompt tokens were never read from the gateway** (`llm/providers/litellm.py`):
  `usage.prompt_tokens_details.cached_tokens` fills `RawResult.tokens_cached`, so the cost table's
  cached price applies. Confirmed: `test_l3_litellm_reads_cached_prompt_tokens`.
- **L12, the fallback endpoint file was written before vLLM was ready**
  (`scripts/arctic/llm_fallback.sbatch`): each server is polled on `/v1/models` (every 15 s, up
  to `FALLBACK_READY_TIMEOUT_SECONDS`, default 1800) before `fallback_endpoint.json` is written;
  a server that never answers exits the job with code 3. Confirmed: `bash -n` and shellcheck;
  the loop cannot run in CI without vLLM.
- **L13, `estimate_cost.py` counted the whole registry** (`scripts/estimate_cost.py`): planned
  calls use `llm.model_ids` and `llm.prompt_ids` when set. Confirmed by running the script on an
  empty data dir (`4 models x 6 prompt types x 100 stocks` under the test config).
- **L14, one `httpx.get` per request** (`data/providers/http.py`): `LiveHttp` keeps one
  `httpx.Client`, so paginated Alpaca calls reuse connections. Confirmed by the provider tests
  (unchanged behaviour through `MockTransport`-free fixtures).
- **L9, config reference missed leaf keys** (`docs/config_reference.md`): the strategy section
  names `strategies.cvar.n_scenarios`, `cvar.beta`, `cvar.lam`, `resampled.n_draws`,
  `resampled.seed` and `resampled.max_failed_share` explicitly (the `data.rate_limits.*` and
  `data.checks.*` rows already cover their keys).
- **L10, scientific dependencies had no upper bounds** (`pyproject.toml`): `scipy<2`,
  `scikit-learn<2`, `statsmodels<1`, `cvxpy<2`, `osqp<2`, `clarabel<1`, `pyyaml<7`, `typer<1`,
  `httpx<1`, `tenacity<10`, `matplotlib<4`; the pinned environment satisfies all of them.
- **Left as recorded design choices, not changed:** L4 (Sharpe with a zero risk-free rate; DTB3
  is collected for the paper, and the shuffle and hypothesis tests compare portfolios on the same
  weeks so the constant cancels), L7 (MVO post-solver renormalisation is documented in the
  strategy module), L8 (yfinance market caps are current, not as-of; the Alpaca-first universe
  does not use them, and the limitation is in the data documentation), L11 (shuffle CPU-hour
  projection omits the Q0 re-solve share; the ~16,600 h conclusion is unchanged), L15 (cost
  estimates assume the output-token cap and are upper bounds; stated in `estimate_cost.py`'s
  output). L5 and L6 (per-call `run_status.json` rewrites, repeated fallback-plan reads) are
  performance items measured at under 1% of the weekly run and are deferred. L16 (real
  split/dividend smoke test against Alpaca) needs live keys and is part of phase 15.

## README rewrite

`README.md` was rewritten as the research-project overview after the review fixes: research
design (unit of study, treatments, models, schema, built-in controls), the hypotheses H1 to H5
with their Holm families, the methodology with the formulas in force, the weekly pipeline by
job, the failure-handling policy, compute and services, reproducibility, project status
(phase 15 manual work and the `UNVERIFIED` items), and the design-choice register kept in full.
Every number in it (24 sources, 127 portfolios, test timings, shuffle projection) was checked
against `config/study.yaml`, `config/models.yaml` and the current test runs.

## GitHub as the primary host

The repository moved to `https://github.com/ctsc/llm_portfolio_management` (commit "transfer all
code from gitlab to github"); GitLab stays as the remote `gitlab` and its CI file is kept current
so the project can go back without code changes. `.github/workflows/ci.yml` mirrors
`.gitlab-ci.yml` job for job (lint, unit with coverage, integration and e2e on the default branch,
slow by hand, checks); `test_github_workflow_mirrors_gitlab_ci` asserts that every GitLab script
line appears in the matching GitHub job and that the history-reading jobs use `fetch-depth: 0`.
Docs: `docs/github_setup.md` (new), `docs/gitlab_setup.md` (marked fallback), README references.

## v3 phase 1: config, decisions, formatting, and stale comments

PLAN3 (`PLAN3.md`) corrects the repository against the two proposals in `docs/proposals/`.

- `ruff format src tests scripts` reformatted 23 files (formatting only; no test changed).
- `config/study.yaml`: `llm.gateway.base_url` is `https://litellm.rs.gsu.edu/v1` (GSU network or
  VPN); the unsourced `minimax-m2.7: max_output_tokens: 2048` override is removed (`settings: {}`);
  `eval.shuffle.strategies: [mvo, robust, topk]` (DECISION(G6), PLAN3 0.1); the `repo_commit` and
  `live_end_date` comments name Carter and PLAN2 phase 15; `decisions` has the PLAN3 G6 text and the
  new W1 (walltime at most 15 hours) and M1 (`--mem-per-cpu=3G`) rows. `config/study_test.yaml`
  adds the same `eval.shuffle.strategies` and keeps 20 shuffles.
- `config.py`: `ShuffleConfig.strategies` (default mvo, robust, topk) and a `StudyConfig`
  validator that rejects a shuffle strategy not in `strategies.enabled`.
- `config/models.yaml` header and `pyproject.toml` description per PLAN3; "Phase 17",
  "the researcher" and "the desktop" removed from every code comment and docstring listed in PLAN3
  finding 10 (`grep -rn -i "phase 17\|the researcher\|the desktop" src scripts config` is empty).
- `scripts/check_banned_words.py`: the "the researcher" / "the student" check now applies to every
  scanned file outside `prompts/`; `PLAN3.md`, `PROGRESS.md` and `results.md` join the skip list
  because they state or quote the rule.
- Tests: `tests/unit/test_v3_phase1_config.py` (config values, validator, wording check).

Choices made while unattended: the wording check already covers `README.md` and `docs/`, so the
five existing "the researcher" phrases there were replaced mechanically with "Carter" now (phases 4
and 5 rewrite those files anyway) to keep the gate green within this phase.

## v3 phase 2: shuffle scope (DECISION(G6)) and corrected projections

Note added 2026-11-08: the v2 phase 9 entry and the "remaining" list above say "about 16,600 shuffle
CPU hours". That figure assumed 52 weeks and all five strategies (Mean-CVaR alone was about 75
percent of it). Under DECISION(G6) and the 24-week study the corrected projection is 85.2 CPU hours
at 300 stocks and 5.6 at 100 (solve time only, 51 sources), measured below. The older text is
history and is left as written.

- `eval/shuffle.py`: `shuffle_portfolios(portfolios, strategies)` keeps only the strategies in
  `eval.shuffle.strategies` (their Q0 first, then their steered portfolios; controls never);
  `plan_shards`, `shuffle_test`, `portstudy shuffle plan/run/collect/run-local` all go through it,
  so Mean-CVaR and Resampled MVO are no longer shuffled. `StudyConfig` narrows the default list to
  the enabled strategies when `eval.shuffle.strategies` is not set (reduced test configs) and
  rejects an explicit entry that names a disabled strategy.
- `eval/report.py`: every excluded strategy section prints "No shuffled-ratings test for this
  strategy (excluded under DECISION(G6) for compute; see the Q0 comparison and factor regression)."
- `scripts/benchmark_pipeline.py`: `--weeks` defaults to 24 (pilot plus live), the projection
  counts only the configured shuffle strategies (each strategy's steered portfolios plus its Q0 at
  that strategy's own mean solve time, printed per strategy), `--replay-seconds` adds a second
  projection with one simulator replay per shuffle (without it the output says replay time is
  not included), and the warning fires only above the 15,000 CPU-hour request.
- Tests: `tests/unit/test_v3_phase2_shuffle.py` (filtered portfolio list, shard count, projection
  arithmetic), `test_v2_phase11_shuffle.py` updated to the G6 scope, and the e2e eval test asserts
  the two exclusion lines (cvar, resampled) in `report.md`.

Projection runs (this VM, `--sources 3 --weeks 24`, projection at 51 sources, 500 shuffles,
156 portfolios = 3 strategies x (51 sources + Q0); solves only, replay time not included):

| stocks | mvo mean solve | robust | topk | cvar (not shuffled) | resampled (not shuffled) | shuffle CPU hours | peak MB |
|---|---|---|---|---|---|---|---|
| 100 | 0.010 s | 0.022 s | 0.001 s | 7.3 s | 1.2 s | 5.6 (mvo 1.7, robust 3.7, topk 0.2) | 587 |
| 300 | 0.077 s | 0.413 s | 0.003 s | 22.1 s | 18.1 s | 85.2 (mvo 13.3, robust 71.5, topk 0.4) | 1301 |

Had Mean-CVaR and Resampled MVO stayed in scope the 300-stock total would be about 4,400 CPU hours
(52 x (22.1 + 18.1) x 24 x 500 / 3600 on top of 85), which is the roughly 99 percent that G6
removes. J7 at 300 stocks and 51 sources is about 2,117 s per week on this VM.

## v3 phase 3: ARCTIC jobs under the time limit (W1) and SU efficiency (M1)

- `J4_open_weight_infer` no longer waits for J2 (the LLM calls never read the Kronos output).
  Healthy open-weight models run side by side when `llm.parallel_models` is true (one thread per
  model, each with its own `WeekRunner` and `llm.concurrency` pool, writing only its own
  `<model>__<config>.jsonl` files; `run_status.json` writes take a shared lock). When one model's
  worker raises, the other models still finish and keep their records; J4 then fails naming the
  crashed model, so `run_weekend.py --from J4` resumes it from the cache (an authentication abort
  still marks that model's sources failed and lets the week proceed, review fix M2). `assess_run`
  keeps any model-wide failed mark (health check, abort) while the run is short of its tasks.
  `config/study_test.yaml` sets `parallel_models: false` so two test runs give identical digests.
- `J7_weights` now refuses to run until J6 is done and, unless J6 planned a hold, J2 is done and
  J4 is done or every open-weight source already carries its fallback mark
  (`_check_j7_prerequisites`).
- `scripts/run_weekend.py --poll-once`: one J5 pass; exit 3 while batches are pending before the
  lock margin, exit 0 when collected, and at the margin the run continues so J6 labels the pending
  sources (N1). `--check-walltime` rejects more than 15 hours (exit 2) and warns above 12.
- New `scripts/arctic/llm.sbatch` (J3 then J4; J4 only in login mode; 2 cores x 3G) and
  `scripts/arctic/finish.sbatch` (1 core x 3G, `FINISH_WALLTIME` 03:00:00: `--from J5 --poll-once`,
  resubmits itself with `--begin=now+${POLL_INTERVAL_MINUTES}minutes` on exit 3, otherwise runs J6
  to J9 in the same job). `submit_weekend.sh` submits kronos and llm at once with no dependency
  and finish with `--dependency=afterany:<kronos>:<llm>`; login mode runs J1 and J3 on the login
  node first and never polls batches there. `weekend.sbatch` is removed.
- `common.sh` `check_walltime` refuses any walltime above `MAX_JOB_HOURS` (15) and warns above 12;
  `arctic.env.example` adds `GPU_MEM`, `LLM_WALLTIME`, `FINISH_WALLTIME`, `POLL_INTERVAL_MINUTES`
  (60), `MAX_JOB_HOURS` and drops `WEEKEND_WALLTIME`. `shuffle_array.sbatch` uses one core and
  `--mem-per-cpu=3G`; GPU jobs take `--mem=${GPU_MEM}`.
- `portstudy shuffle plan` prints the estimated minutes per shard from the newest benchmark file
  (slowest shuffled strategy, one solve per week per shuffle) and lowers the shard size in the plan
  output when a shard would exceed `--max-shard-minutes` (120); the config is unchanged.
- Tests: `tests/unit/test_v3_phase3_arctic.py` (J4 with J2 pending, J7 refusals, parallel equals
  sequential, one crashing model, `--poll-once` exit codes, walltime limits, `common.sh` guard,
  three-job layout in both modes through stubbed `sbatch`, `finish.sbatch` resubmission). The
  review-fix H3 test is replaced (its login-mode wait no longer exists). shellcheck runs in
  `test_shellcheck_every_arctic_script` over the new files.

Runtime budget (DECISION(W1): every job at most 15 hours, target 12). Solve and J7 times are the
phase 2 benchmark on this VM; LLM times are the per-request figures in
`docs/proposals/ARCTIC_Resource_Request.tex` (10 GPU-s small models, 40 GPU-s large); Kronos is the
request's estimate. Largest configuration: 5 open-weight models x 7 prompts.

| Job | 100 stocks | 300 stocks | Basis |
|---|---|---|---|
| kronos.sbatch (J1, J2, GPU) | under 1 h | about 1 h | ARCTIC request, 1 GPU-hour per week |
| llm.sbatch (J3 then J4, 2 cores) | about 1.9 h | about 5.8 h | slowest model: 700 (2,100) requests x 40 s / concurrency 4, models in parallel |
| finish.sbatch (J5 to J9, 1 core) | about 0.2 h | about 0.7 h | J7 446 s (2,117 s) at 51 sources plus J8 and J9 (minutes) |
| one shuffle shard (25 shuffles x 24 weeks) | 0.2 min | 4.1 min | robust mean solve 0.022 s (0.413 s); replay time per shuffle not yet measured |

Every job is under 12 hours; `LLM_WALLTIME` 06:00:00 leaves headroom at 300 stocks only if the
gateway sustains about 20 requests in flight (PLAN2 phase 15 E confirms the real per-request time).

## v3 phase 4: README rewritten from the proposal

- `README.md` (285 lines) now follows the proposal in `docs/proposals/Project_Proposal_IEEE.tex`:
  project identity (Carter Tierney, MS Computer Science, Georgia State University, advisor
  Dr. Parag Tamhankar), the abstract adapted, the research question and H1 to H5 word for word,
  the hypothesis to test to Holm-family table, the study design (sources versus portfolios, the
  prompt treatments, the default and requested model sets, the portfolio count formulas, the
  output schema, the Q0/HRP/equal-weight controls, residualization and common random numbers,
  the frozen config wording without "pre-registered", why a live run and no backtest), the
  methodology (baseline, Ledoit-Wolf risk and prior, views and uncertainty, Black-Litterman,
  the five strategies and controls, guardrails and simulator), the statistics (tie band, sign
  test, exact sign-flip permutation, Newey-West, bootstrap, minimum detectable gap, Fama-French
  primary with the ETF fallback, G6 shuffle scope), the J1 to J9 pipeline, the three-job ARCTIC
  layout with failure handling, the compute table with the phase 3 runtime budget, quick start,
  commands, testing, repository layout, project status and documentation.
- The design-choice section keeps only K1, S4, S5, G3b, P1 and G6, W1, M1 in the plan's wording;
  the old "chosen during build" list moved unchanged to `docs/build_decisions.md`.
- Removed from the README: `config/universe.csv` (never existed in the repository), the 16,600
  CPU hour figure, "pre-registered", "thesis", "quantitative finance", "the researcher" and
  "the desktop". The root `Project_Proposal_IEEE.pdf` is a duplicate of
  `docs/proposals/Project_Proposal_IEEE.pdf` and stays untracked.

## v3 phase 5: documentation aligned with PLAN3

- `docs/design.md`: PLAN3 intro, the G6 shuffle scope and exclusion line in the H5 paragraph,
  Fama-French primary and ETF fallback, and a new "ARCTIC job layout" section.
- `docs/runbook.md`: benchmark command at 24 weeks with the three-strategy projections and the
  measured 5.6 / 85.2 CPU hours, G6 scope and shard-minute sizing in the shuffle section,
  `run-local` as the no-cluster fallback, the three-job layout in both network modes,
  `finish.sbatch` polling, the 15 hour rule with rerun guidance, `sacct` and `squeue` usage.
- `docs/config_reference.md`: `eval.shuffle.strategies`, `llm.parallel_models`, the new
  `arctic.env` variables (`GPU_MEM`, `LLM_WALLTIME`, `FINISH_WALLTIME`, `POLL_INTERVAL_MINUTES`,
  `MAX_JOB_HOURS`, `PORTSTUDY_FINISH_FLAGS`), the gateway URL, no MiniMax `max_output_tokens`
  override, "PLAN2 phase 15 B" instead of "v1 phase 17".
- `docs/open_questions.md`: G6 (changed), W1 and M1 (new) in the decision table.
- `docs/arctic_setup.md`: the known gateway URL with the GSU network or VPN requirement, W1 and
  M1 as the two rules behind every job, the new env variables, `llm.sbatch` and `finish.sbatch`
  in the script table, the new network layouts, and the PLAN2 phase 15 A check of the real
  walltime limit.
- `docs/github_setup.md` and `docs/gitlab_setup.md`: no "researcher" or "desktop" wording, the
  public-repository runner note (no minute budget), GitLab kept as the fallback host,
  `frozen-v2` and PLAN2 phase 15 H instead of `frozen-v1` and v1 phase 17.
- `docs/weekend_checklist.md` replaces `docs/desktop_checklist.md` (ARCTIC three-job flow,
  nothing stays open on a laptop); `docs/arctic_gpu_request.md` stays untracked for phase 15 D.
- `docs/proposals/` now holds the IEEE proposal and the ARCTIC resource request (PDF and LaTeX),
  committed unchanged. `PLAN2.md` gained the one line "Decisions G6 (changed), W1 and M1 (new)
  are updated in PLAN3.md." and `PLAN3.md` is committed. The dated note under v3 phase 2 above
  corrects the 16,600 CPU hour figure.

## v3 phase 6: final verification and push

- `pytest` (unit, integration and e2e; slow excluded): 361 passed (344 unit and integration, 17 e2e) in 27 min on this VM;
  `pytest tests/e2e`: 17 passed in 18 min. `ruff check`, `ruff format --check`, `mypy src
  scripts`, `refresh_manifest.py --check`, `check_placebo_length.py`, `check_banned_words.py`,
  `check_secrets.py` and `yamllint` all clean.
- Consistency greps (nothing found): "the researcher", "the student", "Phase 17", "the desktop" in
  `README.md`, `docs/`, `src/`, `scripts/`, `config/`, `tests/` (the checker and its test excepted);
  "thesis", "pre-registered", "quantitative finance" in `README.md` and `docs/`; "16,600" outside
  `PROGRESS.md`, `results.md` and the plans; `max_output_tokens: 2048` anywhere. Found as expected:
  the gateway URL in `config/study.yaml`, `docs/arctic_setup.md`, `docs/config_reference.md` and
  `docs/weekend_checklist.md`; the README title, research question, H1 to H5 rows and the sign-flip,
  Newey-West, Holm and bootstrap wording; `strategies: [mvo, robust, topk]` in both study configs;
  `--mem-per-cpu=3G` on `llm.sbatch`, `finish.sbatch`, `shuffle_array.sbatch` and the two sbatch
  lines in `submit_weekend.sh`; `--cpus-per-task=1` on `finish.sbatch` and `shuffle_array.sbatch`;
  `MAX_JOB_HOURS` (15) in `common.sh` and `arctic.env.example`.
- `bash scripts/arctic/submit_weekend.sh --week 2026-W05 --dry-run` with a filled test env file,
  exit 0 in both modes. Compute mode: `sbatch ... --gres=gpu:1 --mem=32G --time=01:00:00
  kronos.sbatch`, `sbatch ... --cpus-per-task=2 --mem-per-cpu=3G --time=06:00:00 llm.sbatch`,
  `sbatch ... --cpus-per-task=1 --mem-per-cpu=3G --time=03:00:00 --dependency=afterany:<kronos>:<llm>
  finish.sbatch`. Login mode: `portstudy run-job J1` and `run_weekend.py --from J3 --until J3` on the
  login node first, then the same three sbatch lines with the LLM job starting at J4. The
  `--check-walltime` warning about the 2026-W05 lock deadline is expected for a past week.
- Pushed as ctsc (author and committer) to GitHub `origin master` first and GitLab `gitlab master`
  second; the GitHub Actions run is checked after the push and reported in the session.
- Not done, by design: PLAN3 phase 7 (Carter's manual work) and PLAN2 phase 15.

## Manual work consolidated

- `manualwork.md` now holds every remaining manual step from `PLAN.md` Phase 17, `PLAN2.md` Phase 15
  and `PLAN3.md` Phase 7 (each plan section is replaced by a pointer), keeping the Phase 15 labels
  A to H that the README, this file and `docs/` cite; items already done (A1, A2 for the builds,
  D1, the checklist half of D4, PLAN3 phase 7 item 4) are noted at the top and dropped from the
  list, and the UNVERIFIED list carries the corrected G6 projection.
