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
- Derived view certainty from width using DECISION(E1), retained the Phase 6 view-builder interface, and updated optimizer weight output.
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
- Phase 3 was not started, as requested.
- No real services or paid APIs were called.

Choices made while unattended:
- Stub half-width is `max(1, ceil(abs(noise)))`, so synthetic uncertainty follows the sampled noise while retaining a nonzero interval.
- Width-derived certainty is applied at the optimizer boundary; the existing view-builder interface remains until Phase 6.
- The `endpoint` output field is empty because `RawResult` does not yet expose a resolved serving endpoint; the column is retained for the later serving implementation.
