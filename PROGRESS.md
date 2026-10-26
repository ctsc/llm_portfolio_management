# PROGRESS.md: agent build log

Each entry: what was built, tests passing, anything skipped and why.

## Phase 1: Scaffold, config, and utilities

Built:
- `pyproject.toml` with dependency groups `ml`, `api`, `data`, `dev`; core installs without extras.
- `config/study.yaml` (every section and every decision from devin.md 0.3) and `config/models.yaml` (model registry with stub models for tests).
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
