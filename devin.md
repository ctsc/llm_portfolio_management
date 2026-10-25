# devin.md: AI Build Plan for the LLM Skills and Investor Mindsets Study

This file tells an AI coding agent how to build the entire codebase for the study, start to finish, with no human input until the manual section at the end (Phase 17).

Study in one sentence: with a quantitative portfolio optimizer and guardrails held fixed, test whether an LLM given specialized skills files can beat or improve a quant baseline, using a weekly pipeline simulated on live data.

---

## 0. Rules for the agent (read first, follow always)

### 0.1 How to work
1. Work through the phases in order. Do not start phase N+1 until phase N's "Done when" checks pass.
2. After each phase: run the full test suite, commit with a message `phase N: <title>`, and append a short entry to `PROGRESS.md` (what was built, tests passing, anything skipped and why).
3. If a test fails, fix the code. Never weaken or delete a test to make it pass.
4. Everything must run offline. Use synthetic data, recorded fixtures, and the stub LLM. Never call a real paid API, a real broker, or require a GPU while building. Real-service adapters are written against official SDK or HTTP docs and tested with mocks only.
5. If a detail of an external API is uncertain, read its current official documentation first. If you cannot verify it, implement it behind the interface, mark it with `# UNVERIFIED: needs real-key smoke test (Phase 17)`, and list it in `PROGRESS.md`. Do not guess endpoint names or parameters silently.
6. Keep modules small and pure where possible: inputs in, outputs out, no hidden global state, no hidden randomness (every random call takes an explicit seed from config).
7. Every job and function that writes files must be idempotent: running it twice gives the same result and never double counts.

### 0.2 Style rules
- Python 3.11. Type hints everywhere. `ruff` and `mypy` clean.
- Never use em dashes in code, comments, docstrings, docs, or commit messages. Use commas, colons, or parentheses.
- Never use the word "arm" for experimental conditions. Say "LLM view", "quant baseline view", "portfolio", or "configuration".
- Write short comments above each logical block, not walls of prose.
- No emoji.

### 0.3 Study decisions (set by the researcher)
Each decision below is a config value in `config/study.yaml`, tagged in code with `DECISION(<ID>)`. Never hardcode them. Rows marked `provisional` may change after the real prompt test runs; everything else is confirmed. Never change any of them yourself.

| ID | Setting | Value | Status |
|---|---|---|---|
| D1 | models | local: Qwen 3.5 4B (thinking off), Phi-4 Mini 3.8B, Llama 3.2 3B as a fallback only; API: Claude Haiku 5.5, GPT-6 Luna (reasoning none) | confirmed |
| D3 | budget | goal $50; warn at $50 and require an explicit `--allow-over-budget` flag to go past it; recomputed after real prompt test runs | provisional |
| D4 | temperature | 0 for all models | confirmed |
| B1 | prompt configurations | `bare` (bare prompt, no skills file), `skills` (one skills file), `placebo` (length-matched neutral text); more skills files may be added later as new configurations | confirmed |
| B2 | stock list | sector balanced, drawn from the S&P 500 (see Phase 2, `universe.py`) | confirmed |
| B4 | universe size | 100 | confirmed |
| A2 | start | first live week is the week after the pilot passes | confirmed |
| A3 | live length | runs until `timing.live_end_date`, set by the researcher, no later than the April deadline | confirmed |
| A6 | cold start | first 5 live weeks use fixed IC 0.02 | confirmed |
| C1 | prompt fields | ticker, company name, sector, news | confirmed |
| C2 | stock with no news | still call the model with the news text "No relevant news is available for this week." No web search or tools for any model. | confirmed |
| C4 | skills file token cap | 6000 | provisional |
| C7 | rationale token cap | 150 | provisional |
| E2 | IC used to scale the LLM view | IC of the residualized signal | confirmed |
| E3 | combined score for H2 | Black-Litterman tilt (mu_BL minus pi), ranked | confirmed |
| E4 | Kronos input prices | adjusted OHLCV | confirmed |
| E5 | market-cap weights | over the study's stocks only, renormalized | confirmed |
| G3 | real H3 effect | paired weekly test passes after Holm, placebo as control | confirmed |
| G4 | H4 splits | market up/down, volatility, earnings week (see Phase 11) | confirmed |
| G5 | factor regression | daily returns | confirmed |
| G6 | shuffled-ratings test | all portfolios (every model and configuration) plus Q0, 500 shuffles each | confirmed |
| N1 | failure rule | invalid-output rate above 0.15, or local job not finished by the weight-lock deadline (a crash that resumes and finishes in time is not a failure), or batch not ready by the weight-lock deadline | confirmed |
| N2 | weekly trigger | the researcher runs one script by hand Friday night or Saturday morning; no scheduler | confirmed |
| N3 | storage and backup | all data as CSV files; the researcher downloads the dated CSV folders weekly | confirmed |

### 0.4 Confirmed decisions (hardcode only as config defaults, but treat as fixed design)
- Q0 is mandatory: the same pipeline with only the quant baseline view and no LLM view.
- One stock per model call. Never batch several stocks into one prompt.
- A mindset is a skills file. No stacking of layers.
- Models and code never write or edit prompt or skills files. Only a human changes them, via a commit.
- Output schema: integer score from -5 to 5, confidence from 0 to 1, rationale with a token cap.
- Stop-loss and take-profit are checked at every daily close.
- The pilot week never counts toward results.
- A removed stock is dropped without replacement and logged.
- If an LLM portfolio fails for a week, it trades Q0 weights, labeled as a fallback, and that week is excluded from that portfolio's tests.
- A null result is acceptable. Never tune the evaluation toward a positive result.
- Local models run on the desktop GPU. API models run through batch endpoints with prompt caching.
- GitLab is the source control host and runs lint and tests in CI. It does not run the weekly pipeline.
- All stored data is CSV (UTF-8, header row, ISO dates, floats written with `float_format="%.12g"` so reruns are byte-identical).
- All times are US Eastern (`America/New_York`), stored with an explicit offset (for example `2026-11-13T16:00:00-05:00`). Never use the desktop's local clock without converting. Provider timestamps in UTC are converted to ET on ingest.
- Week labels: `week_id` is the ISO week of the cutoff Friday. Scores made at that cutoff trade at the next Monday open, and their return window runs from that Monday open to the following Monday open. Every table that holds returns stores `week_id`, `trade_date`, and `window_end_date` explicitly, so no code ever infers the alignment.
- API models are pinned to a dated model snapshot ID where the provider offers one, and the provider's returned model version is logged on every call. If the returned version ever changes mid-study, the run logs a warning and the change is flagged in the report.
- A stock is dropped only if it stops trading or its data fails the checks. Leaving or joining the S&P 500 mid-run changes nothing.

---

## 1. Target architecture

```
repo/
  devin.md                    this file
  PROGRESS.md                 agent progress log
  README.md                   how to run everything
  pyproject.toml
  .gitlab-ci.yml
  config/
    study.yaml                every setting and every deferred decision
    models.yaml               model registry
    prompts_manifest.yaml     prompt files, versions, hashes
  prompts/
    base_instructions.md      shared instructions and JSON schema text
    skills/                   HUMAN-WRITTEN skills files (agent writes placeholders only)
    placebo/                  placebo file (placeholder generated to match length)
  src/portstudy/
    config.py  paths.py  io.py  hashing.py  logging_setup.py  calendar.py  types.py
    data/        providers/ (base.py, synthetic.py, alpaca.py, yfinance_provider.py, finnhub.py, fred.py)
                 universe.py  snapshot.py  checks.py
    baseline/    momentum.py  kronos.py  combine.py
    risk/        covariance.py  prior.py
    llm/         schema.py  prompts.py  cache.py  cost.py  validate.py  runner.py
                 providers/ (base.py, stub.py, ollama.py, anthropic_batch.py, openai_batch.py)
    views/       ic.py  residualize.py  build.py
    optimize/    black_litterman.py  mean_variance.py
    sim/         simulator.py  records.py
    pipeline/    status.py  jobs.py  fallback.py  cli.py
    eval/        rank_ic.py  hypotheses.py  tests.py  factors.py  shuffle.py  report.py
    backup.py
  scripts/       make_synthetic_world.py  run_weekend.py  smoke_real_services.py
  tests/         unit/  integration/  e2e/
  docs/          runbook.md  config_reference.md  open_questions.md
  data/          (gitignored) ledger/ snapshots/ outputs/ logs/ portfolios/ backups/ exports/
```

Libraries: pandas, numpy, scipy, scikit-learn, statsmodels, cvxpy (OSQP and CLARABEL solvers), pydantic v2, pyyaml, typer, httpx, tenacity, pytest, pytest-cov, hypothesis (property tests), ruff, mypy. Optional extras (installed only when present): torch and transformers for Kronos, the Anthropic and OpenAI SDKs, alpaca-py, yfinance, finnhub-python. The core must import and test without the optional extras.

---

## Phase 1: Scaffold, config, and utilities

**Goal:** a repo that installs, lints, and runs an empty test suite, with typed config.

**Build:**
1. `pyproject.toml` with the dependency groups: `core`, `ml` (torch, transformers), `api` (SDKs), `data` (alpaca-py, yfinance, finnhub-python), `dev`.
2. `config/study.yaml` with sections: `universe`, `timing`, `baseline`, `risk`, `llm`, `views`, `optimizer`, `simulator`, `pipeline`, `eval`, `backup`, `decisions`. Every value from section 0.3 and the design constants below appear here.
3. `src/portstudy/config.py`: pydantic models that load and validate `study.yaml`; unknown keys are an error; every `decisions.*` entry has `id`, `value`, `status` (`confirmed` or `provisional`).
4. `paths.py`: one function per data folder, rooted at `PORTSTUDY_DATA_DIR` (default `./data`).
5. `hashing.py`: `sha256_file(path)`, `sha256_text(str)`, `stable_hash(obj)` (canonical JSON).
6. `io.py`: `write_csv_atomic`, `read_csv`, `write_json_atomic`, `read_json`. Atomic means write to a temp file then rename.
7. `logging_setup.py`: JSON-lines logger writing to `data/logs/<job>/<date>.jsonl`.
8. `types.py`: dataclasses or pydantic models for `WeekId`, `Snapshot`, `LlmResponse`, `PortfolioId`, `Trade`, `DailyRecord`.
9. `calendar.py`:
   - `trading_days(start, end)` using `pandas_market_calendars` if installed, otherwise a built-in NYSE holiday table for 2026 and 2027 (include early closes).
   - `cutoff_day(week_friday)`: the Friday, or if closed the last prior trading day (decision A4, confirmed).
   - `monday_open_day(week)`: the Monday, or the next trading day if Monday is a holiday.
   - `week_id(date)` like `2026-W46`.

**Design constants to put in config (all editable):**
```
timing.cutoff_time_et: "16:00"
risk.cov_lookback_days: 252
risk.delta: 2.5
views.tau: 0.05
views.ic_window_weeks: 8
views.ic_floor: 0.0
views.cold_start_weeks: 5
views.cold_start_ic: 0.02
views.ic_min_weeks: 3
views.confidence_clip: [0.1, 1.0]
optimizer.max_weight: 0.05
simulator.initial_value: 100000
simulator.cost_bps: 10
simulator.stop_loss: -0.08
simulator.take_profit: 0.15
eval.tie_band: 0.01
```

**Tests:** config loads and rejects unknown keys; atomic writes survive a simulated crash; calendar handles Good Friday 2027, Christmas, New Year (Fridays closed), Thanksgiving half day, and a Monday holiday.

**Done when:** `pip install -e .[dev]`, `ruff check`, `mypy src`, `pytest tests/unit` all pass.

---

## Phase 2: Synthetic world and data layer

**Goal:** a complete fake market the whole pipeline can run on offline, plus the real data adapters behind the same interface.

**Build:**
1. `data/providers/base.py`: abstract `PriceProvider`, `NewsProvider`, `ReferenceProvider` (sectors, market caps, earnings dates), `RateProvider` (T-bill), each returning plain DataFrames with fixed column names:
   - prices: `date, ticker, open, high, low, close, adj_close, volume`
   - news: `ticker, published_at (UTC), headline, summary, source`
   - reference: `ticker, name, sector, market_cap, next_earnings_date`
   - rates: `date, rate`
2. `data/providers/synthetic.py`: a seeded generator:
   - N tickers (default 120), 400 trading days of history plus a configurable number of weeks into the future.
   - Prices from a factor model: market factor plus sector factors plus idiosyncratic noise, with a weak planted momentum effect and a weak planted "LLM-detectable" effect hidden in the news text (for example headline sentiment words correlated with next-week returns), so that the stub LLM and the evaluation can later recover a known effect.
   - Includes a split, a delisting, a halt (missing days), an extreme move, and stocks with no news in some weeks.
   - News with UTC timestamps, some after the Friday cutoff (to test leakage).
   - Sectors, market caps, earnings dates, a T-bill series, and ETF proxies (SPY, IWM, IWD, IWF, QUAL).
3. `scripts/make_synthetic_world.py`: writes the world to `data/synthetic/` so tests can reuse it.
4. Real adapters (mock-tested, `UNVERIFIED` until Phase 17): `alpaca.py` (bars and news), `yfinance_provider.py` (prices, market cap, sectors fallback), `finnhub.py` (earnings dates), `fred.py` (DTB3). Each has retries with backoff and respects rate limits from config. Add a fixture-replay mode that reads recorded JSON instead of the network.
5. `data/checks.py`: functions that return a list of `Issue(severity, ticker, code, message)`:
   missing days, zero or negative prices, absolute daily return above 40 percent without a split flag, duplicate rows, stale prices (unchanged for 5 days), stocks with no news, news published after the cutoff, tickers with fewer than 300 days of history.
   Each issue code maps to a rule in config: `skip_stock`, `forward_fill`, or `halt`. Log every use.
6. `data/snapshot.py`: `make_snapshot(cutoff_date, config, providers) -> SnapshotRef`:
   - Pull prices (at least 300 trading days back), news in `(previous_cutoff, cutoff]` (for the first snapshot, the 7 calendar days before the cutoff), reference data, rates, ETF prices.
   - Drop news published after the cutoff time (strictly).
   - Deduplicate near-identical articles (normalized headline match), cap news per stock at the configured token budget (newest first; use a simple `len(text)/4` token estimate unless a tokenizer is configured).
   - Record `article_count` per stock.
   - Apply the check rules, write to `data/snapshots/<week_id>/` as CSV files plus `manifest.json` (file hashes, provider names, retrieval time).
   - Immutable: if the folder exists, refuse to overwrite unless `--force-new-version`, which writes `<week_id>.v2`.
   - Append every new trading day's raw open, high, low, close and any dividends or splits to the price ledger (`data/ledger/`, see Phase 9). Days already in the ledger are never rewritten.
7. `read_snapshot(week_id)` returns typed access to the snapshot.
8. `data/universe.py` (decision B2): `build_universe(as_of, config, providers) -> DataFrame`:
   - Source list: current S&P 500 members with GICS sectors (Wikipedia table adapter, mock-tested).
   - Rank members within each of the 11 GICS sectors by 60-day average dollar volume (close times volume).
   - Take the top `floor(100 / 11) = 9` per sector (99 stocks), then add the single highest-volume stock left over, for 100.
   - Drop any stock with fewer than 300 trading days of history before ranking.
   - `universe.extra_tickers` (default empty) lets the researcher add a few non-S&P 500 stocks by hand; each one replaces the lowest-volume pick in its sector, so sectors stay balanced.
   - Writes `config/universe.csv` (`ticker, name, sector, rank_in_sector, avg_dollar_volume, version`). Once frozen at the pilot, the file is never rebuilt automatically.
   - Prints a sector count table so the researcher can check the balance.
9. Earnings flag: for every stock and week, `earnings_in_window = True` if a reported or scheduled earnings date falls inside the Monday-open to next-Monday-open return window. Stored in the snapshot for H4 and reporting.

**Tests:**
- Snapshot of a past Friday contains no row dated after the cutoff.
- Re-running gives byte-identical CSV files and manifest hashes.
- A closed Friday (Good Friday) uses Thursday.
- A stock removed from the universe mid-run is dropped and logged, not replaced.
- Each check fires on the planted problem in the synthetic world.
- `build_universe` gives exactly 100 stocks, 9 or 10 per sector, and is deterministic.

**Done when:** `make_snapshot` builds 30 consecutive weekly snapshots from the synthetic world with zero unexpected issues.

---

## Phase 3: Quant baseline

**Goal:** `z_b` for every stock every week.

**Build:**
1. `baseline/momentum.py`, using adjusted close:
   ```
   M12_1 = P[t-21] / P[t-252] - 1
   M3    = P[t]    / P[t-63]  - 1
   R1    = -(P[t]  / P[t-21]  - 1)
   u(x)  = rank(x, method="average") / N      (ranks across stocks, ascending)
   m     = mean(u(M12_1), u(M3), u(R1))
   ```
   Here `t` is the cutoff day index. Stocks missing any of the 252-day history are excluded for the week and logged.
2. `baseline/kronos.py`:
   - `ForecastProvider` interface: `forecast_return(ohlcv_by_ticker, horizon=5, n_samples=20, seed) -> Series k`.
   - `StubForecaster`: deterministic drift plus noise, seeded, so tests need no model.
   - `KronosForecaster`: wraps the Kronos code (clone of `shiyu-coder/Kronos`, vendored at a pinned commit in `vendor/kronos`) with `Kronos-base` weights and `Kronos-Tokenizer-base` from Hugging Face (`NeoQuasar`). Use the last 252 daily candles (the model limit is 512), predict the next 5 candles, `T=1.0`, `top_p=0.9`, 20 sampled paths averaged, fixed seed, use the batch predict method if present. `k_i = mean(P_hat[t+5] / P[t]) - 1`. Mark `UNVERIFIED` until the real run. Free GPU memory after use.
   - Provider chosen by config `baseline.forecaster: stub | kronos`.
3. `baseline/combine.py`:
   ```
   b   = 0.5 * u(m) + 0.5 * u(k)
   z_b = (b - mean(b)) / std(b, ddof=1)
   ```
   Output a DataFrame `ticker, M12_1, M3, R1, m, k, b, z_b` saved to `data/outputs/<week>/baseline.csv`.

**Tests:** hand-computed momentum on a toy price table; ranks with ties; `z_b` has mean 0 and std 1; same input gives same output; baseline rank IC on the synthetic world is positive (planted effect).

**Done when:** baseline builds for all synthetic weeks, deterministic across runs.

---

## Phase 4: Risk model and market prior

**Goal:** `Sigma`, `sigma_i`, `pi` for each week.

**Build:**
1. `risk/covariance.py`:
   - Daily simple returns from adjusted close for the last 252 trading days.
   - `sklearn.covariance.LedoitWolf` fit; this shrinks toward `tr(S)/N * I` with a data-driven alpha. Store `alpha` (`shrinkage_`).
   - Weekly covariance `Sigma = 5 * Sigma_daily`. `sigma_i = sqrt(diag(Sigma))`.
   - Exclude stocks with incomplete history, consistent with Phase 3's stock list. Both modules must use the same final ticker set (`common_universe(snapshot)` helper).
   - Assert symmetric and positive definite (smallest eigenvalue greater than 1e-10), else raise.
2. `risk/prior.py` (decision E5 default):
   ```
   w_mkt = market_cap / sum(market_cap)     over the study tickers only
   pi    = delta * Sigma @ w_mkt            delta = 2.5
   ```

**Tests:** known 3-stock toy case; Sigma positive definite even when stocks exceed days (use a synthetic case with N=300, T=252); `w_mkt` sums to 1; `pi` scales linearly with delta.

**Done when:** `Sigma`, `sigma`, `pi` saved per week and tests pass.

---

## Phase 5: Prompt system

**Goal:** frozen, versioned, hashable prompts assembled identically for every model.

**Build:**
1. `prompts/base_instructions.md`: short shared instructions plus the exact JSON output schema:
   ```
   {"score": <integer -5..5>, "confidence": <number 0..1>, "rationale": "<short reason>"}
   ```
   Respond with JSON only.
2. `prompts/skills/` and `prompts/placebo/`: the agent writes only clearly labeled placeholder files (one `skills_PLACEHOLDER_v0.md` and one `placebo_PLACEHOLDER_v0.md`) of a plausible structure, each starting with a banner `PLACEHOLDER: the researcher writes the real file (Phase 17)`. The real skills file is human work. Never generate investing advice text intended as the final skills. The `bare` configuration (bare prompt) uses no file: its shared prefix is the base instructions only. The design must allow more skills files to be added later as new configurations with no code change.
3. `config/prompts_manifest.yaml`: list of configurations, each with `id`, `file`, `kind` (`bare`, `skills`, `placebo`), `version`, `sha256`, `enabled`. A script `scripts/refresh_manifest.py` recomputes hashes and fails CI if a file changed without a version bump.
4. `llm/prompts.py`:
   - `assemble_prompt(config_id, stock_row, news_text) -> AssembledPrompt(shared_prefix, stock_part, full_text, prefix_hash, prompt_hash)`.
   - Order is fixed: shared prefix first (base instructions plus the configuration's file), then the stock part (ticker, company name, sector, news; decision C1), so provider caching works. No price statistics anywhere in the prompt.
   - If the stock has no news in the window, the news text is exactly the configured no-news sentence (decision C2).
   - `token_count(text, model_id)` abstraction: use `tiktoken` for OpenAI-like, provider counting functions behind an interface for others (mock in tests), and a `len/4` fallback.
5. `scripts/check_placebo_length.py`: asserts the placebo token count is within 2 percent of the skills file it matches, measured with every enabled model's tokenizer (all must pass), and that all skills files are under the cap (`C4`, default 6000 tokens) for every tokenizer.

**Tests:** the shared prefix is byte-identical across stocks; the prefix hash changes when the file changes; the no-news sentence is used; the order is prefix then stock; the manifest check fails on an unversioned edit.

**Done when:** prompts assemble for every enabled configuration and every stock in the synthetic snapshot.

---

## Phase 6: LLM runner

**Goal:** a score and confidence for every stock, model, and configuration, with logging, caching, cost guard, and a failure signal.

**Build:**
1. `llm/schema.py`: `LlmResponse` with `score:int`, `confidence:float`, `rationale:str`, plus `valid:bool`, `raw:str`, `error:str|None`.
2. `llm/validate.py`: `parse_response(raw) -> LlmResponse`. Valid only if JSON parses, `score` is an integer in `[-5,5]`, `confidence` in `[0,1]`, rationale is a string (truncate to the token cap). Strip code fences. Anything else is invalid.
3. Invalid-output guardrail: retry once; if still invalid, store `score=0, confidence=0.1, fallback_flag="invalid_output"` for the record. Count rates per (model, configuration, week). Invalid stocks never become a view: residualization and views skip them (Phase 7). Log a truncated JSON answer (hit the output token limit) as `invalid_truncated`, separate from other invalid output.
4. `llm/providers/base.py`: interface
   ```
   class LlmProvider:
       model_id: str
       supports_batch: bool
       def complete(self, prompt: AssembledPrompt, settings) -> RawResult
       def submit_batch(self, items: list[BatchItem], settings) -> BatchHandle
       def poll_batch(self, handle) -> BatchStatus
       def collect_batch(self, handle) -> list[RawResult]
   ```
   `RawResult` carries text, input tokens, output tokens, cached tokens, latency, provider model version string.
5. `llm/providers/stub.py` (default for all tests): deterministic function of (ticker, week, config, seed) that reads the planted effect from the synthetic news text and returns a plausible integer score and confidence, with configurable noise per configuration (skills configuration slightly better than bare; placebo equal to bare). Supports configurable failure injection: invalid JSON rate, timeout, batch never returns, crash.
6. `llm/providers/ollama.py`: HTTP to the local Ollama server. Settings: `temperature=0`, fixed `seed`, JSON output format, `num_ctx` large enough for the longest prompt, thinking disabled for Qwen (verify the current parameter name in the Ollama docs). Calls are sequential and saved after each call (resumable). Mark `UNVERIFIED`.
7. `llm/providers/anthropic_batch.py` (Claude Haiku 5.5) and `openai_batch.py` (GPT-6 Luna): use the official SDKs and each provider's batch endpoint, with prompt caching on the shared prefix (Anthropic needs explicit cache markers; check the OpenAI docs for automatic or explicit caching), temperature 0, no tools and no web search enabled for any model, GPT-6 Luna reasoning set to none, and a record of any setting the provider ignores. Build requests from `BatchItem`s, poll, collect, and match results by custom id. Mark every call `UNVERIFIED` and cover the logic with HTTP-mocked tests that replay recorded-shape responses you write by hand from the official docs.
8. `llm/cache.py`: key `sha256(model_version, prompt_hash, ticker, week_id, settings)`; store raw results on disk; `get` and `put`. A cache hit costs nothing and is logged.
9. `llm/cost.py`:
   - `CostTable` from `config/models.yaml` (per-million-token prices for input, cached input, output, batch discount). Prices are config values the researcher confirms in Phase 17.
   - `estimate_week_cost(...)` and `RunningCost` log. A guard raises `CostCapExceeded` before submitting any batch if the projected total for the study would pass `llm.cost_goal_usd` (50). The run continues past it only when the CLI gets `--allow-over-budget`, and every such week is logged (decision D3).
   - `scripts/estimate_cost.py`: reads real token counts from logged runs (the prompt test runs in Phase 17) and prints the projected study cost per model and configuration, with and without caching.
10. `llm/runner.py`:
    - `plan_week(week_id) -> list[Task]` where a task is (model, configuration, ticker). Disabled models and configurations are skipped.
    - `run_local(week_id, model)`: resumable, groups tasks by configuration, skips cached.
    - `submit_api(week_id, model)`, `collect_api(week_id, model)`.
    - Output: `data/outputs/<week>/llm_scores.csv` with columns `model, config_id, ticker, score, confidence, rationale, valid, fallback_flag, tokens_in, tokens_cached, tokens_out, cost_usd, latency_s, model_version, prompt_hash, prompt_file_sha256`.
    - `assess_run(week_id, model, config_id) -> RunStatus` returning `ok` or `failed` with a reason using the N1 rules from config (invalid rate above 0.15, local job not finished by the weight-lock deadline, batch not ready by the weight-lock deadline; a local crash that resumes and finishes in time is not a failure). Write it to `data/outputs/<week>/run_status.json`.
11. One config-driven setting `llm.settings.<model>`: temperature, seed, max output tokens (default 300, raise it if test runs show `invalid_truncated`), reasoning mode.

**Tests:**
- Stub run covers every (model, config, ticker) exactly once, and a second run is fully cached.
- Invalid-output injection: fallback values written, rates counted, retry used once.
- Crash injection in `run_local`: second invocation resumes and finishes without repeating completed calls.
- Cost guard blocks submission when the projection exceeds the goal, and `--allow-over-budget` lets it through with a log entry.
- Provider payloads contain no tools or web search options (mock inspects the payload).
- Prompt order and shared prefix survive into the provider request (mock inspects the payload).
- The runner never writes into `prompts/`.

**Done when:** a full synthetic week runs for all enabled models with the stub provider and every artifact above is produced.

---

## Phase 7: Rank IC, residualization, and views

**Goal:** the residualized LLM signal and both views, plus a weekly IC history that feeds scaling.

**Build:**
1. `views/ic.py`:
   - `weekly_returns(ledger, week_id)`: for scores made at the cutoff of `week_id`, the return from the next Monday open (`trade_date`) to the following Monday open (`window_end_date`), from raw ledger prices with dividends added back (total return), per stock.
   - `rank_ic(scores, returns) -> float`: Spearman correlation (average ranks for ties); return NaN if fewer than 10 stocks or zero variance.
   - `ICStore`: a CSV table `week, signal, model, config_id, rank_ic, n_stocks`. A signal is `baseline`, `llm_raw`, `llm_resid`, `combined`.
   - `trailing_ic(signal, model, config_id, as_of_week)`: mean of the last 8 scored weeks before `as_of_week`, floored at 0. Use only weeks whose return window closed before the cutoff (this builds in the one-week lag). Skip weeks flagged as fallback for that portfolio (N1). During the first `cold_start_weeks` live weeks (counting from the first live week, not the pilot) return the fixed `cold_start_ic`. After that, average the clean weeks available (up to 8); if fewer than `views.ic_min_weeks` (3) clean weeks exist, keep returning `cold_start_ic`.
2. `views/residualize.py` (E2 default uses the residual signal's own IC):
   - Cross-sectional OLS of the LLM score on `z_b` with an intercept; the residual is standardized with `ddof=1`. If the residual std is below 1e-9 return zeros.
   - **Invalid outputs are excluded:** stocks with `fallback_flag = invalid_output` are left out of the regression and get no `z_L` value. Their stored score of 0 must never be residualized, because `0 - a - c z_b` would turn a broken output into a view against the baseline.
3. `views/build.py`:
   ```
   q_b = pi + IC_b * sigma * z_b
   q_L = pi + IC_L * sigma * z_L
   Omega_b = tau * diag(Sigma)                    (a vector, one entry per stock)
   Omega_L = tau * diag(Sigma) / clip(conf, 0.1, 1.0)
   ```
   `IC_b` is the trailing IC of the baseline; `IC_L` is the trailing IC of that model and configuration's residualized signal. Output a `Views` object with `q_b, q_L, Omega_b, Omega_L`, plus `use_llm_view: bool`.
   - **Per-stock rule for invalid outputs:** for a stock with no valid LLM output this week, drop that stock's LLM view row, so it keeps only the baseline view. The other stocks keep their LLM view rows. (P becomes the identity stacked with the rows of I for valid stocks only.)
   - **Zero-IC rule:** if `IC_L == 0` (after the floor), set `use_llm_view = False`. The LLM view rows are dropped entirely, so that portfolio's weights equal the Q0 weights for that week. A view of `q_L = pi` is never passed to Black-Litterman, because it would act as a confident vote for no tilt and shrink the baseline view. Log each such week as `llm_view_dropped` (it is a normal week, not a fallback, and stays in the tests).
   - The same rule applies to the baseline view: if `IC_b == 0`, Q0 holds the market prior weights from mean-variance on `pi` alone.

**Tests:**
- Spearman matches `scipy.stats.spearmanr`.
- Residual is uncorrelated with `z_b` (correlation near zero).
- Zero IC gives views equal to `pi`.
- `IC_L == 0` gives `use_llm_view = False`, and the portfolio's weights equal Q0's weights exactly.
- An invalid stock is excluded from the regression and has no LLM view row; if every stock is invalid, the weights equal Q0's.
- Cold start rule: weeks 1 to 5 use 0.02, later weeks use the trailing mean, and fewer than 3 clean weeks keeps 0.02.
- A week whose return window has not closed is never used in `trailing_ic` (look-ahead guard test).

**Done when:** views build for every enabled portfolio over the synthetic run, and the look-ahead guard test passes.

---

## Phase 8: Black-Litterman and mean-variance

**Goal:** target weights for every LLM portfolio and for Q0.

**Build:**
1. `optimize/black_litterman.py`:
   ```
   P      = [I; I]                        (2N x N)
   q      = [q_b ; q_L]
   Omega  = diag([Omega_b ; Omega_L])
   A      = inv(tau*Sigma) + P' inv(Omega) P
   b      = inv(tau*Sigma) pi + P' inv(Omega) q
   mu_BL  = solve(A, b)
   ```
   Use `np.linalg.solve` and Cholesky where possible; never explicit inverses of full matrices when avoidable (`inv(Omega)` is diagonal, so use reciprocals).
   `black_litterman_q0(...)`: the same with `P = I`, `q = q_b`, `Omega = Omega_b`. Any portfolio with `use_llm_view = False` calls this function, so it matches Q0 exactly.
   Return `mu_BL` and the tilt `mu_BL - pi`.
2. `optimize/mean_variance.py`: with cvxpy,
   ```
   maximize  w' mu - (delta/2) w' Sigma w
   s.t.      sum(w) = 1,  0 <= w <= 0.05
   ```
   Use `cp.quad_form` with a symmetrized `Sigma` (or a Cholesky factor). Try OSQP, then CLARABEL; on failure return the previous week's weights with `fallback_flag="solver_failed"`. Assert at least 20 stocks are available (the cap needs it).
3. `optimize/__init__.py`: `build_weights(week_id, model, config_id) -> Weights` and `build_q0_weights(week_id)`.

**Tests:**
- No views (`IC = 0`) gives `mu_BL = pi`.
- A very confident LLM view moves `mu_BL` toward `q_L` for that stock.
- Weights sum to 1 within 1e-8, are in `[0, 0.05]`, and are deterministic.
- Equivalence: BL posterior matches an independent textbook formula implementation on a random case.
- Solver failure injection returns the prior weights with the flag.

**Done when:** weights exist for every enabled portfolio plus Q0 for every synthetic week.

---

## Phase 9: Trade simulator and guardrails

**Goal:** a trustworthy simulator that turns weights into daily portfolio values and a complete trade record.

**Build:**
1. `sim/records.py`: `Trade` fields: `week_id, date, portfolio_id, ticker, side, shares, fill_price, trade_value, cost_bps, cost_usd, reason (rebalance | stop_loss | take_profit), decision_source (llm | fallback_q0 | hold), avg_entry_price_before, avg_entry_price_after, weight_before, weight_after`. `DailyRecord`: `date, portfolio_id, value, cash, cumulative_cost, n_positions, week_status (normal | fallback | no_rebalance)`.
2. `sim/simulator.py`: `class PortfolioSim` with `portfolio_id`, `cash`, `positions {ticker: shares, avg_cost}`.
   - Start at 100000 cash on the same start date for every portfolio.
   - **Rebalance (Monday open):** get target weights; value positions at the open; `target_value_i = w_i * V_open`; trade value = `target_value_i - current_value_i`; shares are fractional; fill at the open price; cost = `cost_bps/10000 * |trade value|` for every trade including stops; cost is paid from cash; if cash is short after costs, scale buys proportionally (document and test this rule).
   - **Daily mark:** at each close, value positions at adjusted close; record the day.
   - **Stop and take-profit (every daily close):** if `close <= 0.92 * avg_cost` or `close >= 1.15 * avg_cost`, queue a full sale at the next trading day's open (reason set accordingly); cost applies. Proceeds stay in cash until the next Monday rebalance (they may be re-bought then if the optimizer selects the stock).
   - **Trigger on the last close before a rebalance:** if the queued sale would execute at the same open as the rebalance, do not sell and rebuy. Execute one trade per stock at that open: move straight to the new target weight (zero if the optimizer dropped it). The trade's reason is `rebalance`, a field `stop_trigger` records `stop_loss` or `take_profit`, and cost is paid once on the net trade. If the stock is held after the rebalance, its average cost resets only by the normal share-weighted rule.
   - Average cost updates on buys as the share-weighted average and is unchanged by sales.
   - **Price ledger (fixes history drift):** the simulator never reads adjusted prices. `data/ledger/prices_raw.csv` is an append-only daily log of raw open, high, low, close per stock, written once per trading day and never rewritten. `data/ledger/corporate_actions.csv` logs dividends (ex-date, cash per share) and splits (ratio) as they happen.
   - Dividends: on the ex-date, credit `shares * dividend` to cash. Splits: on the effective date, multiply shares by the ratio and divide average cost by it, so stop thresholds stay correct.
   - Fills, marks, and stop checks all use raw prices from the ledger. `replay_all` reads only the ledger, so past fills, P&L, and stops can never change when a provider later re-adjusts its history.
   - If a provider later reports a different value for a day already in the ledger, log a `ledger_mismatch` warning and keep the ledger value.
   - Adjusted prices are still used for the baseline, covariance, and Kronos inputs (E4), read from each frozen weekly snapshot.
   - Delisted or halted stock: if no price exists on a day, carry the last close, flag it, and force a sale at the first available open.
3. `simulate_week(portfolio_id, week_id, weights, decision_source, prices) -> (trades, daily_records)` and `replay_all(config) -> None` which replays every week for every portfolio from stored weights, and is safe to re-run (rebuilds from scratch deterministically).
4. Weeks marked `no_rebalance` hold positions; stop and take-profit rules still run.

**Tests:**
- Hand-calculated 3-stock, 5-day example to the cent (write the expected numbers in the test file).
- A stop triggers on the exact close threshold and sells at the next open; same for take-profit.
- Cost is charged on a stop sale.
- Holiday Monday trades on Tuesday.
- Fractional shares; weights after rebalance equal targets within tolerance.
- Replay twice gives identical records (hash compare).
- Sum of daily returns reconciles with the value series.
- A dividend credits cash on the ex-date and a split rescales shares and average cost, with no change in portfolio value.
- Changing a past day in the provider data does not change any replayed record (ledger is the only source).
- A stop triggered at the Friday close before a rebalance yields exactly one trade at the Monday open, cost charged once.
- Week alignment: a test with hand-set dates checks that scores at cutoff `week_id` are paired with the Monday-to-Monday window that starts after that cutoff, never an earlier one.

**Done when:** the simulator runs 30 synthetic weeks for 13 portfolios in under a minute and every reconciliation test passes.

---

## Phase 10: Fallback policy and weekly jobs

**Goal:** the weekly process as independent, resumable jobs with a labeled fallback.

**Build:**
1. `pipeline/status.py`: each job writes `data/outputs/<week>/jobs/<job>.status.json` with `state (pending | running | done | failed | skipped)`, `started_at`, `finished_at`, `inputs_hash`, `outputs_hash`, `error`. A job whose status is `done` and `inputs_hash` is unchanged is skipped unless `--force`.
2. `pipeline/jobs.py`, one function per job, each reading named inputs and writing named outputs:
   - `J1_snapshot`: Friday after close.
   - `J2_baseline`: baseline, covariance, prior.
   - `J3_api_submit`: assemble prompts and submit batches (cost guard first).
   - `J4_local_infer`: run local models (Kronos must be finished first: enforce with the status file, not time).
   - `J5_api_collect`: poll and collect; resubmit only failed items, up to a configured retry count. If a batch is still not ready when J7 must run, that (model, config) is marked failed for the week (N1).
   - `J6_validate`: call `assess_run` for every (model, config); mark failed runs; write `fallback_plan.json`.
   - `J7_weights`: for every portfolio, residualize, views, BL, MV; apply the fallback plan; compute Q0; write weights with `locked_at` timestamp and file hash; refuse to write if the lock time is after `timing.lock_deadline` (default Monday 9:00 am ET, 30 minutes before the open).
   - `J8_replay_score`: after the week ends: simulator replay, rank IC update, cost log.
   - `J9_backup`: call `backup.py`.
3. `pipeline/fallback.py` (confirmed fallback rule):
   - Failed (model, config) for the week: that portfolio's weights are the Q0 weights, `decision_source = fallback_q0`, and the week is marked `fallback` for that portfolio.
   - Data or baseline failure so Q0 cannot be built: hold last week's positions, no trades, `week_status = no_rebalance` for all portfolios.
   - Write `data/outputs/<week>/fallback_log.json` with one entry per fallback and the reason.
   - Provide `clean_weeks(portfolio_id)` returning only normal weeks, used by IC history and evaluation (N1 default).
4. `pipeline/cli.py` (typer): `portstudy run-job J1 --week 2026-W46`, `portstudy run-week --week ...` (runs jobs in order, stopping on failure), `portstudy status --week ...`, `portstudy replay`, `portstudy backup`, `portstudy eval`. Exit code non-zero on failure. Support `--synthetic` to use the synthetic providers and stub LLM.
5. `scripts/run_weekend.py` (decision N2): the one script the researcher runs by hand, ideally Friday night (Saturday morning is the backup). No scheduler. A Friday night start leaves time for one batch retry before the Monday 9:00 am ET lock; if started Saturday, the script prints a warning that the retry buffer is smaller.
   - Step 1: J8 and J9 for any earlier week whose replay or backup is not done (replay marks through the latest close and scores every week whose return window has closed).
   - Step 2: J1 to J6 for the current week, in order, then waits for API batches, polling J5 every few minutes with a visible countdown, until done or `timing.lock_deadline`.
   - Step 3: J7, then J9.
   - Prints each job's start, finish, counts (stocks, calls, invalid rate, cached tokens, cost so far), and any check issue as it happens, in plain readable lines.
   - On a failure it stops, prints what failed, the fallback that will apply, and the exact command to rerun that one job. Rerunning the script resumes where it stopped.
   - `--week` to run a specific week, `--dry-run` to print the plan only, `--synthetic` for the stub world.
   - Ends with a summary table: per portfolio, normal or fallback, and the locked weights file and hash.

**Tests:**
- Every job runs alone given its inputs.
- Running a job twice does not duplicate anything.
- Failure injection per job: J4 crash then resume; J5 batch never returns; J3 cost cap exceeded; invalid-output rate above threshold; J1 data failure. For each, the final weights and records carry the correct label.
- A fallback week for portfolio P is excluded from P's trailing IC and from `clean_weeks(P)`, but appears in the as-traded records.
- The J7 lock refuses a late write.
- `run_weekend.py` resumes correctly after being stopped at each step, and a batch not ready by the deadline produces a labeled fallback.

**Done when:** `python scripts/run_weekend.py --synthetic` for 30 consecutive weeks completes, and each injected failure test ends in a labeled fallback or a clean rerun.

---

## Phase 11: Evaluation engine

**Goal:** all H1 to H5 results and tests, validated by recovering a planted effect.

**Build (`src/portstudy/eval/`):**
1. `rank_ic.py`: weekly IC tables for baseline, LLM raw, LLM residual, and combined (E3 default: rank the tilt `mu_BL - pi`); only clean weeks per portfolio.
2. `hypotheses.py`:
   - **H1:** for each model and configuration, weekly gap `IC_llm_raw - IC_baseline`. Label `win` if gap > tie_band, `loss` if gap < -tie_band, else `tie`. Report counts, mean gap, and a confidence interval (bootstrap or Newey-West).
   - **H2:** weekly gap `IC_combined - IC_baseline`; and for portfolios, the paired weekly return difference against Q0 (primary), plus Sharpe ratio, max drawdown, turnover, and total cost as descriptive columns.
   - **H3 (G3):** paired weekly gaps per model: skills vs bare and skills vs placebo (and any later skills file vs bare). A comparison counts as real only if its paired weekly test passes after Holm correction across the full family.
   - **H4 (G4):** gaps within pre-set splits, labeled exploratory:
     - market up or down: sign of SPY weekly return (Monday open to Monday open);
     - volatility: above or below the median of SPY 20-day realized volatility at the cutoff;
     - earnings week: stocks with `earnings_in_window` vs not (IC computed within each group). Weeks where either group has fewer than 10 stocks give no IC for that week, and the report states how many weeks were usable; outside earnings season this split is mostly descriptive. Also report, as descriptive only, the share of each portfolio's weight and of its stop-loss and take-profit sales that fell in earnings-window stocks.
   - **H5:** see `factors.py` and `shuffle.py`.
3. `tests.py` (every test takes a seed and is deterministic):
   - `newey_west_mean(x, maxlags)`: OLS of x on a constant with HAC (Bartlett) errors via statsmodels; default `maxlags = floor(4 * (T/100)^(2/9))`, minimum 1; return mean, standard error, t, p.
   - `sign_test(x)`: `scipy.stats.binomtest` on count of positives among non-zero values, two-sided.
   - `sign_flip_permutation(x, n=10000, seed)`: randomly flip signs, compare the mean; two-sided p.
   - `holm(pvalues)`: `statsmodels.stats.multitest.multipletests(method="holm")`.
   - Holm families, fixed before the run in `eval.holm_families`: H1 is one family (4 models x 3 configurations = 12 tests); H2 IC gaps are one family (12) and H2 portfolio-vs-Q0 gaps are one family (12); H3 is one family (skills vs bare and skills vs placebo per model = 8); the shuffle test is one family (13 portfolios); each H4 split is its own family and labeled exploratory. Adding a skills file later adds its tests to the same families.
   - `min_detectable_gap(sd, T)`: `2.8 * sd / sqrt(T)`.
4. `factors.py` (G5 default, daily):
   - Build factor stand-ins from ETFs during the run: market = SPY minus T-bill; size = IWM minus SPY; value = IWD minus IWF; profitability = QUAL minus SPY; investment = unavailable (report as a gap).
   - If an official Fama-French daily file is present at the configured path, use MKT-RF, SMB, HML, RMW, CMA, and RF instead.
   - Regress each portfolio's daily excess return on the factors with HAC errors; report alpha, betas, and R squared.
5. `shuffle.py` (G6):
   - Coverage: every portfolio (each model and configuration) plus Q0, so no portfolio is selected after seeing results. For Q0 the baseline score `z_b` is shuffled instead.
   - **Hold view strength fixed:** every shuffle uses the real portfolio's stored weekly `IC_L` (and `IC_b` for Q0) and its real `use_llm_view` flags. Trailing IC is never recomputed from shuffled scores, so only the assignment of scores to stocks is random.
   - For a portfolio, for each week permute the LLM scores (and confidences together) across stocks with a seeded RNG, rebuild residualization, views, BL, MV, and replay the simulator from the ledger for the live period; repeat `n_shuffles` (500) to give a distribution of total return and Sharpe ratio. Report the real portfolio's percentile and the p-value `(1 + count of shuffles at or above the real result) / (n_shuffles + 1)`, then apply Holm correction across all 13 portfolios.
   - Respect `shuffle.n_shuffles` in config; support multiprocessing; cache by seed so reruns resume. Provide a fast path that reuses per-week covariance and prior.
6. `report.py`: write `data/outputs/report/` tables (CSV and Markdown) and simple PNG figures (matplotlib): weekly IC series, gap distributions, win/tie/loss bars, equity curves with Q0, shuffle histograms. Always show as-traded and clean-weeks-only results, with fallback counts per portfolio. Print the minimum detectable gap using the observed standard deviation, and state gaps below it as inconclusive, not as no effect.

**Tests:**
- Each test statistic matches a hand calculation on a small vector.
- Planted effect: on a synthetic world where the stub LLM has a known edge over bare, H1 and H3 detect it at the configured significance; where the stub has no edge (placebo), no effect is reported more often than the false-positive rate (run 200 seeds, allow a tolerance).
- Holm adjusted p-values never decrease below raw p-values and are monotone.
- Fallback weeks are excluded exactly as specified.
- Report builds with no errors on the synthetic run.

**Done when:** `portstudy eval --synthetic` produces every table and figure, and the planted-effect tests pass.

---

## Phase 12: GitLab CI

**Goal:** CI that lints and tests every push. The weekly pipeline is never run from GitLab (decision N2).

**Build:**
1. `.gitlab-ci.yml` with stages `lint`, `test`, and `checks`, on GitLab shared runners (image `python:3.11`):
   - `lint`: ruff and mypy.
   - `test`: pytest with coverage over `tests/unit` and `tests/integration`; e2e tests run only on the default branch to save CI minutes (the free tier has about 400 minutes a month).
   - `checks`: manifest hash check, placebo length check, the em dash and banned-word grep, and a secret pattern grep.
   - Cache pip downloads. No secrets are needed in CI because nothing calls a real service.
2. `docs/gitlab_setup.md`: create the project, push, confirm the pipeline passes.
3. `docs/desktop_checklist.md`: what the desktop needs before each weekend run (Ollama running, GPU free, keys in `.env`, power plugged in, sleep off while the script runs), and the note to start Friday night when possible.

**Tests:** validate the YAML with `yamllint` and check that every command it calls exists.

**Done when:** the CI file validates and its commands run locally.

---

## Phase 13: Backups

**Goal:** a clean, dated CSV export each week that the researcher downloads (decision N3), with a tested restore.

**Build:**
1. `backup.py`:
   - `run_backup(week_id)`: writes `data/backups/<week_id>_<timestamp>/` holding the full price ledger, the week's snapshot CSVs, output CSVs, logs, prompt files and manifest, portfolio and trade CSVs, configs, and run metadata (git commit hash, package versions), plus one zip of the whole folder for easy download.
   - Keeps a `checksums.json` per backup; never overwrites an earlier backup.
   - Also maintains `data/exports/master/`: one running CSV per table across all weeks (`llm_scores_all.csv`, `trades_all.csv`, `daily_values_all.csv`, `weekly_ic_all.csv`, `fallback_log_all.csv`, `cost_log_all.csv`), rebuilt from the weekly files so it is always consistent.
   - Prints the folder and zip path at the end so the researcher knows what to download.
2. `restore_backup(path, target)` and `verify_backup(path)` (recompute hashes).
3. A `--dry-run` flag and clear log lines for each file written.

**Tests:** back up a synthetic run to a temp directory, delete the originals, restore, and compare hashes; a corrupted backup file is detected by `verify_backup`; the master CSVs equal the concatenation of the weekly files.

**Done when:** the backup and restore round trip passes.

---

## Phase 14: End-to-end rehearsal on synthetic data

**Goal:** prove the whole system works together before any real service is touched.

**Build `tests/e2e/`:**
1. `test_full_run_synthetic`: 25 synthetic weeks, 2 local-style models and 2 API-style models (all stubs), 3 configurations (bare, skills, placebo), 100 stocks; runs `run_weekend.py` for each week; asserts all invariants (weights, costs, reconciliation, no look-ahead, no file under `prompts/` changed, fallback labels correct).
2. `test_failure_week`: injects each failure type in different weeks of one run and checks the as-traded and clean-weeks-only outputs.
3. `test_resume`: kills a run mid-week (raise in J4), restarts, and checks final outputs equal an uninterrupted run (hash compare).
4. `test_determinism`: two full runs with the same seed produce identical outputs.
5. `test_no_future_data`: a monkeypatched provider that raises if asked for any date after the cutoff.
6. `test_cost_log`: total projected and actual stub cost lines up with the cost table.
7. `scripts/benchmark_pipeline.py`: wall-clock time per job for 100 stocks, to help the researcher size the weekend window.

**Done when:** `pytest tests/e2e` passes in under 10 minutes and `PROGRESS.md` shows every earlier phase green.

---

## Phase 15: Documentation

**Build:**
1. `README.md`: install, run the synthetic demo, run tests, directory map.
2. `docs/runbook.md`: for a human on a bad Saturday: how to check status (`portstudy status`), rerun one job, handle each fallback type, where logs are, how to restore from backup.
3. `docs/config_reference.md`: every config key, meaning, default, and which deferred decision ID it belongs to.
4. `docs/open_questions.md`: a mirror of the table in section 0.3 with where each default lives in `study.yaml`, and a note that changing a default after the live run starts must be logged and versioned.
5. `docs/design.md`: one page per pipeline stage with the formulas above (copy exactly from this file), so the code and the spec cannot drift.

**Done when:** a new reader can run the synthetic demo from the README alone (verify by following it in a clean virtual environment).

---

## Phase 16: Final agent checklist

Before declaring the code complete, verify and record in `PROGRESS.md`:
- [ ] `ruff`, `mypy`, and the full test suite pass from a clean checkout.
- [ ] No em dashes or the word "arm" in the repository (`grep -r` check, add it as a CI step).
- [ ] No API key or secret appears anywhere in the repository (`grep` for key-like patterns, add as a CI step).
- [ ] Every `UNVERIFIED` marker is listed in `PROGRESS.md` with the file and function.
- [ ] Every `DECISION(...)` marker appears in `docs/open_questions.md`.
- [ ] The runner never writes into `prompts/` (test exists and passes).
- [ ] The synthetic end-to-end run is deterministic.
- [ ] Git history has one commit per phase.

---

## Phase 17: Manual developer work (a human does this; the agent cannot)

Do these in order after Phases 1 to 16 are green. Each item says what the agent should have already prepared.

### A. Accounts, keys, and spending limits
1. Create API accounts and keys for Anthropic (Claude Haiku 5.5) and OpenAI (GPT-6 Luna). Put the keys in a `.env` file on the desktop only (names are listed in `docs/config_reference.md`). Never commit it.
2. Set a monthly spending cap in each provider dashboard, so a bug cannot run far past the $50 goal.
3. Create free accounts and keys for Alpaca (prices and news), Finnhub (earnings dates), and FRED (T-bill rate). Confirm Alpaca's free plan includes the news endpoint and full daily bars. If not, switch `data.price_provider` to yfinance in config.
4. Download the official Fama-French five-factor daily file from the Ken French Data Library when it is needed for the final H5 run, and set its path in config.

### B. Desktop setup (RTX 3050, 6 GB)
1. Install NVIDIA drivers and CUDA, Python 3.11, and Ollama.
2. Pull the local models: Qwen 3.5 4B (thinking off) and Phi-4 Mini, plus Llama 3.2 3B as the fallback. Confirm exact tags in the Ollama library.
3. Clone Kronos at the pinned commit into `vendor/kronos` and download `Kronos-base` and `Kronos-Tokenizer-base` from the `NeoQuasar` Hugging Face organization.
4. Run `scripts/smoke_real_services.py` (the agent writes it to make one real call per service with a tiny prompt and print status, cost, and latency). Fix every `UNVERIFIED` item it exposes (see `PROGRESS.md`).
5. Benchmark: seconds per call and GPU memory per local model with a full-length prompt, and one Kronos-base run over 100 stocks. Confirm the total fits between your Friday night or Saturday morning start and Monday 9:30 am ET with margin. If a local model does not fit in 6 GB, swap it for Llama 3.2 3B in `config/models.yaml` and log why.
6. Power settings: plugged in, no sleep while the weekend script runs, Windows updates paused for the weekend.

### C. GitLab and backups
1. Create the private GitLab project, push the repository, and confirm the CI pipeline passes.
2. Run `run_weekend.py --synthetic` once on the desktop and download the backup zip it prints, then run one restore by hand to confirm it works.

### D. Research content only the researcher can write
1. **Write the skills file** (`prompts/skills/`) and the placebo text. Run `scripts/check_placebo_length.py`, then `scripts/refresh_manifest.py` to freeze versions and hashes.
2. Check the real skills file token count against the 6000 cap (C4) and confirm or change C4 and the 150-token rationale cap (C7).
3. **Build the stock list** (B2): run `build_universe`, review the printed sector table, add any `extra_tickers` by hand, then freeze `config/universe.csv` with a version tag.
4. Enter current model prices in `config/models.yaml`. Test-run the real prompts on a few stocks for each model, then run `scripts/estimate_cost.py` and recompute the budget (D3).
5. Set `timing.live_end_date` (A3), no later than the April deadline.

### E. Freeze, pilot, and go
1. Run a dry run on the last 3 to 4 real weeks with real providers, only as a bug check. Do not tune prompts or settings on the returns.
2. Tag the repository `frozen-v1`. From now on any change is a new version, documented and flagged in the analysis.
3. Run the one-week pilot with `run_weekend.py`. Review job success, valid-output rates, batch return times, cache hits, cost, run time, and fallback events. Hand-check trades, costs, and stops against the simulator output.
4. Fix engineering problems only, record each fix, and drop or swap any model that failed reliability or cost limits (with a log entry saying why).
5. Start the live run the week after the pilot passes (A2). The pilot week stays excluded from all results.
6. Every weekend: run `run_weekend.py` Friday night if possible (Saturday morning as the backup), read its summary, and download the backup zip.

---

## Appendix: Formula sheet (single source of truth)

```
Momentum:   M12_1 = P[t-21]/P[t-252] - 1 ;  M3 = P[t]/P[t-63] - 1 ;  R1 = -(P[t]/P[t-21] - 1)
Rank:       u(x)  = rank(x)/N
Kronos:     k_i   = mean_over_20_paths( P_hat[t+5] / P[t] ) - 1
Baseline:   b = 0.5 u(m) + 0.5 u(k) ;  z_b = (b - mean(b)) / sd(b)
Covariance: Sigma_daily = (1 - alpha) S + alpha (tr(S)/N) I  (Ledoit-Wolf) ;  Sigma = 5 Sigma_daily ;  sigma_i = sqrt(Sigma_ii)
Prior:      pi = delta Sigma w_mkt ;  delta = 2.5
Residual:   s = a + c z_b + e ;  z_L = standardize(e)
Views:      q_b = pi + IC_b sigma z_b ;  q_L = pi + IC_L sigma z_L
Uncertainty: Omega_b,ii = tau Sigma_ii ;  Omega_L,ii = tau Sigma_ii / clip(conf_i, 0.1, 1) ;  tau = 0.05
BL:         mu = [ inv(tau Sigma) + P' inv(Omega) P ]^-1 [ inv(tau Sigma) pi + P' inv(Omega) q ]   P = [I; I]
Q0:         same with P = I, q = q_b, Omega = Omega_b
Optimizer:  max w' mu - (delta/2) w' Sigma w   s.t. sum w = 1, 0 <= w_i <= 0.05
Cost:       C = 0.001 * sum_i |trade value_i|
Stops:      sell at next open if close <= 0.92 avg_cost  or  close >= 1.15 avg_cost  (checked at every daily close)
Rank IC:    Spearman(score at Friday cutoff, total return from next Monday open to the following Monday open)
Zero IC:    IC_L = 0  ->  drop LLM view rows, weights = Q0 weights
Trailing IC: mean of last 8 scored weeks, floored at 0 ; fixed 0.02 for the first 5 live weeks
Min detectable gap: 2.8 * sd / sqrt(T)
```