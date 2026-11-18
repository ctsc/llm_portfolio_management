# Configuration reference

All settings live in `config/study.yaml` (typed by `src/portstudy/config.py`, unknown keys
rejected), `config/models.yaml`, and `config/prompts_manifest.yaml`. The `decision` column
gives the decision id (see `docs/open_questions.md`) where one applies; code tags each use with
`DECISION(<id>)`. Unknown keys fail validation, so a key from a retired design cannot linger.

## Environment variables

| Variable | Meaning |
|---|---|
| `PORTSTUDY_DATA_DIR` | Root of every data folder (default `./data`). Point it at a scratch folder for dry runs so no pre-pilot week lands in the live folder (A7). |
| `PORTSTUDY_ROOT` | Repository root override (default: the checkout holding `src/`). |
| `PORTSTUDY_CONFIG` | Study YAML path (default `config/study.yaml`; CLI `--config` takes precedence). |
| `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY` | Alpaca bars and news. |
| `FINNHUB_API_KEY` | Finnhub earnings calendar. |
| `FRED_API_KEY` | FRED DTB3 series. |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | API models (batch endpoints). |
| `ARCTIC_LITELLM_API_KEY` | Bearer key for the ARCTIC LiteLLM gateway (`llm.gateway.base_url`); never logged. |
| `PORTSTUDY_E2E_DIGEST_FILE` | Tests only: where the e2e suite writes its output digest. |

Keys go in a `.env` file outside the repository (`~/.env` inside WSL, mode 600; `docs/desktop_setup.md`); `scripts/smoke_real_services.py` loads it and `check_secrets.py` fails on any key committed by mistake.

## study.yaml

### universe

| Key | Default | Meaning | Decision |
|---|---|---|---|
| `size` | 100 | Stocks in the study: 100, set by the 1,000 GPU-hour ARCTIC startup allocation (2026-11-16) | B4 |
| `source` | sp500 | Member list source (Wikipedia S&P 500 table) | B2 |
| `sectors` | 11 | GICS sectors | B2 |
| `min_history_days` | 300 | Minimum trading days before a stock can be picked or kept | |
| `dollar_volume_window_days` | 60 | Window for the ranking | |
| `extra_tickers` | [] | Hand-added stocks; each replaces the lowest pick in its sector | |
| `universe_file` | config/universe.csv | Frozen list; never rebuilt by code once written | |
| `etf_proxies` | SPY, IWM, IWD, IWF, QUAL | ETFs for the factor stand-ins | |

### timing

| Key | Default | Meaning | Decision |
|---|---|---|---|
| `timezone` | America/New_York | All times are ET with an explicit offset | |
| `cutoff_time_et` | 16:00 | Friday cutoff (or the prior trading day if closed) | A4 |
| `lock_deadline_time_et` | 09:00 | Monday weight lock, 30 minutes before the open | |
| `pilot_week` | null | ISO week of the pilot, excluded from results | |
| `live_start_week` | null | First live week (the week after the pilot passes) | A2 |
| `live_end_date` | null | Last live date, set by Carter before the freeze | A3 |

### data

| Key | Default | Meaning |
|---|---|---|
| `price_provider`, `news_provider`, `reference_provider`, `rate_provider`, `earnings_provider` | alpaca, alpaca, yfinance, fred, finnhub | Real adapters (`synthetic` in tests) |
| `alpaca_feed` (G16) | sip | Alpaca bar feed: `sip` (consolidated) or `iex` (one small exchange). If the free plan refuses `sip`, set `price_provider: yfinance`. Bars are requested up to the cutoff day 23:59:59 ET, never later than 16 minutes before now |
| `price_history_days` | 300 | Trading days pulled into each snapshot |
| `first_snapshot_news_days` | 7 | News window for the very first snapshot |
| `news_token_budget_per_stock` | 1500 | Newest-first cap per stock (len/4 estimate unless `news_tokenizer`) |
| `rate_limits.*_per_minute` | 200, 60, 120, 60 | Per-service call limits |
| `retries.attempts`, `retries.backoff_seconds` | 3, 2.0 | HTTP retries with exponential backoff |
| `fixture_dir` | tests/fixtures/providers | Recorded JSON for replay mode |
| `checks.<code>` | see file | Rule per check code: `skip_stock`, `forward_fill`, or `halt` |
| `checks.large_move_flag` | 0.10 | 5-day adjusted move since the previous cutoff at or beyond this is flagged (`large_move`) and the stock is kept (U1) |
| `checks.crash_removal` | -0.40 | A single-day or 5-day adjusted return at or below this removes the stock for the rest of the study (`data/universe_removals.csv`); nothing replaces it (U1) |
| `checks.unexplained_split` | skip_stock | A raw jump of the same size absent from the adjusted series with no split action: a data error, the stock is skipped for that week only (U1) |
| `checks.stale_days` | 5 | Unchanged close run length |

### synthetic

`seed`, `n_tickers` (120), `history_days` (400), `future_weeks` (40), `first_week` (2026-W02),
`world_dir`. Only used with `--synthetic`.

### baseline

| Key | Default | Meaning | Decision |
|---|---|---|---|
| `forecaster` | stub | `stub` or `kronos` | |
| `momentum_lookbacks.long/skip/medium/short` | 252/21/63/21 | M12_1, M3, R1 windows | |
| `kronos.repo_commit` | UNPINNED | Pinned `vendor/kronos` commit (set by Carter, `manualwork.md` B) | |
| `kronos.model_id`, `kronos.tokenizer_id` | NeoQuasar/Kronos-base, NeoQuasar/Kronos-Tokenizer-base | Hugging Face ids | |
| `kronos.context_candles`, `horizon`, `n_samples`, `temperature`, `top_p`, `seed` | 252, 5, 20, 1.0, 0.9, 7 | Sampling settings | |
| `kronos.input_prices` | adjusted | Adjusted OHLCV in | E4 |
| `kronos.batch_size`, `kronos.device` | 100, cuda | Stocks per Kronos chunk so the universe fits in 6 GB (100 stocks is one chunk); device. Each (sample, chunk) has its own seed, so `batch_size` changes the sampled paths and is frozen with the config | K2 |
| `on_kronos_failure` | momentum_only | When the forecaster raises: `momentum_only` builds the baseline from the momentum rank alone (neutral K1 spread, `kronos_fallback: true` in `baseline_meta.json`); `hold` fails J2 | K3 |
| `stub_seed` | 11 | Seed of the stub forecaster | |

### risk

| Key | Default | Meaning | Decision |
|---|---|---|---|
| `cov_lookback_days` | 252 | Daily returns window | |
| `delta` | 2.5 | Risk aversion in the prior and the optimizer | |
| `weekly_scale` | 5 | Sigma = 5 Sigma_daily | |
| `min_eigenvalue` | 1e-10 | SPD assertion | |
| `market_cap_weights` | study_only | Market weights over the study stocks only, renormalized | E5 |

### llm

| Key | Default | Meaning | Decision |
|---|---|---|---|
| `cost_goal_usd` | 50.0 | Budget goal; the guard needs `--allow-over-budget` to pass it | D3 |
| `cost_warn_usd` | 50.0 | Warning threshold | D3 |
| `temperature` | 0.0 | For every model. Both open-weight model cards recommend 1.0 (ARCTIC's example uses 0.2); Carter decides, and the paper states the reason for 0 (`manualwork.md` C) | D4 |
| `no_news_sentence` | "No relevant news is available for this week." | Used when a stock has no news | C2 |
| `prompt_fields` | ticker, name, sector, news | Stock part of the prompt | C1 |
| `skills_token_cap` | 6000 | Cap on each enabled skills and placebo file for every tokenizer | C4 |
| `skills_length_tolerance` | 0.05 | Each enabled skills file within 5 percent of the enabled skills mean | P1 |
| `rationale_token_cap` | 150 | Rationale truncation | C7 |
| `placebo_length_tolerance` | 0.02 | Placebo within 2 percent of the enabled skills mean | |
| `invalid_rate_threshold` | 0.15 | Failure rule | N1 |
| `retry_invalid_once` | true | One retry on an invalid answer | |
| `batch_retry_count` | 1 | Resubmissions of failed batch items | |
| `api_poll_seconds` | 300 | J5 polling interval in `run_weekend.py` (without `--poll-once`) | |
| `parallel_models` | true | J4 runs every enabled open-weight model in its own worker with its own `concurrency` pool; `study_test.yaml` sets false for identical digests | |
| `settings_default`, `settings.<model>` | temperature 0, seed 1234, max_output_tokens 300, reasoning none | Per-model overrides of `models.yaml` settings | |
| `model_ids`, `prompt_ids` | [] | Empty selects all enabled models and prompts; lists restrict the source set | |

### views

| Key | Default | Meaning | Decision |
|---|---|---|---|
| `tau` | 0.05 | Black-Litterman scale | |
| `ic_window_weeks` | 8 | Trailing IC window | |
| `ic_floor` | 0.0 | Floor on the trailing IC | |
| `cold_start_weeks`, `cold_start_ic` | 5, 0.02 | Fixed IC for the first live weeks | A6 |
| `ic_min_weeks` | 3 | Fewer clean weeks keeps the cold-start IC | |
| `kronos_spread_clip` | [0.25, 4.0] | Clip on each stock's Kronos spread divided by the weekly median in Omega_b | K1 |
| `width_conf_clip` | [0.1, 1.0] | Clip on width-derived LLM view scale `c` in Omega_L | E1 |
| `ic_signal_for_llm` | llm_resid | IC used to scale the LLM view | E2 |

### optimizer, simulator

`optimizer.max_weight` 0.05, `min_stocks` 20, `solvers` [OSQP, CLARABEL];
`simulator.initial_value` 100000, `cost_bps` 10, `stop_loss` -0.08, `take_profit` 0.15.

### strategies

v2 phase 7 enables `mvo`, `cvar`, `robust`, `resampled`, and `topk`, plus the HRP and EW controls.
`cvar.n_scenarios` defaults to 10,000, `cvar.beta` 0.95, `cvar.lam` 1.0 (`DECISION(S3)` and `DECISION(S4)`),
seed 7, and solvers CLARABEL and HIGHS. Install the `solvers` extra to enable HIGHS.
`robust` uses kappa 1.0 (`DECISION(S5)`) and CLARABEL. `resampled.n_draws` is 200, `resampled.seed` 7, and
`resampled.max_failed_share` 0.10 (more failed draws than that fails the solve). `topk.k` defaults to 20 and HRP linkage to `single`. The shared
optimizer guardrails are no short positions and a 5 percent cap (`DECISION(G1)`). Validation requires
`1 / strategies.topk.k <= optimizer.max_weight` and
`optimizer.min_stocks >= ceil(1 / optimizer.max_weight)`.

`config/study_test.yaml` selects two stub models and four prompt configurations, enables all five
strategies plus HRP and equal weight, and uses 2,000 CVaR scenarios, 20 resampled draws, and 120
synthetic tickers.

### pipeline

`jobs` (J1 to J9 in order), `stop_on_failure` true, `lock_margin_minutes` 30 (J6 declares a run
failed this long before the lock so that J7 can still write; decision N1).

### eval

| Key | Default | Meaning | Decision |
|---|---|---|---|
| `tie_band` | 0.001 | Win, tie, loss band on weekly gaps (10 bps) | G12 |
| `tie_band_sensitivity` | 0, 0.0005, 0.001, 0.005, 0.01 | Bands reported in the tie-band sensitivity table | G12 |
| `bootstrap_samples`, `permutation_samples` | 2000, 10000 | Resampling sizes (Monte Carlo sign flips above the exact limit) | |
| `bootstrap_block_mean` | null | Mean block length of the stationary bootstrap; null means T^(1/3) | G12 |
| `signflip_exact_max_t` | 24 | Exact sign-flip enumeration up to this many weeks | G12 |
| `significance` | 0.05 | After multiple-testing adjustment | |
| `seed` | 2026 | Every test is seeded | |
| `combined_signal` | bl_tilt | Combined score for H2 is the ranked BL tilt | E3 |
| `factor_frequency` | daily | Factor regression frequency | G5 |
| `factor_source` | french | `french` (Kenneth French daily FF5 plus momentum) or `etf_proxy` (live monitoring only) | G11 |
| `fama_french_file`, `momentum_file` | null | Paths to the French daily five-factor and momentum CSVs | G11 |
| `risk_free` | fred | Risk-free rate: the FRED T-bill series stored in each snapshot | G10 |
| `cer_gamma` | 2, 5 | Risk aversions for the certainty-equivalent return | G10 |
| `report_active_weeks` | true | H2 reported intention-to-treat and active-view weeks only | G8 |
| `preregistration_file` | config/preregistration.yaml | The pre-registration (the test config points at `config/preregistration_test.yaml`) | G7 |
| `multiple_testing` | holm, fdr_bh, [H4] | Holm on the preregistered primary family, Benjamini-Hochberg on the rest, H4 exploratory | G7 |
| `h4_splits` | market_direction, volatility, earnings_week | Exploratory splits | G4 |
| `min_group_size` | 10 | Minimum stocks per group for the earnings split IC | |
| `h3b.split`, `h3b.min_group_size` | median, 10 | H3b width split per source-week; weeks with a smaller group or all-equal widths are skipped | G3b |

### config/preregistration.yaml

Fixed before the live run (G7, G9, G13). `frozen` (false until Carter freezes it, with
`frozen_on`); `primary_comparisons` (the four pooled main comparisons: `rq1_llm_vs_baseline`,
`rq2_tilt_gap`, `rq3_skills_vs_bare`, `rq5_skills_vs_placebo`, each with `kind` and optional
`models`, `prompts`, `control_prompt`, `strategies` selectors; at most 10); `residual_ic_controls`
(the nine G9 controls); `main_model_comparison_prompt` (bare);
`view_sensitivity_min_tracking_error` (null until Carter sets it; `frozen: true` is refused while it
is null). The preflight refuses a live run while the file is invalid, names a disabled model,
prompt or strategy, or is not frozen, and the frozen-config check covers it after the pilot.

### backup

`keep_zips` true; `rsync_target` null (optional rsync destination for the dated backup folder and zip, v2 phase 12); `master_tables` (llm_scores, trades, daily_values, weekly_ic, fallback_log,
cost_log). Decision N3.

### decisions

One entry per id with `id`, `value`, `status` (`confirmed` or `provisional`). Never changed by
code. See `docs/open_questions.md`.

## models.yaml

`models` (real registry) and `stub_models` (tests and `--synthetic`). Each entry: `id`,
`display_name`, `kind` (`open_weight` or `api`), `provider` (`litellm`,
`anthropic_batch`, `openai_batch`, `stub`), `provider_model`, `serving` (`gateway`, the only value), `snapshot_id`, `enabled`, `settings`, and `price`
(`input_per_m`, `cached_input_per_m`, `output_per_m`, `batch_discount`). Gateway entries use
zero prices; API prices follow D3.

`llm.concurrency` sets per-model open-weight worker counts, defaulting to four when omitted.
`llm.settings` supports `send_seed` and `json_mode`. ARCTIC's API page documents `response_format`,
and ARCTIC confirmed on 2026-11-12 that `seed` is passed to the model server (outputs are still not
guaranteed identical, because requests are batched with other users'). `json_mode` stays
`UNVERIFIED` until a real run. MiniMax M2.7 always thinks
and its thinking counts toward the output limit, so each gateway model has its own limits in
`config/models.yaml` `settings` (G15): `max_output_tokens` 300 for Gemma and 2,000 for MiniMax,
`request_timeout_seconds` 180 and 600, both starting values until `--time-gateway` measures them.
`llm.settings.<model>` in `study.yaml` overrides a model's settings. The `llm.gateway` section contains the gateway `base_url`
(`https://litellm.rs.gsu.edu/v1`, the ARCTIC LiteLLM gateway; GSU network or VPN required) and
`features_frozen` (default false; set true after the pilot, after which a rejected `seed` or
`response_format` is a failed request instead of a silently disabled feature). API model prices
carry `confirmed` (false until Carter enters real prices) and each API model needs a dated
`snapshot_id` (D5); the preflight checks both. The enabled study
models are Gemma 4 31B and MiniMax M2.7 on the gateway plus Claude Haiku 5.5 and GPT-6 Luna.

## prompts_manifest.yaml

`base_instructions`, `base_sha256`, and `configurations`: `id`, `kind` (`bare`, `skills`,
`placebo`), `file`, `version`, `sha256`, `hashed_version`, `enabled`, `matches` (a skills
configuration id or `skills_mean` for the enabled skills mean). `skills_5` is disabled by default.
Decision B1. Maintained by
`scripts/refresh_manifest.py`.

## Keys added by PLAN4 after the 2026-11-11 review (built in the phase named unless that phase is still to come)

| Key | Default | Meaning | Decision | Phase |
|---|---|---|---|---|
| `strategies.cvar.lam` | set by calibration | Mean-CVaR risk weight, chosen on synthetic data by a fixed rule | S8 | 3B |
| `strategies.resampled.draw_scale` | 1.0, then set by calibration | Fraction of the posterior covariance the resampled means are drawn from | S8 | 3B |
| `views.ic_history_start` | `live_start_week` | Weeks before the first live week never count toward IC or results, and every portfolio starts from cash at the first live week (`first_locked_week` keeps the old behavior for tests) | A7 | 3A |
| `views.ic_mode`, `views.fixed_ic` | `fixed`, 0.02 | Fixed view scale for both views every week (`trailing` keeps the old A6 gating for tests) | A8 | 3A |
| `strategies.conf_only_controls` | `[robust, resampled]` | Strategies that get confidence-only control portfolios | G19 | 3B |
| `simulator.no_price_liquidation_days` | 5 | Trading days without a price before a held stock is sold at its last close | U2 | 3A |
| `llm.settings.<model>.request_timeout_seconds` | 180 (MiniMax 600 until measured) | Gateway request timeout per model | G15 | 3 |
| `llm.gateway.features_frozen` | false | When true, a rejected `seed` or JSON mode is an error and not a silent change | G15 | 3 |
| `arctic.gpu_hours_allocated`, `arctic.gpu_seconds_per_request` | 1000; Gemma 40, MiniMax 90 | ARCTIC usage tracking against the allocation (replace the per-request values after the timing run) | N7 | 3A |
| `arctic.litellm_budget_usd`, `arctic.warn_share`, `arctic.alert_share` | 10, 0.75, 0.90 | Gateway dollar budget checked before J4, and the warning and alert shares of the projections | N7 | 3A |
| `llm.network_outage_tolerance_minutes` | 60 | How long J4 and J5 wait out a connection outage before counting failures | N4, N5 | 3A |
| `view_sensitivity_min_tracking_error` (in `config/preregistration.yaml`) | null | Threshold below which a strategy is reported as insensitive to views | G13 | 4 |

The LLM view formula is `q_L = mu_Q0 + IC_L sigma z_L` (decision E7, built in Phase 3B).
