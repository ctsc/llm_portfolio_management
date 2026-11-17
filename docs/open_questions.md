# Open questions and study decisions

The current decision table. Source of truth: `plans/PLAN4.md` section 0.3 (approved 2026-11-10
and 2026-11-11) on top of the unchanged `plans/PLAN2.md` section 0.3 values. `config/study.yaml`
still holds the older text for D1, D2, N2, G6, W1 and M1 until PLAN4 Phases 2 and 4 update it.

Rule: changing any default after the live run starts must be logged in `plans/PROGRESS.md` with the
date and reason and flagged in the analysis. Code never changes a decision value.

## Current decisions

| ID | Setting | Value | Status |
|---|---|---|---|
| D1 | models | ARCTIC LiteLLM gateway: Gemma 4 31B and MiniMax M2.7 only, no deployment requests; API: both Claude Haiku 5.5 and GPT-6 Luna; no Qwen or DeepSeek | confirmed (2026-11-10; API models 2026-11-17) |
| D2 | gateway outage | the failing model's week is marked failed and left out of its statistics; no Slurm or vLLM fallback | confirmed (2026-11-10) |
| L1 | where it runs | everything except LLM calls on Carter's desktop in WSL2; Kronos on the local RTX 3050; ARCTIC request is gateway GPU hours only | confirmed (2026-11-10) |
| K2 | Kronos batching | forecast in chunks of `baseline.kronos.batch_size` (default 100) to fit 6 GB | confirmed (2026-11-11) |
| D3 | paid API budget | goal 50 USD; gateway models cost 0 USD; recompute with real prices | provisional |
| D4 | temperature | 0 for every model with a fixed seed; the gateway passes the seed through but outputs are not guaranteed identical (ARCTIC, 2026-11-12) | confirmed (2026-11-12) |
| G15 | gateway output handling | per-model `max_output_tokens` (Gemma 300, MiniMax about 2,000 until measured), per-model timeout, reasoning text saved, tokens and `finish_reason` logged, features frozen after the pilot | confirmed (2026-11-11), pending PLAN4 Phase 3 |
| G16 | price feed | Alpaca `sip` by default (not `iex`), yfinance fallback | confirmed (2026-11-11), pending PLAN4 Phase 3 |
| G17 | ARCTIC sizing basis | 300 to 8,000 input tokens; output about 300 (Gemma) and 2,000 (MiniMax); estimated 40 and 90 GPU-seconds per request; 2,730 / 910 GPU-hours requested; corrected after the timing run | confirmed (2026-11-12) |
| E7 | neutral LLM view | `q_L = mu_Q0 + IC_L sigma z_L` (centered on the Q0 posterior mean, not `pi`) | confirmed (2026-11-11), pending PLAN4 Phase 3B |
| G18 | confidence-only counterfactual | posterior with the same LLM view uncertainty and `z_L = 0`; its uncertainty ratio is recorded per source-week | confirmed (2026-11-11), pending PLAN4 Phase 3B |
| S8 | strategy sensitivity calibration | Mean-CVaR `lam` and Resampled `draw_scale` set on synthetic data by a fixed rule before the freeze | confirmed (2026-11-11), pending PLAN4 Phase 3B |
| G1b | stop on a rebalance day | a stop merged into the rebalance resets the position's reference price to the rebalance open | confirmed (2026-11-11), pending PLAN4 Phase 3A |
| N5 | gateway request failures | recorded as invalid `provider_error`; J4 not done while a request has no record | confirmed (2026-11-11), pending PLAN4 Phase 3 |
| C8 | prompt name and sector | from the frozen universe file; yfinance for market cap only, last good value reused | confirmed (2026-11-11), pending PLAN4 Phase 3 |
| A7 | data-folder hygiene | scratch data folder for dry runs; no pre-pilot weeks in the live folder; pilot excluded and every portfolio starts from cash at the first live week | confirmed (2026-11-12), pending PLAN4 Phase 3A |
| U2 | no-price liquidation | sold at the last close after 5 trading days without a price | confirmed (2026-11-12), pending PLAN4 Phase 3A |
| D5 | API model versions | dated `snapshot_id` required for a live run | confirmed (2026-11-11), pending PLAN4 Phase 3 |
| A8 | view scale | IC_b and IC_L fixed at 0.02 every week (`views.ic_mode: fixed`); trailing-IC gating kept only for tests | confirmed (2026-11-12), pending PLAN4 Phase 3A |
| A9 | news cutoff | Friday 16:00 ET, unchanged | confirmed (2026-11-12) |
| K3 | Kronos failure | `momentum_only` for every portfolio that week, flagged | confirmed (2026-11-12), pending PLAN4 Phase 3 |
| G19 | confidence-only control portfolios | Robust and Resampled MVO, rating 0 with the same view uncertainty | confirmed (2026-11-12), pending PLAN4 Phase 3B |
| G20 | pilot repeat check | 50 fixed stocks re-rated under all prompt types per gateway model, cache bypassed; agreement reported | confirmed (2026-11-12), pending PLAN4 Phase 3A |
| G21 | weekly model checks | `model` field, `x-litellm-model-id` and `x-litellm-version` headers, `reasoning_tokens`, `/v1/model/info` `updated_at`; 10 canaries per model before J4; model or deployment change or canary drop flags the week | confirmed (2026-11-16), pending PLAN4 Phase 3A |
| N6 | logging and durable progress | run_id, event timeline, tracebacks, captured warnings, `fallbacks.csv`, fsync on every write, disk check per job, `run_summary.md`, `portstudy audit`, random-kill resume test | confirmed (2026-11-12), pending PLAN4 Phase 3A |
| N7 | ARCTIC usage budget | estimated GPU-hours per week, running total and projection against 1,000; warn at 75%, alert at 90% | confirmed (2026-11-16), pending PLAN4 Phase 3A |
| G22 | measure everything | every metric for every source, portfolio, model, prompt and strategy, per cell and pooled, in `results/eval/all_metrics.csv`; Sharpe net of the T-bill rate leads the portfolio table; null results reported; scope fixed at the freeze | confirmed (2026-11-17), pending PLAN4 Phase 9 |
| B1 | prompt types | bare, placebo, and 4 similar-length specialized skills files (`skills_5` off); one placebo matched to mean skills length | confirmed (2026-11-12) |
| B4 | universe size | 100 stocks; ARCTIC granted the 1,000 GPU-hour startup allocation (2026-11-16) | confirmed (2026-11-16) |
| C3 | output schema | JSON low, likely, high integers in [-5, 5], rationale; low <= likely <= high; no separate certainty field | confirmed |
| E1 | LLM view uncertainty | width = high - low; c = clip(1 - width/10, 0.1, 1); Omega_L,ii = tau Sigma_ii / c | confirmed |
| E2 | IC for LLM view | IC of the residualized likely score | confirmed |
| E6 | baseline view uncertainty | spread of Kronos's 20 sampled paths, mapped by K1 | confirmed |
| K1 | Kronos spread mapping | five-day return spread; ratio clipped to [0.25, 4]; scales Omega_b | confirmed (design choice) |
| S1 | steered strategies | MVO, Mean-CVaR, Robust MVO, Resampled MVO, Top-K; each has its own Q0 | confirmed |
| S2 | controls | HRP and equal weight; no LLM view or Q0 | confirmed |
| S3 | Mean-CVaR | 10,000 Monte Carlo scenarios from the BL posterior; beta 0.95 | confirmed |
| S4 | Mean-CVaR lambda | 1.0 until the S8 calibration sets it (PLAN4 Phase 3B) | changed (2026-11-11) |
| S5 | Robust MVO kappa | 1.0, ellipsoid from the BL posterior covariance of the mean | confirmed (design choice) |
| S6 | Resampled MVO draws | 200 | confirmed |
| S7 | Top-K | K = 20, equal weights | confirmed |
| G1 | guardrails | 5% cap, no shorts, 10 bps cost, -8% stop, +15% take-profit, weekly rebalance | confirmed |
| G3 | H3 | each skills file vs bare and placebo per model on IC and on active return, plus a pooled regression across models; H3b range-width test | confirmed (extended 2026-11-11) |
| G3b | H3b width split | per source and week, narrow width at or below median vs the rest; plus a confidence interaction and a width vs error check | confirmed (design choice) |
| P1 | skills length tolerance | each skills file within 5% of mean skills length; placebo within 2% | confirmed (design choice) |
| G4 | H4 | exploratory: strategy ranking by mean active return vs own Q0, strategy-vs-strategy differences, market, volatility and earnings splits | confirmed |
| G5 | factor frequency | daily | confirmed |
| G7 | multiple testing | Holm on four pooled main comparisons in `config/preregistration.yaml` (LLM-minus-baseline IC gap, steered-minus-Q0 tilt IC, skills minus bare, skills minus placebo); Benjamini-Hochberg on the rest; H4 exploratory | confirmed (2026-11-12; comparison 1 revised 2026-11-17) |
| G8 | H2 reporting | intention-to-treat (all clean weeks) and active-view weeks, with each source's active-week share | confirmed (2026-11-11) |
| G9 | residual IC | weekly Fama-MacBeth regression on the baseline score, momentum, size, volatility, beta, sector and trailing value, profitability and investment loadings; controls fixed in the preregistration | confirmed (2026-11-11) |
| G10 | portfolio metrics | Sharpe net of the T-bill rate; Sortino, max drawdown, CVaR from daily returns; IR and tracking error vs Q0; probabilistic Sharpe; certainty-equivalent return; break-even cost | confirmed (2026-11-11) |
| G11 | factor model | Kenneth French daily FF5 plus momentum on daily active returns; ETF proxies for live monitoring only | confirmed (2026-11-11) |
| G12 | small-sample inference | Student-t with T - 1 df; Newey-West lags floor(4 (T/100)^(2/9)); block bootstrap percentile-t; exact sign-flip when T <= 24; Ledoit-Wolf Sharpe test; tie band 0.001 with sensitivity | confirmed (2026-11-11) |
| N2 | weekly trigger | Carter runs `scripts/run_weekend.py` by hand on the desktop; no scheduler, no Slurm | confirmed (changed 2026-11-10) |
| A2, A3, A6, B2, C1, C2, C4, C7, E3, E4, E5, N1, N3 | unchanged decisions | retain the v1 values listed below | confirmed |

## Retired and superseded decisions

| ID | Was | Retired | Why |
|---|---|---|---|
| G6 | 500 shuffled-rating replays for MVO, Robust MVO and Top-K and their Q0s (PLAN3) | 2026-11-11 | About 638 CPU-hours at 300 stocks on the desktop, and it could not reach significance under Holm (52 / 501 = 0.104); replaced by residual IC, factor regressions of active returns and the placebo control |
| W1 | every Slurm job at most 15 hours, target 12 (PLAN3) | 2026-11-10 | No Slurm jobs; everything but LLM calls runs on the desktop |
| M1 | Slurm CPU jobs at `--mem-per-cpu=3G` (PLAN3) | 2026-11-10 | Same |
| D1 (old) | gateway models plus requested deployments of Gemma 4 E4B, Phi-4 Mini 3.8B, Llama 3.2 3B | 2026-11-10 | Request limited to the two models ARCTIC already hosts |
| D2 (old) | gateway outage served by Carter's own Slurm GPU jobs | 2026-11-10 | No ARCTIC compute requested |
| N2 (old) | one hand-run submit script on ARCTIC | 2026-11-10 | Runs on the desktop |

## Historical v1 decision table

The table below records the values used before PLAN2 superseded them.

| ID | Setting | Value | Status | Config location | Code tags |
|---|---|---|---|---|---|
| D1 | models | local: Qwen 3.5 4B (thinking off), Phi-4 Mini; fallback Llama 3.2 3B; API: Claude Haiku 5.5, GPT-6 Luna (reasoning none) | confirmed | `config/models.yaml` (`models`), `decisions[D1]` | `llm/providers/openai_batch.py` (reasoning none) |
| D3 | budget | goal 50 USD; warn at 50; `--allow-over-budget` to pass it | provisional | `llm.cost_goal_usd`, `llm.cost_warn_usd`, `decisions[D3]` | `llm/cost.py`, `scripts/estimate_cost.py` |
| D4 | temperature | 0 for all models | confirmed | `llm.temperature`, per-model `settings.temperature`, `decisions[D4]` | provider payloads in `llm/providers/*` |
| B1 | prompt configurations | bare, skills, placebo (more skills files may be added as new entries) | confirmed | `config/prompts_manifest.yaml`, `decisions[B1]` | `llm/prompts.py` |
| B2 | stock list | sector balanced, drawn from the S&P 500 | confirmed | `universe.source`, `universe.per_sector`, `decisions[B2]` | `data/universe.py` |
| B4 | universe size | 100 | confirmed | `universe.size`, `decisions[B4]` | `data/universe.py` |
| A2 | start | first live week is the week after the pilot passes | confirmed | `timing.live_start_week`, `decisions[A2]` | `pipeline/context.py` |
| A3 | live length | until `timing.live_end_date`, no later than the April deadline | confirmed | `timing.live_end_date`, `decisions[A3]` | `llm/cost.py` (weeks remaining) |
| A4 | closed Friday | the last prior trading day is the cutoff | confirmed | `decisions[A4]` | `calendar.py` |
| A6 | cold start | first 5 live weeks use IC 0.02 | confirmed | `views.cold_start_weeks`, `views.cold_start_ic`, `decisions[A6]` | `views/ic.py` |
| C1 | prompt fields | ticker, company name, sector, news | confirmed | `llm.prompt_fields`, `decisions[C1]` | `llm/prompts.py` |
| C2 | stock with no news | still call the model with the fixed sentence; no tools or web search | confirmed | `llm.no_news_sentence`, `decisions[C2]` | `llm/prompts.py` |
| C4 | skills file token cap | 6000 | confirmed, v1 value, rechecked in phase 15 G | `llm.skills_token_cap`, `decisions[C4]` | `scripts/check_placebo_length.py` |
| C7 | rationale token cap | 150 | confirmed, v1 value, rechecked in phase 15 G | `llm.rationale_token_cap`, `decisions[C7]` | `llm/validate.py` |
| E2 | IC used to scale the LLM view | IC of the residualized signal | confirmed | `views.ic_signal_for_llm`, `decisions[E2]` | `views/residualize.py`, `optimize/__init__.py` |
| E3 | combined score for H2 | ranked Black-Litterman tilt | confirmed | `eval.combined_signal`, `decisions[E3]` | `pipeline/jobs.py` (`update_ic_history`), `eval/hypotheses.py` |
| E4 | Kronos input prices | adjusted OHLCV | confirmed | `baseline.kronos.input_prices`, `decisions[E4]` | `baseline/combine.py` |
| E5 | market-cap weights | over the study's stocks only, renormalized | confirmed | `risk.market_cap_weights`, `decisions[E5]` | `risk/prior.py` |
| G3 | real H3 effect | paired weekly test passes after Holm, placebo as control | confirmed | `eval.holm_families`, `decisions[G3]` | `eval/hypotheses.py` |
| G4 | H4 splits | market up/down, volatility, earnings week | confirmed | `eval.h4_splits`, `decisions[G4]` | `eval/hypotheses.py` |
| G5 | factor regression | daily returns | confirmed | `eval.factor_frequency`, `decisions[G5]` | `eval/factors.py` |
| G6 | shuffled-ratings test | MVO, Robust MVO and Top-K portfolios plus their Q0s, 500 shuffles each; Mean-CVaR and Resampled MVO excluded (PLAN3) | retired (PLAN4) | `eval.shuffle.n_shuffles`, `eval.shuffle.strategies`, `decisions[G6]` | `eval/shuffle.py` |
| W1 | Slurm walltime | at most 15 hours per job, target 12 or less, no idle waiting on external work (PLAN3) | retired (PLAN4) | `decisions[W1]`, `scripts/arctic/arctic.env` | `scripts/arctic/common.sh` |
| M1 | Slurm memory | `--mem-per-cpu=3G` on CPU jobs, only the cores used (PLAN3) | retired (PLAN4) | `decisions[M1]`, `scripts/arctic/arctic.env` | `scripts/arctic/*.sbatch` |
| N1 | failure rule | invalid rate above 0.15, or local job or batch not finished by the lock | confirmed | `llm.invalid_rate_threshold`, `pipeline.lock_margin_minutes`, `decisions[N1]` | `llm/runner.py` (`assess_run`), `pipeline/jobs.py`, `pipeline/fallback.py` |
| N2 | weekly trigger | one hand-run script, no scheduler | confirmed | `decisions[N2]` | `scripts/run_weekend.py` |
| N3 | storage and backup | CSV files; dated folders downloaded weekly | confirmed | `backup.*`, `decisions[N3]` | `backup.py` |

## Current provisional item

- D3 budget: settle the paid API goal after prompt tests in v2 phase 15.

## Items marked UNVERIFIED until the real-service smoke test

Listed per phase in `plans/PROGRESS.md`; `scripts/smoke_real_services.py` exercises each one.
