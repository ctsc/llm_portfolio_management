# Open questions and study decisions

A mirror of the current decision table in `PLAN2.md` section 0.3. Decision values are not changed
by code. Rows marked provisional are the only open v2 study decisions.

Rule: changing any default after the live run starts must be logged in `PROGRESS.md` with the
date and reason and flagged in the analysis. Code never changes a decision value.

## Current v2 decisions

| ID | Setting | Value | Status |
|---|---|---|---|
| D1 | models | ARCTIC LiteLLM gateway: Gemma 4 31B, MiniMax M2.7; requested deployments: Gemma 4 E4B, Phi-4 Mini 3.8B, Llama 3.2 3B; API: one or two of Claude Haiku 5.5, GPT-6 Luna; no Qwen, DeepSeek, or local GPU | confirmed |
| D2 | gateway outage | use Carter's Slurm GPU jobs; otherwise mark that model's week missed and exclude it from its tests | confirmed |
| D3 | paid API budget | goal 50 USD; gateway models cost 0 USD | provisional |
| D4 | temperature | 0 for all models | confirmed |
| B1 | prompt types | bare, placebo, and 4 to 5 similar-length specialized skills files; one placebo matched to mean skills length | confirmed |
| B4 | universe size | 100 to 300, set after ARCTIC allocation is known | confirmed |
| C3 | output schema | JSON low, likely, high integers in [-5, 5], rationale; low <= likely <= high; no confidence field | confirmed |
| E1 | LLM view uncertainty | width = high - low; c = clip(1 - width/10, 0.1, 1); Omega_L,ii = tau Sigma_ii / c | confirmed |
| E2 | IC for LLM view | IC of the residualized likely score | confirmed |
| E6 | baseline view uncertainty | spread of Kronos's 20 sampled paths, mapped by K1 | confirmed |
| K1 | Kronos spread mapping | five-day return spread; ratio clipped to [0.25, 4]; scales Omega_b | confirmed (design choice) |
| S1 | steered strategies | MVO, Mean-CVaR, Robust MVO, Resampled MVO, Top-K; each has its own Q0 | confirmed |
| S2 | controls | HRP and equal weight; no LLM view, Q0, or shuffle test | confirmed |
| S3 | Mean-CVaR | 10,000 Monte Carlo scenarios from the BL posterior; beta 0.95 | confirmed |
| S4 | Mean-CVaR lambda | 1.0 | confirmed (design choice) |
| S5 | Robust MVO kappa | 1.0, ellipsoid from the BL posterior covariance of the mean | confirmed (design choice) |
| S6 | Resampled MVO draws | 200 | confirmed |
| S7 | Top-K | K = 20, equal weights | confirmed |
| G1 | guardrails | 5% cap, no shorts, 10 bps cost, -8% stop, +15% take-profit, weekly rebalance | confirmed |
| G3 | H3 | each skills file vs bare and placebo per model, plus H3b range-width test | confirmed |
| G3b | H3b width split | per source and week, narrow width at or below median vs the rest; narrow-minus-wide rank IC | confirmed (design choice) |
| P1 | skills length tolerance | each skills file within 5% of mean skills length; placebo within 2% | confirmed (design choice) |
| G4 | H4 | strategy ranking by mean weekly gap vs own Q0, plus v1 splits | confirmed |
| G6 | shuffle test | 500 shuffles for each steered portfolio, including each strategy's Q0 | confirmed |
| G7 | Holm families | IC tests as v1; portfolio return and shuffle tests one family per strategy | confirmed |
| N2 | weekly trigger | one hand-run submit script on ARCTIC; no scheduler | confirmed |
| A2, A3, A6, B2, C1, C2, C4, C7, E3, E4, E5, G5, N1, N3 | unchanged decisions | retain v1 values as listed in `PLAN2.md` | confirmed |

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
| C4 | skills file token cap | 6000 | provisional | `llm.skills_token_cap`, `decisions[C4]` | `scripts/check_placebo_length.py` |
| C7 | rationale token cap | 150 | provisional | `llm.rationale_token_cap`, `decisions[C7]` | `llm/validate.py` |
| E2 | IC used to scale the LLM view | IC of the residualized signal | confirmed | `views.ic_signal_for_llm`, `decisions[E2]` | `views/residualize.py`, `optimize/__init__.py` |
| E3 | combined score for H2 | ranked Black-Litterman tilt | confirmed | `eval.combined_signal`, `decisions[E3]` | `pipeline/jobs.py` (`update_ic_history`), `eval/hypotheses.py` |
| E4 | Kronos input prices | adjusted OHLCV | confirmed | `baseline.kronos.input_prices`, `decisions[E4]` | `baseline/combine.py` |
| E5 | market-cap weights | over the study's stocks only, renormalized | confirmed | `risk.market_cap_weights`, `decisions[E5]` | `risk/prior.py` |
| G3 | real H3 effect | paired weekly test passes after Holm, placebo as control | confirmed | `eval.holm_families`, `decisions[G3]` | `eval/hypotheses.py` |
| G4 | H4 splits | market up/down, volatility, earnings week | confirmed | `eval.h4_splits`, `decisions[G4]` | `eval/hypotheses.py` |
| G5 | factor regression | daily returns | confirmed | `eval.factor_frequency`, `decisions[G5]` | `eval/factors.py` |
| G6 | shuffled-ratings test | every portfolio plus Q0, 500 shuffles each | confirmed | `eval.shuffle.n_shuffles`, `decisions[G6]` | `eval/shuffle.py` |
| N1 | failure rule | invalid rate above 0.15, or local job or batch not finished by the lock | confirmed | `llm.invalid_rate_threshold`, `pipeline.lock_margin_minutes`, `decisions[N1]` | `llm/runner.py` (`assess_run`), `pipeline/jobs.py`, `pipeline/fallback.py` |
| N2 | weekly trigger | one hand-run script, no scheduler | confirmed | `decisions[N2]` | `scripts/run_weekend.py` |
| N3 | storage and backup | CSV files; dated folders downloaded weekly | confirmed | `backup.*`, `decisions[N3]` | `backup.py` |

## Current provisional item

- D3 budget: settle the paid API goal after prompt tests in v2 phase 15.

## Items marked UNVERIFIED until the real-service smoke test

Listed per phase in `PROGRESS.md`; `scripts/smoke_real_services.py` exercises each one.
