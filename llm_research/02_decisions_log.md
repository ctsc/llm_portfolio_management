# 02 Decisions log

Last updated: 2026-11-17. Section A is what the study does now. Section B lists decisions that
were replaced, with what replaced them, so old ideas are never mistaken for current ones.

Decision IDs (D1, S3, and so on) match `config/study.yaml` and `PLAN2.md` section 0.3 in the
repo. Where the repo config still holds an old value, that is flagged in
`10_open_items_and_next_steps.md`.

## A. Current decisions

### Scope and models

| ID | Decision | Notes |
|---|---|---|
| D1 | Open-weight models: Gemma 4 31B and MiniMax M2.7 on ARCTIC's LiteLLM gateway. API models: both Claude Haiku 5.5 and GPT-6 Luna (confirmed 2026-11-17) | No new deployments requested. No models banned by GSU's AI tool registry (Qwen, DeepSeek). No frontier reasoning models |
| Compute split (2026-11-10) | ARCTIC is used only for gateway LLM calls. Kronos, data, optimizers, simulator, evaluation and backups run on Carter's desktop | Confirmed by desktop benchmarks on 2026-11-10 (`04_models_compute_arctic.md`) |
| Gateway outage | If the gateway is down during a weekend run, that model's week is left out of its statistics | Replaces the old Slurm fallback (section B) |
| D3 | Paid API budget goal about 50 USD (provisional) | Must be recomputed with real prices after test prompts. A repo review estimated 100 to 200 USD per API model at 300 stocks, 7 prompt types and 24 weeks with placeholder prices |
| D4 | Temperature 0 for every model with a fixed seed (confirmed 2026-11-12). Andalib: the gateway passes the seed through, but outputs are not guaranteed identical because requests are batched with other users'. Both model cards recommend 1.0; the paper says why 0 was used |
| B1 | Prompt types (changed 2026-11-12): bare, 4 specialized skills files of similar length (`skills_5` stays off), and one placebo matched to their mean length; six types in total, ARCTIC sized for seven |
| P1 | Each skills file within 5% of the mean skills length; placebo within 2%, checked with every model's tokenizer | |
| C4, C7 | Skills cap 6,000 tokens; rationale cap 150 tokens (both provisional until real skills files exist) | |
| B2 | Stocks are sector balanced and drawn from the S&P 500 | A stock that leaves the S&P 500 during the study is not dropped |
| B4 | 100 stocks (decided 2026-11-16): ARCTIC granted the 1,000 GPU-hour startup allocation; 100 stocks is estimated at about 607 GPU-hours, 300 at about 1,820. Revisit only if measured timing shows a larger universe fits with a 25% margin |
| C1 | Prompt fields: ticker, company name, sector, that week's news | |
| C2 | A stock with no news still gets a call with "No relevant news is available for this week." No web search or tools | First snapshot uses 7 days of news |
| C3 | Output: JSON with low, likely, high integers in [-5, 5] (low <= likely <= high) and a short rationale. No separate confidence field | Score-first field order kept |
| N1 | Failure rule: a source falls back if more than 15% of outputs are invalid, or it is not finished by the lock deadline | Invalid outputs are excluded from residualization; the stock keeps only the baseline view |
| No reruns | One run per prompt type (no rerun controls) | |
| No backtest | The study is the live run only. No result depends on a backtest | LLMs may have seen any past period |

### Pipeline and views

| ID | Decision |
|---|---|
| Timing | All times US Eastern. Data cutoff Friday 16:00 ET; weights locked Monday 09:00 ET (30 minutes before the open); lock margin 30 minutes |
| A2 | First live week is the week after the pilot passes; the pilot week is excluded from results |
| A3 | Live end date set by Carter (`timing.live_end_date`) |
| A4 | If Friday is a market holiday, the last prior trading day is the cutoff |
| A6 | Cold start and trailing-IC gating: kept in the code only for `views.ic_mode: trailing` (tests); the live study uses the fixed scale A8 |
| A8 | Fixed view scale (2026-11-12): IC_b and IC_L are fixed at 0.02 every week for every source, so no view is switched on or off by a noisy 8-week IC (its standard error is about 0.02 at 300 stocks). Weekly ICs are still computed for evaluation |
| E1 | LLM view uncertainty from range width: width = high - low, c = clip(1 - width/10, 0.1, 1), Omega_L,ii = tau Sigma_ii / c |
| E2 | The LLM view is scaled by the IC of the residualized likely score |
| E3 | The H2 combined signal is the ranked Black-Litterman tilt |
| E4 | Kronos uses adjusted prices |
| E5 | Market-cap weights over the study stocks only, renormalized |
| E6 / K1 | Baseline uncertainty from the spread of Kronos's 20 sampled five-day paths: r_i = clip(k_sd_i / median(k_sd), 0.25, 4), Omega_b,ii = tau Sigma_ii r_i |
| tau | 0.05; risk aversion delta 2.5; covariance lookback 252 days, Ledoit-Wolf shrinkage |
| Ledger | Append-only raw price ledger with dividends and splits; explicit week labels |
| Stops | A stop triggered right before a rebalance merges into the Monday trade |
| N2 | No scheduler. Carter runs `scripts/run_weekend.py` by hand on his desktop, ideally Friday night (Saturday morning as backup), so he sees any issue |
| N3 | All data in CSV files; dated backup each week |

### Strategies and guardrails

| ID | Decision |
|---|---|
| S1 | Five steered strategies: MVO, Mean-CVaR, Robust MVO, Resampled MVO, Top-K. Each has its own Q0 (baseline view only) |
| S2 | Controls: HRP and equal weight, which use no forecasts |
| S3 | Mean-CVaR: 10,000 Monte Carlo scenarios from the Black-Litterman posterior, beta 0.95 |
| S4 | Mean-CVaR lambda: was fixed at 1.0; now set by the S8 calibration on synthetic data before the freeze |
| S5 | Robust MVO kappa 1.0, ellipsoid from the posterior covariance of the mean |
| S6 | Resampled MVO: 200 draws |
| S7 | Top-K: K = 20, equal weights |
| G1 | 5% max per stock, no shorts, 10 bps per trade, daily stop-loss -8% and take-profit +15%, weekly rebalance |
| Shared posterior | One posterior per rating source, reused across strategies, so strategy comparisons differ only in allocation |

### Evaluation (2026-11-11)

| Decision | Detail |
|---|---|
| Drop the shuffle test | No shuffled-rating portfolios at all. Reasons in `07_dropped_methods_and_why.md` |
| No method may add CPU or GPU hours | Every test runs on saved ratings, weights and returns in seconds to minutes |
| Methods | Rank IC and residual IC (Fama-MacBeth), active return vs Q0, information ratio, Ledoit-Wolf Sharpe test, break-even cost, Fama-French 5 plus momentum regressions of the active return, placebo prompt as the content control (chance is handled by the tests; market and style luck by Q0 and the factor model). Full list in `05_evaluation_methodology.md` |
| Multiple testing | Holm on a small set of main comparisons fixed before the live run; Benjamini-Hochberg on everything else (secondary); H4 exploratory |
| Correspondence | Emails with Parag and Andalib stay local; `11_correspondence.md` is gitignored |
| Reporting | Every result states the smallest effect the study could detect. H2 reported over all weeks and over active-view weeks only |
| Conventions | Sharpe net of the T-bill rate; Sortino, max drawdown and CVaR from daily returns; Student-t critical values |
| Factor data | Kenneth French daily factors for the final regressions; ETF proxies only for live monitoring |
| G5 | Factor regressions use daily returns |
| G13 | View sensitivity (changed 2026-11-11 by review 5): measured as expected weekly tracking error against Q0, not view shift. A strategy below the pre-registered `view_sensitivity_min_tracking_error` is reported as insensitive to views by construction. No default; Carter sets it from the calibration and the pilot |
| G14 | The H2 signal test compares the steered tilt's IC with Q0's tilt IC (not the raw baseline score), and every IC gap uses the same stocks each week |
| U1 | Large moves: a 5-day move of 10% or more either way is kept and flagged; a fall of 40% or more removes the stock permanently (no replacement, position sold at the next open); a raw jump the adjusted prices do not show is a data error that skips the stock for that week only |
| N4 | Week-to-week integrity: weekly returns Monday open to Monday open; a study log row per week; keep-awake, VPN outage retries, checkpoints and progress in the weekly run |
| G15 | Gateway output handling per model (2026-11-11, corrected by review 5): `max_output_tokens` Gemma 300 and MiniMax M2.7 about 2,000 until measured (per-model settings already exist in config); a per-model request timeout; reasoning text saved; tokens, `finish_reason` and seconds logged per request; accepted `seed` and JSON-mode features frozen after the pilot. No strip rule for Gemma's marker is needed because the parser takes the first JSON object |
| G16 | Price data comes from a consolidated feed: Alpaca `feed` becomes a setting that defaults to `sip`, the smoke check confirms SIP bars, and yfinance is the fallback if the free plan refuses SIP (2026-11-11; the code had `iex`, one small exchange) |
| G17 | ARCTIC sizing (revised 2026-11-12): 300 to 8,000 input tokens; about 300 output for Gemma, up to about 2,000 for MiniMax; estimated 40 GPU-seconds per Gemma request and 90 per MiniMax request; 1,820 estimated and 2,730 requested at 300 stocks, 607 and 910 at 100; weekly token totals listed because metering is undecided. Corrected again after the timing run and ARCTIC's metering answer |
| E7 | Neutral LLM view (2026-11-11, review 5, approved by Carter): `q_L = mu_Q0 + IC_L * sigma * z_L`, centered on the Q0 forecast and not on the market prior, so a rating of 0 leaves the forecast mean equal to Q0's and MVO and Top-K equal Q0 when the LLM says nothing. Replaces `q_L = pi + IC_L * sigma * z_L` |
| G18 | Confidence-only counterfactual: the evaluation records how much each LLM view lowers posterior uncertainty with `z_L = 0` (one linear solve), reported next to H2 and used by G13 |
| S8 | Strategy sensitivity calibration (approved by Carter 2026-11-11): Mean-CVaR `lam` and a new Resampled `draw_scale` are set on the synthetic world at 300 stocks by a fixed rule (the largest grid value whose tracking error under noise ratings is at least 25% of MVO's), before the freeze; Carter confirms the values |
| G1b | Stop on a rebalance day (approved by Carter 2026-11-11): when a pending stop is merged into the Monday rebalance and the position is kept, its reference price resets to the rebalance open |
| N5 | A gateway request that fails after its retries is recorded as invalid (`provider_error`) and counts toward the 15% rule; J4 is not done while a request has no record |
| C8 | Prompt name and GICS sector come from the frozen universe file; yfinance supplies market caps only, with the last good value reused on failure |
| A7 | Data-folder hygiene (pilot rule decided 2026-11-12): dry runs use a scratch data folder; the preflight refuses a live run with pre-pilot weeks present; the pilot never counts toward IC or results, and every portfolio starts from cash at the first live week; the ledger is rebuilt at the freeze |
| U2 | A held stock with no price for 5 trading days is liquidated at its last close (confirmed 2026-11-12) |
| D5 | Each API model is pinned to a dated version before a live run |
| G7 | Multiple testing (changed 2026-11-12, revised 2026-11-17): Holm on exactly four pooled main comparisons (LLM-minus-baseline rank IC gap on the same stocks, matching the proposal's RQ1; steered-minus-Q0 tilt IC; skills minus bare; skills minus placebo), each pooled over models (and skills files) with model fixed effects; Benjamini-Hochberg on everything else, including pooled LLM rank IC above zero; H4 exploratory |
| G22 | Measure everything in one fixed scope (2026-11-17): every metric (rank IC, residual IC, hit rate, view shift, annualized Sharpe net of the T-bill rate, PSR, DSR, Sortino, Calmar, max drawdown, CVaR, IR, TE, CER, turnover, costs, break-even cost, factor alpha and loadings) is computed and saved for every source, portfolio, model, prompt and strategy, per cell and pooled, in `results/eval/all_metrics.csv`; null results are reported; no models, prompts, strategies or stocks are added after the freeze |
| A9 | News cutoff Friday 16:00 ET, unchanged (confirmed 2026-11-12); the 2.7-day news age is stated as a limitation |
| K3 | Kronos failure: `momentum_only` (2026-11-12); every portfolio uses the momentum-only baseline that week and the week is flagged |
| G19 | Confidence-only control portfolios (2026-11-12) for Robust and Resampled MVO: same LLM view uncertainty, rating 0; splits steered minus Q0 into information and confidence. Not for Mean-CVaR (cost) or MVO and Top-K (equal to Q0 under E7) |
| G20 | Pilot repeat check (2026-11-12): 50 fixed stocks re-rated under all six prompt types per gateway model, bypassing the cache; agreement reported per model and prompt |
| G21 | Weekly model checks (2026-11-12, revised 2026-11-16 after Andalib): log each response's `model` field, `x-litellm-model-id` and `x-litellm-version` headers and `reasoning_tokens`; save `/v1/models` and `/v1/model/info` (`updated_at`) each weekend; 10 canary prompts per model before J4. A changed model name, deployment ID or canary drop flags the week (kept in main results, dropped in a secondary analysis); a gateway version or `updated_at` change is logged as information only. The gateway has no `system_fingerprint` |
| N6 | Logging and durable progress (2026-11-12): a `run_id` on everything, an event timeline per run, full tracebacks for every error, Python warnings and library logs captured, a weekly `fallbacks.csv` listing every degradation, fsync on every write, a disk check before each job, `run_summary.md` per run, `portstudy audit` per week, logs never deleted and always backed up; tested by killing the run at random points and resuming |
| N7 | ARCTIC usage budget (2026-11-16): the study log tracks estimated GPU-hours used, a running total and an end-of-study projection against the 1,000 allocation; warning above 75%, alert above 90%; Carter checks real usage in Elpis monthly |
| ARCTIC scope | Small models and Slurm not re-added when ARCTIC offered (2026-11-12) |
| API models | Keep both Claude Haiku 5.5 and GPT-6 Luna if the cost estimate fits the 50 USD goal (2026-11-12) |
| Kronos commit | The latest commit on setup day, pinned and never changed after the freeze (2026-11-12) |
| Naming | The proposal and docs say research questions RQ1 to RQ5 with a hypothesis per question at the end of the proposal; code and report keep H1 to H5 |
| G4 | H4 splits: market up or down, volatility, earnings week. Exploratory |

### Documents

| Decision | Detail |
|---|---|
| Proposal references | Trimmed to 51 (10 removed after reading the abstracts; see `08_literature_and_references.md`) |
| Writing rules | First person in ARCTIC documents (no third-person self-reference). No em dashes. No invented facts. No lines ending on 2 to 5 words. The banned word for experimental conditions is never used (`scripts/check_banned_words.py`) |

## B. Superseded decisions (history only, not current)

| Date | Was | Now |
|---|---|---|
| 2026-10-20 | MS research on AI citation hallucination | Dropped for this study (`12_ai_hallucination_research.md`) |
| 2026-10-19 | Cost-per-token ROI framing of LLM trading | Dropped; study is about skills and strategies |
| 2026-10-25 | Local models Qwen 3.5 4B and Phi-4 Mini on the RTX 3050 | Qwen is banned by GSU's tool registry; open-weight models now come from ARCTIC's gateway |
| 2026-10-25 | Universe fixed at 100 stocks; one skills file | 300 preferred (100 fallback); 4 to 5 skills files |
| 2026-10-25 | One fixed optimizer (Black-Litterman plus MVO) | Five fixed strategies, each with its own Q0 (2026-11-01) |
| 2026-10-25 | LLM gives one score plus a confidence | Low / likely / high range on -5 to 5 (2026-11-01) |
| 2026-11-02 | "Investor mindset" prompts | "Specialized skills" prompts |
| 2026-11-02 | Ask ARCTIC to deploy Gemma 4 E4B, Phi-4 Mini 3.8B and Llama 3.2 3B | Dropped 2026-11-10 so the request does not risk denial; only the two existing gateway models |
| 2026-11-02 | If ARCTIC accepts, the local GPU is not used | Kronos now runs on the local GPU |
| PLAN2/PLAN3 | Gateway outage falls back to Carter's own Slurm GPU jobs (D2) | Dropped; the model's week is left out |
| PLAN3 | Weekend jobs as Slurm jobs on ARCTIC (W1 walltime of 15 h or less, M1 memory 3 GB per core, N2 submit script on ARCTIC) | Everything except LLM calls runs on the desktop; `run_weekend.py` is run by hand there |
| Request history | ARCTIC request asked for 15,000 CPU hours, then 1,500 CPU hours plus GPU and storage | GPU hours for gateway LLM calls only; 0 CPU hours, no extra storage |
| Shuffle history | Shuffle on the best portfolio per model plus Q0; then 500 shuffles on all portfolios; then G6 (MVO, Robust and Top-K only, on ARCTIC CPUs) | Dropped entirely 2026-11-11 |
| Multiple testing | Holm across all sources or within each strategy for every test, then 5 to 10 pre-registered main comparisons | Four pooled pre-registered main comparisons only; Benjamini-Hochberg for the rest |
| Factor data | ETF proxies (SPY, IWM, IWD, IWF, QUAL) for H5 | French daily FF5 plus momentum for the final analysis |
| Pilot dates | Earlier targets: pilot Nov 7 to 14, 2026, live from Nov 15 | Live run starts the week after the pilot passes (A2) |
| References | 20, then 54, then 61 | 51 |
| 2026-11-12 | Trailing 8-week IC gating with cold start (A6) | Fixed view scale 0.02 (A8) |
| 2026-11-12 | 4 to 5 skills files | 4 skills files |
| 2026-11-12 | Kronos failure holds every portfolio | `momentum_only` (K3) |
| 2026-11-12 | ARCTIC request 1,700 / 600 GPU-hours with 300 output tokens for every request | 2,730 / 910 with MiniMax at about 2,000 output tokens |
| 2026-11-16 | Universe 300 stocks preferred, 100 as the smaller option | 100 stocks (startup allocation of 1,000 GPU-hours) |
| 2026-11-17 | Main comparison 1: pooled LLM rank IC above zero | Pooled LLM-minus-baseline rank IC gap (G7); IC above zero is secondary |
| 2026-11-17 | One or two API models | Both Claude Haiku 5.5 and GPT-6 Luna (D1) |
