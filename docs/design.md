# Design: one page per pipeline stage

One section per pipeline stage. Decisions are in `docs/open_questions.md` and `plans/PLAN4.md`
section 0.3. Each stage names the module that implements it. Where the code still differs from the
current design, the section says which PLAN4 phase changes it.

## Source and portfolio identities (`types.py`)

A rating source is `<model>__<prompt-id>` and is independent of the strategy using its ratings.
Strategy-steered portfolios use `<strategy>__<model>__<prompt-id>`, each strategy's baseline-only
portfolio uses `<strategy>__Q0`, and controls use `HRP` or `EW`. v2 phase 7 enables five steered
strategies and both controls under the shared 5 percent, no-shorts guardrails. IC history is
recorded per rating source rather than per portfolio.

## v2 phase 7 strategy package

The strategy registry exposes MVO, Mean-CVaR, Robust MVO, Resampled MVO, and Top-K, plus the
forecast-free HRP and equal-weight controls. Mean-CVaR uses 10,000 common posterior scenarios at
beta 0.95 and lambda 1.0. Robust MVO subtracts the posterior-mean uncertainty norm with kappa 1.0.
Resampled MVO averages 200 MVO solutions over common mean draws. Top-K gives 20 highest-mean names
equal weights, breaking boundary ties by ticker. Every portfolio is long-only with a 5 percent
maximum position. Solver failures share the previous-weight, same-strategy Q0, then equal-weight
fallback chain. HRP clusters correlation distance from Sigma and never reads mu.

## v2 phase 8 weights and replay

J7 writes one baseline-only posterior, strategy-specific Q0 weights, both controls, and then one
source-level posterior reused by all strategies. Posterior CSVs retain the inputs and signals needed
for IC history; the full covariance-of-mean matrix is represented by its SHA-256 in weight metadata.
Failed sources and sources arriving after the lock deadline copy each strategy's Q0 CSV and metadata.
The fallback plan is keyed by rating source, while replay plans use each portfolio's persisted
`decision_source`. HRP and equal weight are replayed like every other portfolio.

## v2 phase 9 scale (`optimize/problem_cache.py`, `data/universe.py`, `scripts/benchmark_pipeline.py`)

Every cvxpy problem is compiled once per constant data and reused with `mu` as a `cp.Parameter`:
MVO is keyed by (tickers, SHA-256 of Sigma, delta, cap); robust MVO adds the SHA-256 of M, because
its uncertainty factor changes with every source; mean-CVaR adds M, the week, the seed, and the
scenario settings, because the scenario shocks `Z L'` are built from `Sigma + M`. Resampled MVO
therefore compiles once and solves 200 draws against the cached problem, and every source in a
week reuses the same compiled problem. Solves run with `warm_start=False`
so a cached solve equals a fresh build to 1e-9. The cache holds 16 problems (LRU) and the ticker
tuple is in every key, so a universe change misses rather than reusing stale data.

The universe builder keeps the v1 rule (top floor(size / sectors) per sector by dollar volume,
then the highest-volume leftovers) and logs `universe_thin_sector` for each sector short of its
quota, `universe_leftover_slots` with the slots filled from leftovers, and `universe_short` if
the size still cannot be reached. `scripts/benchmark_pipeline.py` times every strategy and control
on a synthetic 3-factor covariance at a chosen size and projects the weekly J7 time at the full
source count.

## Week alignment (`calendar.py`)

`week_id` is the ISO week of the cutoff Friday (the prior trading day if the Friday is closed,
decision A4). Scores made at that cutoff trade at the next Monday open (`trade_date`, or the
next trading day if Monday is a holiday) and the return window runs to the following Monday
open (`window_end_date`). Every table that holds returns stores all three columns.

## Snapshot (`data/snapshot.py`)

Prices for at least 300 trading days back to the cutoff, news in `(previous cutoff, cutoff]`
(7 calendar days for the first snapshot), reference data, rates, ETF prices. Alpaca bars use the consolidated SIP feed (`data.alpaca_feed`, G16), with yfinance as the fallback
provider. The company name and GICS sector in `reference.csv` come from the frozen universe file,
not from the weekly reference call (C8); that call supplies market caps and earnings dates only,
and a missing cap is carried from the most recent earlier snapshot (`market_cap_carried`). News published
after 16:00 ET on the cutoff day is dropped strictly, near-duplicates are removed, and each
stock's text is capped at the token budget (newest first). The checks in `data/checks.py` map
each issue code to `skip_stock`, `forward_fill`, or `halt`. The folder is immutable
(`--force-new-version` writes `<week>.v2`). Each new trading day's raw OHLC and corporate
actions are appended to the ledger and never rewritten.

## Momentum and baseline (`baseline/`)

```
M12_1 = P[t-21]/P[t-252] - 1 ;  M3 = P[t]/P[t-63] - 1 ;  R1 = -(P[t]/P[t-21] - 1)
u(x)  = rank(x)/N
m     = mean(u(M12_1), u(M3), u(R1))
r_ij  = P_hat_i,j[t+5] / P_i[t] - 1
k_i   = mean_j(r_ij) ;  k_sd_i = sd_j(r_ij, ddof=1) ;  s_i = k_sd_i
b = 0.5 u(m) + 0.5 u(k) ;  z_b = (b - mean(b)) / sd(b)      (sd with ddof=1)
```

Adjusted close is used; stocks missing any of the 252-day history are excluded for the week
and logged. Kronos takes adjusted OHLCV (E4); the stub forecaster is deterministic drift plus
seeded noise. `baseline.csv` stores `k_sd`; paths with a non-positive close are dropped and
logged, and `k_sd` is NaN when fewer than two valid paths remain. v2 phase 6 uses this spread for
the DECISION(K1) baseline uncertainty ratio.

## Risk model and prior (`risk/`)

```
Sigma_daily = (1 - alpha) S + alpha (tr(S)/N) I  (Ledoit-Wolf) ;  Sigma = 5 Sigma_daily
sigma_i = sqrt(Sigma_ii)
w_mkt = market_cap / sum(market_cap)  over the study tickers only (E5)
pi    = delta Sigma w_mkt ;  delta = 2.5
```

Sigma is asserted symmetric and positive definite (smallest eigenvalue above 1e-10). The
baseline and the risk model share one ticker set (`baseline/common.py`).

## Prompts (`llm/prompts.py`)

Shared prefix first (base instructions plus the configuration's file), then the stock part
(ticker, company name, sector, news; C1). Prompt configurations are selected by manifest kind:
`bare`, `placebo`, and `skills_1` through `skills_5`, with `skills_5` disabled by default. The
placebo matches the enabled skills mean, and skills file lengths are checked against that mean.
No price statistics anywhere. A stock with no news gets the fixed sentence (C2). Output schema:

```
{"low": <integer -5..5>, "likely": <integer -5..5>, "high": <integer -5..5>, "rationale": "<short reason>"}
```

The three integer ratings are valid only when `low <= likely <= high`. Extra keys are ignored and
recorded. Invalid output stores `low = likely = high = 0`, `valid = false`, and an empty width.

## LLM runner (`llm/runner.py`)

Open-weight models use the OpenAI-compatible LiteLLM provider with a bounded worker pool, one
stock per call, temperature 0, and no tools. The serving endpoint is included in the cache key
and stored per source-week. The main thread validates and writes completed records so a crash can resume.
Invalid output: retry once, then store
`low = likely = high = 0`, an empty width, and `fallback_flag = invalid_output`
(`invalid_truncated` when the answer hit the output limit). Valid width is `high - low`. Cache
key `sha256(model_version, prompt_hash, ticker, week_id, settings, model_id, endpoint)`. Cost
guard before any batch: projected study total above the goal raises unless
`--allow-over-budget` (D3). Failure rule (N1): invalid rate above 0.15, or open-weight job or
batch not finished by the lock deadline minus the margin.

## Rank IC, residual, views (`views/`)

```
Rank IC:    Spearman(likely rating at Friday cutoff, total return from next Monday open to the following Monday open)
View scale: IC_b = IC_L = views.fixed_ic (0.02) every week (A8, ic_mode: fixed); the trailing
            mean of the last 8 scored weeks is kept for ic_mode: trailing
Residual:   s = a + c z_b + e ;  z_L = standardize(e)        (invalid outputs excluded)
Views:      q_b = pi + IC_b sigma z_b ;  q_L = mu_Q0 + IC_L sigma z_L
            (E7: centered on the Q0 posterior mean, so a rating of 0 is neutral)
Conf-only:  same Omega_L with z_L = 0 (G18); its posterior covariance M_conf drives the
            Robust and Resampled confidence-only controls (G19);
            conf_only_uncertainty_ratio = trace(M_conf) / trace(M_Q0)
Expected TE vs Q0: sqrt((w - w_Q0)' Sigma (w - w_Q0)), in every steered weight meta
Uncertainty: r_i = clip(k_sd_i / median(k_sd), 0.25, 4)
             Omega_b,ii = tau Sigma_ii r_i
             c_i = clip(1 - width_i / 10, 0.1, 1) ;  Omega_L,ii = tau Sigma_ii / c_i
             tau = 0.05
Zero IC:    IC_L = 0  ->  drop LLM view rows, weights = Q0 weights
```

Only weeks whose return window closed before the cutoff enter the trailing IC; fallback weeks
are skipped; fewer than 3 clean weeks keeps 0.02. A stock with no valid LLM output keeps only
the baseline view.

## Black-Litterman and optimizer (`optimize/`)

```
BL:         A = inv(tau Sigma) + P' inv(Omega) P
            mu = solve(A, inv(tau Sigma) pi + P' inv(Omega) q) ;  M = inv(A)
            P = [I; I_L], where I_L selects stocks with valid LLM views
Q0:         same with P = I, q = q_b, Omega = Omega_b
Optimizer:  max w' mu - (delta/2) w' Sigma w   s.t. sum w = 1, 0 <= w_i <= 0.05
```

`black_litterman` builds A once, factors it with Cholesky, and uses triangular solves for both
`mu` and `M`, returned as `Posterior.cov_mu`. A failed factorization retries once with diagonal
jitter of `1e-12 * trace(A) / N`, then raises `OptimizerError`. OSQP, then CLARABEL; on failure
the previous week's weights are used with `fallback_flag = solver_failed`. At least 20 stocks are
required for the cap. The optimizer problem is compiled once per (tickers, Sigma, delta, cap) with
`mu` as a parameter and reused (see v2 phase 9 above).

## Simulator (`sim/`)

```
Cost:   C = 0.001 * sum_i |trade value_i|
Stops:  sell at next open if close <= 0.92 avg_cost  or  close >= 1.15 avg_cost  (checked at every daily close)
```

Start at 100000 cash. Rebalance at the Monday open at raw ledger prices, fractional shares,
sells before buys, buys scaled proportionally if cash is short after costs. Daily mark at the
raw close. A stop queued for the same open as a rebalance becomes one net trade with
`stop_trigger` recorded and one cost charge. Average cost is share-weighted on buys and
unchanged by sales. Dividends credit cash on the ex-date; splits rescale shares and average
cost. The simulator reads only the ledger, so past records never change when a provider
re-adjusts history.

## Fallback policy (`pipeline/fallback.py`)

A failed source trades the corresponding strategy's Q0 weights with `decision_source = fallback_q0` and
the week is marked `fallback` for that portfolio. A data or baseline failure holds every
portfolio (`no_rebalance`). `clean_weeks` returns the normal weeks for IC history and tests.

## Evaluation (`eval/`)

**Current design (the proposal's Evaluation section; built in PLAN4 Phases 4 to 9).** Every test
uses saved ratings, weights and returns and reruns no model or optimizer. Rank IC is computed
across stocks within a week, so market direction does not move it; portfolios are judged on their
active return, the weekly return minus the same strategy's Q0.

- H1: weekly IC gap (LLM likely minus baseline, both on the same stocks) per source with Newey-West SE, Student-t critical
  values, block-bootstrap CI and sign-flip p-value; residual IC from weekly Fama-MacBeth regressions
  on the baseline score, momentum, size, volatility, beta, sector and trailing value, profitability
  and investment loadings; ICIR, hit rate, quintile spread.
- H2: IC of the steered BL tilt minus IC of Q0's tilt, on the same stocks; per steered portfolio the mean active return vs Q0, information
  ratio, tracking error, Ledoit-Wolf Sharpe-difference test, extra turnover and break-even cost,
  reported over all clean weeks and over active-view weeks.
- H3: skills minus bare and minus placebo on IC and on active return, plus a pooled regression with
  model and week fixed effects clustered by week; model-vs-model comparisons on the same prompt.
- H3b: narrow minus wide IC, a score-by-confidence interaction, and a width vs absolute-error check.
- H4 (exploratory): strategy ranking, strategy-vs-strategy active-return differences, regime
  interaction regressions.
- H5: daily active return regressed on Fama-French five factors plus momentum (Kenneth French daily
  data) with Newey-West SE; the placebo prompt as the content control (length and any view change; chance is handled by the tests); deflated Sharpe and a t > 3 flag for any
  headlined portfolio.
- Multiple testing: Holm on four pooled main comparisons fixed in `config/preregistration.yaml` before
  the live run; Benjamini-Hochberg on everything else; H4 unadjusted.
- Metrics: Sharpe net of the T-bill rate, Sortino, maximum drawdown and CVaR from daily returns,
  certainty-equivalent return, turnover, cost, stops, view shift, fallback and solver-failed weeks.
- Minimum detectable effect `2.8 * sd / sqrt(T)` is printed for every main test; a gap below it is
  inconclusive, not evidence of no effect.

Formulas and reasoning: `llm_research/05_evaluation_methodology.md` and
`06_evaluation_research_findings.md`.

**Module map (`src/portstudy/eval/`).**

| Module | Implements |
|---|---|
| `data.py` | `EvalData`: open-to-open weekly returns, daily values, trades, statuses, the stock-week panel with characteristics, active-view, expected-TE and uncertainty-ratio tables, model-change weeks, factors |
| `tests.py` | `mean_test` (Newey-West and t(T-1)), `sign_flip_p` (exact up to T = 24), `block_bootstrap_ci` (stationary, percentile-t), `paired_summary`, Ledoit-Wolf, Jobson-Korkie-Memmel, PSR, DSR |
| `multiple.py` | `fdr_bh`, `assign_families` (Holm primary, BH secondary, H4 exploratory) |
| `metrics.py` | Sharpe net of the T-bill rate, Lo Sharpe, Sortino, drawdown, Calmar, CVaR, IR, TE, CER, hit rate, break-even cost, `arch` SPA, StepM, MCS |
| `factors.py` | French daily FF5 plus momentum loader, coverage check, ETF proxies, synthetic French files |
| `characteristics.py` | stock-week controls as of each cutoff (G9) |
| `signal_tests.py` | H1, H2 signal, H3 signal, model comparisons, H3b interaction and calibration, pooled regressions, the four primary rows |
| `portfolio_tests.py` | H2 portfolio (ITT and active weeks), G19 split, H3 portfolio and the placebo control, model comparisons, controls, H4, H5, SPA and StepM, view sensitivity, model-change sensitivity |
| `hypotheses.py` | portfolio metrics table, H3b narrow minus wide, H4 ranking and splits, earnings exposure |
| `all_metrics.py` | long-form `all_metrics.csv` and the completeness check (G22) |
| `report.py` | `report.md`, every CSV, figures; `portstudy eval` |

## Weekly run on the desktop (`scripts/run_weekend.py`)

Carter runs `scripts/run_weekend.py` by hand in WSL2 on the desktop (decision N2), Friday night or
Saturday morning. Step 1 replays and backs up any earlier week that is behind (J8, J9). Step 2 runs
J1 to J6 in order: J2 runs Kronos on the local GPU, J3 submits the API batches, J4 calls the two
gateway models with one worker per model (`llm.parallel_models`) and `llm.concurrency` requests at
a time, J5 polls the batches with a countdown until they are back or the lock margin, and J6
validates. Step 3 runs J7 and J9. Before J4 a health check sends one tiny call per gateway model; a
model that fails is marked failed for the week and its sources trade Q0 (D2). If the gateway comes
back before the lock, `python scripts/run_weekend.py --from J4` runs the health check again and J4
retries that model (the health result is part of J4's input hash). Every job resumes from its
status files.

## v2 in one page

- **Rating source vs portfolio.** A rating source is `(model, prompt kind)`; it produces one set of
  low/likely/high ratings per stock and week. A portfolio is `(strategy, source)`, a strategy's own
  Q0, or a control. Sources are what the LLM calls pay for; portfolios are what the simulator runs.
  Count: `(models x prompt kinds + 1) x 5 + 2`.
- **Strategy package.** MVO, Mean-CVaR (10,000 posterior scenarios, beta 0.95, lambda 1), Robust MVO
  (kappa 1), Resampled MVO (200 draws), Top-K (20 names, equal weight); controls HRP and equal weight
  never read views or Q0 weights.
- **Posterior covariance.** Black-Litterman returns `mu_BL` and `M = inv(inv(tau Sigma) + P' inv(Omega) P)`,
  the covariance of the posterior mean; CVaR scenarios draw from `Sigma + M`, Robust uses `chol(M)`,
  Resampled draws `mu_d = mu_BL + chol(M) z_d`.
- **Uncertainty.** Baseline `Omega_b,ii = tau Sigma_ii clip(s_i / median(s), 0.25, 4)` with `s_i` the
  Kronos path spread (K1); LLM `Omega_L,ii = tau Sigma_ii / c_i`, `c_i = clip(1 - width_i / 10, 0.1, 1)`.
- **Common random numbers.** Scenario and draw matrices are seeded by `(seed, week, strategy)`, so every
  source and each Q0 of one strategy see the same `Z` in one week; differences between
  portfolios come from the views alone.
- **Fallback chain.** Solver failure -> previous weights -> the same strategy's Q0 -> equal weight. A
  failed source (crash, invalid storm, batch not back, gateway health failure) trades its strategy's Q0
  with `decision_source = fallback_q0`; `IC_L = 0` drops the LLM view rows and the weights equal Q0 exactly.
- **Gateway outage (D2).** A model that fails the pre-J4 health check is marked failed for the week;
  its sources trade Q0 and that week is left out of their statistics. There is no fallback server.
- **Guardrails at each rebalance.** The 5 percent cap, no shorts, and full investment hold at every
  rebalance for every strategy and control; between rebalances weights drift with prices, as in v1, so a
  position can sit above 5 percent until Monday's open. Stops (-8 percent, +15 percent) and the 10 bps
  cost are unchanged.
