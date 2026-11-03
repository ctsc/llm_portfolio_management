# Design: one page per pipeline stage

The formulas below describe the v1 pipeline stages retained in Phase 1. `PLAN2.md` is the
current specification and phases 2 onward replace the formulas they change. Each stage names
the module that implements it.

## Source and portfolio identities (`types.py`)

A rating source is `<model>__<prompt-id>` and is independent of the strategy using its ratings.
Strategy-steered portfolios use `<strategy>__<model>__<prompt-id>`, each strategy's baseline-only
portfolio uses `<strategy>__Q0`, and controls use `HRP` or `EW`. Phase 1 enables MVO only and
does not enable controls. IC history is recorded per rating source rather than per portfolio.

## Week alignment (`calendar.py`)

`week_id` is the ISO week of the cutoff Friday (the prior trading day if the Friday is closed,
decision A4). Scores made at that cutoff trade at the next Monday open (`trade_date`, or the
next trading day if Monday is a holiday) and the return window runs to the following Monday
open (`window_end_date`). Every table that holds returns stores all three columns.

## Snapshot (`data/snapshot.py`)

Prices for at least 300 trading days back to the cutoff, news in `(previous cutoff, cutoff]`
(7 calendar days for the first snapshot), reference data, rates, ETF prices. News published
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
Kronos: k_i = mean_over_20_paths( P_hat[t+5] / P[t] ) - 1
b = 0.5 u(m) + 0.5 u(k) ;  z_b = (b - mean(b)) / sd(b)      (sd with ddof=1)
```

Adjusted close is used; stocks missing any of the 252-day history are excluded for the week
and logged. Kronos takes adjusted OHLCV (E4); the stub forecaster is deterministic drift plus
seeded noise.

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
and stored per source-week. Switching to a fallback archives gateway records and restarts the
source there. The main thread validates and writes completed records so a crash can resume.
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
Trailing IC: mean of last 8 scored weeks, floored at 0 ; fixed 0.02 for the first 5 live weeks
Residual:   s = a + c z_b + e ;  z_L = standardize(e)        (invalid outputs excluded)
Views:      q_b = pi + IC_b sigma z_b ;  q_L = pi + IC_L sigma z_L
Uncertainty: Omega_b,ii = tau Sigma_ii ;  Omega_L,ii = tau Sigma_ii / clip(1 - width_i / 10, 0.1, 1) ;  tau = 0.05
Zero IC:    IC_L = 0  ->  drop LLM view rows, weights = Q0 weights
```

Only weeks whose return window closed before the cutoff enter the trailing IC; fallback weeks
are skipped; fewer than 3 clean weeks keeps 0.02. A stock with no valid LLM output keeps only
the baseline view.

## Black-Litterman and optimizer (`optimize/`)

```
BL:         mu = [ inv(tau Sigma) + P' inv(Omega) P ]^-1 [ inv(tau Sigma) pi + P' inv(Omega) q ]   P = [I; I]
Q0:         same with P = I, q = q_b, Omega = Omega_b
Optimizer:  max w' mu - (delta/2) w' Sigma w   s.t. sum w = 1, 0 <= w_i <= 0.05
```

Omega is diagonal, so the system is solved as `(I + tau Sigma D) mu = pi + tau Sigma d`
without inverting Sigma. OSQP, then CLARABEL; on failure the previous week's weights are used
with `fallback_flag = solver_failed`. At least 20 stocks are required for the cap.

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

```
Min detectable gap: 2.8 * sd / sqrt(T)
```

H1: `IC_llm_raw - IC_baseline` per rating source (win above the tie band, loss below, else tie).
H2: `IC_combined - IC_baseline` with combined = ranked BL tilt (E3); strategy weekly return
minus its Q0. H3: skills vs bare and skills vs placebo per model, Holm across the family (G3).
H4: splits by market direction, volatility, and earnings week, exploratory (G4). H5: factor
regression on daily returns (G5) and the shuffled-ratings test over every steered portfolio and
each strategy's Q0
with view strength held fixed (G6). The primary paired p-value is the exact sign-flip
permutation; Newey-West is reported beside it. Gaps below the minimum detectable gap are
inconclusive, not evidence of no effect. A null result is acceptable.
