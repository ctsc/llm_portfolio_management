# 03 Weekly pipeline and methods

Last updated: 2026-11-17. Describes the study as it runs now: LLM ratings on ARCTIC's gateway,
everything else on Carter's desktop.

## Weekly cycle

1. Friday 16:00 ET: freeze that week's public data.
2. Over the weekend: baseline forecasts, LLM ratings, views, portfolios.
3. Monday 09:00 ET: weights locked (30 minutes before the open). Trades execute at Monday's open.
4. Returns are measured Monday open to next Monday open, total return including dividends and
   splits. Stops are checked daily.

Carter runs one script by hand, ideally Friday night (Saturday morning as backup):

```bash
python scripts/run_weekend.py            # resumes where it stopped if rerun
python scripts/run_weekend.py --synthetic   # full rehearsal on synthetic data
```

Step 1: J8 and J9 for any earlier week not yet replayed or backed up. Step 2: J1 to J6 in order,
waiting on API batches with a countdown. Step 3: J7, then J9. On a failure the script stops,
prints what failed, the fallback that will apply, and the exact command to rerun that job.
Flags: `--week`, `--from J3`, `--until J5`, `--dry-run`, `--allow-over-budget`, `--now`.

## Jobs

| Job | What it does | Where |
|---|---|---|
| J1 | Snapshot: prices, news, earnings dates, reference data, T-bill rate | Desktop (internet) |
| J2 | Quant baseline: momentum ranks plus Kronos 5-day forecasts (20 sampled paths per stock) | Desktop GPU |
| J3 | Submit API batch requests (Claude Haiku 5.5, GPT-6 Luna) | Provider servers |
| J4 | Rate every stock with the gateway models, up to 4 requests at a time per model; a health check sends one tiny call per model first | ARCTIC gateway |
| J5 | Collect API batch results | Provider servers |
| J6 | Validate outputs, apply the failure rule, plan fallbacks | Desktop |
| J7 | Residualize, build views, Black-Litterman, run all strategies and controls, lock weights | Desktop CPU |
| J8 | Replay the simulator on realized prices, score IC | Desktop |
| J9 | Back up the week's CSVs | Desktop, plus an off-machine copy |

## Data sources

| Data | Provider |
|---|---|
| Daily prices and news | Alpaca (yfinance as a fallback for prices). Review finding 2026-11-11: the code requests the IEX feed (`feed: iex`), which is one small exchange and not consolidated prices; PLAN4 switches it to SIP, with a smoke check, and falls back to yfinance if the free plan refuses |
| Reference data (names, sectors, market caps) | yfinance |
| Earnings dates | Finnhub |
| Risk-free rate | FRED (T-bill) |
| Factors for the final analysis | Kenneth French Data Library (daily FF5 and momentum) |

Data checks: forward-fill missing days; skip a stock for non-positive prices, duplicate rows, a
stale price for 5 days, short history (under 300 days), a halt, or a raw-price jump the adjusted
prices do not show (likely a missed split, that week only). Large moves (decision U1, built in PLAN4
Phase 3A): a 5-day move of 10% or more either way is kept and flagged; a fall of 40% or more removes
the stock from the study permanently (no replacement). Weekly portfolio returns run Monday open to
next Monday open, the same window as rank IC; a stock with no
news stays in with the no-news sentence; news after the cutoff is dropped.

## Quant baseline (no LLM)

Each stock is ranked on price momentum (12-1 month, 3-month and reversed 1-month returns) and on
Kronos's next-week return forecast; the two ranks are averaged into the baseline score z_b.
Kronos samples 20 five-day paths per stock. The spread of those paths (k_sd) sets how much the
baseline view is trusted (K1 mapping, `02_decisions_log.md`).

## LLM rating

One request per stock per prompt type. The model sees ticker, company name, sector and that
week's news, and returns JSON: low, likely and high integers on -5 to 5 and a short rationale
(150-token cap). Prompt types: bare, 4 skills files of similar length (decided 2026-11-12), and a placebo
(filler text of the same length with no investing content). Temperature 0, seed sent when the
endpoint accepts it, JSON mode when accepted, max 300 output tokens (Gemma). MiniMax M2.7 always
thinks and the thinking counts toward the cap, so it needs its own cap of about 2,000; PLAN4 adds a
per-model output cap and timeout, saves the reasoning text, and records tokens and
`finish_reason` per request. Gemma's `<|channel>thought` marker needs no handling: the parser takes
the first JSON object in the text.

## Views and Black-Litterman

1. Residualize: remove the part of the LLM's likely score that the baseline already explains,
   so the LLM view only adds new information. Invalid outputs are excluded.
2. Build views: the residualized likely score, the source's trailing rank IC and stock volatility
   set each view's expected return. The LLM view is centered on the Q0 forecast (decision E7,
   2026-11-11): `q_L = mu_Q0 + IC_L * sigma * z_L`, so a rating of 0 changes nothing for MVO and
   Top-K (built in PLAN4 Phase 3B). Range width sets view uncertainty: c = clip(1 - width/10, 0.1,
   1), so a narrow range (3 to 5) counts more than a wide one (1 to 5).
3. View scale: fixed at 0.02 for the LLM and baseline views every week (decision A8, 2026-11-12;
   built in PLAN4 Phase 3A; the older trailing-IC gating remains only as `ic_mode: trailing` for
   tests).
4. Black-Litterman blends market-implied returns (study-only market-cap weights, delta 2.5) with
   the baseline view and the LLM view, using a Ledoit-Wolf covariance (252 days, tau 0.05). The
   same posterior and covariance go to every strategy. Any LLM view also lowers the posterior
   uncertainty, which Robust, Resampled and Mean-CVaR read, so those three still differ from Q0
   when every rating is 0 (measured by decision G18).

## Strategies

| Strategy | What it does |
|---|---|
| Mean-variance (MVO) | Maximizes expected return minus a risk penalty |
| Mean-CVaR | Maximizes return against the average loss in the worst 5% of outcomes, using 10,000 scenarios drawn from the posterior |
| Robust MVO | Optimizes against the worst case within each view's uncertainty (kappa 1) |
| Resampled MVO | Re-optimizes over 200 random draws of expected returns and averages the weights |
| Top-K | Holds the 20 stocks with the highest expected returns in equal weights |
| Q0 (one per strategy) | The same strategy with the baseline view only, no LLM view |
| HRP | Hierarchical risk parity, no forecasts |
| Equal weight | 1/N, no forecasts |

Solvers: OSQP and Clarabel for MVO, Clarabel and HiGHS for CVaR, Clarabel for robust. CVaR and
resampled draws use common random numbers seeded by (week, strategy), so every source sees the
same scenarios. If a solver fails, the portfolio keeps the previous weights and is flagged.

## Guardrails (same for every portfolio)

Weekly rebalance, daily stop-loss -8% and take-profit +15%, 10 bps per trade, no shorts, at
most 5% per stock, at least 20 stocks for optimizers. Initial value 100,000.

## Portfolio count

Sources = models x prompt types. With 2 gateway models plus 2 API models and 6 prompt types,
that is 24 rating sources. Each source steers 5 strategies (120 steered portfolios), plus 5 Q0s,
48 confidence-only portfolios (Robust and Resampled MVO, G19) and 2 controls, about 175 in all. The compute estimates also use the older upper bound of 51 sources (7 models x 7
prompt types + Q0) as a worst case.

## Failure handling

- Invalid rate above 15%, or a source not finished by the lock margin: that source falls back to
  its strategy's Q0 for the week (logged).
- Gateway health check fails: that model's sources are marked failed and the run continues.
- Lock deadline reached mid-J7: finished portfolios keep their weights, the rest get Q0 with
  reason `lock_deadline`.
- Kronos failure: by current design every portfolio holds last week's positions (a cheaper
  momentum-only fallback is an open design choice).
