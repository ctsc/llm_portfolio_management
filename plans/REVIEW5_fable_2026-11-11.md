# Review 5: full code base and PLAN4 review (2026-11-11)

Reviewer: Claude (Fable 5.2), for Carter Tierney. Nothing in the repository was changed for this review. This file is the only output. Every proposed change below is a proposal until Carter approves it.

## 0. How to read this file

- Each finding has an ID, a severity, the evidence (file and line, or a run), why it matters, and a proposed fix.
- "Ran" means I executed the repository's own code. "Read" means I traced the code and did not execute that path. "Unverified" means it needs a real service to confirm.
- Section 8 lists the decisions that are Carter's. Section 9 lists the edits I would make to PLAN4, `manualwork.md` and the docs once he decides.

### What I reviewed

| Area | Coverage |
|---|---|
| `pipeline/jobs.py`, `pipeline/status.py`, `pipeline/context.py`, `pipeline/fallback.py`, `scripts/run_weekend.py` | Read in full |
| `optimize/` (Black-Litterman, all five strategies, HRP, shared code), `views/`, `risk/` | Read in full |
| `sim/simulator.py` | Read in full, and ran a stop-rule test |
| `data/snapshot.py`, `data/checks.py`, `data/ledger.py`, `data/universe.py`, `calendar.py`, Alpaca and yfinance providers, provider factory | Read in full |
| `baseline/` (Kronos wrapper, momentum, combine) | Read in full |
| `llm/runner.py`, `llm/providers/litellm.py`, `llm/validate.py`, `llm/prompts.py`, batch payload builders | Read in full |
| `eval/data.py` | Read. The rest of `eval/` is rewritten by PLAN4 Phases 5 to 9, so I reviewed the plan for it instead |
| `config/study.yaml`, `config/models.yaml`, `config/prompts_manifest.yaml`, `prompts/base_instructions.md` | Read in full |
| `plans/PLAN4.md` | Read Phases 1 (step 0), 3, 3A, 4 to 8, 11, and sections 8 to 14 |
| Tests | Ran the unit and integration suites on the committed code (HEAD `44949aa`) in a Python 3.13 container: 337 passed, 1 skipped, none failed (counted from the progress marks). I left out `tests/unit/test_phase12_ci.py` because `yamllint` was not installed, and did not run the e2e suite. The project pins Python 3.11, so Devin's Phase 1 step 0 baseline is the one to trust |

Not reviewed: `backup.py` internals, `llm/cost.py`, `llm/health.py`, `eval/report.py`, the synthetic data provider, and the test files themselves.

## 1. Summary

| ID | Severity | Type | Finding |
|---|---|---|---|
| R1 | High | Research | A steered portfolio differs from Q0 even when the LLM says nothing |
| R2 | High | Research | Mean-CVaR and Resampled MVO barely react to views, so two of five strategy answers are fixed by design |
| R3 | High | Research | The main tests can only detect effects larger than the literature usually reports |
| R4 | Medium | Research | The trailing-IC on/off switch is mostly noise at 8 weeks |
| R5 | Medium | Research | The LLM's news is 2.7 days old when the trade happens |
| C1 | High | Bug | A stop merged into the Monday rebalance fires again that night; the stock is sold Tuesday and the cash sits idle all week |
| C2 | High | Bug | One failed gateway request blocks the weekend and then sends the whole source to Q0 |
| C3 | High | Bug | The prompt's company name and sector come from yfinance each week, not from the frozen stock list |
| C4 | High | Bug | Dry-run and pilot weeks leak into the IC history and into every portfolio's starting state |
| C5 | Medium | Bug | A delisted or acquired stock is never sold; its value is frozen in the portfolio for the rest of the study |
| C6 | Medium | Bug | A 400 error can silently switch off `seed` or JSON mode for the rest of a run |
| C7 | Medium | Bug | The gateway timeout is fixed at 180 seconds for every model |
| C8 | Medium | Bug | Market caps come from one fragile yfinance call each week; a failure freezes every portfolio |
| C9 | Medium | Bug | The API models are not pinned to dated versions |
| C10 | Medium | Plan | Kronos chunking reuses one random seed per chunk, and the plan's equality test cannot hold for the real model |
| C11 | Medium | Unverified | Alpaca may drop the cutoff day's bar when `end` is a bare date |
| C12 | Low | Gap | MiniMax's reasoning text is thrown away |
| C13 to C18 | Low | Mixed | Six smaller items (section 5) |
| P1 to P9 | Mixed | Plan | Nine PLAN4 gaps (section 6) |

Two of my own earlier statements were wrong and are corrected in section 2.

## 2. Corrections to things I said earlier

1. **Gemma's `<|channel>thought` block does not break our parser.** I said it would be marked invalid. It will not: `parse_response` takes the first `{...}` object anywhere in the text (`llm/validate.py`, `_extract_first_object`), so a leading marker is skipped. The strip rule in G15 is harmless but not needed. The only failure case is a thinking block that itself contains a `{`, which cannot happen while thinking is off.
2. **Per-model output limits already exist.** `_settings_for` (`llm/runner.py`) reads `llm.settings.<model>` and falls back to each model's `settings` in `config/models.yaml`. G15 therefore needs two config values and a change to the cost estimate, not new plumbing.

Still correct from the earlier reviews: MiniMax M2.7 fails at 300 output tokens, the Alpaca feed is hard-coded to `iex`, and the 40 GPU-second figure has no source.

## 3. Research validity

### R1. A steered portfolio differs from Q0 even when the LLM says nothing (High)

**What happens.** The LLM view is `q_L = pi + IC_L * sigma * z_L` with its own uncertainty (`views/build.py` lines 160 to 161). When `z_L = 0`, the view is not neutral. It says "the return equals the market prior" with confidence, which pulls the posterior back toward the prior and shrinks the baseline tilt. It also lowers the posterior uncertainty `M`, which Robust, Resampled and Mean-CVaR all read.

**Evidence (ran).** 100 synthetic stocks, the repo's own `build_views`, `black_litterman` and strategies, `IC_b = IC_L = 0.02`, CVaR at 2,000 scenarios. View shift is `0.5 * sum |w - w_Q0|`.

| LLM input | MVO | Robust | Resampled | CVaR | Top-K |
|---|---|---|---|---|---|
| Every stock rated 0, width 2 | 0.100 | 0.097 | 0.019 | 0.014 | 0.100 |
| Every stock rated 0, width 6 | 0.051 | 0.057 | 0.012 | 0.002 | 0.100 |
| Every stock rated 0, half narrow and half wide | 0.111 | 0.116 | 0.034 | 0.002 | 0.100 |
| Pure noise ratings, width 2 (3 seeds) | 0.25 to 0.38 | 0.18 to 0.22 | 0.024 to 0.031 | 0.013 to 0.028 | 0.25 to 0.40 |

So an LLM that rates everything 0 still moves 5% to 11% of the MVO, Robust and Top-K portfolios.

**Why it matters.** "Steered minus Q0" mixes two things: the information in the ratings, and a mechanical change in view confidence. H2 and RQ2 cannot separate them as designed. The placebo does not fully fix this, because different prompts produce different range widths, and width alone moves weights (row 3).

**Options.**
- **(a) Neutral-view construction.** Center the LLM view on the Q0 posterior instead of the prior: `q_L = mu_Q0 + IC_L * sigma * z_L`. Then a rating of 0 leaves the posterior mean exactly unchanged, so MVO and Top-K equal Q0 when the LLM says nothing. Robust, Resampled and CVaR still see lower uncertainty. This changes decision E1/E2 and the proposal's formula.
- **(b) Report the split.** Keep the design. In evaluation, also compute a "confidence-only" tilt per source-week (same uncertainty, `z_L = 0`). That is one linear solve, not an optimizer run, so it fits working rule 9. Report information effect as steered tilt minus confidence-only tilt. This fixes the signal test only, not the portfolio test.
- **(c) Add a confidence-only control portfolio** for the two fast strategies (MVO 0.08 s, Top-K 0.003 s per solve) in J7. It is a locked live portfolio, not a re-run, and adds seconds.
- My recommendation is (a), with (b) as a cross-check, because (a) removes the problem instead of measuring it. It is Carter's decision because it changes the stated method.

### R2. Mean-CVaR and Resampled MVO barely react to views (High)

**Evidence (ran, same setup).** Under pure-noise ratings at full strength, MVO moves 25% to 38% of the portfolio; Mean-CVaR moves 1.3% to 2.8% and Resampled 2.4% to 3.1%.

**Why.** The posterior tilt is about 0.0004 per week (standard deviation across stocks), while the posterior uncertainty per stock is about 0.007. Resampled draws 200 means from that uncertainty, so the draws are about 17 times larger than the tilt and average it away. Mean-CVaR with `lam = 1.0` weighs a weekly CVaR of a few percent against a tilt of a few basis points.

**Why it matters.** RQ4 asks which strategies benefit most. For these two the answer is set by `lam`, `n_draws` and `tau` before any data arrives. G13 plans to label them "insensitive by construction", which is honest, but two things are wrong with G13 as written:
- The 0.01 threshold would not flag Resampled (0.02 to 0.03), even though it is nearly inert.
- View shift includes the R1 confidence effect, so it overstates sensitivity to information.

**Options.** Decide before the freeze, since no data has been seen and changing settings now is not tuning on results:
- Keep the settings and accept that two strategies are reported as insensitive.
- Or rescale so views matter: for example draw Resampled means from a fraction of `M`, and set CVaR's `lam` so the return and risk terms are the same order of magnitude on synthetic data.
- Either way, replace G13's metric with expected tracking error against Q0, `sqrt((w - w_Q0)' Sigma (w - w_Q0))`, measured against the confidence-only counterfactual from R1.

### R3. The main tests can only detect large effects (High)

**Numbers (analytic, not run).** A weekly rank IC across N independent stocks has a standard error near `1/sqrt(N)`: 0.058 at 300 stocks, 0.100 at 100. Real returns are correlated, so the true figures are larger.

| Quantity, 23 weeks, 80% power | 300 stocks | 100 stocks |
|---|---|---|
| Smallest detectable mean IC, one test at 5% | 0.034 | 0.059 |
| Smallest detectable IC gap between two signals | about 0.048 | about 0.083 |
| Same IC, Holm across 8 main comparisons | about 0.046 | about 0.080 |

Published weekly ICs for news-based LLM signals are usually 0.01 to 0.03. A true effect of that size will most likely not reach significance here. Portfolio tests are weaker still: an active return with 1% weekly tracking error needs about 0.6% per week to be detected.

**Why it matters.** PLAN4 already says to lead with effect sizes and intervals. These numbers should also drive three choices:
- 300 stocks is worth fighting for with ARCTIC. At 100 the detectable effect is 1.7 times larger.
- Fewer main comparisons (3 to 5, not 10), and make one of them a pooled test across models and prompts, which is where the power is.
- Say in the proposal and paper that this is an estimation study. A null result will not show the LLM has no effect.

### R4. The trailing-IC switch is mostly noise (Medium)

**Evidence (read plus arithmetic).** `trailing_ic` (`views/ic.py`) averages the last 8 weekly ICs and drops the view when the mean is at or below 0. Eight weeks at 300 stocks gives a standard error near 0.02. A source with no skill is switched on about 50% of weeks, at a strength similar to the cold-start 0.02. A source with true IC 0.02 is still switched off about 16% of weeks. The same rule applies to the baseline: if Kronos plus momentum has no skill, Q0 loses its view about half the time and becomes the market-cap portfolio.

**Why it matters.** The switch adds variance to every comparison and does little to separate skill from luck. It is free of look-ahead (I checked the two-week lag), so this is about noise, not bias.

**Options.** Keep it and report on/off shares (planned in G8); or fix `IC = 0.02` for the whole study, which is simpler and removes a moving part; or lengthen the window to all weeks so far. Carter's decision.

### R5. The news is 2.7 days old at the trade (Medium)

The LLM reads news up to Friday 16:00 ET and the trade is at Monday's open. Friday-evening earnings and weekend news are excluded but are in the Monday open price. The design is free of look-ahead, which is right, but it tests only slow diffusion of week-old news, which lowers the chance of finding any effect. A later cutoff (for example Saturday or Sunday noon, still well before the Monday lock) would give the LLM fresher news at no compute cost, if the weekend run time allows. Carter's decision; it changes the proposal's timing.

## 4. Bugs and reliability

### C1. A merged stop fires again the same night (High)

**Evidence (ran `sim/simulator.py`).** One stock rising 3% a day, target weight 50%, default stops (+15%, -8%):

```
Mon Nov 16  buy A  499.5 sh @100.00  rebalance
Mon Nov 23  sell A  34.3 sh @115.93  rebalance (stop_trigger=take_profit)
Tue Nov 24  sell A 465.2 sh @119.41  take_profit
Tue to Fri: 50.7% of the portfolio in cash
Mon Nov 30  buy A  457.7 sh @119.41  rebalance
```

**Cause.** A stop that triggers on Friday is merged into Monday's rebalance, which keeps the position (lines 222 to 255). The average cost is unchanged, so Monday's close triggers the same stop again (`check_stops`), the whole position is sold Tuesday, and the cash is idle until the next Monday. Stop-loss behaves the same way.

**Why it matters.** Every portfolio, including Q0, carries random cash drag and an extra round trip of costs whenever a stop lands on a Friday. It adds noise to every return comparison.

**Fix options.** When a pending stop is merged into a rebalance, either reset that position's reference price to the rebalance open (the optimizer has re-approved it), or honor the stop and hold no position in that stock for the week. Carter picks the rule; it belongs with decision G1.

### C2. One failed gateway request can cost a whole source its week (High)

**Evidence (read).**
- A request that fails after its retries is dropped with no record (`llm/runner.py`, `summary.n_dropped += 1`).
- J4 still finishes as `done`, so rerunning `run_weekend.py` skips it (`pipeline/status.py`, same input hash).
- J6 then sees fewer records than tasks, calls the source `incomplete`, and raises "runs still incomplete before the lock margin" on every rerun until Monday 08:30.
- At the margin the source is marked failed and all five of its portfolios trade Q0.

So one stock out of 300 that keeps failing (a 500 error, a context-length error) sends the entire source to fallback, while 44 invalid answers out of 300 would have been tolerated by the 15% rule. The printed hint ("rerun J6") does not help; only `--force` on J4 does.

**Fix.** After the retries, write the request as an invalid record with reason `provider_error`, so it counts toward the 15% rule. Make J4 end as failed or pending, not done, while any request is unrecorded, so a rerun resumes it. PLAN4 Phase 3A's outage tolerance should include both.

### C3. Company name and sector in the prompt come from yfinance (High)

**Evidence (read).** `data/providers/factory.py` builds `YFinanceReferenceProvider(earnings=earnings)` without `members`, so `name` and `sector` come from `yf.Ticker(t).info` every week. `llm/runner.py` puts those into the prompt. yfinance sector labels are not GICS ("Technology", "Consumer Cyclical", "Financial Services"), can change, and can be missing, which would print "Sector: None". The base prompt tells the model it is given "the GICS sector".

**Why it matters.** The prompt text for a stock can change between weeks for reasons unrelated to the study. The residual-IC sector control would also use these labels.

**Fix.** Take name and sector from the frozen `config/universe.csv` for prompts and controls. Use yfinance only for market cap.

### C4. Dry-run and pilot weeks leak into the live study (High)

**Evidence (read).**
- `trailing_ic` has no lower bound at `live_start_week`. From live week 6 the 8-week window reaches back before the live start, so pilot and dry-run weeks set real view strengths.
- `replay_weeks` starts at the first week that has locked weights, so any dry-run week becomes part of every portfolio's holdings, costs and stops.
- `manualwork.md` E5 runs a dry run on 3 to 4 past real weeks, with nothing saying where its data goes.

**Fix.** Run dry runs with `PORTSTUDY_DATA_DIR` pointed at a scratch folder (the code already supports it), and add a preflight check that the live data folder holds no week before the pilot. Decide and write down whether the pilot week counts toward trailing IC and starting positions; today it silently does.

### C5. A delisted stock is never sold (Medium)

**Evidence (read).** A held stock with no price is carried at its last close and queued as `forced_exit`, which executes "at the first open where a price exists" (`execute_pending`). After a cash acquisition or delisting no price ever returns, so the position stays at its last close forever and its value is locked out of every later rebalance. The planned U1 removal has the same gap when the removed stock has stopped trading. Over 24 weeks and 300 S&P 500 names, one to three such events are likely.

**Fix.** After N trading days without a price (for example 5), liquidate at the last close, log it, and count it in the report.

### C6. A 400 error can silently switch off seed or JSON mode (Medium)

**Evidence (read, `llm/providers/litellm.py`).** Any HTTP 400 whose body contains the text `seed` or `response_format` is treated as "parameter unsupported", and that feature is turned off for the rest of the run. LiteLLM error bodies often echo the request, so an unrelated 400 (for example context length) could match. The state is kept per week, so one week could run with JSON mode and the next without.

**Fix.** After the pilot, freeze the accepted features in config and treat a rejection as a failure, not a silent change. Match the error more narrowly. Record the features used per source-week in the study log.

### C7. The gateway timeout is 180 seconds for every model (Medium)

`LiteLLMProvider(timeout=180)`. MiniMax writing about 2,000 reasoning tokens under load can exceed that. A timeout is retried 3 times inside the provider and 2 more rounds by the runner, so one slow request can burn many minutes and then be dropped (which triggers C2). Make the timeout a per-model setting, with MiniMax higher, and set it from the timing run.

### C8. Market caps depend on one fragile weekly call (Medium)

`_yf_info` reads `marketCap` from yfinance's `.info`, an unofficial endpoint that often rate-limits or breaks. A missing cap silently becomes weight 0 in the market prior. If all fail, `market_weights` raises, J2 fails, and every portfolio holds for the week. Cache the last good reference row per ticker and reuse it on failure, and flag when more than a few caps are missing.

### C9. The API models are not pinned (Medium)

`config/models.yaml` has `snapshot_id: null` for Claude Haiku 5.5 and GPT-6 Luna, and the provider IDs are aliases. An alias can move to a new version during a 24-week study. Pin dated IDs and have `check_live_ready.py` require them.

### C10. Kronos chunking and seeds (Medium, plan)

PLAN4 K2 says chunked results "must equal the unbatched result for the same seed". That holds for the test's fake predictor but not for the real model: sampling consumes the random stream per batch, so chunking changes the draws. The code also seeds once per sample (`torch.manual_seed(seed + sample_index)`), so with chunks every chunk would reuse the same stream, and stock j of each chunk would get the same noise. Fix: seed per (sample, chunk), drop the equality claim, and freeze `batch_size` at the freeze because it changes the baseline.

### C11. Alpaca may drop the cutoff day's bar (Unverified)

`_bars` sends `end` as a bare date. If Alpaca reads that as midnight UTC, the cutoff day's daily bar (stamped 04:00 or 05:00 UTC) falls outside the range, every stock would trip `stopped_trading`, and J1 would skip them all. I could not confirm this from Alpaca's docs. The smoke test should assert that the last bar's date equals the cutoff day.

### C12. MiniMax's reasoning text is discarded (Low)

The record keeps `raw` (the answer) but not `reasoning_text`. Saving it costs disk only and is useful for debugging invalid answers and for any later analysis of how the skills prompts change reasoning. At minimum save its length.

### Smaller items

| ID | Item | Evidence | Fix |
|---|---|---|---|
| C13 | A run started on a Thursday evening before a Friday holiday (Dec 24, 2026) computes the previous week | `current_week` only rolls forward on the calendar Friday | Roll forward at the cutoff day's close |
| C14 | After switching the feed to SIP, an existing ledger keeps its IEX rows (append-only, mismatches only logged) | `data/ledger.py` | Rebuild the ledger from empty at the freeze; add to `manualwork.md` |
| C15 | MVO accepts `optimal_inaccurate` silently; HRP reports "single_linkage" even if config changes | `mean_variance.py`, `hrp.py` | Record the solver status; read the label from config |
| C16 | The retry of an invalid answer resends the same prompt at temperature 0 | `runner.py` | Fine to keep; expect little gain, and it doubles MiniMax cost on truncation |
| C17 | Kronos forecasts Friday close to Friday close; the trade window is Monday open to Monday open | `baseline/kronos.py` | Note it in the paper's limitations |
| C18 | "Same length" placebo and skills files are matched by characters divided by 4, not by each model's tokenizer | `llm/prompts.py` | Say so in the paper, or measure with the real tokenizers once |

## 5. What I checked and found correct

- **No look-ahead in view scaling.** A week's IC is used only after its return window has closed (`window_end_date <= cutoff`), which gives the two-week lag.
- **No stuck-at-zero state.** `z_L` is still saved when a source's view is switched off, so its IC keeps updating and it can switch back on.
- **Black-Litterman algebra** matches the documented formulas, including the stacked identity views.
- **Weekly stock returns** handle splits and dividends inside the window correctly.
- **The holiday table** for 2026 and 2027 matches the NYSE calendar, including the Friday closures on Dec 25, 2026 and Jan 1, 2027.
- **Weight capping**, atomic file writes, the response cache and the per-call records behave as described.

## 6. PLAN4 gaps

| ID | Gap | Proposed change |
|---|---|---|
| P1 | G13 uses view shift and a 0.01 threshold (see R2) | Use expected tracking error against the confidence-only counterfactual; set the threshold from synthetic and pilot numbers |
| P2 | G8 counts weeks with no active view as "zero gaps" | Use the realized return gap. Holdings and average costs differ from Q0's, so stops and costs make the gap non-zero |
| P3 | G14 says the tilt gap "is zero when the LLM adds nothing" | Not exact (R1). Reword, and add the confidence-only tilt to Phase 7 |
| P4 | Phase 3A outage tolerance does not mention dropped requests or J4's state | Add the C2 fix and a test that kills the network for one request |
| P5 | No end-of-study procedure | After the last live week, run J8 once the final Monday open exists; state how the last week's return is closed |
| P6 | G15 adds a Gemma strip rule and "per-model limits" as new code | Reduce to config values, the cost estimate, a per-model timeout (C7), and saving reasoning text (C12) |
| P7 | `check_live_ready.py` list is missing checks | Add: API models pinned; name and sector from the universe file; no pre-pilot weeks in the data folder; gateway features frozen; feed is the configured one |
| P8 | Preregistration has no public timestamp | Push the frozen file under a tag to the public repo (or deposit it) so the date is checkable |
| P9 | The Kronos equality criterion (C10) | Replace with: deterministic per (seed, chunk), and results identical across reruns on CPU |

Also confirm both git remotes accept pushes before Devin starts, since every phase ends with a push.

## 7. Suggested order of work

1. Carter decides the section 8 items. R1, R2 and C1 change what the study measures, so they come before the preregistration and the freeze.
2. Fold the approved fixes into PLAN4: C1 and C5 into Phase 3A (simulator); C2, C6, C7, C12 into Phase 3 and 3A (gateway); C3, C8, C11, C14 into Phase 3 (data); C4, C9 into the preflight; R1 and R2 into a new small phase before Phase 4, because they change `views/build.py` and strategy settings.
3. Nothing here needs Andalib's answers.

## 8. Decisions for Carter

| # | Decision | My recommendation |
|---|---|---|
| 1 | R1: neutral-view construction (a), report the split (b), or a control portfolio (c) | (a) plus (b) |
| 2 | R2: keep CVaR and Resampled as is and report them as insensitive, or rescale before the freeze | Rescale on synthetic data before the freeze, and replace the G13 metric |
| 3 | R3: number of main comparisons, and whether one is a pooled test | 3 to 5, with one pooled |
| 4 | R4: keep the trailing-IC switch, fix IC at 0.02, or use all weeks so far | Fixed 0.02 is simplest; keep the switch only if it is part of the research claim |
| 5 | R5: keep the Friday 16:00 news cutoff or move it later in the weekend | Worth considering; needs the measured J4 time first |
| 6 | C1: when a stop lands on a rebalance day, reset the reference price or sit out the week | Reset the reference price |
| 7 | C4: does the pilot week count toward trailing IC and starting positions | No; start clean at the first live week |
| 8 | C5: how many no-price days before a forced liquidation | 5 trading days |

## 9. Files I would edit after approval

- `plans/PLAN4.md`: section 0.3 decisions (new IDs for R1, R2, C1, C4, C5), Phases 3, 3A, 4, 7, 8, completion criteria, caveats, confidence.
- `manualwork.md`: scratch data folder for dry runs, ledger rebuild at the freeze, pin API model versions, preregistration tag, end-of-study step.
- `llm_research/02_decisions_log.md`, `03_pipeline_and_methods.md`, `05_evaluation_methodology.md`, `09_arctic_process_and_model_facts.md` (the Gemma correction), and the proposal's method paragraph if R1 (a) or R5 is chosen.
- `docs/design.md`, `docs/config_reference.md`, `docs/open_questions.md`, `docs/runbook.md`.

## Appendix A. Experiment for R1 and R2

Script: 100 stocks; one-factor daily returns over 252 days; the repo's `ledoit_wolf_covariance`, `build_views`, `black_litterman`, `black_litterman_q0` and `run_strategy`; `tau = 0.05`, `delta = 2.5`, cap 5%, Top-K 20, `IC_b = IC_L = 0.02`; CVaR at 2,000 scenarios instead of 10,000 for speed.

```
pi sd 0.00033  Q0 tilt sd 0.00041  sigma mean 0.0438
Q0 names at the 5% cap: mvo 12, robust 0, resampled 0, cvar 11, topk 20
Q0 names held:          mvo 28, robust 82, resampled 100, cvar 28, topk 20
LLM says nothing (z_L=0), width 2    tilt corr with Q0 0.996  shift mvo 0.0995 robust 0.0966 resampled 0.0190 cvar 0.0135 topk 0.10
LLM says nothing (z_L=0), width 6    tilt corr with Q0 0.999  shift mvo 0.0511 robust 0.0573 resampled 0.0122 cvar 0.0016 topk 0.10
z_L=0, half narrow / half wide       tilt corr with Q0 0.991  shift mvo 0.1105 robust 0.1155 resampled 0.0335 cvar 0.0022 topk 0.10
pure noise z_L seed 0, width 2       tilt corr with Q0 0.842  shift mvo 0.3761 robust 0.1972 resampled 0.0311 cvar 0.0126 topk 0.25
pure noise z_L seed 1, width 2       tilt corr with Q0 0.844  shift mvo 0.2516 robust 0.1786 resampled 0.0240 cvar 0.0216 topk 0.35
pure noise z_L seed 2, width 2       tilt corr with Q0 0.815  shift mvo 0.3151 robust 0.2152 resampled 0.0302 cvar 0.0276 topk 0.40
```

Limits: one synthetic covariance and one draw of baseline scores, 100 stocks, not 300. The direction is structural; the exact sizes will differ on real data. PLAN4 Phase 11 should repeat this on the synthetic world at 300 stocks.

## Appendix B. Experiment for C1

Two stocks (A rises 3% a day for six days then is flat; B is flat), four weeks from 2026-W46, target 50% each, default simulator settings. Output is quoted in C1. Stock A's cash share Tuesday Nov 24 to Friday Nov 27 was 50.7%.
