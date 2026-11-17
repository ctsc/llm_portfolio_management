# 05 Evaluation methodology (current plan)

Last updated: 2026-11-17. This is what the study will test and report, matching the Evaluation
section of `current_documents/Project_Proposal_IEEE.pdf`. The reasoning and sources behind each
choice are in `06_evaluation_research_findings.md`. The code does not implement all of this yet;
see "Code status" at the bottom.

Rule: no evaluation method may add CPU or GPU hours. Every test uses the saved ratings, weights
and returns, reruns no model or optimizer, and finishes in seconds to minutes.

## Two core measurements

**Rank IC (signal level).** Each week, the Spearman correlation across stocks between a score and
the next week's returns (Monday open to next Monday open, total return). It measures whether a
signal ranks winners above losers. Because it is computed across stocks within one week, a
market-wide rise or fall does not change it. Rank IC is the same for every strategy. A "good" IC
in practice is around 0.05; 0.10 is very good.

**Active return (portfolio level).** Each steered portfolio's weekly return minus the return of
its own strategy's Q0. The strategy, constraints, costs, stops and universe are identical, so
the only difference is the LLM views.

## Per research question (code IDs H1 to H5)

All IC gaps are computed on the same stocks each week (decision G14). A strategy whose median weekly
expected tracking error against Q0 is below the pre-registered threshold is reported as insensitive
to views by construction (decision G13, changed 2026-11-11; for Robust and Resampled MVO the confidence-only
control portfolios of G19 split steered minus Q0 into information and confidence: tracking error replaces view shift, and
the threshold has no default). With the neutral view (E7) the H2 tilt gap is exactly zero when every
rating is 0; the confidence-only counterfactual (G18) shows how much uncertainty each view removes. Weekly portfolio returns run Monday open to next Monday open (N4).

### H1: Does the raw LLM signal beat the quant baseline?

- Primary: paired weekly gap, IC of the LLM likely score minus IC of the baseline score, with a
  Newey-West t-test (Student-t critical values), block-bootstrap 95% CI and sign-flip p-value.
- Residual IC: a weekly Fama-MacBeth regression of next-week return ranks on the LLM score,
  controlling for the baseline score z_b, momentum, size, value, profitability, investment,
  volatility (or beta) and sector. The average LLM coefficient tests whether the LLM adds
  information the baseline and known styles do not already carry. The control list must be fixed
  before the live run.
- Also: ICIR (mean IC / sd of IC), IC hit rate (share of weeks with IC > 0), top-minus-bottom
  quintile spread, and optionally IC against week t+2 returns (decay).

### H2: Does adding LLM views improve on the optimizer alone?

- Signal level: IC of the steered Black-Litterman tilt minus IC of Q0's tilt (decision G14), on the
  same stocks, plus the steered tilt's residual IC. Comparing with the raw baseline score is wrong:
  Q0's own tilt reorders stocks, so that gap is not zero when the LLM adds nothing.
- Portfolio level, for each steered portfolio vs its own Q0:
  - mean active return with Newey-West t-test, bootstrap CI and sign-flip p-value;
  - information ratio and tracking error vs Q0 (tracking error near zero means the LLM barely
    changed anything);
  - Sharpe ratio gap tested with the Ledoit-Wolf (2008) bootstrap test;
  - extra turnover and the break-even trading cost at which the gain disappears;
  - daily Fama-French 5 plus momentum regression of the active return (see H5).
- Reported two ways: over all weeks (using the realized gap every week; inactive weeks are not set
  to zero, because stops and costs differ from Q0's), and
  over only the weeks where the LLM view was active. Also report the share of active weeks per
  source.

### H3: Do specialized skills help, beyond length or placebo?

- Signal level: IC of each skills prompt minus bare, and minus placebo, per model.
- Portfolio level: paired weekly return of the skills portfolio minus the placebo portfolio (and
  minus bare), per model and strategy.
- Primary H3 test: one pooled regression across models, with model and week fixed effects and
  standard errors clustered by week, giving the average skills-minus-placebo effect. Per-model
  results are secondary.
- Between models: paired weekly IC difference between models on the same prompt (bare prompt
  pre-specified), plus the same pooled regression with model and prompt effects.

### H3b: Does the LLM's stated uncertainty mean anything?

- IC of narrow-range stocks minus wide-range stocks (split at each source-week's median width).
- Fama-MacBeth interaction of score and confidence c: a positive interaction means confident
  views predict better.
- Calibration check: Spearman correlation between range width and absolute rank error (positive
  means wide ranges flag worse forecasts).

### H4 (exploratory): Which strategy uses LLM views best, and when?

- Rank strategies by mean active return vs their own Q0.
- Strategy-vs-strategy difference of active returns (a paired series).
- Splits fixed in advance: market up or down, high or low volatility (weeks), earnings week or not
  (stocks). Report cell means with confidence intervals; cells have about 5 to 15 weeks.
- Mean view shift per strategy shows how much of each view an optimizer actually expresses.
- Labeled exploratory; no confirmatory claims.

### H5: Is the effect real, or market conditions and luck?

- Regress each portfolio's daily active return (steered minus Q0) on the Fama-French five factors
  plus momentum from Kenneth French's daily data, with Newey-West errors (about 4 lags on about
  115 days). The factor loadings show which market or style exposure the LLM added; alpha is what
  market movement and known styles cannot explain. No risk-free subtraction is needed on the left
  side because it is a difference of two long portfolios.
- Rank IC and residual IC are already immune to market direction.
- The placebo prompt is the content control: it controls for prompt length and any change to the
  views, so a gain the placebo also shows does not come from the LLM's content. Chance is handled by
  the tests and intervals; market and style luck by Q0 and the factor model.
- If a single "best" portfolio is headlined: deflated Sharpe ratio (closed form) and a flag for
  whether its t-statistic clears 3.0 (Harvey, Liu and Zhu).
- ETF proxies (SPY, IWM, IWD, IWF, QUAL) are used only for live monitoring until French data
  covers the study window (it is released with a lag).

### Controls (HRP and equal weight)

Paired active returns and the Ledoit-Wolf Sharpe test of each steered portfolio and each Q0
against equal weight and HRP. With about 23 weeks these comparisons have little power, so they
are reported as estimates with confidence intervals, not pass/fail tests. Equal weight is the
standard honest hurdle (DeMiguel, Garlappi and Uppal 2009).

## Statistics used on every weekly gap

| Statistic | Detail |
|---|---|
| Mean gap | Newey-West standard error (2 lags for 23 weekly points; report next to the plain SE and use the larger), Student-t critical values with T - 1 degrees of freedom |
| Bootstrap CI | 2,000 resamples, stationary or circular block bootstrap, mean block length about 3 (T^(1/3)), percentile-t intervals |
| Sign-flip permutation | Exact enumeration is feasible (2^23 patterns, seconds) or Monte Carlo; robust p-value. Do not call a Monte Carlo version "exact" |
| Win / tie / loss | Narrower tie band than the current 1% (for example 10 bps or half a weekly tracking error), with a sensitivity check |
| Minimum detectable effect | 2.8 x sd / sqrt(T) (80% power, two-sided 5%); about 0.58 sd at T = 23. Report it for every main test so a null reads as "effects above X are ruled out" |

## Multiple testing

| Family | Procedure |
|---|---|
| Main comparisons fixed before the live run: exactly four pooled tests (decision G7, revised 2026-11-17): LLM-minus-baseline rank IC gap on the same stocks, steered-minus-Q0 tilt IC, skills minus bare, skills minus placebo, each pooled over models with model fixed effects | Holm (Romano-Wolf stepdown is a more powerful option when the tests are strongly correlated) |
| Everything else (per-source, per-model, per-prompt, pooled LLM rank IC above zero, Sharpe tests) | Benjamini-Hochberg false discovery rate, reported as secondary |
| H4 | No correction; labeled exploratory |
| Metric values with no test attached (decision G22: every metric for every source, portfolio, model, prompt and strategy, in `all_metrics.csv`) | Reported as estimates with intervals; null results kept |
| "Does any source beat Q0" and "which models are plausibly best" (optional) | Hansen SPA test; Model Confidence Set. Both resample saved series in seconds |

## Portfolio metrics reported for every portfolio (including Q0, HRP, equal weight)

Total return; Sharpe ratio net of the T-bill rate (weekly, annualized by sqrt(52)) with a
probabilistic Sharpe ratio; Sortino ratio, maximum drawdown and realized CVaR (95%) from daily
returns; Calmar (descriptive); information ratio and tracking error vs Q0; certainty-equivalent
return; turnover; total cost; break-even cost; stop-loss and take-profit counts; mean view shift;
fallback and solver-failed weeks. Check lag-1 autocorrelation and apply Lo's (2002) correction to
the annualized Sharpe only if it is material.

## Code status (what still has to change)

What the code does now (`src/portstudy/eval/`): weekly rank IC for baseline, raw LLM, residualized
LLM and BL tilt; H1/H2 IC gaps with Holm across sources; H2 portfolio gaps with Holm per strategy;
H3 IC-only per model; H3b narrow minus wide; H4 rankings and splits; H5 ETF-proxy factor regressions (the shuffle
test and its config were removed in PLAN4 Phase 1, 2026-11-17); win/tie/loss (1% band), Newey-West, bootstrap, sign test, sign-flip
(10,000 draws, mislabeled "exact"), minimum detectable gap; Sharpe without risk-free subtraction.

To build (a Devin phase): residual IC via Fama-MacBeth; portfolio-level H3 and the pooled H3
regression; model-vs-model tests; Ledoit-Wolf Sharpe test; information ratio, tracking error,
break-even cost, Sortino, CVaR and drawdown from daily returns, probabilistic Sharpe, deflated
Sharpe; risk-free subtraction; active-weeks reporting; Holm on a pre-registered family plus
Benjamini-Hochberg; Student-t critical values; block bootstrap; tests vs HRP and equal weight;
French daily FF5 plus momentum loader; and a narrower tie band. See `10_open_items_and_next_steps.md`.
