# 06 Evaluation research findings

Last updated: 2026-11-11. Findings from the evaluation-methodology research done on 2026-11-11
(question: what are the industry-standard, low-compute ways to evaluate each hypothesis), plus
explanations given in chat. The resulting plan is `05_evaluation_methodology.md`. Sources are in
`08_literature_and_references.md`, part B.

## Bottom line

- The shuffle test can be dropped with nothing lost. The standard quant toolkit answers "skill or
  luck" at near-zero compute: rank IC and residual IC on the stock-week panel, paired gap tests of
  each steered portfolio against its own Q0, and factor regressions of the active return on
  Kenneth French daily factors.
- The real weaknesses of the old plan were not compute. They were power (about 23 weeks), too
  many tests in one family, a few metric conventions (no risk-free rate, a wide tie band), and
  missing portfolio-level tests for H3 and for the HRP and equal-weight controls.
- Q0 should be the main benchmark: it is the best-powered and most intuitive comparison.

## Metrics primer (plain language)

| Metric | Formula | What it means | Small-sample note (T about 23 weeks) |
|---|---|---|---|
| Rank IC | Spearman correlation of this week's scores with next week's returns across stocks | Does the signal put winners above losers? 0 means no skill. Per MSCI Barra (citing Grinold), 0.05 is "good" and 0.10 "very good" | 23 weekly values; their mean and t-stat are the core skill measure |
| ICIR | mean(IC) / sd(IC) | Consistency of skill | t = ICIR x sqrt(T) |
| Hit rate | Share of weeks with IC > 0, or weeks the steered portfolio beats Q0 | "How often it worked" | Binomial test: at least 17 of 23 wins for p < 0.05 (two-sided) |
| Quantile spread | Top-quintile mean return minus bottom-quintile | Money version of IC | Pure sorting, market-neutral, no optimizer |
| Sharpe ratio | sqrt(52) x mean(r - rf) / sd(r) | Return per unit of total risk | Very noisy (below) |
| Information ratio | sqrt(52) x mean(r_p - r_b) / sd(r_p - r_b) | Sharpe of the active return vs a benchmark (Q0 here) | Natural headline for H2 |
| Tracking error | sqrt(52) x sd(r_p - r_b) | How far the LLM moves the portfolio from Q0 | Near zero means the LLM barely did anything |
| Sortino | Like Sharpe, but divides by downside deviation only | Penalizes only downside volatility | Use daily data (about 115 days) |
| Max drawdown | Largest peak-to-trough fall | Worst-case pain | One number per path; descriptive only |
| Calmar | Annualized return / max drawdown | Return per unit of worst pain | Unstable under a year; descriptive |
| CVaR / expected shortfall (95%) | Mean of the worst 5% of returns | Average loss in bad periods | 5% of 23 weeks is about 1 point, so use daily returns (worst 5 to 6 days) |
| Omega | Sum of gains above a threshold / sum of losses below it | Gains-to-losses ratio | Optional; adds little beyond Sortino |
| Certainty-equivalent return | mu - (gamma / 2) sigma^2 | Risk-free return an investor with risk aversion gamma would accept instead | Standard in DeMiguel, Garlappi and Uppal (2009) |
| Turnover | Sum of absolute weight changes per rebalance | How much trading the LLM causes | Report the gap vs Q0 |
| Break-even cost | Gross active return / extra turnover | Cost per unit of turnover at which the LLM edge disappears | Robust to the 10 bps assumption |

Two framing points:
- Rank IC already removes market conditions, because it compares stocks within the same week.
- The information ratio vs Q0 is the portfolio-level counterpart of IC. By Grinold-Kahn's
  fundamental law, IR is roughly IC x sqrt(breadth). Ding and Martin's "Redux" shows that when IC
  varies over time, the attainable IR is roughly IC / sqrt((1 - IC^2 - sigma_IC^2)/N + sigma_IC^2).
  With IC = 0.02, N = 300 and an assumed sigma_IC = 0.10, that is about 0.17 per week (about 1.25
  annualized), roughly half the naive 0.02 x sqrt(300 x 52), about 2.5. IC volatility over time,
  not breadth, limits what 23 weeks can show.

## What the stock-week panel buys you

About 23 x 300 = 6,900 stock-weeks per source, but they are not independent: returns in the same
week share market, sector and factor shocks. Petersen (2009) shows ordinary standard errors are
biased when residuals are correlated within time; Fama-MacBeth (or clustering by week) is right.

- The inference unit is still weeks. A Fama-MacBeth t-stat is mean(lambda) / (sd(lambda)/sqrt(T)),
  the same shape as an IC t-test. The panel does not create degrees of freedom.
- What the panel adds: (1) control, estimating the LLM's effect holding z_b, momentum and styles
  fixed; (2) pooling across sources with model and prompt fixed effects (one test instead of 7,
  which keeps power); (3) more precise weekly estimates.
- Cluster by week (or Fama-MacBeth) as primary; two-way clustering by week and stock as a
  robustness check. With 23 clusters use t(22) critical values.

## Replacing the shuffle test

Why it is redundant: under random views, the expected active return vs Q0 is about zero before
costs and negative after costs (noise tilts add turnover). So "beat Q0" is at least as strict as
"beat the middle of the shuffle distribution", and the paired gap tests already test that. The
only extra thing a shuffle gives is a simulated noise-tilt variance; the placebo prompt already
supplies an empirical version of that at zero cost.

| Replacement | What it answers | Compute |
|---|---|---|
| Weekly rank IC with Newey-West t-test | Cross-sectional skill, immune to market direction | Milliseconds |
| Analytic null for Spearman IC: mean 0, variance 1/(N - 1) per week | Exact analytic version of "shuffle ratings across stocks" at the signal level | Zero |
| IC-level permutation (optional) | Same, without asymptotics: permute 300 ranks, not a portfolio | Seconds for 10,000 permutations |
| Fama-MacBeth residual IC | Does the LLM add information beyond Kronos, momentum and known styles? | Milliseconds |
| Quantile long-short spread on the residual LLM score, regressed on factors | Is the residual signal a tradable, factor-neutral return stream? | Milliseconds |
| Daily FF5 plus momentum regression of the active return | Is the portfolio-level LLM effect alpha or a style or market tilt? | Milliseconds |
| Placebo-controlled gap (skills minus placebo) | Is the gain from task content or from any perturbation of views? | Zero |

For N = 300 the analytic null sd per week is 1/sqrt(299), about 0.058, or about 0.012 averaged
over 23 independent weeks. It ignores factor-driven correlation between stocks, so the time-series
t-test on the 23 weekly ICs (more conservative) is primary and the analytic null is a sanity check.

### The residual IC regression

Each week t, on cross-sectional ranks or z-scores:

rank(r_i,t+1) = a_t + lambda_t x s_LLM_i,t + beta_t x z_b_i,t + sum_k delta_k,t x x_k_i,t + e_i,t

where x_k are momentum, size (log market cap), value, profitability, investment, beta or
volatility, and sector dummies. Test mean(lambda_t) with a Newey-West t-test (Fama-MacBeth). A
simpler equivalent: regress the LLM score on z_b and x each week, take the residual, compute its
rank IC.

Why it matters for LLMs: Lopez-Lira, Tang and Zhu show LLMs memorize economic data before their
knowledge cutoff (for example GPT-4o recalls exact pre-cutoff S&P 500 levels; no recall after the
cutoff). A live design prevents memorizing future returns, but an LLM can still encode
pre-cutoff characteristics ("this is a high-quality large cap"), so its ratings could be a style
tilt rather than news. A positive lambda after controls means the information is stock-specific.

### Tests for many strategies at once (resample saved series, no rebuilding)

- Romano-Wolf StepM (2005): which of several strategies truly beat a common benchmark; controls
  familywise error and uses the dependence between tests, so it rejects more false nulls than Holm
  when sources are highly correlated.
- Hansen SPA (2005): does the best of many beat the benchmark; more powerful than White's Reality
  Check and less sensitive to poor or irrelevant alternatives (placebo and weak sources).
- Model Confidence Set (Hansen, Lunde, Nason 2011): which models are plausibly best. With little
  data it honestly returns many models. Loss = -IC per source-week, or -gap for portfolios.
- Deflated Sharpe ratio (Bailey and Lopez de Prado 2014): for headlining the single best portfolio.
  The expected maximum Sharpe of N unskilled trials is sqrt(V[SR]) x ((1 - g) x Phi^-1(1 - 1/N) +
  g x Phi^-1(1 - 1/(N e))), g about 0.5772. For N = 255 portfolios the bracket is about 2.85, so
  the best portfolio sits about 2.85 cross-portfolio standard deviations above the mean by luck
  alone. Closed form, zero compute.
- Probability of backtest overfitting: not recommended (built for backtests; 23 live weeks is too
  short to split).
- Harvey, Liu and Zhu (2016): new factors should clear t > 3.0. Use as a robustness flag, not the
  main rule, because this study pre-specifies a small family.

## LLM vs optimizer alone vs market conditions

Three layers, each with a standard test:
1. Signal layer (does the LLM know anything?): raw rank IC, then residual IC. Market-neutral by
   construction.
2. Portfolio layer (does the optimizer turn it into money?): the paired gap vs Q0 with mean,
   Newey-West SE and t(T-1); IR and tracking error; turnover gap and break-even cost.
3. Attribution layer (alpha or a tilt that happened to pay?): daily active return regressed on
   MKT, SMB, HML, RMW, CMA and MOM. Each loading shows which "market condition" the LLM tilted
   toward. About 115 daily points, Newey-West with about 4 lags (rule floor(4 x (T/100)^(2/9));
   Newey-West 1994 automatic selection as an alternative).

French Data Library: daily FF5 (2x3) and daily momentum (six value-weight portfolios on size and
prior 2-12 month returns) plus the risk-free rate. Fixes the missing investment factor under ETF
proxies. Released with a lag (latest shown during the research was September 2026). Fama and
French (2015) note the five-factor model's main failure is small stocks that invest heavily
despite low profitability, which matters less for S&P 500 stocks.

Regime splits (H4) leave about 5 to 15 weeks per cell, so they are descriptive. Use cell means
with CIs and an interaction regression (gap_t = a + b x Regime_t + e_t).

## Lessons from LLM-finance papers

- Lopez-Lira and Tang: return-predictability regressions of next-period returns on LLM scores,
  long-short portfolios sorted on the score, and hit rates; post-cutoff headlines to avoid
  leakage. GPT-4 had about 90% portfolio-day hit rates for the non-tradable initial reaction and
  its scores predicted the subsequent drift, especially for small stocks and negative news;
  returns declined as LLM adoption rose. This is the evaluation template to copy.
- "Evaluating LLMs in Finance Requires Explicit Bias Consideration" (2026): five recurring biases
  (look-ahead, survivorship, narrative, objective, cost). Across 164 papers from 2023 to 2025, no
  single bias was discussed in more than 28% of studies. In this design: look-ahead is handled by
  forward-only data (disable or log any web retrieval by API models); survivorship by freezing the
  universe and keeping stocks that leave the index; cost by 10 bps plus break-even cost; narrative
  and objective bias by pre-registering metrics.
- Kronos (Shi et al.): pre-trained on over 12 billion K-line records from 45 exchanges and
  evaluated with RankIC (93% higher RankIC than the leading time-series foundation model). Supports
  rank IC as the shared currency for Kronos and the LLMs.
- From the reference abstracts read on 2026-11-11 (`08_literature_and_references.md`): a
  memory-controlled benchmark (KTD-Fin) found that once leakage was masked, LLM agents' returns
  were mostly explained by passive market and style exposure (supports H5's factor attribution);
  FinRankGRPO reported a Spearman correlation of 0.023, a realistic IC scale; "The Price of
  Thought" found extra test-time reasoning does not reliably raise net returns (supports using no
  reasoning models); Cao, Jiang and Xu found that telling an LLM the downstream goal biases its
  outputs, but only before its knowledge cutoff.

## Comparing models, skills and strategies

Rule: compare on the same weeks and the same stocks (paired), then pool with fixed effects for an
average effect.

- Between models (signal): paired weekly IC difference on the same prompt, t(22) test and
  bootstrap CI. Pooled: IC_s,t = mu_t + Model + Prompt + (Model x Prompt) + e, clustered by week;
  the week effects remove market conditions common to all sources.
- Between models (portfolio): paired return difference for the same strategy and the bare prompt,
  pre-specified so the best prompt per model is not picked after the fact.
- Between skills (H3): skills minus bare and skills minus placebo on IC and on portfolio returns;
  primary test pooled across models.
- Between strategies: difference of active returns (steered minus Q0) between two strategies is a
  paired series; Romano-Wolf or MCS across the 5 strategies; mean view shift as a diagnostic
  (related to Grinold-Kahn's transfer coefficient).
- Vs HRP and equal weight: DeMiguel et al. (2009) found none of 14 optimized models consistently
  beat 1/N on Sharpe, certainty-equivalent return or turnover. Expect low power; report estimates
  with CIs.

## Power and small-sample inference

- Minimum detectable effect 2.8 x sd / sqrt(T): about 0.58 sd at T = 23. Examples: if the weekly
  sd of the IC gap is 0.05, the MDE is about 0.029; if the weekly sd of the active return is 0.4%,
  the MDE is about 0.23% per week.
- Use Student-t critical values. Harvey, Leybourne and Newbold (1997): the original
  Diebold-Mariano test with normal critical values rejected 20.3% of the time at a nominal 10%
  (n = 16, h = 2); their correction with t(n-1) critical values brought it to 14.2%. For one-step
  weekly gaps (h = 1) the correction factor is sqrt((n - 1)/n); the main gain is the t(22)
  critical values.
- Newey-West lags: 2 for 23 weekly points, 4 for about 115 daily points.
- Bootstrap: stationary or circular block, mean block length about T^(1/3), about 3, or the
  Politis-White automatic selector (2004, corrected 2009); percentile-t intervals (plain
  percentile intervals under-cover at T = 23).
- Sign-flip: 2^23 (about 8.4 million) sign patterns can be enumerated in seconds.
- Tie band: 1% of weekly return is wider than typical weekly active returns between correlated
  portfolios, so most weeks would become ties. Use a cost-based band (for example 10 bps) or half a
  weekly tracking error, with a sensitivity check.

## No-view weeks: intention-to-treat vs active weeks

The IC gate uses only information available before the week, so conditioning on it is legitimate
(no look-ahead). Report both:
- Intention-to-treat (primary): all weeks, gated weeks count as zero gaps. Answers "what does the
  full system, including its gate, deliver?"
- Active weeks (secondary): only weeks with an active view. Answers "how large is the effect when
  the LLM actually speaks?"

ITT barely costs power in t terms. If a fraction p of weeks is active with effect delta and sd
sigma, the ITT mean is p x delta and the ITT sd is about sqrt(p) x sigma, so t_ITT is about delta x
sqrt(pT) / sigma, about the same as t_active. Dilution shrinks the effect size, not the t-stat. It
does damage the sign test and win/tie/loss counts (gated weeks are exact ties), so use t-based and
sign-flip tests for ITT and report each source's share of active weeks. The cold-start IC of 0.02
makes the first 5 weeks mechanically active.

A design consequence noted in the repo review: with an IC floor of 0, a source with zero or
negative trailing IC copies Q0, so the study can show "helps" or "no effect" for that week but
never "hurts". Reporting active-week shares next to every H2 result addresses this.

## Sharpe ratio conventions

"Sharpe is king" in quant finance, and the study reports it, but with 23 weeks it is noisy, so it
is paired with tests that have more power.

- Subtract the risk-free rate (French RF column or 3-month T-bill, compounded to weekly). Without
  it, every Sharpe is overstated. For steered-minus-Q0 gaps the rate cancels in the mean gap, but
  not inside each portfolio's own Sharpe.
- Annualizing by sqrt(52) assumes i.i.d. returns. Lo (2002): with serial correlation the right
  scaling is SR(q) = eta(q) x SR, eta(q) = q / sqrt(q + 2 x sum_{k=1}^{q-1} (q - k) rho_k), and
  hedge fund annual Sharpes can be overstated by as much as 65%. Report the sqrt(52) version, check
  lag-1 autocorrelation, apply eta only if it is material.
- Precision: SE(SR) is about sqrt((1 + SR^2/2) / T) (Lo 2002). For a weekly SR of 0.1 that is 0.21
  per week, about 1.5 annualized, so the 95% CI on an annualized Sharpe is roughly plus or minus 3.
  Pair any Sharpe with the probabilistic Sharpe ratio.
- Paired Sharpe difference tests (zero compute): Jobson-Korkie with Memmel's correction (assumes
  i.i.d. normal returns); Ledoit-Wolf (2008) studentized time-series bootstrap CI for the
  difference, valid with heavy tails and autocorrelation, so it is primary; Opdyke (2007) closed
  form for stationary, ergodic returns, which shows power rises sharply with the correlation of
  the two series.

Why Q0 is the right benchmark, in numbers (weekly SRs about 0.1, Memmel variance, T = 23):

| Correlation with comparator | SE of annualized SR difference | Minimum detectable difference (2.8 x SE) |
|---|---|---|
| 0.99 (steered vs own Q0, light tilts) | about 0.21 | about 0.6 |
| 0.95 | about 0.48 | about 1.3 |
| 0.80 (steered vs equal weight or HRP) | about 0.95 | about 2.7 |

Only comparisons against Q0 have realistic power. Against equal weight and HRP, report estimates
and CIs and say plainly that 23 weeks cannot settle them.

## Holm, explained

Holm (Holm-Bonferroni) corrects for running many tests at once. Sort the m p-values from smallest
to largest; the adjusted value of the i-th is max over j <= i of min(1, (m - j + 1) x p_(j)); a
result is significant if its adjusted p is below 0.05. The smallest p is multiplied by m, the next
by m - 1, and so on. It controls the chance of any false positive in the family and is never less
powerful than Bonferroni. The cost is power, which is why the study keeps the Holm family small and
uses Benjamini-Hochberg (which controls the expected share of false discoveries) for secondary
results.

## Caveats

- With 23 weeks most portfolio-level comparisons will have wide intervals. Present that as a
  finding ("effects above X are ruled out"), not a failure.
- The residual-IC result depends on the control list; pre-register it.
- Newey-West and block bootstrap are asymptotic and 23 points is near their limit. Lean on paired
  designs and t(T-1), and cross-check with sign-flip p-values.
- The memorization literature is about backtests. A live design avoids memorized outcomes but not
  encoded firm characteristics, which is why factor and characteristic controls are central.
- Schedule the final factor regression for when French data covers the study window.
