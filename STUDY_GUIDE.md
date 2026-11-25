# Study Guide: Topics to Review Before Defending This Project

A checklist of every concept the project touches, organized by area. For each topic:
what it is, where it lives in this codebase, and the questions you should be able to
answer cold. `docs/design.md` and `plans/PLAN2.md` (the formula sheet) are the two
documents that explain most of the "how"; `llm_research/05_evaluation_methodology.md`
and `06_evaluation_research_findings.md` cover the statistics.

---

## 1. Portfolio theory fundamentals (the vocabulary layer)

You need fluent, correct use of all of these — interviewers and committee members will
probe the basics first.

- **Expected return, variance, covariance, standard deviation** — what each means, why
  covariance matters for diversification, units (daily vs weekly scaling: `Sigma = 5 *
  Sigma_daily`).
- **Long-only constraint, fully invested (weights sum to 1), position cap (5%)** —
  why caps matter, what they prevent.
- **Risk-free rate, excess return** — the Sharpe is computed *net of the T-bill rate*.
- **Active return** = portfolio return minus benchmark/Q0 return. Most of this
  project's evaluation is about *active* return, not total return.
- **Tracking error** = std dev of active return; **information ratio** = mean active
  return / tracking error. Also `expected_te_vs_q0` in weight metadata.
- **Beta, market cap, dollar volume, sector** — the universe construction and the
  characteristics used in regressions.
- **Momentum** — the classic anomaly; this project uses 12-1 month momentum, 3-month
  momentum, and 1-month reversal (`baseline/momentum.py`, `docs/design.md`).
- **Turnover** — how much trading a strategy does; matters because of transaction costs.
- **Transaction costs / break-even cost** — 10 bps per unit of traded value in the
  simulator; the report computes the cost at which the edge disappears.
- **Drawdown, max drawdown, Sortino ratio, Calmar ratio, certainty-equivalent return,
  CVaR (expected shortfall)** — all in `eval/metrics.py`.
- **Sharpe ratio variants**: plain Sharpe, Sortino, **probabilistic Sharpe ratio
  (PSR)**, **deflated Sharpe ratio (DSR)** — what each corrects for.

## 2. The quant baseline (what the LLM is compared against)

- **Rank-based scoring**: `u(x) = rank(x)/N`, cross-sectional ranks rather than raw
  values — why ranks are robust to outliers and scale.
- **Cross-sectional standardization** `z = (x - mean)/sd` with `ddof=1`.
- **The baseline score** `b = 0.5 u(momentum) + 0.5 u(Kronos forecast)` — a blend of a
  classical factor and a deep-learning forecaster.
- **Kronos**: a pretrained candlestick/time-series foundation model. Know what a
  foundation model for time series is, why they sample 20 paths per stock, and what
  "forecast spread" (`k_sd`) means — it becomes the baseline view's uncertainty.
- **Residualization** (`views/residualize.py`): regress the LLM score on the baseline
  score; keep only the residual `z_L`. Know *why*: you want the LLM's incremental
  information, not its agreement with momentum.

## 2b. Data layer details (easy to get asked about, easy to forget)

- **Universe construction** (`data/universe.py`): 100 S&P 500 stocks balanced across
  the 11 GICS sectors — top `size/sectors` per sector by dollar volume, then highest-
  volume leftovers. Know why sector balancing matters (avoids a tech-heavy universe
  confounding sector effects with LLM ratings) and what `universe_thin_sector` /
  `universe_short` warnings mean.
- **Where the universe comes from**: Wikipedia's S&P 500 constituents table (ticker,
  name, GICS sector) parsed with the stdlib HTML parser — no API key. Know this is a
  *point-in-time-ish* source: the index membership is whatever Wikipedia lists at
  snapshot time.
- **Data providers** (`data/providers/`): Alpaca (consolidated SIP feed, with
  yfinance fallback), Finnhub, FRED (DTB3 = 3-month T-bill → the risk-free rate),
  Wikipedia, and a synthetic provider for tests. Know the factory pattern and why
  multiple providers exist (redundancy, rate limits, key availability).
- **Snapshot integrity** (`data/snapshot.py`, `data/checks.py`): prices ≥300 trading
  days, news window `(previous cutoff, cutoff]`, dedup of near-duplicate news, per-
  stock token budget (newest first), immutable folders, and the check taxonomy —
  each issue maps to `skip_stock`, `forward_fill`, or `halt` (missing days, stale or
  non-positive prices, extreme moves, short history, delisting).
- **Adjusted vs raw prices**: Kronos gets adjusted OHLCV; the simulator trades at
  *raw* ledger prices with dividends/splits applied separately. Know why mixing these
  up would corrupt returns.
- **The append-only ledger** (`data/ledger.py`): raw OHLC + corporate actions are
  appended daily and never rewritten — the replay can never change history even if a
  provider re-adjusts. This is the reproducibility backbone.
- **News handling confounds**: news after 16:00 ET cutoff is dropped strictly; a
  stock with no news gets a fixed sentence; each prompt's news is capped by tokens.
  Be ready to discuss how news coverage differences across stocks could bias ratings.

## 3. Risk modeling

- **Covariance matrix estimation** — sample covariance is noisy when N is large
  relative to observations. This is the core motivation.
- **Ledoit-Wolf shrinkage**: blends sample covariance toward a scaled identity
  (`(1-alpha) S + alpha (tr(S)/N) I`). Know what shrinkage does, why it helps
  out-of-sample, and what `alpha` is. (`risk/covariance.py`)
- **Positive definiteness** — why the code asserts the smallest eigenvalue > 1e-10
  and why Cholesky factorization requires it.
- **Market-cap equilibrium prior**: `pi = delta * Sigma * w_mkt` with `delta = 2.5`.
  Know what a "market equilibrium" prior means — it's the returns implied by market-cap
  weights being optimal, the Black-Litterman starting point.
- **Correlation distance, distance metrics** — used by HRP.

## 4. Black-Litterman model (the heart of the project — study this hardest)

This is where most defense questions will land. Be able to derive or at least explain
every line of the formula in `docs/design.md` and `optimize/black_litterman.py`.

- **The idea**: Bayesian blend of a prior (equilibrium returns) with investor "views"
  (here, LLM ratings), weighted by confidence.
- **The posterior mean**:
  `mu = solve(A, inv(tau Sigma) pi + P' inv(Omega) q)` where
  `A = inv(tau Sigma) + P' inv(Omega) P`.
- **Each symbol**:
  - `pi` — equilibrium prior returns
  - `tau` — scaling of prior uncertainty (here 0.05); know what tau controls
  - `Sigma` — asset covariance
  - `P` — pick / view matrix (here identity-ish: absolute views on each stock)
  - `q` — the view returns (`q_b`, `q_L`)
  - `Omega` — view uncertainty (diagonal here); tighter Omega = stronger view
- **Why views are "absolute"**: `P = I` means each view says "stock i's expected
  return is q_i" rather than "stock i beats stock j."
- **Omega construction**:
  - Baseline: `Omega_b,ii = tau Sigma_ii * clip(k_sd_i / median(k_sd), 0.25, 4)` —
    a stock Kronos is unsure about gets a wider Omega (weaker view).
  - LLM: `Omega_L,ii = tau Sigma_ii / c_i` where `c_i = clip(1 - width_i/10, 0.1, 1)`
    and `width = high - low` — a narrow rating range means higher confidence →
    *smaller* Omega → stronger pull on the posterior.
- **Posterior covariance of the mean `M = inv(A)`** — used by Robust MVO
  (`chol(M)`), Resampled MVO (draws `mu_d = mu_BL + chol(M) z_d`), and CVaR scenarios
  (`Sigma + M`). Know why it's `Sigma + M`, not just `Sigma`.
- **Neutral view design (decision E7)**: `q_L` is centered on `mu_Q0`, so a rating
  of 0 changes nothing — you can explain why this makes "no opinion" safe.
- **Confidence-only controls (G18/G19)**: same uncertainty, zero signal — separates
  "the LLM has information" from "the LLM's confidence alone shrinks the portfolio."
- **Cholesky factorization and triangular solves** — why the code uses `solve(A, ...)`
  with a Cholesky factor instead of inverting A directly.
- **Numerical stability**: diagonal jitter retry, `OptimizerError`.

## 5. The five strategies + two controls

For each, know the objective, what it optimizes, its known weaknesses, and how it
reacts to views. Files: `optimize/strategies/`.

- **MVO**: `max w'mu - (delta/2) w'Sigma w` — textbook Markowitz. Know the classic
  criticism: estimation-error maximization / "error maximization" — small changes in
  mu produce huge weight changes. This is why the project studies multiple strategies.
- **Mean-CVaR** (Rockafellar-Uryasev): the linear-programming formulation of CVaR at
  beta=0.95 over 10,000 simulated scenarios. Know what CVaR is (expected loss in the
  worst 5%), what zeta is (the VaR auxiliary variable), and why the scenario shocks
  come from `Sigma + M`.
- **Robust MVO**: subtracts `kappa * ||chol(M)' w||_2` — worst case over an ellipsoid
  of posterior-mean uncertainty. Know what robust optimization means intuitively.
- **Resampled MVO** (a.k.a. resampled efficiency / Michaud): average 200 MVO
  solutions at perturbed means `mu_d = mu_BL + chol(M) z_d` — Monte Carlo averaging
  to stabilize weights.
- **Top-K**: pick the 20 highest posterior means, equal weight. Simple, robust to
  optimizer pathologies.
- **HRP** (control): hierarchical risk parity — correlation distance → single-linkage
  clustering → recursive bisection. Know what risk parity is (equal risk contribution)
  and why it never reads `mu` (it's forecast-free by design).
- **Equal weight** (control): `1/N`.

## 6. Optimization / numerical computing (CS side)

- **Convex optimization**: what makes an objective convex, why these problems are
  solvable, what `cvxpy` does (you write the math, it compiles to a solver).
- **Solvers**: OSQP (ADMM, first-order, approximate) then CLARABEL (interior point,
  more accurate) as fallback. Know the tradeoff.
- **Problem compilation & parameterization** (`optimize/problem_cache.py`): compile
  once per (tickers, Sigma hash, delta, cap), re-solve with `mu` as `cp.Parameter` —
  why this is the key performance trick (112 min → much less at 147 portfolios).
- **LRU cache, SHA-256 input/output hashing** — idempotency and resumability.
- **Cholesky factorization, triangular solves, eigenvalue checks, matrix conditioning.**
- **Monte Carlo simulation**: seeded RNG, common random numbers (same `Z` across
  portfolios in a week — variance reduction so differences come only from views).
- **JSON schema validation, structured LLM output, retry-on-invalid logic.**

## 7. Statistics & inference (the evaluation layer — study this second-hardest)

`eval/` implements a small econometrics toolkit. Know each test's purpose and its
assumptions.

- **Rank IC / Spearman rank correlation**: cross-sectional correlation between a
  signal and next-period returns, computed *within each week*. Why weekly (removes
  market-direction effects). Files: `eval/rank_ic.py`, `views/ic.py`.
- **ICIR** = mean(IC)/std(IC) — information coefficient information ratio.
- **Fama-MacBeth regression**: weekly cross-sectional regressions of returns on
  characteristics, then time-series stats on the coefficients — used for residual IC
  and H5 factor regressions.
- **Characteristics / control variables**: baseline score, momentum, size, volatility,
  beta, sector, value, profitability, investment — `eval/characteristics.py`.
- **Newey-West (HAC) standard errors**: corrects for autocorrelation/heteroskedasticity
  in overlapping time series. Know why plain OLS SEs would be wrong here.
- **Student-t critical values, degrees of freedom T-1.**
- **Block bootstrap** (stationary, percentile-t): resample contiguous blocks to get
  a CI that respects time dependence.
- **Sign-flip / permutation test**: exact when T ≤ 24; non-parametric p-value by
  randomly flipping signs of the weekly gaps.
- **Paired tests**: every comparison is paired by week — same stocks, same market,
  only the views differ.
- **Multiple testing** — a huge theme here:
  - **Family-wise error rate vs false discovery rate**.
  - **Holm correction** (sequential Bonferroni, more powerful than plain Bonferroni)
    on the 4 pre-registered primary comparisons.
  - **Benjamini-Hochberg** (FDR) on everything else.
  - **H4 exploratory** — unadjusted, labeled as such.
  - **Preregistration** (`config/preregistration.yaml`) — why fixing comparisons
    *before* the live run matters (prevents p-hacking / garden of forking paths).
- **Deflated Sharpe ratio, probabilistic Sharpe ratio** — haircut Sharpe for
  non-normality, sample length, and number of trials tried.
- **Data-snooping / backtest-overfitting tests**: **SPA (Superior Predictive
  Ability)**, **StepM**, **MCS (Model Confidence Set)** — `eval/metrics.py` uses
  `arch`. Know what each does at a high level.
- **Minimum detectable effect** `2.8·sd/√T` — power analysis; why a null result can
  mean "inconclusive" rather than "no effect" (G22: null results are reported).
- **ITT vs per-protocol**: all-weeks vs active-view-weeks subsets; failed sources
  trade Q0 and are excluded from "clean weeks" — know the analogy to intent-to-treat.

## 8. Factor models & risk premia

- **Fama-French five factors + momentum (FF5+M)**: market, size (SMB), value (HML),
  profitability (RMW), investment (CMA), momentum (UMD). Know what each factor
  captures and why daily French data is used for H5.
- **ETF proxies** (`universe.etf_proxies`) for live monitoring only — why proxies
  differ from the academic factors.
- **Factor loadings as controls** in the residual-IC regression — the LLM score must
  predict returns *beyond* known factors.
- **Earnings/announcement effects** — earnings-week splits in H4; news is the LLM's
  only input, so know why earnings timing matters.
- **Regime splits**: up/down markets, high/low volatility — H4 exploratory.

## 9. Experimental design & bias control (the "why this design" questions)

These are the questions that separate "ran code" from "designed a study."

- **Look-ahead bias / point-in-time data**: cutoff Friday 16:00 ET, trade Monday open,
  news after cutoff dropped strictly, snapshots frozen. Why no backtest: LLMs may
  have seen the data (pretraining contamination).
- **Survivorship bias**: the universe is fixed at study start; stocks that delist are
  handled by checks (`halt`, `skip_stock`, `forward_fill`).
- **Data snooping**: preregistration, frozen config (`frozen-v4` tag), Holm/BH,
  exploratory vs confirmatory labeling.
- **Placebo control**: filler text matched to within 2% of skills-file length —
  controls for "any structured text helps," not just content. Know why length matching
  matters (LLM attention/verbosity confounds).
- **Skills vs bare prompt**: the actual research question — do specialized skill
  files change LLM judgment?
- **Common random numbers**: same seeded scenarios/draws across portfolios in a week.
- **Q0 control**: every strategy run on baseline-only views — isolates the LLM's
  marginal contribution per strategy.
- **Fallback discipline**: failed sources trade Q0 with `fallback_q0` labels; weeks
  excluded from clean-week tests; `no_rebalance` weeks still simulated. Every fallback
  is *labeled in the data* — you can defend this as auditable, not hidden.
- **Intent-to-treat thinking**: ITT vs active-view weeks.
- **Reproducibility**: inputs/outputs hashing per job, immutable snapshots, append-only
  ledger, deterministic seeds, temperature 0, cache keys including endpoint and
  model version.
- **Confounds you should be able to name**: prompt length, output format effects,
  model version drift, market regime, sector composition, earnings clustering,
  Kronos uncertainty vs LLM uncertainty interaction.

## 10. LLM engineering (the systems side)

- **LLM inference concepts**: temperature 0 (determinism), tokens, context windows,
  structured output / JSON mode, system vs user prompts, prompt engineering.
- **Batch APIs** (Anthropic, OpenAI): async batch jobs, 24h windows, cost savings —
  why J3/J5 exist.
- **LiteLLM gateway / OpenAI-compatible endpoints**: unified API across providers,
  concurrency pools, rate limits.
- **Kronos inference on a local GPU**: sampling multiple forecast paths, path spread.
- **Caching & cost guards**: response cache keyed by (model version, prompt hash,
  ticker, week, settings, endpoint); projected-cost check before runs.
- **Validation & failure taxonomy**: invalid JSON, truncated output, `low <= likely
  <= high` constraint, invalid-rate > 15% marks source failed (N1).
- **Resumable/idempotent pipelines**: status files with input/output hashes; jobs
  skipped only if inputs unchanged; crash recovery.
- **Secrets handling**: log redaction of credentials/bearer tokens; env-var keys,
  never committed.
- **Model drift detection (G21)**: LLM providers silently update models behind the
  same API name. The study logs model-version info per source-week and has a canary
  mechanism (`llm/canary.py`, `llm/health.py`) — a pre-J4 health check per gateway
  model; failure marks sources failed for the week rather than silently using a
  different model. Know why "temperature 0" alone does NOT guarantee reproducibility
  across provider-side changes.
- **Batch vs realtime tradeoffs**: batch APIs are cheaper but have up-to-24h latency —
  the schedule is built around this (J3 Friday, J5 polls until lock margin).
- **Reasoning models**: MiniMax M2.7's reasoning tokens count as output — affects cost
  and latency estimates.
- **Cost governance**: projected-total check before batches, per-model concurrency
  pools, response cache, GPU-hour budgeting against the ARCTIC allocation.

## 11. Software architecture & CS topics

- **Pipeline/job orchestration**: J1–J9 weekly jobs, `run_weekend.py`, `--from`/`--until`
  resume, status files. Know why it's jobs-with-hashes rather than a monolith.
- **Provider pattern / factory**: `data/providers/` (alpaca, finnhub, fred,
  yfinance, wikipedia, synthetic) behind a common interface; `llm/providers/` similarly.
- **Append-only ledger** (`data/ledger.py`): raw prices and corporate actions are
  append-only — history cannot drift when providers re-adjust. This is a big deal for
  reproducibility; be ready to explain it.
- **Corporate actions**: dividends (ex-date cash credit), splits (share rescale,
  average-cost adjustment) — why adjusted vs raw prices matter.
- **Market calendar**: ISO week IDs, holiday-aware trade dates, open-to-open returns.
- **Testing**: unit/integration/e2e structure, synthetic data world
  (`scripts/make_synthetic_world.py`) — the whole pipeline is testable without real
  APIs.
- **Type hints, mypy, ruff, pytest** — the project's quality tooling (`pyproject.toml`).
- **Config management**: YAML configs, `DECISION(<id>)` tags linking code to documented
  decisions — a traceability pattern worth mentioning.
- **Git/CI**: `.gitlab-ci.yml`, `.github/` — know how checks run.

## 12. Project status, limitations, and open questions

A defense almost always includes "what's done vs. what's planned" and "what could be
wrong." Read `plans/PROGRESS.md`, `plans/results.md`, `docs/open_questions.md`,
`manualwork.md`, and the status notes in `llm_research/`.

- **Implementation status**: the code currently implements an older eval version
  (Holm everywhere, ETF-proxy regressions of total returns, Sharpe without the
  risk-free rate) — `plans/PLAN4.md` Phases 5–9 bring it to the design in
  `docs/design.md`. Know which parts are built, tested, and which are pending.
- **UNVERIFIED providers**: several adapters (Wikipedia, FRED, Finnhub paths) are
  marked `# UNVERIFIED: needs real smoke test` — be honest about what has and hasn't
  been exercised against live services.
- **Known limitations to name unprompted**:
  - 100-stock universe (set by the ARCTIC GPU-hour allocation, not statistical power).
  - ~24 live weeks planned → low power; the MDE formula exists precisely because of
    this. A null result is "inconclusive," not "no effect."
  - LLM ratings are integers in [-5,5] — coarse signal; width is the only
    confidence channel.
  - Kronos and the LLMs may share correlated failure modes (both read the same news).
  - The study measures *this* universe, *these* models, *this* period — external
    validity is limited by design.
- **Sensitivity analyses built in**: model-change sensitivity, view sensitivity,
  fallback/solver-failed week reporting — the design anticipates its own fragility.
- **`view_shift` diagnostic**: mean |w − w_Q0| per strategy; if a strategy barely
  moves from Q0 it *cannot* show an LLM effect — the report warns at 0.01. This is a
  great example of the project checking its own statistical power per strategy.

## 13. Things to be able to say in one sentence (elevator version)

Practice these — they're the spine of your defense:

1. "We hold data, baseline, risk model, and guardrails fixed; only the LLM's ratings
   and the strategy vary — a controlled comparison."
2. "LLM ratings become Black-Litterman views whose confidence (Omega) comes from the
   rating range width; they update a Ledoit-Wolf covariance and market-cap prior into
   a posterior mean and covariance-of-mean that five different strategies consume."
3. "We compare every steered portfolio to its own quant-only Q0 on *active* return,
   with paired weekly tests, Newey-West errors, and Holm/BH correction — and we
   pre-registered the four main comparisons."
4. "Because LLMs may have seen historical data, we run live forward-only with a
   frozen snapshot and cutoff discipline instead of backtesting."
5. "Every failure is labeled and falls back to Q0 or equal weight, and null results
   are reported — the study measures whether LLM views help, not whether they can be
   made to look good."

## 14. Suggested review order

1. `docs/design.md` end-to-end (it's the map).
2. `plans/PLAN2.md` formula sheet alongside `optimize/` source.
3. Black-Litterman derivation (section 4 above) — be able to reproduce `mu` and `M`.
4. Each strategy's objective and its failure mode (section 5).
5. The evaluation tests in `eval/tests.py`, `eval/multiple.py`, `eval/metrics.py`
   (section 7).
6. `llm_research/05` and `06` for the statistical reasoning and literature.
7. Walk one synthetic week (`run_weekend.py --synthetic`) and trace a rating from
   prompt → validation → view → posterior → weights → simulated returns.
