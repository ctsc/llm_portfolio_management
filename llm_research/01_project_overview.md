# 01 Project overview

Last updated: 2026-11-17. Current as of the latest research proposal and ARCTIC request in
`current_documents/`.

## Title

Specialized LLM Skills Across Fixed Quantitative Portfolio Optimization Strategies

## People

- Carter Tierney, MS Computer Science, Georgia State University (graduating May 2027). The study
  is his MS project (the degree requires a project, not a thesis).
- Advisor: Dr. Parag Tamhankar (address him as "Parag" in emails). He submits the ARCTIC request
  as PI; Carter writes it.
- ARCTIC contact: Andalib (GSU research computing). See `11_correspondence.md`.

## Research question

How can LLM views built from specialized skills improve quantitative portfolio optimization
strategies, and which strategies do LLMs help or hurt the most, and why?

## Research questions (wording used in the proposal)

| ID | Question (code and report keep H1 to H5; Hn tests RQn) |
|---|---|
| RQ1 | How often do the LLM's ratings alone beat the quant baseline's ratings? |
| RQ2 | How often does adding the LLM views improve each strategy's portfolio, compared with the same strategy run on the baseline alone (Q0)? |
| RQ3 | Do specialized skills change the result compared with bare prompts and placebo? (H3b: do narrow rating ranges predict better than wide ones?) |
| RQ4 | Which strategies do the LLM views help or hurt the most, and in which conditions (up or down markets, high or low volatility, earnings weeks)? Exploratory. |
| RQ5 | How much of the result comes from luck or general market movement rather than the LLM? |

How each one is tested now: `05_evaluation_methodology.md`. The proposal ends with a draft
hypothesis for each research question (written by Claude on 2026-11-11 for Carter to revise):
small LLM IC that does not reliably beat the baseline (RQ1); small, mostly noise-level portfolio
effects, clearest in mean-variance (RQ2); skills slightly above bare and placebo, placebo about equal
to bare (RQ3); larger effects in high-volatility and earnings weeks (RQ4); most of any gap explained
by style exposure (RQ5).

## What the study holds fixed and what it varies

Held fixed for every portfolio: the data snapshot, the quant baseline, the Ledoit-Wolf
covariance, the way views are built, Black-Litterman, the guardrails and the trade simulator.

Varied: (1) which LLM ratings are used (model and prompt type), and (2) which optimization
strategy uses them. Each strategy is compared with its own baseline-only version (Q0), so the
strategy cancels out of the comparison.

## Why it is novel (as framed in the proposal and ARCTIC request)

Prior work that puts LLM forecasts into Black-Litterman views mostly:
1. tests one optimizer at a time, so it cannot say which strategies benefit;
2. tests on historical data, where the LLM may have read news about the period (lookahead and
   memorization), which inflates results;
3. rarely checks whether a gain comes from the prompt's content or just from a longer prompt.

This study runs live (forward only, no backtest), compares five fixed strategies on the same
ratings in the same weeks, and uses a same-length placebo prompt as a control. To Carter's
knowledge, no prior study has tested LLM views across several fixed strategies on live data in
the same period.

## Current scope at a glance

| Item | Current value |
|---|---|
| Universe | 100 S&P 500 stocks, sector balanced (B4, set by the 1,000 GPU-hour startup allocation) |
| Rebalance | Weekly. Data frozen Friday close, weights locked before Monday open |
| Study length | One-week pilot, then about 23 live weeks (about 24 weeks used in compute estimates) |
| Open-weight LLMs | Gemma 4 31B and MiniMax M2.7, already on ARCTIC's LiteLLM gateway |
| API LLMs | Two cheap/small models: Claude Haiku 5.5 and GPT-6 Luna (no frontier reasoning models) |
| Prompt types | Bare, 4 specialized skills files of similar length, and a same-length placebo (6 types) |
| LLM output | JSON low / likely / high integers on a -5 to 5 scale plus a short rationale |
| Quant baseline | Momentum plus Kronos (pretrained candlestick forecaster, Kronos-base, 102M parameters) |
| Strategies | Mean-variance (MVO), Mean-CVaR, Robust MVO, Resampled MVO, Top-K (K = 20) |
| Controls | Q0 per strategy (baseline only), plus HRP and equal weight (no forecasts) |
| Where it runs | LLM ratings on ARCTIC's gateway; everything else on Carter's desktop |
| ARCTIC allocation | Default startup allocation granted 2026-11-16: 1,000 GPU and 10,000 CPU service units (requested 2,730 or 910 GPU-hours). The study uses 100 stocks |

## Timeline Carter has stated

- Live run continues as long as possible, ideally through at least April 1, 2027.
- MS project paper written in April 2027; graduation May 2027.
- Results will form the MS project and a conference paper submission (his first).

## Repository

- Code: `C:\dev\portfoliomanagement` (moved out of OneDrive on 2026-11-16; weekly runs use the WSL
  clone `~/pm`) (this research folder is
  `llm_research/` inside it; plans are in `plans/`, code docs in `docs/`) (Python package
  `portstudy`, Python 3.11). GitHub: `ctsc/llm_portfolio_management`.
- Built by Devin from plan files (`PLAN.md`, `PLAN2.md`, `PLAN3.md`). Devin may commit and push,
  but history must show only Carter (`ctsc`), with no AI or co-author lines.
- All manual work Carter has to do is kept in one file, `manualwork.md` at the repo root (not in
  the plans).
- Parts of the repo still describe the older ARCTIC/Slurm design. See
  `10_open_items_and_next_steps.md`.
