# PLAN4.md: Align the repository with the current research (desktop run, LLM-only ARCTIC, new evaluation)

Written 2026-11-11. PLAN.md (v1), PLAN2.md (v2) and PLAN3.md (v3 phases 1 to 6) are built,
committed and pushed (HEAD `44949aa`). PLAN3 phase 7 (manual work) was moved into `manualwork.md`
and is not done. All of Carter's manual work now lives in that one file at the repository root;
this plan has no manual phase. Since then the research changed in three ways that the code does not reflect yet:

1. **Compute moved to Carter's desktop.** ARCTIC is now used only for LLM calls through its LiteLLM
   gateway on the two models it already hosts (Gemma 4 31B, MiniMax M2.7). Kronos, data, all
   optimizers, the simulator, evaluation and backups run on Carter's Windows desktop inside WSL2.
   No Slurm jobs, no CPU hours, no small-model deployments, no vLLM fallback server.
2. **The shuffled-ratings test (H5, decision G6) is dropped.** It cost about 638 CPU-hours at 300
   stocks and could never reach significance under Holm (52 / 501 = 0.104). No evaluation method
   may add CPU or GPU hours.
3. **The evaluation was redesigned** around industry-standard quant tests that run on saved data in
   seconds: rank IC and residual IC (Fama-MacBeth), active return vs Q0, information ratio,
   Ledoit-Wolf Sharpe test, break-even cost, Fama-French 5 plus momentum regressions of the active
   return, the placebo prompt as the content control, Holm on a pre-registered main family and Benjamini-Hochberg
   on the rest.

4. **A full code and plan review on 2026-11-11 (`plans/REVIEW5_fable_2026-11-11.md`) found problems
   that change what the study measures and several bugs.** Carter approved the fixes on 2026-11-11:
   the LLM view is centered on the Q0 forecast so a rating of 0 is neutral (E7), Mean-CVaR and
   Resampled MVO are rescaled on synthetic data before the freeze so views can move them (S8), and a
   stop that lands on a rebalance day resets the stock's reference price (G1b). The review's bug
   fixes (gateway request failures, prompt fields, data-folder hygiene, delisted stocks, and others)
   are folded into Phases 3, 3A and the new Phase 3B. On 2026-11-12 Carter also approved the
   review's remaining recommendations: four pooled main comparisons (G7), a fixed view scale of
   0.02 instead of the trailing-IC switch (A8), a clean start at the first live week (A7), a
   five-day no-price liquidation (U2), the Friday 16:00 ET cutoff unchanged (A9), the
   `momentum_only` Kronos fallback (K3), confidence-only control portfolios for Robust and
   Resampled MVO (G19), and four skills files (B1).
5. **Andalib answered on 2026-11-12** (`llm_research/09_arctic_process_and_model_facts.md` section G).
   The gateway passes `seed` through but outputs are not guaranteed identical, because requests
   are batched with other users'; the service has no changelog, so he suggested logging each
   response's `model` field, running a few fixed canary prompts every weekend, and measuring
   repeatability in the pilot. MiniMax's thinking cannot be turned off and its reasoning counts as
   output. Access route, metering and per-key limits are still being decided by his team. Carter
   approved: temperature 0 stays (D4), a pilot repeat check (G20), and weekly model checks with
   canaries (G21). His follow-ups (2026-11-12 to 2026-11-16): Parag requests access through Elpis
   with the request PDF attached (no Hydra ticket); MiniMax returns its thinking in a separate
   `reasoning_content` field and reports `reasoning_tokens` in `usage`; the gateway returns no model
   version or `system_fingerprint`, and `/v1/models` lists names only, but each response carries the
   `x-litellm-model-id` (deployment ID) and `x-litellm-version` (gateway version) headers, and
   `/v1/model/info` has an `updated_at` per model that changes when gateway settings are edited. Nothing in this plan waits on the open ARCTIC answers.
6. **ARCTIC allocation (2026-11-16).** ARCTIC granted the default startup allocation for GSU users:
   1,000 GPU service units and 10,000 CPU service units, metered as GPU and CPU usage, with a Slurm
   account Carter holds. The universe is therefore 100 stocks (B4), and the weekly run tracks
   estimated ARCTIC usage against the 1,000 GPU-hour limit (N7).
7. **Logging and never losing progress (N6, Carter, 2026-11-12).** Every event in a weekend run must
   be recorded, every fallback must be listed with its reason, every error must keep its full
   traceback, and every completed unit of work must be saved to disk before the next one starts, so
   a crash or power cut at any moment loses at most the work in flight. Phase 3A builds it.
8. **Measure everything in one fixed scope (Carter, 2026-11-17).** Every metric the evaluation can
   compute is computed and saved for every portfolio, source, model, prompt and strategy, both per
   cell and pooled, and null results are reported (G22). The Sharpe ratio net of the T-bill rate
   is a headline portfolio metric. The scope is fixed: 100 stocks, the two gateway models plus both
   API models (Claude Haiku 5.5 and GPT-6 Luna, D1), four skills files, five strategies. RQ1's main
   test becomes the pooled LLM-minus-baseline rank IC gap, matching the proposal (G7).

PLAN4 updates the code, config and tests to match. Every manual step is in `manualwork.md`, not
in this plan. The docs and README were already rewritten on 2026-11-11 to the target
design, with "pending PLAN4 Phase N" notes where the code still differs.

Read first, in this order:
1. `llm_research/current_documents/Project_Proposal_IEEE.pdf` (current research; Evaluation section is the spec
   for Phases 5 to 8).
2. `llm_research/current_documents/ARCTIC_Resource_Request.pdf` (current compute request: gateway GPU hours only).
3. Section 0.3 below (decision changes). The proposal was updated on 2026-11-17 (four skills
   files, 100 stocks, both API models, Sharpe net of the T-bill rate, every metric saved) and
   matches section 0.3; wherever a detail still differs, section 0.3 wins.
4. `plans/REVIEW5_fable_2026-11-11.md` (the review behind E7, S8, G1b, N5, C8, A7, U2, D5 and the
   changed G8, G13, G14, G15, K2; it has the evidence and the experiment outputs).
5. This file.

Supporting notes: `llm_research/` in this repository (research notes, decisions, methodology,
findings; `llm_research/09_arctic_process_and_model_facts.md` holds the ARCTIC gateway and model
facts that Phase 3 depends on). `llm_research/05_evaluation_methodology.md` and `06_evaluation_research_findings.md` give
the formulas and reasoning behind every evaluation phase. Where this plan and those notes differ,
this plan wins. All plans and the build log live in `plans/`.

---

## 0. Working rules

All of PLAN2.md section 0 and PLAN3.md section 0 still apply (offline tests only, no em dashes,
no "arm" for experimental conditions, small pure modules, idempotent jobs, UNVERIFIED tags,
never change a `DECISION` value except the ones section 0.3 changes). The most important ones:

1. **Order.** Work through the phases in order. Do not start phase N+1 until phase N's
   completion criteria pass. A truly blocked phase is logged in `plans/PROGRESS.md` and the next phase
   that does not depend on it continues.
2. **After each phase:** run `ruff check src tests scripts`, `ruff format --check src tests scripts`,
   `mypy`, and `pytest` (default selection). Append a `v4 phase N: <title>` entry to `plans/PROGRESS.md`
   (what changed, test counts, any test changed or deleted and why, choices made while
   unattended). Then commit and push.
3. **Git.** Devin may `git add`, commit and push. Commit subject: `v4 phase N: <title>`.
   Push to `origin master` (`https://github.com/ctsc/llm_portfolio_management.git`), then
   `git push gitlab master`. Author and committer must be exactly
   `ctsc <cartertierney0@gmail.com>` (set `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL`,
   `GIT_COMMITTER_NAME`, `GIT_COMMITTER_EMAIL` for every commit). No `Co-authored-by`,
   `Signed-off-by`, "Generated with", Devin, AI, bot name, session link or any other trailer.
   Check `git log -1 --format='%an <%ae>|%cn <%ce>'` and `git log -1 --format=%B` before every push
   and amend the unpushed commit if either fails. No force push, rebase, history rewrite,
   branches, tags or pull requests.
4. **Uncommitted work already in the tree (from 2026-11-11), now in `C:\dev\portfoliomanagement`.** Carter's reorganization is not
   committed yet: plan files, `PROGRESS.md` and `results.md` moved into `plans/`; `llm_research/`
   added (research notes and `current_documents/` with the current proposal and ARCTIC request);
   `docs/proposals/`, `docs/arctic_setup.md` and `docs/arctic_gpu_request.md` deleted;
   `manualwork.md` rewritten as the single manual list; README and every doc rewritten to the target design with "pending PLAN4 Phase N" notes;
   `docs/desktop_setup.md` added; `scripts/kronos_smoke.py` added; `scripts/check_banned_words.py`
   skips `plans/`; `tests/unit/test_phase16_checklist.py` reads `plans/PROGRESS.md`. Keep all of it
   and include it in the Phase 1 commit. `llm_research/11_correspondence.md` is gitignored (emails
   stay local); never commit it or any other email text. Do not edit `llm_research/` except where a phase says so.
5. **Work autonomously.** Carter is not available during the build. Never stop to ask. When
   something is unclear, choose the option most consistent with the proposal and this plan, write
   it under `Choices made while unattended` in that phase's PROGRESS entry, and continue.
6. **Wording.** Never write "the researcher" or "the student". It is Carter Tierney's MS Computer
   Science research project at Georgia State University, advised by Dr. Parag Tamhankar.
7. **Do not edit prompt files** in `prompts/` (their hashes are frozen by the manifest). Carter
   writes them (`manualwork.md`).
8. **Numbers must come from the code or a run.** Any timing or count written into README or docs
   comes from config, code, or a command run in this session; otherwise it is labeled an estimate
   with its source.
9. **No new compute.** No phase may add a method that re-runs an LLM, Kronos or an optimizer for
   evaluation. Resampling of saved series (bootstrap, sign-flip, Ledoit-Wolf, SPA, MCS) is fine.
10. **Tests stay offline and fast.** Default `pytest` must stay offline (stub providers, synthetic
    world). Keep the e2e suite under the existing 15-minute bound.
11. **Build as if every key exists.** No real key is available during the build
    (`ARCTIC_LITELLM_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `ALPACA_API_KEY_ID`,
    `ALPACA_API_SECRET_KEY`, `FINNHUB_API_KEY`, `FRED_API_KEY`). Build every real-service path in
    full, read keys only from the environment, and test each path with the stub transports. Never
    add a placeholder key, skip a code path because its key is missing, or mark a phase blocked
    for lack of a key. Carter adds the keys and runs the real-service checks (`manualwork.md` C
    and E) after Phase 11.

### 0.1 Git exception to Carter's planning standard

Carter's generic planning standard says an agent never runs Git write commands. For this
repository Carter has a standing rule that overrides it: Devin may commit and push, but history
must show only `ctsc`, with no AI or co-author lines (rule 3 above).

### 0.2 What is already built (audit on 2026-11-11)

| Area | State |
|---|---|
| v1 phases 1 to 16, v2 phases 1 to 14, review fixes, v3 phases 1 to 6 | Built, tested, pushed (`44949aa`) |
| Weekly pipeline J1 to J9, `scripts/run_weekend.py` | Built; runs locally with `--synthetic`; still carries ARCTIC-only flags (`--gateway-fallback`, `--check-walltime`, login-mode wording) |
| LiteLLM gateway provider, health check, concurrency 4 per model | Built; `serving: slurm_fallback` path and `fallback_endpoint_file` still present |
| Five strategies, Q0 per strategy, HRP, EW, simulator, Black-Litterman, views | Built; no change needed except Kronos batching |
| Evaluation (`src/portstudy/eval/`) | Built for the old plan: IC gaps with Holm across sources, H2 portfolio gaps, H3 IC only, H3b, H4, H5 shuffle plus ETF-proxy factor regression of total returns, Sharpe without risk-free subtraction |
| Shuffle test (`eval/shuffle.py`, `portstudy shuffle ...`, `shuffle_array.sbatch`) | Built; to be removed |
| ARCTIC Slurm layer (`scripts/arctic/`, W1 and M1 guards) | Built; to be removed |
| Small models (Gemma 4 E4B, Phi-4 Mini, Llama 3.2 3B) | In `config/models.yaml` as disabled fallbacks with `provider_model: TBD`; to be removed |
| Desktop benchmark (2026-11-10, WSL2, Ryzen 5 8400F, RTX 3050 6 GB) | Done by Carter: J7 at 300 stocks 519 s per week at 3 sources (6,748 s projected at 51); Kronos 300 stocks 3,373 s, peak 4.49 GB GPU memory; 100 stocks 195 s, 1.79 GB |
| PLAN4 | Did not exist before this file |

### 0.3 Decision changes in PLAN4 (Carter approved these on 2026-11-10 and 2026-11-11)

| ID | Change | New value |
|---|---|---|
| D1 (changed) | models | ARCTIC LiteLLM gateway: Gemma 4 31B and MiniMax M2.7 only (no deployment requests). API: both Claude Haiku 5.5 and GPT-6 Luna (confirmed 2026-11-17). No Qwen or DeepSeek (GSU tool registry). |
| D2 (changed) | gateway outage | That model's week is marked failed and left out of its statistics. No Slurm or vLLM fallback. |
| L1 (new) | where it runs | Everything except LLM calls runs on Carter's desktop in WSL2 Ubuntu (Python 3.11); Kronos on the local RTX 3050. ARCTIC request: gateway GPU hours only, no CPU hours or storage. |
| N2 (changed) | weekly trigger | Carter runs `python scripts/run_weekend.py` by hand on the desktop, Friday night (Saturday morning as backup). No scheduler, no Slurm submit. |
| B4 (decided 2026-11-16) | universe size | 100 stocks. ARCTIC granted the default startup allocation (1,000 GPU and 10,000 CPU service units). At the request's estimates, 100 stocks uses about 607 GPU-hours over 24 weeks (about 40% headroom) and 300 stocks about 1,820, which does not fit. `universe.size` stays 100. Revisit before the freeze only if the measured gateway timing (Phase 3 `--time-gateway`) shows a larger universe fits within 1,000 GPU-hours with at least a 25% margin, and record any change in PROGRESS. |
| K2 (new) | Kronos batching | Kronos forecasts in chunks of `baseline.kronos.batch_size` stocks (default 100) so the universe fits in 6 GB of GPU memory (100 stocks is one chunk; 300 also fits). Each (sample, chunk) gets its own seed derived from the config seed, so chunks do not share random draws. Chunking changes the sampled paths, so `batch_size` is frozen with the config at the freeze. |
| G6 (retired) | shuffle test | Dropped. No shuffled-rating portfolios. |
| W1, M1 (retired) | Slurm walltime and memory | Dropped with the Slurm layer. |
| G7 (changed 2026-11-12; revised 2026-11-17) | multiple testing | Holm on a pre-registered main family of exactly four pooled comparisons (`config/preregistration.yaml`): (1) pooled LLM-minus-baseline rank IC gap on the same stocks (RQ1, matching the proposal; pooled LLM rank IC above zero moved to the secondary family on 2026-11-17), (2) pooled steered-minus-Q0 tilt rank IC (RQ2 signal), (3) pooled skills-minus-bare rank IC (RQ3), (4) pooled skills-minus-placebo rank IC (RQ5, content control). Each pools every enabled gateway and API model and, for (3) and (4), every enabled skills file, in a regression with model fixed effects. Benjamini-Hochberg false discovery rate on every other comparison (secondary); H4 exploratory with no correction. Reason: at 100 stocks the smallest detectable weekly IC is about 0.059 for one test and about 0.07 under Holm with four, so fewer, pooled main tests keep the power. |
| G8 (new) | H2 reporting | Every H2 result is reported intention-to-treat (all clean weeks, using the realized return gap in every week, including weeks with no active LLM view) and active-weeks only, with each source's share of active weeks. A week with no active view is not forced to a zero gap: the steered portfolio's holdings and average costs differ from Q0's, so stops and costs can make the realized gap non-zero. |
| G9 (new) | residual IC | H1 and H2 add a residual IC: weekly Fama-MacBeth regression of next-week return ranks on the score, controlling for the baseline score, momentum, size, volatility, beta, sector and trailing value, profitability and investment factor loadings. The control list is fixed in `config/preregistration.yaml`. |
| G10 (new) | portfolio metrics | Sharpe net of the risk-free rate (FRED T-bill already in each snapshot); Sortino, max drawdown and CVaR (95%) from daily returns; information ratio and tracking error vs Q0; probabilistic Sharpe; certainty-equivalent return; break-even cost. |
| G11 (new) | factor model | Final H5 regressions use Kenneth French daily FF5 plus momentum and RF, on daily active returns (steered minus Q0) and on each portfolio's excess return. ETF proxies stay for live monitoring only and are labeled as proxies. |
| G12 (new) | small-sample inference | Student-t critical values with T - 1 degrees of freedom; Newey-West lags from `floor(4 (T/100)^(2/9))`; stationary block bootstrap with percentile-t intervals; exact sign-flip enumeration when T <= 24, Monte Carlo above; Ledoit-Wolf (2008) bootstrap test for Sharpe differences; tie band 0.001 (10 bps) with a sensitivity table. |
| G13 (new, changed by review 5) | view sensitivity | Sensitivity is measured as the expected weekly tracking error against Q0, `sqrt((w - w_Q0)' Sigma (w - w_Q0))`, not as view shift. With E7 a rating of 0 leaves the forecast mean unchanged, so MVO and Top-K have zero tracking error when the LLM says nothing; for Robust, Resampled and Mean-CVaR the part caused only by lower uncertainty is measured with the confidence-only counterfactual (G18) and subtracted. Pre-registered rule: a strategy whose median information tracking error is below `view_sensitivity_min_tracking_error` (set by Carter from the Phase 3B and pilot numbers) is reported as "insensitive to views by construction", not as evidence that the LLM does not help. View shift is still recorded as a descriptive column. |
| G14 (new) | H2 signal baseline | The H2 signal test compares the steered Black-Litterman tilt's rank IC with **Q0's tilt** rank IC (same posterior machinery, baseline view only), not with the raw baseline score `z_b`. With E7 the two tilts are identical when every rating is 0. Every IC gap (H1, H2, H3) is computed on the same stocks each week (the intersection of stocks with a valid score in both signals). |
| U1 (new) | large and extreme moves | Using adjusted prices over the week since the previous cutoff: a stock whose 5-day return is at or beyond plus or minus 10% is kept and flagged (`large_move`, counted in the report); a stock that falls 40% or more (single day or 5-day) is removed from the study permanently from the next snapshot, with no replacement, its position sold at the next Monday open, and the removal logged in `data/universe_removals.csv`. A raw-price jump that the adjusted series does not show (a likely missed split) is a data error that skips the stock for that week only. The old rule (any 40% daily move in the whole 300-day history skips the stock) is removed. |
| N4 (new) | week-to-week integrity | Every week appends one row to `data/outputs/study_log.csv` (job states and durations, git commit, config and preregistration hashes, stock count, removals, flags, fallbacks, invalid rates, cost, backup verification). Weekly returns are measured Monday open to next Monday open (the same window as rank IC). `run_weekend.py` keeps Windows awake while it runs, survives VPN drops by pausing and retrying, checkpoints long jobs, and shows progress and projected finish against the Monday lock. |
| G15 (new, changed by review 5) | gateway output handling | Per-model settings already exist (`llm.settings.<model>` and each model's `settings` in `config/models.yaml`), so this is config plus small code changes: Gemma 4 31B keeps `max_output_tokens: 300`; MiniMax M2.7 starts at 2,000 (it always thinks, the thinking counts toward the limit, and a too-small limit returns empty content with `finish_reason: length`). The request timeout becomes a per-model setting (MiniMax higher than the current fixed 180 s). The reasoning text is saved in each record. Every gateway request records prompt, completion and reasoning tokens, `finish_reason` and seconds. The parser already skips Gemma's `<\|channel>thought` marker because it takes the first JSON object in the text, so no strip rule is needed; a test pins that behavior. After the pilot the accepted `seed` and JSON-mode features are frozen in config, and a later rejection is a failure, not a silent change. |
| G16 (new) | price feed | The Alpaca `feed` becomes a config key (`data.alpaca_feed`) that defaults to `sip` (consolidated) instead of the hard-coded `iex` (one small exchange, which distorts opens, closes and volume). The real-service smoke check confirms the bars come from the configured feed. If the free plan refuses SIP, yfinance becomes the price source and the choice is recorded in `study_log.csv`. |
| G17 (new, revised 2026-11-12) | ARCTIC sizing basis | The request now uses about 300 to 8,000 input tokens per request, about 300 output tokens for Gemma and up to about 2,000 for MiniMax (reasoning counts as output, confirmed by Andalib), and estimated V100-equivalent costs of about 40 GPU-seconds per Gemma request and about 90 per MiniMax request (from model size and memory bandwidth, not measured): 1,820 GPU-hours estimated and 2,730 requested at 300 stocks, 607 and 910 at 100 stocks, with weekly token totals listed because metering is undecided. The timing tool (Phase 3) measures tokens and seconds per request on the gateway; the request and `llm_research/04_models_compute_arctic.md` are corrected from those numbers and from ARCTIC's metering answer. |
| E7 (new, review 5) | neutral LLM view | The LLM view is centered on the Q0 forecast, not on the market prior: `q_L = mu_Q0 + IC_L * sigma * z_L`, where `mu_Q0` is that week's posterior mean with the baseline view only (equal to `pi` when the baseline view is dropped). `Omega_L` is unchanged (E1). A rating of 0 then leaves the posterior mean exactly equal to Q0's, so MVO and Top-K equal Q0 when the LLM says nothing. Replaces `q_L = pi + IC_L * sigma * z_L`. Review evidence: with every stock rated 0, the old formula moved 5% to 11% of the MVO, Robust and Top-K portfolios. |
| G18 (new, review 5) | confidence-only counterfactual | For each source-week the evaluation also computes the posterior with the same LLM view uncertainty and `z_L = 0` (one linear solve from saved files, no optimizer). It gives the confidence-only tilt and posterior covariance used by G13 and reported next to the H2 signal test. |
| S8 (new, review 5) | strategy sensitivity calibration | Before the freeze, Mean-CVaR's `lam` and a new `strategies.resampled.draw_scale` (the fraction of the posterior covariance the resampled means are drawn from, default 1.0) are set on the synthetic world at the study's universe size (100 stocks, B4) by a fixed rule (Phase 3B), so LLM views can move those two strategies. This replaces S4's fixed `lam = 1.0`. Carter confirms the values before the freeze; no real data is used. Review evidence: under noise ratings MVO moved 25% to 38% of the portfolio, Mean-CVaR and Resampled 1% to 3%. |
| G1b (new, review 5) | stop on a rebalance day | When a pending stop-loss or take-profit is merged into the Monday rebalance and the position is kept, that position's reference price (average cost) is reset to the rebalance open price. The old behavior left the reference unchanged, so the same stop fired again at Monday's close, the stock was sold on Tuesday, and the cash sat idle until the next Monday. |
| N5 (new, review 5) | gateway request failures | A gateway request that still fails after its retries is written as an invalid record with reason `provider_error`, so it counts toward the 15% rule (N1) and no longer leaves the source incomplete. J4 ends as failed, not done, while any planned request has no record, so a rerun resumes it. |
| C8 (new, review 5) | prompt name and sector | The company name and GICS sector in every prompt, and the sector control in the residual IC, come from the frozen `config/universe.csv`, not from a weekly yfinance call. yfinance supplies market caps only, with the last good value reused when a call fails. |
| A7 (new, review 5; pilot rule decided 2026-11-12) | data-folder hygiene | Dry runs use a scratch `PORTSTUDY_DATA_DIR`. The preflight refuses a live run when the live data folder holds any week before the pilot. New key `views.ic_history_start` with default `live_start_week`: weeks before the first live week (the pilot) never count toward any IC used for views and are excluded from `ic_history.csv` used in evaluation, and every portfolio, Q0 and the controls included, starts from cash at the first live week's Monday open, so all start with the same costs. `first_locked_week` keeps the old behavior for tests. The price ledger is rebuilt from empty at the freeze, because it is append-only and would keep rows from an earlier feed. |
| U2 (new, review 5; confirmed 2026-11-12) | no-price liquidation | A held stock with no price for `simulator.no_price_liquidation_days` trading days (default 5, one weekly cycle) is liquidated at its last close with the usual cost, logged, and counted in the report. Covers delistings and cash acquisitions, where no price ever returns; a trading halt usually clears within a day or two. |
| D5 (new, review 5) | API model versions | Each enabled API model must have a dated `snapshot_id` in `config/models.yaml`; the preflight refuses a live run without it. Carter enters the IDs (`manualwork.md`). |
| A8 (new, 2026-11-12) | fixed view scale | `views.ic_mode: fixed` (new key; `trailing` keeps the old A6 behavior for tests). In `fixed` mode both `IC_b` and `IC_L` equal `views.fixed_ic` (0.02) every week, for every source, so no view is switched on or off by a noisy 8-week IC. Weekly rank ICs are still computed and stored in `ic_history.csv` for evaluation. Replaces the trailing-IC gating and cold start (A6) for the live study. Reason: an 8-week mean IC has a standard error near 0.02 at 300 stocks and near 0.035 at 100, so the switch turned a no-skill source on about half the weeks and could drop Q0's own baseline view. |
| A9 (unchanged, confirmed 2026-11-12) | news cutoff | Friday 16:00 ET stays the data cutoff. Reasons: little S&P 500 news arrives after Friday's close, and a later cutoff would shrink the buffer before the Monday 09:00 lock while J4's time is unmeasured. Revisit only if the measured weekend run is short. |
| K3 (new, 2026-11-12) | Kronos failure | `baseline.on_kronos_failure: momentum_only` (default changed from `hold`). A failed J2 uses the momentum rank alone for every portfolio that week, with `k_sd` set to the median, and the week is flagged. Every portfolio, Q0 included, uses the same backup baseline, so comparisons stay fair. |
| G19 (new, 2026-11-12) | confidence-only control portfolios | For every steered source and for Robust MVO and Resampled MVO only, J7 also builds a locked portfolio from the posterior with the same LLM view uncertainty and `z_L = 0` (posterior mean equal to Q0's by E7; posterior covariance `M_conf`). Portfolio key `<strategy>__CONF__<model>__<config_id>`. These are simulated and replayed like every other portfolio. The evaluation splits steered minus Q0 into information (steered minus confidence-only) and confidence (confidence-only minus Q0). Not built for MVO and Top-K (their confidence-only portfolio equals Q0 by E7) or Mean-CVaR (about 114 s per solve at 300 stocks, and it barely reacts to views). Added local J7 time: about 15 s per source per week at 300 stocks (Robust about 3.6 s, Resampled about 11.7 s), so about 6 minutes for 24 sources. This is a live locked portfolio, not an evaluation re-run, so working rule 9 does not apply. |
| B1 (changed 2026-11-12) | prompt types | Four skills files (`skills_1` to `skills_4`) plus bare and placebo, six prompt types. `skills_5` stays disabled. ARCTIC is still sized for seven types, which leaves headroom. |
| D4 (confirmed 2026-11-12) | temperature | 0 for every model, with `seed` sent (the gateway passes it to the model server). Andalib: outputs are still not guaranteed identical, because requests are batched with other users'. Temperature 0 keeps the variation as small as the service allows, and G20 measures what remains. Both open-weight model cards recommend 1.0; the paper states why 0 was used. |
| G20 (new, 2026-11-12) | pilot repeat check | In the pilot week, 50 fixed stocks (the first 50 of the frozen universe by ticker) are rated a second time under all six prompt types by each gateway model, bypassing the response cache (50 stocks, 6 prompts, 2 models: 600 extra requests). Agreement per model and prompt is reported as the share of identical `likely` scores, the share within 1 point, and the rank correlation of `likely` across the 50 stocks. The exact-agreement share sets the canary threshold in G21 and is reported in the paper. |
| G21 (new, 2026-11-12; revised 2026-11-16) | weekly model checks and canaries | Free checks on every response and every weekend (what ARCTIC's gateway exposes, per Andalib): the response's `model` field (already saved as `model_version`), the `x-litellm-model-id` header (the gateway's deployment ID) and the `x-litellm-version` header (the gateway's software version) are saved in every record; before J4 the run saves `/v1/models` and `/v1/model/info` (with each model's `updated_at`). The gateway returns no model version or `system_fingerprint`. Canaries: before J4, each gateway model answers 10 fixed canary prompts (bare prompt, 10 fixed stocks, inputs from one frozen past snapshot saved in the repository), bypassing the cache. A model's week is flagged `model_change` when its `model` field or `x-litellm-model-id` differs from the pilot's, or when its canary answers match the pilot canary answers on fewer stocks than the lower 1% binomial bound given that model's G20 exact-agreement share (for example 5 or fewer of 10 when agreement is 0.9). A change in `x-litellm-version` or in a model's `updated_at` is logged as `gateway_changed` (information only, because it can come from changes such as pricing that do not affect answers) and does not flag the week by itself. Flagged weeks stay in the main results; a secondary analysis repeats the main comparisons without them. A flag never stops the run. |
| G22 (new, 2026-11-17) | measure everything | Every metric in Phases 5 to 8 is computed for every unit the study has: each source (per model and prompt), each steered portfolio (per model, prompt and strategy), each Q0, each confidence-only portfolio, HRP and equal weight, and pooled over models, prompts and strategies where a pooled estimator exists. Metrics include rank IC, residual IC, ICIR, IC hit rate, top-minus-bottom spread, view shift, steered tilt IC, annualized Sharpe net of the T-bill rate, probabilistic and deflated Sharpe, Sortino, Calmar, max drawdown, CVaR (95%), volatility, active return, information ratio, tracking error, certainty-equivalent return, turnover, costs, break-even cost, share of active weeks, factor alpha and loadings, and usage counts. Phase 9 writes them in long form to `results/eval/all_metrics.csv` (one row per unit and metric, with estimate, CI, p, `p_adj`, family, T and a `reason` when the value is NaN). The scope does not grow: no new models, prompts, strategies or stocks are added after the freeze. Every non-main test is in the Benjamini-Hochberg secondary family (H4 stays exploratory). Null results are reported, never dropped. |
| N6 (new, 2026-11-12) | logging, fallbacks and durable progress | (1) Each `run_weekend.py` invocation gets a `run_id` (timestamp plus short random suffix) that is attached to every log line, status file, record and study-log row it writes. (2) An append-only event timeline `data/logs/runs/<run_id>.jsonl` records every run start and end, job start, end, skip and failure (with duration and input hash), preflight result, retry, fallback, warning, flag and resume. (3) Every caught exception is logged with its full traceback (status files keep a `traceback` field; J4 model crashes, solver failures and provider errors keep theirs). (4) Python warnings and the `cvxpy`, `httpx` and `urllib3` loggers are routed into the JSON-lines logs. (5) A weekly fallback ledger `data/outputs/<week>/fallbacks.csv` lists every degradation with portfolio or source, stock if any, kind, reason and job: Q0 fallbacks, `no_rebalance`, Kronos fallback, `provider_error`, invalid outputs, solver fallback to the next solver, `optimal_inaccurate`, carried market caps, forward-filled prices, skipped and removed stocks, no-price liquidations, disabled gateway features, `model_change` flags. (6) Durable writes: atomic file writes `fsync` the temporary file before the rename and the folder after it; JSON-lines appends (LLM records, logs, the event timeline) are flushed and `fsync`ed after each line. (7) A free-disk check (at least 2 GB) runs before every job, not only at preflight. (8) Each run writes `data/outputs/<week>/run_summary.md`: every job with state and duration, counts of records, invalid outputs, fallbacks, warnings and errors, and the list of anything needing attention. (9) `portstudy audit --week W` verifies a week end to end: every expected file exists, hashes match the status files, every planned LLM request has a record, every portfolio has weights, and the fallback ledger matches the weight metadata. (10) Logs are never deleted during the study and are included in every backup. |
| N7 (new, 2026-11-16) | ARCTIC usage budget | Every gateway request's tokens and seconds are already logged (G15). Each week the study log adds `arctic_gpu_hours_est` (the week's requests times the per-request estimate in config, `arctic.gpu_seconds_per_request` per model: Gemma 40, MiniMax 90, replaced by measured values after the timing run), a running total, and a projection to the end of the study (total so far plus remaining weeks times the mean weekly use). `arctic.gpu_hours_allocated: 1000`. The weekend summary warns when the projection passes 75% of the allocation and prints a clear alert above 90%; the run never stops on its own. Carter checks the real usage in Elpis each month and records it in the study log notes (`manualwork.md`). |
| G5 (unchanged) | factor frequency | daily |
| G3, G3b, G4 (extended) | H3, H3b, H4 | H3 adds portfolio-level skills minus placebo and a pooled regression across models; H3b adds a confidence interaction and a calibration check; H4 adds strategy-vs-strategy active-return differences and regime interaction regressions. |

Everything else in PLAN2 section 0.3 is unchanged (A2 to A5, B2, C1 to C7, D3, E1 to
E6, K1, N1, N3, P1, S1 to S7, G1). A6 stays in the code for `views.ic_mode: trailing` only.

---

## 1. Repository review

### Existing architecture observations

- **Language and tooling:** Python 3.11 package `portstudy` (`src/` layout), setuptools build,
  `typer` CLI (`portstudy`), pydantic v2 frozen config models, pandas/numpy, cvxpy with OSQP,
  Clarabel and HiGHS, statsmodels, matplotlib, httpx plus tenacity. Ruff (line length 100), strict
  mypy on `src`, pytest with `slow` deselected by default. CI on GitHub Actions
  (`.github/workflows/ci.yml`) mirrored on GitLab (`.gitlab-ci.yml`).
- **Structure:** `config.py` (study config schema), `data/` (providers, snapshot, ledger,
  universe), `baseline/` (momentum, Kronos), `llm/` (prompts, providers, runner, validate, cost,
  health, cache), `views/` (IC, residualize, build), `risk/`, `optimize/` (Black-Litterman,
  strategies, problem cache), `sim/` (simulator, records), `pipeline/` (jobs J1 to J9, context,
  fallback, status, cli), `eval/` (data, rank_ic, tests, hypotheses, factors, report, shuffle),
  `backup.py`.
- **Data layer:** CSV files under `data/` (decision N3), weekly snapshots, append-only price ledger,
  `ICStore` (`ic_history.csv`), run status JSON per source, dated backups with optional rsync.
- **Frontend, database, auth, API server:** none. Secrets via `.env` (never committed).

### Relevant files and directories

| Path | Why it matters |
|---|---|
| `config/study.yaml`, `config/study_test.yaml`, `src/portstudy/config.py` | Decisions block, `eval.shuffle`, `eval.holm_families`, gateway fallback path, model `serving` literal |
| `config/models.yaml` | Small-model entries to remove; API prices are placeholders |
| `src/portstudy/eval/` | Every evaluation phase lands here |
| `src/portstudy/eval/shuffle.py`, `src/portstudy/pipeline/cli.py` (shuffle sub-app), `eval/report.py` | Shuffle removal |
| `scripts/arctic/`, `scripts/run_weekend.py`, `src/portstudy/llm/health.py`, `llm/providers/factory.py`, `pipeline/context.py`, `pipeline/jobs.py` (J4 health handling) | Slurm and fallback removal |
| `src/portstudy/baseline/kronos.py`, `scripts/kronos_smoke.py` | Kronos batching and GPU memory |
| `scripts/benchmark_pipeline.py` | Shuffle projection, `ARCTIC_REQUEST_CPU_HOURS`, Linux-only `resource` import |
| `src/portstudy/backup.py` | `backup.rsync_target` for the off-desktop copy |
| `src/portstudy/eval/factors.py` | ETF proxies, French loader, `daily_rf_from_snapshots` (reuse) |
| `tests/unit/test_v2_phase*`, `test_v3_phase*`, `tests/integration/test_v2_phase11_shuffle.py`, `test_v2_phase12_arctic.py`, `tests/e2e/*`, `test_phase16_checklist.py` | Tests to delete or update |
| `README.md`, `docs/*.md` | Rewritten on 2026-11-11 to the target design; their "pending PLAN4" notes are removed as each phase lands |
| `plans/`, `llm_research/` | Plans and build log; research notes (read-only for Devin unless a phase says otherwise) |

### Existing coding conventions

- Small pure functions per module; pydantic models are frozen; config values are never hard-coded
  (every setting lives in `study.yaml`, tagged `DECISION(<ID>)` in code).
- Evaluation functions take `EvalData` and return a tidy `pd.DataFrame` (one row per comparison
  with `n`, `mean_gap`, `nw_se`, `t`, `p`, `p_signflip`, `ci_low`, `ci_high`, `wins`, `ties`,
  `losses`, `mdg`, then a Holm column). Keep that shape and add columns rather than inventing a new
  result type.
- `eval/tests.py` holds statistics primitives (`newey_west_mean`, `sign_flip_permutation`, `holm`,
  `bootstrap_ci`, `paired_summary`, `min_detectable_gap`). New primitives go there.
- Reports are written by `eval/report.py` as CSV tables plus `report.md` and PNG figures.
- Logging via `logging_setup` with `extra_fields`; user-facing script output via `print`.
- Tests are named `test_<plan>_phase<N>_<topic>.py`; PLAN4 uses `test_v4_phase<N>_<topic>.py`.
- `plans/PROGRESS.md` has one entry per phase with "Choices made while unattended".

### Existing patterns to follow and reuse

- `paired_summary` and `_gap_rows` for every paired weekly gap.
- `EvalData` (weekly returns, daily values, statuses, `stock_week` with `likely:<source>` and
  `width:<source>`, view shift) as the single evaluation input.
- `daily_rf_from_snapshots` for the risk-free rate; `fama_french_factors` for a local French file.
- `views/build.py` `use_llm_view` flag and `has_llm_view` column for active-week detection.
- `ICStore` series for weekly IC by signal.
- `backup.rsync_target` for the off-desktop backup copy.
- `llm/health.py` `mark_sources_failed` for the D2 outage rule (keep; drop only the fallback path).

### Potential conflicts and how the plan handles them

| Conflict | Handling |
|---|---|
| `test_phase16_checklist.py` requires one commit per phase and mirrors every `DECISION` ID in `docs/open_questions.md` | Update the checklist test for `v4 phase N` subjects; add every new ID (L1, K2, G8 to G12) to `open_questions.md`; retired IDs move to a retired table and are removed from code tags |
| Removing `serving: slurm_fallback` changes the run registry hash used by J4 | Acceptable: no live week has run yet; note in PROGRESS that synthetic caches are invalidated |
| Removing shuffle changes `report.py` output files | Expected; update report tests |
| Residual IC needs stock characteristics; snapshots hold market cap and sector but no fundamentals | Use characteristics computable from data already stored (prices, market cap, sector) plus trailing factor loadings from French daily factors (G9). No new data provider |
| French data is released with a lag | Loader works on any local file; ETF proxies remain for monitoring; final report notes which source was used |
| OneDrive folders refuse WSL `chmod` and editable installs | Desktop setup clones the repo inside the WSL home (`~/pm`) instead of copying from OneDrive |
| New dependency for SPA, MCS and stationary bootstrap | Add only `arch` (Kevin Sheppard's package) to an optional `stats` extra; everything else uses numpy, scipy and statsmodels already present |

### Missing context

- Gateway behavior (rate limits, JSON-mode support, actual seconds and tokens per call) is
  unverified until Carter has a gateway key. Build for the documented behavior
  (`llm_research/09_arctic_process_and_model_facts.md`, `13_litellm_gateway_reference.md`) and
  leave the real checks to `manualwork.md` E.
- Whether WSL can write `backup.rsync_target` into a OneDrive-synced folder is unverified; Phase 3
  adds a check and a documented non-OneDrive default.
- Exact French Data Library file names and formats must be confirmed when the fetch script is
  first run (Phase 6 marks the URLs UNVERIFIED).
- The four main comparisons (G7) and the residual-IC controls (G9) are decided; Phase 4 writes
  them. Carter sets the view-sensitivity threshold and `frozen: true` (`manualwork.md` D4).

---

## 2. Overall summary

PLAN4 removes the ARCTIC compute layer and the shuffle test, adapts the pipeline to run end to end
on Carter's desktop under WSL2 with Kronos on a 6 GB GPU, and rebuilds the evaluation package so it
implements exactly the Evaluation section of the current proposal using zero-compute,
industry-standard tests. It also applies the fixes from the 2026-11-11 review: a neutral LLM view,
strategy settings that let views matter, a corrected stop rule, and the gateway, data and
simulator bug fixes. Config, decisions, tests, README and docs are brought in line, and every
remaining manual step is in `manualwork.md`.

## 3. Estimated total LOC

Approximately 9,700 to 12,800 lines changed in total:
- Added: about 4,700 to 6,100 (evaluation statistics, factor data, hypothesis rewrites, report,
  preregistration, desktop checks, tests, docs).
- Modified: about 1,700 to 2,600 (config, jobs, CLI, run_weekend, views, strategies, simulator,
  LLM runner and providers, data providers, report, README, docs).
- Deleted: about 2,300 to 2,900 (shuffle module and tests, Slurm scripts and tests, small-model
  config).

## 4. Assumptions

- No live week has run, so changing config hashes, removing serving modes and changing report
  outputs breaks no real data.
- The desktop runs WSL2 Ubuntu with Python 3.11 via `uv`, CUDA visible in WSL (confirmed
  2026-11-10).
- ARCTIC granted the startup allocation (2026-11-16) and the study uses only the LiteLLM gateway;
  nothing in the repo runs on ARCTIC compute nodes.
- The FRED T-bill series in each snapshot (`rates.csv`) is the risk-free source for weekly Sharpe;
  French RF is used inside factor regressions.
- `has_llm_view` is recorded per ticker in each steered portfolio's J7 weight output
  (`optimize/__init__.py`); Phase 8 verifies it can be read for every source-week before relying
  on it.
- About 23 live weeks, 100 stocks, 4 models (2 gateway, 2 API) and 6 prompt types (bare,
  placebo, four skills files), 5 strategies.
- Every API key exists at run time (working rule 11); none exists during the build.

## 5. Existing code to reuse

`paired_summary`, `_gap_rows`, `newey_west_mean`, `bootstrap_ci`, `holm`, `min_detectable_gap`,
`EvalData` and `load_eval_data`, `daily_rf_from_snapshots`, `fama_french_factors`,
`regress_portfolio`, `ICStore`, `rank_ic`, `residualize` (views), `momentum` (baseline),
`mark_sources_failed` / `unhealthy_models`, `run_weekend.py` resume logic, `backup.rsync_backup`,
`benchmark_pipeline.py` solver timing.

## 6. Files likely touched

```
config/                 study.yaml, study_test.yaml, models.yaml, preregistration.yaml (new)
src/portstudy/config.py
src/portstudy/eval/     tests.py, data.py, factors.py, hypotheses.py, report.py, rank_ic.py,
                        characteristics.py (new), metrics.py (new), multiple.py (new),
                        shuffle.py (deleted)
src/portstudy/baseline/kronos.py
src/portstudy/views/    build.py
src/portstudy/optimize/ __init__.py, black_litterman.py, mean_variance.py, strategies/
src/portstudy/sim/      simulator.py, records.py
src/portstudy/data/     checks.py, snapshot.py, ledger.py, providers/ (alpaca, yfinance, factory)
src/portstudy/llm/      health.py, runner.py, validate.py, cost.py, providers/ (litellm, factory)
src/portstudy/pipeline/ cli.py, context.py, jobs.py, status.py, study_log.py (new), events.py (new),
                        fallback_ledger.py (new)
src/portstudy/          logging_setup.py, io.py
src/portstudy/backup.py
scripts/                run_weekend.py, benchmark_pipeline.py, kronos_smoke.py,
                        smoke_real_services.py, fetch_french_factors.py (new),
                        check_live_ready.py (new), calibrate_view_sensitivity.py (new),
                        arctic/ (deleted)
tests/                  unit, integration, e2e (delete shuffle and Slurm tests, add v4 tests)
.github/workflows/ci.yml, .gitlab-ci.yml, pyproject.toml, uv.lock (new)
README.md, docs/ (remove pending notes, add new keys), plans/PROGRESS.md
```

## 7. AI execution order

| Phase | Title | Depends on |
|---|---|---|
| 1 | Remove the shuffle test | none |
| 2 | Remove the ARCTIC Slurm layer and small models | 1 |
| 3 | Desktop runtime: WSL2, Kronos batching, live-readiness checks | 2 |
| 3A | Week-to-week integrity: open-to-open returns, move rules, study log, keep-awake, checkpoints, logging and durable progress (N6), canaries and repeat check | 3 |
| 3B | Neutral LLM view (E7), confidence-only counterfactual (G18), strategy sensitivity calibration (S8) | 3A |
| 4 | Decisions, evaluation config and pre-registration file | 3B |
| 5 | Statistics primitives and portfolio metrics | 4 |
| 6 | Factor data and stock characteristics | 5 |
| 7 | Signal-level hypothesis tests (H1, H2 signal, H3 signal, model comparisons, H3b) | 5, 6 |
| 8 | Portfolio-level hypothesis tests (H2, H3, H4, H5, controls) and multiple testing | 5, 6, 7 |
| 9 | Report rewrite | 7, 8 |
| 10 | Documentation recheck and repository cleanup | 1 to 9 |
| 11 | End-to-end verification on synthetic data | 1 to 10 |

---

## Phase 1: Remove the shuffle test

### Goal

Delete every trace of the shuffled-ratings test (decision G6 retired) so no code path, config key,
CLI command, report section or doc can run or describe it.

### Step 0: verify the repository migration (before any other change)

Carter moved the repository by hand on 2026-11-16 from the OneDrive folder
(`C:\Users\carte\OneDrive\Desktop\projects\portfoliomanagement`, now renamed
`portfoliomanagement_olddd`) to `C:\dev\portfoliomanagement` (`manualwork.md` 0.1). A first check
on 2026-11-17 found `git fsck` clean, HEAD `44949aa`, both remotes and the gitignored correspondence
present; this step still runs in full. The move means OneDrive no longer syncs `.git`, `.venv` and
caches. Devin works only in `C:\dev\portfoliomanagement`; weekly runs use the WSL clone (`~/pm`).
Before touching any code, Devin checks that nothing was lost and records the result in
`plans/PROGRESS.md` under "v4 phase 1, step 0":

- `git -C C:\dev\portfoliomanagement fsck --no-dangling` reports no errors; `git log -1` is
  `44949aa` (or the commit Carter names); `git remote -v` lists `origin` (GitHub) and `gitlab`.
- `git status --porcelain` shows exactly the uncommitted work listed in working rule 4 (the
  reorganization into `plans/` and `llm_research/`, deleted `docs/proposals/`, rewritten docs,
  `manualwork.md`, `scripts/kronos_smoke.py`, the `.gitignore` change) and nothing unexpected.
- Gitignored local files that git cannot restore exist in the new folder:
  `llm_research/11_correspondence.md`, anything under `data/` that existed in the old folder, and any
  `.env` file. Compare file counts and SHA-256 hashes of every non-ignored and ignored file (excluding
  `.venv`, `__pycache__`, `.mypy_cache`, `.ruff_cache`, `.pytest_cache`) between the old and new
  folders; list every difference.
- Recreate the virtual environment in the new folder (the old `.venv` is not copied) and run the
  default test suite once on the unchanged code; record the pass count as the baseline.
- If any check fails, stop and write the problem in PROGRESS; do not continue to the shuffle removal.
  Carter deletes the old folder (`portfoliomanagement_olddd`) only after this step passes. The
  empty `vendor/` folder was deleted on 2026-11-17 (Phase 3 creates `vendor/kronos`).

### High level summary

- Delete `src/portstudy/eval/shuffle.py` and the `portstudy shuffle` sub-app (`plan`, `run`,
  `run-local`, `collect`) in `pipeline/cli.py`; remove `--skip-shuffle` from `portstudy eval`.
- Remove `ShuffleConfig`, `eval.shuffle`, its validator and the `H5_shuffle` Holm family from
  `config.py`, `study.yaml` and `study_test.yaml`.
- Remove the shuffle calls, `SHUFFLE_EXCLUDED_LINE`, `_fig_shuffle` and the H5 shuffle sections from
  `eval/report.py`. Leave H5 as the existing factor table until Phase 8 rebuilds it.
- In `scripts/benchmark_pipeline.py`, remove the shuffle projection columns, `--replay-seconds`,
  `--project-sources` shuffle math and `ARCTIC_REQUEST_CPU_HOURS`. Keep per-strategy solve times,
  J7 seconds per week (at the run's sources and a projected source count) and peak memory.
- Update comments in `optimize/mean_variance.py` and `optimize/problem_cache.py` that mention
  shuffles (the cache still serves resampled draws and many sources per week).
- Delete `tests/integration/test_v2_phase11_shuffle.py` and `tests/unit/test_v3_phase2_shuffle.py`;
  remove shuffle assertions from `test_phase11_report.py`, `test_v2_phase9_scale.py`,
  `test_v3_phase1_config.py`, `test_review_fixes.py` and the e2e tests.
- Commit the 2026-11-11 reorganization in this phase (working rule 4).

### Implementation areas

Evaluation, CLI, config schema, benchmark script, tests.

### Estimated LOC

Approximately 900 to 1,200 deleted, 80 to 150 modified.

### Files expected to change

`src/portstudy/eval/shuffle.py` (deleted), `eval/report.py`, `pipeline/cli.py`, `config.py`,
`config/study.yaml`, `config/study_test.yaml`, `scripts/benchmark_pipeline.py`,
`optimize/mean_variance.py`, `optimize/problem_cache.py`, the tests listed above, and the
reorganized files from working rule 4 (commit only).

### Dependencies

None.

### Can AI complete independently?

Yes.

### Risk

Low. The shuffle is isolated; the only coupling is the report and CLI.

### Complexity

2/5

### AI completion criteria

- `grep -rni shuffle src scripts config tests` returns nothing except the word in unrelated
  synthetic-data code (`rng.shuffle(words)` in `data/providers/synthetic.py`) and new negative
  tests that assert the shuffle CLI and config key are gone.
- `portstudy eval --synthetic` runs without the shuffle and without `--skip-shuffle`.
- `python scripts/benchmark_pipeline.py --n-stocks 100 --sources 3 --weeks 24` prints solve times,
  J7 seconds and peak memory with no shuffle line.
- Lint, format, mypy and default pytest pass; PROGRESS entry written; committed and pushed.

---

## Phase 2: Remove the ARCTIC Slurm layer and small models

### Goal

Make the code match L1 and D2: ARCTIC is only an HTTP endpoint, a gateway outage just fails that
model's week, and there is no Slurm, vLLM or small-model path.

### High level summary

- Delete `scripts/arctic/` (`common.sh`, `submit_weekend.sh`, `kronos.sbatch`, `llm.sbatch`,
  `finish.sbatch`, `llm_fallback.sbatch`, `shuffle_array.sbatch`, `arctic.env.example`).
- `config.py` and `models.yaml`: remove `serving: slurm_fallback`, `fallback_eligible`, the
  slurm-fallback validator, `gateway.fallback_endpoint_file` and the `ARCTIC_FALLBACK_API_KEY`
  lookup (`llm/providers/factory.py`, `scripts/check_secrets.py`, `scripts/smoke_real_services.py`);
  remove the Gemma 4 E4B, Phi-4 Mini
  and Llama 3.2 3B entries; remove the model `role` field (only the `fallback` value used it).
- `llm/providers/factory.py`: drop the fallback endpoint file lookup. `pipeline/context.py`: drop
  the `slurm_fallback` registry rewrite. `pipeline/jobs.py` J4: keep the health check and
  `mark_sources_failed` (D2), drop the "runs on the fallback endpoint" branch.
- `llm/health.py`: remove the Slurm walltime parser and any walltime check.
- `scripts/run_weekend.py` and `pipeline/cli.py`: remove `--gateway-fallback`, `--check-walltime`,
  the finish-job single-pass mode (`--poll-once`, used only by `finish.sbatch`) and every
  "login mode" or Slurm wording in help text and the module docstring. Keep `--week`, `--dry-run`,
  `--synthetic`, `--allow-over-budget`, `--config`, `--now`, `--poll-seconds`, `--from`, `--until`.
- `config/study.yaml` decisions block: update D1, D2, N2 to section 0.3 values; delete the W1, M1
  and G6 entries (they move to the retired table in `docs/open_questions.md` in Phase 4).
- Delete `tests/integration/test_v2_phase12_arctic.py` and `tests/unit/test_v3_phase3_arctic.py`;
  update `test_v2_phase4_litellm.py`, `test_v2_phase1_config_identity.py`, `test_review_fixes.py`
  and the e2e rehearsal (`tests/e2e/conftest.py::INJECTIONS` week 12 used `--gateway-fallback`:
  change it to a gateway health failure that marks the model's sources failed and the run
  continues with Q0 for those sources).
- Update `.github/workflows/ci.yml` and `.gitlab-ci.yml` if they reference `scripts/arctic` or
  shellcheck on sbatch files.

### Implementation areas

Config schema, model registry, providers, pipeline context and J4, CLI, run script, CI, tests.

### Estimated LOC

Approximately 900 to 1,100 deleted, 150 to 250 modified.

### Files expected to change

`scripts/arctic/` (deleted), `config.py`, `config/models.yaml`, `config/study.yaml`,
`llm/providers/factory.py`, `llm/health.py`, `pipeline/context.py`, `pipeline/jobs.py`,
`pipeline/cli.py`, `scripts/run_weekend.py`, CI files, tests listed above.

### Dependencies

Phase 1 (both touch `cli.py` and `config.py`; doing them in order avoids conflicts).

### Can AI complete independently?

Yes.

### Risk

Medium. J4's health-check path and the e2e failure injections change; a mistake could make a
failed model abort the whole run instead of falling back to Q0. The updated e2e test must prove
the run continues.

### Complexity

3/5

### AI completion criteria

- `grep -rniE "slurm|sbatch|walltime|gateway[-_]fallback|fallback_endpoint|arctic_fallback|login mode|e4b|phi-4|llama"`
  over `src scripts config tests .github .gitlab-ci.yml` returns nothing (docs are Phase 10).
- A unit test: a failed gateway health check for one model marks only that model's sources failed
  for the week, J6 serves their strategies' Q0, and the other model's sources run normally.
- `python scripts/run_weekend.py --synthetic --dry-run` lists J1 to J9 with no ARCTIC wording.
- Lint, format, mypy, pytest pass; PROGRESS entry; committed and pushed.

---

## Phase 3: Desktop runtime (WSL2, Kronos batching, live-readiness checks)

### Goal

Make the weekly run reliable on Carter's desktop: Kronos for the universe (100 stocks; 300 also fits) in 6 GB of GPU memory,
reproducible installs, an off-desktop backup, and a single preflight that refuses a live run while
anything is still a placeholder.

### High level summary

- **Kronos batching (K2).** Add `baseline.kronos.batch_size` (default 100) and `device`
  (default `cuda`, `cpu` allowed). `KronosForecaster.forecast` processes tickers in chunks of
  `batch_size`, calls `torch.cuda.empty_cache()` between chunks, and logs per-chunk seconds and
  `torch.cuda.max_memory_allocated()`. Seed every (sample, chunk) separately, with a seed derived
  from `(kronos.seed, sample_index, chunk_index)`: today one seed per sample would be reused by
  every chunk, so stock j of each chunk would get the same sampling noise. Chunking changes the
  sampled paths of the real model, so do not claim equality with the unbatched result; the tests are
  (a) a fake predictor returns the same values for any chunk size, and (b) two runs with the same
  seed and `batch_size` give identical outputs on CPU. `batch_size` is part of the frozen config.
  Keep the existing `predict_paths` / repeated-sample adapter.
- **`scripts/kronos_smoke.py`** (already in the tree, uncommitted): add `--batch-size`, print
  per-chunk timing and peak memory, and make it exit cleanly with a message when torch or CUDA is
  missing. Commit it. Add a `slow` test that runs it only when torch and CUDA are present.
- **Kronos failure fallback (K3).** Add `baseline.on_kronos_failure: hold | momentum_only` with
  default `momentum_only` (decided 2026-11-12). With `momentum_only`, a failed J2 uses the momentum
  rank alone, sets `k_sd` to the median so K1 gives a neutral spread, writes
  `kronos_fallback: true` in the baseline meta and the study log, and the week runs normally for
  every portfolio. Covariance failures still stop J2 (`hold` behavior), because no strategy can run
  without it. Test: a Kronos forecaster that raises gives a complete J2 with the flag set.
- **Dependencies.** Add `einops` and `safetensors` to the `ml` extra (Kronos needs them; they had to
  be installed by hand on 2026-11-10). Add an optional `stats` extra with `arch` (used in Phases 5
  and 8). Generate and commit `uv.lock` (`uv lock`) so the freeze installs exact versions; document
  `uv pip sync` / `uv sync` from the lock in the setup doc.
- **Cross-platform benchmark.** `scripts/benchmark_pipeline.py` imports `resource` (Linux only).
  Guard it so peak memory falls back to `psutil` if present or prints "n/a" on Windows; the
  documented path is still WSL2.
- **Backups off the desktop.** Keep `backup.rsync_target`. Add a writability check at J9 start
  (create and delete a probe file) that prints a clear warning when the target is not writable
  (for example a OneDrive folder that refuses WSL writes). Document a default target such as
  `/mnt/c/Users/carte/portstudy_backups` plus a manual copy to cloud storage.
- **Gateway timing and token logging (G15, G17).** Extend `scripts/smoke_real_services.py` with
  `--time-gateway N`, which sends N real rating requests per gateway model at the configured
  concurrency and prints, per model: mean, p50 and p95 seconds per call (both client-side and the
  gateway's `x-litellm-response-duration-ms`), the `x-ratelimit-*` limits the gateway reports for the
  key, invalid rate, the share of
  responses with `finish_reason: length`, and mean and max prompt, completion and reasoning tokens
  (read from the response `usage`, including reasoning tokens when the gateway reports them).
  It also writes a small CSV of the raw measurements to `data/outputs/gateway_timing.csv` so the
  numbers in the ARCTIC request can be traced to a run. The same fields are recorded for every
  live request in the LLM call log. Offline tests use the stub transport.
- **Gateway output handling (G15).** Per-model settings already work (`_settings_for` in
  `llm/runner.py` reads `llm.settings.<model>`, then the model's `settings` in
  `config/models.yaml`), so:
  - Set `max_output_tokens` to 300 for Gemma and 2,000 for MiniMax in config, and make the cost
    estimate in `llm/cost.py` use each model's own limit.
  - Add a per-model `request_timeout_seconds` setting (default 180, MiniMax 600 until measured) and
    pass it to `LiteLLMProvider`, replacing the fixed `timeout=180`.
  - Save `reasoning_text` (and its length) in every record and in `llm_scores.csv` as
    `reasoning_chars`; today it is dropped.
  - Treat empty content with `finish_reason: length` as `invalid_truncated` with a log line that
    names the model and its limit.
  - No strip rule for Gemma's `<|channel>thought ... <channel|>` marker is needed, because
    `parse_response` takes the first JSON object in the text. Add tests that pin this: Gemma output
    with an empty marker before the JSON; MiniMax output with `reasoning_content` and blank lines
    before the JSON; empty content with `finish_reason: length`.
- **Gateway error classes (N5).** Classify gateway errors before retrying: HTTP 429 (rate limit or
  a deployment cooldown) waits and retries with backoff, using `x-ratelimit-reset-*` when present;
  a budget error (text `ExceededBudget`, `ExceededTokenBudget` or type `budget_exceeded`, whatever
  the status code) is never retried, stops new requests to that model for the week, and is written to
  the fallback ledger and the weekend summary as an ARCTIC problem; 5xx and timeouts retry as today;
  other 4xx are not retried. Tests for each class with the stub transport.
- **Gateway request failures (N5).** In `WeekRunner.run_open_weight`, a request that is not
  requeued (retries used up, or not retryable) is written as an invalid record with
  `fallback_flag: provider_error` and the error text, so it counts toward the 15% rule and the
  source is no longer left incomplete. `J4_open_weight_infer` raises (state failed) while any
  planned request of a non-failed model has no record, so rerunning `run_weekend.py` resumes J4
  instead of skipping it. Update the J4 and J6 hints in `run_weekend.py`. Tests: one request that
  always returns HTTP 500 gives a complete source with one `provider_error` record and a normal
  J6; a request with no record leaves J4 failed and a rerun completes it.
- **Frozen gateway features (G15).** In `LiteLLMProvider`, match an unsupported-parameter 400 more
  narrowly (the error must name the parameter as unsupported or unknown, not just contain the
  word), and log the response body. Add `llm.gateway.features_frozen` (default false): when true,
  a rejected `seed` or `response_format` raises a provider error instead of silently disabling the
  feature. Record the features used per source-week in the study log (Phase 3A).
- **Seed and JSON mode.** Andalib confirmed on 2026-11-12 that the gateway passes `seed` to the
  model server: change the `send_seed` comment in `config/study.yaml` from `UNVERIFIED` to
  "passed through by the gateway (ARCTIC, 2026-11-12); outputs not guaranteed identical". The
  ARCTIC page documents `response_format` (`json_object` and `json_schema`), so `json_mode` stays
  `UNVERIFIED` only until the timing run shows it accepted. `--time-gateway` prints which features
  the gateway accepted and `check_live_ready.py` lists any still marked `UNVERIFIED`.
- **Response identity fields (G21).** The response's `model` field is already saved in every
  record as `model_version` (`llm/providers/litellm.py`), with a warning on mismatch; reuse it
  (do not add a second column) and carry it into `llm_scores.csv`. Also save the response headers
  `x-litellm-model-id` and `x-litellm-version` in every record and in `llm_scores.csv`
  (`deployment_id`, `gateway_version`), plus `x-litellm-model-group`,
  `x-litellm-attempted-fallbacks`, `x-litellm-attempted-retries`, `x-litellm-response-duration-ms`,
  `x-litellm-response-cost`, `x-litellm-call-id` and every `x-ratelimit-*` header when present
  (header meanings in `llm_research/13_litellm_gateway_reference.md`). A response whose
  `x-litellm-attempted-fallbacks` is above 0, or whose `x-litellm-model-group` differs from the
  requested model, is an invalid answer with reason `gateway_fallback` (another deployment or model
  may have answered) and counts toward the 15% rule. Also save `usage` in full, including MiniMax's
  `reasoning_tokens` (Andalib: MiniMax returns its thinking in `reasoning_content` and the count in
  `usage`). The gateway returns no `system_fingerprint`, so do not build for it. G21's checks
  compare `model_version` and `deployment_id` with the pilot's. Add a `bypass_cache` flag to the runner for one-off requests (used by the
  G20 repeat check and the G21 canaries) so they are never answered from, or written to, the
  response cache.
- **Price feed (G16).** Add `data.alpaca_feed` (default `sip`) to the config schema and pass it to
  `AlpacaPriceProvider` and `AlpacaNewsProvider` where the feed applies. In
  `scripts/smoke_real_services.py` the Alpaca check requests a few days of daily bars for one
  ticker with the configured feed, prints the feed, bar count and the volume of the most recent
  bar, and fails clearly (with the provider's error) if the feed is refused. If SIP is refused, Carter
  sets the existing `data.price_provider: yfinance` (no new key) and the choice is written to the
  study log. Tests use the stub transport.
- **Live-readiness preflight.** New `scripts/check_live_ready.py` (also called by
  `run_weekend.py` for any non-synthetic run, with `--skip-preflight` to override) refuses to start
  while any of these hold, listing every failure: an enabled prompt file name contains
  `PLACEHOLDER`; `baseline.kronos.repo_commit` is `UNPINNED`; `baseline.forecaster` is `stub`;
  any enabled model has `provider_model: TBD`; API prices are the placeholder values;
  `timing.live_end_date` or `timing.pilot_week` is null; `ARCTIC_LITELLM_API_KEY` or a required
  provider key is missing from the environment; `vendor/kronos` is missing or not at the pinned
  commit; less than 5 GB free disk; GPU free memory below what `batch_size` needs (warning only);
  the gateway URL does not answer (VPN off); an enabled API model has no `snapshot_id` (D5); the
  live data folder holds a week before `timing.pilot_week` (A7); the universe file lacks a name or
  sector for a ticker (C8); `llm.gateway.features_frozen` is false after the pilot week;
  `simulator` or `strategies` values differ from the ones recorded by Phase 3B's calibration
  without a note. It also prints the count of `TBD`, `UNVERIFIED` and `UNPINNED` markers in code
  and config.
- **Config hash scope.** Change the job input hash to cover only the config keys each job reads
  (so setting `timing.live_end_date` mid-study does not mark J1 to J4 stale), or, if that is too
  invasive, add a frozen-config check that refuses a live run when `study.yaml` differs from the
  copy saved at the pilot. Record which option was chosen.
- **Gateway model mismatch.** `LiteLLMProvider`: a response naming a different model than
  requested becomes an invalid output (currently a warning).
- **Prompt name and sector from the frozen stock list (C8).** `data/providers/factory.py` builds
  `YFinanceReferenceProvider` without `members`, so prompts get yfinance's name and sector each
  week (non-GICS labels, sometimes missing). Make `make_snapshot` take `name` and `sector` for
  every ticker from the frozen universe file and use the reference provider only for market cap
  and the earnings date. A missing name or sector in the universe file is a `halt` issue.
  Test: a reference provider returning a different or missing sector does not change the prompt.
- **Market-cap fallback (C8).** When the reference call fails or returns no cap for a ticker, reuse
  that ticker's value from the most recent earlier snapshot and add an `info` issue
  (`market_cap_carried`). If more than 10% of caps are carried or missing, add a `warning` that is
  printed in the weekend summary. J2 no longer fails when a single week's reference call is down.
- **Cutoff-day bar check (review C11, UNVERIFIED).** `AlpacaPriceProvider._bars` sends `end` as a
  bare date, which may exclude the cutoff day's bar. Send `end` as the cutoff day 23:59:59 in
  America/New_York as an RFC 3339 timestamp, and make the Alpaca check in
  `scripts/smoke_real_services.py` assert that the last returned bar's date equals the last
  trading day requested.
- **API model versions (D5).** `check_live_ready.py` fails while any enabled API model has
  `snapshot_id: null`.
- **Small fixes from the review.** `current_week` rolls forward at the cutoff day's close, also
  when the cutoff is a Thursday before a Friday holiday (test: Thursday 2026-12-24 after 13:00 ET
  returns 2026-W52). `solve_mean_variance` records `optimal_inaccurate` in the weight meta
  (`solver_status`). HRP reports the configured linkage name.

### Implementation areas

Baseline (Kronos), config schema, scripts, backup, LLM provider, pipeline hashing, pyproject, tests.

### Estimated LOC

Approximately 1,000 to 1,400 added, 280 to 420 modified.

### Files expected to change

`baseline/kronos.py`, `config.py`, `config/study.yaml`, `config/study_test.yaml`,
`scripts/kronos_smoke.py`, `scripts/benchmark_pipeline.py`, `scripts/smoke_real_services.py`,
`scripts/check_live_ready.py` (new), `scripts/run_weekend.py`, `backup.py`,
`llm/providers/litellm.py`, `llm/runner.py`, `llm/validate.py`, `llm/cost.py`,
`data/providers/alpaca.py`, `data/providers/factory.py`, `data/snapshot.py`,
`optimize/mean_variance.py`, `optimize/strategies/hrp.py`, `pipeline/jobs.py` (J4 state,
`current_week`), `config/models.yaml`,
`pipeline/jobs.py` or `hashing.py`, `pyproject.toml`, `uv.lock` (new), tests.

### Dependencies

Phase 2.

### Can AI complete independently?

Yes for the code. The real Kronos timing at 100 stocks with batching, the real gateway timing
(needs the ARCTIC key and the VPN), the SIP check (needs Alpaca keys) and the backup writability
check need Carter's desktop and keys (`manualwork.md`).

### Risk

Medium. Kronos chunking must be deterministic for a fixed seed and chunk size; the J4 state change
must not break resume; the preflight must not block synthetic runs or CI.

### Complexity

3/5

### AI completion criteria

- Unit test: batched and unbatched Kronos forecasts are identical for a fake predictor and fixed
  seed; chunk boundaries at 1, `batch_size`, and `n` not divisible by `batch_size`.
- `check_live_ready.py` on the current repo fails with a list naming the placeholders, `UNPINNED`,
  `stub`, null dates and missing keys; it passes against a fixture config with everything set.
- `run_weekend.py --synthetic` does not run the preflight; a non-synthetic run without
  `--skip-preflight` stops at the preflight in tests.
- Gemma marker, MiniMax reasoning and `finish_reason: length` tests pass; per-model
  `max_output_tokens` and `request_timeout_seconds` are read by the provider and by `llm/cost.py`;
  `reasoning_chars` is in the records; `--time-gateway` offline test writes `gateway_timing.csv`
  with token and `finish_reason` columns.
- N5 tests pass (a permanently failing request becomes a `provider_error` record; J4 is not done
  while a request has no record). Frozen-features test: with `features_frozen: true` a rejected
  `seed` raises; an unrelated 400 whose body contains the word `seed` does not disable it.
- C8 tests pass (prompt name and sector come from the universe file; a failed reference call
  carries last week's market caps). The cutoff-day bar assertion exists in the smoke script.
- `check_live_ready.py` also fails on a missing `snapshot_id` and on a pre-pilot week in the data
  folder.
- `data.alpaca_feed` defaults to `sip`; no hard-coded `iex` remains in `src/`.
- `uv.lock` committed; `pip install -e ".[ml,solvers]"` pulls `einops` and `safetensors`.
- Lint, format, mypy, pytest pass; PROGRESS entry; committed and pushed.

---

## Phase 3A: Week-to-week integrity and data tracking

### Goal

Make the weekly run survive the real desktop (sleep, Windows Update, VPN drops, crashes mid-job)
without losing progress, record every week in one study log, measure weekly returns on the right
window, and apply the new large-move and crash-removal rules (decisions N4 and U1).

### High level summary

- **Open-to-open weekly returns (N4).** The simulator marks values at each daily close, and
  `eval/data.py` currently computes weekly returns from Friday close to Friday close, so the
  weekend gap (held with last week's weights) is credited to the new week. Record the portfolio
  value at each Monday open before the rebalance (`value_open` on the trade-day record; the
  rebalance already computes it) and compute weekly returns from one Monday open to the next, the
  same window rank IC uses. A `no_rebalance` week still records its Monday-open value. The last
  week waits for the next Monday open before it is scored. Daily close values stay for daily
  metrics and factor regressions.
- **Large moves and crash removal (U1).** Rewrite the `extreme_move` check in `data/checks.py`:
  look only at days since the previous cutoff (the first snapshot looks back one week), use adjusted
  prices for real moves, flag 5-day moves at or beyond plus or minus `data.checks.large_move_flag`
  (0.10) without skipping the stock, and permanently remove a stock whose single-day or 5-day
  adjusted return is at or below `data.checks.crash_removal` (-0.40). Removals are appended to
  `data/universe_removals.csv` (week, ticker, date, return, reason), every later snapshot excludes
  the ticker, the next rebalance sets its weight to 0 (sold at the Monday open), and nothing replaces
  it. A raw jump not present in the adjusted series becomes `unexplained_split` (data error, skip the
  stock for that week only). Keep `stopped_trading` as is.
- **ARCTIC usage tracking (N7).** Add `arctic.gpu_hours_allocated` (1000) and
  `arctic.gpu_seconds_per_request` (per model; Gemma 40, MiniMax 90) to the config schema. After J4,
  compute the week's estimated GPU-hours from the logged request counts (and, once measured values
  replace the estimates, from them), the running total and the end-of-study projection; write them to
  the study log and the weekend summary; warn above 75% projected and alert above 90%. Test: a
  synthetic study whose projection crosses 75% prints the warning and keeps running.
- **Study log.** New `pipeline/study_log.py` upserts one row per week into
  `data/outputs/study_log.csv`: start and finish times, each job's state and duration, git commit,
  config hash, preregistration hash, universe size, removals, large-move flags, per-source fallback
  and invalid rates, API cost, Kronos peak GPU memory, backup path and verification result, the
  price feed and provider actually used, the gateway model list returned by `/v1/models`, and
  per gateway model the mean seconds, mean completion tokens and `finish_reason: length` share
  (so a gateway model or setting change shows up in the log), and per gateway model the canary
  match count, the `model_version` value seen and the `model_change` flag (G21).
  `run_weekend.py` writes it at the end of every run (and on failure, with the failing job), and
  `portstudy status` prints the last few rows.
- **Weekly model checks and canaries (G21).** New `llm/canary.py` and a pre-J4 step in
  `run_weekend.py` (also `portstudy canary --week W`):
  - Save `/v1/models` and `/v1/model/info` (every field returned, including each model's
    `updated_at`) to `runs/<week>/gateway_models.json`. If `/v1/model/info` is refused for a user
    key, log that once and continue.
  - Send 10 canary prompts per gateway model with `bypass_cache`. The canary set (10 tickers and
    their frozen inputs from one past snapshot) is stored in `config/canary/` and frozen with the
    config; Devin builds it from the synthetic world for tests, and Carter replaces it with a real
    past week before the freeze (`manualwork.md`).
  - Compare with the pilot's canary answers saved in `data/outputs/canary_baseline.csv`: the
    number of identical `likely` scores and any change in `model_version` or `deployment_id`
    (these flag), and any change in `gateway_version` or a model's `updated_at` (logged as
    `gateway_changed`, information only).
  - Write the result and the `model_change` flag per model to the study log and to
    `data/outputs/model_checks.csv`. A flag prints a warning in the weekend summary and never stops
    the run. Canary requests are not part of any portfolio and their records are kept separately.
  - Tests with the stub provider: identical answers give no flag; a changed `model` field or
    `x-litellm-model-id` header flags; a changed `x-litellm-version` or `updated_at` logs
    `gateway_changed` without flagging;
    answers below the binomial bound flag; the canary never reaches the cache.
- **Pilot repeat check (G20).** New `portstudy repeat-check --week W` (run once in the pilot after
  J4): re-rates the first 50 universe tickers under all six prompt types for each gateway model with
  `bypass_cache`, then writes `data/outputs/pilot_repeatability.csv` (per model and prompt: share of
  identical `likely`, share within 1, rank correlation, invalid count) and stores each model's
  exact-agreement share for G21. It also writes the pilot canary answers to
  `data/outputs/canary_baseline.csv`. Test with the stub provider (identical answers give
  agreement 1.0; injected noise lowers it).
- **Keep the desktop awake.** When `run_weekend.py` runs under WSL it starts a small
  `powershell.exe` helper that calls `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`
  and stops it on exit (normal, error or Ctrl+C). `--no-keep-awake` disables it. Tag UNVERIFIED until
  Carter confirms it on the desktop. Windows Update is handled by pausing updates (manual).
- **Stop on a rebalance day (G1b).** In `PortfolioSim.rebalance`, when a ticker had a pending
  stop-loss or take-profit and the position is still held after the rebalance, set its `avg_cost`
  to the rebalance open price. The trade record keeps `stop_trigger`. Without this the stop fires
  again at Monday's close and the stock is sold on Tuesday (review C1, reproduced by running the
  simulator). Test: the review's two-stock case (one stock rising 3% a day) shows no Tuesday sale
  and no idle cash; a stock that keeps rising another 15% from the Monday open still triggers.
- **No-price liquidation (U2).** A held ticker with no positive price for
  `simulator.no_price_liquidation_days` consecutive ledger days (default 5) is sold at its last
  close with the usual cost, with trade reason `no_price_liquidation`, and listed in the report.
  The U1 removal uses the same path when the removed stock has no Monday open. Test: a ticker that
  disappears from the ledger is liquidated on the fifth day and its cash is used at the next
  rebalance.
- **Fixed view scale (A8).** Add `views.ic_mode: fixed | trailing` (default `fixed`) and
  `views.fixed_ic: 0.02`. In `portfolio_ics` (`optimize/__init__.py`), `fixed` returns
  `(fixed_ic, fixed_ic)` for every week and source; `trailing` keeps today's `trailing_ic` path.
  J8 still computes and stores every weekly rank IC (`update_ic_history`). Record `ic_mode` and the
  IC values used in each weight meta. Update `config/study_test.yaml` so existing trailing-IC tests
  set `ic_mode: trailing` explicitly. Tests: in fixed mode a source with negative trailing IC still
  gets its LLM view and Q0 always keeps its baseline view.
- **Data-folder hygiene (A7).** Add `views.ic_history_start` (`live_start_week` default, or
  `first_locked_week`). With `live_start_week`: `trailing_ic` and the evaluation ignore every week
  before the first live week; `replay_weeks` starts every portfolio from cash at the first live
  week's Monday open; pilot weights stay on disk for the pilot review but are never replayed into
  live results. Document `PORTSTUDY_DATA_DIR` for dry runs. The preflight check for
  pre-pilot weeks is in Phase 3. Add `portstudy ledger rebuild`, which rebuilds the price ledger
  from empty for the configured feed (used once at the freeze).
- **End of study.** Add `portstudy closeout`: after the last live week's window has closed (the
  following Monday open is in the ledger) it runs the final replay and scoring, writes the last
  week's open-to-open return, and marks the study complete in the study log. It refuses to run
  earlier and prints the date it can run.
- **Checkpoints in long jobs.** J2 writes Kronos results per chunk (`baseline` chunk files) and
  resumes from the last finished chunk. Verify with tests that J4 resumes from its per-call cache
  after being killed mid-run, that J7 keeps finished portfolios and only solves the rest, and that
  J5 polling resumes. Partial outputs are never treated as complete (status files plus output hashes).
- **Network outage tolerance.** On connection errors to the gateway or a provider (for example the
  VPN dropping), J4 and J5 pause and retry with backoff for up to
  `llm.network_outage_tolerance_minutes` (default 60) before counting the calls as failed, and print
  "check the GSU VPN" while waiting. Retries never extend past the lock margin. Connection errors
  during the tolerance window do not use up a request's retry count (N5), so a VPN drop cannot turn
  hundreds of requests into `provider_error` records.
- **Progress and deadline.** J2, J4 and J7 write `data/outputs/<week>/progress.json` (done, total,
  elapsed, projected finish). `run_weekend.py` prints progress and warns when the projected finish of
  the remaining jobs passes the Monday 08:30 ET lock margin.
- **Interrupted runs.** On start, `run_weekend.py` detects jobs left in `running` state (process
  died), says so, records the interruption in the event timeline, and resumes them.
- **Logging, fallback ledger and durable progress (N6).** Build all ten parts of N6:
  - `run_id`: created in `run_weekend.py` (and in `portstudy run-job` when run alone), passed in
    `RunContext`, added by `JsonLinesHandler` to every line, and saved in each status file, LLM
    record, weight meta and study-log row.
  - Event timeline: new `pipeline/events.py` with one `emit(kind, **fields)` function used by the
    run script, `run_job`, retries, fallbacks, flags and resume; one JSON line per event.
  - Tracebacks: `run_job` stores `traceback.format_exc()` in the status file next to `error`; the
    J4 model-crash handler, solver-fallback loops (`mean_variance.py`, `strategies/common.py`),
    provider errors and the health check log the traceback with `log.exception` or an explicit
    `traceback` field. No caught exception may be dropped without a log line (add a test that greps
    `src/` for `except` blocks with no logging call and fails on new ones).
  - Warnings and library logs: `logging.captureWarnings(True)` and handlers on `cvxpy`, `httpx`,
    `urllib3` in `get_logger`'s setup, at WARNING and above, into `data/logs/libraries/<date>.jsonl`.
  - Fallback ledger: new `pipeline/fallback_ledger.py` with `record(week, kind, reason, job, ...)`;
    every code path listed in N6 (5) calls it. J6 and J7 also cross-check the ledger against the
    weight metadata and log any mismatch.
  - Durable writes: `io._atomic_replace` fsyncs the temp file and the parent folder; every JSON-lines
    writer (`JsonLinesHandler`, the LLM record store, the event timeline, the fallback ledger) flushes
    and fsyncs after each line. Measure the cost on the synthetic weekend; it should be seconds.
  - Disk check: `run_job` refuses to start a job with less than 2 GB free and records why.
  - Run summary: `run_weekend.py` writes `run_summary.md` at the end of every run, including failed
    runs, and prints its "needs attention" list.
  - Audit: `portstudy audit --week W` (exit code non-zero on any problem); `run_weekend.py` runs it
    after J9 and the result goes into the study log.
  - Backups: confirm `data/logs/` (including `runs/` and `libraries/`) is in every backup and that
    nothing rotates or deletes logs.
- **Clock check.** WSL2's clock can drift after the PC sleeps, and the lock deadline depends on it.
  The preflight compares WSL time with the `Date` header of an HTTPS response (gateway or a provider)
  and refuses a live run if they differ by more than 2 minutes (prints the fix: `sudo hwclock -s`).

### Implementation areas

Simulator records, evaluation data loader, data checks, snapshot and universe, pipeline jobs, run
script, new study log, tests.

### Estimated LOC

Approximately 1,250 to 1,800 added, 350 to 520 modified.

### Files expected to change

`sim/simulator.py`, `sim/records.py`, `eval/data.py`, `data/checks.py`, `data/snapshot.py`,
`data/universe.py`, `config.py`, `config/study.yaml`, `config/study_test.yaml`, `pipeline/jobs.py`,
`pipeline/study_log.py` (new), `pipeline/events.py` (new), `pipeline/fallback_ledger.py` (new),
`pipeline/status.py`, `pipeline/context.py`, `logging_setup.py`, `io.py`, `llm/cache.py` or the
record store, `optimize/mean_variance.py`, `optimize/strategies/common.py`, `backup.py`,
`pipeline/cli.py`, `baseline/kronos.py`, `llm/runner.py`,
`llm/providers/*`, `scripts/run_weekend.py`, `scripts/check_live_ready.py`, tests.

### Dependencies

Phase 3.

### Can AI complete independently?

Yes for code and tests. The keep-awake helper and the clock check need confirmation on Carter's
desktop (`manualwork.md`).

### Risk

Medium to high. The weekly-return change touches every result; the removal rule changes the
universe. Both need tests on the synthetic world with planted cases.

### Complexity

4/5

### AI completion criteria

- Synthetic test: a planted weekend gap moves the old Friday-close weekly return but not the new
  open-to-open return; open-to-open portfolio returns match a hand calculation.
- Planted 12% weekly move: stock kept and flagged. Planted -45% drop: stock removed from the next
  snapshot onward, weight 0 at the next rebalance, row in `universe_removals.csv`. Planted raw jump
  without an adjusted jump: skipped for that week only. A 40% move 200 days before the cutoff no
  longer skips the stock.
- Kill tests: killing J2 mid-chunk, J4 mid-run and J7 mid-portfolio, then rerunning, produces the
  same outputs as an uninterrupted run, with no repeated LLM calls beyond the in-flight ones.
- Random-kill test (N6): on a synthetic weekend, kill the process with SIGKILL at 20 random points
  spread across J1 to J9 (seeded), rerun after each kill, and confirm the final outputs, hashes and
  record counts equal an uninterrupted run, no completed LLM call is repeated, and the event timeline
  shows every interruption and resume.
- Logging tests (N6): every line of a synthetic weekend's logs carries the `run_id`; a forced
  exception in J4, in a solver and in a provider leaves a traceback in the logs and status file; a
  `warnings.warn` inside a job reaches the log; every planted fallback appears once in
  `fallbacks.csv`; `run_summary.md` is written after a forced failure; `portstudy audit` passes on
  a clean week and fails when a weight file or an LLM record is deleted; atomic writes and JSON-lines
  appends call `fsync` (monkeypatched counter).
- Simulated network outage shorter than the tolerance recovers with no `provider_error` records;
  longer marks the calls failed.
- G1b and U2 simulator tests pass; `portstudy closeout` refuses before the final window closes and
  completes after; `views.ic_history_start: live_start_week` ignores earlier weeks and starts every
  portfolio from cash at the first live week in a test; fixed-IC mode tests pass.
- `study_log.csv` has one row per week after the synthetic multi-week run and survives reruns.
- Lint, format, mypy, pytest pass; PROGRESS entry; committed and pushed.

---

## Phase 3B: Neutral LLM view, confidence-only counterfactual and strategy sensitivity

### Goal

Make a rating of 0 neutral (E7), make the confidence effect measurable (G18), and set Mean-CVaR and
Resampled MVO so LLM views can move them (S8), before the evaluation is built and before the
freeze. Evidence and experiment outputs: `plans/REVIEW5_fable_2026-11-11.md` (R1, R2, Appendix A).

### High level summary

- **Neutral LLM view (E7).** `views/build.py` builds `q_L = pi + IC_L * sigma * z_L`. Change it to
  `q_L = mu_Q0 + IC_L * sigma * z_L`, where `mu_Q0` is the week's Q0 posterior mean (prior plus the
  baseline view; equal to `pi` when the baseline view is dropped). `build_views` takes the Q0
  posterior mean as an input; `_source_views`, `posterior_for_source` and `build_weights` in
  `optimize/__init__.py` pass it (J7 already computes the Q0 posterior first). `Omega_L` is
  unchanged. Save `q_L` and `mu_Q0` in the posterior file. Update the module docstrings, the
  formulas in `docs/design.md` and the README.
  Checked on 2026-11-11 with the repo's own functions: with every rating 0 the posterior mean
  equals Q0's to 1e-18 and MVO and Top-K weights equal Q0 exactly; Robust still moved 5% to 11%,
  Resampled 1% to 3% and Mean-CVaR under 0.3%, from lower uncertainty alone.
- **Confidence-only counterfactual (G18).** New `optimize/counterfactual.py` with one function that
  returns, for a source-week, the posterior mean and covariance with the same `Omega_L` and
  `z_L = 0` (one linear solve). With E7 its mean equals Q0's, so the useful output is the posterior
  covariance `M_conf`. J7 records `conf_only_uncertainty_ratio = trace(M_conf) / trace(M_Q0)` per
  source-week in the weight meta. The counterfactual itself runs no optimizer; the G19 bullet below
  solves confidence-only portfolios for Robust and Resampled MVO only. Phases 7 and 8 read the ratio.
- **Confidence-only control portfolios (G19).** In `PortfolioId` add a `conf_only(strategy,
  source)` constructor with key `<strategy>__CONF__<model>__<config_id>`, and make `parse`, the
  replay, scoring and status code accept it. In `build_weights`, after each steered source's
  posterior is built, solve Robust MVO and Resampled MVO once more with the posterior mean of Q0
  and the source's `M_conf` (from `optimize/counterfactual.py`), using the same caps, previous
  weights and solver settings as the steered portfolio, and write them with the normal weight
  files and meta (`portfolio_kind: conf_only`). When the source has no valid LLM view that week,
  the confidence-only portfolio copies Q0's weights, like the steered one. A fallback week for the
  source applies the same fallback to its confidence-only portfolios. The set of strategies is a
  config list, `strategies.conf_only_controls: [robust, resampled]`.
- **Expected tracking error.** J7 records `expected_te_vs_q0 = sqrt((w - w_Q0)' Sigma (w - w_Q0))`
  in each steered portfolio's weight meta next to `view_shift`.
- **Calibration script (S8).** New `scripts/calibrate_view_sensitivity.py`, run on the synthetic
  world at `universe.size` stocks (100) for at least 12 synthetic weeks. For each strategy it records the median
  expected tracking error against Q0 under three inputs: every rating 0 (confidence only), the
  stub's noise ratings, and the stub's skilled ratings. It then picks settings by this fixed rule,
  so no judgment is needed while Carter is away:
  - Mean-CVaR `lam`: from the grid `[1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01]`, the largest value
    whose median tracking error under noise ratings is at least 25% of MVO's.
  - Resampled `draw_scale`: add `strategies.resampled.draw_scale` (draws use
    `draw_scale * M`; default 1.0 keeps today's behavior). From the grid
    `[1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01]`, the largest value that meets the same 25% rule.
  - If no grid value meets the rule, keep the smallest grid value and say so.
  The script writes `data/outputs/calibration/view_sensitivity.csv` and a short markdown summary,
  and Devin writes the chosen values into `config/study.yaml` with a comment naming the script and
  date. Robust's `kappa` and Top-K's `k` are not changed. Carter confirms or overrides the values
  before the freeze (`manualwork.md`); nothing here uses real data.
- **Decision text.** Update S4 (now "set by the S8 calibration") and add E7, G18, S8 in
  `config/study.yaml` decisions and `docs/open_questions.md` (Phase 4 covers the rest).

### Implementation areas

Views, Black-Litterman callers, weight building, strategy config (Resampled), a new script, tests,
docs for the formula.

### Estimated LOC

Approximately 550 to 800 added, 180 to 300 modified.

### Files expected to change

`views/build.py`, `optimize/__init__.py`, `optimize/counterfactual.py` (new),
`optimize/strategies/resampled.py`, `types.py`, `sim/` replay and scoring, `pipeline/jobs.py`,
`config.py`, `config/study.yaml`, `config/study_test.yaml`,
`scripts/calibrate_view_sensitivity.py` (new), `docs/design.md`, `README.md`,
new `tests/unit/test_v4_phase3b_views.py`, updates to `tests/unit/test_phase7_views.py` and
`test_v2_phase6_views_bl.py`.

### Dependencies

Phase 3A.

### Can AI complete independently?

Yes. The calibration rule is fixed above. Carter confirms the chosen values later.

### Risk

Medium to high. E7 changes every steered portfolio. Existing view and Black-Litterman tests encode
the old formula and must be updated on purpose, not loosened.

### Complexity

3/5

### AI completion criteria

- With every valid rating equal (so `z_L = 0`), the steered posterior mean equals the Q0 posterior
  mean to 1e-12, and MVO and Top-K weights equal Q0's exactly, for both `IC_b > 0` and `IC_b = 0`.
- With non-zero `z_L` the tilt moves in the direction of `z_L` for a diagonal covariance test case.
- `draw_scale = 1.0` reproduces the current Resampled weights; a smaller value raises the tracking
  error under noise ratings in a test.
- `expected_te_vs_q0` and `conf_only_uncertainty_ratio` are in the weight meta for every steered
  portfolio in a synthetic week.
- Every steered source has `robust__CONF__...` and `resampled__CONF__...` portfolios in a
  synthetic week; with every rating 0 they equal the steered portfolio exactly; they replay and
  score like other portfolios; a source fallback week gives them Q0 weights.
- The calibration script runs on the synthetic world, writes its CSV and summary, and the chosen
  values and the full grid are recorded in PROGRESS.
- Lint, format, mypy, pytest pass; PROGRESS entry; committed and pushed as
  `v4 phase 3B: neutral LLM view and strategy sensitivity`.

---

## Phase 4: Decisions, evaluation config and pre-registration file

### Goal

Put every section 0.3 decision into config and the decision record, and add the pre-registration
file that fixes the main comparisons and residual-IC controls before the live run.

### High level summary

- `config/study.yaml` decisions block: add L1, K2, K3, G8 to G21, E7, S8, G1b, N4, N5, N6, N7, C8, A7,
  A8, A9, U1, U2, D5, G22, and the changed D4, B1 and B4 (`value: 100`, the 2026-11-16 allocation;
  today it still reads "100 to 300"); update G7, G3, G3b, G4 text; keep
  every unchanged ID. Remove `DECISION(G6)`, `DECISION(W1)`, `DECISION(M1)` tags from code.
- New `eval` keys (pydantic models in `config.py`): `tie_band: 0.001` (G12), `tie_band_sensitivity:
  [0.0, 0.0005, 0.001, 0.005, 0.01]`, `bootstrap_samples: 2000`, `bootstrap_block_mean: null`
  (null means `T^(1/3)`), `signflip_exact_max_t: 24`, `permutation_samples: 10000` (above the exact
  limit), `significance: 0.05`, `factor_source: french | etf_proxy` with `fama_french_file` and new
  `momentum_file`, `risk_free: fred`, `cer_gamma: [2, 5]`, `report_active_weeks: true` (G8).
- Replace `eval.holm_families` with `eval.multiple_testing`: `primary_procedure: holm`,
  `secondary_procedure: fdr_bh`, `exploratory: [H4]`.
- New `config/preregistration.yaml`, loaded and validated by `config.py`:
  - `frozen: false` (Carter sets true, with a date, before the pilot). Extend
    `scripts/check_live_ready.py` (built in Phase 3) in this phase so a live run refuses while
    `frozen` is false or the file's hash changed after it was frozen.
  - `primary_comparisons`: exactly the four pooled comparisons of G7, each with `id`,
    `hypothesis`, `kind` (`pooled_source_ic_gap`, `pooled_tilt_ic_gap`, `pooled_prompt_ic_gap`) and
    selectors: (1) `rq1_llm_vs_baseline`: pooled LLM-minus-baseline rank IC gap over every enabled
    source, on the same stocks each week (G14); pooled LLM rank IC above zero is a secondary row; (2) `rq2_tilt_gap`: pooled steered-minus-Q0 tilt rank IC gap over every steered source;
    (3) `rq3_skills_vs_bare`: pooled skills-minus-bare rank IC gap over every enabled skills file
    and model; (4) `rq5_skills_vs_placebo`: pooled skills-minus-placebo rank IC gap. Each pooled
    test is a source-week regression with model fixed effects and standard errors clustered by
    week (Phase 7 builds the estimator). Validation rejects selectors that name a disabled model,
    prompt or strategy, and more than 10 entries, so Carter can still change the list before the
    freeze.
  - `residual_ic_controls`: default `[baseline_score, momentum_12_1, log_market_cap, volatility_63d,
    beta_252d, sector, loading_hml, loading_rmw, loading_cma]`.
  - `main_model_comparison_prompt: bare` (prompt used for model-vs-model portfolio tests).
  - `view_sensitivity_min_tracking_error: null` (G13). Carter sets it from the Phase 3B
    calibration and the pilot; validation refuses `frozen: true` while it is null.
  - Note in the file: `baseline_score` already contains momentum, so `momentum_12_1` is partly
    redundant as a control; it is kept so the residual IC also nets out momentum measured alone.
- `docs/open_questions.md`: rewrite the current-decisions table from section 0.3; move G6, W1, M1
  and the old D1, D2, N2 values to a "Retired and superseded" table with dates; keep the v1 table.
- Update `test_phase16_checklist.py`: accept `v4 phase N` subjects; every `DECISION(...)` tag and
  every `study.yaml` decision ID must appear in `open_questions.md`; retired IDs must not appear as
  code tags.

### Implementation areas

Config schema and YAML, decision docs, checklist test.

### Estimated LOC

Approximately 250 to 400 added, 100 to 200 modified.

### Files expected to change

`config.py`, `config/study.yaml`, `config/study_test.yaml`, `config/preregistration.yaml` (new),
`docs/open_questions.md`, `docs/config_reference.md`, `tests/unit/test_phase16_checklist.py`,
new `tests/unit/test_v4_phase4_config.py`.

### Dependencies

Phase 2.

### Can AI complete independently?

Yes. The main comparisons are decided (G7); the view-sensitivity threshold and the `frozen: true`
flag are Carter's (`manualwork.md` D4).

### Risk

Low.

### Complexity

2/5

### AI completion criteria

- Config loads; invalid preregistration (unknown model, 11 entries, unknown control) raises a clear
  error; the test config has its own small preregistration.
- Checklist test passes with the new IDs.
- Lint, format, mypy, pytest pass; PROGRESS entry; committed and pushed.

---

## Phase 5: Statistics primitives and portfolio metrics

### Goal

Add every statistical building block the proposal's Tests paragraph and metric list require, as
small tested functions, before any hypothesis code uses them.

### High level summary

In `eval/tests.py` (statistics) and a new `eval/metrics.py` (portfolio metrics):

- **Paired gap summary (extend `paired_summary`):** Newey-West SE with lags
  `floor(4 (T/100)^(2/9))` (report the plain SE too and use the larger for the t-stat); t critical
  values and p-values from Student-t with T - 1 degrees of freedom; win/tie/loss with `eval.tie_band`;
  sign test; sign-flip p-value by exact enumeration of all 2^T sign patterns when
  T <= `signflip_exact_max_t` (vectorized in chunks to bound memory), Monte Carlo otherwise, with a `signflip_method`
  column; stationary block bootstrap with percentile-t 95% CI (mean block length from config or
  `T^(1/3)`); minimum detectable effect `2.8 sd / sqrt(T)` (keep `mdg`).
- **Multiple testing:** `holm` (keep) and `fdr_bh` (Benjamini-Hochberg, via statsmodels
  `multipletests`) in a new `eval/multiple.py` with one function that assigns each result row to
  primary, secondary or exploratory from the preregistration and fills `p_adj` and `family`.
- **Sharpe tests:** Ledoit-Wolf (2008) studentized time-series bootstrap test of the difference of
  two Sharpe ratios on paired weekly excess returns (p-value and CI); Jobson-Korkie with Memmel's
  correction as a secondary column; probabilistic Sharpe ratio (Bailey and Lopez de Prado) against
  0; deflated Sharpe ratio given the number of trials.
- **Portfolio metrics (`eval/metrics.py`):** weekly and daily excess returns using the FRED rate
  from snapshots (reuse `daily_rf_from_snapshots`, compound to weekly); Sharpe net of risk-free
  (annualized by sqrt(52)); lag-1 autocorrelation and Lo (2002) adjusted annual Sharpe (reported,
  not used for tests); Sortino, max drawdown, Calmar and realized CVaR at 95% from daily returns;
  information ratio and tracking error vs a benchmark series; certainty-equivalent return for each
  `cer_gamma`; turnover; break-even cost = gross active return / extra turnover vs Q0 (NaN with a
  reason when extra turnover is not positive); hit rate with binomial p-value.
- Optional, behind the `stats` extra: wrappers for `arch.bootstrap` SPA, StepM (Romano-Wolf) and
  MCS, used only by Phase 8 for "any source beats Q0" and "set of best models". If `arch` is not
  installed those tables are skipped with a note.
- Replace `portfolio_metrics` in `hypotheses.py` with a call into `metrics.py` that adds the new
  columns; keep the existing columns (`total_return`, `turnover`, `total_cost_usd`, stops, view
  shift, fallback and solver-failed weeks).

### Implementation areas

Evaluation statistics and metrics, tests.

### Estimated LOC

Approximately 550 to 800 added, 100 to 150 modified.

### Files expected to change

`eval/tests.py`, `eval/metrics.py` (new), `eval/multiple.py` (new), `eval/hypotheses.py`
(`portfolio_metrics`), `eval/factors.py` (risk-free helper reuse), new
`tests/unit/test_v4_phase5_stats.py`.

### Dependencies

Phase 4 (config keys).

### Can AI complete independently?

Yes.

### Risk

Medium. Statistical code is easy to get subtly wrong; every function needs a test against a known
value.

### Complexity

4/5

### AI completion criteria

Unit tests with known answers:
- Exact sign-flip on a small vector equals a hand count; exact and Monte Carlo agree within
  tolerance at T = 20.
- `fdr_bh` and `holm` match statsmodels on a fixed p-value vector.
- Ledoit-Wolf test: identical series give p near 1; a series with clearly higher Sharpe gives a
  small p; result is deterministic for a fixed seed.
- PSR and DSR match values computed from the published formulas on a fixed input.
- Sortino, CVaR, max drawdown, IR, tracking error and CER on hand-made series.
- Student-t p-values equal `scipy.stats.t` values for T - 1 degrees of freedom.
- Lint, format, mypy, pytest pass; PROGRESS entry; committed and pushed.

---

## Phase 6: Factor data and stock characteristics

### Goal

Provide the Kenneth French daily factors and a stock-week characteristic panel so H1, H2 and H5
can control for market, style and the baseline without any new data provider.

### High level summary

- **French loader (`eval/factors.py`):** extend `fama_french_factors` to read the daily FF5 (2x3)
  file and the daily momentum file, convert percent to decimals, join on date, and return
  `MKT_RF, SMB, HML, RMW, CMA, MOM, RF`. Accept the CSV inside the zip or an extracted CSV.
- **Fetch script:** new `scripts/fetch_french_factors.py` that downloads the two zip files from the
  French Data Library into `data/factors/`, records the download date and the last date covered,
  and refuses to overwrite a newer file. URLs tagged UNVERIFIED (confirm names on first run).
  Tests use a small fixture file in `tests/fixtures/factors/`.
- **Factor source selection:** `eval.factor_source: french` uses the French files and fails
  clearly if they do not cover the study's last week; `etf_proxy` keeps today's ETF method
  (labeled "ETF proxy, monitoring only" in every output). The ETF method gains a momentum proxy
  only if one is already in the snapshot ETFs; otherwise MOM is omitted and the omission is printed.
- **Characteristics (`eval/characteristics.py`, new):** one row per stock-week, built only from data
  already saved: baseline score `z_b`; 12-1 month momentum and 1-month reversal (reuse
  `baseline/momentum.py`); log market cap and sector (`reference.csv`); 63-day volatility and 252-day
  beta to SPY (from the price ledger); trailing 252-day loadings on HML, RMW and CMA from a
  regression of each stock's daily excess returns on the French factors (as of the week's cutoff,
  so nothing after the cutoff is used). Cross-sectionally rank or z-score each week. Add these
  columns to `EvalData.stock_week`.
- The synthetic provider gets a seeded synthetic factor file so tests and the synthetic e2e run can
  exercise the whole path.

### Implementation areas

Evaluation data, factor data, scripts, synthetic data, tests.

### Estimated LOC

Approximately 350 to 500 added, 50 to 100 modified.

### Files expected to change

`eval/factors.py`, `eval/characteristics.py` (new), `eval/data.py`, `scripts/fetch_french_factors.py`
(new), `data/providers/synthetic.py` (synthetic factors), `tests/fixtures/factors/` (new),
new `tests/unit/test_v4_phase6_factors.py`.

### Dependencies

Phase 5.

### Can AI complete independently?

Yes, offline with fixtures. The first real download is a manual step (`manualwork.md`).

### Risk

Medium. Look-ahead in the trailing loadings would bias residual IC; tests must prove each week's
loadings use only data up to that week's cutoff.

### Complexity

3/5

### AI completion criteria

- Loader returns the seven columns in decimals from the fixture; coverage check fails clearly when
  the file ends before the study.
- Characteristic test: changing returns after a cutoff does not change that week's characteristics.
- Lint, format, mypy, pytest pass; PROGRESS entry; committed and pushed.

---

## Phase 7: Signal-level hypothesis tests

### Goal

Implement H1, the signal half of H2, the signal half of H3, model-vs-model signal tests and H3b
exactly as the proposal states.

### High level summary

All in `eval/hypotheses.py` (or a split `eval/signal_tests.py` if the file passes about 600 lines),
returning the existing row shape plus new columns:

- **Same stocks (G14).** Every IC gap below is computed each week on the intersection of stocks that
  have a valid value in both signals (today H1 compares LLM IC on valid outputs with baseline IC on
  every stock). Report the intersection size per week.
- **H1:** paired weekly gap IC(LLM likely) minus IC(baseline) per source (keep); add ICIR, IC hit
  rate with binomial p, top-minus-bottom quintile spread per week (mean and t), and optional IC
  against week t+2 returns (decay). Add **residual IC (G9):** each week regress next-week return
  ranks on the source's score and the preregistered controls (Phase 6 panel); the slope series is
  tested with Newey-West and t(T-1) (Fama-MacBeth). Also report the simpler "residualize the score
  on the controls, then rank IC" version as a cross-check column.
- **H2 signal (G14, changed):** IC of the steered Black-Litterman tilt (`mu_BL - pi` from
  `posterior/<source>.csv`) minus IC of Q0's tilt (`posterior/Q0.csv`), on the same stocks, plus the
  residual IC of the steered tilt. With E7 the two tilts are identical when every rating is 0, so
  the gap is exactly zero when the LLM says nothing (test this on a synthetic week). Report the
  mean `conf_only_uncertainty_ratio` (G18) per source next to the result.
- **H3 signal:** per model, skills minus bare and skills minus placebo IC gaps (keep). Add the
  **pooled test:** a stacked regression of source-week IC on prompt type with model fixed effects
  and week fixed effects, standard errors clustered by week (statsmodels OLS with
  `cov_type="cluster"`); report the skills-minus-placebo and skills-minus-bare coefficients with
  t(T-1) p-values.
- **Model comparisons (new):** paired weekly IC difference between each pair of models on the
  preregistered prompt (`main_model_comparison_prompt`), plus a pooled regression of source-week IC
  on model and prompt with week fixed effects, clustered by week.
- **H3b:** keep narrow-minus-wide IC (median width split per source-week). Add a weekly
  Fama-MacBeth interaction of score and confidence `c` (from E1) and a calibration check: weekly
  Spearman correlation between range width and absolute rank error, tested over weeks.
- Every result row carries `hypothesis`, `comparison_id` (matching preregistration ids when it is a
  primary comparison) and is passed through `eval/multiple.py`.

### Implementation areas

Evaluation hypotheses, tests.

### Estimated LOC

Approximately 450 to 650 added, 100 to 200 modified.

### Files expected to change

`eval/hypotheses.py` (and optionally `eval/signal_tests.py`), `eval/rank_ic.py`, `eval/data.py`,
new `tests/unit/test_v4_phase7_signal.py`, updates to `tests/unit/test_phase11_eval.py` and
`test_v2_phase10_eval.py`.

### Dependencies

Phases 5 and 6.

### Can AI complete independently?

Yes.

### Risk

Medium. Pooled regressions and Fama-MacBeth need careful alignment of weeks and sources.

### Complexity

4/5

### AI completion criteria

- On a synthetic panel with a planted stock-specific effect, residual IC is positive and
  significant; with a planted pure style tilt (score equal to a control), raw IC is positive but
  residual IC is near zero.
- Pooled H3 regression recovers a planted skills-minus-placebo IC difference.
- The stub noise progression (`skills_1` to `skills_4` less noisy than controls) produces positive
  skills-minus-placebo estimates in the synthetic world.
- Lint, format, mypy, pytest pass; PROGRESS entry; committed and pushed.

---

## Phase 8: Portfolio-level hypothesis tests and multiple testing

### Goal

Implement the portfolio half of H2 and H3, H4, H5 and the HRP and equal-weight comparisons, and
apply the primary, secondary and exploratory families.

### High level summary

- **Weekly returns.** Use the open-to-open weekly returns from Phase 3A everywhere (paired gaps, IR,
  Sharpe, Ledoit-Wolf); daily close values only for daily metrics and factor regressions.
- **View sensitivity (G13).** For each strategy compute the median weekly expected tracking error
  against Q0 from the saved `expected_te_vs_q0`. Label strategies below
  `view_sensitivity_min_tracking_error` "insensitive to views by construction" in every table and
  keep them out of any "LLM does not help" statement. Keep view shift as a descriptive column. For
  Robust, Resampled and Mean-CVaR, state in the table notes that part of their difference from Q0
  comes from lower uncertainty alone (G18), and show the uncertainty ratio.
- **Model-change sensitivity (G21).** Repeat the four main comparisons and the H2 portfolio tests
  without the weeks flagged `model_change` for the affected model, as a secondary analysis, and list
  the flagged weeks in the report. Report the G20 repeat agreement per model and prompt in the
  data-quality section.
- **Information vs confidence split (G19).** For Robust and Resampled MVO, report steered minus
  Q0 split into information (steered minus `CONF`) and confidence (`CONF` minus Q0), with the same
  tests as H2 (secondary family). Mean-CVaR's table note says no confidence-only control exists for
  it and gives its uncertainty ratio instead.
- **Realized gaps (G8).** The intention-to-treat series uses the realized weekly return gap in
  every clean week. Do not set inactive weeks to zero. Report how large the inactive-week gaps are
  (mean absolute gap) as a check on path effects from stops and costs.
- **Active-week flags (G8):** add `active_view` (weeks by steered portfolio) to `EvalData` from the
  recorded `use_llm_view` / `has_llm_view`; verify it is recorded for every steered source-week and
  that fallback weeks are excluded as today by `clean_mask`.
- **H2 portfolio:** for each steered portfolio vs its own Q0, two analyses (ITT on clean weeks,
  active weeks only) with: mean active return (paired summary), information ratio, tracking error,
  Ledoit-Wolf Sharpe-difference test, turnover gap, break-even cost, share of active weeks. Daily
  active-return factor regression (next bullet) joined in.
- **H5:** regress each steered portfolio's daily active return (steered minus Q0, no RF on the left)
  on MKT_RF, SMB, HML, RMW, CMA, MOM with Newey-West SE (lags from the daily rule, about 4 for 115
  days); report alpha (daily and annualized), t, p and each loading. Also keep a regression of each
  portfolio's excess return (including Q0 and controls) on the same factors. Add the placebo
  content-control row: for each model and strategy, active return of skills portfolios minus the
  placebo portfolio's active return. Wording everywhere: the placebo controls for prompt length and
  any change to the views; chance is handled by the tests and intervals, market and style luck by Q0
  and the factor model. For any portfolio a reader would headline (largest Sharpe or IR),
  report the deflated Sharpe ratio (trials = number of steered portfolios) and a flag for |t| > 3.
- **H3 portfolio:** paired weekly return of each skills portfolio minus its placebo portfolio and
  minus its bare portfolio, per model and strategy; plus the pooled regression of weekly active
  return on prompt type with model and week fixed effects, clustered by week.
- **Model comparisons (portfolio):** paired weekly return difference between models for the
  preregistered prompt, per strategy.
- **H4 (exploratory):** keep the strategy ranking by mean active return and the v1 splits; add
  strategy-vs-strategy differences of active returns (paired series) and regime interaction
  regressions `gap_t = a + b Regime_t + e_t` for market direction and volatility, with t(T-1) CIs.
  Earnings split stays on source IC. Mean view shift per strategy stays. Optional MCS over
  strategies when `arch` is installed.
- **Controls:** paired weekly return and Ledoit-Wolf Sharpe test of each steered portfolio and each
  Q0 against equal weight and HRP, reported as estimates with CIs (secondary family only).
- **Any-source tests (optional, `arch`):** Hansen SPA per strategy ("does any source beat Q0") and
  StepM (Romano-Wolf) "which sources beat Q0", both on saved weekly active returns.
- **Families:** `eval/multiple.py` assigns each row: preregistered ids to the primary Holm family;
  H4 rows exploratory with no adjustment; everything else secondary with Benjamini-Hochberg. Each
  table has `family` and `p_adj` columns.
- Remove the old per-strategy Holm-only logic (`_holm_within` usage) once every caller uses
  `multiple.py`.

### Implementation areas

Evaluation data and hypotheses, factors, multiple testing, tests.

### Estimated LOC

Approximately 550 to 800 added, 150 to 300 modified.

### Files expected to change

`eval/data.py`, `eval/hypotheses.py` (or `eval/portfolio_tests.py`), `eval/factors.py`,
`eval/multiple.py`, new `tests/unit/test_v4_phase8_portfolio.py`, updates to existing eval tests.

### Dependencies

Phases 5, 6, 7.

### Can AI complete independently?

Yes.

### Risk

Medium to high. This is the core of the paper's evidence; misaligned weeks between steered, Q0,
daily values and factors would silently bias results.

### Complexity

4/5

### AI completion criteria

- On synthetic data with a planted LLM effect, H2 ITT and active-week mean active returns are
  positive, active-week effect size is larger, and the t-stats are similar (the ITT dilution
  property in section 7 of the research notes).
- A planted pure factor tilt (steered portfolio equal to Q0 plus a momentum tilt) shows a positive
  MOM loading and an alpha near zero.
- A source whose views are off every week has tracking error 0, active return 0 and NaN break-even
  cost with a reason.
- Family assignment: preregistered ids get Holm, H4 rows unadjusted, others BH.
- Lint, format, mypy, pytest pass; PROGRESS entry; committed and pushed.

---

## Phase 9: Report rewrite

### Goal

Make `portstudy eval` produce a report that reads like the proposal's Evaluation section: main
results first, every number with its interval, and no shuffle anywhere.

### High level summary

- `eval/report.py` sections in this order: header (decision values in force, factor source,
  preregistration frozen date or "not frozen", weeks, models, prompts, strategies, fallback and
  active-week shares); **Main results** (the primary family table with effect, CI, p, Holm-adjusted
  p and minimum detectable effect); H1; H2 (ITT and active weeks side by side); H3; model
  comparisons; H3b; H4 (marked exploratory); H5 (factor loadings and alpha, placebo control, DSR and
  t > 3 flags); controls vs HRP and equal weight; portfolio metrics table (all G10 metrics);
  tie-band sensitivity; view sensitivity per strategy (G13); data integrity (removals, large-move
  flags, data-error skips per week from Phase 3A); the study log summary; a "how to read a null" note that prints each main test's minimum detectable
  effect.
- **All metrics (G22):** `eval/all_metrics.py` gathers every metric from Phases 5 to 8 for every
  unit (source, steered portfolio, Q0, confidence-only portfolio, HRP, equal weight, and pooled
  rows) into `all_metrics.csv` in long form, with a `reason` for each NaN. The report's portfolio
  table leads with annualized Sharpe net of the T-bill rate, PSR and the Ledoit-Wolf Sharpe test.
  A completeness check lists any enabled unit and metric pair with no row and fails the eval.
- CSV for every table, `report.md`, and figures: weekly IC series, IC gap bars with CIs, equity
  curves, active-return bars with CIs per strategy, H3b, H4 ranking, factor loadings per strategy.
  Remove `fig_shuffle_histograms.png`.
- `pipeline/cli.py` `eval` command: options `--synthetic`, `--config`, `--factor-source`,
  `--out`; no shuffle options.

### Implementation areas

Report, CLI, tests.

### Estimated LOC

Approximately 350 to 500 added, 200 to 300 modified or deleted.

### Files expected to change

`eval/report.py`, `pipeline/cli.py`, `tests/integration/test_phase11_report.py`, new
`tests/integration/test_v4_phase9_report.py`.

### Dependencies

Phases 7 and 8.

### Can AI complete independently?

Yes.

### Risk

Low to medium. Mostly presentation; the risk is a section silently missing.

### Complexity

3/5

### AI completion criteria

- Integration test runs `portstudy eval --synthetic` and checks every section heading and every CSV
  exists, the Main results table has one row per preregistered comparison, and the word "shuffle"
  does not appear.
- Report runtime on the synthetic study is a few minutes at most (log it in PROGRESS).
- Lint, format, mypy, pytest pass; PROGRESS entry; committed and pushed.

---

## Phase 10: Documentation recheck and repository cleanup

### Goal

Make README and docs describe the code exactly as it now is, with no "pending PLAN4" notes left.

### High level summary

- README and every doc were rewritten on 2026-11-11 to the target design, with notes such as
  "pending PLAN4 Phase 3" or "removed in PLAN4 Phase 1" where the code still differed. Remove each
  note now that the code matches, and correct anything Phases 1 to 9 built differently from what the
  docs say (record those differences in PROGRESS).
- `docs/config_reference.md`: document every new key with default, meaning and decision id
  (`kronos.batch_size`, `kronos.device`, `on_kronos_failure`, every new `eval` key,
  `multiple_testing`, `config/preregistration.yaml`); delete the rows for removed keys and the retired
  environment variables.
- `docs/design.md`: replace the "What the code does today" paragraph with the final module map of
  `eval/` (which function implements which test).
- `docs/runbook.md`: add `scripts/check_live_ready.py`, `scripts/fetch_french_factors.py`,
  `--time-gateway`, and the report sections; remove the `--skip-shuffle` notes.
- `docs/desktop_setup.md`: switch the install line to the `uv.lock` command and the extras added in
  Phase 3; add the J9 backup probe; replace the Kronos timing note with the chunked figure if the
  build VM measured one (label it "build VM").
- `docs/build_decisions.md`: add the v4 entries (newest first).
- Code comments: sweep `src/` and `scripts/` for stale mentions of ARCTIC jobs, Slurm, shuffles,
  CPU-hour budgets and small models; fix or remove them.
- Do not move or rename anything in `plans/` or `llm_research/`.

### Implementation areas

Documentation, comments.

### Estimated LOC

Approximately 250 to 450 modified or added.

### Files expected to change

`README.md`, `docs/*.md`, comments across `src/` and `scripts/`, `plans/PROGRESS.md`.

### Dependencies

Phases 1 to 9.

### Can AI complete independently?

Yes.

### Risk

Low.

### Complexity

2/5

### AI completion criteria

- `grep -rniE "pending PLAN4|removed in PLAN4|until PLAN4"` over `README.md docs/` returns nothing.
- `grep -rniE "slurm|sbatch|shuffle|walltime|cpu.hour|e4b|phi-4|llama 3.2|login mode"` over
  `README.md docs/*.md src scripts config` returns nothing, except retired-decision rows in
  `open_questions.md` and the retired note in `build_decisions.md`.
- Every relative link in README and docs resolves.
- Lint, format, mypy, pytest pass; PROGRESS entry; committed and pushed.

---

## Phase 11: End-to-end verification on synthetic data

### Goal

Prove the whole updated system works together before Carter touches real services.

### High level summary

- Run the full default test suite, the e2e suite and `pytest -m slow` where the VM allows.
- Run `python scripts/make_synthetic_world.py` and `python scripts/run_weekend.py --synthetic`
  for several weeks, then `portstudy eval --synthetic`; confirm every report section and that the
  planted synthetic skill effects show up in the expected direction.
- Record each strategy's median view shift and median expected tracking error on the synthetic
  study in PROGRESS (G13 input for Carter's pre-registration), and rerun
  `scripts/calibrate_view_sensitivity.py` to confirm the Phase 3B values still meet the rule.
- Repeat the review's two experiments on the final code at 100 stocks and record the outputs: every
  rating 0 gives MVO and Top-K weights equal to Q0 (E7), and the simulator stop case shows no
  Tuesday sale (G1b).
- Run `portstudy closeout` on the synthetic study after its last window.
- Run the random-kill test on the full synthetic study (not only one weekend) and `portstudy audit`
  on every synthetic week; record the results and the fsync overhead in PROGRESS.
- Run the canary step on every synthetic week and `portstudy repeat-check` on the synthetic pilot;
  inject a changed `model` field in one week and confirm only that week is flagged and the report's
  sensitivity table drops it.
- Confirm in the synthetic run: `ic_mode: fixed` is used in every weight meta; no pilot week is in
  the replayed results; the `CONF` portfolios exist for every steered source; a forced Kronos
  failure week runs with `kronos_fallback: true`.
- Run the kill-and-resume tests from Phase 3A once more on the full synthetic weekend.
- Run `python scripts/benchmark_pipeline.py --n-stocks 300 --sources 3 --weeks 24` and
  `--n-stocks 100` on the build VM and record the numbers in PROGRESS as "build VM" figures (the
  desktop figures from 2026-11-10 stay the reference in README).
- Run `python scripts/check_live_ready.py` and record the expected failure list (it should list
  exactly the open items in `manualwork.md`).
- Final consistency greps from Phases 1, 2 and 10; confirm `git log` shows only `ctsc` with no
  trailers for every v4 commit.
- Write a "v4 complete" PROGRESS entry summarizing what changed, test counts, runtime of the e2e
  suite, and the open items in `manualwork.md`.

### Implementation areas

Tests, verification, PROGRESS.

### Estimated LOC

Approximately 50 to 150 (fixes found during verification plus PROGRESS).

### Files expected to change

`plans/PROGRESS.md`, any files needing fixes.

### Dependencies

Phases 1 to 10.

### Can AI complete independently?

Yes.

### Risk

Low.

### Complexity

2/5

### AI completion criteria

All suites pass, the synthetic weekend and evaluation complete, greps are clean, the preflight
lists only items from `manualwork.md`, PROGRESS entry written, committed and pushed.

---

## 8. Human commit plan

Devin commits and pushes one commit per phase as `ctsc` (working rule 3): `v4 phase 1: remove the
shuffle test` through `v4 phase 11: end-to-end verification`
(Phase 3A commits as `v4 phase 3A: week-to-week integrity`, Phase 3B as
`v4 phase 3B: neutral LLM view and strategy sensitivity`). Carter's own commits while doing `manualwork.md`
(prompts, universe, preregistration, prices, config values) should each be separate and descriptive
(for example `prompts: skills files 1 to 4 and placebo`, `config: freeze preregistration`), and the
freeze is marked by the `frozen-v4` tag.

## 9. Manual developer tasks

See `manualwork.md` at the repository root (the only manual list). Nothing in Phases 1 to 11
requires Carter.

## 10. Testing checklist

### Existing tests to update or delete
- Delete: `tests/integration/test_v2_phase11_shuffle.py`, `tests/integration/test_v2_phase12_arctic.py`,
  `tests/unit/test_v3_phase2_shuffle.py`, `tests/unit/test_v3_phase3_arctic.py`.
- Update: `test_phase11_report.py`, `test_phase11_eval.py`, `test_v2_phase10_eval.py`,
  `test_v2_phase9_scale.py`, `test_v2_phase4_litellm.py`, `test_v2_phase1_config_identity.py`,
  `test_v3_phase1_config.py`, `test_review_fixes.py`, `test_phase16_checklist.py`, `tests/e2e/*`
  (week 12 injection), CI workflows.

### New tests
- Unit: `test_v4_phase3_desktop.py` (Kronos chunk seeding, preflight, backup probe, gateway
  model mismatch, N5 request failures, frozen features, prompt fields from the universe file,
  market-cap fallback), `test_v4_phase3a_integrity.py` (open-to-open returns, U1, G1b stop reset,
  U2 liquidation, outage tolerance, closeout, `ic_history_start`, canaries and model checks,
  repeat check, run_id and event timeline, tracebacks, fallback ledger, fsync, audit,
  random-kill resume), `test_v4_phase3b_views.py`
  (E7 neutrality, G18, `draw_scale`), `test_v4_phase4_config.py`, `test_v4_phase5_stats.py`,
  `test_v4_phase6_factors.py`, `test_v4_phase7_signal.py`, `test_v4_phase8_portfolio.py`.
- Integration: `test_v4_phase9_report.py` (includes the G22 completeness check on
  `all_metrics.csv`); updated synthetic e2e run through the new evaluation.
- Slow (manual): `kronos_smoke` with CUDA; 100-stock benchmark.

### Edge cases
- T below 3 weeks (tests return NaN with a reason, never crash); a source with zero active weeks;
  all ties; a strategy with every week solver-failed; missing Q0 for a week; French file ending
  before the study; a characteristic missing for a stock (row dropped that week, counted);
  preregistration naming a disabled model; exact sign-flip at T = 24 and Monte Carlo at T = 25.

### Regression testing
- Weekly pipeline J1 to J9 on synthetic data produces the same weights as before for the same
  inputs (Kronos stub, no batching effect), apart from hash changes from removed config keys.

### Performance testing
- `portstudy eval --synthetic` finishes in minutes; exact sign-flip enumeration at T = 23 is a
  vectorized operation (seconds); e2e suite stays under 15 minutes.

### Security testing
- `scripts/check_secrets.py` still passes; no keys in config, tests or docs; preflight reads keys
  only from the environment.

## 11. Documentation checklist

- README rewritten (Phase 10); `docs/desktop_setup.md` new; runbook and weekend checklist rewritten;
  `config_reference.md` lists every new key (`kronos.batch_size`, `kronos.device`,
  `data.alpaca_feed`, `views.ic_history_start`, `simulator.no_price_liquidation_days`,
  `strategies.resampled.draw_scale`, per-model `request_timeout_seconds`,
  `llm.gateway.features_frozen`, `llm.network_outage_tolerance_minutes`,
  `on_kronos_failure`, `views.ic_mode`, `views.fixed_ic`, `strategies.conf_only_controls`,
  all `eval` keys, `preregistration.yaml`); `open_questions.md` current and
  retired tables; `runbook.md` gains a "where to look" section (event timeline, job logs,
  library logs, `fallbacks.csv`, `run_summary.md`, `portstudy audit`); `build_decisions.md` v4 entries; PROGRESS entries per phase; no "pending PLAN4"
  notes left; stale comments removed.

## 12. Migration notes

Migration required: Yes (config and outputs only; no real data exists yet).
- Config: removed keys (`eval.shuffle`, `eval.holm_families`, `gateway.fallback_endpoint_file`,
  model `serving: slurm_fallback`, small models); new keys listed above; new
  `config/preregistration.yaml`.
- Outputs: report files change; old synthetic `data/` folders and caches should be deleted before
  the first v4 synthetic run (job input hashes change).
- Rollback: v3 is commit `44949aa`; any v4 phase can be reverted with git by Carter.
- Compatibility: nothing outside the repo consumes its outputs.

## 13. Major caveats

- **Power.** About 23 weeks gives wide intervals on every portfolio comparison; the report must lead
  with effect sizes, CIs and minimum detectable effects. Comparisons against equal weight and HRP
  cannot be settled in 23 weeks and are reported as estimates.
- **Residual IC depends on the control list.** Pre-register it (`manualwork.md` D4).
- **Small-sample asymptotics.** Newey-West and block bootstrap are near their limits at T = 23;
  the exact sign-flip and paired designs are the backstop.
- **Single machine.** The study depends on one desktop staying on, online and on the VPN for a
  weekend. Phase 3A adds keep-awake, outage retries, checkpoints, progress and a study log, but an
  outage near the Monday lock can still cost a week. The gateway needs the GSU VPN for every
  request, and its session limit is unknown; Carter watches for drops in the pilot.
- **Repository location.** Carter moved the repository out of OneDrive to
  `C:\dev\portfoliomanagement` on 2026-11-16 (`manualwork.md` 0.1); Phase 1 Step 0 verifies
  nothing was lost. OneDrive must never sync `.git`, `.venv` or caches. Runs happen in the WSL clone.
- **View sensitivity.** On the old settings Mean-CVaR and Resampled MVO moved 1% to 3% of the
  portfolio under noise ratings while MVO moved 25% to 38% (review R2). S8 rescales them on
  synthetic data before the freeze; G13 still reports any strategy that stays insensitive. The
  calibration uses a synthetic world, so the real sensitivity must be read again in the pilot.
- **Confidence effect.** E7 makes a rating of 0 neutral for the forecast mean, so MVO and Top-K
  equal Q0 when the LLM says nothing. Robust, Resampled and Mean-CVaR also read the posterior
  uncertainty, which any LLM view lowers; Robust moved 5% to 11% from that alone in the review's
  test. Their steered-minus-Q0 result therefore mixes information with a confidence effect, which
  the report states and G18 quantifies. G19's confidence-only control portfolios separate the two
  at portfolio level for Robust and Resampled MVO. Mean-CVaR has no such control (cost), so its
  steered-minus-Q0 result is labeled as mixing both.
- **Universe of 100 stocks (B4).** The startup allocation fits 100 stocks, so the power numbers below
  apply at 100: the smallest detectable mean weekly IC for one source is about 0.059 for a single
  test and about 0.07 under Holm with the four main comparisons. Pooling across models and prompts
  (G7) is what brings the main tests within reach of published effect sizes.
- **Power in numbers.** At 23 weeks and 100 stocks the smallest detectable mean weekly IC is about
  0.059 for one test and about 0.07 under Holm with the four main comparisons (review R3, analytic;
  at 300 stocks it would be about 0.034). Published weekly ICs for news-based LLM signals are usually 0.01 to 0.03,
  so this is an estimation study and a null result does not show the LLM has no effect.
- **Fixed view scale.** A8 replaces the trailing-IC switch with a fixed 0.02. A source whose ratings
  hurt keeps its view all study long; that is intended, because the study measures the effect of
  using the ratings, not of a gate that guesses which ones work.
- **News age.** The LLM reads news up to Friday 16:00 ET and the trade is at Monday's open (review
  R5). Kept on purpose (A9); the paper states it as a limitation.
- **The gateway is an alpha service.** ARCTIC's own page says models may be slow or unavailable and
  settings may change, and not to rely on it for deadlines. A model or setting change mid-study can
  break comparability. There is no status page or changelog, and the gateway exposes no model
  version (Andalib, 2026-11-12 and 2026-11-16). G21 logs each response's `model` field and the
  deployment ID and gateway version headers, reads `/v1/model/info` each weekend, and runs 10 canary
  prompts per model. Andalib noted the deployment ID will not catch every change (for example a
  change on the model server behind it), so the canaries are the main check. A change that alters
  answers only slightly can still pass the canary test.
- **MiniMax M2.7 always thinks.** Its reasoning counts toward the output limit and adds seconds per
  request; the 2,000-token starting limit is a guess to be measured (`--time-gateway`). If more
  than 15% of its answers are invalid, N1 sends every MiniMax week to the Q0 fallback, so check this
  before the pilot.
- **The ARCTIC request's numbers are estimates.** The token figures were corrected on 2026-11-12,
  but 40 GPU-seconds per Gemma request and 90 per MiniMax request are estimates from model size and
  memory bandwidth (G17). Access is approved (startup allocation, 2026-11-16), but how gateway
  requests are converted into GPU service units and what each key's limits are is not yet known;
  N7 tracks the estimate weekly and Carter checks real usage in Elpis monthly.
- **Both open-weight model cards recommend temperature 1.0.** The study uses 0 (D4) to keep
  variation as small as the shared gateway allows; the paper states the reason.
- **Gateway limits are unknown** (rate, concurrency, tokens per key) until the key exists, and the
  VPN session limit is unknown (Carter is asking GSU IT). Saving each rating as it completes and
  resuming after a drop (N4, N5) is the protection either way.
- **Third-party data.** Alpaca's free tier coverage, yfinance stability and French library timing are
  outside the project's control; the ETF proxy path remains for monitoring. The old default (IEX
  feed) is replaced by SIP (G16); whether the free plan serves SIP daily bars for older dates is
  checked in `manualwork.md` E before the pilot.
- **Durability limits.** N6 makes every completed unit of work survive a crash or power cut, but
  work in flight (an LLM call or a solver run that has not finished) is redone on resume, and a disk
  failure can still lose everything since the last off-desktop backup (J9 each week).
- **Reproducibility.** Temperature 0 with a seed does not give identical gateway outputs, because
  ARCTIC batches requests from many users (Andalib, 2026-11-12). The response cache keeps reruns of
  a saved week identical; G20 measures how often a fresh request agrees, and the paper reports it.

## 14. Confidence and unknowns

Confidence: 80%

Unknowns:
- Whether `use_llm_view` / `has_llm_view` is recorded for every steered source-week in a form
  `EvalData` can read without re-running J7 (Phase 8 verifies first).
- Exact French Data Library file names and the lag before the study window is covered.
- Whether WSL can write into the chosen backup folder.
- Whether the `powershell.exe` keep-awake helper works from WSL on Carter's desktop, and how often
  WSL2's clock drifts after sleep.
- Real gateway throughput and whether 4 concurrent requests per model is allowed.
- How gateway requests convert into GPU service units, and each key's limits (access approved
  2026-11-16; Carter is waiting to be added to the project's team to create keys). How many reasoning tokens a rating needs (2,000 is a starting
  guess; `reasoning_tokens` in each response's `usage` will show it).
- Whether a user key can read `/v1/model/info` (G21 continues without it).
- How often repeated gateway answers agree (G20 measures it in the pilot).
- Whether the Alpaca free plan returns SIP daily bars for the study dates, and how Alpaca reads a
  bare `end` date (review C11).
- Whether the S8 grid reaches the 25% rule for Mean-CVaR and Resampled MVO at 100 stocks.
- How many existing view and Black-Litterman tests encode the old `q_L` formula.
- Whether the steered-minus-Q0 confidence effect in Robust is large on real data.
- Whether the job-input hash can be narrowed per job without large changes (Phase 3 allows the
  frozen-config alternative).
- Runtime of exact sign-flip enumeration plus all bootstraps across about 175 portfolios (120 steered, 5 Q0, 48 G19 controls, HRP and equal weight) on the
  build VM (expected minutes; measured in Phase 11).
