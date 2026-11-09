# PLAN3.md: Corrections Plan (README, docs, config, ARCTIC jobs, shuffle scope)

PLAN.md (v1) and PLAN2.md (v2) are built and their tests pass. A full review of the repository at commit `b00f01b` against the actual research (the two proposals now in `docs/proposals/`) found that the README describes a different project in places, several docs are stale, a few config values are wrong, and the ARCTIC job layout breaks a cluster time limit. PLAN3 fixes all of it. It does not add new research features.

Read first, in this order: `docs/proposals/Project_Proposal_IEEE.pdf` (the research), `docs/proposals/ARCTIC_Resource_Request.pdf` (the compute request), PLAN2.md section 0.3 (decisions), then this file.

---

## 0. Working rules

All of PLAN2.md section 0 still applies (offline only, no em dashes, no "arm", small pure modules, idempotent jobs, UNVERIFIED tags, no changing `DECISION` values except the ones this file changes). The rules below repeat the ones that matter most and add three.

1. **Order.** Work through the phases in order. Do not start phase N+1 until phase N's "Done when" list passes, except that a truly blocked phase is logged in PROGRESS.md and the next phase that does not depend on it continues.
2. **After each phase:** run `ruff check src tests scripts`, `ruff format --check src tests scripts`, `mypy`, and `pytest` (default selection). Append a `v3 phase N: <title>` entry to PROGRESS.md (what changed, test counts, any test changed and why, choices made while unattended). Then commit and push.
3. **Git (same as PLAN2 rule 0.1.3).** Commit and push to `origin master` (`https://github.com/ctsc/llm_portfolio_management.git`), then `git push gitlab master` to keep the fallback mirror current. Author and committer must be exactly `ctsc <cartertierney0@gmail.com>`; set `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL`, `GIT_COMMITTER_NAME`, `GIT_COMMITTER_EMAIL` for every commit. No `Co-authored-by`, `Signed-off-by`, "Generated with", Devin, AI, bot name, session link, or any other trailer. Check `git log -1 --format='%an <%ae>|%cn <%ce>'` and `git log -1 --format=%B` before every push and amend the unpushed commit if either fails. No force push, rebase, history rewrite, branches, tags, or pull requests.
4. **Work autonomously.** Carter is not available. Never stop to ask. When something is unclear, choose the option most consistent with the proposals and PLAN2, write it under `Choices made while unattended` in that phase's PROGRESS entry, and continue.
5. **Wording.** Never write "the researcher" or "the student" in any file you create or edit (use "Carter" where a person is meant, or rewrite the sentence without a person). Do not call the project a thesis or say it is in quantitative finance: it is Carter Tierney's MS Computer Science research project at Georgia State University, advised by Dr. Parag Tamhankar.
6. **Do not edit prompt files.** Leave `prompts/` untouched (including the placeholder text that mentions "Phase 17"); changing them changes their hashes. Carter replaces them in PLAN2 phase 15 G.
7. **Numbers must come from the code or a run.** Every number you put in README or docs (timings, counts, CPU hours) must come from the config, the code, or a command you ran in this session. If you cannot measure it, say it is an estimate and where it comes from.

### 0.1 Decision changes in PLAN3 (Carter approved these on 2026-11-08)

| ID | Change | New value |
|---|---|---|
| G6 (changed) | shuffle test scope | 500 shuffles for the **MVO, Robust MVO and Top-K** steered portfolios and those three strategies' Q0s only. Mean-CVaR and Resampled MVO are excluded from H5 shuffles because they are about 99 percent of the shuffle compute. Runs on ARCTIC CPUs as a Slurm array; `portstudy shuffle run-local` stays as the backup. |
| W1 (new) | Slurm job length | ARCTIC is reported to cancel any job over 24 hours (unverified). Every Slurm job must request a walltime of at most 15 hours, with a target of 12 hours or less. No job may sit idle waiting on something outside the cluster (for example API batches). |
| M1 (new) | Slurm memory and cores | CPU jobs request `--mem-per-cpu=3G` (one CPU hour is one SU only at 3 GB or less per core, as the ARCTIC request states) and only the cores the work actually uses. |
| (config fix) | gateway URL | `llm.gateway.base_url: https://litellm.rs.gsu.edu/v1` (given by ARCTIC; needs the GSU network or VPN) |
| (config fix) | MiniMax output limit | remove the `minimax-m2.7: max_output_tokens: 2048` override; MiniMax uses the default until PLAN2 phase 15 E shows its answers are cut off. The 2048 value was never sourced. |

Everything else in PLAN2 section 0.3 is unchanged.

---

## 1. Review findings (what is wrong now)

Checked: every tracked file (167), the full test suite (passes, 279 test functions, about 31 minutes on a 2-core VM), `mypy` (clean), `ruff check` (clean).

**Code and config**
1. `ruff format --check` fails: 23 files would be reformatted. PLAN2 rule 0.1.2 requires it to pass.
2. `config/study.yaml`: `llm.gateway.base_url` is empty; `llm.settings.minimax-m2.7.max_output_tokens: 2048` is an unsourced value; `kronos.repo_commit` comment says "researcher ... Phase 17"; `timing.live_end_date` comment says "set by the researcher, no later than the April deadline".
3. `config/models.yaml` header says prices are confirmed "by the researcher in Phase 17".
4. `pyproject.toml` description says "LLM skills and investor mindsets study" (the investor-mindset framing was dropped).
5. Shuffle test (H5) runs every steered portfolio of all five strategies (`eval/shuffle.py::shuffle_portfolios`); G6 now limits it to MVO, Robust and Top-K.
6. `scripts/benchmark_pipeline.py` projects shuffle CPU hours with `--weeks 52` by default. The study is about 24 weeks (1 pilot plus about 23 live), so the "about 16,600 CPU hours" figure in README, PROGRESS and runbook is more than double the real projection, and it also includes the strategies G6 now excludes.
7. `pipeline/jobs.py::J4_open_weight_infer` refuses to start until J2 (Kronos) is done, and runs the open-weight models one after another. LLM calls go to the ARCTIC gateway and do not need Kronos, and running models in sequence makes J4 the longest step (see 9).
8. `scripts/arctic/*.sbatch` set no memory per CPU (violates M1). `shuffle_array.sbatch` asks for 2 cores per task, which doubles the CPU hours charged for single-threaded work; `weekend.sbatch` asks for 8 cores while it mostly waits.
9. `weekend.sbatch` runs `run_weekend.py --from J3`, which calls `wait_for_batches` and can keep polling until the Monday lock margin: up to about 60 hours from a Friday night start. That breaks W1 and holds 8 cores idle the whole time.
10. Code comments and docstrings still say "Phase 17" (now PLAN2 phase 15), "the researcher", or "the desktop": `backup.py`, `__init__.py`, `llm/cost.py`, `llm/providers/anthropic_batch.py`, `llm/providers/openai_batch.py`, `data/universe.py`, `data/providers/{alpaca,finnhub,fred,wikipedia,yfinance_provider}.py`, `scripts/run_weekend.py`, `scripts/estimate_cost.py`, `scripts/refresh_manifest.py`, `scripts/smoke_real_services.py` ("run this on the desktop"; it now runs on ARCTIC).

**README.md (wrong in substance, not just style)**
11. It calls the project "a master's thesis in quantitative finance". It is Carter Tierney's MS Computer Science research project at GSU, advised by Dr. Parag Tamhankar. No author, advisor or university is named anywhere.
12. The research question and hypotheses are not the proposal's. The proposal's question is: "How can LLM views built from specialized skills improve quantitative portfolio optimization strategies, and which strategies do LLMs help or hurt the most, and why?", with H1 to H5 as written in `docs/proposals/Project_Proposal_IEEE.pdf`. The title should match the proposal: "Specialized LLM Skills Across Fixed Quantitative Portfolio Optimization Strategies".
13. "Pre-registered" overstates it: nothing is filed with a registry. The accurate claim is that every study setting is fixed in config and frozen with the `frozen-v2` tag before the live run.
14. Statistics: it names only the sign-flip permutation and the bootstrap. The code (`eval/tests.py::paired_summary`) also runs the sign test and Newey-West standard errors (both in the proposal), and Holm uses the sign-flip p (`p_perm`). It should list all four and say which p-value Holm adjusts.
15. H5 (a) says factor regressions use ETF proxies. The proposal uses the Fama-French five-factor model; the code (`eval/factors.py::factor_table`) uses Fama-French whenever `eval.fama_french_file` is set and falls back to the ETF proxies only when it is not (offline tests).
16. Compute table says the shuffle test is "roughly 16,600 CPU hours at 300 stocks", which exceeds the 15,000 CPU hours requested. See findings 5 and 6.
17. "a planted rating effect that decreases from skills_1 to skills_5" is backwards: stub noise falls from 1.2 (`skills_1`) to 0.4 (`skills_5`), so the planted effect gets stronger from skills_1 to skills_5.
18. Repository layout omits `optimize/strategies/`, `optimize/problem_cache.py`, `llm/health.py`, `scripts/arctic/`, and `docs/proposals/`; it lists the desktop checklist.
19. Says "the researcher's machine", "researcher only", "the GitLab data and backup folders" (GitHub is primary), "the five skills files" (it is 4 to 5).
20. The documentation table lists `docs/desktop_checklist.md` (obsolete: nothing runs on the desktop) and `docs/arctic_gpu_request.md` (untracked, outdated; the real request is `docs/proposals/ARCTIC_Resource_Request.pdf`).
21. The "Design choices" section holds about 70 build-time entries. Only the five study design choices (K1, S4, S5, G3b, P1) belong in the README; the build log belongs in its own doc.
22. Timing claims ("about 30 seconds", "about 9 minutes", "about 14 minutes") were not re-measured after the review fixes.

**docs/**
23. `design.md` opens with "the v1 pipeline stages retained through v2 phase 1", which is out of date; its H5 text describes shuffles over every steered portfolio; it has no section on ARCTIC job layout.
24. `runbook.md` section 7 repeats the 15,000 CPU hour warning based on the 52-week projection; section 8 and the ARCTIC weekend section describe the single long weekend job.
25. `config_reference.md` says "set by the researcher" and "(v1 phase 17)"; it needs the new keys from this plan.
26. `open_questions.md` lacks the G6 change and W1, M1.
27. `arctic_setup.md` treats the gateway URL as unknown and says nothing about walltime limits or memory per core.
28. `github_setup.md`: "the researcher's hand-run", "The desktop checkout", "A private repository on the free plan has 2,000 Actions minutes" (the repository is public).
29. `gitlab_setup.md`: "the researcher's ... on the desktop", "tagged `frozen-v1` (v1 phase 17 E2)", "From the repository root on the desktop".
30. `desktop_checklist.md` is obsolete (Windows power plan, `.env` on the desktop).
31. `docs/arctic_gpu_request.md` is untracked and outdated.

---

## Phase 1: Config, decisions, formatting, and stale comments

**Goal:** config and code comments match the research and PLAN3 decisions, and formatting passes.

**Goals checklist:**
1. `ruff format --check src tests scripts` passes.
2. Config holds the PLAN3 values and decision entries.
3. No code comment, docstring, or config comment says "Phase 17", "the researcher", or "the desktop" (except `prompts/`, which is not touched).

**Build:**
1. Run `ruff format src tests scripts`. Do not change behavior; if a test changes because of formatting alone, something is wrong.
2. `config/study.yaml`:
   - `llm.gateway.base_url: https://litellm.rs.gsu.edu/v1` with the comment `# ARCTIC LiteLLM gateway; GSU network or VPN required`.
   - Remove the `minimax-m2.7` block under `llm.settings` (keep `settings: {}` if it becomes empty).
   - `eval.shuffle.strategies: [mvo, robust, topk]` with `# DECISION(G6)`.
   - Comments: `kronos.repo_commit` -> `# Carter pins the vendor/kronos commit in PLAN2 phase 15 C`; `timing.live_end_date` -> `# DECISION(A3): set by Carter before the freeze`.
   - `decisions`: update G6's value to the PLAN3 text; add W1 and M1 entries (status confirmed).
3. `config/study_test.yaml`: keep 20 shuffles; add `eval.shuffle.strategies: [mvo, robust, topk]`.
4. `config.py::ShuffleConfig`: add `strategies: list[StrategyName] = ["mvo", "robust", "topk"]` with a validator that every entry is in `strategies.enabled`.
5. `config/models.yaml` header: "Prices are per million tokens in USD and are placeholders until Carter confirms them in PLAN2 phase 15 G."
6. `pyproject.toml` description: "Specialized LLM skills across fixed quantitative portfolio optimization strategies: LLM views in a weekly Black-Litterman pipeline".
7. Fix every comment listed in finding 10: "Phase 17" -> "PLAN2 phase 15" (with the right sub-letter where obvious: smoke tests are 15 E, prices 15 G, Kronos pin 15 C); "the researcher" -> "Carter" or a rewrite; "on the desktop" -> "on ARCTIC" (smoke tests) or remove.
8. `scripts/check_banned_words.py`: apply the "the researcher"/"the student" check to `README.md`, `docs/**/*.md`, `src/**`, `scripts/**` and `config/**`, excluding `PLAN.md`, `PLAN2.md`, `PLAN3.md`, `PROGRESS.md`, `results.md`, and `prompts/`.

**Outputs:** reformatted files; the config, models, pyproject, and banned-word script changes above.

**Tests:** config loads with the new keys; the `ShuffleConfig.strategies` validator rejects a strategy not in `strategies.enabled`; the banned-word script passes on the repo and fails on a temp file containing "the researcher".

**Done when:** all quality gates pass and `grep -rn -i "phase 17\|the researcher\|the desktop" src scripts config` returns nothing.

---

## Phase 2: Shuffle test scope (G6) and correct projections

**Goal:** H5 shuffles run only for MVO, Robust and Top-K portfolios and their Q0s, and every CPU-hour projection uses the real study length and scope.

**Goals checklist:**
1. `shuffle_portfolios` returns only portfolios whose strategy is in `eval.shuffle.strategies`.
2. The report says plainly which strategies have H5 shuffle results and why the others do not.
3. The benchmark projects shuffle CPU hours for the shuffled strategies only, over 24 weeks by default, and prints a breakdown per strategy.

**Build:**
1. `eval/shuffle.py::shuffle_portfolios(portfolios, cfg)`: keep a portfolio only if its `PortfolioId.parse(key).strategy` is in `cfg.eval.shuffle.strategies` (steered and Q0). Controls stay excluded. `plan_shards`, `run_shard`, `run_shards_local`, `collect`, and `shuffle_test` all go through it.
2. `eval/report.py`: in each excluded strategy's section, write one line: "No shuffled-ratings test for this strategy (excluded under DECISION(G6) for compute; see the Q0 comparison and factor regression)." The H5 Holm family per strategy covers only the included strategies.
3. `scripts/benchmark_pipeline.py`:
   - `--weeks` default 24.
   - Projection uses only `cfg.eval.shuffle.strategies` and prints CPU hours per shuffled strategy plus the total.
   - Add a second projection that includes one simulator replay per shuffle (time one real replay on the synthetic world, or say clearly that replay time is not included if it cannot be timed offline).
   - Warn only when the total exceeds 15,000 CPU hours.
4. Run `python scripts/benchmark_pipeline.py --n-stocks 300 --sources 3 --weeks 24` (and `--n-stocks 100`) and record the output in PROGRESS.md. These numbers feed README and runbook in Phases 4 and 5.

**Tests:** `shuffle_portfolios` returns only mvo, robust and topk steered and Q0 keys under the default config; `plan_shards` count equals (shuffled portfolios) x (n_shuffles / shard_size); the report on the synthetic run has the exclusion line for cvar and resampled and H5 tables for the other three; projection arithmetic for 24 weeks.

**Done when:** tests pass and PROGRESS records the 24-week projection for 100 and 300 stocks.

---

## Phase 3: ARCTIC jobs under the time limit (W1) and SU efficiency (M1)

**Goal:** no Slurm job can run longer than 15 hours, no job waits idle, and each job requests only the cores and memory it uses.

**Goals checklist:**
1. J4 no longer waits on J2 and runs the open-weight models in parallel.
2. API batch collection happens in short polling jobs, not one long job.
3. Every sbatch file and the submit script respect W1 and M1.
4. Login mode and compute mode both work under the same rules.

**Build:**
1. **Drop the J2 gate in J4.** Remove the `read_status(week_id, "J2")` check from `J4_open_weight_infer` (results.md H3 recommended this; LLM calls do not use Kronos output). J7 must still refuse to run until J2, J4 (or its fallback marks) and J6 are done; add that check to J7 if it is not already enforced, with a test.
2. **Parallel models in J4.** Run each enabled open-weight model in its own worker (thread or process), each with its own `llm.concurrency` pool. Each model writes only its own records files, so the "main thread writes records" rule from PLAN2 phase 4 holds per model. Add `llm.parallel_models: true` (config) so tests can force sequential runs. A model that crashes must not stop the others; its sources are marked by J6 as usual.
3. **Short batch polling.**
   - Add `--poll-once` to `run_weekend.py`: run J5 once; exit 0 if every batch is collected, exit 3 if some are still pending and the lock margin has not passed, and at the lock margin let J6 mark the pending sources failed (N1) and continue.
   - New `scripts/arctic/finish.sbatch` (1 core, `--mem-per-cpu=3G`, `FINISH_WALLTIME`, default 03:00:00): runs `run_weekend.py --from J5 --poll-once`; on exit 3 it resubmits itself with `--begin=now+${POLL_INTERVAL_MINUTES}minutes` (default 60) and exits; on exit 0 it runs `--from J6` (J6 to J9) in the same job.
   - The chain stops at the lock margin by design, so it never loops past Monday.
4. **New weekend layout (`submit_weekend.sh`).**
   - compute mode: submit `kronos.sbatch` (GPU) and `llm.sbatch` (new: J3 then J4, CPU, 2 cores, `--mem-per-cpu=3G`, `LLM_WALLTIME`) at the same time (no dependency between them), then submit `finish.sbatch` with `--dependency=afterany:<kronos>:<llm>`.
   - login mode: J1 and J3 on the login node, submit kronos and llm (llm runs J4 only), then `finish.sbatch` as above; batch polling never runs in the login shell.
   - Remove `weekend.sbatch`, or keep it only as a thin wrapper that calls the new layout. If removed, update every reference.
5. **Resources in every sbatch file.**
   - `--mem-per-cpu=3G` on every CPU job (`llm.sbatch`, `finish.sbatch`, `shuffle_array.sbatch`).
   - `shuffle_array.sbatch`: `--cpus-per-task=1` (the solvers run single-threaded with BLAS pinned to one thread).
   - `kronos.sbatch` and `llm_fallback.sbatch` keep their GPU requests; set memory explicitly from `arctic.env` (`GPU_MEM`), since GPU jobs are not limited to 3 GB per core.
6. **Walltime guard.**
   - `arctic.env.example` adds `MAX_JOB_HOURS=15`, `LLM_WALLTIME`, `FINISH_WALLTIME`, `POLL_INTERVAL_MINUTES`, `GPU_MEM`; removes `WEEKEND_WALLTIME`.
   - `common.sh` refuses to submit any job whose walltime is above `MAX_JOB_HOURS` and prints a warning above 12 hours.
   - `--check-walltime` in `run_weekend.py` applies the same limits.
7. **Shard sizing.** `portstudy shuffle plan` prints the estimated minutes per shard from the benchmark solve times; if a shard would exceed 2 hours, lower `shard_size` in the plan output (not in config) and say so.
8. **Runtime budget in PROGRESS.** Using the benchmark and the synthetic run, record the expected wall time of each job at 100 and 300 stocks: Kronos (estimate, GPU), J3+J4 at the documented per-request times (estimate), finish (J5 to J9, measured on synthetic data), and one shuffle shard. Each must be under 12 hours; if one is not, say which and by how much.

**Tests (`tests/unit/test_v3_phase3_arctic.py` plus updates):**
- J4 runs with J2 pending.
- J7 refuses without J2 done.
- Parallel J4 gives the same records as sequential J4 on the stub models.
- One model raising does not stop the other.
- `--poll-once` exits 3 with pending batches before the margin, 0 when collected, and marks pending sources failed after the margin.
- `finish.sbatch` resubmission logic tested through a fake `sbatch` on PATH.
- `common.sh` refuses `16:00:00` and warns at `13:00:00`.
- `shellcheck` on all scripts (skip with a PROGRESS note if not installed).
- The e2e rehearsal still passes.

**Done when:** all tests pass, `submit_weekend.sh --dry-run` prints the new three-job layout with memory flags in both network modes, and PROGRESS has the runtime budget table.

---

## Phase 4: README rewrite

**Goal:** the README describes Carter's actual research and the system exactly as built, in a form a reviewer or advisor can read.

**Build:** replace `README.md` with the structure below. Keep it under about 300 lines. Use the proposals for research wording and the code and config for every system fact.

1. **Title and one-paragraph overview.**
   - Title: "Specialized LLM Skills Across Fixed Quantitative Portfolio Optimization Strategies".
   - Under it: "MS Computer Science research project, Georgia State University. Carter Tierney, advised by Dr. Parag Tamhankar. Compute: GSU ARCTIC."
   - Overview: adapt the proposal abstract.
2. **Research question and hypotheses.** The proposal's question verbatim, then H1 to H5 as in the IEEE proposal. Below them, a table mapping each hypothesis to its test and Holm family (H1 and H2 IC: all sources; H2 returns: per strategy; H3: per model; H3b: all sources; H4: exploratory, no Holm; H5 shuffle: per strategy for MVO, Robust, Top-K).
3. **Study design.**
   - Rating source vs portfolio.
   - Prompt types: bare, placebo, 4 to 5 skills files; `skills_5` disabled by default.
   - Models: default enabled set (Gemma 4 31B, MiniMax M2.7, Claude Haiku 5.5, GPT-6 Luna) and the requested small models.
   - Portfolio count formula with the default numbers (4 x 6 sources = 24, so 127 portfolios) and the largest planned configuration (7 x 7 = 49 sources, 252 portfolios).
   - Output schema.
   - The controls: each strategy's Q0, HRP, equal weight, residualization, placebo, common random numbers.
   - "Every study setting is fixed in `config/study.yaml` and frozen with the `frozen-v2` tag before the live run". Do not say pre-registered.
   - Live run only, no backtest, and why.
4. **Methodology.** Keep the current formulas (they were verified against PLAN2 and results.md): baseline, risk and prior, views and uncertainty (K1, E1), Black-Litterman with M, the strategy table, guardrails, simulator. Correct anything that changed in PLAN3.
5. **Statistics.**
   - Win/tie/loss with the 0.01 tie band.
   - Sign test, exact sign-flip permutation (primary p, adjusted by Holm), Newey-West standard errors, paired bootstrap interval.
   - Minimum detectable gap.
   - Factor regressions: Fama-French five-factor daily when `eval.fama_french_file` is set (the live study), ETF proxies only as the offline fallback.
   - Shuffled-ratings test scope (G6) and why CVaR and Resampled are excluded.
6. **Weekly pipeline.** Jobs J1 to J9 table and the ARCTIC job layout from Phase 3 (Kronos and LLM jobs in parallel, short finish jobs that poll the batches, every job under 15 hours).
7. **Failure handling.** Keep the current list; update the gateway-outage wording ("Carter's own Slurm vLLM job") and add the walltime rule.
8. **Compute.** A table with each job, where it runs, cores and memory, and expected wall time from the Phase 3 budget. Include the shuffle total from the Phase 2 projection (24 weeks, three strategies) and the local backup command `portstudy shuffle run-local --processes N`.
9. **Quick start, commands, testing.** Re-measure every timing you state (run them) or drop the number. Include `ruff format --check`.
10. **Repository layout.** Complete and current, including `optimize/strategies/`, `optimize/problem_cache.py`, `llm/health.py`, `scripts/arctic/`, and `docs/proposals/`.
11. **Project status.**
    - Built and tested (v1, v2, PLAN3).
    - Remaining manual work: PLAN2 phase 15 and the PLAN2 project completion checklist.
    - Not yet verified against live services: the UNVERIFIED list.
12. **Documentation table.** Every doc after Phase 5, with no `desktop_checklist.md` and no `arctic_gpu_request.md`.
13. **Design choices.**
    - Only the five study design choices (K1, S4, S5, G3b, P1), keeping their current text.
    - The PLAN3 decisions G6, W1 and M1, with one line of reason each.
    - Link `docs/build_decisions.md` for everything else.

**Also:** move the current README "chosen during build" list, unchanged, into a new `docs/build_decisions.md` with a one-line intro.

**Done when:** every check in Section 2 below passes for README.md.

---

## Phase 5: Docs

**Goal:** every doc matches the code, the proposals, and PLAN3.

**Build:**
1. `docs/design.md`:
   - Replace the intro with: "One section per pipeline stage, current as of PLAN3. PLAN2.md section 0.3 and PLAN3.md section 0.1 hold the decisions."
   - H5 text: shuffles for MVO, Robust and Top-K only (G6).
   - Fama-French primary, ETF fallback.
   - Add a section "ARCTIC job layout" describing Phase 3.
2. `docs/runbook.md`:
   - Section 7: projections from Phase 2 (24 weeks, three strategies).
   - Section 8: shuffle scope, shard sizing, and the local fallback.
   - ARCTIC weekend section: the new three-job layout, `finish.sbatch` polling, what to do if a job hits its walltime (rerun from the named job), how to read `sacct` for SU use.
3. `docs/config_reference.md`:
   - New keys: `eval.shuffle.strategies`, `llm.parallel_models`, and the new `arctic.env` variables.
   - `llm.gateway.base_url` value.
   - Remove the MiniMax override; fix "set by the researcher" and "(v1 phase 17)".
4. `docs/open_questions.md`: G6 row changed; W1 and M1 rows added; D3 still the only provisional study decision.
5. `docs/arctic_setup.md`:
   - The gateway URL is known; it needs the GSU network or VPN.
   - W1 and M1 explained, with the new env variables.
   - Check the real walltime limit and memory policy with ARCTIC (PLAN2 phase 15 A).
6. `docs/github_setup.md`:
   - Replace "the researcher's hand-run" with "Carter's hand-run" and "The desktop checkout" with "Carter's local checkout".
   - Fix the Actions-minutes note: the repository is public, so standard GitHub-hosted runners are free.
   - Keep the rest.
7. `docs/gitlab_setup.md`:
   - Same wording fixes.
   - `frozen-v1 (v1 phase 17 E2)` -> `frozen-v2 (PLAN2 phase 15 H)`.
   - Remove the "on the desktop" steps or reword them for any machine.
8. Replace `docs/desktop_checklist.md` with `docs/weekend_checklist.md`: an ARCTIC-based checklist with these parts:
   - before: VPN, gateway key, `arctic.env` filled, frozen tag, last backup pulled;
   - running: `submit_weekend.sh`, then `squeue`/`sacct`;
   - if something fails: point to the runbook;
   - after: summary read, backup pulled.

   Delete `desktop_checklist.md` and update every reference.
9. `docs/arctic_gpu_request.md` is untracked and outdated: leave it uncommitted and list it in PROGRESS for Carter to delete. Reference `docs/proposals/` instead.
10. Commit `docs/proposals/` (four files Carter added: the ARCTIC request and the IEEE project proposal, each as PDF and LaTeX source) in this phase. Do not edit them.
11. PROGRESS.md:
    - The v2 phase 9 entry and the "remaining" list say "about 16,600 CPU hours". Do not rewrite history; add a dated note under the PLAN3 entries that this figure used 52 weeks and all five strategies, and give the corrected figure from Phase 2.
12. PLAN2.md: add one line under the title: "Decisions G6 (changed), W1 and M1 (new) are updated in PLAN3.md." Change nothing else in PLAN2.

**Done when:** every check in Section 2 passes for every file under `docs/`.

---

## Phase 6: Final verification

**Goal:** prove the repo is consistent and green.

**Build and run:**
1. Full suite including e2e (`pytest`, then `pytest tests/e2e`), `ruff check`, `ruff format --check`, `mypy`, `scripts/refresh_manifest.py --check`, `scripts/check_placebo_length.py`, `scripts/check_banned_words.py`, `scripts/check_secrets.py`, `yamllint`.
2. The consistency checks in Section 2, run as greps, with output pasted into PROGRESS.
3. `bash scripts/arctic/submit_weekend.sh --dry-run` in both modes (with a filled test env file) and paste the printed commands into PROGRESS.
4. Push to `origin` and `gitlab`, then confirm the GitHub Actions run is green (read-only `gh run list` if available; otherwise note it for Carter).

**Done when:** everything passes and the last PROGRESS entry lists the commands and their results.

---

## 2. Consistency checks (run in Phases 4 to 6)

Each must return nothing (or the expected value) across `README.md`, `docs/` (excluding `docs/proposals/` and `docs/build_decisions.md`), `src/`, `scripts/`, `config/`:

| Check | Expected |
|---|---|
| `grep -rn -i "thesis\|quantitative finance\|pre-registered\|investor mindset"` | nothing |
| `grep -rn -i "the researcher\|the student\|phase 17\|desktop"` | nothing (except `docs/build_decisions.md` history, if quoted) |
| `grep -rn "16,600\|16,595\|15,000 CPU hour request" README.md docs` | nothing outside a clearly dated correction note |
| `grep -rn "2048" config/study.yaml` | nothing |
| `grep -n "base_url" config/study.yaml` | `https://litellm.rs.gsu.edu/v1` |
| README title | matches the proposal title |
| README research question | matches the proposal verbatim |
| README H1 to H5 | match the proposal |
| README statistics | names sign test, sign-flip permutation, Newey-West, bootstrap, Holm on `p_perm`, Fama-French five-factor |
| README shuffle scope | MVO, Robust, Top-K and their Q0s, 500 shuffles |
| every `#SBATCH` CPU job | has `--mem-per-cpu=3G` (directly or via submit flags) |
| `shuffle_array.sbatch` | `--cpus-per-task=1` |
| every walltime in `arctic.env.example` and docs | at most 15 hours |

---

## Phase 7: Manual work (Carter only, after Phases 1 to 6)

1. Ask Andalib (ARCTIC) to confirm the real maximum job length and the memory-per-core policy; if the limit differs from 24 hours, set `MAX_JOB_HOURS` to stay well under it.
2. Delete the untracked `docs/arctic_gpu_request.md`.
3. Read the new README top to bottom against the proposal.
4. The project proposal (`docs/proposals/Project_Proposal_IEEE.pdf`) already says H5 shuffles cover only the mean-variance, robust and top-K portfolios. The ARCTIC request still budgets 10,420 CPU hours for permutation tests. That is fine as headroom, so no change is needed unless ARCTIC asks.
5. Continue with PLAN2 phase 15 (accounts, ARCTIC setup, smoke tests, skills files, freeze, pilot).
