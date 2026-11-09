# PLAN2.md: Build Plan v2 for the LLM Skills and Portfolio Strategies Study

Decisions G6 (changed), W1 and M1 (new) are updated in PLAN3.md.

This file is the second build plan. PLAN.md (v1) built the full pipeline in 17 phases, and every phase is done. PLAN2.md changes that codebase to match the updated proposal. Read PLAN.md and PROGRESS.md first: everything in v1 that this file does not change stays as it is.

Study in one sentence: with the views, covariance and guardrails held fixed, test whether LLM ratings (given a bare prompt, specialized skills files, or a placebo) improve five portfolio optimization strategies compared with the same strategies run on the quant baseline alone, using a weekly pipeline on live data.

---

## 0. Working rules (read first, follow always)

### 0.1 How to work
1. Work through the phases in order. Do not start phase N+1 until every item in phase N's "Done when" list passes. The one exception is rule 10: if a phase is truly blocked, record why and continue with the next phase that does not depend on it.
2. After each phase, run the full test suite, `ruff check`, `ruff format --check` and `mypy`. Then append an entry to `PROGRESS.md` headed `v2 phase N: <title>` with: what was built, test counts, every v1 test that was changed and why, every UNVERIFIED item added, and anything skipped and why. Then commit and push under rule 3 with the message from Section 0.6.
3. **Git: the agent commits and pushes, but every commit is Carter's.** This replaces rule 0.1.2 of PLAN.md.
   - Repository: `origin` = `https://gitlab.com/cartertierney0/llm_portfolio_management.git`, branch `master` (the branch checked out now). Commit on `master` and push to `origin master`. Do not create, switch, or delete branches or tags.
   - Identity: every commit's author and committer must be exactly `ctsc <cartertierney0@gmail.com>` (the identity on every v1 commit). Before the first commit, run `git config user.name "ctsc"` and `git config user.email "cartertierney0@gmail.com"` in the repo, and set `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL`, `GIT_COMMITTER_NAME` and `GIT_COMMITTER_EMAIL` to the same values for every commit, so no tool default can replace them.
   - Message: only the subject line from Section 0.6 (plus an optional plain body describing the change). Never add `Co-authored-by`, `Signed-off-by`, `Generated with`, `Devin`, `AI`, a bot name, a session link, or any other trailer or attribution. Turn off any setting in the agent's environment that adds such trailers automatically.
   - Check before every push: `git log -1 --format='%an <%ae>|%cn <%ce>'` must print `ctsc <cartertierney0@gmail.com>|ctsc <cartertierney0@gmail.com>`, and `git log -1 --format=%B` must contain no trailer or attribution. If either check fails, fix that unpushed commit with `git commit --amend` (with the identity variables set) and check again. Never push a commit that fails the check.
   - Allowed: `git add`, `git commit`, `git commit --amend` on the latest unpushed commit, `git push origin master`, `git pull --ff-only`, and every read command.
   - Forbidden: `git push --force` or `--force-with-lease`, `rebase`, `reset --hard`, `filter-branch` or any history rewrite of pushed commits, merge commits, opening or merging pull requests or merge requests, changing remotes, and editing repository settings.
   - Commit only files the phase touched, plus `PLAN2.md` itself in the Phase 1 commit. Leave the untracked `docs/arctic_gpu_request.md` uncommitted. Never commit `data/`, `.env`, keys, or anything `scripts/check_secrets.py` flags; run that script before every commit.
   - If a push fails (auth, network), keep committing locally, retry the push at the end of the next phase, and note it in PROGRESS.md. Never stop work because of a failed push.
4. If a test fails, fix the code. Never weaken or delete a test to make it pass. A v1 test may change only where this plan changes the behavior it checks (for example a hardcoded v1 portfolio key), and the PROGRESS entry names each one.
5. Everything must run offline: synthetic data, recorded fixtures, and the stub LLM. Never call the ARCTIC gateway, a paid API, a broker, or Slurm while building. Real-service code is written against official docs and tested with mocks.
6. If an external detail is uncertain (LiteLLM, Kronos sampling, Slurm flags on ARCTIC), read the current official documentation. If it cannot be verified, implement it behind the interface, mark it `# UNVERIFIED: needs real smoke test (v2 Phase 15)`, and list it in `PROGRESS.md`. Never guess silently.
7. Keep modules small and pure. Every random call takes an explicit seed from config. Every job that writes files is idempotent: a rerun gives byte-identical files.
8. Do not delete files except those a phase lists. Do not revert or overwrite unrelated work. Do not change any `DECISION` value.
9. No temporary code, debug prints, or commented-out blocks survive the end of a phase.
10. **Work autonomously.** Carter is not available during the build. Never stop to ask for permission, approval, or input, and never wait for a reply. When something is unclear, pick the option most consistent with this plan and the existing code, write it under a `Choices made while unattended` heading in that phase's PROGRESS.md entry, and keep going. When something needs a real key, account, or ARCTIC access, implement it behind the interface with an UNVERIFIED tag and move on. The only limits are rule 3 (git), rule 8 (no deleting unrelated files, no changing `DECISION` values), and rule 5 (no real services). Run all 14 phases back to back; a phase that cannot fully pass gets its blocker written in PROGRESS.md and the next independent phase continues.

### 0.2 Style rules (unchanged from v1)
- Python 3.11, type hints everywhere, `ruff` and `mypy` clean.
- Never use em dashes in code, comments, docstrings, docs, or commit messages.
- Never use the word "arm" for experimental conditions. Say "portfolio", "prompt type", "strategy", "LLM view", or "quant baseline view".
- Comments explain intent, at most 3 lines each. Remove stale comments when code changes.
- No emoji. No generic helpers or abstraction layers with only one caller.
- ARCTIC-facing docs are written in the first person and never refer to "the researcher" or "the student".

### 0.3 Study decisions v2 (set by Carter, never changed by the agent)
Each row is a config value in `config/study.yaml` or `config/models.yaml`, tagged in code with `DECISION(<ID>)`. Rows marked **new** or **changed** differ from v1. "Design choice" rows are not in the proposal text; Carter has adopted them, and the README documents each one with its reason (Phase 14).

| ID | Setting | Value | Status |
|---|---|---|---|
| D1 (changed) | models | ARCTIC LiteLLM gateway: Gemma 4 31B, MiniMax M2.7; requested deployments: Gemma 4 E4B, Phi-4 Mini 3.8B, Llama 3.2 3B; API: one or two of Claude Haiku 5.5, GPT-6 Luna. No Qwen or DeepSeek (GSU tool registry). No local GPU. | confirmed |
| D2 (new) | gateway outage | the affected small models run in Carter's own Slurm GPU jobs on ARCTIC; otherwise that model's week is marked missed (Q0 fallback, excluded from its tests) | confirmed |
| D3 | budget | goal $50 for paid API models only; gateway models cost $0 in the cost table | provisional |
| D4 | temperature | 0 for all models | confirmed |
| B1 (changed) | prompt types | `bare`, `placebo`, and 4 to 5 specialized skills files of similar length; one placebo matched to the mean skills length | confirmed |
| B4 (changed) | universe size | 100 to 300, set after the ARCTIC allocation is known | confirmed |
| C3 (changed) | output schema | JSON `{"low": int, "likely": int, "high": int, "rationale": str}`, integers in [-5, 5], valid only if low <= likely <= high; no confidence field | confirmed |
| E1 (new) | LLM view uncertainty | width w = high - low ; c = clip(1 - w/10, 0.1, 1) ; Omega_L,ii = tau Sigma_ii / c | confirmed |
| E2 | IC used to scale the LLM view | IC of the residualized `likely` score | confirmed |
| E6 (new) | baseline view uncertainty | from the spread of Kronos's 20 sampled paths (mapping in K1) | confirmed |
| K1 (new) | Kronos spread mapping | s_i = sd over the 20 paths of the 5-day return ; r_i = clip(s_i / median(s), 0.25, 4) ; Omega_b,ii = tau Sigma_ii r_i | confirmed (design choice) |
| S1 (new) | steered strategies | MVO, Mean-CVaR, Robust MVO, Resampled MVO, Top-K; each has its own Q0 | confirmed |
| S2 (new) | forecast-free controls | HRP and equal weight; no LLM view, no Q0, no shuffle test | confirmed |
| S3 (new) | Mean-CVaR | 10,000 Monte Carlo scenarios from the BL posterior, beta = 0.95 (worst 5%) | confirmed |
| S4 (new) | Mean-CVaR risk weight lambda | 1.0 | confirmed (design choice) |
| S5 (new) | Robust MVO radius kappa | 1.0, ellipsoid from the BL posterior covariance of the mean | confirmed (design choice) |
| S6 (new) | Resampled MVO draws | 200 | confirmed |
| S7 (new) | Top-K | K = 20, equal weights | confirmed |
| G1 (new) | guardrails | identical for every strategy: 5% cap and no shorts at each rebalance, 10 bps cost, stop-loss -8% and take-profit +15% checked at every daily close, weekly rebalance | confirmed |
| G3 (changed) | H3 | each skills file vs bare and vs placebo per model, plus the range-width test H3b | confirmed |
| G3b (new) | H3b width split | per source and week, stocks with width <= the median width vs the rest; rank IC gap narrow minus wide | confirmed (design choice) |
| P1 (new) | skills length tolerance | each skills file within 5% of the mean skills token length; placebo within 2% of that mean | confirmed (design choice) |
| G4 (changed) | H4 | strategy ranking by mean weekly gap vs own Q0, plus the v1 splits (market direction, volatility, earnings week) | confirmed |
| G6 (changed) | shuffle test | 500 shuffles for every steered portfolio, including each strategy's Q0 | confirmed |
| G7 (new) | Holm families | IC tests: as v1. Portfolio return tests (H2 vs Q0, H5 shuffle): one family per strategy | confirmed |
| N2 (changed) | weekly trigger | Carter runs one submit script by hand on ARCTIC; no scheduler | confirmed |

Unchanged from v1: A2, A3, A6, B2, C1, C2, C4, C7, E3, E4, E5, G5, N1, N3, live only with no backtest, LightGBM out, Ledoit-Wolf covariance, CSV storage with dated backups.

### 0.4 Confirmed design (hardcode only as config defaults)
- A **rating source** is one (model, prompt type) pair. A **portfolio** is one (strategy, rating source) pair, one (strategy, Q0), or one of the two controls.
- Rank IC belongs to the rating source, not the portfolio. The same rating gives the same IC under every strategy.
- Black-Litterman runs once per rating source per week. Every steered strategy for that source reads the same posterior mean and posterior covariance. Q0 BL runs once per week and feeds every strategy's Q0.
- Portfolio count: (models x prompt types + 1) x 5 + 2.
- LLM calls per week: models x prompt types x stocks. Strategies never add calls.
- If a rating source fails for a week, each of its 5 portfolios trades its own strategy's Q0 weights, labeled `fallback_q0`, and that week is excluded from that source's tests.
- If the LLM view is dropped (IC_L = 0 or no valid scores), each portfolio trades exactly its own strategy's Q0 weights.
- One rating source in one week is served by exactly one endpoint (gateway or Slurm fallback). Mixed serving inside a source-week is not allowed.
- Random draws (CVaR scenarios, resampled draws) use common random numbers: the same standard normal draws per (week, strategy) for every source, its Q0, and every shuffle. Only the mean and covariance used to transform them differ. This keeps simulation noise out of every LLM vs Q0 and real vs shuffle comparison.

### 0.5 Phase template (every phase below follows it)
- **Goal:** one sentence.
- **Goals checklist:** numbered outcomes that must all be true at the end.
- **Build:** the exact changes.
- **Outputs:** each file created or changed and what it must contain (functions, classes, columns).
- **Edge cases:** each one needs code that handles it and a test.
- **Tests:** file name and cases.
- **Meta:** LOC estimate, dependencies, whether the agent can finish alone, risk, complexity.
- **Done when:** checks that must all pass, on top of rule 0.1.2.

### 0.6 Commit plan (the agent commits and pushes these under rule 3)
| Phase | Commit message |
|---|---|
| 1 | `v2 phase 1: strategy dimension in portfolio ids and config` |
| 2 | `v2 phase 2: low likely high output schema` |
| 3 | `v2 phase 3: skills manifest and placeholders` |
| 4 | `v2 phase 4: ARCTIC LiteLLM provider, remove Ollama and Qwen` |
| 5 | `v2 phase 5: Kronos path spread` |
| 6 | `v2 phase 6: width-based view uncertainty and BL posterior covariance` |
| 7 | `v2 phase 7: strategy package (MVO, CVaR, robust, resampled, Top-K, HRP, EW)` |
| 8 | `v2 phase 8: weights and replay over strategies` |
| 9 | `v2 phase 9: 300-stock scale and compute projection` |
| 10 | `v2 phase 10: per-strategy evaluation` |
| 11 | `v2 phase 11: sharded shuffle test` |
| 12 | `v2 phase 12: ARCTIC Slurm scripts` |
| 13 | `v2 phase 13: end-to-end rehearsal and CI` |
| 14 | `v2 phase 14: docs and final checklist` |

One commit per phase, made only after that phase's "Done when" list passes, then pushed. A phase left blocked (rule 10) is committed with `v2 phase N: <title> (partial, see PROGRESS)`.

---

## 1. Repository review (state at commit `030faf4`, v1 complete)

### 1.1 Architecture observations
- Package `portstudy`, Python 3.11, pandas, numpy, scipy, scikit-learn, statsmodels, cvxpy (OSQP, CLARABEL), pydantic v2 configs with `StrictModel` (extra keys rejected), typer CLI, httpx transport, tenacity retries, pytest with hypothesis.
- No web frontend, database, or auth. All state is CSV and JSON under `data/` (gitignored), written atomically through `io.write_csv_atomic` with `float_format="%.12g"`.
- Weekly flow is nine idempotent jobs in `pipeline/jobs.py`: J1 snapshot, J2 baseline, J3 API submit, J4 local infer, J5 API collect, J6 validate, J7 weights, J8 replay and score, J9 backup. `scripts/run_weekend.py` runs them in order against a lock deadline.
- Build tooling: `pyproject.toml`, `.gitlab-ci.yml` (lint, type check, tests; never runs the weekly pipeline), helper scripts under `scripts/`.

### 1.2 Relevant files and why they matter
| Area | Files | Why |
|---|---|---|
| Identity | `types.py` (`PortfolioId`, `LlmResponse`, `DecisionSource`) | Every table, folder and test keys on `PortfolioId.key`; it has no strategy field |
| Config | `config.py`, `config/study.yaml`, `config/models.yaml`, `config/prompts_manifest.yaml` | D1, B1, B4 outdated; Qwen still listed; one skills file; `universe.per_sector` unused |
| LLM | `llm/schema.py`, `llm/validate.py`, `llm/runner.py` (`SCORE_COLUMNS`), `llm/providers/*`, `prompts/base_instructions.md` | Score plus confidence schema; no OpenAI-compatible provider; Ollama is sequential |
| Baseline | `baseline/kronos.py`, `baseline/combine.py` | Kronos returns only the mean path return `k`, no spread |
| Views | `views/build.py`, `views/residualize.py` | Omega_b = tau Sigma_ii; Omega_L uses the confidence field |
| Optimize | `optimize/black_litterman.py`, `optimize/mean_variance.py`, `optimize/__init__.py` | BL has no posterior covariance; MV is hardwired in `_optimize` |
| Pipeline | `pipeline/jobs.py` (J7, `build_plans`, `update_ic_history`), `pipeline/context.py` (`portfolio_keys`), `pipeline/fallback.py` | Loops are model x config; fallback copies one Q0 |
| Eval | `eval/hypotheses.py`, `eval/shuffle.py`, `eval/report.py`, `eval/data.py`, `eval/factors.py` | Families, shuffle, figures and tables assume 13 MV portfolios |
| Universe | `data/universe.py` | Size from config; at 300 some sectors cannot fill floor(300/11) = 27 slots |
| Docs | `README.md`, `docs/*.md` | Describe the v1 scope, the desktop GPU and Ollama |

### 1.3 Coding conventions to follow
- One module per concept, pure functions taking `cfg` sections, frozen dataclasses for records, pydantic `StrictModel` for config.
- `DECISION(<ID>)` tags beside every decision value use.
- Module-level exception classes (`OptimizerError`, `RiskModelError`, `LlmProviderError`, `CostCapExceeded`).
- Structured logging through `logging_setup` with `extra={"extra_fields": {...}}`.
- Every output CSV has a fixed column list constant (`WEIGHT_COLUMNS`, `SCORE_COLUMNS`, `BASELINE_COLUMNS`).
- v1 tests are `tests/unit/test_phaseN_<area>.py`; new v2 tests go in `tests/unit/test_v2_phaseN_<area>.py`.

### 1.4 Existing code to reuse
- `solve_mean_variance` stays the MVO strategy and is reused inside Resampled MVO.
- `black_litterman` keeps its linear solve; posterior covariance is added beside it.
- `views.build_views` keeps its shape; only the two Omega inputs change.
- `LlmTransport` / `HttpxTransport` and the non-batch provider path are the template for the LiteLLM provider.
- `eval/tests.py` (Newey-West, sign test, sign-flip permutation, Holm, bootstrap) needs no change.
- `eval/shuffle.py` already has a per-portfolio cache directory, which becomes the shard output target.
- The simulator (`sim/`) is strategy independent and needs no logic change.

### 1.5 Potential conflicts and how this plan handles them
- **Portfolio key format.** Changing `PortfolioId.key` breaks every stored path. Phase 1 moves to the v2 key format with one parser. No live data exists, so there is nothing to convert.
- **Runtime growth.** v1 e2e takes 6 min 23 s for 13 portfolios; v2 has up to 252. A small test config keeps CI bounded, and compiled cvxpy problems are reused (Phase 9).
- **Solver types.** OSQP solves only QPs. Robust MVO is an SOCP and Mean-CVaR is an LP, so each strategy has its own solver list.
- **Kronos API.** It is not verified whether `KronosPredictor.predict` exposes per-path samples. Phase 5 handles both cases behind the interface.
- **Reasoning output from MiniMax M2.7.** It may return thinking text inline or in a separate field and needs a larger output budget. Phases 2 and 4 handle both.
- **Gateway vs self-hosted fallback.** A model served from a Slurm job may differ in precision or chat template. Every call logs its endpoint and returned model version, and fallback-served weeks are flagged in the report.
- **Shuffle compute.** At full scale the shuffle is about 125,000 portfolio replays. Phase 9 projects the CPU hours against the ARCTIC request before the freeze, so any gap shows up early.

### 1.6 Missing context (each is answered in Phase 15)
- Gateway model IDs for the three requested small models (not deployed yet).
- Gateway rate limits, JSON mode support, whether `seed` is honored, and how MiniMax returns reasoning.
- Whether ARCTIC compute nodes reach the internet and the gateway, or only login nodes do.
- ARCTIC Slurm partitions, account string, GPU flags, walltime caps, and inode or file-count quotas.
- Final names of the 4 to 5 skills files.

---

## 2. Overall summary

v1 built one optimizer (mean-variance) fed by score-plus-confidence ratings from local Ollama models. v2 adds a strategy dimension (five steered strategies, each with its own Q0, plus HRP and equal-weight controls), switches the LLM output to a low / likely / high range whose width sets the view uncertainty, uses the Kronos path spread for the baseline view uncertainty, and moves open-weight inference to the ARCTIC LiteLLM gateway with Slurm jobs for Kronos, the fallback, and the shuffle test. The approach extends existing modules in place: BL gains a posterior covariance, `optimize/` gains a strategy package, and the job, eval and shuffle loops gain a strategy axis.

**Estimated total LOC:** approximately 4,500 to 6,500 lines: about 2,900 to 4,100 added source and test lines, 900 to 1,400 modified, 150 to 300 deleted (Ollama provider and its tests, Qwen config), and 600 to 900 lines of docs.

### 2.1 Assumptions
- No live or pilot data exists yet, so storage formats can change without migrating real results.
- The gateway is OpenAI-compatible at `https://litellm.rs.gsu.edu/v1` with a bearer API key, reachable only from the GSU network or VPN, and is in alpha.
- The paid API providers keep their v1 batch endpoints; their adapters change only for the schema.
- The v1 synthetic world can test every strategy once it has a Kronos-like spread and range outputs from the stub.

### 2.2 Files likely touched
```
config/            study.yaml, models.yaml, prompts_manifest.yaml, study_test.yaml (new)
prompts/           base_instructions.md, skills/ (5 placeholders), placebo/
src/portstudy/
  types.py  config.py
  llm/        schema.py validate.py runner.py prompts.py cost.py providers/ (litellm.py new, ollama.py removed, stub.py, factory.py)
  baseline/   kronos.py combine.py
  views/      build.py residualize.py
  optimize/   black_litterman.py mean_variance.py __init__.py strategies/ (new package)
  pipeline/   context.py jobs.py fallback.py status.py cli.py
  eval/       data.py hypotheses.py shuffle.py report.py factors.py
  data/       universe.py
scripts/      run_weekend.py estimate_cost.py check_placebo_length.py check_secrets.py smoke_real_services.py
              benchmark_pipeline.py check_banned_words.py arctic/ (new)
tests/        unit/test_v2_*  integration/  e2e/
docs/         README.md design.md runbook.md config_reference.md open_questions.md arctic_setup.md (new)
```

---

## 3. Target changes at a glance

```
optimize/
  black_litterman.py      + Posterior.cov_mu (posterior covariance of the mean, M)
  mean_variance.py        MVO, plus a cached parameterized problem
  strategies/
    __init__.py           STRATEGIES, CONTROLS, run_strategy, run_control
    common.py             StrategyInputs, StrategyResult, check_weights, cap_and_renormalize,
                          clean_weights, solve_with, fallback_chain, standard_normals
    mvo.py  cvar.py  robust.py  resampled.py  topk.py  hrp.py  equal_weight.py
llm/providers/litellm.py  OpenAI-compatible chat completions with bounded concurrency
scripts/arctic/           arctic.env.example, kronos.sbatch, weekend.sbatch, shuffle_array.sbatch,
                          llm_fallback.sbatch, submit_weekend.sh
```

Portfolio key v2:
```
<strategy>__<model>__<prompt_type>    e.g. mvo__gemma-4-31b__skills_1
<strategy>__Q0                        e.g. cvar__Q0
HRP, EW                               the two controls
```

Phase order and dependencies:
```
1 identity -> 2 schema -> 3 prompt manifest -> 4 LiteLLM
1 -> 5 Kronos spread ;  2 + 5 -> 6 views and BL -> 7 strategies -> 8 weights job -> 9 scale
8 -> 10 eval -> 11 shuffle ;  4 + 8 + 11 -> 12 ARCTIC ;  all -> 13 e2e -> 14 docs -> 15 manual
```

---

## Phase 1: Strategy dimension, identity, and config

**Goal:** every part of the code names a portfolio by strategy and rating source, and config holds the v2 decisions.

**Goals checklist:**
1. `PortfolioId` has a strategy; `RatingSource` exists and is used wherever v1 meant "model x config".
2. `config/study.yaml` has the `strategies` section and the Section 0.3 decision table.
3. A test config can be selected without editing `study.yaml`.
4. The pipeline runs end to end with MVO as the only enabled strategy, writing v2 keys.

**Build:**
1. `types.py`:
   - `RatingSource(model: str, config_id: str)`, frozen, `key = f"{model}__{config_id}"`, `parse(key)`.
   - `PortfolioId(strategy: str, model: str, config_id: str)` with `steered(strategy, source)`, `q0(strategy)`, `control(name)`, `parse(key)`, and properties `key`, `is_q0`, `is_control`, `source -> RatingSource | None`.
   - `StrategyName = Literal["mvo", "cvar", "robust", "resampled", "topk"]`, `ControlName = Literal["HRP", "EW"]`.
   - `DecisionSource` adds `"control"`.
2. `config.py`:
   - `StrategiesConfig(enabled: list[StrategyName], controls: list[ControlName], cvar: CvarConfig, robust: RobustConfig, resampled: ResampledConfig, topk: TopKConfig, hrp: HrpConfig)`.
   - `CvarConfig(n_scenarios=10000, beta=0.95, lam=1.0, seed, solvers=["CLARABEL", "HIGHS"])`, `RobustConfig(kappa=1.0, solvers=["CLARABEL"])`, `ResampledConfig(n_draws=200, seed, max_failed_share=0.10)`, `TopKConfig(k=20)`, `HrpConfig(linkage="single")`.
   - Validators: `1 / topk.k <= optimizer.max_weight`, and `optimizer.min_stocks >= ceil(1 / max_weight)`.
   - Remove `UniverseConfig.per_sector` (unused by `build_universe`).
3. Config selection: if the loader has no override yet, add env var `PORTSTUDY_CONFIG` (path to a study YAML) read by the loader, plus a `--config` option on the CLI and `run_weekend.py`. Default stays `config/study.yaml`.
4. `config/study.yaml`: `strategies` section with `enabled: [mvo]` and `controls: []` for this phase (Phase 7 enables the rest); `decisions` list replaced by the Section 0.3 table; `per_sector` removed.
5. `config/study_test.yaml` (new): 2 stub models, the v1 three prompt types (Phase 3 switches it to `bare`, `placebo`, `skills_1`, `skills_2`), 2,000 CVaR scenarios, 20 resampled draws, 20 shuffles, 120 synthetic tickers.
6. `pipeline/context.py`: `sources()`, `steered_keys()`, `q0_keys()`, `control_keys()`, `portfolio_keys()` (fixed order: Q0s by strategy, controls, steered by strategy then source). Replace every v1 call of `portfolio_keys()` that meant rating source with `sources()`.
7. `pipeline/jobs.py` and `optimize/__init__.py`: minimal rename so files are written under v2 keys (`mvo__Q0`, `mvo__<source>`). No logic change.

**Outputs:**
| File | Must contain |
|---|---|
| `types.py` | `RatingSource`, `PortfolioId` v2, `StrategyName`, `ControlName`, `DecisionSource` with `control` |
| `config.py` | `StrategiesConfig` and sub-configs with validators; `per_sector` gone; config path override |
| `config/study.yaml`, `config/study_test.yaml` | as above |
| `pipeline/context.py` | the five key and source methods |

**Edge cases:** model IDs with dots and hyphens (`claude-haiku-5.5`); a model ID containing `__` (rejected at registry load); unknown strategy in YAML (validation error); `Q0`, `HRP` or `EW` used as a model ID (rejected).

**Tests (`test_v2_phase1_identity.py`):** round trip `parse(key).key == key` for all three key forms; `parse` rejects malformed keys; `portfolio_keys()` length equals (sources + 1) x strategies + controls; registry rejects `__` and reserved names in IDs; validators fire on K = 10 with a 5% cap; `PORTSTUDY_CONFIG` selects the test config.

**Meta:** LOC 300 to 450. Depends on nothing. Agent can finish alone: yes. Risk: medium (touches every key). Complexity: 2/5.

**Done when:** the full suite passes with MVO only; weight files on disk use v2 keys; PROGRESS lists every v1 test changed for key names.

---

## Phase 2: Range output schema (low, likely, high)

**Goal:** LLM outputs are ranges, validated strictly, and every reader uses `likely` and the width.

**Goals checklist:**
1. `parse_response` accepts only valid ordered integer triples plus a rationale.
2. Stored scores have `low, likely, high, width` and no `score` or `confidence`.
3. The stub provider emits ranges whose width carries information, so H3b has a planted effect in the synthetic world.
4. Residualization, the shuffle's permutation, and every v1 test run on the new columns.

**Build:**
1. `llm/schema.py`: `RANGE_MIN = -5`, `RANGE_MAX = 5`; remove `FALLBACK_CONFIDENCE`; `fallback_response()` returns `low = likely = high = 0`, `valid = False`.
2. `types.LlmResponse`: `low`, `likely`, `high`, `rationale`, `valid`, `raw`, `error`, `fallback_flag`, `extra_keys`, `had_thinking`.
3. `llm/validate.py` `parse_response(raw, rationale_token_cap=150, stop_reason=None, reasoning_text=None)`:
   - Strip code fences (v1). Strip one leading `<think>...</think>` block and set `had_thinking`. An opening `<think>` with no closing tag is `invalid_truncated` if `stop_reason == "max_tokens"`, else `unclosed_thinking`.
   - Empty content after stripping: `empty` (or `invalid_truncated` when truncated).
   - Parse the first JSON object. Required keys `low`, `likely`, `high`, `rationale`. Extra keys are ignored, listed in `extra_keys`, and logged.
   - `low`, `likely`, `high`: a Python `int` that is not a `bool`, or a float equal to an integer (`3.0`); strings such as `"3"` and floats such as `2.5` are `not_integer`.
   - Range check (`out_of_range`), then order check (`bad_order`).
   - `rationale` must be a string; truncated to the C7 cap as in v1; an empty string is valid and logged.
4. `llm/runner.py`: `SCORE_COLUMNS` replaces `score, confidence` with `low, likely, high, width`, and adds `extra_keys`, `had_thinking`, `endpoint`. `width = high - low` for valid rows, empty otherwise.
5. `prompts/base_instructions.md`: the agent rewrites the JSON instruction for the new schema: give the lowest plausible, most likely, and highest plausible rating for next week on the -5 to 5 scale, integers only, low <= likely <= high, then a short rationale, JSON only. Carter's review is a Phase 15 F task. Run `scripts/refresh_manifest.py`.
6. `llm/providers/stub.py`: `likely` from the v1 signal plus noise; half-width drawn so that, within a prompt type, stocks with less noise get narrower ranges, and lower-noise prompt types get narrower ranges on average. Output is clipped to [-5, 5] and ordered.
7. `views/residualize.py`: `valid_scores` returns `ticker, low, likely, high, width`; `residualize` regresses `likely` on z_b.
8. `eval/shuffle.py`: permute `low, likely, high, width` together in place of `score, confidence` (the strategy rework is Phase 11).
9. Batch adapters (`anthropic_batch.py`, `openai_batch.py`): no format change beyond calling the new `parse_response`.

**Outputs:**
| File | Must contain |
|---|---|
| `llm/schema.py` | range constants, new `fallback_response` |
| `llm/validate.py` | `parse_response` with error codes `out_of_range`, `bad_order`, `not_integer`, `missing_key`, `not_json`, `empty`, `unclosed_thinking`, `invalid_truncated` |
| `llm/runner.py` | new `SCORE_COLUMNS` |
| `llm/providers/stub.py` | range output with informative width |
| `views/residualize.py`, `eval/shuffle.py` | new columns |
| `prompts/base_instructions.md`, `config/prompts_manifest.yaml` | new text, refreshed hash |

**Edge cases:** `true` as a value; `2.5`; `"3"`; `-0`; JSON-like text inside the rationale; two JSON objects in one reply (first wins, logged); JSON inside a thinking block (only the content after the block is parsed); every output invalid for a source (no LLM view, Q0 weights).

**Tests (`test_v2_phase2_schema.py`, edits to `test_phase6_llm.py`, `test_phase7_views.py`):** each valid and invalid case above with its error code; hypothesis property: every ordered in-range triple is valid and every unordered one is `bad_order`; stub determinism and narrower mean width for lower-noise prompt types; residualize ignores invalid rows.

**Meta:** LOC 350 to 500. Depends on Phase 1. Agent can finish alone: yes. Risk: medium. Complexity: 2/5.

**Done when:** `grep -rn "confidence" src/` returns only `ViewsConfig.confidence_clip` and `views/build.py` (both renamed in Phase 6); the suite passes.

---

## Phase 3: Prompt manifest and skills placeholders

**Goal:** the manifest holds `bare`, `placebo`, and `skills_1` to `skills_5`, and the length checks enforce B1 and P1.

**Goals checklist:**
1. Five skills placeholders exist; `skills_5` is disabled by default.
2. The placebo is checked against the mean skills length, and skills files against each other.
3. Prompt types are discovered from the manifest `kind`, never from hardcoded names.

**Build:**
1. `prompts/skills/skills_1_PLACEHOLDER_v0.md` to `skills_5_PLACEHOLDER_v0.md`: neutral placeholder text of equal length. Remove `skills_PLACEHOLDER_v0.md`.
2. `config/prompts_manifest.yaml`: entries `bare` (kind `bare`), `placebo` (kind `placebo`, `matches: skills_mean`), `skills_1` to `skills_5` (kind `skills`), each with sha256 and hashed version. Remove the old `skills` entry.
3. `llm/prompts.py`: `PromptManifest.by_kind(kind) -> list[PromptConfiguration]`; `matches` accepts `skills_mean`.
4. `config.py` `LlmConfig`: `skills_length_tolerance: float = 0.05` tagged `DECISION(P1)`.
5. `scripts/check_placebo_length.py`: per model tokenizer, target = mean token count of enabled skills files; each skills file within 5% of the target; placebo within the existing 2% of the target; any file over C4 (6000 tokens) fails.
6. `llm/providers/stub.py`: noise levels per prompt type: `bare` and `placebo` equal (no planted gap between them), `skills_1` to `skills_5` lower by different amounts.
7. `config/study_test.yaml`: prompt types `bare`, `placebo`, `skills_1`, `skills_2`.

**Outputs:** the placeholder files, manifest, `by_kind`, the updated check script, the stub noise table.

**Edge cases:** only one skills file enabled (mean equals that file; the cross-skills check passes trivially); zero skills enabled (check fails with a clear message); a placebo `matches` an unknown ID (fails, as v1).

**Tests (updates to `test_phase5*`, new `test_v2_phase3_manifest.py`):** mean-length rule; cross-skills tolerance pass and fail; `by_kind` ordering is stable; disabled `skills_5` is excluded from sources.

**Meta:** LOC 120 to 220. Depends on Phase 2. Agent can finish alone: yes (the real skills text is a Phase 15 G task). Risk: low. Complexity: 1/5.

**Done when:** both length checks pass on the placeholders, and `ctx.sources()` under the test config returns 2 models x 4 prompt types.

---

## Phase 4: ARCTIC LiteLLM provider and model registry

**Goal:** open-weight models are called through the gateway or a self-hosted OpenAI-compatible fallback with one provider, and the registry matches D1.

**Goals checklist:**
1. `LiteLLMProvider` sends and parses OpenAI chat completions with retries and bounded concurrency.
2. The registry lists the D1 models, no Qwen, and gateway models priced at $0.
3. Ollama code and tests are removed; J4 runs open-weight models through the new provider.
4. A source-week is served by one endpoint, and switching to the fallback reruns the whole source.

**Build:**
1. `llm/providers/litellm.py` `LiteLLMProvider(model_id, provider_model, base_url, api_key_env, transport=None, timeout=180)`:
   - `POST {base_url}/chat/completions`, header `Authorization: Bearer <key>`. Gateway key from `ARCTIC_LITELLM_API_KEY`, fallback key from `ARCTIC_FALLBACK_API_KEY` (may be empty).
   - Payload: `model`, system message = shared prefix, user message = stock part, `temperature`, `max_tokens`, `seed` when `settings.send_seed`, `response_format={"type": "json_object"}` when `settings.json_mode`. Both default true and are `# UNVERIFIED`.
   - Parse `choices[0].message.content`; pass `choices[0].message.reasoning_content` (if present) to `parse_response` as `reasoning_text`, for logging only; `usage.prompt_tokens`, `usage.completion_tokens`; `model` as `model_version`; `finish_reason == "length"` to `stop_reason = "max_tokens"`.
   - If the server rejects `response_format` or `seed` with a 400 naming that field, retry once without it, log a warning, and keep it off for the rest of the run (stored in the run status so the whole week is consistent).
   - Retries: tenacity with exponential backoff on 429, 5xx and timeouts, honoring `Retry-After`. A 401 or 403 raises at once with a message naming the GSU VPN requirement.
2. `llm/runner.py` `run_open_weight(model) -> RunSummary`: a thread pool of `llm.concurrency[model]` workers (default 4). Workers only call the provider; the main thread validates and appends records, so the resume file stays consistent. No new calls start after the J6 cutoff (lock deadline minus margin). The cache key adds the endpoint name.
3. Single-endpoint rule: `RunStatus` stores `endpoint` per source-week. When `--gateway-fallback` switches a source to the fallback, its gateway records for that week move to `llm/<week>/archive/<source>.gateway.jsonl` and the source reruns from the start on the fallback.
4. `config.py`: `ModelSpec.kind: Literal["open_weight", "api"]`, `serving: Literal["gateway", "slurm_fallback"] = "gateway"`, `fallback_eligible: bool = False`. `LlmSettings` adds `send_seed`, `json_mode`; removes the Ollama-only `think`. `LlmConfig` adds `concurrency: dict[str, int]` and `gateway: GatewayConfig(base_url, fallback_endpoint_file="data/runtime/fallback_endpoint.json")`.
5. `config/models.yaml`:
   - `gemma-4-31b`: gateway, `provider_model: google/gemma-4-31B-it`, enabled.
   - `minimax-m2.7`: gateway, `provider_model: MiniMaxAI/MiniMax-M2.7`, enabled, `max_output_tokens: 2048` in `llm.settings` (confirmed in Phase 15 E).
   - `gemma-4-e4b`, `phi-4-mini`, `llama-3.2-3b`: gateway, `provider_model: TBD`, `enabled: false`, `fallback_eligible: true`.
   - `claude-haiku-5.5`, `gpt-6-luna`: `kind: api`, otherwise unchanged.
   - Remove `qwen3.5-4b` and the Ollama `llama3.2-3b` entry. Gateway models get price 0. Stub models: `stub-ow-a`, `stub-ow-b` (open weight), `stub-api-a`, `stub-api-b`.
   - A model with `enabled: true` and `provider_model: TBD` fails registry load.
6. `llm/providers/factory.py`: `litellm` maps to the new provider with `base_url` chosen by `serving` (fallback URL read from the endpoint file); remove the `ollama` branch. Delete `llm/providers/ollama.py` and its tests.
7. `pipeline/jobs.py`: `J4_local_infer` becomes `J4_open_weight_infer`; the job ID stays `J4`.
8. `scripts/check_secrets.py`: add both env var names. `scripts/estimate_cost.py`: gateway calls and tokens at $0, API cost as before, using the new call count (models x prompt types x stocks). `scripts/smoke_real_services.py`: one tiny gateway call per enabled gateway model; `--dry-run` prints payloads only.

**Outputs:** `litellm.py`, updated `runner.py`, `factory.py`, `config.py`, `models.yaml`, the `study.yaml` gateway section, the scripts; `ollama.py` deleted.

**Edge cases:** a 429 storm past the retry limit (calls left unfinished, J6 decides failure by N1); the gateway returns a different `model` string than requested (logged; a change mid-study warns, as v1 does for API models); empty `choices`; `content` null with reasoning only; fallback endpoint file missing or older than the current run (fails with a clear message, never silently uses the gateway); crash mid-run (resume skips finished records).

**Tests (`test_v2_phase4_litellm.py`):** payload shape from a recorded fixture; bearer header; flags omit `seed` and `response_format`; a 400 on `response_format` retries without it; 429 with `Retry-After` then 200; 401 raises with the VPN hint; parsing of content, usage, `finish_reason: length` and `reasoning_content`; 50 stub-transport calls with 4 workers give the same sorted records as 1 worker; a crash after 20 calls resumes with no duplicates; switching a source to the fallback archives its gateway records and reruns it; registry rejects enabled `TBD` models and any ID containing `qwen`.

**Meta:** LOC 400 to 550 (about 150 deleted). Depends on Phase 2. Agent can finish alone: yes (small-model IDs are entered in Phase 15 C). Risk: medium (gateway behavior unverified). Complexity: 3/5.

**Done when:** the suite passes with stubs; `smoke_real_services.py --dry-run` prints the planned gateway payloads; `grep -rni "ollama\|qwen" src/ config/` returns nothing.

---

## Phase 5: Kronos path spread

**Goal:** the baseline stores the spread of Kronos's 20 paths per stock (E6).

**Goals checklist:**
1. The forecaster returns `k` and `k_sd` per stock.
2. `baseline.csv` has `k_sd`; z_b is unchanged.
3. Both Kronos code paths (per-path samples, or repeated single-sample calls) are implemented and tested with mocks.

**Build:**
1. `baseline/kronos.py`: `ForecastProvider.forecast(ohlcv_by_ticker, horizon, n_samples, seed) -> pd.DataFrame` with index ticker and columns `k`, `k_sd`. Renamed from `forecast_return`; update its one caller.
   - `KronosForecaster`: if the vendored predictor can return each sampled path, compute `k` and `k_sd` (ddof 1) from them. Otherwise call it `n_samples` times with `sample_count=1` and seeds `seed + j`. Which path ran is logged once per run and tagged `# UNVERIFIED`.
   - `StubForecaster`: `k` as v1; `k_sd` from a seeded lognormal per ticker.
2. `baseline/combine.py`: `BASELINE_COLUMNS` adds `k_sd`; `combine` joins it.
3. `optimize/__init__.py`: `WeekInputs.k_sd() -> pd.Series`.

**Outputs:** `forecast` on both forecasters, `BASELINE_COLUMNS` with `k_sd`, `WeekInputs.k_sd`.

**Edge cases:** `k_sd` NaN for a stock (kept as NaN, handled in Phase 6); fewer than 2 valid paths (NaN); a path with a non-positive close (that path dropped, logged).

**Tests (`test_v2_phase5_kronos.py`):** stub determinism; a mocked predictor exercising both paths gives the same `k` and `k_sd` for the same samples; z_b equal to v1 on the same inputs.

**Meta:** LOC 150 to 250. Depends on Phase 1. Agent can finish alone: yes. Risk: medium (Kronos API). Complexity: 2/5.

**Done when:** every synthetic week's `baseline.csv` has `k_sd`, and z_b matches v1 byte for byte.

---

## Phase 6: View uncertainty and BL posterior covariance

**Goal:** Omega_b follows K1, Omega_L follows E1, and BL returns the posterior covariance of the mean, M.

**Goals checklist:**
1. Omega_b uses the Kronos spread ratio; Omega_L uses the width.
2. `Posterior` carries `cov_mu` (M), symmetric positive definite.
3. No `confidence` name remains in `src/`.

**Build:**
1. `views/build.py` `build_views(pi, sigma_vec, z_b, ic_b, cfg, k_sd=None, z_l=None, width=None, ic_l=None, ...)`:
   - Baseline: `ratio = clip(k_sd / median(k_sd), 0.25, 4)`; `Omega_b,ii = tau Sigma_ii ratio_i`. Config `views.kronos_spread_clip: [0.25, 4.0]`, `DECISION(K1)`.
   - LLM: `c_i = clip(1 - width_i / 10, 0.1, 1)`; `Omega_L,ii = tau Sigma_ii / c_i`. Rename `confidence_clip` to `width_conf_clip`, `DECISION(E1)`.
   - `views_frame` writes `k_sd`, `ratio`, `width`, `c`.
2. `optimize/black_litterman.py`:
   - `Posterior(mu_bl, tilt, n_views, cov_mu)`.
   - A = inv(tau Sigma) + P' inv(Omega) P, built once; Cholesky of A; `mu_bl` by two triangular solves; `cov_mu` = A^-1 from the same factor (solve against I), then symmetrized.
   - Q0: P = I, Omega = Omega_b.
3. `optimize/__init__.py`: `WEIGHT_COLUMNS` replaces `confidence` with `width`, `c`, and adds `k_sd`, `ratio`.

**Outputs:** the updated `build_views`, `Posterior.cov_mu`, column constants.

**Edge cases:** `median(k_sd) <= 1e-12` or all NaN (ratio 1 for every stock, logged); NaN `k_sd` for one stock (ratio 1 for that stock, logged); Cholesky of A fails (add 1e-12 x trace(A)/N to the diagonal once and log, else raise `OptimizerError`); width outside [0, 10] (cannot happen after validation; assert).

**Tests (`test_v2_phase6_views_bl.py`):** equal spreads give Omega_b = tau Sigma_ii, so Q0 equals v1 Q0; a stock at twice the median spread gets twice the baseline Omega; clipping at both ends; c values for widths 0, 4 and 10; a narrow range moves mu_BL closer to q_L than a wide one; M is symmetric positive definite; with no views M = tau Sigma; adding views never raises a diagonal entry of M; M and mu_BL match a textbook explicit-inverse implementation within 1e-10 on a random 30-stock case.

**Meta:** LOC 200 to 300. Depends on Phases 2 and 5. Agent can finish alone: yes. Risk: low. Complexity: 2/5.

**Done when:** the suite passes and `grep -rn "confidence" src/` returns nothing.

---

## Phase 7: Strategy package

**Goal:** five steered strategies and two controls behind one interface, all obeying the G1 guardrails.

**Goals checklist:**
1. Each strategy and control returns weights that pass `check_weights`, or a flagged fallback.
2. Random draws use common random numbers (Section 0.4).
3. Solver failure follows one fallback chain for every strategy.
4. The view-shift diagnostic exists for every steered portfolio.
5. `study.yaml` enables all five strategies and both controls.

**Build (formulas also in the Appendix):**
1. `common.py`:
   - `StrategyInputs(week_id, tickers, mu, sigma, cov_mu, delta, previous: pd.Series | None, q0_weights: pd.Series | None)`.
   - `StrategyResult(weights, solver, objective, fallback_flag, diagnostics: dict)`.
   - `check_weights(w, cfg)`: sum 1 within 1e-8, 0 <= w <= max_weight + 1e-9, at least `min_stocks` tickers in the universe.
   - `clean_weights(w)`: entries in [-1e-9, 0) set to 0, renormalize, then `check_weights`.
   - `cap_and_renormalize(w, cap)`: water-filling; names above the cap are set to it and the excess spreads over uncapped names in proportion to their weights; repeat until none exceed.
   - `solve_with(problem, solvers)`: tries each installed solver in order; accepts only status `optimal`; `optimal_inaccurate` moves to the next solver.
   - `fallback_chain(...)`: on failure use (a) this portfolio's previous weights, reindexed to this week's tickers, dropped names removed, renormalized and capped; else (b) this week's same-strategy Q0 weights; else (c) equal weight. `fallback_flag` records which (`solver_failed_prev`, `solver_failed_q0`, `solver_failed_ew`).
   - `standard_normals(week_id, strategy, shape, seed) -> np.ndarray`: seeded by SHA-256 of `(seed, week_id, strategy)`; shared by every source, Q0 and shuffle.
2. `mvo.py`: wraps `solve_mean_variance`; failure goes through `fallback_chain`.
3. `cvar.py` (S3, S4):
   - Scenarios R = mu' + Z chol(Sigma + M)', with Z = `standard_normals(week, "cvar", (S, N))`.
   - Rockafellar-Uryasev: maximize w' mu - lam (zeta + (1 / ((1 - beta) S)) sum_s u_s) subject to u_s >= -R_s w - zeta, u_s >= 0, sum w = 1, 0 <= w <= cap.
   - Solvers `CLARABEL`, then `HIGHS` if installed (`highspy` as an optional extra).
   - Diagnostics: in-sample CVaR, return term, CVaR term.
4. `robust.py` (S5): maximize w' mu - kappa ||L' w||_2 - (delta / 2) w' Sigma w, with L = chol(M); solver `CLARABEL`. Diagnostics: the three terms.
5. `resampled.py` (S6): mu_d = mu + chol(M) z_d, with Z = `standard_normals(week, "resampled", (D, N))`; MVO for each draw; average the successful solutions; `clean_weights`. If more than 10% of draws fail, use `fallback_chain`. Diagnostics: failed draws, weight dispersion.
6. `topk.py` (S7): sort by mu descending, ties by ticker; the top K get 1/K.
7. `hrp.py` (S2): correlation from Sigma; distance sqrt((1 - rho) / 2); `scipy.cluster.hierarchy.linkage` (single); quasi-diagonal order; recursive bisection with inverse-variance cluster weights; then `cap_and_renormalize`. Never reads mu.
8. `equal_weight.py` (S2): 1/N over the week's investable tickers.
9. `__init__.py`: `STRATEGIES`, `CONTROLS`, `run_strategy(name, inputs, cfg) -> StrategyResult`, `run_control(name, inputs, cfg) -> StrategyResult`.
10. `view_shift(w, w_q0) = 0.5 * sum |w - w_q0|`, stored in each steered portfolio's meta (Phase 8 fills it once both weights exist). It shows when a strategy's risk term outweighs the views (Section 8).
11. `config/study.yaml`: `strategies.enabled` lists all five; `controls: [HRP, EW]`.

**Outputs:** the nine modules above with these functions; `pyproject.toml` optional extra `highspy`.

**Edge cases:** fewer than `min_stocks` tickers (config error before solving); Sigma + M not positive definite in floating point (one diagonal jitter, then the fallback chain); every CVaR scenario identical (the LP still solves); Top-K ties at the K boundary; HRP with two perfectly correlated stocks; previous weights containing a dropped ticker; first live week with no previous weights.

**Tests (`test_v2_phase7_strategies.py`):**
- Every strategy and control passes `check_weights` on 10 seeded cases each at N = 25, 100 and 300.
- Byte-identical weights for identical inputs.
- MVO through the registry equals v1 `solve_mean_variance`.
- CVaR: with lam = 0 the cap goes to the top mu names; larger lam never raises in-sample CVaR.
- Robust: kappa = 0 equals MVO; larger kappa never raises w' mu.
- Resampled: as M goes to 0 it converges to MVO.
- Top-K: exactly 20 names at 0.05; boundary ties broken by ticker.
- HRP: two equal uncorrelated blocks get about half each before capping; output unchanged when mu changes.
- `cap_and_renormalize`: property test for sum, cap, and order preservation.
- Common random numbers: two sources in the same week and strategy receive the same Z.
- Fallback chain: forced failures walk prev, then Q0, then EW, with the right flag each time.

**Meta:** LOC 800 to 1,100. Depends on Phase 6. Agent can finish alone: yes. Risk: medium (solver behavior at scale). Complexity: 4/5.

**Done when:** all strategy tests pass, and one 300-stock CVaR solve with 10,000 scenarios finishes, with its time recorded in PROGRESS.md.

---

## Phase 8: Weights job, fallback, and replay over strategies

**Goal:** J7 writes weights for every portfolio in Section 3, J8 replays all of them, and the fallback rules work per strategy.

**Goals checklist:**
1. The weight file count per week equals the Section 0.4 count.
2. BL runs once per source and once for Q0 per week.
3. A failed source's five portfolios hold their strategies' Q0 weights, labeled `fallback_q0`.
4. IC history stays one row per source and signal per week.
5. Reruns are byte-identical.

**Build:**
1. `optimize/__init__.py`:
   - `posterior_for_q0(inputs) -> Posterior`, `posterior_for_source(inputs, source) -> Posterior | None` (None when the LLM view is dropped).
   - `build_q0_weights(week_id, strategy)`, `build_weights(week_id, strategy, source)`, `build_control_weights(week_id, name)`.
   - `_save` writes `weights/<portfolio_key>.csv` and `.meta.json` with `strategy`, `source`, `solver`, `fallback_flag`, `use_llm_view`, `decision_source`, `endpoint`, `view_shift`, and strategy diagnostics.
   - `posterior/<source_key>.csv` (and `posterior/Q0.csv`): ticker, pi, mu_bl, tilt, z_b, z_L, k_sd, ratio, width, c, has_llm_view. M is not stored (it is recomputed from stored inputs for audits and shuffles); its SHA-256 goes in the meta file.
2. `pipeline/jobs.py` `J7_weights`, in this order so a late source can still fall back before the lock:
   1. Q0 posterior, then each strategy's Q0 weights.
   2. Controls HRP and EW (`decision_source = "control"`).
   3. Each source: if the fallback plan marks it failed, copy each strategy's Q0 CSV and meta with `decision_source = "fallback_q0"`; if `use_llm_view` is false, copy each strategy's Q0 weights with `decision_source = "llm"` and `use_llm_view = false`; otherwise build the posterior and run every strategy.
   4. If the lock deadline arrives mid-loop, every unfinished source is written as `fallback_q0` with reason `lock_deadline`.
   5. `weights/locked.json` lists every file hash, as v1.
3. `build_plans`: keys from `ctx.portfolio_keys()`; `decision_source` from meta.
4. `update_ic_history`: loops `ctx.sources()`; signals `baseline`, `llm_raw` (likely), `llm_resid`, `combined` (tilt), read from `posterior/<source>.csv`.
5. `pipeline/fallback.py`: `FallbackPlan` keyed by source; `week_status_for(week_id, portfolio_key)` maps a steered portfolio to its source; Q0s and controls are `normal` unless the week is `no_rebalance`.
6. `pipeline/status.py`, `cli.py status`: summary grouped by strategy; per source: endpoint, valid rate, fallback reason; per strategy: solver failures and mean `view_shift`.
7. `J8_replay_score`: loops all portfolio keys; logic unchanged.

**Outputs:** the functions above, the `posterior/` folder, the new meta fields, the status summary.

**Edge cases:** a source failed and also past the deadline (labeled once, reason from the plan); a strategy's Q0 itself solver-failed (its steered fallbacks copy the Q0 result, flag carried); a `no_rebalance` week (all portfolios hold, as v1); a stock dropped mid-week (v1 simulator rule).

**Tests (`test_v2_phase8_jobs.py`, updates to `tests/integration/test_phase10*`):** file count; every file passes `check_weights`; forced source failure; `use_llm_view = false` byte-identical to the strategy's Q0; lock deadline mid-loop; IC history row count; J7 and J8 reruns give identical hashes; `view_shift` is 0 for fallbacks and Q0 copies.

**Meta:** LOC 350 to 500. Depends on Phase 7. Agent can finish alone: yes. Risk: medium (largest edit to the weekly flow). Complexity: 3/5.

**Done when:** the 30-week integration test passes with all strategies and both controls under the test config.

---

## Phase 9: Universe of 100 to 300, runtime, and compute projection

**Goal:** the pipeline runs at 300 stocks within the weekend window, the universe builder handles thin sectors, and the full shuffle cost is projected before the freeze.

**Goals checklist:**
1. The universe fills to the configured size even when sectors are thin, and logs it.
2. Resampled MVO, Robust MVO and the shuffle reuse compiled cvxpy problems.
3. A benchmark reports per-strategy solve times, weekly J7 time, and projected shuffle CPU hours.

**Build:**
1. `data/universe.py`: keep the v1 rule (top floor(size / sectors) per sector by dollar volume, then the highest-volume leftovers). Log each sector that could not fill its quota and how many leftover slots were used.
2. `optimize/mean_variance.py`: a cached problem keyed by (tickers tuple, SHA-256 of Sigma) with mu as a `cp.Parameter`; outputs equal the uncached path. Same reuse in `robust.py` and `cvar.py` (mu as a parameter; R rebuilt per source because M changes).
3. `scripts/benchmark_pipeline.py`: options `--n-stocks`, `--sources`, `--strategies`, `--weeks`. Prints seconds per solve per strategy, J7 seconds per week, peak memory, and a projection: weekly J7 time at the full source count, and shuffle CPU hours = solves per replay x weeks x 500 x steered portfolios x mean solve time. Writes the same table to `data/benchmarks/<date>.csv`.
4. `config/study.yaml`: `universe.size` comment `# DECISION(B4): 100 to 300, set after the ARCTIC allocation`.

**Outputs:** the universe logging, cached problems, the benchmark script and its CSV.

**Edge cases:** a sector with fewer eligible names than its quota; extra tickers in a thin sector; cache invalidation when tickers change mid-run (the key includes tickers).

**Tests (`test_v2_phase9_scale.py`):** synthetic members with a 15-name sector at size 300 give 300 names and the right log; cached vs uncached MVO within 1e-9 on 20 random mu vectors; a `@pytest.mark.slow` benchmark at 300 stocks, 3 sources, all strategies (excluded from default CI).

**Meta:** LOC 200 to 350. Depends on Phase 8. Agent can finish alone: yes. Risk: low. Complexity: 2/5.

**Done when:** PROGRESS.md records the benchmark at 300 stocks scaled to 51 sources (7 models x 7 prompt types + Q0) and the projected shuffle CPU hours, with a note if the projection exceeds 15,000 CPU hours (the ARCTIC request). The agent reports any gap and does not change a decision to close it.

---

## Phase 10: Evaluation over strategies

**Goal:** H1 to H5 follow the proposal, with Holm families per strategy for return tests and the new H3b and H4 ranking.

**Goals checklist:**
1. IC tests are per source; return tests are per portfolio, corrected within each strategy.
2. H3 covers every skills prompt type against both controls; H3b tests range width.
3. H4 ranks strategies by their mean gap vs their own Q0.
4. Controls appear in metric and factor tables but in no hypothesis family.
5. The report has a section per strategy and prints every design choice value used.

**Build:**
1. `eval/data.py`: `EvalData` carries `sources`, `portfolios`, `strategy_of`, `source_of`; weekly returns for every portfolio including HRP and EW; `clean_mask(portfolio)` uses the source's fallback weeks for steered portfolios.
2. `eval/hypotheses.py`:
   - H1: per source, likely vs baseline rank IC, with win, loss and tie counts; Holm across sources.
   - H2 IC: per source, BL tilt vs baseline; Holm across sources.
   - H2 return: each steered portfolio vs its own strategy's Q0 (paired weekly gap on the source's clean weeks); Holm within each strategy.
   - H3: per model, each `skills` kind vs `bare` and vs `placebo`; Holm per model. Prompt types come from `manifest.by_kind`.
   - H3b: per source and week, split valid stocks at the median width (width <= median is narrow); rank IC of likely in each group; skip the week if either group has fewer than `eval.min_group_size` (10) stocks or all widths are equal; weekly gap narrow minus wide; Holm across sources. Skipped weeks are counted in the table.
   - H4 ranking: per strategy and week, the mean gap vs its Q0 across the sources clean that week; strategy mean with a bootstrap CI; rank. Weeks where no source is clean are dropped and counted. The v1 market direction, volatility and earnings week splits run per strategy. Exploratory, no Holm.
   - Solver-failed weeks stay in the data and are counted per portfolio in the metrics table.
3. `config/study.yaml` `eval.holm_families`: `H1_ic_gap: all_sources`, `H2_ic_gap: all_sources`, `H2_portfolio_vs_q0: per_strategy`, `H3_skills_vs_controls: per_model`, `H3b_width: all_sources`, `H5_shuffle: per_strategy` (each strategy's steered portfolios plus its Q0). Add `eval.h3b: {split: median, min_group_size: 10}` tagged `DECISION(G3b)`.
4. `eval/factors.py`: factor regression for every portfolio, controls included.
5. `eval/report.py`:
   - Header: portfolio count, weeks, fallback and fallback-served weeks, and every design choice value (K1, S4, S5, G3b, P1) read from config.
   - Per strategy: metrics (return, Sharpe, max drawdown, turnover, mean `view_shift`, fallback and solver-failed weeks), H2 return table, H5 table, an equity curve figure (its Q0, its steered portfolios, HRP and EW).
   - H1, H2 IC, H3 and H3b tables; an H3b figure of IC by width group.
   - H4 ranking table and figure.
   - A warning line for any strategy whose mean `view_shift` is below 0.01 (views barely change its weights).

**Outputs:** the functions above and the report sections.

**Edge cases:** a source with no clean weeks (reported, excluded from tests); a strategy whose Q0 solver-failed every week; integer-width ties at the median; fewer than 3 weeks for a test (NaN p-value, reported, never dropped silently).

**Tests (`test_v2_phase10_eval.py`, updates to `test_phase11*`):** Holm adjusts within each strategy only on a planted 2-strategy case; H2 pairs each portfolio with its own strategy's Q0; H3 finds every `skills` kind; H3b is positive on planted data and skips all-equal weeks; H4 ranking matches a planted order; controls are in no family; the report builds on the synthetic 30-week run with every section.

**Meta:** LOC 550 to 750. Depends on Phase 8. Agent can finish alone: yes. Risk: medium (statistical correctness). Complexity: 3/5.

**Done when:** `portstudy eval --synthetic` writes a report with every section above.

---

## Phase 11: Shuffle test over strategies, sharded for Slurm

**Goal:** 500 shuffles for every steered portfolio (G6), runnable as a Slurm array.

**Goals checklist:**
1. Shuffles permute range triples (and z_b with k_sd for Q0) and rerun the portfolio's own strategy.
2. Shards can run in any order, any number of times, with identical results.
3. The file count stays small enough for cluster quotas (one file per shard, not per shuffle).

**Build:**
1. `eval/shuffle.py`:
   - Steered portfolio: per clean week, permute the `low, likely, high, width` rows across stocks with an RNG seeded by SHA-256 of `(seed, portfolio, index, week)`; rerun residualize, views, BL and `run_strategy`. IC_L and `use_llm_view` stay fixed at the real values (v1 rule). Q0: permute z_b and k_sd together.
   - CVaR and Resampled use the same Z as the real portfolio (common random numbers), so the identity permutation reproduces the real weights.
   - Fallback and `no_rebalance` weeks use stored weights or hold (v1).
   - `Shard(portfolio, start, stop)`; `plan_shards(portfolios, n_shuffles, shard_size) -> list[Shard]`; `run_shard(shard)` writes `data/shuffle/<portfolio>/<start>_<stop>.json` atomically, with per-index metrics and a completion marker; an existing complete file is skipped.
   - `shuffle_test()` reads all shard files, fails loudly listing missing ranges, computes p-values, then applies Holm per strategy.
2. `pipeline/cli.py`: `portstudy shuffle plan --out data/shuffle/shards.csv`, `portstudy shuffle run --shard-index N --shards <file>`, `portstudy shuffle run-local --processes P`, `portstudy shuffle collect`.
3. `config/study.yaml` `eval.shuffle`: add `shard_size: 25`; keep `n_shuffles: 500`, `seed: 99`.

**Outputs:** the sharded shuffle functions, the CLI commands, the shard files.

**Edge cases:** a half-written shard file after a killed job (no completion marker, so it reruns); two array tasks given the same shard (atomic write, identical content); a portfolio with no clean weeks (skipped, reported).

**Tests (`test_v2_phase11_shuffle.py`):** shards in any order, or twice, equal one serial run; a missing shard makes `collect` fail with the range listed; each shuffle keeps the multiset of triples per week; the identity permutation equals the real portfolio for every strategy; a shard file without its marker is rerun.

**Meta:** LOC 300 to 450. Depends on Phases 7 and 10. Agent can finish alone: yes. Risk: medium (compute size). Complexity: 3/5.

**Done when:** 20 shuffles in 4 shards run in separate processes and match the serial result.

---

## Phase 12: ARCTIC execution (Slurm templates and weekend submit)

**Goal:** Carter runs the whole weekend on ARCTIC with one command, the D2 fallback is a flag, and both possible network layouts work.

**Goals checklist:**
1. One submit command runs the week; jobs chain with Slurm dependencies.
2. Network jobs run on compute nodes or on the login node, chosen by one env value.
3. The D2 fallback starts with one sbatch and one flag, and follows the single-endpoint rule.
4. `run_weekend.py` resumes from any job.

**Build:**
1. `scripts/arctic/arctic.env.example`: `SLURM_ACCOUNT`, `GPU_PARTITION`, `CPU_PARTITION`, `GPU_GRES`, `KRONOS_WALLTIME`, `WEEKEND_WALLTIME`, `SHUFFLE_WALLTIME`, `PYTHON_MODULE`, `VENV_PATH`, `DATA_ROOT`, `HF_HOME`, `ARCTIC_NETWORK_MODE=compute|login`, `SHUFFLE_MAX_PARALLEL`. All `TBD`, pointing to `docs/arctic_setup.md`.
2. `kronos.sbatch`: J2 for the week (with J1 first when `ARCTIC_NETWORK_MODE=compute`).
3. `weekend.sbatch`: `run_weekend.py --from J3` (compute mode) or `--from J6` (login mode).
4. `submit_weekend.sh`:
   - compute mode: submit kronos, then weekend with `--dependency=afterok`.
   - login mode: run J1 here, submit kronos, run J3 to J5 here in the foreground (API submit, gateway calls, API collect), then submit weekend for J6 to J9 once kronos has finished.
   - Before submitting, warn if `WEEKEND_WALLTIME` ends before the Monday lock deadline.
   - Print job IDs, log paths, and the exact rerun command for each job.
5. `shuffle_array.sbatch`: `--array=0-<n_shards - 1>%${SHUFFLE_MAX_PARALLEL}` calling `portstudy shuffle run`.
6. `llm_fallback.sbatch`: starts an OpenAI-compatible server for the fallback-eligible models on the job's node (serving tool chosen at setup, `# UNVERIFIED`), writes host, port, start time and job ID to the fallback endpoint file, and stays up for its walltime. Model weights load from `HF_HOME` (downloaded in advance, since compute nodes may lack internet).
7. `scripts/run_weekend.py`: `--from <job>`, `--gateway-fallback`, `--config`. Before J4, a health check sends one tiny call per enabled gateway model. On failure without the flag, it prints the D2 steps (submit `llm_fallback.sbatch`, then rerun with `--from J4 --gateway-fallback`) and, if the run continues, marks only the affected sources failed. With the flag, fallback-eligible models switch to the fallback endpoint for this week (Phase 4 single-endpoint rule), and the other models stay on the gateway.
8. `backup.py`: optional `backup.rsync_target` for pulling the dated CSV folders off ARCTIC; the zip step stays.

**Outputs:** the six scripts, the env example, the new `run_weekend.py` options, the rsync option.

**Edge cases:** the gateway passes the health check but fails mid-J4 (affected sources fail by N1, or Carter reruns them with the flag); the kronos job fails (the weekend job never starts because of `afterok`, and the summary says so); a fallback endpoint file from a previous week (rejected by start time); a walltime kill mid-J7 (rerun with `--from J7`, which is idempotent).

**Tests (`test_v2_phase12_arctic.py`):** `shellcheck` on every script (skipped with a PROGRESS note if not installed); `run_weekend.py --from J3 --synthetic` runs J3 to J9 only; a health failure without the flag marks the right sources and continues; with a fake endpoint file, fallback-eligible models switch and the large models do not; a stale endpoint file is rejected.

**Meta:** LOC 300 to 450. Depends on Phases 4, 8 and 11. Agent can finish alone: yes; every ARCTIC-specific value is a `TBD` placeholder, filled in during Phase 15 C. Risk: high (cluster details unknown). Complexity: 3/5.

**Done when:** the scripts pass `shellcheck`, and the synthetic run passes through `--from`, the health check, and the fallback flag.

---

## Phase 13: End-to-end rehearsal and CI

**Goal:** one synthetic run proves every v2 path together, and CI stays fast.

**Goals checklist:**
1. The e2e test covers every strategy, both controls, every fallback path, eval, and a sharded shuffle.
2. Two runs give identical output hashes.
3. CI runtime stays bounded.

**Build:**
1. `tests/e2e/test_full_run.py`, using `study_test.yaml` over 30 synthetic weeks:
   - one forced source failure, one forced solver failure per optimizing strategy, one lock-deadline cutoff, one gateway health failure handled with the fallback flag, one stock removed mid-run;
   - then eval and a 20-shuffle test in 4 shards.
2. Invariants: the v1 Phase 14 list, plus: weight file count per week, one Q0 per strategy, controls never read views, IC rows per source only, every weight file passes `check_weights`, H3b has a positive gap on the planted width effect, and `view_shift` is above 0 for every steered strategy on the planted views.
3. `.gitlab-ci.yml`: lint, type check and unit tests on every push; integration and e2e on the default branch; `slow` tests manual.
4. Record e2e runtime in PROGRESS.md. If it exceeds 15 minutes, shrink the test config (fewer weeks or tickers), never the checks.

**Outputs:** the e2e test, the CI changes, the runtime record.

**Meta:** LOC 200 to 350. Depends on Phases 1 to 12. Agent can finish alone: yes. Risk: low. Complexity: 2/5.

**Done when:** e2e passes twice in a row with identical hashes.

---

## Phase 14: Documentation and final checklist

**Goal:** every doc describes v2, the design choices are recorded in the README with their reasons, and nothing stale remains.

**Goals checklist:**
1. The README has a "Design choices" section for K1, S4, S5, G3b and P1.
2. Every doc and config comment matches Section 0.3.
3. The final checks all pass.

**Build:**
1. `README.md`:
   - v2 scope, the portfolio count formula, strategy list, controls, and how to run synthetic and on ARCTIC.
   - **Design choices** section, one entry per choice with its value, config key, and reason (use this text):
     - **K1, Kronos spread to baseline uncertainty** (`views.kronos_spread_clip`): Omega_b,ii = tau Sigma_ii clip(s_i / median(s), 0.25, 4). Scaling by the median keeps the average baseline uncertainty at the v1 level, so the spread only shifts trust between stocks. The clip stops one unusual forecast from making a view almost certain or almost ignored.
     - **S4, Mean-CVaR lambda = 1.0** (`strategies.cvar.lam`): expected return and tail loss are both weekly returns, so equal weight needs no tuning and is not fit to past returns.
     - **S5, Robust MVO kappa = 1.0** (`strategies.robust.kappa`): the worst case is taken over a one standard deviation ellipsoid of the posterior mean, a neutral setting not fit to past returns.
     - **G3b, median width split** (`eval.h3b`): splitting at each week's median gives two groups of similar size, so neither group's rank IC is much noisier than the other; weeks with a group under 10 stocks are skipped.
     - **P1, skills length tolerance 5%** (`llm.skills_length_tolerance`): keeps the skills files close enough in length that length cannot explain differences between them; the placebo stays within 2% of their mean.
   - A pointer to `docs/open_questions.md` for D3, the only provisional decision left.
2. `docs/design.md`: rating source vs portfolio, the strategy package, posterior covariance, uncertainty formulas, common random numbers, the fallback chain, the single-endpoint rule, and that the 5% cap applies at each rebalance (weights can drift above it between rebalances, as in v1).
3. `docs/runbook.md`: the ARCTIC weekend in both network modes, the D2 steps, backups, and the post-run shuffle and eval.
4. `docs/config_reference.md`: every new key and env var (`ARCTIC_LITELLM_API_KEY`, `ARCTIC_FALLBACK_API_KEY`, `PORTSTUDY_CONFIG`).
5. `docs/open_questions.md`: the Section 0.3 table, with D3 the only provisional row.
6. `docs/arctic_setup.md` (new, first person): env file, venv, Kronos and fallback model weights in `HF_HOME`, Hugging Face license acceptance for Gemma and Llama, VPN for the gateway, quotas.
7. Leave `docs/arctic_gpu_request.md` and `docs/desktop_checklist.md` in place and list them in PROGRESS.md (what happens to them is a Phase 15 D task).
8. `scripts/check_banned_words.py`: ban "the researcher" and "the student" in `docs/arctic_setup.md`.
9. Final checks: em dash scan over the repo, banned words, `ruff`, `mypy`, the full suite, `check_secrets.py`, and `grep -rni "qwen\|ollama\|confidence" src/ config/` returning nothing.

**Outputs:** the updated docs and the passing checks.

**Meta:** LOC 600 to 900 (mostly docs). Depends on Phase 13. Agent can finish alone: yes. Risk: low. Complexity: 1/5.

**Done when:** every check in item 9 passes, the README design choices match `study.yaml`, and PROGRESS.md lists each remaining UNVERIFIED item.

---

## 4. Testing checklist

**Existing tests needing updates:** `test_phase5*` (manifest), `test_phase6_llm.py` (schema), `test_phase7_views.py` (width, spread), `test_phase8_optimize.py` (posterior, columns), `test_phase10*` and `test_phase11*` integration (keys, counts), `snapshots_30_weeks`, `test_full_run.py`; the Ollama tests are deleted with the provider.

**New tests:** one `test_v2_phaseN_*.py` per phase, as listed.

**Edge cases (collected from the phases):** booleans and strings as range values; unclosed thinking; all outputs invalid; all widths equal; zero or NaN Kronos spreads; Cholesky failures; fewer than 20 names; Top-K and HRP ties; first-week solver failure; dropped tickers in previous weights; gateway 401, 429 storms, and field rejections; a stale fallback endpoint; mixed serving attempts; the lock deadline mid-J7; walltime kills; half-written shard files.

**Regression:** v1 invariants (simulator, stops, costs, dividends, splits, lock deadline, CSV byte identity) pass unchanged.

**Performance:** the Phase 9 benchmark and projection; e2e under 15 minutes.

**Security:** no keys in the repo (`check_secrets.py` with the new names); the gateway key is never logged (a test checks that log records redact `Authorization`); the fallback server binds to the job's node only.

---

## 5. Documentation checklist
- README with the design choices section, plus design, runbook, config reference and open questions (Phase 14).
- `docs/arctic_setup.md` in the first person.
- Env vars documented: `ARCTIC_LITELLM_API_KEY`, `ARCTIC_FALLBACK_API_KEY`, `PORTSTUDY_CONFIG`, plus the v1 API keys.
- `config/study.yaml` decisions, the README design choices, and `docs/open_questions.md` all match Section 0.3.
- Stale references removed: Ollama, desktop GPU, Qwen, confidence, "13 portfolios".
- PROGRESS.md lists every UNVERIFIED item and every changed v1 test.

---

## 6. Decisions

All study decisions are in Section 0.3. The five design choices (K1, S4, S5, G3b, P1) are confirmed and documented in the README with their reasons. D3 (the paid API budget) stays provisional until test prompts run in Phase 15 G. The agent never changes any of them; if a phase shows a problem with one (for example the Phase 9 projection or a `view_shift` warning), it records it in PROGRESS.md and keeps building.

---

## 7. Migration notes

**Migration required:** yes, code and config only.
- No live or pilot data exists, so stored files are regenerated, not converted. Deleting the old synthetic `data/` folders is a Phase 15 D task.
- Portfolio keys and the weight, score and baseline columns change format. Each reader is updated in the same phase as its writer.
- Rollback: v1 stays at commit `030faf4` (reverting is a Phase 15 A task).

---

## 8. Major caveats
- **Risk terms can outweigh the views.** Weekly view tilts are small (about IC x sigma, under 0.1% per stock). With lambda = 1 the CVaR term, and with kappa = 1 the robust penalty, are much larger than those tilts, so both strategies may sit close to their minimum-risk portfolios and move little when the views change. That is a real property of these strategies, not a bug, and it is part of what H4 measures. The `view_shift` diagnostic and the report warning make it visible before the freeze.
- **Top-K ranks on mu_BL, which includes the prior pi.** Unless views are strong, it will mostly hold high-pi stocks every week, and the LLM effect shows up only at the edge of the top 20.
- **Unverified gateway.** It is in alpha; JSON mode, seeds, rate limits, reasoning output and small-model IDs are unconfirmed. A long outage falls back under D2 or costs that week for the affected models.
- **Fallback serving differences.** A small model served from a Slurm job may not match the gateway copy; such weeks are flagged.
- **Compute size.** The full shuffle is about 125,000 replays, with CVaR and Resampled the slowest; Phase 9 projects whether it fits the 15,000 CPU hours requested. It runs once after the live run, not weekly.
- **Statistical power.** More strategies add portfolios, not weeks, so small IC gaps stay hard to detect, and the report says so.
- **Kronos sampling.** Without per-path samples, 20 single-sample calls cost about 20 times the GPU time.
- **Network and queues on ARCTIC.** Compute-node network access and Slurm queue waits are unknown until the allocation; the login mode and the walltime warning cover the known cases.

---

## 9. Confidence and unknowns

**Confidence:** about 80% that Phases 1 to 11, 13 and 14 build as written on synthetic data; about 60% for Phase 12 until the allocation details are known.

**Unknowns:** small-model gateway IDs, gateway rate limits, JSON mode and seed support, the MiniMax reasoning format, Kronos per-path access, ARCTIC partitions, walltimes, network and quotas, CVaR solve time at 300 names and 10,000 scenarios, and whether the full shuffle fits the CPU request.

---

## Appendix: Formula sheet v2 (single source of truth, replaces the v1 sheet where they differ)

```
Momentum:     M12_1 = P[t-21]/P[t-252] - 1 ;  M3 = P[t]/P[t-63] - 1 ;  R1 = -(P[t]/P[t-21] - 1)
Rank:         u(x) = rank(x)/N
Kronos:       k_i = mean over 20 paths of (P_hat[t+5] / P[t] - 1) ;  s_i = sd over the same 20 paths (ddof 1)
Baseline:     b = 0.5 u(m) + 0.5 u(k) ;  z_b = (b - mean(b)) / sd(b)
Covariance:   Sigma_daily = (1 - alpha) S + alpha (tr(S)/N) I (Ledoit-Wolf) ;  Sigma = 5 Sigma_daily
Prior:        pi = delta Sigma w_mkt ;  delta = 2.5
LLM output:   low <= likely <= high, integers in [-5, 5] ;  width w = high - low
Residual:     likely = a + c z_b + e ;  z_L = standardize(e)
Views:        q_b = pi + IC_b sigma z_b ;  q_L = pi + IC_L sigma z_L
Baseline Omega: Omega_b,ii = tau Sigma_ii clip(s_i / median(s), 0.25, 4)
LLM Omega:    c_i = clip(1 - w_i/10, 0.1, 1) ;  Omega_L,ii = tau Sigma_ii / c_i ;  tau = 0.05
BL:           A = inv(tau Sigma) + P' inv(Omega) P ;  mu_BL = solve(A, inv(tau Sigma) pi + P' inv(Omega) q) ;  P = [I; I]
BL cov of mean: M = inv(A)
Q0:           same with P = I, q = q_b, Omega = Omega_b
Constraints:  sum w = 1 ;  0 <= w_i <= 0.05 at every rebalance (every strategy and control)
Shared draws: Z ~ N(0, I), seeded by (seed, week_id, strategy), same for every source, Q0 and shuffle
MVO:          max w' mu_BL - (delta/2) w' Sigma w
Mean-CVaR:    r_s = mu_BL + chol(Sigma + M) z_s, s = 1..10,000 ;  beta = 0.95 ;  lam = 1
              max w' mu_BL - lam [ zeta + 1/((1-beta) S) sum_s u_s ]   s.t. u_s >= -r_s' w - zeta, u_s >= 0
Robust MVO:   max w' mu_BL - kappa || chol(M)' w ||_2 - (delta/2) w' Sigma w ;  kappa = 1
Resampled:    mu_d = mu_BL + chol(M) z_d, d = 1..200 ;  w = mean_d MVO(mu_d)
Top-K:        top 20 by mu_BL (ties by ticker), w_i = 1/20
HRP:          distance sqrt((1 - rho)/2), single linkage, recursive bisection, then cap at 0.05
Equal weight: w_i = 1/N
Fallback:     solver failure -> previous weights -> same-strategy Q0 -> equal weight
View shift:   0.5 sum_i |w_i - w_Q0,i|
Cost:         C = 0.001 sum_i |trade value_i|
Stops:        sell at next open if close <= 0.92 avg_cost or close >= 1.15 avg_cost (every daily close)
Rank IC:      Spearman(likely at Friday cutoff, return from next Monday open to the following Monday open)
Zero IC:      IC_L = 0 -> drop LLM view rows -> weights = that strategy's Q0 weights
Trailing IC:  mean of last 8 scored weeks, floored at 0 ;  fixed 0.02 for the first 5 live weeks
H3b:          per source and week, IC(width <= median) - IC(width > median), each group >= 10 stocks
Portfolios:   (models x prompt types + 1) x 5 + 2
Min detectable gap: 2.8 sd / sqrt(T)
```

---

## Phase 15: Manual work (Carter only)

Moved to `manualwork.md`, which keeps the Phase 15 letters and numbers (A to H) that the README, PROGRESS.md and docs refer to, plus the PLAN.md Phase 17 crosswalk.

---

## Project completion checklist

The project is complete only when every box below is checked. Each item names how to confirm it.

### 1. Data sources
- [ ] **Alpaca** (daily prices and news): free plan confirmed to include news and full daily bars; smoke test passes; or prices switched to yfinance.
- [ ] **yfinance** (names, sectors, market caps; price fallback): smoke test passes.
- [ ] **Finnhub** (earnings dates for the H4 earnings week split): smoke test passes.
- [ ] **FRED** (T-bill rate for the risk-free series): smoke test passes.
- [ ] **Wikipedia S&P 500 member list** (universe source): smoke test passes and the frozen universe has the chosen size.
- [ ] **Ken French Data Library** (Fama-French five-factor daily file for H5): file downloaded and its path set in config.
- [ ] **ETF proxies** (SPY, IWM, IWD, IWF, QUAL) present in the price data for every live week.

### 2. ARCTIC compute
- [ ] ARCA allocation granted (requested: 2,500 GPU hours, 15,000 CPU hours, 200 GB for 12 months) and `arctic.env` filled in.
- [ ] VPN and gateway key work from where the weekend jobs run.
- [ ] A GPU Slurm job runs Kronos over the full universe and writes `baseline.csv` with `k_sd`.
- [ ] A CPU Slurm job runs J7 and J8 for every portfolio before the Monday lock.
- [ ] The shuffle array runs at least one shard, and the projected CPU hours fit the allocation (or the gap is resolved).
- [ ] Storage path and file-count quota confirmed for a full study of data, logs, and shuffle shards.

### 3. Models running
- [ ] Gemma 4 31B and MiniMax M2.7 return valid outputs on the gateway, with invalid rates under the N1 threshold (0.15) on test prompts.
- [ ] Gemma 4 E4B, Phi-4 Mini and Llama 3.2 3B deployed on the gateway, enabled in `models.yaml`, and passing the same check.
- [ ] The Slurm fallback server runs the small models end to end.
- [ ] Claude Haiku 5.5 and/or GPT-6 Luna batch runs work with prompt caching, pinned snapshot IDs, and spending caps set.
- [ ] Kronos-base and its tokenizer load on an ARCTIC GPU at the pinned commit.

### 4. Code
- [ ] Phases 1 to 14 green; CI passes on GitLab.
- [ ] Every UNVERIFIED item in PROGRESS.md (v1 and v2) resolved and the tag removed from the code.
- [ ] Synthetic run and backup restore done on ARCTIC.

### 5. Research content and settings
- [ ] 4 to 5 skills files and the placebo written, passing the length checks, with hashes frozen in the manifest.
- [ ] Base instruction wording approved.
- [ ] Universe frozen in `config/universe.csv` with a version tag.
- [ ] Budget D3 recomputed from test prompts; `timing.live_end_date` set.
- [ ] Design choice values confirmed on the synthetic report header.

### 6. Runs
- [ ] Dry run on 3 to 4 real weeks done as a bug check.
- [ ] Repository tagged `frozen-v2`.
- [ ] Pilot week passed and its fixes logged.
- [ ] Live run completed through `timing.live_end_date`, with every week's summary read and every dated CSV backup pulled.
- [ ] Fallback log and model version log complete for every week.

### 7. Analysis
- [ ] Shuffle test complete: every shard present and `portstudy shuffle collect` passes.
- [ ] `portstudy eval` report built with H1 to H5, H3b, the H4 strategy ranking, factor regressions, and the per-strategy sections.
- [ ] Every fallback-served, solver-failed, and skipped week accounted for in the report.

### 8. Deliverables
- [ ] MS project write-up.
- [ ] Conference paper submission, with AI use disclosed.
- [ ] Code Performance and Scaling document for ARCTIC (per the ARCTIC proposal), with the measured per-request and per-job times.
- [ ] Final archive: code at the frozen tag, prompts and skills files with hashes, and every weekly CSV folder, so each result traces back to its inputs.
