# Code review results (read-only review, no code changed)

Reviewer scope: every module under `src/portstudy`, `scripts/run_weekend.py`, `scripts/arctic/*`,
`.gitlab-ci.yml`, and the docs, against PLAN2 (formula sheet and phase specs). Verification
method: read every formula against the sheet, then ran throwaway checks (never saved in the
repo) that compare the code's numbers with independent implementations (numpy closed forms,
scipy SLSQP, statsmodels, scipy.stats, sklearn, a from-scratch HRP). Outputs are pasted as proof.

Severity: **HIGH** = wrong results or a silent failure in a live week; **MEDIUM** = robustness
or correctness gap in an edge case, or a spec deviation; **LOW** = cleanliness, performance,
documentation.

Working tree reviewed: `master` at v2 phase 14 (`9283a8c` on GitLab).

---

## 1. Summary

**Math: verified correct.** Every formula on the PLAN2 sheet that is implemented (momentum, rank
transform, Ledoit-Wolf and weekly scaling, market prior, residualization, both views, both Omega
terms, Black-Litterman posterior and its covariance of the mean, Q0, MVO, Mean-CVaR, Robust MVO,
Resampled MVO, Top-K, HRP, equal weight, view shift, trade cost, stop thresholds, dividend and split
handling in the return window, rank IC, trailing IC rules, Holm) matches an independent computation
to floating-point precision or to solver tolerance (1e-7). Proofs are in section 3.

**Bugs found: 4 HIGH, 11 MEDIUM, 16 LOW (pass 2 added M10, M11, L13 to L16 and an improvements list; see the Pass 2 section).** None of the HIGH items is in the math. They are all in
how the pipeline behaves around failures and on the cluster:

| # | Severity | Where | One line |
|---|---|---|---|
| H1 | HIGH | `data/snapshot.py` make_snapshot | Forward-filled (fabricated) price rows are appended to the raw price ledger; the simulator later "trades" at those made-up opens on halted days, defeating the carry rule. |
| H2 | HIGH | `scripts/run_weekend.py`, `pipeline/jobs.py` | A week whose J1/J2 fails never gets `locked.json` unless someone runs J6 and J7 by hand; the replay then skips that week entirely, so stops are never checked and daily values are missing for those days. |
| H3 | HIGH | `scripts/arctic/submit_weekend.sh` login mode | J3 to J5 run on the login node right after Kronos is *submitted*; J4 refuses to run until J2 is `done`, so the login-mode weekend aborts before the weekend job is submitted. |
| H4 | HIGH | `scripts/run_weekend.py` main + both sbatch files | J8 (replay, IC history) only runs inside `step1_catch_up`, which only runs when `--from J1`; both ARCTIC flows start at J3 or J6, so on the cluster J8 never runs: no replay, no IC history, trailing IC stuck at the cold-start value. |

Everything else (MEDIUM and LOW) is listed in section 2 with file and line, what happens, and the
smallest fix.

**Pipeline silent-failure audit (section 4):** no job can fail silently. Every job writes a
`failed` status file with the traceback and `run_weekend.py` exits non-zero and prints the rerun
command. The two ways a problem can be *missed* are H2 (a failed week is simply absent from the
replay) and H4 (J8 never runs on ARCTIC); both are visible only if someone looks for the missing
rows. The LLM runner cannot lose data silently: every call is appended to a JSONL before the
status moves on, and the N1 rule turns any incomplete source into a labelled fallback.

**What is good.** The optimizer layer is cleanly separated and every strategy's objective is
exactly the sheet's. Determinism holds (two full e2e runs give identical digests). The lock
deadline, the fallback chain (solver -> previous -> Q0 -> equal weight), the single-endpoint rule,
the stale fallback endpoint check and the cost cap all behave as specified.

---

## 2. Findings

(appended as the review proceeds; see sections 3 to 6 for the proofs)

### 2.1 HIGH

#### H1. Forward-filled price rows are appended to the raw ledger (`src/portstudy/data/snapshot.py`)

- `make_snapshot` runs `prices = _forward_fill(prices, expected_days)` (line 187) and then, after
  writing the snapshot, does `all_raw = pd.concat([prices, etf_prices]); ledger.append_prices(all_raw, ...)`
  (lines 221-222). The forward-filled frame is what reaches the ledger.
- `_forward_fill` invents rows for trading days the provider did not return (halts, suspensions):
  `open = high = low = close = previous close`, `volume = 0`.
- The ledger is documented as "raw, written once, never rewritten" and is the only thing the
  simulator reads. With fabricated opens present, `PortfolioSim.rebalance` and `execute_pending`
  see a price on a halted day and fill trades at the stale close; the carry rule ("a stock with no
  price on a day carries its last close ... and is sold at the first open where a price exists")
  never triggers. Because the ledger is append-only, the fabricated rows are permanent: the later
  `refresh_ledger` call keeps the ledger value and only logs `ledger_mismatch`.
- Proof (synthetic world, planted halt on `SYN009`, J1 for the week after the halt, fresh data dir):

```
halt ticker SYN009 provider rows 601 days missing from provider for this ticker: [2026-03-10, 2026-03-11, 2026-03-12]
snapshot week 2026-W11 cutoff 2026-03-13
J1 done
LEDGER rows for SYN009 on days the provider had NO data: 3
             date  ticker        open        high         low       close  volume
31093  2026-03-10  SYN009  324.425694  324.425694  324.425694  324.425694       0
31198  2026-03-11  SYN009  324.425694  324.425694  324.425694  324.425694       0
31303  2026-03-12  SYN009  324.425694  324.425694  324.425694  324.425694       0
```

- Fix: keep the pre-fill frame (`raw_prices = prices.copy()` before `_forward_fill`) and append
  `raw_prices` + `etf_prices` to the ledger; forward fill only what goes into the snapshot CSV
  (the models need a complete adjusted-close matrix; the simulator must not). One assertion in a
  test: no ledger row with `volume == 0` on a day the provider had no row.

#### H2. A failed data/baseline week disappears from the replay (`scripts/run_weekend.py`, `pipeline/jobs.py`)

- `run_weekend.main` stops at the first `JobFailed` (returns 1). If J1 or J2 fails, J6 and J7 never
  run, so no `fallback_plan.json` and no `weights/locked.json` exist for the week. The documented
  intent is "every portfolio holds last week's positions (no_rebalance)", and J6+J7 do implement
  that, but only if someone reruns them by hand (the e2e test does exactly that, which is why it
  passes).
- `weeks_with_weights()` (jobs.py line 501) lists only weeks with `locked.json`; `build_plans` and
  `step1_catch_up` iterate that list; `simulate_week` only walks the days of weeks that have a plan.
  So the days of the missing week are never simulated: no stop-loss or take-profit is evaluated,
  no dividends are credited, no daily values exist, and the next week's rebalance happens as if
  the gap week never existed.
- Proof (two-stock simulator, X falls to 80 during week 2 and recovers; week 2 either has a
  `no_rebalance` plan or no plan at all):

```
with the week-2 no_rebalance plan (what J6+J7 would write): [('2026-03-17', 'X', 'stop_loss')]
   daily records: 15 weeks: ['2026-W10', '2026-W11', '2026-W12']
with week 2 absent (J1 failed, J6/J7 never run): []
   daily records: 10 weeks: ['2026-W10', '2026-W12']
   final value full vs gap: 89820.23 99900.1
```

- Fix (two parts): in `run_weekend.main`, when J1 or J2 fails, still run J6 and J7 (J6 already
  writes the `baseline_unavailable` no-rebalance plan when J2 is not done, and J7 writes a
  `no_rebalance` lock) before exiting 1; and make `build_plans` treat any week between the first
  and last locked week that has no lock as `no_rebalance` instead of skipping it, so the replay
  always covers every calendar week. The FALLBACK_HINTS text for J1/J2 should say that the hold is
  written by J6+J7.

#### H3. Login-mode ARCTIC flow cannot pass J4 (`scripts/arctic/submit_weekend.sh`, `pipeline/jobs.py` J4)

- Login mode does: `run-job J1` on the login node, `sbatch kronos.sbatch` (J2), then immediately
  `run_weekend.py --from J3 --until J5` in the foreground.
- `J4_open_weight_infer` begins with `if read_status(week_id, "J2").state != "done": raise RuntimeError("J2 ... is not done; J4 waits for it")`
  (jobs.py lines 183-185). The Kronos job has only just been queued, so J2 is `pending`; J4 raises,
  `run_weekend` returns 1, and with `set -euo pipefail` the script exits before the weekend job
  (J6 to J9) is submitted. Only if the Kronos job happens to finish during J3 does this work.
- Fix: in login mode, wait for the Kronos job before J4 (`while squeue -j $KRONOS_ID ...; sleep`),
  or split the foreground run into `--from J3 --until J3`, wait, then `--from J4 --until J5`; or
  drop the J2 gate in J4 (it exists only to order GPU work, which Slurm's dependency already does).

#### H4. J8 never runs on ARCTIC (`scripts/run_weekend.py` main, `scripts/arctic/*.sbatch`)

- `main` runs `step1_catch_up` (J8 and J9 for earlier weeks) only `if args.from_job == "J1"` (line 230),
  and the main loop skips J8 explicitly (`elif job != "J8"`, line 237). So J8 runs only through the catch-up
  step of a run that started at J1.
- On ARCTIC, J1 runs via `portstudy run-job J1` (kronos.sbatch in compute mode, the login node in
  login mode), and `weekend.sbatch` starts `run_weekend.py` at `--from J3` or `--from J6`. Neither
  path ever reaches `step1_catch_up`, so J8 (ledger refresh, replay, IC history upsert) never runs:
  no `data/portfolios/*` records, no `ic_history.csv` rows, and `trailing_ic` keeps returning the
  cold-start 0.02 for every week because `store.series(...)` stays empty (`views/ic.py` ~150-170).
  Nothing reports this; J7 happily uses IC 0.02 forever.
- Also: `--from J8` runs nothing for J8 (the loop skips it), which contradicts the `--from <job>`
  help text.
- Fix: run `step1_catch_up` whenever the run includes J6 or earlier (or always, it is idempotent),
  and let the loop run J8 when it is explicitly in the `--from/--until` range. Add a check in J7
  that warns when `ic_history.csv` has no row for the most recent closed week.

### 2.2 MEDIUM

#### M1. Provider errors drop tasks for the rest of the run (`llm/runner.py` run_open_weight, line 541)

`except LlmProviderError: ... log warning; continue` discards the task. A 429 burst that outlives
tenacity's 3 attempts (`litellm.py` `_post`), or any transport error, leaves those stocks without a
record; the source is then `incomplete`, J6 raises before the margin and the whole run must be
rerun by hand, or at the margin the source is marked failed (N1) and trades Q0. There is no
re-queue. Fix: push the task back onto `remaining` with a retry budget (for example 3 passes) and
an exponential pause; abort the model early on `LlmAuthenticationError`.

#### M2. `LlmAuthenticationError.retryable = False` is never read (`llm/providers/litellm.py:24`)

A missing or rejected key (401/403) is handled exactly like a timeout: one warning per stock, 400+
warnings per model, every task skipped. The pre-J4 health check catches a key that is wrong from the
start, but not a key that expires mid-run. Fix: in `run_open_weight`, treat `LlmAuthenticationError`
as `fatal` (stop the model, keep the records, raise) so J4 fails loudly with one message.

#### M3. Under `--gateway-fallback`, a non-eligible model that fails the health check is not marked (`scripts/run_weekend.py` gateway_health)

With the flag, `gateway_health` prints "continuing" and returns without calling
`mark_sources_failed`, and J4 uses `unhealthy = set()`. A non-eligible model (the two primaries are
not fallback-eligible in `models.yaml`) that fails the probe is then run anyway; every call fails
(M1) and the source ends incomplete instead of being marked failed immediately. Fix: with the flag,
still mark the failed models that are not `fallback_eligible`.

#### M4. `current_week` ignores DST (`pipeline/jobs.py` lines 700-702)

`now.astimezone(dt.timezone(dt.timedelta(hours=-5)))` is a fixed UTC-5. In summer (EDT, UTC-4) a
run started between 00:00 and 01:00 ET on Friday picks the previous week. Proof:

```
Friday 2026-07-10 00:30 EDT -> 2026-W27   ISO week of that Friday: 2026-W28
Saturday 2026-07-11 00:30 EDT -> 2026-W28
```

Fix: `now.astimezone(ZoneInfo("America/New_York"))` (ET is already defined in `llm/runner.py`).

#### M5. A `done` job is skipped even when its outputs are gone (`pipeline/status.py` run_job, line 91)

`outputs_hash` is recorded but never checked on the skip path. If `weights/` or a snapshot folder
is deleted or restored from an older backup, J7/J1 report `skipped` and the downstream job fails
with a confusing error. Fix: when `outputs` is given, rerun if any output file is missing or its
hash differs from `outputs_hash`.

#### M6. Steered-portfolio `view_shift` is forced to 0 on any solver fallback (`optimize/__init__.py` `_optimize`, line 244)

`shift = ... if not result.fallback_flag else 0.0`. A steered portfolio that fell back to previous
weights can be far from Q0 but reports shift 0, which lowers the strategy's mean view shift in the
report and can trigger the `< 0.01` warning for the wrong reason. Fix: compute the shift from the
weights actually written, and record `fallback_flag` next to it (the report already has a
solver-failed column to explain it).

#### M7. `build_weights` recomputes and overwrites the strategy Q0 when the LLM view is dropped (`optimize/__init__.py` line 488 onward)

In the `not views.use_llm_view` branch it calls `build_q0_weights(...)` again with the *steered*
portfolio's `previous_week`, rewriting `<strategy>__Q0.csv` and `.meta.json`. J7 never reaches this
branch (it copies Q0 itself when the posterior is None), so today it is dead for the weekly run, but
any other caller (shuffle, CLI, tests) that reaches it re-solves CVaR/Resampled for every dropped
source and can change the Q0 file when `previous` differs. Fix: copy the existing Q0 file (as J7
does) instead of re-solving.

#### M8. The IC history used by J7 is not part of J7's inputs hash (`pipeline/jobs.py` J7_weights)

J7 hashes the plan, baseline, prior, sigma, scores and config, but not `ic_history.csv`. A J7 rerun
after J8 added rows for an earlier week is skipped as "done" even though `IC_b`/`IC_L` would now
differ. In the normal weekly order this cannot happen (J8 for week w-1 runs before J7 for week w),
so this is idempotency hygiene, not a live bug; but `--force` is the only way to refresh.

#### M9. Documentation says the gateway key is redaction-tested; nothing redacts (`PLAN2` section 4, `logging_setup.py`)

No log filter removes `Authorization` or the key, and the only test touching the header checks that
it *is* sent (`tests/unit/test_v2_phase4_litellm.py:108`). Today nothing logs the header (httpx does
not log headers at INFO; error messages include the response body only), so there is no leak, but
the promised guard does not exist. Fix: add a `logging.Filter` on the JSON handler that masks
`Bearer ...` and the two env values, plus the test.

### 2.3 LOW

- **L1** `scripts/run_weekend.py`: `--until J5` in login mode still runs `gateway_health`, good; but
  `--from J8` is a silent no-op (see H4).
- **L2** `llm/validate.py`: floats with zero fraction (`3.0`) are accepted as integers; the schema
  says integers. Harmless, but `3.0` from a model that ignored the format is a weak signal of
  non-compliance; consider counting it in `extra_keys`-style diagnostics.
- **L3** `llm/providers/litellm.py`: `tokens_cached` is never filled from
  `usage.prompt_tokens_details.cached_tokens`; gateway cost is 0 so it only affects the token
  columns.
- **L4** `eval/data.py` `sharpe`: annualised mean/sd with no risk-free rate although DTB3 rates are
  collected in every snapshot. Either subtract the weekly T-bill return or say "excess over zero".
- **L5** `llm/runner.py` `_update_open_weight_status` rewrites `run_status.json` after every single
  call (read + write JSON, 400 to 700 times per model per week). Batch it per N calls or per
  config; it is also the only non-atomic ordering between the JSONL append and the status file.
- **L6** `pipeline/fallback.py` `fallback_weeks(portfolio)` reads every week's plan and log JSON for
  every source in J7 (O(weeks x sources) file reads); cache the plans once per J7 run.
- **L7** `optimize/mean_variance.py` `_normalize_capped_weights` moves the solver's solution by up to
  solver tolerance to hit the cap/sum exactly; fine, but `headroom.sum()` can be 0 only when every
  weight is at the cap, which the feasibility check excludes. No bug; documenting the invariant.
- **L8** `data/providers/yfinance_provider.py` `get_reference`: `market_cap` is yfinance's *current*
  value, not as of the cutoff. For the live weekly run the lag is hours; for any backfill it is
  look-ahead in the prior weights.
- **L9** `docs/config_reference.md` omits these keys: `data.rate_limits.*`, `data.checks.missing_days`,
  `nonpositive_price`, `duplicate_rows`, `stale_price`, `news_after_cutoff`, `short_history`,
  `stopped_trading`, `strategies.cvar.n_scenarios`, `strategies.resampled.n_draws`,
  `strategies.resampled.max_failed_share` (they appear as prose or not at all).
- **L10** `pyproject.toml`: `scipy`, `scikit-learn`, `statsmodels`, `cvxpy`, `osqp`, `clarabel`,
  `httpx`, `tenacity`, `matplotlib` have no upper bound; pandas, numpy and pydantic do. A major
  release of cvxpy or clarabel can change solver defaults and break the 1e-9 cached/uncached
  equality test. Pin `<2` style upper bounds like the others.

- **L11** `scripts/benchmark_pipeline.py` `projection`: the shuffle CPU-hour projection counts
  `projected_sources x strategies` steered portfolios only; the five strategy Q0 shuffles and the
  residualize + Black-Litterman work per shuffle are not counted (about 2 to 3 percent under), and
  `j7_projected` uses `(projected_sources + 1)` while the `--projected-sources` help text says 51
  already includes Q0, so J7 is projected for 52 posteriors. Neither changes the 16,600-hour
  conclusion materially; state the convention once.
- **L12** `scripts/arctic/llm_fallback.sbatch` writes `fallback_endpoint.json` as soon as the
  servers are launched, before they answer; poll `GET /v1/models` on each port first so a
  `--gateway-fallback` rerun started from the printed command does not hit connection errors.


---

## 3. Mathematical verification (proof)

Method: throwaway script (not saved in the repo) that calls the production functions and recomputes
each quantity independently with numpy closed forms, scipy SLSQP, statsmodels, scipy.stats, sklearn,
and a from-scratch Lopez de Prado HRP. Random inputs, seed 0. Full output:

```
=== (a) Black-Litterman closed form, 4 assets ===
views omega_b match: True  omega_L match: True
q_b match: True  q_L match: True
mu_BL max abs diff: 6.938893903907228e-18
M (cov of mean) max abs diff: 6.938893903907228e-18
mu_BL vs textbook form max abs diff: 9.540979117872439e-18
Q0 mu diff: 2.439454888092385e-18

=== (b) MVO cvxpy vs SLSQP, 30 assets, cap 0.05 ===
solver: OSQP  cvxpy obj: 0.002684683805090147  slsqp obj: 0.002684688997097136
max |w_cvxpy - w_slsqp|: 1.0832075865123377e-07  sum: 0.9999999999999999  max w: 0.05  min w: 1.0832075865123377e-07

=== (c) Mean-CVaR: optimizer objective equals sample CVaR of its own solution; vs SLSQP ===
solver: CLARABEL  flag: None
problem.value: 0.0008999629601765327  mu'w - lam*CVaR(sample, RU form): 0.000899962930981968
diagnostics in_sample_cvar: 0.002049861359591097  independent: 0.0020498613595910974
SLSQP objective (nonsmooth, local): 0.0008999614083964958  cvxpy objective: 0.000899962930981968  cvxpy >= slsqp: True

=== (d) Robust MVO: objective check ===
solver: CLARABEL  problem.value: 0.0024741276221109253  recomputed mu'w - k||L'w|| - d/2 w'Sw: 0.002474127622110924
SLSQP obj: 0.0024741276465453247  max |dw|: 1.073646378842684e-06

=== (e) Resampled: equals mean of per-draw MVO solutions ===
max |w - mean(draw solutions)|: 0.0

=== (f) Top-K ===
selected == top 20 by mu: True  weights: [np.float64(0.05)]

=== (g) HRP vs independent Lopez de Prado implementation ===
max |hrp - LdP reference| (no cap): 0.0

=== (h) residualize vs statsmodels; rank_ic vs scipy; Ledoit-Wolf vs sklearn ===
z_L max diff vs statsmodels: 4.440892098500626e-16
rank_ic vs spearmanr: -0.052004801920768305 -0.052004801920768305
Sigma == 5 * [(1-a)S + a tr(S)/N I]: True  weekly_scale: 5.0  alpha: 1.0

=== (i) weekly_returns with a split and dividend, hand example ===
X: 100 -> 2:1 split -> 55 (=110 pre-split) + div 0.5 x 2 shares = 111 -> ret 0.1100000000000001  expected 0.1100000000000001
Y: 40 -> 42 ret 0.050000000000000044  expected 0.050000000000000044

=== (j) simulator stop thresholds and cost, 1 stock ===
2026-03-09 X buy 499.5005 100.0 rebalance 49.95
2026-03-09 Y buy 999.001 50.0 rebalance 49.95
2026-03-12 X sell 499.5005 90.0 stop_loss 44.955
expected: buy 500 X @100 cost 50, buy 1000 Y @50 cost 50; stop_loss sell X at the open after the 91.9 close (day 4 open 90), cost 0.001*500*90=45
```

Reading of each block against the PLAN2 formula sheet:

| Sheet line | Code | Check | Result |
|---|---|---|---|
| `Omega_b,ii = tau Sigma_ii clip(s_i/median(s), 0.25, 4)` | `views/build.py::_baseline_ratio`, `build_views` | recomputed from k_sd | identical |
| `c_i = clip(1 - w_i/10, 0.1, 1); Omega_L,ii = tau Sigma_ii / c_i` | `views/build.py::width_to_c` | recomputed | identical |
| `q_b = pi + IC_b sigma z_b; q_L = pi + IC_L sigma z_L` | `build_views` | recomputed | identical |
| `A = inv(tau Sigma) + P' inv(Omega) P; mu_BL = solve(A, b); M = inv(A)` | `optimize/black_litterman.py` (Cholesky solves) | dense closed form and the textbook form `pi + tau S P'(P tau S P' + Om)^-1 (q - P pi)` | max diff 1e-17 |
| Q0: `P = I, q = q_b, Omega = Omega_b` | `black_litterman_q0` | closed form | 2e-18 |
| MVO `max w'mu - (delta/2) w'Sigma w`, `0 <= w <= 0.05`, `sum w = 1` | `mean_variance.py` (OSQP, Cholesky risk) | SLSQP from a different start | objective agrees to 5e-9, weights to 1e-7 (solver tolerance) |
| Mean-CVaR, Rockafellar-Uryasev: `max mu'w - lam[zeta + 1/((1-beta)S) sum u_s]`, `u_s >= -r_s'w - zeta`, `r_s = mu + chol(Sigma + M) z_s` | `strategies/cvar.py` | the LP optimum equals `mu'w - lam * sampleCVaR(w)` recomputed as VaR + mean excess over the same 400 scenarios; `in_sample_cvar` diagnostic identical; SLSQP on the non-smooth objective cannot beat it | 1e-11; True |
| Robust `max w'mu - kappa ||chol(M)'w|| - (delta/2) w'Sigma w` | `strategies/robust.py` | objective recomputed from the weights; SLSQP | 1e-18; weights within 1e-6 |
| Resampled `w = mean_d MVO(mu + chol(M) z_d)` | `strategies/resampled.py` | re-ran the 20 draws through `solve_mean_variance` with the same `standard_normals` | exactly equal (0.0) |
| Top-K: top 20 by `mu_BL`, ties by ticker, `1/20` each | `strategies/topk.py` | set comparison | equal; weights 0.05 |
| HRP: `sqrt((1-rho)/2)`, single linkage, recursive bisection by inverse-variance cluster variance | `strategies/hrp.py` | independent LdP implementation (cap removed) | exactly equal (0.0) |
| `likely = a + c z_b + e; z_L = standardize(e)` | `views/residualize.py` | statsmodels OLS residuals, ddof 1 | 4e-16 |
| Rank IC = Spearman with average ranks | `views/ic.py::rank_ic` | `scipy.stats.spearmanr` | identical |
| `Sigma = 5 [(1-a) S + a tr(S)/N I]` | `risk/covariance.py` | sklearn LedoitWolf shrinkage recomposed | True |
| Return window with a 2:1 split and a 0.50 dividend | `views/ic.py::weekly_returns` | hand: (55*2 + 0.5*2)/100 - 1 = 0.11 | 0.11 |
| Cost `0.001 * |trade value|`; stop `close <= 0.92 avg_cost` sells at the next open | `sim/simulator.py` | hand example | buys cost 49.95 each (buys scaled by 1/(1+c) so cash covers cost, as the docstring states), stop sell at 90 cost 44.955 |

Also checked by reading (no numeric test needed): momentum `M12_1 = P[t-21]/P[t-252]-1`, `M3`,
`R1 = -(P[t]/P[t-21]-1)`, `u(x) = rank/N` (`baseline/momentum.py`, `common.py`); `b = 0.5 u(m) + 0.5 u(k)`,
`z_b` with ddof 1 (`combine.py`); `pi = delta Sigma w_mkt` with caps renormalised over the study
stocks (`risk/prior.py`); Kronos `k = mean`, `k_sd = sd(ddof=1)` over paths (`baseline/kronos.py`);
view shift `0.5 sum |w - w_Q0|` (`strategies/common.py`); trailing IC = mean of the last 8 closed,
clean, earlier weeks, floored at 0, cold start 0.02 for the first 5 live weeks or under 3 weeks
(`views/ic.py`); common random numbers seeded by `(seed, week, strategy)` (`standard_normals`);
Holm via `statsmodels.multipletests` within the configured family (`eval/tests.py`, `hypotheses.py`);
H3b median split with `<= median` narrow, min group 10, equal-width weeks skipped and counted;
`2.8 sd / sqrt(T)` min detectable gap; the fallback chain previous -> Q0 -> equal weight with the
cap re-applied (`fallback_chain`). The 5 percent cap is enforced at every rebalance for every
strategy and control (`check_weights` after each solve) and weights drift between rebalances, as the
plan says.

One stub-only remark: the stub forecaster's `k_sd` is a seeded lognormal unrelated to its paths
(documented in README design choices); the Kronos path uses the real path sd.

---

## 4. Pipeline robustness audit (can anything fail silently?)

Trace of one week through `scripts/run_weekend.py` and `pipeline/jobs.py`:

| Step | Failure mode | What happens | Verdict |
|---|---|---|---|
| any job | exception inside `fn` | `run_job` writes `state=failed`, `error`, traceback tail to `jobs/<J>.status.json`, raises `JobFailed`; `run_weekend.run` prints FAILED, the fallback hint and the exact rerun command; `main` returns 1 | loud |
| any job | process killed (walltime, Ctrl-C) | status stays `running`; the next run treats it as not done and reruns; LLM records are JSONL appends so nothing is repeated | safe |
| J1/J2 | provider or baseline failure | loud, but the no-rebalance lock is only written if J6+J7 are run afterwards; otherwise the week is missing from the replay | **H2** |
| J3 | cost cap | `CostCapExceeded` unless `--allow-over-budget`; batches already submitted are reused via the handle file | loud |
| J4 | gateway down before start | health probe marks sources failed, prints D2 steps, J4 skips them, J6 labels them fallback | loud and labelled |
| J4 | gateway errors mid-run | per-task warnings, task dropped, source incomplete; J6 raises until the margin, then labels it failed | labelled, but no retry (**M1**, **M2**) |
| J4 | key missing/expired | same as above, hundreds of warnings | **M2** |
| J5 | batch never returns | `pending` state, poll loop until the margin, then `batch_not_ready_by_lock` fallback | labelled |
| J6 | sources incomplete before the margin | raises (by design) so the run is rerun later | loud |
| J7 | after the lock | `LockViolation` at start; mid-run the remaining portfolios get Q0 with reason `lock_deadline` and the lock file records `locked_at > deadline` | labelled |
| J7 | solver failure | `fallback_chain` previous -> Q0 -> EW, flag in meta and in `fallback_log.json`, counted in the report | labelled |
| J8 | never scheduled on ARCTIC | silent: IC history and replay simply do not advance | **H4** |
| J9 | rsync failure | printed and logged, local backup intact | loud |
| skip path | outputs deleted after `done` | job skipped, downstream fails confusingly | **M5** |
| data | halted stock | fabricated rows in the ledger | **H1** |

Endpoint and connection rules checked: a source runs on one endpoint per week (`_prepare_open_weight_sources`
archives records from a different endpoint before switching; `assess_run` logs `mixed_endpoints_for_source`);
the fallback endpoint file must exist, name the current week and be newer than the source run's
start (`factory._fallback_base_url`); the cache key includes the endpoint so gateway and fallback
answers never collide (`llm/cache.py`); `ARCTIC_LITELLM_API_KEY` missing raises before any request;
429/5xx retried 3 times honouring `Retry-After` (capped at 60 s); 400 mentioning `response_format`
or `seed` disables that feature for the run and records the change in `run_status.json`; Alpaca news
uses `created_at` and the snapshot drops anything after the 16:00 ET cutoff; prices and news never
exceed the cutoff (`test_no_future_data` and my reading agree).

Slurm scripts: `common.sh` refuses `TBD` values; every sbatch file writes to `logs/arctic/` and the
paths printed by `submit_weekend.sh` match the `#SBATCH --output` lines; `--export=ALL` carries
`PORTSTUDY_WEEK`, `PORTSTUDY_FROM`, `PORTSTUDY_GATEWAY_FALLBACK`; `afterok` makes a Kronos failure stop
the weekend job and the script says so. Problems: **H3** (login mode J4 gate) and **H4** (no J8).
`shuffle_array.sbatch` is correct (`--shard-index $SLURM_ARRAY_TASK_ID`, idempotent shard files).
`llm_fallback.sbatch` writes the endpoint file atomically after starting the servers but before they
are ready; `run_weekend.py --gateway-fallback` started too early would get connection errors (M1
applies). Add a readiness poll on `/v1/models` before writing the file.

CI (`.gitlab-ci.yml`): lint, type check, unit on every push; integration and e2e on master; slow
manual; `GIT_DEPTH: "0"` for the phase-checklist test. Correct. One gap: `yamllint` in CI skips
`config/study_test.yaml`? No: it is included since phase 13. OK.

---

## 5. Things I confirmed are right (so you do not have to re-check them)

- Portfolio identity and counts: `(models x prompt kinds + 1) x 5 + 2`; controls never read views
  (`decision_source = control`, `use_llm_view = False`, `view_shift = 0`), checked for every week in e2e.
- Dropped LLM view (`IC_L = 0` or no valid outputs) gives weights byte-identical to the strategy's Q0
  (J7 copies the file).
- Failed source trades Q0 with `decision_source = fallback_q0` and the week is excluded from IC history
  and from every hypothesis test (`ic_tables(..., clean_weeks)`, `clean_mask`).
- Determinism: two full e2e runs produce the same SHA-256 over every output CSV
  (`48d8385d...98c`, recorded in PROGRESS.md).
- Lock deadline: 09:00 ET Monday, DST-aware (`ZoneInfo`), N1 margin 30 min before it.
- Atomic writes everywhere (`os.replace` of a temp file in the same directory).
- Cached and uncached cvxpy problems agree to 1e-9 (phase 9 test); the cache key contains tickers,
  Sigma hash, delta, cap (+ M hash, week, seed, scenario parameters for CVaR/Robust).
- Holm is applied within the configured family only; controls are excluded from every family.
- No em dash, no banned word, no secret pattern, no stale `qwen/ollama/confidence` in tracked
  `src/` or `config/`.

---

## 6. Suggested order of fixes

1. **H1** ledger forward fill (10 lines, one test). Do this before any live week: the ledger is
   append-only, so a fabricated row written in week 1 stays for the whole study.
2. **H4** run `step1_catch_up` regardless of `--from`, and let `--from J8` run J8.
3. **H2** write the no-rebalance plan and lock when J1/J2 fail; make `build_plans` fill lock-less
   weeks with `hold`.
4. **H3** wait for the Kronos job before J4 in login mode (or drop the J2 gate in J4).
5. **M1/M2/M3** re-queue provider errors with a budget, abort on auth errors, mark non-eligible
   models under the fallback flag.
6. **M4** `current_week` with `ZoneInfo`.
7. **M5** check outputs on the skip path.
8. The rest as time allows; none affects results on the clean path.

Nothing in this file changes a DECISION value. No code was edited; the only file written is
`results.md`.

---

## Pass 2 (deeper, file by file)

Second pass after the first write-up. Everything below is new; numbering continues (H/M/L).

### Pass 2 HIGH / MEDIUM

#### M10. `current_week` rolls to the new week at Friday 00:00, not at the 16:00 ET cutoff (`pipeline/jobs.py` current_week)

`current_week(now)` picks the week whose Friday is at or before `now.date()`. On Friday before
the close, that is the *current* Friday, so `run_weekend.py` started on Friday at 15:00 ET
snapshots a week whose cutoff day has not closed: Alpaca returns no (or a partial) daily bar for
Friday, `check_prices` flags `stopped_trading`/`missing_days` for every stock, and the snapshot is
frozen without Friday's close (snapshots are immutable once written). Proof from the calendar run:

```
Fri 2026-03-13 15:00 -> 2026-W11      (cutoff 2026-03-13 16:00 has not happened)
Fri 2026-03-13 17:00 -> 2026-W11
```

Fix: in `current_week`, roll forward only when `now >= cutoff_datetime_et(cutoff_day(week))`
(use the ET zone, which also fixes M4), and have `run_weekend.py` refuse to run J1 before the
cutoff unless `--week` is given explicitly. The docs say "start Friday night", so this only bites
on an early start, but the result (a frozen bad snapshot) is expensive.

#### M11. Shuffle p-values compare a fresh-start replay with a real portfolio that did not start fresh when weeks are excluded (`eval/shuffle.py` real_metrics, `_metrics`)

`real_metrics` filters the real `daily.csv` to `weeks` and then `_metrics` computes
`total_return = last value / initial_value - 1`; the shuffled replays start at `initial_value` on
the first week in `weeks`. `load_eval_data` removes `timing.pilot_week` from `weeks`, so once a
pilot week is set the real total return includes the pilot week's P&L and the shuffles do not; the
same happens for any `weeks` subset passed to `shuffle_test`. The Sharpe comparison is unaffected
(weekly returns), the total-return p-value is biased. Fix: compute the real total return as
`last / first-day-previous-value - 1` over the selected weeks (or `prod(1 + weekly) - 1`), which
is what `weekly_portfolio_returns` already gives.

### Pass 2 confirmations (calendar, report, shuffle machinery, context)

- Calendar: ISO year boundaries (`2026-W52 -> 2026-W53 -> 2027-W01`, `2024-W52 -> 2025-W01`), a
  Good Friday cutoff (Thursday 2026-04-02), Monday holidays (trade Tuesday 2026-01-20, 2026-05-26,
  2026-09-08), Christmas week (cutoff Thu 2026-12-24, trade Mon 2026-12-28), and
  `week_id(cutoff_day(w)) == w` for 60 consecutive weeks all behave; years outside 2024-2027 raise
  a clear error unless `pandas_market_calendars` is installed; 251 trading days in 2026 (correct).
- `shuffle_table`: `p = (1 + #{shuffle >= real}) / (n + 1)`, Holm within strategy including that
  strategy's Q0, controls never shuffled; shard files are complete only with the marker and the
  right count; `collect` raises `MissingShards` naming every gap; `shuffle_portfolios` dedupes.
- Report: header prints K1/S4/S5/G3b/P1 with the config value and the decision text; the
  `view_shift < 0.01` warning is per strategy on steered portfolios only; every table listed in
  the plan is written as CSV and Markdown; figures closed after saving (no handle leak).
- `RunContext.models()`/`prompt_ids()` fail loudly on unknown ids; `live_start_week` raises in live
  mode when unset; `run_registry` keeps non-selected models out of the inputs hash.

### Pass 2 LOW and improvements

- **L13** `scripts/estimate_cost.py`: `planned_calls = len(registry.enabled()) * len(manifest.enabled()) * n_stocks`
  ignores `llm.model_ids` / `llm.prompt_ids`, so with a selected subset the estimate is too high
  (today 7 models x 7 prompts, while `study.yaml` selects two gateway models). Use
  `RunContext.sources()`.
- **L14** `data/providers/http.py` `LiveHttp._get` calls `httpx.get(...)` per request (new
  connection every call, no keep-alive) while `LiteLLMProvider` keeps an `httpx.Client`. Reuse a
  client; it also makes the rate limiter the only throttle, as intended.
- **L15** `llm/cost.py` `estimate_week_cost` assumes every answer uses `max_output_tokens` (300 by
  default, 2048 for MiniMax with reasoning). Conservative by design, but for the cost-cap check it
  means the gateway-only study always projects 0 USD (prices 0) while an API model projects at the
  cap; document that the guard is an upper bound.
- **L16** `data/providers/yfinance_provider.py` `get_prices` returns `_rows` as both raw and adjusted
  (whatever `_rows` built); I could not verify offline that `adj_close` is the dividend-and-split
  adjusted series and `close` the raw one, because yfinance's `auto_adjust` default changed in 2024.
  Add to the phase 15 smoke test: for a stock with a recent split, assert `close != adj_close`
  before the split date.
- Improvement: `WeekRunner._update_open_weight_status` and `fb.fallback_weeks` (L5, L6) are the two
  hot spots that will matter at 300 stocks x 51 sources; both are one-line caches.
- Improvement: `run_job` could record `outputs_hash` *and* verify it on the skip path (M5); with
  that, `portstudy status` can also report "done but outputs missing".
- Improvement: a single `portstudy doctor` command that runs the health probe, checks the
  fallback endpoint file freshness, the ledger for `volume == 0` rows (H1), the presence of
  `locked.json` for every calendar week since `live_start_week` (H2), and the last IC row date
  (H4) would make all four HIGH items visible in one line each weekend.

### Pass 2 confirmations (providers, prompts, cost, backup, tests)

- Alpaca: raw bars for the ledger, `adjustment=all` close merged as `adj_close` (falls back to the
  raw close only when the adjusted row is missing); split ratio `new_rate / old_rate`; news uses
  `created_at`, 50 per page, paginated; the http layer retries 429/5xx with backoff and a sliding
  60-second rate limiter; fixtures are keyed without headers so keys never enter fixture names.
- Synthetic providers filter strictly by `[start, end]` dates and `(start_utc, end_utc]` news
  timestamps; the synthetic market cap uses the latest adjusted close at or before `as_of` (no
  look-ahead).
- Prompts: shared prefix (base instructions + skills/placebo file) is byte-identical across stocks
  so provider caching works; the stock part holds only ticker, name, sector, news and the no-news
  sentence; `prompt_hash` covers the full text and `prompt_file_sha256` the skills file; the
  manifest check refuses a file whose hash changed without a version bump.
- P1 check: each skills file within 5 percent of the mean skills token length and the placebo
  within 2 percent of its match, per model tokenizer, as the plan states.
- Backup: dated folder never overwritten, checksums for every file, zip, optional rsync that never
  raises, master exports rebuilt from the weekly files on every backup, `verify_backup` detects
  missing, changed and unexpected files.
- Tests: 212 unit, 34 integration, 17 e2e; no tautological assertions found (the only bare
  `is not None` checks guard a following attribute access); the e2e covers every strategy, both
  controls, four injected failure kinds, the planted delisting, eval and a sharded shuffle; live
  providers are exercised only through recorded fixtures.
