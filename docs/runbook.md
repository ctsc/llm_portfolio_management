# Runbook: for a human on a bad Saturday

Everything below assumes the repository root as the working directory and the virtual
environment activated. Add `--synthetic` to any command to work on the synthetic world. Use
`--config PATH` to select a study file, or set `PORTSTUDY_CONFIG` for the default.

## 1. Where things are

| What | Where |
|---|---|
| Job status files | `data/outputs/<week>/jobs/<job>.status.json` (`state`, `started_at`, `finished_at`, `error`) |
| Logs (JSON lines) | `data/logs/<job>/<date>.jsonl` |
| Snapshot | `data/snapshots/<week>/` (CSV files plus `manifest.json`) |
| Baseline, prior, covariance | `data/outputs/<week>/baseline.csv`, `prior.csv`, `sigma_matrix.csv` |
| LLM calls | `data/outputs/<week>/llm/<model>__<config>.jsonl` (one line per call), `llm_scores.csv`, `run_status.json`, `cost_log.csv` |
| Archived endpoint records | `data/outputs/<week>/llm/archive/<source>.gateway.jsonl` |
| Batch handles | `data/outputs/<week>/llm/batch_<model>__<config>.json` (deleted once collected) |
| Fallback plan and log | `data/outputs/<week>/fallback_plan.json` (J6), `fallback_log.json` (J7) |
| Weights and posteriors | `data/outputs/<week>/weights/<portfolio>.csv`, `.meta.json`, `locked.json`; `posterior/<source>.csv` and `posterior/Q0.csv` |
| Price ledger | `data/ledger/prices_raw.csv`, `corporate_actions.csv` (append-only) |
| Portfolio records | `data/portfolios/<portfolio>/trades.csv`, `daily.csv` |
| IC history | `data/outputs/ic_history.csv` |
| Backups | `data/backups/<week>_<timestamp>/` and `.zip`; master exports in `data/exports/master/` |
| LLM cache | `data/cache/llm/` |

## 2. Check status

```
portstudy status --week 2026-W46
```

One line per job: `pending`, `running`, `done`, `failed`, `skipped`, with timestamps and the
error text for failures. Strategy summaries show solver failures and mean view shift, with endpoint,
valid rate, and fallback reason per source. `running` after a crash means the process died; rerunning
the job resumes it (every job is idempotent and resumable).

## 3. Rerun one job

```
portstudy run-job J4 --week 2026-W46
portstudy run-job J7 --week 2026-W46 --force      # rerun even if done with the same inputs
```

`scripts/run_weekend.py` prints exactly this command when a job fails. Rerunning
`run_weekend.py` itself resumes where it stopped.

## 4. What each failure means and the fallback that applies

| Failed job | Effect | What to do |
|---|---|---|
| J1 snapshot (data provider down, a check with rule `halt`) | No snapshot. If still failing at the lock, J6 writes `no_rebalance`: every portfolio holds last week's positions, stops still run. | Check the provider status page and the key in `.env`; rerun J1. For a `halt` issue read `issues.csv` in the partial logs and decide whether the stock should be dropped (`data.checks` rules in `study.yaml`). |
| J2 baseline (Kronos, covariance) | Same as J1: `no_rebalance` for all. | Rerun J2. If Kronos cannot load, set `baseline.forecaster: stub` for the week and log the change. |
| J3 API submit (`CostCapExceeded`) | Batches not submitted. The API portfolios will trade their strategy's Q0 weights (`fallback_q0`). | Read the projected total in the log line. Either run `portstudy run-job J3 --week ... --allow-over-budget` (the week is logged as over budget, decision D3) or accept the fallback. |
| J4 open-weight inference (crash or gateway unavailable) | Partial records are saved as calls complete. A model that fails the pre-J4 health check is marked failed for the week (decision D2) and its portfolios trade their strategy's Q0 (`fallback_q0`). If a source is not finished by the lock margin, it does the same. | Check the GSU VPN and `ARCTIC_LITELLM_API_KEY`, then rerun J4 to resume from the cache. There is no fallback server; a model that stays down misses the week and that week is left out of its statistics. |
| J5 API collect (`pending`) | Batches not back yet. `run_weekend.py` polls until the lock margin (Monday 8:30 am ET by default). | Wait, or rerun J5 later. After the margin J6 labels the batch failed and the portfolios trade their strategy's Q0. |
| J6 validate (`runs still incomplete before the lock margin`) | Nothing written yet. | Finish J4 or J5 first, or wait for the margin. |
| J6 says runs are incomplete although J4 finished | Until PLAN4 Phase 3 (decision N5) a gateway request that fails after its retries leaves no record, J4 still shows done, and J6 waits until the lock margin and then sends the source to Q0. | Rerun J4 with `--force` to retry the missing requests. After Phase 3 such a request is recorded as an invalid `provider_error` answer and J4 stays failed until every request has a record. |
| J6 marks `invalid_rate` failures | That rating source trades its strategy's Q0 weights for the week (decision N1). | Nothing to do for the week; investigate the model's raw answers in the jsonl records. If `invalid_truncated` dominates, raise `max_output_tokens` for that model in `llm.settings`. This is expected for MiniMax M2.7 at the default 300 tokens: it always thinks, the thinking counts toward the limit, and the answer comes back empty with `finish_reason: length` (ARCTIC suggests about 2,000). |
| J7 weights (`LockViolation`) | Weights not written because the clock is past Monday 9:00 am ET. | Too late for the week: the simulator holds positions (`no_rebalance`). Record what happened. Never backdate the clock. |
| J7 `solver_failed` | The portfolio keeps last week's weights with the flag. | Check `sigma_matrix.csv` for a bad stock; rerun J7 with `--force` after fixing the data. |
| J8 replay and score | Results lag. | Rerun J8 any time; it rebuilds every portfolio from the ledger and rescores every closed week. |
| J9 backup | No download for the week. | `portstudy backup --week ...` |

Fallback labels are visible in `fallback_log.json`, in `daily.csv` (`week_status`), in
`trades.csv` (`decision_source`), and in the report's fallback-count table. Fallback weeks are
excluded from that portfolio's IC history and from the clean-weeks tests, and kept in the
as-traded records.

## 5. Restore from a backup

```
python - <<'EOF'
from pathlib import Path
from portstudy.backup import verify_backup, restore_backup
src = Path("data/backups/2026-W46_20261114T210000")
print(verify_backup(src))                      # [] means every hash matches
restore_backup(src, Path("data_restored"))     # then point PORTSTUDY_DATA_DIR at it
EOF
```

`restore_backup` refuses a backup that fails verification. Restoring over the live `data/`
folder is also possible, but keep a copy first.

## 6. Prompt or config changes mid-run

Models and code never edit prompt files. If a human changes a prompt, bump its `version` in
`config/prompts_manifest.yaml`, rename the file to carry the version, run
`python scripts/refresh_manifest.py --write`, commit, and note the change in the analysis.
Use `config/study_test.yaml` for a small offline run; the test suite's shared `study_config`
fixture loads it too, so pytest exercises every strategy and control at test scale. Any change to a `decisions.*` value after
the live start must be logged and versioned the same way (see `docs/open_questions.md`).

## 7. Sizing the universe

`universe.size` (DECISION(B4)) is 300 if ARCTIC approves the full gateway request and 100 if it
approves the smaller option. Before freezing it, run the optimizer benchmark at the candidate size:

```
python scripts/benchmark_pipeline.py --n-stocks 300 --sources 3 --weeks 24
```

It prints seconds per solve for every strategy and control, the weekly J7 time at the given and at
the projected source count (`--project-sources`, default 51), and peak memory; the same table is
written to `data/benchmarks/<date>.csv`. Desktop measurements from 2026-11-10 are in
`docs/desktop_setup.md`. On native Windows peak memory prints `n/a` unless `psutil` is installed;
the documented path is WSL2. `--jobs` times J1 to J9 once
on the synthetic world instead. When the universe is built, the log shows `universe_thin_sector`
for every sector that could not fill its quota and `universe_leftover_slots` for the slots filled
by the highest-volume leftovers.

## 8. The desktop weekend

Setup is in `docs/desktop_setup.md`; the per-weekend routine is `docs/weekend_checklist.md`.

```
python scripts/run_weekend.py                      # resumes where it stopped
python scripts/run_weekend.py --from J7            # rerun from one job
```

If the health check before J4 reports a gateway model down, its sources are marked failed and trade
Q0 (D2); check the GSU VPN and run `python scripts/run_weekend.py --from J4` before the lock to
retry that model. A job that dies (sleep,
restart, network) is rerun with the same command; every job resumes from its status files.

**Backups.** J9 writes `data/backups/<week>_<timestamp>/` and a zip, and with `backup.rsync_target`
set copies both with `rsync -a --partial`; a failed rsync is printed and logged, never raised.
Restore as in section 5.

## 9. After the last live week

```
portstudy eval            # report under data/outputs/report/
```

The final evaluation (residual IC, active-return factor regressions on
Kenneth French daily factors, Ledoit-Wolf Sharpe tests, pre-registered main comparisons) is
specified in `plans/PLAN4.md` Phases 5 to 9 and `llm_research/05_evaluation_methodology.md`.

## Where to look (pending PLAN4 Phase 3A, decision N6)

After Phase 3A every weekend run will leave: an event timeline `data/logs/runs/<run_id>.jsonl`
(every job start, end, retry, fallback and resume), job logs `data/logs/<job>/<date>.jsonl` and
library warnings `data/logs/libraries/<date>.jsonl` (all tagged with the `run_id`), status files
with full tracebacks, `data/outputs/<week>/fallbacks.csv` (every degradation and its reason),
`data/outputs/<week>/run_summary.md` (what happened and what needs attention), and the result of
`portstudy audit --week <week>`. Until then, the job logs and status files are the record.

