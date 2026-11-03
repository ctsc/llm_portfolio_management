# Desktop checklist before each weekend run

Start Friday night when possible. A Saturday morning start still works, but it leaves a smaller
buffer for one batch retry before the Monday 9:00 am ET weight lock (`run_weekend.py` prints a
warning when started on a Saturday).

## Before starting

- [ ] Power: plugged in; Windows power plan set so the machine never sleeps while the script runs.
- [ ] Windows Update paused for the weekend (Settings, Windows Update, Pause updates).
- [ ] ARCTIC LiteLLM gateway URL and `ARCTIC_LITELLM_API_KEY` are provisioned.
- [ ] If using `--gateway-fallback`, `data/runtime/fallback_endpoint.json` contains a fresh
      `base_url` and optional matching `week_id`; `ARCTIC_FALLBACK_API_KEY` may be empty.
- [ ] `.env` on the desktop holds the keys named in `docs/config_reference.md`
      (ARCTIC, Anthropic, OpenAI, Alpaca, Finnhub, FRED). Never commit this file.
- [ ] Spending caps are set in the Anthropic and OpenAI dashboards.
- [ ] `vendor/kronos` is at the pinned commit and the Kronos weights are downloaded, if
      `baseline.forecaster` is `kronos`.
- [ ] Free disk space: at least a few GB under `data/` (snapshots, backups, and the cache grow
      every week).
- [ ] The repository is on the frozen tag (`git describe --tags`) and the working tree is clean.
- [ ] Last week's backup zip was downloaded (decision N3).

## Running

```
python scripts/run_weekend.py
```

Watch the plain log lines. The script prints each job's start, finish, counts (stocks, calls,
invalid rate, cached tokens, cost so far) and any data check issue. While API batches are
pending it shows a countdown to the lock margin.

## If something fails

The script stops, prints the failed job, the fallback that will apply, and the exact command to
rerun that one job. Fix the cause (see `docs/runbook.md`), then rerun `run_weekend.py`; it
resumes where it stopped.

## After the run

- [ ] Read the summary table: every portfolio should say `normal`; a `fallback` row is expected
      only after a labeled failure.
- [ ] Download the backup folder or zip the script printed.
- [ ] Leave the desktop on through the Monday open only if a rerun is still pending; the weight
      files are locked once J7 has written them.
