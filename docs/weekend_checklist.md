# Weekend checklist (ARCTIC)

One pass per study week. Start Friday evening after the 16:00 ET close when possible; a Saturday
start still works but leaves less room for a batch retry before the Monday 09:00 ET weight lock.

## Before

- [ ] On the GSU network or VPN: the LiteLLM gateway (`https://litellm.rs.gsu.edu/v1`) answers
      `python scripts/smoke_real_services.py --dry-run` and `ARCTIC_LITELLM_API_KEY` is in `$HOME/.env`
      (mode 600, never in the repo).
- [ ] `scripts/arctic/arctic.env` is filled (no `TBD`), every walltime is at most 15 hours, and
      `bash scripts/arctic/submit_weekend.sh --dry-run` prints the three sbatch commands.
- [ ] The checkout is on the frozen tag (`git describe --tags` shows `frozen-v2`) and the working
      tree is clean.
- [ ] Last week's backup zip was pulled off the cluster (decision N3) and `$DATA_ROOT` has a few GB free.
- [ ] If the gateway is down: `sbatch scripts/arctic/llm_fallback.sbatch` first and add
      `--gateway-fallback` to the submit command.

## Running

```
bash scripts/arctic/submit_weekend.sh --week <YYYY-Www>
squeue -u $USER                                   # kronos and llm run at once; finish waits on both
sacct -j <finish jobid> --format=JobID,Elapsed,State,ExitCode
```

The finish job polls the API batches once per run and resubmits itself every `POLL_INTERVAL_MINUTES`
minutes until they are collected, then runs J6 to J9. Logs are under `logs/arctic/`. Nothing needs to
stay open on a laptop; the login shell can be closed once the jobs are submitted.

## If something fails

`portstudy status --week <YYYY-Www>` shows the failed job and its error. `docs/runbook.md` section 4
names the fallback that applies and section 9 the rerun command; every job resumes from its status
files, so a job killed by its walltime is rerun from where it stopped.

## After

- [ ] Read the summary at the end of the finish job's log: every portfolio should say `normal`; a
      `fallback` row is expected only after a labeled failure.
- [ ] Pull the backup folder or zip that J9 printed (or confirm the `rsync` target received it).
