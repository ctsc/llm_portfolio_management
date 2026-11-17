# Weekend checklist (desktop)

One pass per study week, on the desktop in WSL2 (`docs/desktop_setup.md`). Start Friday evening
after the 16:00 ET close when possible; a Saturday morning start still works but leaves less room
for a batch retry before the Monday 09:00 ET weight lock.

## Before

- [ ] Desktop awake with sleep set to Never; Windows Update paused (renew the pause before it ends).
- [ ] GSU VPN connected (the LiteLLM gateway `https://litellm.rs.gsu.edu/v1` needs it; `curl -sS https://litellm.rs.gsu.edu/health/liveliness` prints "I'm alive!"). The gateway is an alpha service, so read its status in the J4 log before trusting a week. After PLAN4 Phase 3A, also read the canary result (`model_change` flag) in the weekend summary.
- [ ] GPU-heavy apps closed (Kronos needs most of the 6 GB).
- [ ] In Ubuntu: `source ~/.local/bin/env && source ~/pm311/bin/activate && cd ~/pm`.
- [ ] The checkout is on the frozen tag (`git describe --tags`) and `git status` is clean.
- [ ] `~/.env` holds every key (`ARCTIC_LITELLM_API_KEY`, API and data keys); never in the repo.
- [ ] Last week's backup was copied off the desktop and `data/` has a few GB free.

## Running

```bash
python scripts/run_weekend.py
```

It replays and backs up any earlier week that is behind, runs J1 to J6 (polling API batches until
they are back or the lock margin), then J7 and J9. Before J4 a health check sends one tiny call per
gateway model; a model that fails is marked failed for the week and its portfolios trade their
strategy's Q0 (decision D2). If the script stops, it prints what failed, the fallback that will
apply, and the exact rerun command; rerunning `run_weekend.py` resumes where it stopped.

Every live run starts with the readiness check (`python scripts/check_live_ready.py` runs it
alone). It lists every problem and stops the run with exit code 2 while prompts, the Kronos
commit, model versions, prices, dates or keys are still placeholders, the gateway does not answer,
the data folder holds a week before the pilot, or the config changed since the pilot.
`--skip-preflight` overrides it; `--synthetic` never runs it.

## After

- [ ] The summary shows every portfolio with a weights file and `locked.json` exists for the week.
- [ ] `portstudy status --week <YYYY-Www>` shows no `failed` job, or the failure and its fallback
      are understood (`docs/runbook.md`).
- [ ] Fallback and invalid-output rates per model look normal.
- [ ] The J9 backup zip is copied off the desktop.
- [ ] Anything unusual is written down for the analysis.
