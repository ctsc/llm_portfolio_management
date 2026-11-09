# ARCTIC setup

This is my setup list for running the study on ARCTIC. Everything cluster-specific lives in
`scripts/arctic/arctic.env`, my copy of `arctic.env.example`. Every site value starts as `TBD` and
the scripts refuse to run until I fill it in; I do that in PLAN2 phase 15 C once I know the
allocation. Two rules from PLAN3 shape every job: W1, no Slurm job asks for more than 15 hours
(target 12 or less) and none waits idle on anything outside the cluster; M1, CPU jobs request
`--mem-per-cpu=3G` and only the cores they use. I confirm the real walltime limit and memory
policy with ARCTIC in PLAN2 phase 15 A and lower `MAX_JOB_HOURS` if the limit is lower.

## My checklist

1. **Environment file.** `cp scripts/arctic/arctic.env.example scripts/arctic/arctic.env`, then fill
   every row of the table below from the allocation email and `sinfo`. The file is ignored by git.
2. **Python.** `module load $PYTHON_MODULE`, `python -m venv $VENV_PATH`, `source $VENV_PATH/bin/activate`,
   `pip install -e ".[dev]"` from the checkout, then `pytest tests/unit -q` once on the login node.
3. **Model weights in `HF_HOME`.** On a node with network access I run `huggingface-cli download` for the
   Kronos checkpoint and for every model with `fallback_eligible: true` in `config/models.yaml`, so
   `kronos.sbatch` and `llm_fallback.sbatch` never download on a compute node.
4. **Licenses.** Gemma and Llama are gated on Hugging Face: I accept each license with my account and log
   in with `huggingface-cli login` before step 3, or the downloads fail with 401.
5. **Gateway access.** The LiteLLM gateway is `https://litellm.rs.gsu.edu/v1` (`llm.gateway.base_url`)
   and needs the GSU network or VPN plus my key in `ARCTIC_LITELLM_API_KEY` (in `$HOME/.env`, mode 600,
   never in the repo); `python scripts/smoke_real_services.py --dry-run` prints the payloads, and the
   live smoke test confirms one call per model.
6. **Network mode.** I check whether a compute node can reach the gateway (`srun --pty curl -sI $base_url`);
   if not, `ARCTIC_NETWORK_MODE=login`.
7. **Quotas.** I check `$DATA_ROOT` space (the 30-week synthetic run is a few hundred MB; the live run with
   backups and shuffle shards needs a few GB), the GPU hours for Kronos (one short job per week plus the
   fallback server when needed), and the CPU hours for the shuffle against `scripts/benchmark_pipeline.py`.
8. **Dry run.** `bash scripts/arctic/submit_weekend.sh --dry-run` prints the sbatch commands and the
   walltime check without submitting.

| Variable | Meaning | Filled in |
|---|---|---|
| `SLURM_ACCOUNT` | `sbatch --account` | phase 15 C |
| `GPU_PARTITION`, `GPU_GRES`, `GPU_MEM` | partition, GRES and host memory (`sbatch --mem`) for Kronos and the LLM fallback server | phase 15 C |
| `CPU_PARTITION` | partition for the LLM, finish and shuffle jobs | phase 15 C |
| `KRONOS_WALLTIME`, `LLM_WALLTIME`, `SHUFFLE_WALLTIME`, `LLM_FALLBACK_WALLTIME` | Slurm walltimes, each at most `MAX_JOB_HOURS`; `submit_weekend.sh` warns when the LLM walltime ends after the Monday lock | phase 15 C |
| `FINISH_WALLTIME`, `POLL_INTERVAL_MINUTES`, `MAX_JOB_HOURS` | finish job walltime (03:00:00), its resubmission delay while batches are pending (60), and the W1 ceiling (15) | defaults set |
| `PYTHON_MODULE`, `VENV_PATH` | `module load` name and the virtualenv with `portstudy` installed | phase 15 C |
| `DATA_ROOT` | exported as `PORTSTUDY_DATA_DIR` | phase 15 C |
| `HF_HOME` | Hugging Face cache; fallback model weights are downloaded here in advance | phase 15 C |
| `ARCTIC_NETWORK_MODE` | `compute` when compute nodes reach the internet, `login` when only the login node does | phase 15 C |
| `SHUFFLE_MAX_PARALLEL` | array throttle for `shuffle_array.sbatch` | phase 15 C |

## Scripts

| Script | Runs |
|---|---|
| `submit_weekend.sh [--week W] [--gateway-fallback] [--dry-run]` | the whole weekend with one command; prints job ids, log paths, and the exact rerun command for each job |
| `kronos.sbatch` | J2 (with J1 first in compute mode) on a GPU node, `--mem=$GPU_MEM` |
| `llm.sbatch` | J3 then J4 (compute mode) or J4 only (login mode), 2 cores, `--mem-per-cpu=3G`; `PORTSTUDY_FROM` overrides |
| `finish.sbatch` | 1 core, `--mem-per-cpu=3G`: `run_weekend.py --from J5 --until J5 --poll-once`; exit 3 resubmits itself `POLL_INTERVAL_MINUTES` later, exit 0 runs J6 to J9 |
| `shuffle_array.sbatch` | one shard per array task on one core via `portstudy shuffle run --shard-index $SLURM_ARRAY_TASK_ID` |
| `llm_fallback.sbatch` | the DECISION(D2) fallback server for the fallback-eligible models; writes `data/runtime/fallback_endpoint.json` |
| `common.sh` | shared loader; refuses `TBD` values and any walltime above `MAX_JOB_HOURS`, warns above 12 hours |

Network layouts: in `compute` mode kronos and llm are submitted at the same time with no
dependency (J4 never reads the Kronos output) and finish is submitted with
`--dependency=afterany` on both, so it runs even if one of them failed and J6 or J7 records the
hold. In `login` mode J1 and J3 run on the login node first, then kronos (J2), llm (J4 only) and
finish are submitted the same way; the login shell never polls a batch.

## Gateway health check and the D2 fallback

Before J4, `run_weekend.py` sends one tiny prompt per enabled open-weight model served by the
gateway. On a failure without `--gateway-fallback` it prints the D2 steps, marks every source of
the failing models failed in `run_status.json` (reason `gateway_health_check_failed`), writes
`gateway_health.json`, and continues; J4 skips those models and J6 serves their strategy's Q0.
With the flag, fallback-eligible models switch to the endpoint in the fallback file for this
week (single-endpoint rule from v2 phase 4) and the other models stay on the gateway. A fallback
file older than the source run's start time, or for another week, is rejected.

Walltime kill mid-J7: `python scripts/run_weekend.py --from J7` (idempotent). A gateway that
passes the check but fails mid-J4 leaves the affected sources failed by N1, or rerun them with
`--from J4 --gateway-fallback` once the fallback server is up.

## Pulling results off the cluster

I set `backup.rsync_target` (for example `user@laptop:/backups/portstudy`) and J9 runs
`rsync -a --partial` of the dated backup folder and its zip after writing them; a failed rsync is
printed and logged, never raised. I have not verified the serving tool for the fallback server: vLLM is assumed
(`# UNVERIFIED`) and I choose it at setup.
