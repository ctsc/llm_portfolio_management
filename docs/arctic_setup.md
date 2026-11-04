# ARCTIC setup (v2 phase 12)

Everything cluster-specific lives in `scripts/arctic/arctic.env` (copy of
`arctic.env.example`). Every value starts as `TBD` and the scripts refuse to run until it is
filled in; that happens in v2 phase 15 C once the allocation is known.

| Variable | Meaning | Filled in |
|---|---|---|
| `SLURM_ACCOUNT` | `sbatch --account` | phase 15 C |
| `GPU_PARTITION`, `GPU_GRES` | partition and GRES for Kronos and the LLM fallback server | phase 15 C |
| `CPU_PARTITION` | partition for the weekend pipeline and the shuffle array | phase 15 C |
| `KRONOS_WALLTIME`, `WEEKEND_WALLTIME`, `SHUFFLE_WALLTIME`, `LLM_FALLBACK_WALLTIME` | Slurm walltimes; `submit_weekend.sh` warns when the weekend walltime ends after the Monday lock | phase 15 C |
| `PYTHON_MODULE`, `VENV_PATH` | `module load` name and the virtualenv with `portstudy` installed | phase 15 C |
| `DATA_ROOT` | exported as `PORTSTUDY_DATA_DIR` | phase 15 C |
| `HF_HOME` | Hugging Face cache; fallback model weights are downloaded here in advance | phase 15 C |
| `ARCTIC_NETWORK_MODE` | `compute` when compute nodes reach the internet, `login` when only the login node does | phase 15 C |
| `SHUFFLE_MAX_PARALLEL` | array throttle for `shuffle_array.sbatch` | phase 15 C |

## Scripts

| Script | Runs |
|---|---|
| `submit_weekend.sh [--week W] [--gateway-fallback] [--dry-run]` | the whole weekend with one command; prints job ids, log paths, and the exact rerun command for each job |
| `kronos.sbatch` | J2 (with J1 first in compute mode) on a GPU node |
| `weekend.sbatch` | `run_weekend.py --from J3` (compute mode) or `--from J6` (login mode); `PORTSTUDY_FROM` overrides |
| `shuffle_array.sbatch` | one shard per array task via `portstudy shuffle run --shard-index $SLURM_ARRAY_TASK_ID` |
| `llm_fallback.sbatch` | the DECISION(D2) fallback server for the fallback-eligible models; writes `data/runtime/fallback_endpoint.json` |
| `common.sh` | shared loader; refuses `TBD` values |

Network layouts: in `compute` mode kronos is submitted first and the weekend job waits on it
with `--dependency=afterok`, so a kronos failure means the weekend job never starts and the
summary says so. In `login` mode J1 runs on the login node, kronos is submitted, J3 to J5 run in
the foreground on the login node (`run_weekend.py --from J3 --until J5`), and the weekend job
(J6 to J9) is submitted with the same dependency.

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

Set `backup.rsync_target` (for example `user@laptop:/backups/portstudy`) and J9 runs
`rsync -a --partial` of the dated backup folder and its zip after writing them; a failed rsync is
printed and logged, never raised. Serving tool for the fallback server: vLLM assumed
(`# UNVERIFIED`), chosen at setup.
