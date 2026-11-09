#!/usr/bin/env bash
# One command for the weekend on ARCTIC (PLAN3 phase 3 layout; DECISION(W1), DECISION(M1)).
# Reads scripts/arctic/arctic.env. Three jobs, none longer than MAX_JOB_HOURS:
#   kronos.sbatch  GPU: J1 (compute mode) and J2.
#   llm.sbatch     CPU, 2 cores: J3 then J4 (compute mode) or J4 only (login mode, J3 ran here).
#                  No dependency on kronos: the LLM calls do not use the Kronos output.
#   finish.sbatch  CPU, 1 core, after both (afterany): one J5 poll, then J6 to J9; while API
#                  batches are pending it resubmits itself POLL_INTERVAL_MINUTES later.
#   compute mode: everything above is submitted from here.
#   login mode:   J1 and J3 run on the login node first (only it reaches the data and API
#                 providers); batch polling never runs in the login shell.
# Usage: scripts/arctic/submit_weekend.sh [--week 2026-W46] [--gateway-fallback] [--dry-run]
set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

WEEK=""
GATEWAY_FALLBACK=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --week) WEEK="$2"; shift 2 ;;
    --gateway-fallback) GATEWAY_FALLBACK=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

require_set SLURM_ACCOUNT GPU_PARTITION CPU_PARTITION GPU_GRES GPU_MEM ARCTIC_NETWORK_MODE
check_walltime KRONOS_WALLTIME
check_walltime LLM_WALLTIME
check_walltime FINISH_WALLTIME
activate_env

LOGS="$(log_dir)"
export PORTSTUDY_WEEK="${WEEK}"
export PORTSTUDY_GATEWAY_FALLBACK="${GATEWAY_FALLBACK}"
WEEK_ARG=()
if [[ -n "${WEEK}" ]]; then WEEK_ARG=(--week "${WEEK}"); fi
FALLBACK_ARG=()
if [[ "${GATEWAY_FALLBACK}" == "1" ]]; then FALLBACK_ARG=(--gateway-fallback); fi

# Warn when the longest chain (LLM job then one finish job) ends after the Monday lock deadline.
python scripts/run_weekend.py --dry-run --check-walltime "${LLM_WALLTIME}" "${WEEK_ARG[@]}" || true

run_or_echo() {
  if [[ "${DRY_RUN}" == "1" ]]; then echo "+ $*"; else "$@"; fi
}

submit() {  # prints the job id only
  if [[ "${DRY_RUN}" == "1" ]]; then echo "+ sbatch $*" >&2; echo "DRYRUN"; return; fi
  sbatch --parsable "$@"
}

KRONOS_FLAGS=(--account="${SLURM_ACCOUNT}" --partition="${GPU_PARTITION}" --gres="${GPU_GRES}"
  --mem="${GPU_MEM}" --time="${KRONOS_WALLTIME}" --export=ALL)
LLM_FLAGS=(--account="${SLURM_ACCOUNT}" --partition="${CPU_PARTITION}" --cpus-per-task=2
  --mem-per-cpu=3G --time="${LLM_WALLTIME}" --export=ALL)
FINISH_FLAGS=(--account="${SLURM_ACCOUNT}" --partition="${CPU_PARTITION}" --cpus-per-task=1
  --mem-per-cpu=3G --time="${FINISH_WALLTIME}")
export PORTSTUDY_FINISH_FLAGS="${FINISH_FLAGS[*]}"   # finish.sbatch resubmits itself with these

case "${ARCTIC_NETWORK_MODE}" in
  compute) ;;
  login)
    # Only the login node reaches the data providers and the API providers: J1 (data) and J3
    # (API batch submit) run here, in the foreground, before anything is queued.
    run_or_echo portstudy run-job J1 "${WEEK_ARG[@]}"
    run_or_echo python scripts/run_weekend.py --from J3 --until J3 "${WEEK_ARG[@]}" "${FALLBACK_ARG[@]}"
    ;;
  *) echo "ARCTIC_NETWORK_MODE must be compute or login, got ${ARCTIC_NETWORK_MODE}" >&2; exit 2 ;;
esac

KRONOS_ID="$(submit "${KRONOS_FLAGS[@]}" "${ARCTIC_DIR}/kronos.sbatch")"
LLM_ID="$(submit "${LLM_FLAGS[@]}" "${ARCTIC_DIR}/llm.sbatch")"
FINISH_ID="$(submit "${FINISH_FLAGS[@]}" --export=ALL --dependency=afterany:"${KRONOS_ID}":"${LLM_ID}" "${ARCTIC_DIR}/finish.sbatch")"
echo "kronos job ${KRONOS_ID} (GPU, ${KRONOS_WALLTIME}): ${LOGS}/kronos-${KRONOS_ID}.out"
echo "  rerun: sbatch ${KRONOS_FLAGS[*]} ${ARCTIC_DIR}/kronos.sbatch"
echo "llm job ${LLM_ID} (2 cores x 3G, ${LLM_WALLTIME}): ${LOGS}/llm-${LLM_ID}.out"
echo "  rerun: sbatch ${LLM_FLAGS[*]} ${ARCTIC_DIR}/llm.sbatch   (set PORTSTUDY_FROM=J4 to skip J3)"
echo "finish job ${FINISH_ID} (1 core x 3G, ${FINISH_WALLTIME}, after both): ${LOGS}/finish-${FINISH_ID}.out"
echo "  rerun: sbatch ${FINISH_FLAGS[*]} --export=ALL ${ARCTIC_DIR}/finish.sbatch"
echo "finish runs even if kronos or llm failed (afterany): J6 plans the hold or the fallbacks;"
echo "while API batches are pending it resubmits itself every ${POLL_INTERVAL_MINUTES:-60} minutes"
echo "and stops at the lock margin. Watch with: squeue -u \$USER; sacct -j ${FINISH_ID} --format=JobID,State,Elapsed,MaxRSS"
