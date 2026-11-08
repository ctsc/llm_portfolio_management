#!/usr/bin/env bash
# One command for the weekend on ARCTIC. Reads scripts/arctic/arctic.env.
#   compute mode: sbatch kronos, then sbatch weekend with --dependency=afterok.
#   login mode:   J1 here, sbatch kronos, J3 to J5 here in the foreground (API submit, gateway
#                 calls, API collect), then sbatch weekend (J6 to J9) after kronos finishes.
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

require_set SLURM_ACCOUNT GPU_PARTITION CPU_PARTITION GPU_GRES KRONOS_WALLTIME \
  WEEKEND_WALLTIME ARCTIC_NETWORK_MODE
activate_env

LOGS="$(log_dir)"
export PORTSTUDY_WEEK="${WEEK}"
export PORTSTUDY_GATEWAY_FALLBACK="${GATEWAY_FALLBACK}"
WEEK_ARG=()
if [[ -n "${WEEK}" ]]; then WEEK_ARG=(--week "${WEEK}"); fi
FALLBACK_ARG=()
if [[ "${GATEWAY_FALLBACK}" == "1" ]]; then FALLBACK_ARG=(--gateway-fallback); fi

# Warn when the weekend walltime ends after the Monday lock deadline.
python scripts/run_weekend.py --dry-run --check-walltime "${WEEKEND_WALLTIME}" "${WEEK_ARG[@]}" || true

run_or_echo() {
  if [[ "${DRY_RUN}" == "1" ]]; then echo "+ $*"; else "$@"; fi
}

wait_for_job() {  # block until a Slurm job is no longer queued or running
  local job_id="$1"
  if [[ "${DRY_RUN}" == "1" ]]; then echo "+ wait for job ${job_id}"; return; fi
  echo "waiting for job ${job_id} (J2, Kronos) before J4"
  while squeue -h -j "${job_id}" 2>/dev/null | grep -q .; do sleep "${WAIT_POLL_SECONDS:-30}"; done
  echo "job ${job_id} finished; see ${LOGS}/kronos-${job_id}.out"
}

submit() {  # prints the job id only
  if [[ "${DRY_RUN}" == "1" ]]; then echo "+ sbatch $*" >&2; echo "DRYRUN"; return; fi
  sbatch --parsable "$@"
}

KRONOS_FLAGS=(--account="${SLURM_ACCOUNT}" --partition="${GPU_PARTITION}" --gres="${GPU_GRES}"
  --time="${KRONOS_WALLTIME}" --export=ALL)
WEEKEND_FLAGS=(--account="${SLURM_ACCOUNT}" --partition="${CPU_PARTITION}"
  --time="${WEEKEND_WALLTIME}" --export=ALL)

if [[ "${ARCTIC_NETWORK_MODE}" == "compute" ]]; then
  KRONOS_ID="$(submit "${KRONOS_FLAGS[@]}" "${ARCTIC_DIR}/kronos.sbatch")"
  WEEKEND_ID="$(submit "${WEEKEND_FLAGS[@]}" --dependency=afterok:"${KRONOS_ID}" "${ARCTIC_DIR}/weekend.sbatch")"
  echo "kronos job ${KRONOS_ID}: ${LOGS}/kronos-${KRONOS_ID}.out"
  echo "  rerun: sbatch ${KRONOS_FLAGS[*]} ${ARCTIC_DIR}/kronos.sbatch"
  echo "weekend job ${WEEKEND_ID} (after kronos): ${LOGS}/weekend-${WEEKEND_ID}.out"
  echo "  rerun: sbatch ${WEEKEND_FLAGS[*]} ${ARCTIC_DIR}/weekend.sbatch   (set PORTSTUDY_FROM=J7 to resume)"
elif [[ "${ARCTIC_NETWORK_MODE}" == "login" ]]; then
  run_or_echo portstudy run-job J1 "${WEEK_ARG[@]}"
  KRONOS_ID="$(submit "${KRONOS_FLAGS[@]}" "${ARCTIC_DIR}/kronos.sbatch")"
  echo "kronos job ${KRONOS_ID}: ${LOGS}/kronos-${KRONOS_ID}.out"
  # J3 (API batch submit) does not need the baseline; J4 does (review fix H3): wait for the
  # Kronos job to leave the queue before the gateway calls start.
  run_or_echo python scripts/run_weekend.py --from J3 --until J3 "${WEEK_ARG[@]}" "${FALLBACK_ARG[@]}"
  wait_for_job "${KRONOS_ID}"
  run_or_echo python scripts/run_weekend.py --from J4 --until J5 "${WEEK_ARG[@]}" "${FALLBACK_ARG[@]}"
  WEEKEND_ID="$(submit "${WEEKEND_FLAGS[@]}" --dependency=afterok:"${KRONOS_ID}" "${ARCTIC_DIR}/weekend.sbatch")"
  echo "weekend job ${WEEKEND_ID} (J6 to J9 after kronos): ${LOGS}/weekend-${WEEKEND_ID}.out"
  echo "  rerun: PORTSTUDY_FROM=J6 sbatch ${WEEKEND_FLAGS[*]} ${ARCTIC_DIR}/weekend.sbatch"
else
  echo "ARCTIC_NETWORK_MODE must be compute or login, got ${ARCTIC_NETWORK_MODE}" >&2
  exit 2
fi
echo "if kronos fails the weekend job never starts (afterok); fix and rerun both commands above"
