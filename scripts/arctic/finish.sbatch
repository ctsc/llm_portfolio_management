#!/usr/bin/env bash
#SBATCH --job-name=portstudy-finish
#SBATCH --output=logs/arctic/finish-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
# One J5 poll, then J6 to J9 in the same job (PLAN3 phase 3, DECISION(W1)). Runs after the Kronos
# and LLM jobs. If API batches are still pending it exits 3 and resubmits itself to start
# POLL_INTERVAL_MINUTES later, so no job sits idle waiting on the providers. Set
# PORTSTUDY_FINISH_FLAGS to the sbatch flags used for the resubmission (submit_weekend.sh does).
set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env
POLL_INTERVAL_MINUTES="${POLL_INTERVAL_MINUTES:-60}"

ARGS=()
if [[ -n "${PORTSTUDY_WEEK:-}" ]]; then ARGS+=(--week "${PORTSTUDY_WEEK}"); fi
if [[ "${PORTSTUDY_GATEWAY_FALLBACK:-0}" == "1" ]]; then ARGS+=(--gateway-fallback); fi

set +e
python scripts/run_weekend.py --from J5 --until J5 --poll-once "${ARGS[@]}"
STATUS=$?
set -e
if [[ "${STATUS}" == "3" ]]; then
  echo "API batches still pending; resubmitting in ${POLL_INTERVAL_MINUTES} minutes"
  # shellcheck disable=SC2086
  ${SBATCH_CMD:-sbatch} ${PORTSTUDY_FINISH_FLAGS:-} --begin="now+${POLL_INTERVAL_MINUTES}minutes" \
    --export=ALL "${ARCTIC_DIR}/finish.sbatch"
  exit 0
elif [[ "${STATUS}" != "0" ]]; then
  echo "J5 failed with exit ${STATUS}; see the log above" >&2
  exit "${STATUS}"
fi
python scripts/run_weekend.py --from J6 --until J9 "${ARGS[@]}"
