#!/usr/bin/env bash
#SBATCH --job-name=portstudy-weekend
#SBATCH --output=logs/arctic/weekend-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# The weekend pipeline after Kronos: J3 onward in compute mode (the node reaches the gateway and
# the API providers), J6 onward in login mode (J3 to J5 already ran on the login node).
set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env
require_set ARCTIC_NETWORK_MODE

FROM_JOB="J3"
if [[ "${ARCTIC_NETWORK_MODE}" == "login" ]]; then FROM_JOB="J6"; fi
FROM_JOB="${PORTSTUDY_FROM:-${FROM_JOB}}"

ARGS=(--from "${FROM_JOB}")
if [[ -n "${PORTSTUDY_WEEK:-}" ]]; then ARGS+=(--week "${PORTSTUDY_WEEK}"); fi
if [[ "${PORTSTUDY_GATEWAY_FALLBACK:-0}" == "1" ]]; then ARGS+=(--gateway-fallback); fi

python scripts/run_weekend.py "${ARGS[@]}"
