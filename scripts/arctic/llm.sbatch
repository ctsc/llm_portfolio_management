#!/usr/bin/env bash
#SBATCH --job-name=portstudy-llm
#SBATCH --output=logs/arctic/llm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
# J3 (API batch submit) then J4 (open-weight gateway calls) for the week. Neither needs the
# Kronos output, so this job runs alongside kronos.sbatch (PLAN3 phase 3). Account, partition and
# walltime come from submit_weekend.sh. In login mode J3 already ran on the login node.
set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env
require_set ARCTIC_NETWORK_MODE

FROM_JOB="J3"
if [[ "${ARCTIC_NETWORK_MODE}" == "login" ]]; then FROM_JOB="J4"; fi
ARGS=(--from "${PORTSTUDY_FROM:-${FROM_JOB}}" --until J4)
if [[ -n "${PORTSTUDY_WEEK:-}" ]]; then ARGS+=(--week "${PORTSTUDY_WEEK}"); fi
if [[ "${PORTSTUDY_GATEWAY_FALLBACK:-0}" == "1" ]]; then ARGS+=(--gateway-fallback); fi

python scripts/run_weekend.py "${ARGS[@]}"
