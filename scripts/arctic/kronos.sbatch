#!/usr/bin/env bash
#SBATCH --job-name=portstudy-kronos
#SBATCH --output=logs/arctic/kronos-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# J2 (baseline with the Kronos forecaster) for the week. In compute mode J1 runs first here,
# because the compute node can reach the data providers. Account, partition, GRES and walltime
# come from submit_weekend.sh (sbatch flags), so this file holds no TBD.
set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env
require_set ARCTIC_NETWORK_MODE

WEEK_ARG=()
if [[ -n "${PORTSTUDY_WEEK:-}" ]]; then WEEK_ARG=(--week "${PORTSTUDY_WEEK}"); fi

if [[ "${ARCTIC_NETWORK_MODE}" == "compute" ]]; then
  portstudy run-job J1 "${WEEK_ARG[@]}"
fi
portstudy run-job J2 "${WEEK_ARG[@]}"
