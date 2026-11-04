#!/usr/bin/env bash
#SBATCH --job-name=portstudy-shuffle
#SBATCH --output=logs/arctic/shuffle-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
# One shard of the H5 shuffle test per array task. Submit with
#   N=$(($(wc -l < data/shuffle/shards.csv) - 1))
#   sbatch --array=0-$((N - 1))%${SHUFFLE_MAX_PARALLEL} scripts/arctic/shuffle_array.sbatch
# after `portstudy shuffle plan`. Shards are idempotent, so a killed task is simply resubmitted.
set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env

SHARDS="${PORTSTUDY_SHARDS:-${PORTSTUDY_DATA_DIR}/shuffle/shards.csv}"
portstudy shuffle run --shard-index "${SLURM_ARRAY_TASK_ID}" --shards "${SHARDS}"
