#!/usr/bin/env bash
#SBATCH --job-name=portstudy-shuffle
#SBATCH --output=logs/arctic/shuffle-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
# One shard of the H5 shuffle test per array task (DECISION(G6): MVO, Robust MVO and Top-K
# portfolios only; DECISION(M1): one core, 3G). Submit with
#   N=$(($(wc -l < data/shuffle/shards.csv) - 1))
#   sbatch --array=0-$((N - 1))%${SHUFFLE_MAX_PARALLEL} scripts/arctic/shuffle_array.sbatch
# after `portstudy shuffle plan` (which prints the estimated minutes per shard). Shards are
# idempotent, so a killed task is simply resubmitted.
set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env
check_walltime SHUFFLE_WALLTIME

SHARDS="${PORTSTUDY_SHARDS:-${PORTSTUDY_DATA_DIR}/shuffle/shards.csv}"
portstudy shuffle run --shard-index "${SLURM_ARRAY_TASK_ID}" --shards "${SHARDS}"
