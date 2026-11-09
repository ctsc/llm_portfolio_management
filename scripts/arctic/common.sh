#!/usr/bin/env bash
# Shared by the ARCTIC scripts: load arctic.env, refuse TBD values, check walltimes (DECISION(W1)),
# activate the environment.
set -euo pipefail

ARCTIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ARCTIC_DIR}/../.." && pwd)"
ARCTIC_ENV="${ARCTIC_ENV:-${ARCTIC_DIR}/arctic.env}"

if [[ ! -f "${ARCTIC_ENV}" ]]; then
  echo "missing ${ARCTIC_ENV}: copy arctic.env.example and fill it in (docs/arctic_setup.md)" >&2
  exit 2
fi
set -a
# shellcheck disable=SC1090
source "${ARCTIC_ENV}"
set +a
MAX_JOB_HOURS="${MAX_JOB_HOURS:-15}"
WARN_JOB_HOURS="${WARN_JOB_HOURS:-12}"

require_set() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" || "${!name:-}" == "TBD" ]]; then
      echo "${name} is not set in ${ARCTIC_ENV} (docs/arctic_setup.md)" >&2
      exit 2
    fi
  done
}

walltime_minutes() {  # [D-]HH:MM[:SS] or MM -> minutes
  local value="$1" days=0 rest
  if [[ "${value}" == *-* ]]; then days="${value%%-*}"; rest="${value#*-}"; else rest="${value}"; fi
  local IFS=:
  read -r -a parts <<< "${rest}"
  case "${#parts[@]}" in
    1) echo $((10#${parts[0]} + days * 1440)) ;;
    2) echo $((10#${parts[0]} * 60 + 10#${parts[1]} + days * 1440)) ;;
    *) echo $((10#${parts[0]} * 60 + 10#${parts[1]} + days * 1440)) ;;
  esac
}

check_walltime() {  # check_walltime NAME: refuse above MAX_JOB_HOURS, warn above WARN_JOB_HOURS
  local name="$1" minutes
  require_set "${name}"
  minutes="$(walltime_minutes "${!name}")"
  if (( minutes > MAX_JOB_HOURS * 60 )); then
    echo "${name}=${!name} exceeds the ${MAX_JOB_HOURS} hour limit (DECISION(W1))" >&2
    exit 2
  fi
  if (( minutes > WARN_JOB_HOURS * 60 )); then
    echo "warning: ${name}=${!name} is above the ${WARN_JOB_HOURS} hour target (DECISION(W1))" >&2
  fi
}

activate_env() {
  require_set PYTHON_MODULE VENV_PATH DATA_ROOT
  if command -v module >/dev/null 2>&1; then
    module load "${PYTHON_MODULE}"
  fi
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
  export PORTSTUDY_DATA_DIR="${DATA_ROOT}"
  export HF_HOME="${HF_HOME:-${DATA_ROOT}/hf}"
  cd "${REPO_ROOT}"
}

log_dir() {
  mkdir -p "${REPO_ROOT}/logs/arctic"
  echo "${REPO_ROOT}/logs/arctic"
}
