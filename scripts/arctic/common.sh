#!/usr/bin/env bash
# Shared by the ARCTIC scripts: load arctic.env, refuse TBD values, activate the environment.
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

require_set() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" || "${!name:-}" == "TBD" ]]; then
      echo "${name} is not set in ${ARCTIC_ENV} (docs/arctic_setup.md)" >&2
      exit 2
    fi
  done
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
