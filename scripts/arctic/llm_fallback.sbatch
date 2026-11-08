#!/usr/bin/env bash
#SBATCH --job-name=portstudy-llm-fallback
#SBATCH --output=logs/arctic/llm-fallback-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# DECISION(D2) fallback: serve the fallback-eligible open-weight models from this node with an
# OpenAI-compatible server and publish the endpoint for `run_weekend.py --gateway-fallback`.
# Serving tool and launch flags are chosen at setup (# UNVERIFIED: vLLM assumed); weights load
# from HF_HOME, downloaded in advance because compute nodes may lack internet access.
set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env
require_set LLM_FALLBACK_WALLTIME

PORT="${PORTSTUDY_FALLBACK_PORT:-8000}"
HOST="$(hostname -f)"
ENDPOINT_FILE="${PORTSTUDY_DATA_DIR}/runtime/fallback_endpoint.json"
WEEK="${PORTSTUDY_WEEK:-$(python -c 'from portstudy.pipeline.jobs import current_week; import datetime as dt; from zoneinfo import ZoneInfo; print(current_week(dt.datetime.now(tz=ZoneInfo("America/New_York"))))')}"
MODELS="$(python -c 'from portstudy.config import load_model_registry; print(" ".join(m.provider_model for m in load_model_registry().models if m.fallback_eligible))')"

if [[ -z "${MODELS}" ]]; then
  echo "no fallback-eligible models in config/models.yaml" >&2
  exit 2
fi

# UNVERIFIED serving command; one server per fallback-eligible model on consecutive ports.
PIDS=()
PORT_I="${PORT}"
for MODEL in ${MODELS}; do
  python -m vllm.entrypoints.openai.api_server --model "${MODEL}" --host 0.0.0.0 \
    --port "${PORT_I}" --download-dir "${HF_HOME}" &
  PIDS+=("$!")
  PORT_I=$((PORT_I + 1))
done

# Review fix L12: write the endpoint file only once every server answers /v1/models, so a
# --gateway-fallback run never reads an endpoint whose model is still loading.
READY_TIMEOUT="${FALLBACK_READY_TIMEOUT_SECONDS:-1800}"
PORT_I="${PORT}"
for MODEL in ${MODELS}; do
  waited=0
  until curl -sf "http://127.0.0.1:${PORT_I}/v1/models" >/dev/null 2>&1; do
    if (( waited >= READY_TIMEOUT )); then
      echo "server for ${MODEL} on port ${PORT_I} not ready after ${READY_TIMEOUT}s" >&2
      kill "${PIDS[@]}" 2>/dev/null || true
      exit 3
    fi
    sleep 15
    waited=$((waited + 15))
  done
  echo "server for ${MODEL} ready on port ${PORT_I} after ${waited}s"
  PORT_I=$((PORT_I + 1))
done

mkdir -p "$(dirname "${ENDPOINT_FILE}")"
python - "${ENDPOINT_FILE}" "${HOST}" "${PORT}" "${WEEK}" "${SLURM_JOB_ID:-local}" <<'PY'
import datetime as dt, json, sys
path, host, port, week, job = sys.argv[1:]
payload = {
    "base_url": f"http://{host}:{port}/v1",
    "host": host,
    "port": int(port),
    "week_id": week,
    "started_at": dt.datetime.now(dt.UTC).isoformat(),
    "slurm_job_id": job,
}
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
import os
os.replace(tmp, path)
print(f"fallback endpoint written to {path}: {payload['base_url']}")
PY

echo "serving ${MODELS} on ${HOST}:${PORT}; rerun the week with:"
echo "  python scripts/run_weekend.py --from J4 --gateway-fallback"
wait "${PIDS[@]}"
