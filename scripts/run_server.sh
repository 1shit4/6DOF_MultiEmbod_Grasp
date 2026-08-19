#!/usr/bin/env bash
# Start the GraspGen-X ZMQ inference server on CPU.
#
#   scripts/run_server.sh              # foreground
#   scripts/run_server.sh --daemon     # background, logs to outputs/server.log
#
# The model loads once (~3 s) and serves every request after that, which is the
# whole point of the two-process split: GraspMAS calls grasp_detection several
# times per query and would otherwise reload 1.6 GB of weights each time.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./env.sh
source "${REPO_ROOT}/scripts/env.sh"

SERVER_PY="${REPO_ROOT}/GraspGenX/client-server/graspgenx_server.py"

# `conda run` forks a child python, so a recorded PID names the wrapper, not the
# server. Match on the script path instead — that reliably finds the real process.
stop_server() {
  if pkill -f "graspgenx_server.py.*--port ${GRASPGEN_SERVER_PORT}" 2>/dev/null; then
    echo "Stopping GraspGen-X server on port ${GRASPGEN_SERVER_PORT} ..."
    for _ in $(seq 1 30); do
      pgrep -f "graspgenx_server.py.*--port ${GRASPGEN_SERVER_PORT}" >/dev/null || break
      sleep 1
    done
    pkill -9 -f "graspgenx_server.py.*--port ${GRASPGEN_SERVER_PORT}" 2>/dev/null || true
  fi
  rm -f "${REPO_ROOT}/outputs/server.pid"
}

DAEMON=0
case "${1:-}" in
  --daemon) DAEMON=1; shift ;;
  --stop)   stop_server; exit 0 ;;
  --restart) stop_server; DAEMON=1; shift ;;
esac

# Never leave two servers fighting over the port.
stop_server

CMD=(conda run -n graspgenx --no-capture-output python
     "${SERVER_PY}"
     --config "${GRASPGENX_CHECKPOINT_DIR}/release"
     --assets_dir "${GRASPGENX_ASSETS_DIR}"
     --default_gripper franka_panda
     --host 127.0.0.1
     --port "${GRASPGEN_SERVER_PORT}"
     --device cpu
     "$@")

if [[ "${DAEMON}" == "1" ]]; then
  mkdir -p "${REPO_ROOT}/outputs"
  LOG="${REPO_ROOT}/outputs/server.log"
  nohup "${CMD[@]}" >"${LOG}" 2>&1 &
  echo $! > "${REPO_ROOT}/outputs/server.pid"
  echo "GraspGen-X server starting (pid $(cat "${REPO_ROOT}/outputs/server.pid")), log: ${LOG}"
else
  exec "${CMD[@]}"
fi
