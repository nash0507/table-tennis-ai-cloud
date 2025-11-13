#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

BACKEND_HOST=${BACKEND_HOST:-0.0.0.0}
BACKEND_PORT=${BACKEND_PORT:-8000}
FRONTEND_PORT=${FRONTEND_PORT:-8501}

command -v uvicorn >/dev/null 2>&1 || {
  echo "[launch_web] 請先安裝 uvicorn (pip install uvicorn[standard])" >&2
  exit 1
}
command -v streamlit >/dev/null 2>&1 || {
  echo "[launch_web] 請先安裝 streamlit (pip install streamlit)" >&2
  exit 1
}

cleanup() {
  if [[ -n "${UVICORN_PID:-}" ]] && ps -p "${UVICORN_PID}" >/dev/null 2>&1; then
    kill "${UVICORN_PID}" >/dev/null 2>&1 || true
    wait "${UVICORN_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

uvicorn backend.app:app \
  --host "${BACKEND_HOST}" \
  --port "${BACKEND_PORT}" \
  --reload &
UVICORN_PID=$!

echo "[launch_web] FastAPI 伺服器已啟動：http://${BACKEND_HOST}:${BACKEND_PORT}"
echo "[launch_web] 開始啟動 Streamlit 儀表板..."

streamlit run frontend/dashboard.py \
  --server.port "${FRONTEND_PORT}" \
  --server.headless true
