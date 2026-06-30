#!/usr/bin/env bash
# Start the OCR server, Whiteboard API, and Vite app for local development.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHITEBOARD_2_DIR="${WHITEBOARD_2_DIR:-"$(cd "$ROOT_DIR/.." && pwd)/whiteboard_2"}"

OCR_PORT="${OCR_PORT:-8000}"
OCR_URL="${OCR_URL:-http://127.0.0.1:${OCR_PORT}}"
API_PORT="${API_PORT:-8010}"
API_URL="${API_URL:-http://127.0.0.1:${API_PORT}}"
WIFI_IP="$(ipconfig getifaddr en0 2>/dev/null || true)"

OCR_PID=""
API_PID=""

cleanup() {
  if [ -n "$API_PID" ]; then
    kill "$API_PID" 2>/dev/null || true
  fi
  if [ -n "$OCR_PID" ]; then
    kill "$OCR_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

if [ ! -f "$WHITEBOARD_2_DIR/server/server.py" ]; then
  echo "Could not find the OCR server at:"
  echo "  $WHITEBOARD_2_DIR/server/server.py"
  echo
  echo "Set WHITEBOARD_2_DIR=/path/to/whiteboard_2 if it lives somewhere else."
  exit 1
fi

OCR_PYTHON="python3"
if [ -x "$WHITEBOARD_2_DIR/.venv/bin/python" ]; then
  OCR_PYTHON="$WHITEBOARD_2_DIR/.venv/bin/python"
elif [ -x "$WHITEBOARD_2_DIR/.venv-1/bin/python" ]; then
  OCR_PYTHON="$WHITEBOARD_2_DIR/.venv-1/bin/python"
fi

echo "Starting OCR server from whiteboard_2 on $OCR_URL ..."
(cd "$WHITEBOARD_2_DIR" && "$OCR_PYTHON" -m uvicorn server.server:app --host 0.0.0.0 --port "$OCR_PORT") &
OCR_PID=$!

echo "Starting Whiteboard API on $API_URL ..."
(cd "$ROOT_DIR" && python3 -m src.server.app --host 0.0.0.0 --port "$API_PORT" --upstream-api-url "$OCR_URL") &
API_PID=$!

echo "Starting Whiteboard app ..."
echo
echo "When Vite prints a localhost URL, open it in your browser."
if [ -n "$WIFI_IP" ]; then
  echo "From the Chromebook, open: http://${WIFI_IP}:5500/"
fi
echo "Press Ctrl+C here to stop everything."
echo

cd "$ROOT_DIR"
VITE_API_URL="$API_URL" npm run dev
