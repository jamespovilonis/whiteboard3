#!/usr/bin/env bash
# Stop local development servers for the whiteboard stack.

set -euo pipefail

PORTS="${PORTS:-8000 8010 5500}"

for port in $PORTS; do
  pids="$(lsof -ti "tcp:${port}" 2>/dev/null || true)"
  if [ -z "$pids" ]; then
    echo "Port ${port}: nothing running"
    continue
  fi

  echo "Port ${port}: stopping process(es) ${pids}"
  kill $pids 2>/dev/null || true
done

