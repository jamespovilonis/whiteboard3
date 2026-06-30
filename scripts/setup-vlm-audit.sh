#!/usr/bin/env bash
set -euo pipefail

MODEL="${VLM_AUDIT_MODEL:-qwen3-vl:8b}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama is not installed or not on PATH." >&2
  exit 1
fi

if ollama list | awk 'NR > 1 { print $1 }' | grep -Fxq "$MODEL"; then
  echo "VLM audit model already installed: $MODEL"
  exit 0
fi

echo "Pulling VLM audit model: $MODEL"
ollama pull "$MODEL"
