#!/usr/bin/env python3
"""Compatibility launcher for the FastAPI recognition gateway.

The backend now lives under ``src.server``. This script remains so older docs,
tests, and local commands that run ``python3 testing/semantic_score_server.py``
continue to start the same gateway.
"""

from __future__ import annotations

from pathlib import Path
import sys

TESTING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTING_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.server.app import main
from src.server.services.proxy import proxy_request_headers, proxy_target_url
from src.server.services.scoring import (
    SemanticScoringTimeout,
    score_payload_with_timeout as score_semantic_payload_with_timeout,
)

__all__ = [
    "SemanticScoringTimeout",
    "proxy_request_headers",
    "proxy_target_url",
    "score_semantic_payload_with_timeout",
]


if __name__ == "__main__":
    raise SystemExit(main())
