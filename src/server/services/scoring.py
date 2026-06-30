"""Timeout-isolated scoring helpers used by API routes."""

from __future__ import annotations

import multiprocessing
import queue
from typing import Any, Callable

from testing.latex_semantics import score_semantic_payload
from src.grading import grade_equation_payload


class SemanticScoringTimeout(TimeoutError):
    """Raised when semantic scoring exceeds the request budget."""


def score_payload_with_timeout(
    payload: dict[str, Any],
    timeout_seconds: float = 2.5,
    scorer: Callable[[dict[str, Any]], dict[str, Any]] = score_semantic_payload,
) -> dict[str, Any]:
    timeout = float(timeout_seconds)
    if timeout <= 0:
        return scorer(payload)

    if "fork" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("fork")
    else:
        context = multiprocessing.get_context()
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_score_payload_worker,
        args=(payload, result_queue, scorer),
        daemon=True,
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
        raise SemanticScoringTimeout(f"Semantic scoring exceeded {timeout:.2f}s")

    try:
        status, result = result_queue.get(timeout=1)
    except queue.Empty as exc:
        raise RuntimeError("semantic scoring worker exited without a result") from exc

    if status == "ok":
        return result
    raise RuntimeError(result)


def _score_payload_worker(
    payload: dict[str, Any],
    result_queue,
    scorer: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    try:
        result_queue.put(("ok", scorer(payload)))
    except Exception as exc:
        result_queue.put(("error", str(exc)))


__all__ = [
    "SemanticScoringTimeout",
    "grade_equation_payload",
    "score_payload_with_timeout",
    "score_semantic_payload",
]
