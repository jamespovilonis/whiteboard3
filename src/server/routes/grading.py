"""Semantic scoring and grading routes."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from src.server.services.scoring import SemanticScoringTimeout, score_payload_with_timeout


router = APIRouter()


@router.post("/score-latex-candidates")
def score_latex_candidates(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    return _score(payload, request, scorer=request.app.state.semantic_scorer, failure_label="Semantic scoring")


@router.post("/grade-equation-work")
def grade_equation_work(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    return _score(
        {**payload, "problemType": "equation-solving"},
        request,
        scorer=request.app.state.grading_scorer,
        failure_label="Grading",
    )


@router.post("/grade-math-work")
def grade_math_work(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    scorer = getattr(request.app.state, "math_grading_scorer", request.app.state.grading_scorer)
    return _score(payload, request, scorer=scorer, failure_label="Math grading")


def _score(payload: dict[str, Any], request: Request, *, scorer, failure_label: str) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = score_payload_with_timeout(
            payload,
            timeout_seconds=request.app.state.settings.semantic_timeout,
            scorer=scorer,
        )
    except SemanticScoringTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{failure_label} failed: {exc}") from exc

    result["elapsedSeconds"] = round(time.monotonic() - started, 3)
    return result
