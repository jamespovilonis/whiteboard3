"""Primitive math feedback routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request


router = APIRouter()


@router.post("/feedback/math-work")
def math_work_feedback(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    service = request.app.state.feedback_service
    try:
        return service.generate(payload or {})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feedback failed: {exc}") from exc
