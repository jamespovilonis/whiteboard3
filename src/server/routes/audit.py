"""VLM audit routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, status


router = APIRouter()


@router.post("/audit-recognition", status_code=status.HTTP_202_ACCEPTED)
def audit_recognition(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    service = request.app.state.audit_service
    return service.enqueue(payload or {})


@router.get("/audit-recognition/{audit_id}")
def audit_recognition_status(audit_id: str, request: Request) -> dict[str, Any]:
    service = request.app.state.audit_service
    return service.status(audit_id)
