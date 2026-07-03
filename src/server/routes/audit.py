"""VLM audit routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status


router = APIRouter()


@router.post("/audit-recognition", status_code=status.HTTP_202_ACCEPTED)
def audit_recognition(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    service = request.app.state.audit_service
    return service.enqueue(payload or {})


@router.post("/audit-recognition-note", status_code=status.HTTP_201_CREATED)
@router.post("/audit-recognition/notes", status_code=status.HTTP_201_CREATED)
def add_audit_note(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    service = request.app.state.audit_service
    try:
        return service.add_personal_note(payload or {})
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/audit-recognition-feedback", status_code=status.HTTP_201_CREATED)
@router.post("/audit-recognition/feedback", status_code=status.HTTP_201_CREATED)
def attach_audit_feedback(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    service = request.app.state.audit_service
    try:
        return service.attach_feedback(payload or {})
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/audit-recognition/{audit_id}")
def audit_recognition_status(audit_id: str, request: Request) -> dict[str, Any]:
    service = request.app.state.audit_service
    return service.status(audit_id)
