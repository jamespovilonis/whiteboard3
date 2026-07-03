"""Gateway health route."""

from __future__ import annotations

from fastapi import APIRouter, Request


router = APIRouter()


@router.get("/gateway/health")
def gateway_health(request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    return {
        "status": "ok",
        "semantic": True,
        "grading": True,
        "audit": bool(settings.audit_enabled),
        "auditLogDir": settings.audit_log_dir,
        "vlmAuditBaseUrl": settings.vlm_audit_base_url,
        "vlmAuditModel": settings.vlm_audit_model,
        "feedbackBaseUrl": settings.feedback_base_url,
        "feedbackModel": settings.feedback_model,
        "upstreamApiUrl": settings.upstream_api_url,
    }
