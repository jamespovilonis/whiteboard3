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
        "upstreamApiUrl": settings.upstream_api_url,
    }
