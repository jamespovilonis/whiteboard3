"""OCR/model proxy routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool

from src.server.services.proxy import UpstreamNotConfigured, proxy_request


router = APIRouter()


@router.api_route("/recognize", methods=["GET", "POST", "OPTIONS"])
async def proxy_recognize(request: Request) -> Response:
    return await _proxy(request)


@router.api_route("/segment-lines", methods=["GET", "POST", "OPTIONS"])
async def proxy_segment_lines(request: Request) -> Response:
    return await _proxy(request)


@router.api_route("/segment-lines/{rest:path}", methods=["GET", "POST", "OPTIONS"])
async def proxy_segment_lines_child(request: Request, rest: str) -> Response:
    return await _proxy(request)


@router.api_route("/health", methods=["GET", "POST", "OPTIONS"])
async def proxy_upstream_health(request: Request) -> Response:
    return await _proxy(request)


async def _proxy(request: Request) -> Response:
    settings = request.app.state.settings
    body = await request.body()
    path = request.url.path
    if request.url.query:
        path = f"{path}?{request.url.query}"

    try:
        upstream = await run_in_threadpool(
            proxy_request,
            upstream_api_url=settings.upstream_api_url,
            path=path,
            method=request.method,
            headers=request.headers,
            body=body,
            timeout=settings.upstream_timeout,
        )
    except UpstreamNotConfigured as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    return Response(
        content=upstream.body,
        status_code=upstream.status,
        media_type=upstream.headers.get("Content-Type", "application/octet-stream"),
    )
