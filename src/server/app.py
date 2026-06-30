"""FastAPI application for the whiteboard backend."""

from __future__ import annotations

import argparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.grading import grade_equation_payload
from src.server.config import ServerSettings, settings_from_env
from src.server.routes import grading, health, recognition
from testing.latex_semantics import score_semantic_payload


def create_app(
    settings: ServerSettings | None = None,
    *,
    semantic_scorer=score_semantic_payload,
    grading_scorer=grade_equation_payload,
) -> FastAPI:
    app = FastAPI(title="Whiteboard Recognition API")
    app.state.settings = settings or settings_from_env()
    app.state.semantic_scorer = semantic_scorer
    app.state.grading_scorer = grading_scorer

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    app.include_router(health.router)
    app.include_router(grading.router)
    app.include_router(recognition.router)
    return app


app = create_app()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--upstream-api-url",
        default=None,
        help="Optional CoMER/DBNet API base URL to proxy /recognize, /segment-lines, and /health.",
    )
    parser.add_argument("--upstream-timeout", type=float, default=None)
    parser.add_argument(
        "--semantic-timeout",
        type=float,
        default=None,
        help="Maximum seconds to spend scoring one semantic or grading request.",
    )
    args = parser.parse_args()

    env_settings = settings_from_env()
    settings = ServerSettings(
        host=args.host or env_settings.host,
        port=args.port if args.port is not None else env_settings.port,
        upstream_api_url=args.upstream_api_url if args.upstream_api_url is not None else env_settings.upstream_api_url,
        upstream_timeout=args.upstream_timeout if args.upstream_timeout is not None else env_settings.upstream_timeout,
        semantic_timeout=args.semantic_timeout if args.semantic_timeout is not None else env_settings.semantic_timeout,
    )

    print(f"Recognition API listening on http://{settings.host}:{settings.port}")
    if settings.upstream_api_url:
        print(f"Proxying CoMER/DBNet endpoints to {settings.upstream_api_url}")
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
