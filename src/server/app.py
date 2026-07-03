"""FastAPI application for the whiteboard backend."""

from __future__ import annotations

import argparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.grading import grade_equation_payload, grade_math_payload
from src.server.config import ServerSettings, settings_from_env
from src.server.routes import audit, feedback, grading, health, recognition
from src.server.services.audit import RecognitionAuditService
from src.server.services.feedback import MathFeedbackService
from testing.latex_semantics import score_semantic_payload


def create_app(
    settings: ServerSettings | None = None,
    *,
    semantic_scorer=score_semantic_payload,
    grading_scorer=grade_equation_payload,
    math_grading_scorer=grade_math_payload,
) -> FastAPI:
    app = FastAPI(title="Whiteboard Recognition API")
    app.state.settings = settings or settings_from_env()
    app.state.semantic_scorer = semantic_scorer
    app.state.grading_scorer = grading_scorer
    app.state.math_grading_scorer = math_grading_scorer
    app.state.audit_service = RecognitionAuditService(app.state.settings)
    app.state.feedback_service = MathFeedbackService(app.state.settings)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    app.include_router(health.router)
    app.include_router(grading.router)
    app.include_router(feedback.router)
    app.include_router(audit.router)
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
    parser.add_argument("--audit-enabled", action="store_true", default=None)
    parser.add_argument("--audit-disabled", action="store_true", default=None)
    parser.add_argument("--audit-log-dir", default=None)
    parser.add_argument("--vlm-audit-base-url", default=None)
    parser.add_argument("--vlm-audit-model", default=None)
    parser.add_argument("--vlm-audit-timeout", type=float, default=None)
    parser.add_argument("--vlm-audit-normal-sample-rate", type=float, default=None)
    parser.add_argument("--vlm-audit-circuit-failures", type=int, default=None)
    parser.add_argument("--vlm-audit-circuit-seconds", type=float, default=None)
    args = parser.parse_args()

    env_settings = settings_from_env()
    audit_enabled = env_settings.audit_enabled
    if args.audit_enabled:
        audit_enabled = True
    if args.audit_disabled:
        audit_enabled = False

    settings = ServerSettings(
        host=args.host or env_settings.host,
        port=args.port if args.port is not None else env_settings.port,
        upstream_api_url=args.upstream_api_url if args.upstream_api_url is not None else env_settings.upstream_api_url,
        upstream_timeout=args.upstream_timeout if args.upstream_timeout is not None else env_settings.upstream_timeout,
        semantic_timeout=args.semantic_timeout if args.semantic_timeout is not None else env_settings.semantic_timeout,
        audit_enabled=audit_enabled,
        audit_log_dir=args.audit_log_dir if args.audit_log_dir is not None else env_settings.audit_log_dir,
        vlm_audit_base_url=args.vlm_audit_base_url if args.vlm_audit_base_url is not None else env_settings.vlm_audit_base_url,
        vlm_audit_model=args.vlm_audit_model if args.vlm_audit_model is not None else env_settings.vlm_audit_model,
        vlm_audit_timeout_seconds=args.vlm_audit_timeout if args.vlm_audit_timeout is not None else env_settings.vlm_audit_timeout_seconds,
        vlm_audit_normal_sample_rate=args.vlm_audit_normal_sample_rate if args.vlm_audit_normal_sample_rate is not None else env_settings.vlm_audit_normal_sample_rate,
        vlm_audit_circuit_failures=args.vlm_audit_circuit_failures if args.vlm_audit_circuit_failures is not None else env_settings.vlm_audit_circuit_failures,
        vlm_audit_circuit_seconds=args.vlm_audit_circuit_seconds if args.vlm_audit_circuit_seconds is not None else env_settings.vlm_audit_circuit_seconds,
        feedback_base_url=env_settings.feedback_base_url,
        feedback_model=env_settings.feedback_model,
        feedback_timeout_seconds=env_settings.feedback_timeout_seconds,
    )

    print(f"Recognition API listening on http://{settings.host}:{settings.port}")
    if settings.upstream_api_url:
        print(f"Proxying CoMER/DBNet endpoints to {settings.upstream_api_url}")
    if settings.audit_enabled:
        print(f"Writing VLM audit logs to {settings.audit_log_dir}")
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
