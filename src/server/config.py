"""Server configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8010
    upstream_api_url: str = ""
    upstream_timeout: float = 25.0
    semantic_timeout: float = 2.5
    audit_enabled: bool = True
    audit_log_dir: str = "/Users/jpovj/Documents/dev/log_whiteboard_3"
    vlm_audit_base_url: str = "http://127.0.0.1:11434/v1"
    vlm_audit_model: str = "qwen3-vl:8b"
    vlm_audit_timeout_seconds: float = 120.0
    vlm_audit_normal_sample_rate: float = 0.10


def settings_from_env() -> ServerSettings:
    return ServerSettings(
        host=os.environ.get("WHITEBOARD_API_HOST", "127.0.0.1"),
        port=_env_int("WHITEBOARD_API_PORT", 8010),
        upstream_api_url=os.environ.get("UPSTREAM_OCR_API_URL", ""),
        upstream_timeout=_env_float("UPSTREAM_OCR_TIMEOUT", 25.0),
        semantic_timeout=_env_float("SEMANTIC_TIMEOUT", 2.5),
        audit_enabled=_env_bool("WHITEBOARD_AUDIT_ENABLED", True),
        audit_log_dir=os.environ.get("WHITEBOARD_AUDIT_LOG_DIR", "/Users/jpovj/Documents/dev/log_whiteboard_3"),
        vlm_audit_base_url=os.environ.get("VLM_AUDIT_BASE_URL", "http://127.0.0.1:11434/v1"),
        vlm_audit_model=os.environ.get("VLM_AUDIT_MODEL", "qwen3-vl:8b"),
        vlm_audit_timeout_seconds=_env_float("VLM_AUDIT_TIMEOUT_SECONDS", 120.0),
        vlm_audit_normal_sample_rate=_env_float("VLM_AUDIT_NORMAL_SAMPLE_RATE", 0.10),
    )


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}
