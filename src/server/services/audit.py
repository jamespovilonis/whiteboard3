"""Filesystem-backed VLM audit service for recognition results."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any, Callable, Optional
import urllib.error
import urllib.request

from PIL import Image, ImageDraw

from src.grading import grade_math_payload
from src.server.config import ServerSettings


JSON_HEADERS = {"Content-Type": "application/json"}
MAX_RENDER_SIDE = 1800
RENDER_PADDING = 80
VLM_TIMEOUT_CIRCUIT_FAILURES = 2
VLM_TIMEOUT_CIRCUIT_SECONDS = 300.0
AUDIT_PROMPT_VERSION = "recognition-audit-v2"
AUDIT_IMAGE_NAMES = ("problemCrop", "answerCrop", "answerContext")


class AuditSchemaError(ValueError):
    """Raised when the VLM response cannot be normalized."""

    def __init__(self, message: str, *, raw_response: Any = None):
        super().__init__(message)
        self.raw_response = raw_response


class OpenAICompatibleVlmClient:
    """Small OpenAI-compatible chat-completions client for local VLM servers."""

    def __init__(self, *, base_url: str, model: str, timeout_seconds: float):
        self.base_url = str(base_url or "").rstrip("/")
        self.model = str(model or "").strip()
        self.timeout_seconds = max(1.0, float(timeout_seconds or 120.0))

    def complete(self, *, prompt: str, image_paths: list[Path]) -> dict[str, Any]:
        if not self.base_url:
            raise RuntimeError("VLM audit base URL is not configured")
        if not self.model:
            raise RuntimeError("VLM audit model is not configured")

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_to_data_url(path)},
            })

        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a careful math handwriting auditor. Return only valid JSON.",
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=JSON_HEADERS,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"VLM audit HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"VLM audit request failed: {exc.reason}") from exc


class RecognitionAuditService:
    """Queue and execute VLM audit jobs without blocking recognition requests."""

    def __init__(
        self,
        settings: ServerSettings,
        *,
        vlm_client: Optional[OpenAICompatibleVlmClient] = None,
        grader: Callable[[dict[str, Any]], dict[str, Any]] = grade_math_payload,
    ):
        self.settings = settings
        self.log_dir = Path(settings.audit_log_dir).expanduser()
        self.grader = grader
        self.vlm_client = vlm_client or OpenAICompatibleVlmClient(
            base_url=settings.vlm_audit_base_url,
            model=settings.vlm_audit_model,
            timeout_seconds=settings.vlm_audit_timeout_seconds,
        )
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whiteboard-audit")
        self._status_lock = Lock()
        self._statuses: dict[str, dict[str, Any]] = {}
        self._vlm_health_lock = Lock()
        self._consecutive_vlm_timeouts = 0
        self._vlm_circuit_open_until = 0.0

    def enqueue(self, payload: dict[str, Any]) -> dict[str, Any]:
        audit_id = build_audit_id(payload)
        if not self.settings.audit_enabled:
            status_payload = {
                "auditId": audit_id,
                "status": "disabled",
                "queued": False,
                "done": True,
                "disabled": True,
            }
            self._set_status(audit_id, status_payload)
            return {"auditId": audit_id, "queued": False, "disabled": True}
        self._set_status(audit_id, {
            "auditId": audit_id,
            "status": "queued",
            "queued": True,
            "done": False,
            "problemId": payload.get("problemId"),
            "triggerReasons": payload.get("triggerReasons") or [],
        })
        self.executor.submit(self.run_audit, payload, audit_id)
        return {"auditId": audit_id, "queued": True}

    def status(self, audit_id: str) -> dict[str, Any]:
        with self._status_lock:
            status_payload = self._statuses.get(str(audit_id or ""))
            if not status_payload:
                return {
                    "auditId": audit_id,
                    "status": "unknown",
                    "queued": False,
                    "done": False,
                    "found": False,
                }
            return dict(status_payload)

    def add_personal_note(self, payload: dict[str, Any]) -> dict[str, Any]:
        note = str(payload.get("note") or "").strip()
        if not note:
            raise ValueError("Personal note is required")
        if len(note) > 2000:
            raise ValueError("Personal note must be 2000 characters or fewer")

        audit_id = str(payload.get("auditId") or "").strip() or None
        problem_id = str(payload.get("problemId") or "").strip() or None
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        record = {
            "eventStage": "personal_note",
            "createdAt": created_at,
            "auditId": audit_id,
            "problemId": problem_id,
            "note": note,
            "source": str(payload.get("source") or "debugger"),
        }

        audit_dir: Path | None = None
        if audit_id:
            with self._status_lock:
                status_payload = self._statuses.get(audit_id, {})
                audit_dir_value = status_payload.get("auditDir")
            if audit_dir_value:
                audit_dir = Path(str(audit_dir_value))
                record["auditDir"] = str(audit_dir)

        append_jsonl(self.log_dir / "personal_notes.jsonl", record)
        if audit_dir is not None:
            append_jsonl(audit_dir / "personal_notes.jsonl", record)
        if audit_id:
            self._set_status(audit_id, {
                "auditId": audit_id,
                "problemId": problem_id,
                "personalNote": note,
                "personalNoteAt": created_at,
                "personalNoteCount": int(self.status(audit_id).get("personalNoteCount") or 0) + 1,
            })
        return record

    def run_audit(self, payload: dict[str, Any], audit_id: Optional[str] = None) -> dict[str, Any]:
        audit_id = audit_id or build_audit_id(payload)
        now = datetime.now(timezone.utc)
        audit_dir = self.log_dir / now.strftime("%Y-%m-%d") / audit_id
        audit_dir.mkdir(parents=True, exist_ok=True)
        self._set_status(audit_id, {
            "auditId": audit_id,
            "status": "processing",
            "queued": True,
            "done": False,
            "problemId": payload.get("problemId"),
            "triggerReasons": payload.get("triggerReasons") or [],
            "auditDir": str(audit_dir),
        })

        input_payload = sanitize_json(payload)
        fast_result = sanitize_json(payload.get("fastResult") or {})
        write_json(audit_dir / "input.json", input_payload)
        write_json(audit_dir / "fast_result.json", fast_result)

        artifact_paths: dict[str, str] = {}
        crop_boxes: dict[str, Any] = {}
        audit_attempts: list[dict[str, Any]] = []
        failure_kind: str | None = None
        failure_stage = "initializing"
        comparison: dict[str, Any] | None = None
        normalized: dict[str, Any] | None = None
        vlm_grading: dict[str, Any] | None = None
        raw_vlm: dict[str, Any] | None = None
        prompt_version = str(payload.get("promptVersion") or AUDIT_PROMPT_VERSION)
        attached_images: list[str] = []

        try:
            failure_stage = "rendering_artifacts"
            rendered = render_audit_images(payload, audit_dir)
            crop_boxes = rendered.pop("cropBoxes", {})
            crops = {key: path for key, path in rendered.items() if isinstance(path, Path)}
            artifact_paths.update({key: str(path) for key, path in crops.items()})
            prompt = build_vlm_prompt(payload)
            attached_images = [name for name in AUDIT_IMAGE_NAMES if name in crops]
            failure_stage = "requesting_vlm"
            self._raise_if_vlm_circuit_open()
            raw_vlm, normalized = self._complete_and_normalize_vlm(
                prompt=prompt,
                image_paths=[crops[name] for name in attached_images],
                image_names=attached_images,
                audit_dir=audit_dir,
                attempts=audit_attempts,
                prompt_version=prompt_version,
                audit_id=audit_id,
            )
            self._record_vlm_success()
            write_json(audit_dir / "vlm_raw.json", raw_vlm)

            write_json(audit_dir / "vlm_normalized.json", normalized)

            failure_stage = "grading_vlm"
            if is_problem_input_audit(payload):
                vlm_grading = {"status": "skipped", "failed": False, "reason": "problem_input_audit"}
            else:
                vlm_grading = self.grader({
                    "problemLatex": payload.get("problemLatex") or "",
                    "problemMetadata": payload.get("problemMetadata") or {},
                    "lines": [
                        {"lineIndex": index, "latex": latex}
                        for index, latex in enumerate(normalized.get("latexLines") or [])
                    ],
                })
                vlm_grading = {"status": "complete", "failed": False, **vlm_grading}
            write_json(audit_dir / "vlm_grading.json", vlm_grading)
            failure_stage = "comparing_results"
            comparison = compare_audit_results(fast_result, normalized, vlm_grading)
        except AuditSchemaError as exc:
            failure_kind = "vlm_schema_error"
            failure_stage = "normalizing_vlm"
            comparison = failure_comparison("vlm_schema_error", str(exc), fast_result)
            write_json(audit_dir / "vlm_raw.json", schema_error_payload(exc, raw_vlm))
        except Exception as exc:
            if failure_stage in {"requesting_vlm", "normalizing_vlm"}:
                failure_kind = "vlm_unavailable"
                self._record_vlm_unavailable(str(exc))
                comparison = failure_comparison("vlm_unavailable", str(exc), fast_result)
                write_json(audit_dir / "vlm_raw.json", unavailable_error_payload(exc, failure_stage))
            else:
                failure_kind = "audit_internal_error"
                comparison = failure_comparison("audit_internal_error", str(exc), fast_result)
        if comparison is None:
            failure_kind = failure_kind or "audit_internal_error"
            comparison = failure_comparison(failure_kind, "audit did not produce a comparison result", fast_result)

        failure_kind = failure_kind or first_failure_kind(comparison)
        try:
            write_json(audit_dir / "comparison.json", comparison)
            write_json(audit_dir / "audit_metadata.json", build_audit_metadata(
                settings=self.settings,
                audit_id=audit_id,
                payload=payload,
                comparison=comparison,
                artifact_paths=artifact_paths,
                crop_boxes=crop_boxes,
                attempts=audit_attempts,
                failure_kind=failure_kind,
                failure_stage=failure_stage,
                prompt_version=prompt_version,
                normalized=normalized,
                attached_images=attached_images,
            ))
            summary = build_event_summary(
                audit_id=audit_id,
                created_at=now,
                payload=payload,
                comparison=comparison,
                audit_dir=audit_dir,
                artifact_paths=artifact_paths,
                normalized=normalized,
                vlm_grading=vlm_grading,
                failure_kind=failure_kind,
                failure_stage=failure_stage,
                prompt_version=prompt_version,
                attempts=audit_attempts,
                attached_images=attached_images,
            )
            append_jsonl(self.log_dir / "audit_events.jsonl", summary)
        except Exception as exc:
            summary = {
                "auditId": audit_id,
                "createdAt": now.isoformat().replace("+00:00", "Z"),
                "eventStage": "terminal",
                "problemId": payload.get("problemId"),
                "comparisonStatus": "failed",
                "failureKind": "audit_internal_error",
                "failureStage": "writing_artifacts",
                "description": f"Failed to persist audit artifacts: {exc}",
                "auditDir": str(audit_dir),
                "prompt_version": prompt_version,
            }
            append_jsonl(self.log_dir / "audit_events.jsonl", summary)
        self._set_status(audit_id, {
            "auditId": audit_id,
            "status": "logged",
            "queued": True,
            "done": True,
            "problemId": payload.get("problemId"),
            "triggerReasons": payload.get("triggerReasons") or [],
            "auditDir": str(audit_dir),
            "eventLogPath": str(self.log_dir / "audit_events.jsonl"),
            "discrepancyCount": len(comparison.get("discrepancies") or []),
            "comparisonStatus": comparison.get("status"),
            "failureKind": failure_kind,
        })
        return summary

    def _raise_if_vlm_circuit_open(self) -> None:
        now = time.monotonic()
        with self._vlm_health_lock:
            remaining = self._vlm_circuit_open_until - now
        if remaining > 0:
            raise RuntimeError(f"VLM audit circuit open after repeated timeouts; retry in {remaining:.0f}s")

    def _record_vlm_success(self) -> None:
        with self._vlm_health_lock:
            self._consecutive_vlm_timeouts = 0
            self._vlm_circuit_open_until = 0.0

    def _record_vlm_unavailable(self, error: str) -> None:
        normalized_error = str(error).lower()
        if "circuit open" in normalized_error:
            return
        if "timeout" not in normalized_error and "timed out" not in normalized_error:
            return
        with self._vlm_health_lock:
            self._consecutive_vlm_timeouts += 1
            if self._consecutive_vlm_timeouts >= VLM_TIMEOUT_CIRCUIT_FAILURES:
                self._vlm_circuit_open_until = time.monotonic() + VLM_TIMEOUT_CIRCUIT_SECONDS

    def _complete_and_normalize_vlm(
        self,
        *,
        prompt: str,
        image_paths: list[Path],
        image_names: list[str],
        audit_dir: Path,
        attempts: list[dict[str, Any]],
        prompt_version: str,
        audit_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        last_schema_error: AuditSchemaError | None = None
        for attempt_index in range(2):
            started = time.perf_counter()
            raw_vlm: dict[str, Any] | None = None
            request_id = f"{audit_id}:attempt:{attempt_index + 1}"
            try:
                raw_vlm = self.vlm_client.complete(prompt=prompt, image_paths=image_paths)
                elapsed = time.perf_counter() - started
                response_request_id = request_id_for_response(raw_vlm) or request_id
                normalized = normalize_vlm_response(raw_vlm)
                attempts.append({
                    "attempt": attempt_index + 1,
                    "status": "complete",
                    "elapsedSeconds": round(elapsed, 3),
                    "requestId": response_request_id,
                    "request_id": response_request_id,
                    "promptVersion": prompt_version,
                    "prompt_version": prompt_version,
                    "attachedImages": list(image_names),
                    "queueMs": None,
                    "queue_ms": None,
                    "inferenceMs": round(elapsed * 1000, 1),
                    "inference_ms": round(elapsed * 1000, 1),
                })
                if attempt_index > 0:
                    write_json(audit_dir / f"vlm_raw_attempt_{attempt_index + 1}.json", raw_vlm)
                return raw_vlm, normalized
            except AuditSchemaError as exc:
                elapsed = time.perf_counter() - started
                response_request_id = request_id_for_response(raw_vlm) or request_id
                last_schema_error = AuditSchemaError(str(exc), raw_response=raw_vlm)
                attempts.append({
                    "attempt": attempt_index + 1,
                    "status": "failed",
                    "failureKind": "vlm_schema_error",
                    "error": str(exc),
                    "elapsedSeconds": round(elapsed, 3),
                    "failureStage": "normalizing_vlm",
                    "requestId": response_request_id,
                    "request_id": response_request_id,
                    "promptVersion": prompt_version,
                    "prompt_version": prompt_version,
                    "attachedImages": list(image_names),
                    "queueMs": None,
                    "queue_ms": None,
                    "inferenceMs": round(elapsed * 1000, 1),
                    "inference_ms": round(elapsed * 1000, 1),
                })
                if raw_vlm is not None:
                    write_json(audit_dir / f"vlm_raw_attempt_{attempt_index + 1}.json", raw_vlm)
            except Exception as exc:
                elapsed = time.perf_counter() - started
                attempts.append({
                    "attempt": attempt_index + 1,
                    "status": "failed",
                    "failureKind": "vlm_unavailable",
                    "error": str(exc),
                    "elapsedSeconds": round(elapsed, 3),
                    "failureStage": "requesting_vlm",
                    "requestId": request_id,
                    "request_id": request_id,
                    "promptVersion": prompt_version,
                    "prompt_version": prompt_version,
                    "attachedImages": list(image_names),
                    "queueMs": None,
                    "queue_ms": None,
                    "inferenceMs": round(elapsed * 1000, 1),
                    "inference_ms": round(elapsed * 1000, 1),
                })
                raise
        if last_schema_error is not None:
            raise last_schema_error
        raise RuntimeError("VLM audit failed without an attempt result")

    def _set_status(self, audit_id: str, status_payload: dict[str, Any]) -> None:
        with self._status_lock:
            existing = self._statuses.get(audit_id, {})
            self._statuses[audit_id] = {**existing, **sanitize_json(status_payload)}


def build_audit_id(payload: dict[str, Any]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    seed = json.dumps({
        "problemId": payload.get("problemId"),
        "inputSignature": payload.get("inputSignature"),
        "problemLatex": payload.get("problemLatex"),
    }, sort_keys=True)
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:10]
    return f"audit_{now}_{digest}"


def build_vlm_prompt(payload: dict[str, Any]) -> str:
    if is_problem_input_audit(payload):
        return build_problem_input_vlm_prompt(payload)

    problem_latex = str(payload.get("problemLatex") or "").strip()
    trigger_reasons = payload.get("triggerReasons") or []
    fast_attachments = flatten_annotation_attachments(
        (payload.get("fastResult") or {}).get("annotationAttachments") or [],
        source="fast",
    )
    attachment_context = ""
    if fast_attachments:
        summarized = [
            {
                "operator_latex": item.get("operator_latex"),
                "target_line_index": item.get("target_line_index"),
                "equation_side": item.get("equation_side"),
            }
            for item in fast_attachments[:6]
        ]
        attachment_context = (
            "Fast OCR-side attachment hints (may be wrong): "
            f"{json.dumps(summarized, sort_keys=True)}\n"
        )
    return (
        "Read the student's handwritten math answer from the attached images. "
        "The printed problem is not rendered in the image; it is provided here as LaTeX. "
        "The third image includes wider answer context only for interpreting detached operation marks.\n\n"
        f"Problem LaTeX: {problem_latex}\n"
        f"Audit trigger reasons: {', '.join(map(str, trigger_reasons)) or 'none'}\n\n"
        f"{attachment_context}"
        "Return only a JSON object with this shape:\n"
        "{\n"
        "  \"latexLines\": [\"one LaTeX string per visible student math line\"],\n"
        "  \"lineObservations\": [{\"lineIndex\": 0, \"latex\": \"...\", \"confidence\": 0.0, \"notes\": \"...\"}],\n"
        "  \"visualMarks\": [{\"type\": \"circled_answer|boxed_answer|crossed_out|arrow|scratch|detached_operation_annotation|other\", \"lineIndex\": 0, \"latex\": \"...\", \"confidence\": 0.0, \"notes\": \"...\"}],\n"
        "  \"annotationAttachments\": [{\"operatorLatex\": \"...\", \"targetLineIndex\": 0, \"equationSide\": \"left|right|both\", \"pairedAnnotationId\": \"...\", \"attachmentConfidence\": 0.0, \"notes\": \"...\"}],\n"
        "  \"overallConfidence\": 0.0,\n"
        "  \"notes\": \"short reason for anything suspicious\"\n"
        "}\n\n"
        "Confidence values must be numbers from 0 to 1. Use 0 only when the line or mark is unreadable, not as a placeholder. "
        "Call out detached operation annotations such as multiplier/divider marks written beside both sides of an equation, "
        "and attach them to left, right, or both sides when possible. "
        "Do not decide whether the app is correct. Do not include markdown. "
        "Preserve visible intermediate lines, even if a final answer is circled."
    )


def build_problem_input_vlm_prompt(payload: dict[str, Any]) -> str:
    trigger_reasons = payload.get("triggerReasons") or []
    fast_lines = (payload.get("fastResult") or {}).get("latexLines") or []
    return (
        "Read the user's handwritten math problem input from the attached images. "
        "This is the problem-entry box, not a student's answer to a printed problem. "
        "The app's fast OCR thought the input was: "
        f"{json.dumps(fast_lines, sort_keys=True)}\n"
        f"Audit trigger reasons: {', '.join(map(str, trigger_reasons)) or 'none'}\n\n"
        "Return only a JSON object with this shape:\n"
        "{\n"
        "  \"latexLines\": [\"one LaTeX string per visible handwritten problem line\"],\n"
        "  \"lineObservations\": [{\"lineIndex\": 0, \"latex\": \"...\", \"confidence\": 0.0, \"notes\": \"...\"}],\n"
        "  \"visualMarks\": [],\n"
        "  \"annotationAttachments\": [],\n"
        "  \"overallConfidence\": 0.0,\n"
        "  \"notes\": \"short reason for anything suspicious\"\n"
        "}\n\n"
        "Preserve radicals, fraction nesting, exponents, logarithm bases, and index notation exactly when visible. "
        "If the input was split into multiple visible rows, return multiple latexLines. "
        "Do not solve or simplify the problem. Do not include markdown."
    )


def render_audit_images(payload: dict[str, Any], audit_dir: Path) -> dict[str, Any]:
    strokes = [stroke for stroke in payload.get("strokes") or [] if isinstance(stroke, dict)]
    problem_box = bbox_or_none(payload.get("problemBox"))
    answer_box = bbox_or_none(payload.get("answerBox")) or bbox_for_strokes(strokes)
    problem_crop_box = padded_bbox(union_bbox(problem_box, answer_box) or answer_box or problem_box, RENDER_PADDING)
    answer_crop_box = padded_bbox(answer_box or problem_crop_box, RENDER_PADDING)
    answer_context_box = padded_bbox(answer_box or problem_crop_box, RENDER_PADDING * 2)

    if problem_crop_box is None:
        problem_crop_box = {"xMin": 0, "yMin": 0, "xMax": 800, "yMax": 500}
    if answer_crop_box is None:
        answer_crop_box = problem_crop_box
    if answer_context_box is None:
        answer_context_box = answer_crop_box

    problem_crop = audit_dir / "problem_crop.png"
    answer_crop = audit_dir / "answer_crop.png"
    answer_context = audit_dir / "answer_context.png"
    fast_overlay = audit_dir / "fast_overlay.png"
    line_boxes = fast_line_boxes(payload.get("fastResult") or {})
    render_stroke_crop(strokes, problem_crop_box, problem_crop)
    render_stroke_crop(strokes, answer_crop_box, answer_crop)
    render_stroke_crop(
        strokes,
        answer_context_box,
        answer_context,
        boxes=line_boxes,
    )
    render_stroke_crop(
        strokes,
        problem_crop_box,
        fast_overlay,
        boxes=line_boxes,
    )
    return {
        "problemCrop": problem_crop,
        "answerCrop": answer_crop,
        "answerContext": answer_context,
        "fastOverlay": fast_overlay,
        "cropBoxes": {
            "problemCrop": problem_crop_box,
            "answerCrop": answer_crop_box,
            "answerContext": answer_context_box,
            "fastOverlay": problem_crop_box,
        },
    }


def render_stroke_crop(
    strokes: list[dict[str, Any]],
    crop_box: dict[str, float],
    path: Path,
    *,
    boxes: Optional[list[dict[str, float]]] = None,
) -> None:
    width = max(1.0, float(crop_box["xMax"]) - float(crop_box["xMin"]))
    height = max(1.0, float(crop_box["yMax"]) - float(crop_box["yMin"]))
    scale = min(1.0, MAX_RENDER_SIDE / max(width, height))
    image = Image.new("RGB", (max(1, int(width * scale)), max(1, int(height * scale))), "white")
    draw = ImageDraw.Draw(image)

    def point_xy(point: dict[str, Any]) -> tuple[float, float]:
        return (
            (float(point.get("x", 0)) - float(crop_box["xMin"])) * scale,
            (float(point.get("y", 0)) - float(crop_box["yMin"])) * scale,
        )

    for stroke in strokes:
        outline = stroke.get("outlinePoints") or []
        raw = stroke.get("rawPoints") or []
        if isinstance(outline, list) and len(outline) >= 3:
            draw.polygon([point_xy(point) for point in outline if isinstance(point, dict)], fill="black")
        elif isinstance(raw, list) and len(raw) >= 2:
            draw.line([point_xy(point) for point in raw if isinstance(point, dict)], fill="black", width=max(2, int(7 * scale)))

    for box in boxes or []:
        normalized = bbox_or_none(box)
        if not normalized:
            continue
        xy = [
            (normalized["xMin"] - crop_box["xMin"]) * scale,
            (normalized["yMin"] - crop_box["yMin"]) * scale,
            (normalized["xMax"] - crop_box["xMin"]) * scale,
            (normalized["yMax"] - crop_box["yMin"]) * scale,
        ]
        draw.rectangle(xy, outline="red", width=max(2, int(3 * scale)))

    image.save(path)


def normalize_vlm_response(raw_vlm: dict[str, Any]) -> dict[str, Any]:
    content = raw_vlm
    if isinstance(raw_vlm, dict) and isinstance(raw_vlm.get("choices"), list):
        try:
            content = raw_vlm["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AuditSchemaError("VLM response did not include choices[0].message.content") from exc

    if isinstance(content, str):
        parsed = parse_json_content(content)
    elif isinstance(content, dict):
        parsed = content
    else:
        raise AuditSchemaError("VLM response content was not JSON")

    latex_lines = parsed.get("latexLines")
    if not isinstance(latex_lines, list):
        raise AuditSchemaError("VLM JSON must include latexLines as a list")

    normalized_lines = [normalize_vlm_latex_line(item) for item in latex_lines if str(item or "").strip()]
    line_observations = parsed.get("lineObservations") if isinstance(parsed.get("lineObservations"), list) else []
    visual_marks = parsed.get("visualMarks") if isinstance(parsed.get("visualMarks"), list) else []
    annotation_attachments = parsed.get("annotationAttachments") if isinstance(parsed.get("annotationAttachments"), list) else []
    normalized_attachments = flatten_annotation_attachments(annotation_attachments, source="vlm")
    if not normalized_attachments:
        normalized_attachments = attachments_from_visual_marks(visual_marks)
    return {
        "latexLines": normalized_lines,
        "lineObservations": [sanitize_json(item) for item in line_observations if isinstance(item, dict)],
        "visualMarks": [sanitize_json(item) for item in visual_marks if isinstance(item, dict)],
        "annotationAttachments": normalized_attachments,
        "overallConfidence": clamp_float(parsed.get("overallConfidence"), default=None),
        "notes": str(parsed.get("notes") or "").strip(),
    }


def parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise AuditSchemaError("VLM response was not valid JSON") from None
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise AuditSchemaError("VLM JSON root must be an object")
    return parsed


def normalize_vlm_latex_line(value: Any) -> str:
    text = str(value or "").strip()
    if "=" not in text:
        return text
    compact = re.sub(r"\s+", "", text).lower()
    if "o" in compact and not re.search(r"[a-np-z]", compact):
        text = re.sub(r"(?<![A-Za-z])O(?![A-Za-z])", "0", text)
        text = re.sub(r"(?<![A-Za-z])o(?![A-Za-z])", "0", text)
    return text


def compare_audit_results(
    fast_result: dict[str, Any],
    normalized_vlm: dict[str, Any],
    vlm_grading: dict[str, Any],
) -> dict[str, Any]:
    discrepancies: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    observation_only_discrepancies: list[dict[str, Any]] = []
    fast_grading = fast_result.get("grading") or {}
    fast_status = nested_get(fast_grading, ["result", "problemStatus"])
    vlm_status = nested_get(vlm_grading, ["result", "problemStatus"])
    if fast_status != vlm_status:
        discrepancies.append({
            "type": "problem_status_mismatch",
            "source": "grading",
            "description": f"Fast grading status {fast_status!r} differs from VLM transcript grading status {vlm_status!r}.",
            "fast": fast_status,
            "slow": vlm_status,
        })

    fast_found = sorted(map(str, nested_get(fast_grading, ["result", "foundSolutions"], []) or []))
    vlm_found = sorted(map(str, nested_get(vlm_grading, ["result", "foundSolutions"], []) or []))
    if fast_found != vlm_found:
        discrepancies.append({
            "type": "solution_set_mismatch",
            "source": "grading",
            "description": f"Fast found solutions {fast_found!r}; VLM transcript found {vlm_found!r}.",
            "fast": fast_found,
            "slow": vlm_found,
        })

    fast_lines = [str(item or "").strip() for item in fast_result.get("latexLines") or []]
    vlm_lines = normalized_vlm.get("latexLines") or []
    if len(fast_lines) != len(vlm_lines):
        discrepancies.append({
            "type": "line_count_mismatch",
            "source": "ocr",
            "description": f"Fast pipeline returned {len(fast_lines)} lines; VLM read {len(vlm_lines)} lines.",
            "fast": len(fast_lines),
            "slow": len(vlm_lines),
        })
    for index, (fast_line, vlm_line) in enumerate(zip(fast_lines, vlm_lines)):
        if normalize_latex_for_compare(fast_line) != normalize_latex_for_compare(vlm_line):
            discrepancies.append({
                "type": "line_latex_mismatch",
                "source": "ocr",
                "lineIndex": index,
                "description": f"Line {index + 1} differs between fast OCR and VLM read.",
                "fast": fast_line,
                "slow": vlm_line,
            })

    fast_attachments = flatten_annotation_attachments(
        fast_result.get("annotationAttachments") or extract_fast_annotation_attachments(fast_result),
        source="fast",
    )
    vlm_attachments = flatten_annotation_attachments(
        normalized_vlm.get("annotationAttachments") or [],
        source="vlm",
    )
    if attachment_signatures(fast_attachments) != attachment_signatures(vlm_attachments):
        discrepancies.append({
            "type": "equation_side_operation_annotation_mismatch",
            "source": "repair",
            "description": "Detached operation annotation attachments differ between fast OCR repair and VLM context read.",
            "fast": fast_attachments,
            "slow": vlm_attachments,
        })

    visual_marks = normalized_vlm.get("visualMarks") or []
    if visual_marks:
        visual_observation = {
            "type": "visual_intent_observed",
            "source": "vlm_normalization",
            "description": f"VLM observed {len(visual_marks)} visual mark(s).",
            "visualMarks": visual_marks,
        }
        observations.append(visual_observation)
        observation_only_discrepancies.append(visual_observation)

    low_confidence = vlm_confidence_is_low(normalized_vlm)
    if low_confidence:
        discrepancies.append({
            "type": "vlm_low_confidence",
            "source": "vlm_normalization",
            "description": "VLM reported low confidence for the whole answer or one line.",
        })

    return {
        "status": "complete",
        "fastProblemStatus": fast_status,
        "vlmProblemStatus": vlm_status,
        "discrepancies": discrepancies,
        "observations": observations,
        "observation_only_discrepancies": observation_only_discrepancies,
        "description": describe_discrepancies(discrepancies),
    }


def is_problem_input_audit(payload: dict[str, Any]) -> bool:
    metadata = payload.get("problemMetadata") or {}
    return metadata.get("auditSubject") == "problem-input"


def failure_comparison(kind: str, error: str, fast_result: dict[str, Any]) -> dict[str, Any]:
    source = "requesting_vlm"
    if kind == "vlm_schema_error":
        source = "vlm_normalization"
    elif kind == "audit_internal_error":
        source = "audit_internal"
    return {
        "status": "failed",
        "fastProblemStatus": nested_get(fast_result.get("grading") or {}, ["result", "problemStatus"]),
        "vlmProblemStatus": None,
        "discrepancies": [{
            "type": kind,
            "source": source,
            "description": error,
        }],
        "observations": [],
        "observation_only_discrepancies": [],
        "description": error,
    }


def build_event_summary(
    *,
    audit_id: str,
    created_at: datetime,
    payload: dict[str, Any],
    comparison: dict[str, Any],
    audit_dir: Path,
    artifact_paths: dict[str, str],
    normalized: Optional[dict[str, Any]],
    vlm_grading: Optional[dict[str, Any]],
    failure_kind: Optional[str],
    failure_stage: str,
    prompt_version: str,
    attempts: list[dict[str, Any]],
    attached_images: list[str],
) -> dict[str, Any]:
    discrepancies = comparison.get("discrepancies") or []
    observations = comparison.get("observations") or []
    last_attempt = attempts[-1] if attempts else {}
    flattened_attachments = flatten_annotation_attachments((normalized or {}).get("annotationAttachments") or [], source="vlm")
    return {
        "auditId": audit_id,
        "createdAt": created_at.isoformat().replace("+00:00", "Z"),
        "eventStage": "terminal",
        "problemId": payload.get("problemId"),
        "inputSignature": payload.get("inputSignature"),
        "attemptId": payload.get("attemptId"),
        "previousAuditId": payload.get("previousAuditId"),
        "triggerReasons": payload.get("triggerReasons") or [],
        "auditSubject": (payload.get("problemMetadata") or {}).get("auditSubject"),
        "comparisonStatus": comparison.get("status"),
        "failureKind": failure_kind,
        "failureStage": failure_stage,
        "failure_stage": failure_stage,
        "discrepancyCount": len(discrepancies),
        "discrepancyTypes": [item.get("type") for item in discrepancies if isinstance(item, dict)],
        "discrepancySources": [item.get("source") for item in discrepancies if isinstance(item, dict)],
        "observationCount": len(observations),
        "observationTypes": [item.get("type") for item in observations if isinstance(item, dict)],
        "observation_only_discrepancies": comparison.get("observation_only_discrepancies") or [],
        "description": comparison.get("description") or "",
        "auditDir": str(audit_dir),
        "artifacts": artifact_paths,
        "artifactTypes": sorted(artifact_paths.keys()),
        "fastProblemStatus": comparison.get("fastProblemStatus"),
        "vlmProblemStatus": comparison.get("vlmProblemStatus"),
        "vlmLatexLines": (normalized or {}).get("latexLines") or [],
        "annotation_attachments": flattened_attachments,
        "annotationAttachmentCount": len(flattened_attachments),
        "vlmGradingFailed": bool((vlm_grading or {}).get("failed")),
        "prompt_version": prompt_version,
        "request_id": last_attempt.get("request_id"),
        "queue_ms": last_attempt.get("queue_ms"),
        "inference_ms": last_attempt.get("inference_ms"),
        "attached_images": list(attached_images),
    }


def build_audit_metadata(
    *,
    settings: ServerSettings,
    audit_id: str,
    payload: dict[str, Any],
    comparison: dict[str, Any],
    artifact_paths: dict[str, str],
    crop_boxes: dict[str, Any],
    attempts: list[dict[str, Any]],
    failure_kind: Optional[str],
    failure_stage: str,
    prompt_version: str,
    normalized: Optional[dict[str, Any]],
    attached_images: list[str],
) -> dict[str, Any]:
    last_attempt = attempts[-1] if attempts else {}
    flattened_attachments = flatten_annotation_attachments((normalized or {}).get("annotationAttachments") or [], source="vlm")
    return {
        "auditId": audit_id,
        "model": settings.vlm_audit_model,
        "timeoutSeconds": settings.vlm_audit_timeout_seconds,
        "attempts": attempts,
        "vlmElapsedSeconds": sum(
            float(attempt.get("elapsedSeconds") or 0)
            for attempt in attempts
            if isinstance(attempt, dict)
        ),
        "failureKind": failure_kind,
        "failureStage": failure_stage,
        "failure_stage": failure_stage,
        "comparisonStatus": comparison.get("status"),
        "observations": comparison.get("observations") or [],
        "observation_only_discrepancies": comparison.get("observation_only_discrepancies") or [],
        "problemId": payload.get("problemId"),
        "inputSignature": payload.get("inputSignature"),
        "attemptId": payload.get("attemptId"),
        "previousAuditId": payload.get("previousAuditId"),
        "triggerReasons": payload.get("triggerReasons") or [],
        "auditSubject": (payload.get("problemMetadata") or {}).get("auditSubject"),
        "cropBoxes": crop_boxes,
        "imagePaths": artifact_paths,
        "artifactTypes": sorted(artifact_paths.keys()),
        "promptVersion": prompt_version,
        "prompt_version": prompt_version,
        "requestId": last_attempt.get("request_id"),
        "request_id": last_attempt.get("request_id"),
        "queueMs": last_attempt.get("queue_ms"),
        "queue_ms": last_attempt.get("queue_ms"),
        "inferenceMs": last_attempt.get("inference_ms"),
        "inference_ms": last_attempt.get("inference_ms"),
        "attachedImages": list(attached_images),
        "attached_images": list(attached_images),
        "annotationAttachments": flattened_attachments,
        "annotation_attachments": flattened_attachments,
    }


def first_failure_kind(comparison: dict[str, Any]) -> Optional[str]:
    if comparison.get("status") != "failed":
        return None
    for discrepancy in comparison.get("discrepancies") or []:
        if isinstance(discrepancy, dict):
            kind = str(discrepancy.get("type") or "").strip()
            if kind:
                return kind
    return "failed"


def request_id_for_response(raw_vlm: Any) -> Optional[str]:
    if isinstance(raw_vlm, dict):
        value = raw_vlm.get("id") or nested_get(raw_vlm, ["response", "id"])
        if value:
            return str(value)
    return None


def schema_error_payload(exc: AuditSchemaError, raw_vlm: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": str(exc),
        "failureKind": "vlm_schema_error",
        "failureStage": "normalizing_vlm",
        "parserError": str(exc),
    }
    raw_response = raw_vlm if raw_vlm is not None else exc.raw_response
    if raw_response is not None:
        payload["rawResponse"] = sanitize_json(raw_response)
    return payload


def unavailable_error_payload(exc: Exception, failure_stage: str) -> dict[str, Any]:
    return {
        "error": str(exc),
        "failureKind": "vlm_unavailable",
        "failureStage": failure_stage,
    }


def fast_line_boxes(fast_result: dict[str, Any]) -> list[dict[str, float]]:
    boxes: list[dict[str, float]] = []
    for line in fast_result.get("lines") or []:
        box = bbox_or_none(line.get("tightBbox") if isinstance(line, dict) else None)
        if box:
            boxes.append(box)
    if boxes:
        return boxes
    for candidate in nested_get(fast_result, ["segmentation", "selected"], []) or []:
        box = bbox_or_none(candidate.get("tightBbox") if isinstance(candidate, dict) else None)
        if box:
            boxes.append(box)
    return boxes


def extract_fast_annotation_attachments(fast_result: dict[str, Any]) -> list[dict[str, Any]]:
    attachments: list[dict[str, Any]] = []
    for line in fast_result.get("lines") or []:
        if not isinstance(line, dict):
            continue
        attachment = line.get("annotationAttachment")
        if isinstance(attachment, dict):
            attachments.append(attachment)
    return attachments


def image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def bbox_for_strokes(strokes: list[dict[str, Any]]) -> Optional[dict[str, float]]:
    boxes = [bbox_or_none(stroke.get("canvasBbox")) for stroke in strokes if isinstance(stroke, dict)]
    boxes = [box for box in boxes if box]
    current = None
    for box in boxes:
        current = union_bbox(current, box)
    return current


def bbox_or_none(value: Any) -> Optional[dict[str, float]]:
    if not isinstance(value, dict):
        return None
    try:
        box = {
            "xMin": float(value["xMin"]),
            "yMin": float(value["yMin"]),
            "xMax": float(value["xMax"]),
            "yMax": float(value["yMax"]),
        }
    except (KeyError, TypeError, ValueError):
        return None
    if box["xMax"] <= box["xMin"] or box["yMax"] <= box["yMin"]:
        return None
    return box


def union_bbox(left: Optional[dict[str, float]], right: Optional[dict[str, float]]) -> Optional[dict[str, float]]:
    if not left:
        return dict(right) if right else None
    if not right:
        return dict(left)
    return {
        "xMin": min(left["xMin"], right["xMin"]),
        "yMin": min(left["yMin"], right["yMin"]),
        "xMax": max(left["xMax"], right["xMax"]),
        "yMax": max(left["yMax"], right["yMax"]),
    }


def padded_bbox(box: Optional[dict[str, float]], padding: float) -> Optional[dict[str, float]]:
    if not box:
        return None
    return {
        "xMin": box["xMin"] - padding,
        "yMin": box["yMin"] - padding,
        "xMax": box["xMax"] + padding,
        "yMax": box["yMax"] + padding,
    }


def clamp_float(value: Any, *, default: Optional[float]) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def flatten_annotation_attachments(attachments: list[dict[str, Any]], *, source: str) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for item in attachments or []:
        if not isinstance(item, dict):
            continue
        operator_latex = str(item.get("operator_latex") or item.get("operatorLatex") or "").strip()
        if not operator_latex:
            operator_latex = str(item.get("latex") or "").strip()
        equation_side = str(item.get("equation_side") or item.get("equationSide") or "").strip().lower()
        if equation_side not in {"left", "right", "both"}:
            equation_side = "both" if "both" in str(item.get("notes") or "").lower() else "unknown"
        flattened.append(sanitize_json({
            "source": source,
            "operator_latex": operator_latex,
            "operand": str(item.get("operand") or "").strip() or None,
            "target_line_index": safe_int(item.get("target_line_index") if "target_line_index" in item else item.get("targetLineIndex")),
            "target_candidate_id": str(item.get("target_candidate_id") or item.get("targetCandidateId") or "").strip() or None,
            "equation_side": equation_side,
            "paired_annotation_id": str(item.get("paired_annotation_id") or item.get("pairedAnnotationId") or "").strip() or None,
            "attachment_confidence": clamp_float(item.get("attachment_confidence") if "attachment_confidence" in item else item.get("attachmentConfidence"), default=None),
            "notes": str(item.get("notes") or "").strip() or None,
            "anchor_bbox": sanitize_json(item.get("anchor_bbox") or item.get("anchorBbox")),
            "annotation_bbox": sanitize_json(item.get("annotation_bbox") or item.get("annotationBbox")),
            "repaired_latex": str(item.get("repaired_latex") or item.get("repairedLatex") or "").strip() or None,
        }))
    return flattened


def attachments_from_visual_marks(visual_marks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attachments: list[dict[str, Any]] = []
    for mark in visual_marks or []:
        if not isinstance(mark, dict):
            continue
        if str(mark.get("type") or "").strip() != "detached_operation_annotation":
            continue
        attachments.append({
            "source": "vlm",
            "operator_latex": str(mark.get("latex") or "").strip(),
            "operand": None,
            "target_line_index": safe_int(mark.get("lineIndex")),
            "target_candidate_id": None,
            "equation_side": str(mark.get("equationSide") or "unknown").strip().lower(),
            "paired_annotation_id": None,
            "attachment_confidence": clamp_float(mark.get("confidence"), default=None),
            "notes": str(mark.get("notes") or "").strip() or None,
            "anchor_bbox": None,
            "annotation_bbox": None,
            "repaired_latex": None,
        })
    return attachments


def attachment_signatures(attachments: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    signatures = []
    for item in attachments or []:
        if not isinstance(item, dict):
            continue
        signatures.append((
            safe_int(item.get("target_line_index")),
            str(item.get("equation_side") or "unknown"),
            normalize_latex_for_compare(str(item.get("operand") or item.get("operator_latex") or "")),
        ))
    return sorted(signatures)


def safe_int(value: Any) -> Optional[int]:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return integer


def vlm_confidence_is_low(normalized: dict[str, Any]) -> bool:
    overall = normalized.get("overallConfidence")
    if isinstance(overall, (int, float)) and overall < 0.5:
        return True
    for observation in normalized.get("lineObservations") or []:
        confidence = observation.get("confidence") if isinstance(observation, dict) else None
        try:
            if float(confidence) < 0.5:
                return True
        except (TypeError, ValueError):
            continue
    return False


def describe_discrepancies(discrepancies: list[dict[str, Any]]) -> str:
    if not discrepancies:
        return "Fast pipeline and VLM audit agreed on the checked signals."
    labels = [str(item.get("type") or "unknown") for item in discrepancies if isinstance(item, dict)]
    return "VLM audit raised: " + ", ".join(labels)


def normalize_latex_for_compare(value: str) -> str:
    normalized = str(value or "").lower()
    previous = None
    while previous != normalized:
        previous = normalized
        normalized = normalized.replace(r"\\", "\\")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"\^\{([^{}])\}", r"^\1", normalized)
    normalized = re.sub(r"_\{([^{}])\}", r"_\1", normalized)
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    if "=" in normalized:
        normalized = re.sub(r"(?<![a-z])o(?![a-z])", "0", normalized)
    return normalized


def nested_get(value: Any, keys: list[str], default: Any = None) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def sanitize_json(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(key): sanitize_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [sanitize_json(item) for item in value]
        return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(sanitize_json(payload), indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_json(payload), sort_keys=True))
        handle.write("\n")


__all__ = [
    "AuditSchemaError",
    "OpenAICompatibleVlmClient",
    "RecognitionAuditService",
    "compare_audit_results",
    "normalize_vlm_response",
]
