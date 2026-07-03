"""Primitive tutoring feedback service backed by local Ollama."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from typing import Any, Optional
import urllib.error
import urllib.request

from src.server.config import ServerSettings


JSON_HEADERS = {"Content-Type": "application/json"}
FEEDBACK_PROMPT_VERSION = "math-feedback-v1"
SYSTEM_PROMPT = (
    "You provide concise math feedback. Do not grade correctness. "
    "Use only the supplied JSON. Give one valid correction or next step. "
    "Return plain text only, no markdown, in at most two sentences."
)


class OpenAICompatibleTextClient:
    """Small OpenAI-compatible chat-completions client for local text models."""

    def __init__(self, *, base_url: str, model: str, timeout_seconds: float):
        self.base_url = str(base_url or "").rstrip("/")
        self.model = str(model or "").strip()
        self.timeout_seconds = max(1.0, float(timeout_seconds or 10.0))

    def complete(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if not self.base_url:
            raise RuntimeError("Feedback base URL is not configured")
        if not self.model:
            raise RuntimeError("Feedback model is not configured")

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": 120,
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
            raise RuntimeError(f"Feedback LLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Feedback LLM request failed: {exc.reason}") from exc


class MathFeedbackService:
    """Generate short post-grading feedback without evaluating correctness."""

    def __init__(
        self,
        settings: ServerSettings,
        *,
        llm_client: Optional[OpenAICompatibleTextClient] = None,
    ):
        self.settings = settings
        self.llm_client = llm_client or OpenAICompatibleTextClient(
            base_url=settings.feedback_base_url,
            model=settings.feedback_model,
            timeout_seconds=settings.feedback_timeout_seconds,
        )

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = payload or {}
        grading = feedback_grading(payload)
        status = problem_status(grading)
        base = {
            "attemptId": payload.get("attemptId"),
            "inputSignature": payload.get("inputSignature"),
            "status": "complete",
            "model": self.settings.feedback_model,
            "promptVersion": FEEDBACK_PROMPT_VERSION,
            "createdAt": now_iso(),
        }

        if status == "correct":
            return {
                **base,
                "source": "deterministic",
                "text": correct_feedback_text(),
                "skippedReason": "correct",
            }

        prompt_context = build_prompt_context(payload, grading)
        user_prompt = (
            "Provide feedback for this graded math attempt. "
            "Give one valid correction or next step.\n"
            f"{json.dumps(prompt_context, sort_keys=True)}"
        )
        try:
            raw = self.llm_client.complete(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
            text = trim_feedback_text(extract_chat_content(raw))
            if not text:
                raise RuntimeError("Feedback LLM returned empty text")
            return {
                **base,
                "source": "ollama",
                "text": text,
                "promptContext": prompt_context,
            }
        except Exception as exc:
            return {
                **base,
                "source": "fallback",
                "text": deterministic_fallback_feedback(prompt_context),
                "error": str(exc),
                "promptContext": prompt_context,
            }


def build_prompt_context(payload: dict[str, Any], grading: dict[str, Any] | None = None) -> dict[str, Any]:
    grading = grading or feedback_grading(payload)
    fast_result = payload.get("fastResult") or {}
    steps = list(grading.get("steps") or []) if isinstance(grading, dict) else []
    status = problem_status(grading)
    first_invalid = first_invalid_step(steps)
    valid_steps = [
        compact_step(step)
        for step in steps
        if isinstance(step, dict) and step.get("classification") == "valid_step"
    ]
    return {
        "problemId": payload.get("problemId"),
        "problemLatex": payload.get("problemLatex") or "",
        "problemType": (payload.get("problemMetadata") or {}).get("problemType") or payload.get("problemType") or "",
        "problemStatus": status or "unknown",
        "problem": compact_problem(grading.get("problem") if isinstance(grading, dict) else None),
        "result": grading.get("result") if isinstance(grading, dict) else None,
        "latexLines": fast_result.get("latexLines") or payload.get("latexLines") or [],
        "validSteps": valid_steps,
        "firstInvalidLine": compact_step(first_invalid) if first_invalid else None,
        "instruction": "Give one valid correction or next step.",
    }


def feedback_grading(payload: dict[str, Any]) -> dict[str, Any] | None:
    grading = payload.get("grading")
    if isinstance(grading, dict):
        return grading
    fast_grading = (payload.get("fastResult") or {}).get("grading")
    return fast_grading if isinstance(fast_grading, dict) else None


def problem_status(grading: dict[str, Any] | None) -> str:
    return str(((grading or {}).get("result") or {}).get("problemStatus") or "")


def first_invalid_step(steps: list[Any]) -> dict[str, Any] | None:
    for step in steps:
        if isinstance(step, dict) and step.get("classification") == "invalid_step":
            return step
    return None


def compact_problem(problem: Any) -> dict[str, Any]:
    if not isinstance(problem, dict):
        return {}
    return {
        "manifest": problem.get("manifest"),
        "solutionSet": problem.get("solutionSet"),
        "cardinality": problem.get("cardinality"),
        "manifestResponseKind": problem.get("manifestResponseKind"),
    }


def compact_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "lineIndex": step.get("lineIndex"),
        "studentLatex": step.get("studentLatex") or step.get("latex") or "",
        "classification": step.get("classification"),
        "solutionCoverage": step.get("solutionCoverage"),
        "matchedSolutions": step.get("matchedSolutions") or [],
        "answerFinality": step.get("answerFinality"),
        "countsTowardCompletion": step.get("countsTowardCompletion"),
    }


def deterministic_fallback_feedback(context: dict[str, Any]) -> str:
    status = str(context.get("problemStatus") or "")
    first_invalid = context.get("firstInvalidLine") or {}
    missing = (((context.get("result") or {}).get("missingSolutions")) or [])
    if status == "incorrect" and first_invalid.get("studentLatex"):
        return f"Check line {int(first_invalid.get('lineIndex') or 0) + 1}: {first_invalid.get('studentLatex')} is not a valid step. Try one valid correction or next step from the previous line."
    if status == "incomplete" and missing:
        return f"Keep going toward {missing[0]}. Give one valid next step from your last correct line."
    if status == "not_started":
        return "Start by rewriting the problem or applying one valid operation to both sides."
    return "Give one valid correction or next step from your latest work."


def correct_feedback_text() -> str:
    return "Correct! Great job!"


def extract_chat_content(raw: dict[str, Any]) -> str:
    choices = raw.get("choices") if isinstance(raw, dict) else None
    if not choices:
        return ""
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return " ".join(parts)
    return ""


def trim_feedback_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return ""
    sentences = re.findall(r"[^.!?]+[.!?]?", compact)
    trimmed = " ".join(sentence.strip() for sentence in sentences[:2] if sentence.strip())
    return trimmed[:500].strip()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "FEEDBACK_PROMPT_VERSION",
    "MathFeedbackService",
    "OpenAICompatibleTextClient",
    "build_prompt_context",
    "correct_feedback_text",
    "deterministic_fallback_feedback",
    "trim_feedback_text",
]
