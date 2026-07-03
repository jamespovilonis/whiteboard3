"""Primitive tutoring feedback service backed by local Ollama."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from typing import Any, Optional
import urllib.error
import urllib.request

import sympy

from src.grading.equation_grader import GradingParseFailure, parse_math
from src.server.config import ServerSettings


JSON_HEADERS = {"Content-Type": "application/json"}
FEEDBACK_PROMPT_VERSION = "math-feedback-v1"
SYSTEM_PROMPT = (
    "You provide concise math feedback. Do not grade correctness. "
    "Use only the supplied context. Use the exact targetLine in your response. "
    "Do not say 'give one valid correction or next step'. "
    "Return plain text only, no markdown, in at most two sentences."
)


class OllamaChatClient:
    """Small native Ollama chat client for local text models."""

    def __init__(self, *, base_url: str, model: str, timeout_seconds: float):
        self.base_url = normalize_ollama_base_url(base_url)
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
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": 120,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers=JSON_HEADERS,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Feedback Ollama HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Feedback Ollama request failed: {exc.reason}") from exc


class MathFeedbackService:
    """Generate short post-grading feedback without evaluating correctness."""

    def __init__(
        self,
        settings: ServerSettings,
        *,
        llm_client: Optional[OllamaChatClient] = None,
    ):
        self.settings = settings
        self.llm_client = llm_client or OllamaChatClient(
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
        user_prompt = build_feedback_prompt(prompt_context)
        try:
            raw = self.llm_client.complete(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
            text = trim_feedback_text(extract_chat_content(raw))
            if not acceptable_llm_feedback(text, prompt_context):
                raise RuntimeError("Feedback LLM did not include the target line")
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
        **build_target_line_context(
            problem_latex=payload.get("problemLatex") or "",
            problem_type=(payload.get("problemMetadata") or {}).get("problemType") or payload.get("problemType") or "",
            status=status,
            grading=grading if isinstance(grading, dict) else {},
            steps=steps,
            first_invalid=first_invalid,
        ),
        "instruction": "Use the exact targetLine in the response.",
    }


def build_feedback_prompt(context: dict[str, Any]) -> str:
    status = str(context.get("problemStatus") or "unknown")
    target_line = str(context.get("targetLine") or "").strip()
    reason = str(context.get("targetLineReason") or "").strip()
    first_invalid = context.get("firstInvalidLine") or {}
    anchor = context.get("anchorLine") or {}
    student_line = str(first_invalid.get("studentLatex") or "").strip()
    if not student_line:
        student_line = str(anchor.get("studentLatex") or "").strip()
    return "\n".join([
        "Write one short student-facing feedback message.",
        f"Status: {status}",
        f"Student line to respond to: {student_line or 'not started'}",
        f"The correct target line is exactly: {target_line}",
        f"Reason to mention: {reason or 'This is the next valid math line.'}",
        "Rules: include the exact target line, do not output JSON, do not mention these rules, and do not say 'give one valid correction or next step'.",
    ])


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


def build_target_line_context(
    *,
    problem_latex: str,
    problem_type: str,
    status: str,
    grading: dict[str, Any],
    steps: list[Any],
    first_invalid: dict[str, Any] | None,
) -> dict[str, Any]:
    anchor = anchor_step_for_target(status=status, steps=steps, first_invalid=first_invalid)
    target = target_line_for_attempt(
        problem_latex=problem_latex,
        problem_type=problem_type,
        grading=grading,
        anchor=anchor,
    )
    return {
        "anchorLine": compact_step(anchor) if anchor else None,
        "targetLine": target.get("line") or "",
        "targetLineSource": target.get("source") or "fallback",
        "targetLineReason": target.get("reason") or "",
    }


def anchor_step_for_target(*, status: str, steps: list[Any], first_invalid: dict[str, Any] | None) -> dict[str, Any] | None:
    if status == "incorrect" and first_invalid:
        invalid_index = safe_int(first_invalid.get("lineIndex"))
        prior = [
            step for step in steps
            if isinstance(step, dict) and
            step.get("classification") == "valid_step" and
            safe_int(step.get("lineIndex")) is not None and
            invalid_index is not None and
            safe_int(step.get("lineIndex")) < invalid_index
        ]
        if prior:
            return sorted(prior, key=lambda item: safe_int(item.get("lineIndex")) or 0)[-1]
        return None
    valid = [step for step in steps if isinstance(step, dict) and step.get("classification") == "valid_step"]
    if valid:
        return sorted(valid, key=lambda item: safe_int(item.get("lineIndex")) or 0)[-1]
    return None


def target_line_for_attempt(
    *,
    problem_latex: str,
    problem_type: str,
    grading: dict[str, Any],
    anchor: dict[str, Any] | None,
) -> dict[str, str]:
    if str(problem_type or "") == "equation-solving":
        linear = next_linear_equation_line(anchor_latex(anchor) or problem_latex)
        if linear:
            return linear
        answer = answer_form_target(grading)
        if answer:
            return answer
    answer = expression_target(grading)
    if answer:
        return answer
    return {
        "line": str(problem_latex or "").strip(),
        "source": "problem",
        "reason": "Start from the original problem.",
    }


def next_linear_equation_line(latex: str) -> dict[str, str] | None:
    try:
        parsed = parse_math(latex)
    except GradingParseFailure:
        return None
    if parsed.kind != "equation" or parsed.right is None:
        return None
    variables = sorted(parsed.residual.free_symbols, key=lambda item: item.name)
    if len(variables) != 1:
        return None
    variable = variables[0]
    try:
        polynomial = sympy.Poly(sympy.expand(parsed.residual), variable)
    except Exception:
        return None
    if polynomial.degree() != 1:
        return None
    left = sympy.expand(parsed.left)
    right = sympy.simplify(parsed.right)
    coefficient = sympy.simplify(left.coeff(variable))
    constant = sympy.simplify(left - coefficient * variable)

    if coefficient != 0 and not constant.equals(0) and not parsed.right.has(variable):
        next_right = sympy.simplify(right - constant)
        return {
            "line": f"{sympy.sstr(coefficient * variable)} = {sympy.sstr(next_right)}",
            "source": "linear_isolate_term",
            "reason": "Move the constant term to the other side.",
        }

    if coefficient != 0 and constant.equals(0) and not parsed.right.has(variable):
        next_right = sympy.simplify(right / coefficient)
        return {
            "line": f"{sympy.sstr(variable)} = {sympy.sstr(next_right)}",
            "source": "linear_divide_coefficient",
            "reason": "Divide both sides by the coefficient of the variable.",
        }
    return None


def answer_form_target(grading: dict[str, Any]) -> dict[str, str] | None:
    problem = grading.get("problem") or {}
    manifest = problem.get("manifest") or {}
    variable = manifest.get("variable") or problem.get("variable") or infer_variable_from_steps(grading.get("steps") or [])
    solutions = (
        manifest.get("exact_set") or
        problem.get("solutionSet") or
        (grading.get("result") or {}).get("missingSolutions") or
        []
    )
    if not variable or not solutions:
        return None
    return {
        "line": f"{variable} = {solutions[0]}",
        "source": "answer_manifest",
        "reason": "Use the expected solution from the grader manifest.",
    }


def expression_target(grading: dict[str, Any]) -> dict[str, str] | None:
    problem = grading.get("problem") or {}
    manifest = problem.get("manifest") or {}
    solutions = (
        manifest.get("exact_set") or
        problem.get("solutionSet") or
        (grading.get("result") or {}).get("missingSolutions") or
        []
    )
    if not solutions:
        return None
    return {
        "line": str(solutions[0]),
        "source": "answer_manifest",
        "reason": "Use the expected value or simplified expression from the grader manifest.",
    }


def anchor_latex(anchor: dict[str, Any] | None) -> str:
    if not isinstance(anchor, dict):
        return ""
    return str(anchor.get("studentLatex") or anchor.get("latex") or "").strip()


def infer_variable_from_steps(steps: list[Any]) -> str:
    for step in steps:
        text = str((step or {}).get("studentLatex") or "")
        match = re.search(r"\b([A-Za-z])\b", text)
        if match:
            return match.group(1)
    return ""


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def deterministic_fallback_feedback(context: dict[str, Any]) -> str:
    status = str(context.get("problemStatus") or "")
    first_invalid = context.get("firstInvalidLine") or {}
    target_line = str(context.get("targetLine") or "").strip()
    reason = str(context.get("targetLineReason") or "").strip()
    reason_sentence = f" {reason}" if reason else ""
    if status == "incorrect" and first_invalid.get("studentLatex"):
        line_number = int(first_invalid.get("lineIndex") or 0) + 1
        return f"Line {line_number} should be: {target_line or 'a valid next line'}.{reason_sentence}"
    if status == "incomplete":
        return f"A good next line is: {target_line or 'the next valid line'}.{reason_sentence}"
    if status == "not_started":
        return f"Start with: {target_line or 'the original problem'}.{reason_sentence}"
    return f"Use this line: {target_line or 'the next valid line'}.{reason_sentence}"


def acceptable_llm_feedback(text: str, context: dict[str, Any]) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    if stripped.startswith("{") or stripped.startswith("["):
        return False
    lowered = stripped.lower()
    if "give one valid correction or next step" in lowered:
        return False
    if "student line to respond" in lowered or "reason to mention" in lowered:
        return False
    target_line = str(context.get("targetLine") or "").strip()
    if target_line and target_line not in stripped:
        return False
    return True


def correct_feedback_text() -> str:
    return "Correct! Great job!"


def extract_chat_content(raw: dict[str, Any]) -> str:
    message = raw.get("message") if isinstance(raw, dict) else None
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message.get("content") or ""
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


def normalize_ollama_base_url(base_url: str) -> str:
    normalized = str(base_url or "http://127.0.0.1:11434").rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized


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
    "OllamaChatClient",
    "acceptable_llm_feedback",
    "build_feedback_prompt",
    "build_prompt_context",
    "build_target_line_context",
    "correct_feedback_text",
    "deterministic_fallback_feedback",
    "extract_chat_content",
    "trim_feedback_text",
]
