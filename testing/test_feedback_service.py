#!/usr/bin/env python3
"""Tests for primitive math feedback generation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.server.config import ServerSettings
from src.server.services.feedback import (
    MathFeedbackService,
    acceptable_llm_feedback,
    build_feedback_prompt,
    build_prompt_context,
    extract_chat_content,
)


class FakeTextClient:
    def __init__(self, content: str = "", *, error: Exception | None = None):
        self.content = content
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if self.error is not None:
            raise self.error
        return {
            "choices": [{
                "message": {
                    "content": self.content,
                },
            }],
        }


class MathFeedbackServiceTests(unittest.TestCase):
    def test_correct_grading_skips_llm(self):
        client = FakeTextClient("should not be used")
        service = service_with(client)

        result = service.generate(feedback_payload(problem_status="correct"))

        self.assertEqual(result["source"], "deterministic")
        self.assertEqual(result["text"], "Correct! Great job!")
        self.assertEqual(result["skippedReason"], "correct")
        self.assertEqual(client.calls, [])

    def test_incorrect_grading_builds_prompt_context_and_uses_llm(self):
        client = FakeTextClient("Line 2 should be: x = 4. Divide both sides by 2.")
        service = service_with(client)

        result = service.generate(feedback_payload(problem_status="incorrect"))

        self.assertEqual(result["source"], "ollama")
        self.assertIn("Line 2", result["text"])
        self.assertEqual(len(client.calls), 1)
        self.assertIn("The correct target line is exactly: x = 4", client.calls[0]["user_prompt"])
        self.assertNotIn("Give one valid correction or next step", client.calls[0]["user_prompt"])
        context = build_prompt_context(feedback_payload(problem_status="incorrect"))
        self.assertEqual(context["firstInvalidLine"]["lineIndex"], 1)
        self.assertEqual(context["firstInvalidLine"]["studentLatex"], "x = 5")
        self.assertEqual(context["targetLine"], "x = 4")
        self.assertEqual(context["anchorLine"]["studentLatex"], "2x = 8")

    def test_llm_error_returns_fallback_with_metadata(self):
        service = service_with(FakeTextClient(error=TimeoutError("too slow")))

        result = service.generate(feedback_payload(problem_status="incomplete"))

        self.assertEqual(result["source"], "fallback")
        self.assertIn("too slow", result["error"])
        self.assertTrue(result["text"])
        self.assertIn("x = 4", result["text"])
        self.assertNotIn("Give one valid correction or next step", result["text"])
        self.assertEqual(result["attemptId"], "attempt_a")

    def test_llm_json_echo_falls_back_to_concrete_target_line(self):
        service = service_with(FakeTextClient('{"targetLine":"x = 4"}'))

        result = service.generate(feedback_payload(problem_status="incomplete"))

        self.assertEqual(result["source"], "fallback")
        self.assertIn("x = 4", result["text"])
        self.assertNotIn("Give one valid correction or next step", result["text"])

    def test_llm_output_must_include_exact_target_line(self):
        context = build_prompt_context(feedback_payload(problem_status="incomplete"))
        self.assertFalse(acceptable_llm_feedback("A good next line is x=4.", context))
        self.assertFalse(acceptable_llm_feedback("Student line to respond to: 2x = 8. The correct target line is exactly: x = 4.", context))
        self.assertTrue(acceptable_llm_feedback("A good next line is: x = 4.", context))

    def test_feedback_prompt_is_not_raw_json(self):
        context = build_prompt_context(feedback_payload(problem_status="incomplete"))
        prompt = build_feedback_prompt(context)

        self.assertIn("The correct target line is exactly: x = 4", prompt)
        self.assertFalse(prompt.lstrip().startswith("{"))

    def test_native_ollama_response_parser_extracts_message_content(self):
        self.assertEqual(
            extract_chat_content({
                "message": {
                    "role": "assistant",
                    "content": "Line 1 should be: 4*x = 11.",
                    "reasoning": "hidden thinking",
                },
            }),
            "Line 1 should be: 4*x = 11.",
        )

    def test_linear_problem_target_prefers_next_step_from_problem(self):
        context = build_prompt_context(feedback_payload(
            problem_latex="4x - 3 = 8",
            problem_status="incorrect",
            steps=[{
                "lineIndex": 0,
                "studentLatex": "x = 44",
                "classification": "invalid_step",
                "solutionCoverage": "none",
                "matchedSolutions": [],
            }],
            solution="11/4",
        ))

        self.assertEqual(context["targetLine"], "4*x = 11")
        self.assertEqual(context["targetLineSource"], "linear_isolate_term")
        self.assertIsNone(context["anchorLine"])

    def test_linear_incomplete_target_divides_coefficient(self):
        context = build_prompt_context(feedback_payload(
            problem_latex="4x - 3 = 8",
            problem_status="incomplete",
            steps=[{
                "lineIndex": 0,
                "studentLatex": "4x = 11",
                "classification": "valid_step",
                "solutionCoverage": "none",
                "matchedSolutions": [],
            }],
            solution="11/4",
        ))

        self.assertEqual(context["targetLine"], "x = 11/4")
        self.assertEqual(context["targetLineSource"], "linear_divide_coefficient")
        self.assertEqual(context["anchorLine"]["studentLatex"], "4x = 11")


def service_with(client: FakeTextClient) -> MathFeedbackService:
    return MathFeedbackService(ServerSettings(feedback_model="qwen3:1.7b"), llm_client=client)


def feedback_payload(
    problem_status: str = "incorrect",
    *,
    problem_latex: str = "2x + 3 = 11",
    steps: list[dict[str, Any]] | None = None,
    solution: str = "4",
) -> dict[str, Any]:
    if steps is None:
        steps = [{
            "lineIndex": 0,
            "studentLatex": "2x = 8",
            "classification": "valid_step",
            "solutionCoverage": "none",
            "matchedSolutions": [],
        }]
        if problem_status == "incorrect":
            steps.append({
                "lineIndex": 1,
                "studentLatex": "x = 5",
                "classification": "invalid_step",
                "solutionCoverage": "none",
                "matchedSolutions": [],
            })
    grading = {
        "status": "complete",
        "failed": False,
        "problem": {
            "manifest": {
                "cardinality": "finite",
                "exact_set": [solution],
                "variable": "x",
            },
            "solutionSet": [solution],
        },
        "steps": steps,
        "result": {
            "problemStatus": problem_status,
            "foundSolutions": [],
            "missingSolutions": [solution],
        },
    }
    return {
        "problemId": "problem-a",
        "problemLatex": problem_latex,
        "problemMetadata": {"problemType": "equation-solving"},
        "inputSignature": "sig-a",
        "attemptId": "attempt_a",
        "grading": grading,
        "fastResult": {
            "latexLines": [step["studentLatex"] for step in steps],
            "grading": grading,
        },
    }


if __name__ == "__main__":
    unittest.main()
