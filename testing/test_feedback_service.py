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
from src.server.services.feedback import MathFeedbackService, build_prompt_context


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
        client = FakeTextClient("Line 2 should keep the equation balanced. Try subtracting 5 from both sides.")
        service = service_with(client)

        result = service.generate(feedback_payload(problem_status="incorrect"))

        self.assertEqual(result["source"], "ollama")
        self.assertIn("Line 2", result["text"])
        self.assertEqual(len(client.calls), 1)
        self.assertIn("Give one valid correction or next step", client.calls[0]["user_prompt"])
        context = build_prompt_context(feedback_payload(problem_status="incorrect"))
        self.assertEqual(context["firstInvalidLine"]["lineIndex"], 1)
        self.assertEqual(context["firstInvalidLine"]["studentLatex"], "x = 5")

    def test_llm_error_returns_fallback_with_metadata(self):
        service = service_with(FakeTextClient(error=TimeoutError("too slow")))

        result = service.generate(feedback_payload(problem_status="incomplete"))

        self.assertEqual(result["source"], "fallback")
        self.assertIn("too slow", result["error"])
        self.assertTrue(result["text"])
        self.assertEqual(result["attemptId"], "attempt_a")


def service_with(client: FakeTextClient) -> MathFeedbackService:
    return MathFeedbackService(ServerSettings(feedback_model="qwen3:1.7b"), llm_client=client)


def feedback_payload(problem_status: str = "incorrect") -> dict[str, Any]:
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
                "exact_set": ["4"],
            },
            "solutionSet": ["4"],
        },
        "steps": steps,
        "result": {
            "problemStatus": problem_status,
            "foundSolutions": [],
            "missingSolutions": ["4"],
        },
    }
    return {
        "problemId": "problem-a",
        "problemLatex": "2x + 3 = 11",
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
