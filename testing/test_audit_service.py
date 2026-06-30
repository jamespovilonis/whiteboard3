#!/usr/bin/env python3
"""Tests for the filesystem-backed VLM audit service."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.grading import grade_equation_payload
from src.server.config import ServerSettings
from src.server.services.audit import RecognitionAuditService, normalize_vlm_response


class FakeVlmClient:
    def __init__(self, content: Any = None, *, error: Exception | None = None):
        self.content = content
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, prompt: str, image_paths: list[Path]) -> dict[str, Any]:
        self.calls.append({"prompt": prompt, "image_paths": image_paths})
        if self.error is not None:
            raise self.error
        return {
            "choices": [{
                "message": {
                    "content": self.content,
                },
            }],
        }


class AuditServiceTests(unittest.TestCase):
    def test_agreement_writes_artifacts_and_empty_discrepancies(self):
        with tempfile.TemporaryDirectory() as directory:
            service = service_for(directory, {
                "latexLines": ["x = 4"],
                "lineObservations": [{"lineIndex": 0, "latex": "x = 4", "confidence": 0.9}],
                "visualMarks": [],
                "overallConfidence": 0.9,
                "notes": "clear",
            })
            summary = service.run_audit(audit_payload(), "audit_agreement")

            self.assertEqual(summary["discrepancyCount"], 0)
            audit_dir = Path(summary["auditDir"])
            self.assertTrue((audit_dir / "problem_crop.png").exists())
            self.assertTrue((audit_dir / "answer_crop.png").exists())
            self.assertTrue((audit_dir / "fast_overlay.png").exists())
            self.assertTrue((audit_dir / "vlm_grading.json").exists())
            self.assertTrue((Path(directory) / "audit_events.jsonl").exists())

    def test_disagreement_records_status_and_solution_discrepancies(self):
        with tempfile.TemporaryDirectory() as directory:
            service = service_for(directory, {
                "latexLines": ["x = 5"],
                "lineObservations": [{"lineIndex": 0, "latex": "x = 5", "confidence": 0.8}],
                "visualMarks": [],
                "overallConfidence": 0.8,
                "notes": "different answer",
            })
            summary = service.run_audit(audit_payload(), "audit_disagreement")

            self.assertGreaterEqual(summary["discrepancyCount"], 1)
            self.assertIn("problem_status_mismatch", summary["discrepancyTypes"])
            comparison = json.loads((Path(summary["auditDir"]) / "comparison.json").read_text())
            self.assertEqual(comparison["fastProblemStatus"], "correct")
            self.assertEqual(comparison["vlmProblemStatus"], "incorrect")

    def test_visual_mark_only_records_visual_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            service = service_for(directory, {
                "latexLines": ["x = 4"],
                "lineObservations": [{"lineIndex": 0, "latex": "x = 4", "confidence": 0.9}],
                "visualMarks": [{"type": "circled_answer", "lineIndex": 0, "confidence": 0.8}],
                "overallConfidence": 0.9,
                "notes": "answer circled",
            })
            summary = service.run_audit(audit_payload(), "audit_visual_mark")

            self.assertEqual(summary["discrepancyCount"], 1)
            self.assertEqual(summary["discrepancyTypes"], ["visual_intent_observed"])

    def test_invalid_json_records_schema_error(self):
        with tempfile.TemporaryDirectory() as directory:
            service = service_for(directory, "not json")
            summary = service.run_audit(audit_payload(), "audit_invalid_json")

            self.assertEqual(summary["discrepancyTypes"], ["vlm_schema_error"])
            self.assertTrue((Path(summary["auditDir"]) / "vlm_raw.json").exists())

    def test_unavailable_vlm_records_unavailable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            service = service_for(directory, None, error=RuntimeError("ollama offline"))
            summary = service.run_audit(audit_payload(), "audit_unavailable")

            self.assertEqual(summary["discrepancyTypes"], ["vlm_unavailable"])
            self.assertIn("ollama offline", summary["description"])

    def test_normalize_vlm_response_accepts_fenced_json(self):
        normalized = normalize_vlm_response({
            "choices": [{
                "message": {
                    "content": '```json\n{"latexLines":["x = 4"],"overallConfidence":0.7}\n```',
                },
            }],
        })

        self.assertEqual(normalized["latexLines"], ["x = 4"])
        self.assertEqual(normalized["overallConfidence"], 0.7)


def service_for(directory: str, content: Any, *, error: Exception | None = None) -> RecognitionAuditService:
    settings = ServerSettings(
        audit_enabled=True,
        audit_log_dir=directory,
        vlm_audit_base_url="http://127.0.0.1:11434/v1",
        vlm_audit_model="qwen3-vl:8b",
    )
    return RecognitionAuditService(
        settings,
        vlm_client=FakeVlmClient(content, error=error),
    )


def audit_payload() -> dict[str, Any]:
    problem_latex = "x + 1 = 5"
    fast_grading = grade_equation_payload({
        "problemLatex": problem_latex,
        "lines": [{"lineIndex": 0, "latex": "x = 4"}],
    })
    fast_grading = {"status": "complete", "failed": False, **fast_grading}
    return {
        "problemId": "problem-a",
        "problemLatex": problem_latex,
        "problemMetadata": {},
        "problemBox": {"xMin": 0, "yMin": 0, "xMax": 500, "yMax": 160},
        "answerBox": {"xMin": 0, "yMin": 160, "xMax": 500, "yMax": 360},
        "inputSignature": "problem-a::stroke-a",
        "triggerReasons": ["normal_sample"],
        "strokes": [{
            "id": "stroke-a",
            "rawPoints": [
                {"x": 80, "y": 210, "pressure": 0.5},
                {"x": 160, "y": 260, "pressure": 0.5},
            ],
            "outlinePoints": [
                {"x": 80, "y": 210},
                {"x": 170, "y": 210},
                {"x": 170, "y": 270},
                {"x": 80, "y": 270},
            ],
            "canvasBbox": {"xMin": 80, "yMin": 210, "xMax": 170, "yMax": 270},
        }],
        "fastResult": {
            "latex": "x = 4",
            "latexLines": ["x = 4"],
            "grading": fast_grading,
            "lines": [{
                "lineIndex": 0,
                "candidateId": "line-a",
                "strokeIds": ["stroke-a"],
                "tightBbox": {"xMin": 80, "yMin": 210, "xMax": 170, "yMax": 270},
                "acceptedLatex": "x = 4",
                "latex": "x = 4",
            }],
            "segmentation": {
                "selected": [{
                    "candidateId": "line-a",
                    "strokeIds": ["stroke-a"],
                    "tightBbox": {"xMin": 80, "yMin": 210, "xMax": 170, "yMax": 270},
                }]
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
