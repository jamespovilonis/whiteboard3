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

from src.grading import grade_equation_payload, grade_math_payload
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
        content = self.content
        if isinstance(content, list):
            index = min(len(self.calls) - 1, len(content) - 1)
            content = content[index]
        return {
            "choices": [{
                "message": {
                    "content": content,
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
            self.assertTrue((audit_dir / "answer_context.png").exists())
            self.assertTrue((audit_dir / "fast_overlay.png").exists())
            self.assertTrue((audit_dir / "audit_metadata.json").exists())
            self.assertTrue((audit_dir / "vlm_grading.json").exists())
            self.assertTrue((Path(directory) / "audit_events.jsonl").exists())
            metadata = json.loads((audit_dir / "audit_metadata.json").read_text())
            self.assertEqual(metadata["comparisonStatus"], "complete")
            self.assertIn("answerContext", metadata["artifactTypes"])
            self.assertIn("answerContext", metadata["cropBoxes"])
            self.assertEqual(metadata["prompt_version"], "recognition-audit-v2")
            self.assertEqual(metadata["attached_images"], ["answerCrop", "answerContext"])
            self.assertEqual(metadata["vlmRequestProfile"], "answer-local-compact")
            self.assertEqual(metadata["attempts"][0]["attachedImages"], ["answerCrop", "answerContext"])
            self.assertEqual(metadata["attempts"][0]["vlmRequestProfile"], "answer-local-compact")
            self.assertIn("auditIdTimestamp", metadata)
            self.assertIn("queuedAt", metadata)
            self.assertIn("startedAt", metadata)
            self.assertIn("completedAt", metadata)
            self.assertIsInstance(metadata["queueMs"], (int, float))
            self.assertIsInstance(metadata["runElapsedSeconds"], (int, float))
            self.assertEqual(summary["comparisonStatus"], "complete")
            self.assertIn("answerContext", summary["artifactTypes"])
            self.assertEqual(summary["attached_images"], ["answerCrop", "answerContext"])
            self.assertEqual(summary["vlmRequestProfile"], "answer-local-compact")
            self.assertEqual(summary["eventStage"], "terminal")
            self.assertEqual(len(service.vlm_client.calls[0]["image_paths"]), 2)
            self.assertEqual(
                [path.name for path in service.vlm_client.calls[0]["image_paths"]],
                ["answer_crop.png", "answer_context.png"],
            )

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
            self.assertIn("grading", comparison["discrepancies"][0]["source"])

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

            self.assertEqual(summary["discrepancyCount"], 0)
            self.assertEqual(summary["observationTypes"], ["visual_intent_observed"])
            comparison = json.loads((Path(summary["auditDir"]) / "comparison.json").read_text())
            self.assertEqual(comparison["observations"][0]["type"], "visual_intent_observed")
            self.assertEqual(len(comparison["observation_only_discrepancies"]), 1)

    def test_evaluate_expression_vlm_transcript_uses_math_grader(self):
        with tempfile.TemporaryDirectory() as directory:
            service = service_for(directory, {
                "latexLines": ["7"],
                "lineObservations": [{"lineIndex": 0, "latex": "7", "confidence": 0.9}],
                "visualMarks": [],
                "overallConfidence": 0.9,
                "notes": "clear",
            })
            summary = service.run_audit(evaluate_audit_payload(), "audit_evaluate")

            self.assertEqual(summary["discrepancyCount"], 0)
            self.assertEqual(summary["vlmProblemStatus"], "correct")
            vlm_grading = json.loads((Path(summary["auditDir"]) / "vlm_grading.json").read_text())
            self.assertEqual(vlm_grading["problem"]["resolvedProblemType"], "evaluate-expression")
            self.assertEqual(vlm_grading["result"]["problemStatus"], "correct")

    def test_invalid_json_records_schema_error(self):
        with tempfile.TemporaryDirectory() as directory:
            service = service_for(directory, "not json")
            summary = service.run_audit(audit_payload(), "audit_invalid_json")

            self.assertEqual(summary["discrepancyTypes"], ["vlm_schema_error"])
            self.assertEqual(summary["failureKind"], "vlm_schema_error")
            self.assertTrue((Path(summary["auditDir"]) / "vlm_raw.json").exists())
            raw = json.loads((Path(summary["auditDir"]) / "vlm_raw.json").read_text())
            self.assertEqual(raw["failureKind"], "vlm_schema_error")
            self.assertIn("parserError", raw)

    def test_schema_failure_is_retried_and_preserves_attempt_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            service = service_for(directory, [
                '{"lineObservations":[]}',
                {
                    "latexLines": ["x = 4"],
                    "lineObservations": [{"lineIndex": 0, "latex": "x = 4", "confidence": 0.9}],
                    "visualMarks": [],
                    "overallConfidence": 0.9,
                },
            ])
            summary = service.run_audit(audit_payload(), "audit_retry")

            audit_dir = Path(summary["auditDir"])
            metadata = json.loads((audit_dir / "audit_metadata.json").read_text())
            self.assertEqual(summary["discrepancyCount"], 0)
            self.assertEqual(len(metadata["attempts"]), 2)
            self.assertEqual(metadata["attempts"][0]["failureKind"], "vlm_schema_error")
            self.assertTrue((audit_dir / "vlm_raw_attempt_1.json").exists())
            self.assertEqual(metadata["attempts"][1]["prompt_version"], "recognition-audit-v2")

    def test_unavailable_vlm_records_unavailable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            service = service_for(directory, None, error=RuntimeError("ollama offline"))
            summary = service.run_audit(audit_payload(), "audit_unavailable")

            self.assertEqual(summary["discrepancyTypes"], ["vlm_unavailable"])
            self.assertEqual(summary["failureKind"], "vlm_unavailable")
            self.assertIn("ollama offline", summary["description"])
            metadata = json.loads((Path(summary["auditDir"]) / "audit_metadata.json").read_text())
            self.assertEqual(len(metadata["attempts"]), 1)
            self.assertEqual(metadata["failure_stage"], "requesting_vlm")

    def test_repeated_timeouts_open_circuit_without_extra_vlm_call(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeVlmClient(None, error=RuntimeError("timed out"))
            settings = ServerSettings(
                audit_enabled=True,
                audit_log_dir=directory,
                vlm_audit_base_url="http://127.0.0.1:11434/v1",
                vlm_audit_model="qwen3-vl:8b",
                vlm_audit_circuit_failures=2,
                vlm_audit_circuit_seconds=60,
            )
            service = RecognitionAuditService(settings, vlm_client=client)

            first = service.run_audit(audit_payload(), "audit_timeout_1")
            second = service.run_audit(audit_payload(), "audit_timeout_2")
            third = service.run_audit(audit_payload(), "audit_timeout_3")
            suppressed = service.run_audit(audit_payload(), "audit_timeout_4")

            self.assertEqual(len(client.calls), 2)
            self.assertEqual(first["discrepancyTypes"], ["vlm_timeout"])
            self.assertEqual(second["failureKind"], "vlm_timeout")
            self.assertEqual(third["comparisonStatus"], "skipped")
            self.assertEqual(third["failureKind"], "vlm_circuit_open")
            self.assertEqual(third["discrepancyTypes"], [])
            self.assertIn("retryAfterSeconds", third)
            self.assertEqual(suppressed["eventStage"], "suppressed")
            self.assertEqual(suppressed["comparisonStatus"], "skipped")
            self.assertEqual(suppressed["suppressedByAuditId"], "audit_timeout_3")
            self.assertNotIn("auditDir", suppressed)
            event_lines = [
                json.loads(line)
                for line in (Path(directory) / "audit_events.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(event_lines[-1]["eventStage"], "suppressed")
            self.assertEqual(event_lines[-1]["suppressedByAuditId"], "audit_timeout_3")

    def test_new_discrepancy_categories_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            service = service_for(directory, {
                "latexLines": ["x = 4"],
                "lineObservations": [{"lineIndex": 0, "latex": "x = 4", "confidence": 0.9}],
                "visualMarks": [],
                "overallConfidence": 0.9,
            })
            payload = audit_payload()
            payload["fastResult"]["latex"] = ""
            payload["fastResult"]["latexLines"] = []
            payload["fastResult"]["lines"][0]["acceptedLatex"] = ""
            payload["fastResult"]["lines"][0]["latex"] = ""
            payload["fastResult"]["lines"][0]["ocrLatex"] = ""
            payload["fastResult"]["selectionSummary"] = {
                "highConfidenceDiscarded": [{
                    "candidateId": "discarded-full",
                    "latex": "x = 4",
                    "strokeIds": ["stroke-a"],
                    "grading": {
                        "classification": "valid_step",
                        "solutionCoverage": "full",
                        "matchedSolutions": ["4"],
                    },
                }],
            }
            summary = service.run_audit(payload, "audit_new_categories")

            self.assertIn("line_segmentation_empty", summary["discrepancyTypes"])
            self.assertIn("candidate_present_not_selected", summary["discrepancyTypes"])

    def test_simplification_policy_disagreement_category_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            service = service_for(directory, {
                "latexLines": ["7x - 2"],
                "lineObservations": [{"lineIndex": 0, "latex": "7x - 2", "confidence": 0.9}],
                "visualMarks": [],
                "overallConfidence": 0.9,
            })
            summary = service.run_audit(simplification_policy_audit_payload(), "audit_simplification_policy")

            self.assertIn("simplification_policy_disagreement", summary["discrepancyTypes"])

    def test_zero_and_letter_o_are_normalized_for_line_compare(self):
        with tempfile.TemporaryDirectory() as directory:
            service = service_for(directory, {
                "latexLines": ["5 = O"],
                "lineObservations": [{"lineIndex": 0, "latex": "5 = O", "confidence": 0.9}],
                "visualMarks": [],
                "overallConfidence": 0.9,
                "notes": "clear",
            })
            summary = service.run_audit(zero_o_audit_payload(), "audit_zero_o")

            self.assertEqual(summary["discrepancyCount"], 0)

    def test_annotation_attachment_mismatch_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            service = service_for(directory, {
                "latexLines": [r"\frac{7}{3}"],
                "lineObservations": [{"lineIndex": 0, "latex": r"\frac{7}{3}", "confidence": 0.9}],
                "visualMarks": [],
                "annotationAttachments": [{
                    "operatorLatex": "(x+1)",
                    "targetLineIndex": 0,
                    "equationSide": "left",
                    "attachmentConfidence": 0.9,
                }],
                "overallConfidence": 0.9,
                "notes": "detached operation seen on left side",
            })
            summary = service.run_audit(annotation_audit_payload(), "audit_annotation_mismatch")

            self.assertIn("equation_side_operation_annotation_mismatch", summary["discrepancyTypes"])
            metadata = json.loads((Path(summary["auditDir"]) / "audit_metadata.json").read_text())
            self.assertEqual(metadata["annotation_attachments"][0]["equation_side"], "left")

    def test_grader_failure_still_writes_terminal_event(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = ServerSettings(
                audit_enabled=True,
                audit_log_dir=directory,
                vlm_audit_base_url="http://127.0.0.1:11434/v1",
                vlm_audit_model="qwen3-vl:8b",
            )
            service = RecognitionAuditService(
                settings,
                vlm_client=FakeVlmClient({
                    "latexLines": ["x = 4"],
                    "lineObservations": [{"lineIndex": 0, "latex": "x = 4", "confidence": 0.9}],
                    "visualMarks": [],
                    "overallConfidence": 0.9,
                }),
                grader=lambda _payload: (_ for _ in ()).throw(RuntimeError("grader exploded")),
            )
            summary = service.run_audit(audit_payload(), "audit_internal_failure")

            self.assertEqual(summary["failureKind"], "audit_internal_error")
            self.assertEqual(summary["failureStage"], "grading_vlm")
            event_log = (Path(directory) / "audit_events.jsonl").read_text()
            self.assertIn("audit_internal_error", event_log)

    def test_personal_note_writes_global_and_audit_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            service = service_for(directory, {
                "latexLines": ["x = 4"],
                "lineObservations": [{"lineIndex": 0, "latex": "x = 4", "confidence": 0.9}],
                "visualMarks": [],
                "overallConfidence": 0.9,
                "notes": "clear",
            })
            summary = service.run_audit(audit_payload(), "audit_note")

            note = service.add_personal_note({
                "auditId": "audit_note",
                "problemId": "problem-1",
                "note": "Check the circled answer handling.",
                "source": "test",
            })

            self.assertEqual(note["eventStage"], "personal_note")
            self.assertEqual(note["note"], "Check the circled answer handling.")
            global_notes = (Path(directory) / "personal_notes.jsonl").read_text()
            audit_notes = (Path(summary["auditDir"]) / "personal_notes.jsonl").read_text()
            self.assertIn("Check the circled answer handling.", global_notes)
            self.assertIn("Check the circled answer handling.", audit_notes)
            status = service.status("audit_note")
            self.assertEqual(status["personalNote"], "Check the circled answer handling.")
            self.assertEqual(status["personalNoteCount"], 1)

    def test_attach_feedback_writes_audit_artifacts_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            service = service_for(directory, {
                "latexLines": ["x = 4"],
                "lineObservations": [{"lineIndex": 0, "latex": "x = 4", "confidence": 0.9}],
                "visualMarks": [],
                "overallConfidence": 0.9,
                "notes": "clear",
            })
            payload = audit_payload()
            summary = service.run_audit(payload, "audit_feedback")

            record = service.attach_feedback({
                "auditId": "audit_feedback",
                "problemId": payload["problemId"],
                "inputSignature": payload["inputSignature"],
                "attemptId": payload["attemptId"],
                "feedback": {
                    "attemptId": payload["attemptId"],
                    "inputSignature": payload["inputSignature"],
                    "status": "complete",
                    "source": "deterministic",
                    "text": "Correct! Great job!",
                    "model": "qwen3:1.7b",
                    "promptVersion": "math-feedback-v1",
                    "skippedReason": "correct",
                },
            })

            self.assertEqual(record["eventStage"], "feedback")
            self.assertEqual(record["feedbackText"], "Correct! Great job!")
            audit_dir = Path(summary["auditDir"])
            feedback = json.loads((audit_dir / "feedback.json").read_text())
            self.assertEqual(feedback["text"], "Correct! Great job!")
            metadata = json.loads((audit_dir / "audit_metadata.json").read_text())
            self.assertEqual(metadata["feedbackText"], "Correct! Great job!")
            self.assertEqual(metadata["feedbackSource"], "deterministic")
            self.assertIn("Correct! Great job!", (Path(directory) / "feedback_events.jsonl").read_text())
            self.assertIn("Correct! Great job!", (audit_dir / "feedback_events.jsonl").read_text())

    def test_attach_feedback_rejects_mismatched_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            service = service_for(directory, {
                "latexLines": ["x = 4"],
                "lineObservations": [{"lineIndex": 0, "latex": "x = 4", "confidence": 0.9}],
                "visualMarks": [],
                "overallConfidence": 0.9,
                "notes": "clear",
            })
            payload = audit_payload()
            service.run_audit(payload, "audit_feedback_mismatch")

            with self.assertRaises(ValueError):
                service.attach_feedback({
                    "auditId": "audit_feedback_mismatch",
                    "problemId": payload["problemId"],
                    "inputSignature": "different",
                    "attemptId": payload["attemptId"],
                    "feedback": {
                        "attemptId": payload["attemptId"],
                        "inputSignature": "different",
                        "status": "complete",
                        "source": "fallback",
                        "text": "Stale feedback",
                        "promptVersion": "math-feedback-v1",
                    },
                })

    def test_problem_input_adjustment_audit_skips_grading_and_compares_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            service = service_for(directory, {
                "latexLines": [r"\sqrt{\frac{x^{10}}{x^2}}"],
                "lineObservations": [{"lineIndex": 0, "latex": r"\sqrt{\frac{x^{10}}{x^2}}", "confidence": 0.9}],
                "visualMarks": [],
                "overallConfidence": 0.9,
                "notes": "problem input",
            })
            summary = service.run_audit(problem_input_audit_payload(), "audit_problem_input")

            self.assertEqual(summary["auditSubject"], "problem-input")
            self.assertIn("user_adjusted_problem_input", summary["triggerReasons"])
            self.assertIn("line_latex_mismatch", summary["discrepancyTypes"])
            self.assertIn("problem_input_ocr_mismatch", summary["discrepancyTypes"])
            audit_dir = Path(summary["auditDir"])
            vlm_grading = json.loads((audit_dir / "vlm_grading.json").read_text())
            self.assertEqual(vlm_grading["status"], "skipped")
            metadata = json.loads((audit_dir / "audit_metadata.json").read_text())
            self.assertEqual(metadata["auditSubject"], "problem-input")
            self.assertEqual(metadata["attached_images"], ["answerCrop", "problemCrop"])
            self.assertEqual(metadata["vlmRequestProfile"], "problem-input-local-compact")
            self.assertIn("problem-entry box", service.vlm_client.calls[0]["prompt"])
            self.assertEqual(
                [path.name for path in service.vlm_client.calls[0]["image_paths"]],
                ["answer_crop.png", "problem_crop.png"],
            )

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
        "attemptId": "attempt_test",
        "triggerReasons": ["normal_sample"],
        "promptVersion": "recognition-audit-v2",
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
            "annotationAttachments": [],
            "grading": fast_grading,
            "lines": [{
                "lineIndex": 0,
                "candidateId": "line-a",
                "strokeIds": ["stroke-a"],
                "tightBbox": {"xMin": 80, "yMin": 210, "xMax": 170, "yMax": 270},
                "acceptedLatex": "x = 4",
                "latex": "x = 4",
                "annotationAttachment": None,
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


def problem_input_audit_payload() -> dict[str, Any]:
    payload = audit_payload()
    payload.update({
        "problemId": "problem-input-simplify",
        "problemLatex": "",
        "problemMetadata": {
            "auditSubject": "problem-input",
            "mode": "simplify",
            "problemType": "simplify-expression",
            "source": "user-handwriting",
        },
        "inputSignature": "problem-input-adjust::sig",
        "triggerReasons": ["user_adjusted_problem_input"],
        "fastResult": {
            "latex": r"\sqrt{\frac{x^10}{x^2}",
            "latexLines": [r"\sqrt{\frac{x^10}{x^2}"],
            "lines": [{
                "lineIndex": 0,
                "latex": r"\sqrt{\frac{x^10}{x^2}",
                "tightBbox": {"xMin": 40, "yMin": 60, "xMax": 560, "yMax": 200},
            }],
            "candidatePredictions": [],
            "annotationAttachments": [],
            "segmentation": {"selected": [], "candidates": []},
            "grading": None,
        },
    })
    return payload


def simplification_policy_audit_payload() -> dict[str, Any]:
    payload = audit_payload()
    payload.update({
        "problemId": "simplify-policy",
        "problemLatex": "2 x - 3 + 5 x + 1",
        "problemMetadata": {"problemType": "simplify-expression"},
        "inputSignature": "simplify-policy::sig",
        "triggerReasons": ["normal_sample"],
        "fastResult": {
            **payload["fastResult"],
            "latex": "2 x - 3 + 5 x + 1",
            "latexLines": ["2 x - 3 + 5 x + 1"],
            "grading": {
                "status": "complete",
                "failed": False,
                "problem": {"manifestResponseKind": "simplified_expression"},
                "steps": [{
                    "lineIndex": 0,
                    "studentLatex": "2 x - 3 + 5 x + 1",
                    "classification": "valid_step",
                    "solutionCoverage": "full",
                    "matchedSolutions": ["7*x - 2"],
                    "answerFinality": "unsimplified",
                    "countsTowardCompletion": False,
                }],
                "result": {
                    "problemStatus": "incomplete",
                    "foundSolutions": [],
                    "missingSolutions": ["7*x - 2"],
                },
            },
            "lines": [{
                **payload["fastResult"]["lines"][0],
                "acceptedLatex": "2 x - 3 + 5 x + 1",
                "latex": "2 x - 3 + 5 x + 1",
            }],
        },
    })
    return payload


def evaluate_audit_payload() -> dict[str, Any]:
    payload = audit_payload()
    problem_latex = "10 - 3"
    fast_grading = grade_math_payload({
        "problemLatex": problem_latex,
        "problemMetadata": {"problemType": "evaluate-expression"},
        "lines": [{"lineIndex": 0, "latex": "7"}],
    })
    fast_grading = {"status": "complete", "failed": False, **fast_grading}
    payload.update({
        "problemLatex": problem_latex,
        "problemMetadata": {"problemType": "evaluate-expression"},
        "fastResult": {
            **payload["fastResult"],
            "latex": "7",
            "latexLines": ["7"],
            "grading": fast_grading,
            "lines": [{
                **payload["fastResult"]["lines"][0],
                "acceptedLatex": "7",
                "latex": "7",
            }],
        },
    })
    return payload


def zero_o_audit_payload() -> dict[str, Any]:
    payload = audit_payload()
    payload["problemLatex"] = "x + 5 = x"
    payload["problemMetadata"] = {"problemType": "equation-solving"}
    payload["fastResult"]["latex"] = "5 = 0"
    payload["fastResult"]["latexLines"] = ["5 = 0"]
    payload["fastResult"]["lines"][0]["acceptedLatex"] = "5 = 0"
    payload["fastResult"]["lines"][0]["latex"] = "5 = 0"
    payload["fastResult"]["grading"] = {
        "status": "complete",
        "failed": False,
        **grade_equation_payload({
            "problemLatex": "x + 5 = x",
            "lines": [{"lineIndex": 0, "latex": "5 = 0"}],
        }),
    }
    return payload


def annotation_audit_payload() -> dict[str, Any]:
    payload = audit_payload()
    payload["problemLatex"] = r"\frac{x-1}{x+1}=4"
    payload["triggerReasons"] = ["detached_operation_annotation"]
    payload["fastResult"]["latex"] = r"\frac{x-1}{x+1}=4"
    payload["fastResult"]["latexLines"] = [r"\frac{x-1}{x+1}=4"]
    payload["fastResult"]["lines"][0]["acceptedLatex"] = r"\frac{x-1}{x+1}=4"
    payload["fastResult"]["lines"][0]["latex"] = r"\frac{x-1}{x+1}=4"
    payload["fastResult"]["annotationAttachments"] = [{
        "operatorLatex": "(x+1)",
        "targetLineIndex": 0,
        "equationSide": "both",
        "pairedAnnotationId": "pair:0|(x+1)",
        "attachmentConfidence": 0.95,
    }]
    return payload


if __name__ == "__main__":
    unittest.main()
