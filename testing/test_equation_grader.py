#!/usr/bin/env python3
"""Tests for product equation grading helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.grading import create_answer_manifest, grade_candidate_group, grade_equation_payload, grade_equation_work


class EquationGraderManifestTests(unittest.TestCase):
    def test_creates_linear_manifest_with_variable_inference(self):
        manifest = create_answer_manifest("3x + 5 = 17")

        self.assertEqual(manifest["variable"], "x")
        self.assertEqual(manifest["cardinality"], "finite")
        self.assertEqual(manifest["exact_set"], ["4"])
        self.assertEqual(manifest["decimal_set"], [4.0])
        self.assertIn("x=4", manifest["acceptable_strings"])
        self.assertIn("4", manifest["acceptable_strings"])

    def test_creates_quadratic_manifest(self):
        manifest = create_answer_manifest("x^2 - 5x + 6 = 0")

        self.assertEqual(manifest["cardinality"], "finite")
        self.assertEqual(manifest["exact_set"], ["2", "3"])
        self.assertEqual(manifest["decimal_set"], [2.0, 3.0])

    def test_creates_irrational_manifest(self):
        manifest = create_answer_manifest("x^2 - x - 1 = 0")

        self.assertEqual(manifest["cardinality"], "finite")
        self.assertEqual(
            manifest["exact_set"],
            ["1/2 - sqrt(5)/2", "1/2 + sqrt(5)/2"],
        )
        self.assertEqual(manifest["decimal_set"], [-0.618, 1.618])
        self.assertEqual(manifest["tolerance"], 0.005)

    def test_creates_no_solution_manifest(self):
        manifest = create_answer_manifest("x + 1 = x")

        self.assertEqual(manifest["variable"], "x")
        self.assertEqual(manifest["cardinality"], "none")
        self.assertEqual(manifest["exact_set"], [])
        self.assertIn("nosolution", manifest["acceptable_strings"])

    def test_creates_infinite_solution_manifest(self):
        manifest = create_answer_manifest("x + 1 = x + 1")

        self.assertEqual(manifest["variable"], "x")
        self.assertEqual(manifest["cardinality"], "infinite")
        self.assertEqual(manifest["exact_set"], [])
        self.assertIn("allrealnumbers", manifest["acceptable_strings"])

    def test_uses_explicit_variable_override(self):
        manifest = create_answer_manifest("3t + 5 = 17", variable="t")

        self.assertEqual(manifest["variable"], "t")
        self.assertEqual(manifest["cardinality"], "finite")
        self.assertEqual(manifest["exact_set"], ["4"])


class EquationGraderWorkTests(unittest.TestCase):
    def test_top_five_candidate_recovery_selects_valid_prediction(self):
        manifest = create_answer_manifest("3x + 5 = 17")

        result = grade_equation_work(manifest, [{
            "latex": "x = 5",
            "candidates": [
                {"latex": "x = 5"},
                {"latex": "x = 4"},
            ],
        }])

        self.assertEqual(result["result"]["problemStatus"], "correct")
        self.assertEqual(result["steps"][0]["classification"], "valid_step")
        self.assertEqual(result["steps"][0]["studentLatex"], "x = 4")
        self.assertEqual(result["steps"][0]["selectedCandidateIndex"], 1)

    def test_candidate_group_grading_selects_valid_top_five_prediction(self):
        manifest = create_answer_manifest("3x + 5 = 17")

        verdict = grade_candidate_group(manifest, {
            "latex": "x = 5",
            "candidates": [
                {"latex": "x = 5"},
                {"latex": "x = 4"},
            ],
        }, problem_latex="3x + 5 = 17")

        self.assertEqual(verdict["classification"], "valid_step")
        self.assertEqual(verdict["studentLatex"], "x = 4")
        self.assertEqual(verdict["selectedCandidateIndex"], 1)
        self.assertEqual(verdict["matchedSolutions"], ["4"])
        self.assertEqual(verdict["candidateVerdicts"][1]["classification"], "valid_step")

    def test_separate_multi_solution_lines_complete_problem(self):
        manifest = create_answer_manifest("x^2 - 5x + 6 = 0")

        result = grade_equation_work(manifest, [
            {"latex": "x = 2"},
            {"latex": "x = 3"},
        ])

        self.assertEqual(result["result"]["problemStatus"], "correct")
        self.assertEqual(result["result"]["foundSolutions"], ["2", "3"])
        self.assertEqual([step["solutionCoverage"] for step in result["steps"]], ["partial", "partial"])

    def test_combined_multi_solution_line_completes_problem(self):
        manifest = create_answer_manifest("x^2 - 5x + 6 = 0")

        result = grade_equation_work(manifest, [{"latex": "x = 2, 3"}])

        self.assertEqual(result["result"]["problemStatus"], "correct")
        self.assertEqual(result["steps"][0]["solutionCoverage"], "full")
        self.assertEqual(result["steps"][0]["matchedSolutions"], ["2", "3"])

    def test_partial_solution_is_incomplete(self):
        manifest = create_answer_manifest("x^2 - 5x + 6 = 0")

        result = grade_equation_work(manifest, [{"latex": "x = 2"}])

        self.assertEqual(result["result"]["problemStatus"], "incomplete")
        self.assertEqual(result["result"]["foundSolutions"], ["2"])
        self.assertEqual(result["result"]["missingSolutions"], ["3"])

    def test_invalid_equation_reports_breakdown_line(self):
        manifest = create_answer_manifest("x^2 - 5x + 6 = 0")

        result = grade_equation_work(manifest, [
            {"lineIndex": 4, "latex": "x = 5"},
            {"lineIndex": 5, "latex": "x + 1"},
        ])

        self.assertEqual(result["result"]["problemStatus"], "incorrect")
        self.assertEqual(result["result"]["breakdownLineIndex"], 4)
        self.assertEqual(result["steps"][0]["classification"], "invalid_step")
        self.assertEqual(result["steps"][1]["classification"], "other")

    def test_scratch_only_work_is_not_started(self):
        manifest = create_answer_manifest("3x + 5 = 17")

        result = grade_equation_work(manifest, [
            {"latex": "/ 4 / 4"},
            {"latex": ""},
        ])

        self.assertEqual(result["result"]["problemStatus"], "not_started")
        self.assertEqual([step["classification"] for step in result["steps"]], ["other", "other"])

    def test_accepts_exact_radical_solutions(self):
        manifest = create_answer_manifest("x^2 - 2 = 0")

        result = grade_equation_work(manifest, [{
            "latex": r"x = - \sqrt { 2 }, \sqrt { 2 }",
        }])

        self.assertEqual(result["result"]["problemStatus"], "correct")
        self.assertEqual(result["steps"][0]["solutionCoverage"], "full")

    def test_accepts_rounded_decimal_solutions_with_tolerance(self):
        manifest = create_answer_manifest("x^2 - 2 = 0")

        result = grade_equation_work(manifest, [{"latex": "x = -1.414, 1.414"}])

        self.assertEqual(result["result"]["problemStatus"], "correct")
        self.assertEqual(result["result"]["missingSolutions"], [])

    def test_accepts_no_solution_answer(self):
        manifest = create_answer_manifest("x + 1 = x")

        result = grade_equation_work(manifest, [{"latex": r"\emptyset"}])

        self.assertEqual(result["result"]["problemStatus"], "correct")
        self.assertEqual(result["steps"][0]["acceptedSpecialAnswer"], "none")

    def test_accepts_infinite_solution_answer(self):
        manifest = create_answer_manifest("x + 1 = x + 1")

        result = grade_equation_work(manifest, [{"latex": r"x \in \mathbb { R }"}])

        self.assertEqual(result["result"]["problemStatus"], "correct")
        self.assertEqual(result["steps"][0]["acceptedSpecialAnswer"], "infinite")

    def test_payload_generates_manifest_from_problem_latex(self):
        result = grade_equation_payload({
            "problemLatex": "3x + 5 = 17",
            "lines": [{"latex": "x = 4"}],
        })

        self.assertEqual(result["problem"]["manifest"]["cardinality"], "finite")
        self.assertEqual(result["result"]["problemStatus"], "correct")


if __name__ == "__main__":
    unittest.main()
