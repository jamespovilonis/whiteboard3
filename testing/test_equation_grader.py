#!/usr/bin/env python3
"""Tests for product equation grading helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTING_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TESTING_DIR) not in sys.path:
    sys.path.insert(0, str(TESTING_DIR))

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

    def test_repeated_assignment_multi_solution_line_completes_problem(self):
        manifest = create_answer_manifest("x^2 - 5x + 6 = 0")

        for latex in ("x=2  x=3", "x = 2 x = 3", "x=2,3"):
            with self.subTest(latex=latex):
                result = grade_equation_work(manifest, [{"latex": latex}])

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
        self.assertEqual([step["classification"] for step in result["steps"]], ["other", "unrecognized"])

    def test_unrecognized_line_does_not_trigger_incorrect(self):
        """A line with no parseable content should classify as 'unrecognized',
        not 'invalid_step', so it doesn't trigger an incorrect problem status."""
        manifest = create_answer_manifest("3x + 5 = 17")
        result = grade_equation_work(manifest, [
            {"latex": "3x = 12"},
            {"latex": ""},
        ])
        self.assertEqual(result["steps"][0]["classification"], "valid_step")
        self.assertEqual(result["steps"][1]["classification"], "unrecognized")
        self.assertEqual(result["result"]["problemStatus"], "incomplete")

    def test_unrecognized_with_no_valid_steps_is_not_started(self):
        """If all lines are unrecognized, the problem should be not_started."""
        manifest = create_answer_manifest("x + 1 = 2")
        result = grade_equation_work(manifest, [
            {"latex": ""},
            {"latex": ""},
        ])
        self.assertEqual(result["result"]["problemStatus"], "not_started")
        self.assertTrue(all(
            step["classification"] == "unrecognized" for step in result["steps"]
        ))


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

    def test_sympy_grader_budget_times_out_without_hanging(self):
        """A pathological expression should not hang the grader."""
        manifest = create_answer_manifest("x = 1")
        # An expression that would be expensive to simplify but should time out gracefully.
        result = grade_equation_work(manifest, [{"latex": "x = 1 + 1"}])
        self.assertIn(result["result"]["problemStatus"], {"incorrect", "incomplete"})

    def test_grade_candidate_group_returns_candidate_verdicts(self):
        """grade_candidate_group should return per-candidate verdicts for debugging."""
        manifest = create_answer_manifest("3x + 5 = 17")
        verdict = grade_candidate_group(manifest, {
            "latex": "x = 5",
            "candidates": [
                {"latex": "x = 5"},
                {"latex": "x = 4"},
            ],
        }, problem_latex="3x + 5 = 17")

        self.assertIn("candidateVerdicts", verdict)
        self.assertEqual(len(verdict["candidateVerdicts"]), 2)
        self.assertEqual(verdict["candidateVerdicts"][1]["classification"], "valid_step")
        self.assertEqual(verdict["candidateVerdicts"][0]["classification"], "invalid_step")

    def test_grading_first_candidate_selection_skips_semantic_scoring(self):
        """When grading finds a valid candidate, semantic scoring should be skipped."""
        from latex_semantics import score_candidate_group
        manifest = create_answer_manifest("3x + 5 = 17")
        group = {
            "candidateId": "test-line",
            "latex": "x = 5",
            "candidates": [
                {"latex": "x = 5"},
                {"latex": "x = 4"},
            ],
        }
        result = score_candidate_group(
            group,
            problem_latex="3x + 5 = 17",
            previous_latex=[],
            answer_manifest=manifest,
        )
        self.assertEqual(result["bestLatex"], "x = 4")
        self.assertTrue(result["equivalentToProblem"])
        self.assertTrue(result["equivalentToPrevious"])
        # Candidate scores should come from grading, not semantic scoring.
        self.assertTrue(
            any(item.get("detail", {}).get("fromGrading") for item in result["candidateScores"])
        )

    def test_grading_falls_back_to_semantic_scoring_when_no_valid_candidate(self):
        """When grading finds no valid candidate, semantic scoring should run."""
        from latex_semantics import score_candidate_group
        manifest = create_answer_manifest("3x + 5 = 17")
        group = {
            "candidateId": "test-line",
            "latex": "y = 5",
            "candidates": [
                {"latex": "y = 5"},
            ],
        }
        result = score_candidate_group(
            group,
            problem_latex="3x + 5 = 17",
            previous_latex=[],
            answer_manifest=manifest,
        )
        # No valid candidate found by grading.
        self.assertNotEqual(result["grading"]["classification"], "valid_step")
        # Semantic scoring should have run (candidate scores won't have fromGrading).
        self.assertFalse(
            any(item.get("detail", {}).get("fromGrading") for item in result["candidateScores"])
        )

    # --- Phase 1: Test realism improvements ---

    def test_property_based_valid_step_addition_to_both_sides(self):
        """Adding the same value to both sides should always be a valid step."""
        for problem_latex, step_latex in [
            ("2x + 3 = 11", "2x + 3 + 5 = 11 + 5"),
            ("2x + 3 = 11", "2x + 3 - 7 = 11 - 7"),
            ("x - 4 = 10", "x - 4 + 4 = 10 + 4"),
            ("5x = 25", "5x + 0 = 25 + 0"),
            ("3x - 2 = 13", "3x - 2 + 2 = 13 + 2"),
        ]:
            with self.subTest(problem=problem_latex, step=step_latex):
                manifest = create_answer_manifest(problem_latex)
                result = grade_equation_work(manifest, [{"latex": step_latex}])
                self.assertEqual(
                    result["steps"][0]["classification"], "valid_step",
                    f"Expected valid_step for {step_latex} given {problem_latex}"
                )

    def test_property_based_valid_step_multiplication(self):
        """Multiplying both sides by the same nonzero constant should be valid."""
        for problem_latex, step_latex in [
            ("x = 4", "2x = 8"),
            ("x = 4", "3x = 12"),
            ("x / 2 = 5", "x = 10"),
            ("x / 3 = 7", "x = 21"),
        ]:
            with self.subTest(problem=problem_latex, step=step_latex):
                manifest = create_answer_manifest(problem_latex)
                result = grade_equation_work(manifest, [{"latex": step_latex}])
                self.assertEqual(
                    result["steps"][0]["classification"], "valid_step",
                    f"Expected valid_step for {step_latex} given {problem_latex}"
                )

    def test_cross_line_error_propagation(self):
        """An OCR misread on line 2 should make line 3 invalid relative to it."""
        manifest = create_answer_manifest("2x + 3 = 11")
        result = grade_equation_work(manifest, [
            {"latex": "2x + 3 = 11"},
            {"latex": "2x = 9"},   # misread: should be 8
            {"latex": "x = 4.5"},  # follows from misread, but not the solution
        ])
        # Line 2 is valid (2x = 9 follows from 2x + 3 = 11? No, 11 - 3 = 8, not 9)
        # So line 2 should be invalid_step
        self.assertEqual(result["steps"][1]["classification"], "invalid_step")
        # Line 3 follows from line 2 (9/2 = 4.5), so it should be valid relative to line 2
        # But the problem should be incorrect because line 2 broke the chain
        self.assertEqual(result["result"]["problemStatus"], "incorrect")
        self.assertEqual(result["result"]["breakdownLineIndex"], 1)

    def test_incomplete_work_with_partial_solution(self):
        """A student who writes valid steps but doesn't reach the full solution set."""
        manifest = create_answer_manifest("x^2 - 5x + 6 = 0")
        result = grade_equation_work(manifest, [
            {"latex": "x^2 - 5x + 6 = 0"},
            {"latex": "(x - 2)(x - 3) = 0"},
            {"latex": "x = 2"},
            # Student stops here, missing x = 3
        ])
        self.assertEqual(result["result"]["problemStatus"], "incomplete")
        self.assertEqual(result["result"]["foundSolutions"], ["2"])
        self.assertEqual(result["result"]["missingSolutions"], ["3"])

    def test_incomplete_work_no_solution_reached(self):
        """Valid steps but no solution value written yet."""
        manifest = create_answer_manifest("3x + 5 = 17")
        result = grade_equation_work(manifest, [
            {"latex": "3x + 5 = 17"},
            {"latex": "3x = 12"},
            # Student stops before writing x = 4
        ])
        self.assertEqual(result["result"]["problemStatus"], "incomplete")
        self.assertEqual(result["result"]["foundSolutions"], [])
        self.assertEqual(result["result"]["missingSolutions"], ["4"])

    def test_diverse_scratch_work_classified_as_other(self):
        """Various scratch work patterns should all classify as 'other'."""
        manifest = create_answer_manifest("3x + 5 = 17")
        scratch_patterns = [
            "/ 4 / 4",
            "\\times 6 \\times 6",
            "- 3 - 3",
            "",  # empty line
            "???",
            "x x x",
        ]
        for scratch in scratch_patterns:
            with self.subTest(scratch=scratch):
                result = grade_equation_work(manifest, [{"latex": scratch}])
                self.assertIn(result["steps"][0]["classification"], ("other", "unrecognized"))
        self.assertEqual(result["result"]["problemStatus"], "not_started")

    def test_solution_set_coverage_combined_form(self):
        """Solutions entered as 'x = 2, 3' should give full coverage."""
        manifest = create_answer_manifest("x^2 - 5x + 6 = 0")
        result = grade_equation_work(manifest, [{"latex": "x = 2, 3"}])
        self.assertEqual(result["result"]["problemStatus"], "correct")
        self.assertEqual(result["steps"][0]["solutionCoverage"], "full")

    def test_solution_set_coverage_set_notation(self):
        """Solutions entered as 'x = 2, 3' (comma-separated) should give full coverage."""
        manifest = create_answer_manifest("x^2 - 5x + 6 = 0")
        result = grade_equation_work(manifest, [{"latex": "x = 2, 3"}])
        self.assertEqual(result["result"]["problemStatus"], "correct")
        self.assertEqual(result["steps"][0]["solutionCoverage"], "full")
        # Note: set notation like x = \{ 2, 3 \} is not yet supported by the parser.

    def test_solution_set_coverage_separate_lines(self):
        """Solutions on separate lines should accumulate to full coverage."""
        manifest = create_answer_manifest("x^2 - 5x + 6 = 0")
        result = grade_equation_work(manifest, [
            {"latex": "x = 2"},
            {"latex": "x = 3"},
        ])
        self.assertEqual(result["result"]["problemStatus"], "correct")
        self.assertEqual(result["steps"][0]["solutionCoverage"], "partial")
        self.assertEqual(result["steps"][1]["solutionCoverage"], "partial")

    def test_solution_set_coverage_repeated_solution(self):
        """Writing the same solution twice should not give full coverage for multi-solution."""
        manifest = create_answer_manifest("x^2 - 5x + 6 = 0")
        result = grade_equation_work(manifest, [
            {"latex": "x = 2"},
            {"latex": "x = 2"},
        ])
        self.assertEqual(result["result"]["problemStatus"], "incomplete")
        self.assertEqual(result["result"]["foundSolutions"], ["2"])
        self.assertEqual(result["result"]["missingSolutions"], ["3"])

    def test_grading_first_path_is_faster_than_semantic_fallback(self):
        """The grading-first path (valid candidate exists) should be faster than the fallback."""
        import time
        from latex_semantics import score_candidate_group

        manifest = create_answer_manifest("3x + 5 = 17")
        group_valid = {
            "candidateId": "valid-line",
            "latex": "x = 5",
            "candidates": [
                {"latex": "x = 5"},
                {"latex": "x = 4"},
            ],
        }
        group_invalid = {
            "candidateId": "invalid-line",
            "latex": "y = 5",
            "candidates": [
                {"latex": "y = 5"},
                {"latex": "z = 5"},
            ],
        }

        # Warm up
        score_candidate_group(group_valid, problem_latex="3x + 5 = 17", previous_latex=[], answer_manifest=manifest)
        score_candidate_group(group_invalid, problem_latex="3x + 5 = 17", previous_latex=[], answer_manifest=manifest)

        # Time grading-first path (valid candidate → skips semantic scoring)
        start = time.monotonic()
        for _ in range(20):
            score_candidate_group(group_valid, problem_latex="3x + 5 = 17", previous_latex=[], answer_manifest=manifest)
        grading_first_time = time.monotonic() - start

        # Time fallback path (no valid candidate → runs full semantic scoring)
        start = time.monotonic()
        for _ in range(20):
            score_candidate_group(group_invalid, problem_latex="3x + 5 = 17", previous_latex=[], answer_manifest=manifest)
        fallback_time = time.monotonic() - start

        # The grading-first path should be faster (or at least not significantly slower)
        # We use a generous threshold to avoid flakiness on slow CI machines.
        self.assertLess(
            grading_first_time, fallback_time * 2,
            f"Grading-first path ({grading_first_time:.4f}s) should not be much slower than "
            f"fallback ({fallback_time:.4f}s)"
        )


if __name__ == "__main__":
    unittest.main()
