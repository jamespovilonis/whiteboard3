#!/usr/bin/env python3
"""Tests for SymPy-backed OCR candidate scoring."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
if str(TESTING_DIR) not in sys.path:
    sys.path.insert(0, str(TESTING_DIR))

from latex_semantics import (
    character_overlap,
    check_equivalence,
    is_sound_latex,
    math_symbol_weights,
    parse_math,
    score_ocr_predictions,
    score_semantic_payload,
)
from fixture_catalog import PROBLEMS


class LatexSemanticsTests(unittest.TestCase):
    def test_parses_implicit_multiplication_and_fractions(self):
        parsed = parse_math(r"3x + \frac { 1 } { 2 } = 8")
        self.assertEqual(parsed.kind, "equation")
        self.assertIsNotNone(parsed.right)

    def test_parses_greek_variable_commands(self):
        parsed = parse_math(r"\eta + 1 = 6")
        self.assertEqual(parsed.kind, "equation")
        self.assertEqual({symbol.name for symbol in parsed.left.free_symbols}, {"eta"})

    def test_rejects_structurally_unsound_latex(self):
        self.assertTrue(is_sound_latex(r"2 x + 3 = 11"))
        self.assertFalse(is_sound_latex(r"\frac { x + 1 } { } = 5"))
        self.assertFalse(is_sound_latex(r"2 x + = 11"))
        self.assertFalse(is_sound_latex(r"x = \frac { - x + x } { 0 }"))

    def test_treats_operation_annotations_as_sound_lines(self):
        for latex in (r"/ 4       / 4", r"\times 6       \times 6", r"- 3      - 3"):
            with self.subTest(latex=latex):
                parsed = parse_math(latex)
                self.assertEqual(parsed.kind, "operation")
                self.assertTrue(is_sound_latex(latex))

    def test_fixture_catalog_rows_are_parseable_or_sound_annotations(self):
        failures = []
        for problem in PROBLEMS:
            for index, latex in enumerate(problem.lines):
                try:
                    parse_math(latex)
                except Exception as exc:
                    failures.append((problem.name, index, latex, str(exc)))
        self.assertEqual(failures, [])

    def test_detects_equivalent_equation_solution_sets(self):
        result = check_equivalence(r"2 x + 3 = 11", r"2 x = 8")
        self.assertTrue(result.equivalent)
        self.assertIn(result.method, {"constant_residual_factor", "real_solution_set", "residual_difference"})

    def test_parses_logarithms_with_bases_and_detects_equivalent_steps(self):
        parsed = parse_math(r"\log _ { 2 } ( x ) + 3 = 7")
        self.assertEqual(parsed.kind, "equation")

        subtract_step = check_equivalence(
            r"\log _ { 2 } ( x ) + 3 = 7",
            r"\log _ { 2 } ( x ) = 4",
        )
        self.assertTrue(subtract_step.equivalent)

        exponent_step = check_equivalence(
            r"\log _ { 2 } ( x ) = 4",
            r"2 ^ { 4 } = x",
        )
        self.assertTrue(exponent_step.equivalent)

    def test_parses_integral_and_leading_equals_continuation_rows(self):
        integral = parse_math(r"\int _ { 0 } ^ { 2 } ( 3 x ^ { 2 } + 1 ) d x")
        self.assertEqual(integral.kind, "expression")
        integral_with_limits = parse_math(r"\int \limits _ { 1 } ^ { 3 } 2 x d x")
        self.assertEqual(integral_with_limits.kind, "expression")
        continuation = parse_math(r"= ( 8 + 2 ) - ( 0 + 0 )")
        self.assertEqual(continuation.kind, "expression")

        antiderivative = check_equivalence(
            r"\int _ { 0 } ^ { 2 } ( 3 x ^ { 2 } + 1 ) d x",
            r"= [ x ^ { 3 } + x ] _ { 0 } ^ { 2 }",
        )
        self.assertTrue(antiderivative.equivalent)
        final_value = check_equivalence(
            r"\int _ { 0 } ^ { 2 } ( 3 x ^ { 2 } + 1 ) d x",
            r"= 10",
        )
        self.assertTrue(final_value.equivalent)

    def test_scores_any_good_top_five_prediction_against_problem_and_history(self):
        predictions = [
            {"latex": r"2 x + = 8", "score": -5.0, "elapsedSeconds": 1.2},
            {"latex": r"2 x = 8", "score": -1.0, "elapsedSeconds": 1.0},
            {"latex": r"z = 8", "score": -0.7, "elapsedSeconds": 1.0},
        ]
        scored = score_ocr_predictions(
            predictions,
            problem_latex=r"2 x + 3 = 11",
            previous_latex=[r"2 x + 3 = 11"],
        )
        self.assertEqual(scored[0].latex, r"2 x = 8")
        self.assertTrue(scored[0].sound)
        self.assertTrue(scored[0].equivalent_to_problem)

    def test_semantic_payload_returns_grading_and_prefers_solution_candidate(self):
        payload = score_semantic_payload({
            "problemLatex": "3x + 5 = 17",
            "candidateGroups": [{
                "candidateId": "line-1",
                "latex": "x = 5",
                "candidates": [
                    {"latex": "x = 5", "score": 1.0},
                    {"latex": "x = 4", "score": -5.0},
                ],
            }],
        })

        self.assertEqual(payload["answerManifest"]["cardinality"], "finite")
        score = payload["candidateScores"][0]
        self.assertEqual(score["bestLatex"], "x = 4")
        self.assertEqual(score["grading"]["classification"], "valid_step")
        self.assertEqual(score["grading"]["selectedCandidateIndex"], 1)
        self.assertEqual(score["grading"]["matchedSolutions"], ["4"])

    def test_scores_transformed_first_student_line_against_explicit_prompt(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"2 x = 8", "score": -1.0},
                {"latex": r"q = 8", "score": 1.0},
            ],
            problem_latex=r"2 x + 3 = 11",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"2 x = 8")
        self.assertTrue(scored[0].equivalent_to_problem)

    def test_unsound_division_by_zero_does_not_beat_sound_top_candidate(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"x = \frac { - 4 + 6 } { 2 }", "score": -0.8},
                {"latex": r"x = \frac { - x + x } { 0 }", "score": -0.1},
            ],
            problem_latex=r"x ^ { 2 } + 4 x - 5 = 0",
            previous_latex=[
                r"x = \frac { - 4 + \sqrt { 4 ^ { 2 } - 4 ( 1 ) ( - 5 ) } } { 2 ( 1 ) }",
            ],
        )

        self.assertNotEqual(scored[0].latex, r"x = \frac { - x + x } { 0 }")
        self.assertTrue(scored[0].sound)
        unsound = next(item for item in scored if item.latex == r"x = \frac { - x + x } { 0 }")
        self.assertFalse(unsound.sound)

    def test_repairs_quadratic_formula_coefficient_from_problem_context(self):
        scored = score_ocr_predictions(
            [
                {
                    "latex": r"x = \frac { - 1 + \sqrt { 1 ^ { 2 } - 4 ( 1 ) ( - 5 ) } } { 2 ( 1 ) }",
                    "score": -0.5,
                },
            ],
            problem_latex=r"x ^ { 2 } + 4 x - 5 = 0",
            previous_latex=[],
        )

        self.assertEqual(
            scored[0].latex,
            r"x = \frac { - 4 + \sqrt { 4 ^ { 2 } - 4 ( - 5 ) } } { 2 }",
        )
        self.assertEqual(scored[0].detail["repair"], "contextual_quadratic_formula_coefficient")

    def test_repairs_simplified_quadratic_formula_numerator_from_context(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"x = \frac { - 1 + 6 } { 2 }", "score": -0.5},
            ],
            problem_latex=r"x ^ { 2 } + 4 x - 5 = 0",
            previous_latex=[
                r"x = \frac { - 4 + \sqrt { 4 ^ { 2 } - 4 ( 1 ) ( - 5 ) } } { 2 ( 1 ) }",
            ],
        )

        self.assertEqual(scored[0].latex, r"x = \frac { - 4 + 6 } { 2 }")
        self.assertEqual(scored[0].detail["repair"], "contextual_quadratic_formula_coefficient")

    def test_quadratic_formula_numerator_repair_preserves_visible_arithmetic(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"x = \frac { - 1 + 6 } { 2 }", "score": -0.0255},
                {"latex": r"x = \frac { - a + 6 } { 2 }", "score": -0.6164},
            ],
            problem_latex=r"x ^ { 2 } + 4 x - 5 = 0",
            previous_latex=[
                r"x = \frac { - 4 + \sqrt { 4 ^ { 2 } - 4 ( 1 ) ( - 5 ) } } { 2 ( 1 ) }",
            ],
        )

        self.assertEqual(scored[0].latex, r"x = \frac { - 4 + 6 } { 2 }")
        self.assertEqual(scored[0].detail["repair"], "contextual_quadratic_formula_coefficient")
        self.assertNotEqual(scored[0].latex, r"x = \frac { 2 } { 2 }")

    def test_repairs_simplified_quadratic_formula_numerator_without_previous_formula(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"x = \frac { - 1 + 6 } { 2 }", "score": -0.5},
            ],
            problem_latex=r"x ^ { 2 } + 4 x - 5 = 0",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"x = \frac { - 4 + 6 } { 2 }")
        self.assertEqual(scored[0].detail["repair"], "contextual_quadratic_formula_coefficient")

    def test_recent_subtraction_annotation_blocks_stale_denominator_clear_repair(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"3 x = 2 0", "score": -0.06},
                {"latex": r"3 x = 1 0", "score": -1.21},
            ],
            problem_latex=r"\frac { x } { 2 } + \frac { 1 } { 3 } = 2",
            previous_latex=[
                r"\times 6 \times 6",
                r"3 x + 2 = 1 2",
                r"- 2 - 2",
            ],
        )

        self.assertEqual(scored[0].latex, r"3 x = 10")
        self.assertNotEqual(scored[0].detail.get("repair"), "contextual_denominator_clear")

    def test_repairs_malformed_quadratic_formula_sqrt_fraction_boundary(self):
        scored = score_ocr_predictions(
            [
                {
                    "latex": r"x = \frac { - 1 + \sqrt { 4 ^ { 2 } - ( 1 ) ( - 5 ) } { 2 ( 1 ) }",
                    "score": -0.5,
                },
            ],
            problem_latex=r"x ^ { 2 } + 4 x - 5 = 0",
            previous_latex=[],
        )

        self.assertEqual(
            scored[0].latex,
            r"x = \frac { - 4 + \sqrt { 4 ^ { 2 } - 4 ( 1 ) ( - 5 ) } } { 2 ( 1 ) }",
        )
        self.assertTrue(scored[0].sound)
        self.assertEqual(scored[0].detail["repair"], "contextual_quadratic_formula_coefficient")

    def test_repairs_malformed_quadratic_formula_letter_and_multiplier_confusions(self):
        scored = score_ocr_predictions(
            [
                {
                    "latex": r"x = \frac { - a + \sqrt { 4 ^ { 2 } - 1 ( 1 ) ( - 5 ) } { 2 ( 1 ) }",
                    "score": -0.5,
                },
            ],
            problem_latex=r"x ^ { 2 } + 4 x - 5 = 0",
            previous_latex=[],
        )

        self.assertEqual(
            scored[0].latex,
            r"x = \frac { - 4 + \sqrt { 4 ^ { 2 } - 4 ( 1 ) ( - 5 ) } } { 2 ( 1 ) }",
        )
        self.assertTrue(scored[0].sound)
        self.assertEqual(scored[0].detail["repair"], "contextual_quadratic_formula_coefficient")

    def test_repairs_badly_malformed_quadratic_formula_from_problem_coefficients(self):
        scored = score_ocr_predictions(
            [
                {
                    "latex": r"x = \frac { 1 + \sqrt { 4 ^ { 2 } - 1 ( 1 ) ( - 5 } } { 2 ( 1 ) } - 1",
                    "score": 0,
                },
            ],
            problem_latex=r"x ^ { 2 } + 4 x - 5 = 0",
            previous_latex=[],
        )

        self.assertEqual(
            "".join(scored[0].latex.split()),
            "".join(r"x = \frac { - 4 + \sqrt { 4 ^ { 2 } - 4 ( 1 ) ( - 5 ) } } { 2 ( 1 ) }".split()),
        )
        self.assertTrue(scored[0].sound)
        self.assertTrue(scored[0].detail["solutionSupportedByProblem"])
        self.assertEqual(scored[0].detail["repair"], "contextual_quadratic_formula_coefficient")

    def test_scores_logarithmic_top_five_against_previous_line(self):
        predictions = [
            {"latex": r"\log _ { 2 } ( x ) = ", "score": 1.0},
            {"latex": r"2 ^ { 4 } = x", "score": -2.0},
            {"latex": r"q = 4", "score": -0.5},
        ]
        scored = score_ocr_predictions(
            predictions,
            problem_latex=r"\log _ { 2 } ( x ) + 3 = 7",
            previous_latex=[r"\log _ { 2 } ( x ) = 4"],
        )
        self.assertEqual(scored[0].latex, r"2 ^ { 4 } = x")
        self.assertTrue(scored[0].equivalent_to_previous)

    def test_previous_line_duplicate_does_not_override_sound_new_solution(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"x = 3", "score": -0.2},
                {"latex": r"x = 2", "score": -0.3},
            ],
            problem_latex=r"x ^ { 2 } - 5 x + 6 = 0",
            previous_latex=[
                r"( x - 2 ) ( x - 3 ) = 0",
                r"x = 2",
            ],
        )

        self.assertEqual(scored[0].latex, r"x = 3")
        duplicate = next(item for item in scored if item.latex == r"x = 2")
        self.assertTrue(duplicate.detail["duplicatePreviousLatex"])

    def test_repairs_contextual_uppercase_variable_and_zero_lookalike(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"( X - 2 ) ( x - 3 ) = s", "score": -0.4},
            ],
            problem_latex=r"x ^ { 2 } - 5 x + 6 = 0",
            previous_latex=[r"x ^ { 2 } - 5 x + 6 = 0"],
        )

        self.assertEqual(scored[0].latex, r"( x - 2 ) ( x - 3 ) = 0")
        self.assertTrue(scored[0].sound)
        self.assertTrue(scored[0].equivalent_to_problem)

    def test_repairs_adjacent_fraction_sum_symbol_lookalikes(self):
        problem = r"\frac { x } { 2 } + \frac { 1 } { 3 } = 2"
        for latex in (
            r"\frac { \pi } { 2 } t \frac { 1 } { 3 } = 2",
            r"\frac { \pi } { 2 } + t \frac { 1 } { 3 } = 2",
        ):
            with self.subTest(latex=latex):
                scored = score_ocr_predictions(
                    [{"latex": latex, "score": -0.2}],
                    problem_latex=problem,
                    previous_latex=[],
                )

                self.assertEqual(scored[0].latex, problem)
                self.assertTrue(scored[0].equivalent_to_problem)
                self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_simplifies_parenthesized_unit_products_in_recovered_line(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"5 x + 2 = 1 5", "score": -0.2},
            ],
            problem_latex=r"\frac { x } { 2 } + \frac { 1 } { 3 } = 2",
            previous_latex=[
                r"\frac { x } { 2 } + \frac { 1 } { 3 } = 2",
                r"\times 6 \times 6",
            ],
        )

        self.assertEqual(scored[0].latex, r"3 x + 2 = 12")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_denominator_clear")

    def test_equivalence_returns_unknown_for_over_budget_expression(self):
        noisy_candidate = "x = " + " + ".join(f"a{i}" for i in range(160))

        started = time.monotonic()
        result = check_equivalence(r"x = 1", noisy_candidate)
        elapsed = time.monotonic() - started

        self.assertIsNone(result.equivalent)
        self.assertIn("budget", result.method)
        self.assertLess(elapsed, 1.0)

    def test_scores_operation_annotation_above_stray_symbol_lookalike(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"\times 6 \times 6", "score": -1.07},
                {"latex": r"x 6 \times 6", "score": -1.12},
            ],
            problem_latex=r"\frac { x } { 2 } + \frac { 1 } { 3 } = 2",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"\times 6 \times 6")
        self.assertTrue(scored[0].detail["operationAnnotation"])

    def test_repairs_trailing_one_that_balances_parentheses(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"= ( 8 + 2 ) - ( 0 + 0 1", "score": -0.66},
            ],
            problem_latex=r"\int _ { 0 } ^ { 2 } ( 3 x ^ { 2 } + 1 ) d x",
            previous_latex=[r"= [ x ^ { 3 } + x ] _ { 0 } ^ { 2 }"],
        )

        self.assertEqual(scored[0].latex, r"= ( 8 + 2 ) - ( 0 + 0 )")
        self.assertTrue(scored[0].sound)
        self.assertEqual(scored[0].detail["repair"], "trailing_one_to_parenthesis")

    def test_repairs_malformed_arithmetic_continuation_when_equivalent_to_previous(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"- 1 2 + 2 ) - ( o + n", "score": -0.55},
                {"latex": r"- 1 2 + 2 ) - ( 0 + n 1", "score": -0.4},
                {"latex": r"- ( 2 + 2 ) - ( o + n )", "score": -0.71},
            ],
            problem_latex=r"\int _ { 0 } ^ { 2 } ( x + 1 ) d x",
            previous_latex=[r"= [ \frac { x ^ { 2 } } { 2 } + x ] _ { 0 } ^ { 2 }"],
        )

        self.assertEqual(scored[0].latex, r"= ( 2 + 2 ) - ( 0 + 0 )")
        self.assertTrue(scored[0].sound)
        self.assertTrue(scored[0].equivalent_to_previous)
        self.assertEqual(scored[0].detail["repair"], "malformed_arithmetic_continuation")
        self.assertNotIn(r"0 + 01", [item.latex for item in scored])

    def test_repairs_second_equals_as_minus_in_arithmetic_continuation(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"= ( 2 + 2 ) = ( o + o )", "score": 0},
            ],
            problem_latex=r"\int _ { 0 } ^ { 2 } ( x + 1 ) d x",
            previous_latex=[
                r"= [ \frac { x ^ { 2 } } { 2 } + x ] _ { 0 } ^ { 2 }",
            ],
        )

        self.assertEqual(scored[0].latex, r"= ( 2 + 2 ) - ( 0 + 0 )")
        self.assertTrue(scored[0].equivalent_to_previous)
        self.assertEqual(scored[0].detail["repair"], "malformed_arithmetic_continuation")

    def test_repairs_adjacent_parenthesized_arithmetic_as_subtraction_continuation(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"= ( 8 + 2 ) ( o + o )", "score": 0},
            ],
            problem_latex=r"\int _ { 0 } ^ { 2 } ( 3 x ^ { 2 } + 1 ) d x",
            previous_latex=[
                r"= [ x ^ { 3 } + x ] _ { 0 } ^ { 2 }",
            ],
        )

        self.assertEqual(scored[0].latex, r"= ( 8 + 2 ) - ( 0 + 0 )")
        self.assertTrue(scored[0].equivalent_to_previous)
        self.assertEqual(scored[0].detail["repair"], "malformed_arithmetic_continuation")

    def test_malformed_arithmetic_repair_does_not_rewrite_variable_algebra(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"- x + 2", "score": -0.2},
            ],
            problem_latex=r"x + 2 = 5",
            previous_latex=[],
        )

        self.assertNotEqual(scored[0].latex, r"= ( x + 2 )")
        self.assertNotIn("repair", scored[0].detail)

    def test_repairs_symbolic_final_value_from_previous_numeric_continuation(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"= q", "score": -2.02},
                {"latex": r"c = q", "score": -2.22},
            ],
            problem_latex=r"\int _ { 0 } ^ { 2 } ( x + 1 ) d x",
            previous_latex=[r"= ( 2 + 2 ) - ( 0 + 0 )"],
        )

        self.assertEqual(scored[0].latex, r"= 4")
        self.assertTrue(scored[0].equivalent_to_previous)
        self.assertEqual(scored[0].detail["repair"], "symbolic_numeric_continuation")

    def test_repairs_stray_variable_numeric_continuation_from_previous_value(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"m = 2 \cdot 0", "score": -1.78},
                {"latex": r"= 2 . 0 0", "score": -2.22},
            ],
            problem_latex=r"\int _ { 0 } ^ { 2 } ( 3 x ^ { 2 } + 1 ) d x",
            previous_latex=[r"= ( 8 + 2 ) - ( 0 + 0 )"],
        )

        self.assertEqual(scored[0].latex, r"= 10")
        self.assertTrue(scored[0].equivalent_to_previous)
        self.assertEqual(scored[0].detail["repair"], "stray_variable_numeric_continuation")

    def test_stray_variable_numeric_repair_preserves_context_variables(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"x = 0", "score": -0.1},
            ],
            problem_latex=r"x + 1 = 1",
            previous_latex=[r"= 10"],
        )

        self.assertEqual(scored[0].latex, r"x = 0")
        self.assertNotEqual(scored[0].detail.get("repair"), "stray_variable_numeric_continuation")

    def test_symbolic_final_value_repair_requires_numeric_previous_context(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"= q", "score": -2.02},
            ],
            problem_latex=r"x + 2 = 5",
            previous_latex=[r"x + 2"],
        )

        self.assertNotEqual(scored[0].latex, r"= 4")
        self.assertNotIn("repair", scored[0].detail)

    def test_scores_function_evaluation_against_previous_symbolic_line(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"f ^ { \prime } ( 2 ) = \frac { 3 } { 1 }", "score": -0.22},
                {"latex": r"f ^ { \prime } ( 2 ) = \frac { 3 } { 4 }", "score": -0.28},
                {"latex": r"f ^ { 1 } ( 2 ) = \frac { 3 } { 1 }", "score": -0.35},
            ],
            problem_latex=r"f ( x ) = \frac { x ^ { 2 } + 1 } { x }",
            previous_latex=[r"f ^ { \prime } ( x ) = \frac { x ^ { 2 } - 1 } { x ^ { 2 } }"],
        )

        self.assertEqual(scored[0].latex, r"f ^ { \prime } ( 2 ) = \frac { 3 } { 4 }")
        self.assertTrue(scored[0].equivalent_to_previous)
        self.assertEqual(scored[0].detail["previousEquivalence"], "function_evaluation_substitution")

    def test_scores_product_rule_evaluation_against_previous_symbolic_line(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"f ^ { \prime } ( 2 ) = h ( 5 ) + 4", "score": -0.2},
                {"latex": r"f ^ { \prime } ( 2 ) = 4 ( 5 ) + 4", "score": -0.4},
            ],
            problem_latex=r"f ( x ) = x ^ { 2 } ( x + 3 )",
            previous_latex=[r"f ^ { \prime } ( x ) = 2 x ( x + 3 ) + x ^ { 2 }"],
        )

        self.assertEqual(scored[0].latex, r"f ^ { \prime } ( 2 ) = 4 ( 5 ) + 4")
        self.assertTrue(scored[0].equivalent_to_previous)

    def test_scores_derivative_line_against_function_definition(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"f ( x ) = 2 x ( x + 3 ) + x ^ { 2 }", "score": -0.01},
                {"latex": r"f ^ { \prime } ( x ) = 2 x ( x + 3 ) + x ^ { 2 }", "score": -0.44},
            ],
            problem_latex=r"f ( x ) = x ^ { 2 } ( x + 3 )",
            previous_latex=[r"f ( x ) = x ^ { 2 } ( x + 3 )"],
        )

        self.assertEqual(scored[0].latex, r"f ^ { \prime } ( x ) = 2 x ( x + 3 ) + x ^ { 2 }")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertTrue(scored[0].equivalent_to_previous)

    def test_repairs_contextual_function_and_power_lookalikes(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"v ( x ) = 2 x ( x + 3 ) + n 2", "score": -0.2},
            ],
            problem_latex=r"f ( x ) = x ^ { 2 } ( x + 3 )",
            previous_latex=[r"f ( x ) = x ^ { 2 } ( x + 3 )"],
        )

        self.assertEqual(scored[0].latex, r"f ^ { \prime } ( x ) = 2 x ( x + 3 ) + x ^ { 2 }")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_repairs_stray_noncontext_letter_before_numeric_coefficient(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"f ( x ) = x ^ { 3 } + t 2 x ^ { 2 }", "score": 0},
            ],
            problem_latex=r"f ( x ) = x ^ { 3 } + 2 x ^ { 2 }",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"f ( x ) = x ^ { 3 } + 2 x ^ { 2 }")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_repairs_contextual_function_header_for_evaluation_line(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"i ( 2 ) = 4 ( 5 ) + 1", "score": -0.2},
                {"latex": r"r ( 2 ) = 4 ( 5 ) + 4", "score": -0.8},
            ],
            problem_latex=r"f ( x ) = x ^ { 2 } ( x + 3 )",
            previous_latex=[
                r"f ( x ) = x ^ { 2 } ( x + 3 )",
                r"f ^ { \prime } ( x ) = 2 x ( x + 3 ) + x ^ { 2 }",
            ],
        )

        self.assertEqual(scored[0].latex, r"f ^ { \prime } ( 2 ) = 4 ( 5 ) + 4")
        self.assertTrue(scored[0].equivalent_to_previous)
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_contextual_repairs_are_deduplicated_before_scoring(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"v ( x ) = 2 x ( x + 3 ) + n 2", "score": -0.2},
                {"latex": r"/ ( x ) = 2 x ( x + 3 ) + n 2", "score": -0.4},
                {"latex": r"v ( x ) = 2 x ( x + 3 ) + m 2", "score": -0.5},
            ],
            problem_latex=r"f ( x ) = x ^ { 2 } ( x + 3 )",
            previous_latex=[r"f ( x ) = x ^ { 2 } ( x + 3 )"],
        )

        repaired = r"f ^ { \prime } ( x ) = 2 x ( x + 3 ) + x ^ { 2 }"
        repaired_scores = [item for item in scored if item.latex == repaired]

        self.assertEqual(len(repaired_scores), 1)
        self.assertEqual(repaired_scores[0].detail["modelScore"], -1.25)
        self.assertEqual(repaired_scores[0].detail["repairedFrom"], r"v ( x ) = 2 x ( x + 3 ) + n 2")

    def test_contextual_symbol_repair_requires_semantic_support(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"q ( x ) = x + 1", "score": 1.0},
            ],
            problem_latex=r"f ( x ) = x ^ { 2 }",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"q ( x ) = x + 1")
        self.assertNotIn("repair", scored[0].detail)

    def test_repairs_parenthesis_letter_and_context_number_when_equivalent(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"x - 2 = 5 ( x + 1 n", "score": -0.3},
            ],
            problem_latex=r"\frac { x - 1 } { x + 1 } = 5",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"x - 1 = 5 ( x + 1 )")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_parenthesis_number_repair_requires_equivalence_support(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"x - 2 = 5 ( x + 1 n", "score": 1.0},
            ],
            problem_latex=r"z + 1 = 6",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"x - 2 = 5 ( x + 1 n")
        self.assertNotIn("repair", scored[0].detail)

    def test_repairs_extra_spaced_digit_when_equivalent_to_problem(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"4 ( 2 x + 1 ) - 3 ( x - 2 ) = 6 0 0", "score": -0.2},
            ],
            problem_latex=r"\frac { 2 x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"4 ( 2 x + 1 ) - 3 ( x - 2 ) = 6 0")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_repairs_trig_n_number_lookalike_from_previous_context(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"8 x + 4 - 3 x + 6 = \cos n", "score": -0.2},
            ],
            problem_latex=r"\frac { 2 x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
            previous_latex=[r"4 ( 2 x + 1 ) - 3 ( x - 2 ) = 6 0"],
        )

        self.assertEqual(scored[0].latex, r"8 x + 4 - 3 x + 6 = 60")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertTrue(scored[0].equivalent_to_previous)
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_repairs_c_n_number_lookalike_from_previous_context(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"8 x + 4 - 3 x + 6 = c n", "score": -0.2},
            ],
            problem_latex=r"\frac { 2 x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
            previous_latex=[r"4 ( 2 x + 1 ) - 3 ( x - 2 ) = 6 0"],
        )

        self.assertEqual(scored[0].latex, r"8 x + 4 - 3 x + 6 = 60")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertTrue(scored[0].equivalent_to_previous)
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_repairs_ellipsis_row_from_previous_distributive_step(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"\cdots + 4 - 3 X + 6 = w", "score": -0.2},
            ],
            problem_latex=r"\frac { 2 x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
            previous_latex=[r"4 ( 2 x + 1 ) - 3 ( x - 2 ) = 6 0"],
        )

        self.assertEqual(scored[0].latex, r"8 x + 4 - 3 x + 6 = 60")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertTrue(scored[0].equivalent_to_previous)
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_denominator_clear_repair_does_not_repeat_previous_context_line(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"\cdots + 4 - 3 x + 6 = w", "score": -0.2},
            ],
            problem_latex=r"\frac { 2 x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
            previous_latex=[
                r"\times 12 \times 12",
                r"4 ( 2 x + 1 ) - 3 ( x - 2 ) = 6 0",
            ],
        )

        self.assertEqual(scored[0].latex, r"8 x + 4 - 3 x + 6 = 60")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertTrue(scored[0].equivalent_to_previous)
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_repairs_linear_simplification_from_previous_context(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"3 x + 1 0 = \cos n", "score": -0.2},
            ],
            problem_latex=r"\frac { 2 x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
            previous_latex=[r"8 x + 4 - 3 x + 6 = 60"],
        )

        self.assertEqual(scored[0].latex, r"5 x + 10 = 60")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertTrue(scored[0].equivalent_to_previous)
        self.assertEqual(scored[0].detail["repair"], "contextual_linear_simplification")

    def test_linear_simplification_does_not_revert_to_fractional_problem_context(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"8 X + 4 - 3 x + 6 = 6 0", "score": -0.2},
            ],
            problem_latex=r"\frac { 2 x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
            previous_latex=[
                r"\frac { 2 x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
                r"\times 12 \times 12",
                r"4 ( 2 x + 1 ) - 3 ( x - 2 ) = 60",
            ],
        )

        latexes = [item.latex for item in scored[:3]]
        self.assertEqual(scored[0].latex, r"8 x + 4 - 3 x + 6 = 60")
        self.assertNotIn(r"\frac { 5 } { 12 } x + \frac { 5 } { 6 } = 5", latexes)

    def test_repairs_ellipsis_row_from_contextual_denominator_clear_step(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"\cdots + 1 ) - 3 ( x - \ldots", "score": -0.2},
            ],
            problem_latex=r"\frac { 2 x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
            previous_latex=[r"\times 1 2 \times 1 2"],
        )

        self.assertEqual(scored[0].latex, r"4 ( 2 x + 1 ) - 3 ( x - 2 ) = 60")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_denominator_clear")

    def test_repairs_two_sided_fraction_denominator_clear_step(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"3 ( x + 1 1 = 1 0", "score": -0.2},
            ],
            problem_latex=r"\frac { x + 1 } { 2 } = \frac { 5 } { 3 }",
            previous_latex=[r"\times 6 \times 6"],
        )

        self.assertEqual(scored[0].latex, r"3 ( x + 1 ) = 10")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_denominator_clear")

    def test_repairs_spurious_exponent_on_final_answer_variable(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"x ^ { 2 } = \frac { 7 } { 3 }", "score": -0.2},
            ],
            problem_latex=r"\frac { x + 1 } { 2 } = \frac { 5 } { 3 }",
            previous_latex=[
                r"\times 6 \times 6",
                r"3 ( x + 1 ) = 10",
                r"3 x + 3 = 10",
                r"- 3 - 3",
                r"3 x = 7",
                r"/ 3 / 3",
            ],
        )

        self.assertEqual(scored[0].latex, r"x = \frac { 7 } { 3 }")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertTrue(scored[0].equivalent_to_previous)
        self.assertEqual(scored[0].detail["repair"], "contextual_exponent_confusion")

    def test_final_assignment_does_not_revert_to_denominator_cleared_context(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"x ^ { 2 } = \frac { 7 } { 3 }", "score": -0.2},
            ],
            problem_latex=r"\frac { x + 1 } { 2 } = \frac { 5 } { 3 }",
            previous_latex=[
                r"\times 6 \times 6",
                r"3 ( x + 1 ) = 10",
                r"3 x + 3 = 10",
                r"- 3 - 3",
                r"3 x = 7",
                r"/ 3 / 3",
            ],
        )

        latexes = [item.latex for item in scored[:3]]
        self.assertEqual(scored[0].latex, r"x = \frac { 7 } { 3 }")
        self.assertNotIn(r"3 ( x + 1 ) = 10", latexes)

    def test_denominator_clear_repair_requires_contextual_multiplier(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"\cdots + 1 ) - 3 ( x - \ldots", "score": -0.2},
            ],
            problem_latex=r"\frac { 2 x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
            previous_latex=[],
        )

        self.assertNotEqual(scored[0].latex, r"4 ( 2 x + 1 ) - 3 ( x - 2 ) = 60")
        self.assertNotIn("repair", scored[0].detail)

    def test_repairs_missing_derivative_prime_when_rhs_matches_problem_derivative(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"f ( x ) = \frac { x ^ { 2 } - 1 } { x ^ { 2 } }", "score": -0.1},
            ],
            problem_latex=r"f ( x ) = \frac { x ^ { 2 } + 1 } { x }",
            previous_latex=[r"f ( x ) = \frac { x ^ { 2 } + 1 } { x }"],
        )

        self.assertEqual(scored[0].latex, r"f ^ { \prime } ( x ) = \frac { x ^ { 2 } - 1 } { x ^ { 2 } }")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_repairs_missing_derivative_prime_and_contextual_rhs_variable(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"f ( x ) = \frac { x ^ { 2 } - 1 } { n ^ { 2 } }", "score": -0.1},
            ],
            problem_latex=r"f ( x ) = \frac { x ^ { 2 } + 1 } { x }",
            previous_latex=[r"f ( x ) = \frac { x ^ { 2 } + 1 } { x }"],
        )

        self.assertEqual(scored[0].latex, r"f ^ { \prime } ( x ) = \frac { x ^ { 2 } - 1 } { x ^ { 2 } }")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_repairs_redundant_trailing_brace_in_derivative_fraction(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"f ^ { \prime } ( x ) = \frac { x ^ { 2 } - 1 } { x ^ { 2 } } }", "score": -0.1},
            ],
            problem_latex=r"f ( x ) = \frac { x ^ { 2 } + 1 } { x }",
            previous_latex=[r"f ( x ) = \frac { x ^ { 2 } + 1 } { x }"],
        )

        self.assertEqual(scored[0].latex, r"f ^ { \prime } ( x ) = \frac { x ^ { 2 } - 1 } { x ^ { 2 } }")
        self.assertTrue(scored[0].sound)
        self.assertEqual(scored[0].detail["repair"], "redundant_trailing_brace")

    def test_repairs_bad_derivative_header_when_candidate_denominator_is_contextual(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"f ^ { f } ( 2 ) = \frac { 3 } { 1 }", "score": -0.3},
                {"latex": r"f ^ { f } ( 2 ) = \frac { 3 } { 4 }", "score": -0.5},
            ],
            problem_latex=r"f ( x ) = \frac { x ^ { 2 } + 1 } { x }",
            previous_latex=[r"f ^ { \prime } ( x ) = \frac { x ^ { 2 } - 1 } { x ^ { 2 } }"],
        )

        self.assertEqual(scored[0].latex, r"f ^ { \prime } ( 2 ) = \frac { 3 } { 4 }")
        self.assertTrue(scored[0].equivalent_to_previous)
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_repairs_derivative_substitution_step_from_previous_derivative(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"f ^ { l } ( 2 ) = h ( 5 ) + 4", "score": -0.2},
            ],
            problem_latex=r"f ( x ) = x ^ { 2 } ( x + 3 )",
            previous_latex=[r"f ^ { \prime } ( x ) = 2 x ( x + 3 ) + x ^ { 2 }"],
        )

        self.assertEqual(scored[0].latex, r"f ^ { \prime } ( 2 ) = 4 ( 5 ) + 4")
        self.assertTrue(scored[0].equivalent_to_previous)
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_derivative_substitution_prime_repair_beats_numeric_rewrite(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"f ( 2 ) = 3 ( 2 ) ^ { 2 } + 4 ( 2 )", "score": 0},
            ],
            problem_latex=r"f ( x ) = x ^ { 3 } + 2 x ^ { 2 }",
            previous_latex=[r"f ^ { \prime } ( x ) = 3 x ^ { 2 } + 4 x"],
        )

        self.assertEqual(scored[0].latex, r"f ^ { \prime } ( 2 ) = 3 ( 2 ) ^ { 2 } + 4 ( 2 )")
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")
        self.assertNotEqual(scored[0].latex, r"f ( 2 ) = 2 ( 2 ) ^ { 2 } + 4 ( 2 )")

    def test_low_overlap_complex_continuation_does_not_win_by_collapsing_to_problem_value(self):
        scored = score_ocr_predictions(
            [
                {
                    "latex": r"= ( \frac { 2 } { \frac { 2 } { 2 } } + v _ { 0 } ^ { 2 } } ^ { 2 }",
                    "score": -1.4,
                    "elapsedSeconds": 26.2,
                },
                {
                    "latex": r"= 1 \frac { x ^ { 2 } } { 2 } + x ] _ { 0 } ^ { 2 }",
                    "score": 0,
                    "elapsedSeconds": 26.2,
                },
            ],
            problem_latex=r"\int _ { 0 } ^ { 2 } ( x + 1 ) d x",
            previous_latex=[
                r"\int _ { 0 } ^ { 2 } ( x + 1 ) d x",
            ],
        )

        self.assertEqual(scored[0].latex, r"= [ \frac { x ^ { 2 } } { 2 } + x ] _ { 0 } ^ { 2 }")
        self.assertEqual(scored[0].detail["repair"], "contextual_bound_evaluation_bracket")
        malformed = r"= ( \frac { 2 } { \frac { 2 } { 2 } } + v _ { 0 } ^ { 2 } } ^ { 2 }"
        demoted = next(
            item
            for item in scored
            if item.latex == malformed
        )
        self.assertTrue(demoted.detail["lowVisualSupportForProblemValue"])

    def test_low_overlap_final_value_keeps_previous_equivalence_credit(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"= 4", "score": -1.4, "elapsedSeconds": 3.3},
                {
                    "latex": r"= ( \frac { 2 } { \frac { 2 } { 2 } } + v _ { 0 } ^ { 2 } } ^ { 2 }",
                    "score": -1.4,
                    "elapsedSeconds": 26.2,
                },
            ],
            problem_latex=r"\int _ { 0 } ^ { 2 } ( x + 1 ) d x",
            previous_latex=[
                r"\int _ { 0 } ^ { 2 } ( x + 1 ) d x",
                r"= ( 2 + 2 ) - ( 0 + 0 )",
            ],
        )

        self.assertEqual(scored[0].latex, r"= 4")
        self.assertTrue(scored[0].equivalent_to_previous)
        self.assertNotIn("lowVisualSupportForProblemValue", scored[0].detail)

    def test_bound_evaluation_bracket_repair_requires_integral_context(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"= 1 x + 1 ] _ { 0 } ^ { 2 }", "score": 0},
            ],
            problem_latex=r"x + 1 = 3",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"= 1 x + 1 ] _ { 0 } ^ { 2 }")
        self.assertNotEqual(scored[0].detail.get("repair"), "contextual_bound_evaluation_bracket")

    def test_repairs_malformed_quotient_derivative_equation_from_problem(self):
        scored = score_ocr_predictions(
            [
                {
                    "latex": r"f ^ { \prime } ( x ) \frac { - x ^ { 2 } - 1 } { - x ^ { 2 } }",
                    "score": -0.2,
                },
            ],
            problem_latex=r"f ( x ) = \frac { x ^ { 2 } + 1 } { x }",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"f ^ { \prime } ( x ) = \frac { x ^ { 2 } - 1 } { x ^ { 2 } }")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_repairs_quotient_derivative_prime_and_numeric_denominator_base(self):
        scored = score_ocr_predictions(
            [
                {
                    "latex": r"f ( x ) = \frac { x ^ { 2 } - 1 } { 2 ^ { 2 } }",
                    "score": 0,
                },
            ],
            problem_latex=r"f ( x ) = \frac { x ^ { 2 } + 1 } { x }",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"f ^ { \prime } ( x ) = \frac { x ^ { 2 } - 1 } { x ^ { 2 } }")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_repairs_accidental_exponent_as_equals_in_solve_final_line(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"x ^ { n } = \frac { 1 0 } { i 3 }", "score": -0.3},
            ],
            problem_latex=r"\frac { x } { 2 } + \frac { 1 } { 3 } = 2",
            previous_latex=[r"3 x = 10"],
        )

        self.assertEqual(scored[0].latex, r"x = \frac { 1 0 } { 3 }")
        self.assertTrue(scored[0].equivalent_to_previous)
        self.assertEqual(scored[0].detail["repair"], "contextual_exponent_confusion")

    def test_repairs_stray_noncontext_letter_before_fraction_with_semantic_support(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"\frac { x } { 2 } + t \frac { 1 } { 3 } = 2", "score": -0.3},
            ],
            problem_latex=r"\frac { x } { 2 } + \frac { 1 } { 3 } = 2",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"\frac { x } { 2 } + \frac { 1 } { 3 } = 2")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_repairs_stray_noncontext_letter_between_adjacent_fractions_as_plus(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"\frac { x } { 2 } t \frac { 1 } { 3 } = 2", "score": -0.3},
            ],
            problem_latex=r"\frac { x } { 2 } + \frac { 1 } { 3 } = 2",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"\frac { x } { 2 } + \frac { 1 } { 3 } = 2")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_repairs_log_numeric_argument_from_problem_context(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"\log _ { 2 } ( x ) + \log _ { 2 } ( 1 ) = 5", "score": -0.2},
            ],
            problem_latex=r"\log _ { 2 } ( x ) + \log _ { 2 } ( 4 ) = 5",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"\log _ { 2 } ( x ) + \log _ { 2 } ( 4 ) = 5")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_log_numeric_argument")

    def test_repairs_latex_rhs_number_from_log_problem_context(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"\log _ { 2 } ( x ) + \log _ { 2 } ( 4 ) = 0", "score": -0.2},
            ],
            problem_latex=r"\log _ { 2 } ( x ) + \log _ { 2 } ( 4 ) = 5",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"\log _ { 2 } ( x ) + \log _ { 2 } ( 4 ) = 5")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_latex_numeric_equivalence")

    def test_repairs_malformed_log_base_from_problem_context(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"\log _ { 2 } ( x ) + \log _ { 0 } 2 } ( 4 ) = 5", "score": -0.2},
            ],
            problem_latex=r"\log _ { 2 } ( x ) + \log _ { 2 } ( 4 ) = 5",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"\log _ { 2 } ( x ) + \log _ { 2 } ( 4 ) = 5")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_malformed_log_base")

    def test_repairs_malformed_copied_log_equation_from_problem_context(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"\tan v _ { 0 } 3 ( x + x = n", "score": 0},
                {"latex": r"v _ { 5 } ( x + 1 ) = 1", "score": -0.4},
            ],
            problem_latex=r"\log _ { 3 } ( x + 1 ) = 2",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"\log _ { 3 } ( x + 1 ) = 2")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_malformed_log_equation")

    def test_repairs_parseable_copied_log_equation_symbol_confusions(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"\log _ { 2 } ( x ) + x = q", "score": 0},
            ],
            problem_latex=r"\log _ { 2 } ( x ) + 3 = 7",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"\log _ { 2 } ( x ) + 3 = 7")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_malformed_log_equation")

    def test_repairs_copied_problem_log_numeric_term_confusion(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"\log _ { 2 } ( x ) + x = 7", "score": 0},
            ],
            problem_latex=r"\log _ { 2 } ( x ) + 3 = 7",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"\log _ { 2 } ( x ) + 3 = 7")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_malformed_log_equation")

    def test_repairs_copied_problem_fraction_operator_and_denominator_confusions(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"\frac { 2 x + 1 } { 3 } \infty \frac { x - 2 } { 1 } = 5", "score": 0},
            ],
            problem_latex=r"\frac { 2 x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"\frac { 2 x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_copied_problem_equation")

    def test_repairs_copied_problem_quadratic_symbol_and_operator_confusions(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"x ^ { 2 } - 5 x x = o", "score": 0},
            ],
            problem_latex=r"x ^ { 2 } - 5 x + 6 = 0",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"x ^ { 2 } - 5 x + 6 = 0")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_copied_problem_equation")

    def test_copied_problem_repair_does_not_override_later_work(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"\log _ { 2 } ( x ) + x = 7", "score": 0},
            ],
            problem_latex=r"\log _ { 2 } ( x ) + 3 = 7",
            previous_latex=[r"- 3 - 3"],
        )

        self.assertNotEqual(scored[0].latex, r"\log _ { 2 } ( x ) + 3 = 7")
        self.assertNotEqual(scored[0].detail.get("repair"), "contextual_copied_problem_equation")

    def test_stray_fraction_letter_repair_preserves_context_variable(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"x + t \frac { 1 } { 3 } = 2", "score": -0.3},
            ],
            problem_latex=r"x + t \frac { 1 } { 3 } = 2",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"x + t \frac { 1 } { 3 } = 2")
        self.assertNotIn("repair", scored[0].detail)

    def test_adjacent_fraction_letter_repair_preserves_context_variable(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"\frac { x } { 2 } t \frac { 1 } { 3 } = 2", "score": -0.3},
            ],
            problem_latex=r"\frac { x } { 2 } t \frac { 1 } { 3 } = 2",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"\frac { x } { 2 } t \frac { 1 } { 3 } = 2")
        self.assertNotIn("repair", scored[0].detail)

    def test_repairs_context_variable_fraction_denominator_with_equivalence_support(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"f ( x ) = \frac { z ^ { 2 } + 1 } { 2 }", "score": -0.3},
            ],
            problem_latex=r"f ( x ) = \frac { x ^ { 2 } + 1 } { x }",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"f ( x ) = \frac { x ^ { 2 } + 1 } { x }")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_repairs_spurious_numeric_subscript_on_context_variable(self):
        scored = score_ocr_predictions(
            [
                {
                    "latex": r"f ^ { \prime } ( x ) = \frac { x ^ { 2 } - 1 } { x _ { 0 } ^ { 2 } }",
                    "score": -0.3,
                },
            ],
            problem_latex=r"f ( x ) = \frac { x ^ { 2 } + 1 } { x }",
            previous_latex=[],
        )

        self.assertEqual(
            scored[0].latex,
            r"f ^ { \prime } ( x ) = \frac { x ^ { 2 } - 1 } { x ^ { 2 } }",
        )
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_character_overlap_uses_problem_and_previous_context(self):
        overlap = character_overlap(r"5 z + 2 = 17", r"z = 3", candidate=r"5 z = 15")
        self.assertGreater(overlap, 0.45)
        weak = character_overlap(r"5 z + 2 = 17", candidate=r"q = 9")
        self.assertLess(weak, overlap)

    def test_character_overlap_weights_variables_above_shared_digits(self):
        correct_variable = character_overlap(r"5 z + 2 = 17", r"z = 3", candidate=r"5 z = 15")
        wrong_variable = character_overlap(r"5 z + 2 = 17", r"z = 3", candidate=r"5 q = 15")

        self.assertGreater(correct_variable, wrong_variable + 0.3)
        self.assertGreater(math_symbol_weights(r"z")["z"], math_symbol_weights(r"5")["5"])

    def test_previous_greek_symbol_context_beats_latin_lookalike(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"z + 1 = 6", "score": 1.0},
                {"latex": r"\eta + 1 = 6", "score": 0.6},
            ],
            problem_latex=r"u = 5",
            previous_latex=[r"\eta = 5"],
        )

        self.assertEqual(scored[0].latex, r"\eta + 1 = 6")

    def test_repairs_latin_eta_lookalike_from_problem_context(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"n + 1 = 6", "score": -0.1},
                {"latex": r"n = 5", "score": -0.2},
            ],
            problem_latex=r"\eta + 1 = 6",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"\eta + 1 = 6")
        self.assertEqual(scored[0].detail["repair"], "contextual_greek_variable")

    def test_greek_variable_repair_requires_greek_context(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"n + 1 = 6", "score": -0.1},
            ],
            problem_latex=r"x + 1 = 6",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"n + 1 = 6")
        self.assertNotIn("repair", scored[0].detail)

    def test_repairs_contextual_numeric_lookalikes_with_semantic_support(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"\infty x + 3 = n", "score": -1.0},
            ],
            problem_latex=r"2 x + 3 = 11",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"2 x + 3 = 11")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_numeric_lookalike")

    def test_contextual_numeric_repair_preserves_real_context_variable(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"n + 1 = 12", "score": -0.1},
            ],
            problem_latex=r"n + 1 = 12",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"n + 1 = 12")
        self.assertNotIn("repair", scored[0].detail)

    def test_repairs_n_as_zero_only_with_semantic_support(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"( X - 2 ) ( x - 3 ) = n", "score": 0},
            ],
            problem_latex=r"x ^ { 2 } - 5 x + 6 = 0",
            previous_latex=[r"x ^ { 2 } - 5 x + 6 = 0"],
        )

        self.assertEqual(scored[0].latex, r"( x - 2 ) ( x - 3 ) = 0")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertIn(scored[0].detail["repair"], {
            "contextual_numeric_lookalike",
            "contextual_symbol_confusion",
        })

    def test_repairs_alpha_as_contextual_two_only_without_alpha_context(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"\alpha x + 3 = 1 1", "score": -1.0},
            ],
            problem_latex=r"2 x + 3 = 11",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"2 x + 3 = 1 1")
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_numeric_lookalike")

    def test_contextual_numeric_repair_preserves_real_alpha_context(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"\alpha + 1 = 3", "score": -0.1},
            ],
            problem_latex=r"\alpha + 1 = 3",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"\alpha + 1 = 3")
        self.assertNotIn("repair", scored[0].detail)

    def test_repairs_split_c_alpha_as_contextual_two_x_with_semantic_support(self):
        scored = score_ocr_predictions(
            [
                {
                    "latex": r"\frac { c \alpha + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
                    "score": -0.8,
                },
            ],
            problem_latex=r"\frac { 2 x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
            previous_latex=[],
        )

        self.assertEqual(
            scored[0].latex,
            r"\frac { 2 x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
        )
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_numeric_lookalike")

    def test_repairs_c_before_context_variable_as_two_with_semantic_support(self):
        scored = score_ocr_predictions(
            [
                {
                    "latex": r"\frac { c x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
                    "score": -0.8,
                },
            ],
            problem_latex=r"\frac { 2 x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
            previous_latex=[],
        )

        self.assertEqual(
            scored[0].latex,
            r"\frac { 2 x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
        )
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_numeric_lookalike")

    def test_repairs_plain_alpha_as_context_variable_with_numeric_support(self):
        scored = score_ocr_predictions(
            [
                {
                    "latex": r"\frac { 2 \alpha + 1 } { 3 } - \frac { x - 2 } { 1 } = 5",
                    "score": -0.8,
                },
            ],
            problem_latex=r"\frac { 2 x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
            previous_latex=[],
        )

        self.assertEqual(
            scored[0].latex,
            r"\frac { 2 x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
        )
        self.assertTrue(scored[0].equivalent_to_problem)
        self.assertEqual(scored[0].detail["repair"], "contextual_symbol_confusion")

    def test_subtraction_repair_skips_non_finite_previous_equations(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"x = 2", "score": 0.2},
            ],
            problem_latex=r"x = 2",
            previous_latex=[
                r"x + \frac { 1 } { x - x } = 3",
                r"- 1 - 1",
            ],
        )

        self.assertEqual(scored[0].latex, r"x = 2")
        self.assertNotEqual(scored[0].detail.get("repair"), "contextual_subtraction_step")

    def test_visible_expanded_row_beats_contextual_linear_simplification(self):
        scored = score_ocr_predictions(
            [
                {
                    "latex": r"8 x + 4 3 x + 6 = 6 0",
                    "score": 0,
                    "elapsedSeconds": 8.2,
                },
            ],
            problem_latex=r"\frac { 2 x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
            previous_latex=[
                r"\frac { 2 x + 1 } { 3 } - \frac { x - 2 } { 4 } = 5",
                r"\times 12 \times 12",
                r"4 ( 2 x + 1 ) - 3 ( x - 2 ) = 60",
            ],
        )

        self.assertEqual(scored[0].latex, r"8 x + 4 - 3 x + 6 = 60")
        self.assertEqual(scored[0].detail.get("repair"), "contextual_symbol_confusion")

    def test_split_c_alpha_repair_preserves_real_c_context(self):
        scored = score_ocr_predictions(
            [
                {"latex": r"c \alpha + 1 = 3", "score": -0.8},
            ],
            problem_latex=r"2 x + c + 1 = 3",
            previous_latex=[],
        )

        self.assertEqual(scored[0].latex, r"c \alpha + 1 = 3")
        self.assertNotIn("repair", scored[0].detail)

    def test_scores_payload_candidate_groups_for_browser_pipeline(self):
        payload = {
            "problemLatex": r"2 x + 3 = 11",
            "previousLatex": [r"2 x + 3 = 11"],
            "candidateGroups": [
                {
                    "candidateId": "candidate-a",
                    "candidates": [
                        {"latex": r"2 x + = 8", "score": 1.5},
                        {"latex": r"2 x = 8", "score": -1.0},
                    ],
                },
                {
                    "candidateId": "candidate-b",
                    "candidates": [
                        {"latex": r"q = 100", "score": 2.0},
                    ],
                },
            ],
        }

        result = score_semantic_payload(payload)
        by_id = {item["candidateId"]: item for item in result["candidateScores"]}
        self.assertEqual(by_id["candidate-a"]["bestLatex"], r"2 x = 8")
        self.assertTrue(by_id["candidate-a"]["equivalentToProblem"])
        self.assertGreater(by_id["candidate-a"]["semanticScore"], by_id["candidate-b"]["semanticScore"])

    def test_group_elapsed_time_is_applied_to_candidate_scores(self):
        payload = {
            "problemLatex": r"2 x + 3 = 11",
            "candidateGroups": [
                {
                    "candidateId": "slow-candidate",
                    "elapsedSeconds": 8.0,
                    "candidates": [
                        {"latex": r"2 x = 8", "score": -1.0},
                    ],
                },
            ],
        }

        result = score_semantic_payload(payload)
        score = result["candidateScores"][0]["candidateScores"][0]
        self.assertEqual(score["detail"]["elapsedSeconds"], 8.0)
        self.assertLess(result["candidateScores"][0]["semanticScore"], 3.0)


if __name__ == "__main__":
    unittest.main()
