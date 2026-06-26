#!/usr/bin/env python3
"""Fast checks for standalone synthetic math fixture generation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
if str(TESTING_DIR) not in sys.path:
    sys.path.insert(0, str(TESTING_DIR))

from fixture_catalog import PROBLEMS, build_board, gaps_for_spacing, get_problem, placements_for
from synthetic_handwriting import normalize_latex, place_handwriting_lines, render_handwriting


class SyntheticHandwritingTests(unittest.TestCase):
    def test_renders_simple_solution_to_ink_contours(self):
        fixture = render_handwriting("x = 4", seed=4)
        self.assertGreater(fixture.width, 10)
        self.assertGreater(fixture.height, 10)
        self.assertGreater(len(fixture.contours), 0)
        self.assertTrue(fixture.data_url.startswith("data:image/png;base64,"))

    def test_renders_nested_math_to_ink_contours(self):
        fixture = render_handwriting(
            r"x = \frac { - b \pm \sqrt { b ^ { 2 } - 4 a c } } { 2 a }",
            seed=9,
        )
        self.assertGreater(fixture.width, 100)
        self.assertGreater(fixture.height, 40)
        self.assertGreater(len(fixture.contours), 5)

    def test_normalizes_token_spaced_latex_for_mathtext(self):
        self.assertEqual(normalize_latex(r"x ^ { 3 } + 2 x ^ { 2 }"), r"x^{3}+ 2 x^{2}")
        self.assertEqual(normalize_latex(r"\log _ { 2 } ( x )"), r"\log_{2}( x )")
        self.assertEqual(normalize_latex(r"\int _ { 0 } ^ { 2 } x d x"), r"\int_{0}^{2}x d x")
        self.assertEqual(normalize_latex(r"f ' ( 2 ) = 20"), r"f^{\prime}( 2 ) = 20")

    def test_places_multiple_algebra_lines_on_board(self):
        board = place_handwriting_lines(
            [
                r"2 x + 3 = 11",
                r"2 x = 8",
                r"x = 4",
            ],
            board_width=1200,
            board_height=720,
            placements=[
                {"x": 120, "y": 90},
                {"x": 188, "y": 260},
                {"x": 250, "y": 430},
            ],
            seed=22,
        )
        self.assertEqual(len(board.lines), 3)
        self.assertGreater(len(board.contours), 8)
        self.assertTrue(board.data_url.startswith("data:image/png;base64,"))
        self.assertLess(board.lines[0].bbox["yMax"], board.lines[1].bbox["yMin"])
        self.assertLess(board.lines[1].bbox["yMax"], board.lines[2].bbox["yMin"])

    def test_places_lines_by_board_anchors(self):
        board = place_handwriting_lines(
            [r"x = 4", r"y = 2", r"z = 6"],
            board_width=900,
            board_height=600,
            placements=[
                {"anchor": "top-left"},
                {"anchor": "center"},
                {"anchor": "bottom-right"},
            ],
            seed=30,
        )
        self.assertLess(board.lines[0].x, board.lines[1].x)
        self.assertLess(board.lines[0].y, board.lines[1].y)
        self.assertGreater(board.lines[2].x, board.lines[1].x)
        self.assertGreater(board.lines[2].y, board.lines[1].y)

    def test_default_placement_accepts_custom_non_uniform_line_gaps(self):
        gaps = [70, 12, 45, 6]
        board = place_handwriting_lines(
            [
                r"2 x + 3 = 11",
                r"- 3      - 3",
                r"2 x = 8",
                r"/ 2      / 2",
                r"x = 4",
            ],
            board_width=1200,
            board_height=900,
            seed=42,
            max_line_height=104,
            max_line_width=850,
            line_gaps=gaps,
        )
        top_deltas = [
            board.lines[index + 1].y - board.lines[index].y
            for index in range(len(board.lines) - 1)
        ]
        self.assertEqual(top_deltas, [104 + gap for gap in gaps])

    def test_custom_line_gaps_must_match_line_count(self):
        with self.assertRaises(ValueError):
            place_handwriting_lines(
                [r"x = 1", r"x = 2", r"x = 3"],
                board_width=800,
                board_height=600,
                line_gaps=[30],
            )


class FixtureCatalogTests(unittest.TestCase):
    def test_catalog_contains_required_problem_families(self):
        families = {problem.family for problem in PROBLEMS}
        self.assertTrue({"algebra", "rational", "logarithmic", "derivative", "integral"} <= families)

    def test_builds_representative_simple_and_complex_boards(self):
        for problem_name in [
            "algebra_simple",
            "algebra_steps",
            "rational_quadratic_solve",
            "logarithmic_solve",
            "derivative_evaluate",
            "integral_evaluate",
        ]:
            with self.subTest(problem_name=problem_name):
                problem = get_problem(problem_name)
                board = build_board(problem_name, spacing="dense", seed=75)
                self.assertEqual(len(board.lines), len(problem.lines))
                self.assertGreater(len(board.contours), len(problem.lines))
                self.assertTrue(board.data_url.startswith("data:image/png;base64,"))

    def test_mixed_spacing_uses_non_uniform_gaps(self):
        problem = get_problem("rational_quadratic_solve")
        gaps = gaps_for_spacing(len(problem.lines), "mixed")
        self.assertGreater(len(set(gaps)), 1)
        placements, _, _, placement_gaps = placements_for(problem, "mixed")
        self.assertEqual(gaps, placement_gaps)
        top_deltas = [
            placements[index + 1]["y"] - placements[index]["y"]
            for index in range(len(placements) - 1)
        ]
        self.assertEqual(top_deltas, [problem.max_line_height + gap for gap in gaps])

    def test_build_board_custom_line_gaps_override_spacing(self):
        gaps = [64, 12, 48]
        problem = get_problem("derivative_evaluate")
        board = build_board(problem.name, spacing="standard", line_gaps=gaps, seed=91)
        top_deltas = [
            board.lines[index + 1].y - board.lines[index].y
            for index in range(len(board.lines) - 1)
        ]
        self.assertEqual(top_deltas, [problem.max_line_height + gap for gap in gaps])

    def test_build_board_rejects_wrong_custom_line_gap_count(self):
        with self.assertRaises(ValueError):
            build_board("derivative_evaluate", line_gaps=[64, 12], seed=91)


if __name__ == "__main__":
    unittest.main()
