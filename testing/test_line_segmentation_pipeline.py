#!/usr/bin/env python3
"""Regression checks for JavaScript math-line segmentation."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

TESTING_DIR = Path(__file__).resolve().parent
ROOT = TESTING_DIR.parent
if str(TESTING_DIR) not in sys.path:
    sys.path.insert(0, str(TESTING_DIR))

from fixture_catalog import build_board, get_problem, line_gaps_for_pattern
from synthetic_handwriting import place_handwriting_lines


SEGMENTER = ROOT / "testing" / "segment_fixture_with_js.mjs"
REAL_HANDWRITING_FIXTURE_DIR = TESTING_DIR / "fixtures" / "real_handwriting"


def run_segmentation(board, order: str = "line-order") -> dict[str, Any]:
    return run_segmentation_payload(board.to_json(), order=order)


def run_segmentation_payload(payload: dict[str, Any], order: str = "line-order") -> dict[str, Any]:
    completed = subprocess.run(
        ["node", str(SEGMENTER), order],
        cwd=str(ROOT),
        input=json.dumps(payload),
        text=True,
        check=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def load_real_handwriting_fixtures() -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(REAL_HANDWRITING_FIXTURE_DIR.glob("*.json"))
    ]


def stroke_group_key(stroke_ids: list[str]) -> str:
    return "|".join(sorted(map(str, stroke_ids or [])))


def assert_clean_line_cover(testcase: unittest.TestCase, result: dict[str, Any], expected_count: int):
    testcase.assertEqual(
        result["selectedCount"],
        expected_count,
        msg=json.dumps(result["selected"], indent=2),
    )
    covered = []
    for selected in result["selected"]:
        line_set = selected["syntheticLineSets"]
        testcase.assertEqual(
            len(line_set),
            1,
            msg=f"Candidate mixes fixture lines: {json.dumps(selected, indent=2)}",
        )
        covered.extend(line_set)
    testcase.assertEqual(sorted(covered), list(range(expected_count)))


class LineSegmentationPipelineTests(unittest.TestCase):
    def test_segments_varied_problem_families_and_stroke_orders(self):
        cases = [
            ("algebra_steps", "tight-steps", "line-order", 21),
            ("algebra_steps", "mixed", "interleaved-lines", 22),
            ("rational_two_fraction_solve", "tight-steps", "reverse-lines", 23),
            ("rational_quadratic_solve", "standard", "interleaved-lines", 101),
            ("logarithmic_product_solve", "mixed", "line-order", 24),
            ("derivative_quotient_evaluate", "dense", "interleaved-lines", 25),
            ("integral_fraction_antiderivative", "tight-steps", "reverse-lines", 26),
        ]

        for problem_name, spacing, order, seed in cases:
            with self.subTest(problem_name=problem_name, spacing=spacing, order=order):
                problem = get_problem(problem_name)
                board = build_board(problem.name, spacing=spacing, seed=seed)
                result = run_segmentation(board, order=order)
                assert_clean_line_cover(self, result, len(problem.lines))

    def test_keeps_single_fraction_expression_together(self):
        latex = r"\frac { x - 1 } { x + 1 } = 5"
        board = place_handwriting_lines(
            [latex],
            board_width=1200,
            board_height=520,
            placements=[{"x": 120, "y": 90}],
            seed=41,
            max_line_width=900,
            max_line_height=150,
        )
        result = run_segmentation(board, order="interleaved-lines")
        assert_clean_line_cover(self, result, 1)

    def test_preserves_parent_child_hypotheses_before_selection(self):
        problem = get_problem("algebra_steps")
        board = build_board(problem.name, spacing="tight-steps", seed=29)
        result = run_segmentation(board, order="line-order")

        self.assertGreater(result["candidateCount"], result["selectedCount"])
        self.assertGreaterEqual(result["partitions"].get("parent", 0), 1)
        self.assertGreaterEqual(result["partitions"].get("row-line", 0), len(problem.lines))
        assert_clean_line_cover(self, result, len(problem.lines))

    def test_segments_messy_writer_style(self):
        problem = get_problem("rational_two_fraction_solve")
        board = build_board(problem.name, spacing="tight-steps", seed=119, ink_style="messy")
        result = run_segmentation(board, order="interleaved-lines")
        assert_clean_line_cover(self, result, len(problem.lines))

    def test_segments_distilled_real_handwriting_trace_fixtures(self):
        fixtures = load_real_handwriting_fixtures()
        self.assertGreaterEqual(len(fixtures), 6)

        for fixture in fixtures:
            with self.subTest(slug=fixture["slug"]):
                result = run_segmentation_payload(fixture, order="line-order")
                expected_keys = sorted(
                    stroke_group_key(group["strokeIds"])
                    for group in fixture["expectedLineGroups"]
                )
                actual_keys = sorted(
                    stroke_group_key(candidate["strokeIds"])
                    for candidate in result["selected"]
                )

                self.assertEqual(result["strokeCount"], len(fixture["strokes"]))
                replay_mismatch = fixture.get("expectedSegmentationReplayMismatch") or fixture.get("expectedReplayMismatch")
                if replay_mismatch:
                    self.assertTrue(replay_mismatch.get("issueId"))
                    mismatch_keys = sorted(
                        stroke_group_key(group)
                        for group in replay_mismatch.get("selectedGroups", [])
                    )
                    self.assertEqual(actual_keys, mismatch_keys, msg=json.dumps(result["selected"], indent=2))
                    continue
                self.assertEqual(actual_keys, expected_keys, msg=json.dumps(result["selected"], indent=2))
                for candidate in result["selected"]:
                    self.assertEqual(
                        len(candidate.get("expectedLineSets") or []),
                        1,
                        msg=f"Selected trace candidate mixes reviewed groups: {json.dumps(candidate, indent=2)}",
                    )

    def test_compact_final_fraction_survives_messy_tight_steps(self):
        problem = get_problem("rational_solve")
        for spacing in ("dense", "tight-steps"):
            for order in ("interleaved-lines", "reverse-lines"):
                with self.subTest(spacing=spacing, order=order):
                    board = build_board(problem.name, spacing=spacing, seed=515, ink_style="messy")
                    result = run_segmentation(board, order=order)
                    assert_clean_line_cover(self, result, len(problem.lines))

    def test_two_sided_fraction_problem_survives_pinched_messy_spacing(self):
        problem = get_problem("rational_two_sided_fraction_solve")
        gaps = line_gaps_for_pattern(len(problem.lines), "pinched-middle")
        board = build_board(
            problem.name,
            spacing="dense",
            line_gaps=gaps,
            seed=2441,
            ink_style="messy",
        )
        result = run_segmentation(board, order="interleaved-lines")
        assert_clean_line_cover(self, result, len(problem.lines))

    def test_collapsed_fraction_bar_stays_with_final_answer_line(self):
        problem = get_problem("rational_solve")
        board = build_board(problem.name, spacing="tight-steps", seed=998, ink_style="messy")
        result = run_segmentation(board, order="interleaved-lines")
        assert_clean_line_cover(self, result, len(problem.lines))

    def test_integral_evaluation_continuation_does_not_merge_with_antiderivative(self):
        problem = get_problem("integral_fraction_antiderivative")
        board = build_board(problem.name, spacing="tight-steps", seed=1720, ink_style="messy")
        result = run_segmentation(board, order="interleaved-lines")
        assert_clean_line_cover(self, result, len(problem.lines))

    def test_zero_gap_log_lines_prefer_independent_children_over_loose_parent(self):
        problem = get_problem("logarithmic_product_solve")
        gaps = line_gaps_for_pattern(len(problem.lines), "accordion")
        board = build_board(
            problem.name,
            spacing="dense",
            line_gaps=gaps,
            seed=1720,
            ink_style="messy",
        )
        result = run_segmentation(board, order="interleaved-lines")
        assert_clean_line_cover(self, result, len(problem.lines))

    def test_zero_gap_derivative_rows_do_not_capture_next_line_strokes(self):
        problem = get_problem("derivative_product_evaluate")
        gaps = line_gaps_for_pattern(len(problem.lines), "accordion")
        board = build_board(
            problem.name,
            spacing="dense",
            line_gaps=gaps,
            seed=616,
            ink_style="messy",
        )
        result = run_segmentation(board, order="interleaved-lines")
        assert_clean_line_cover(self, result, len(problem.lines))

    def test_pinched_integral_substitution_rows_split_at_center_gap(self):
        problem = get_problem("integral_evaluate")
        gaps = line_gaps_for_pattern(len(problem.lines), "pinched-middle")
        board = build_board(
            problem.name,
            spacing="dense",
            line_gaps=gaps,
            seed=1720,
            ink_style="messy",
        )
        result = run_segmentation(board, order="interleaved-lines")
        assert_clean_line_cover(self, result, len(problem.lines))

    def test_fixture_bridge_can_reselect_exact_cover_with_candidate_scores(self):
        board = place_handwriting_lines(
            [r"x = 1", r"y = 2"],
            board_width=800,
            board_height=420,
            placements=[{"x": 120, "y": 80}, {"x": 122, "y": 210}],
            seed=72,
            max_line_width=520,
            max_line_height=86,
        )
        first = run_segmentation(board, order="line-order")
        parent = next(candidate for candidate in first["candidates"] if "parent" in candidate["profiles"])
        rescored = run_segmentation_payload({
            **board.to_json(),
            "scoreByCandidateId": {
                parent["candidateId"]: 50,
            },
        }, order="line-order")

        self.assertEqual(rescored["rescoredSelectedCount"], 1)
        self.assertEqual(rescored["rescoredSelected"][0]["candidateId"], parent["candidateId"])


if __name__ == "__main__":
    unittest.main()
