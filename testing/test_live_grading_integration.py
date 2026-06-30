#!/usr/bin/env python3
"""Blind integration test: render synthetic handwriting → real OCR → real grading.

This test is the most "blind" test in the suite. It:
1. Renders synthetic handwriting to PNG images via fixture_catalog.py
2. Sends each line image to the real CoMER OCR model
3. Feeds the OCR output through the real Python grader
4. Asserts the final grading verdict matches expectations

Unlike test_equation_grader.py (which uses hardcoded LaTeX strings) or
test_recognition_pipeline.mjs (which mocks the OCR), this test exercises
the actual OCR model on actual rendered handwriting.

Requirements:
- The CoMER OCR server must be running at http://127.0.0.1:8010
- Set LIVE_GRADING_E2E=1 to enable this test (skipped by default)
"""

from __future__ import annotations

import json
import os
import sys
import time
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTING_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TESTING_DIR) not in sys.path:
    sys.path.insert(0, str(TESTING_DIR))

from src.grading import create_answer_manifest, grade_equation_work
from fixture_catalog import get_problem, build_board, fixture_payload, placements_for
from run_live_recognition_matrix import (
    render_fixture,
    post_image,
    recognize_crop_with_retries,
    DEFAULT_INITIAL_RASTER_HEIGHT,
    DEFAULT_INITIAL_RASTER_MIN_HEIGHT,
)


def ocr_server_available(api_url: str = "http://127.0.0.1:8010") -> bool:
    """Check if the OCR server is reachable."""
    try:
        request = urllib.request.Request(f"{api_url}/health", method="GET")
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


@unittest.skipUnless(
    os.environ.get("LIVE_GRADING_E2E") == "1",
    "Set LIVE_GRADING_E2E=1 and start the OCR server to run the blind grading integration test."
)
class LiveGradingIntegrationTest(unittest.TestCase):
    """Blind integration test: render → OCR → grade."""

    @classmethod
    def setUpClass(cls):
        cls.api_url = os.environ.get("OCR_API_URL", "http://127.0.0.1:8010")
        if not ocr_server_available(cls.api_url):
            raise unittest.SkipTest(
                f"OCR server not reachable at {cls.api_url}. "
                "Start it with: python3 -m src.server.app --upstream-api-url http://127.0.0.1:8000"
            )

    def test_algebra_simple_blind_grading(self):
        """Render algebra_simple fixture, OCR each line, grade the work."""
        problem = get_problem("algebra_simple")
        fixture = render_fixture(problem, spacing="standard", seed=101, ink_style="normal")

        # The fixture renders lines as separate crops. We OCR each line image
        # and collect the recognized LaTeX.
        recognize_url = f"{self.api_url}/recognize?model=comer&timeout_seconds=20"
        recognized_lines = []

        # The board payload contains line bounding boxes.
        board = fixture.board
        lines = getattr(board, "lines", []) or []
        if not lines:
            # Fallback: use the fixture payload's line data
            payload_lines = fixture.payload.get("lines", [])
            for line_data in payload_lines:
                bbox = line_data.get("bbox", {})
                if not bbox:
                    continue
                # Create a crop from the board PNG using the line bbox
                crop_path = self._crop_line_from_board(fixture.png_path, bbox)
                if crop_path is None:
                    continue
                payload = post_image(recognize_url, crop_path, 20.0)
                latex = self._extract_latex(payload)
                if latex:
                    recognized_lines.append(latex)
        else:
            for line in lines:
                bbox = getattr(line, "bbox", None) or line.get("bbox", {})
                if not bbox:
                    continue
                crop_path = self._crop_line_from_board(fixture.png_path, bbox)
                if crop_path is None:
                    continue
                payload = post_image(recognize_url, crop_path, 20.0)
                latex = self._extract_latex(payload)
                if latex:
                    recognized_lines.append(latex)

        self.assertGreater(len(recognized_lines), 0, "OCR should recognize at least one line")

        # Create the answer manifest from the problem
        manifest = create_answer_manifest(problem.context_latex)

        # Grade the recognized work
        work = [{"latex": latex} for latex in recognized_lines]
        result = grade_equation_work(manifest, work)

        # The algebra_simple problem is 2x + 3 = 11 with solution x = 4.
        # The expected lines are: 2x + 3 = 11, 2x = 8, x = 4.
        # If OCR recognizes all lines correctly, the problem should be correct.
        # If OCR misreads some lines, the problem may be incomplete or incorrect,
        # but it should NOT be "not_started" (we have recognized math lines).
        self.assertNotEqual(
            result["result"]["problemStatus"], "not_started",
            f"Problem should not be 'not_started' with recognized lines: {recognized_lines}"
        )

        # Verify the grading structure is well-formed
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result.get("failed", False))
        self.assertGreater(len(result["steps"]), 0)

        for step in result["steps"]:
            self.assertIn(step["classification"], ["valid_step", "invalid_step", "other"])
            self.assertTrue(step["studentLatex"])

        # Attach diagnostic info
        print(f"\n  Problem: {problem.context_latex}")
        print(f"  Recognized lines: {recognized_lines}")
        print(f"  Problem status: {result['result']['problemStatus']}")
        print(f"  Found solutions: {result['result']['foundSolutions']}")
        print(f"  Missing solutions: {result['result']['missingSolutions']}")
        for i, step in enumerate(result["steps"]):
            print(f"  Step {i}: {step['studentLatex']} → {step['classification']}")

    def _crop_line_from_board(self, board_png: Path, bbox: dict[str, Any]) -> Path | None:
        """Crop a line region from the board PNG."""
        try:
            from PIL import Image
        except ImportError:
            return None

        x_min = int(bbox.get("xMin", 0))
        y_min = int(bbox.get("yMin", 0))
        x_max = int(bbox.get("xMax", x_min + 100))
        y_max = int(bbox.get("yMax", y_min + 50))

        # Add padding
        padding = 24
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = x_max + padding
        y_max = y_max + padding

        try:
            img = Image.open(board_png)
            cropped = img.crop((x_min, y_min, x_max, y_max))
            crop_path = board_png.parent / f"{board_png.stem}_crop_{x_min}_{y_min}.png"
            cropped.save(crop_path)
            return crop_path
        except Exception:
            return None

    def _extract_latex(self, payload: dict[str, Any]) -> str:
        """Extract the top LaTeX string from an OCR response."""
        if not payload or payload.get("failed") or payload.get("timedOut"):
            return ""
        top = payload.get("top") or {}
        latex = str(top.get("latex") or payload.get("latex") or "").strip()
        if not latex:
            candidates = payload.get("candidates") or []
            if candidates:
                latex = str(candidates[0].get("latex", "")).strip()
        return latex


if __name__ == "__main__":
    unittest.main()
