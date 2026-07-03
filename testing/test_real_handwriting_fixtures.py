#!/usr/bin/env python3
"""Schema and anonymization checks for distilled real handwriting fixtures."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "real_handwriting"
EXPECTED_FIXTURE_COUNT = 37


class RealHandwritingFixtureSchemaTests(unittest.TestCase):
    def test_committed_fixture_set_is_present(self):
        fixtures = load_fixtures()

        self.assertEqual(len(fixtures), EXPECTED_FIXTURE_COUNT)
        self.assertTrue(any(item["visualMarks"] for item in fixtures))
        self.assertTrue(any(item["knownDiscrepancyTypes"] for item in fixtures))

    def test_fixtures_are_anonymized_and_relative(self):
        for fixture in load_fixtures():
            with self.subTest(slug=fixture["slug"]):
                self.assertEqual(fixture["fixtureKind"], "real-handwriting-trace")
                self.assertIn("sourceAuditId", fixture)
                self.assertNotIn("sourceAuditDate", fixture)
                self.assert_no_absolute_paths(fixture)
                for stroke in fixture["strokes"]:
                    self.assertGreaterEqual(stroke["startTime"], 0)
                    self.assertLess(stroke["startTime"], 8 * 60 * 60 * 1000)
                    self.assertLess(stroke["endTime"], 8 * 60 * 60 * 1000)
                self.assertLess(
                    max(stroke["startTime"] for stroke in fixture["strokes"]),
                    8 * 60 * 60 * 1000,
                    "stroke times should be relative to the fixture, not epoch timestamps",
                )

    def test_strokes_have_valid_geometry_and_relations(self):
        for fixture in load_fixtures():
            stroke_ids = {stroke["id"] for stroke in fixture["strokes"]}
            with self.subTest(slug=fixture["slug"]):
                self.assertGreater(len(stroke_ids), 0)
                self.assertEqual(len(stroke_ids), len(fixture["strokes"]))
                for stroke in fixture["strokes"]:
                    self.assert_valid_box(stroke["canvasBbox"])
                    self.assert_valid_box(stroke["bbox"], normalized=True)
                    self.assertGreaterEqual(stroke["endTime"], stroke["startTime"])
                    self.assertTrue(stroke["rawPoints"])
                    self.assertTrue(stroke["outlinePoints"])
                    self.assertEqual(len(stroke["points"]), len(stroke["rawPoints"]))
                    self.assertIn("relationsToPrev", stroke)

                for group in fixture["expectedLineGroups"]:
                    self.assertTrue(group["latex"])
                    self.assertTrue(group["strokeIds"])
                    self.assertTrue(set(group["strokeIds"]) <= stroke_ids)

                for detection in fixture.get("detections") or []:
                    self.assert_valid_box(detection["bbox"])
                    self.assert_valid_box(detection["normalizedBbox"], normalized=True)
                    if "polygon" in detection:
                        self.assertGreaterEqual(len(detection["polygon"]), 4)

                visual_only = set(fixture.get("visualOnlyStrokeIds") or [])
                grouped = set(
                    stroke_id
                    for group in fixture["expectedLineGroups"]
                    for stroke_id in group["strokeIds"]
                )
                self.assertFalse(visual_only & grouped)
                for stroke in fixture["strokes"]:
                    if stroke["id"] in visual_only:
                        self.assertTrue(stroke.get("visualOnly"))

    def test_vlm_transcript_and_fast_lines_are_both_preserved(self):
        mismatched = 0
        for fixture in load_fixtures():
            self.assertTrue(fixture["expectedLatexLines"], fixture["slug"])
            if not fixture["fastLatexLines"]:
                self.assertIn("line_segmentation_empty", fixture["knownDiscrepancyTypes"], fixture["slug"])
            else:
                self.assertTrue(fixture["fastLatexLines"], fixture["slug"])
            if fixture["expectedLatexLines"] != fixture["fastLatexLines"]:
                mismatched += 1

        self.assertGreaterEqual(mismatched, 4)

    def assert_no_absolute_paths(self, value: Any):
        if isinstance(value, dict):
            for child in value.values():
                self.assert_no_absolute_paths(child)
        elif isinstance(value, list):
            for child in value:
                self.assert_no_absolute_paths(child)
        elif isinstance(value, str):
            self.assertNotIn("/Users/", value)
            self.assertNotIn(str(Path.home()), value)

    def assert_valid_box(self, box: dict[str, Any], *, normalized: bool = False):
        self.assertLess(box["xMin"], box["xMax"])
        self.assertLess(box["yMin"], box["yMax"])
        if normalized:
            for key in ("xMin", "yMin", "xMax", "yMax"):
                self.assertGreaterEqual(box[key], 0)
                self.assertLessEqual(box[key], 1)


def load_fixtures() -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(FIXTURE_DIR.glob("*.json"))
    ]


if __name__ == "__main__":
    unittest.main()
