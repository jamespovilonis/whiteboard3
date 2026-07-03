#!/usr/bin/env python3
"""Checks for real-user calibrated handwriting fixture generation."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


TESTING_DIR = Path(__file__).resolve().parent
ROOT = TESTING_DIR.parent
if str(TESTING_DIR) not in sys.path:
    sys.path.insert(0, str(TESTING_DIR))

from realistic_handwriting import (
    DEFAULT_AUDIT_LOG_DIR,
    build_calibration_summary,
    build_realistic_fixture,
    classify_features,
    fixture_metrics,
    load_audit_input_payloads,
    summarize_distribution,
    validate_fixture_against_calibration,
)


SEGMENTER = ROOT / "testing" / "segment_fixture_with_js.mjs"


class RealisticHandwritingHarnessTests(unittest.TestCase):
    def test_builds_mixed_fixture_from_real_stroke_trajectories(self):
        fixture = build_realistic_fixture("mixed-marks", seed=17)
        features = classify_features(fixture)
        metrics = fixture_metrics(fixture)

        self.assertEqual(fixture["fixtureKind"], "realistic-handwriting-trace")
        self.assertGreater(len(fixture["strokes"]), 20)
        self.assertGreater(len(fixture["expectedLineGroups"]), 2)
        self.assertTrue(features["has_pressure"])
        self.assertTrue(features["has_multi_stroke_symbols"])
        self.assertTrue(features["has_visual_only_marks"])
        self.assertTrue(features["has_circled_answer"])
        self.assertTrue(features["has_crossout_or_scratch"])
        self.assertTrue(features["has_non_sequential_writing"])
        self.assertTrue(any(value > 500 for value in metrics["inter_stroke_gap_ms"]))

        for stroke in fixture["strokes"]:
            self.assertTrue(stroke["rawPoints"])
            self.assertTrue(stroke["outlinePoints"])
            self.assertIn("pressure", stroke["rawPoints"][0])
            self.assertIn("relationsToPrev", stroke)
            self.assertLess(stroke["bbox"]["xMin"], stroke["bbox"]["xMax"])
            self.assertLess(stroke["bbox"]["yMin"], stroke["bbox"]["yMax"])

    def test_generated_fixture_validates_against_distilled_real_distribution(self):
        calibration = build_calibration_summary(audit_log_dir=Path("/path/that/does/not/exist"))
        fixture = build_realistic_fixture("crossout-scratch", seed=9)
        report = validate_fixture_against_calibration(
            fixture,
            calibration,
            required_features=[
                "has_pressure",
                "has_multi_stroke_symbols",
                "has_visual_only_marks",
                "has_crossout_or_scratch",
            ],
        )

        self.assertEqual(calibration.source_kind, "distilled-real-fixtures")
        self.assertGreaterEqual(calibration.source_count, 30)
        self.assertTrue(report["ok"], json.dumps(report, indent=2))

    def test_live_audit_input_distribution_is_read_when_available(self):
        payloads = load_audit_input_payloads(DEFAULT_AUDIT_LOG_DIR, limit=8)
        if not payloads:
            self.skipTest("local audit input.json logs are not available")

        calibration = summarize_distribution(payloads, source_kind="audit-input-json")
        self.assertEqual(calibration.source_kind, "audit-input-json")
        self.assertEqual(calibration.source_count, len(payloads))
        self.assertIn("stroke_count", calibration.bands)
        self.assertIn("point_count", calibration.bands)
        self.assertGreater(calibration.bands["stroke_count"].median, 0)

    def test_generated_fixture_can_be_consumed_by_segmentation_bridge(self):
        fixture = build_realistic_fixture("circled-answer", seed=33)
        completed = subprocess.run(
            ["node", str(SEGMENTER), "line-order"],
            cwd=str(ROOT),
            input=json.dumps(fixture),
            text=True,
            check=True,
            capture_output=True,
        )
        result = json.loads(completed.stdout)

        self.assertEqual(result["strokeCount"], len(fixture["strokes"]))
        self.assertGreaterEqual(result["candidateCount"], result["selectedCount"])
        self.assertGreater(result["selectedCount"], 0)


if __name__ == "__main__":
    unittest.main()
