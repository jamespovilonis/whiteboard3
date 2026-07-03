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
    build_calibration_report,
    build_calibration_summary,
    build_hybrid_linear_equation_fixture,
    build_realistic_fixture,
    classify_features,
    fixture_metrics,
    fixture_summary,
    load_handwriting_atom_catalog,
    load_audit_input_payloads,
    load_corpus_records,
    summarize_distribution,
    validate_fixture_against_calibration,
)


SEGMENTER = ROOT / "testing" / "segment_fixture_with_js.mjs"


class RealisticHandwritingHarnessTests(unittest.TestCase):
    def test_curated_catalog_extracts_real_stroke_atoms(self):
        catalog = load_handwriting_atom_catalog()

        for label in ["3x", "+", "2", "=", "12", "x=4", "circle", "crossout"]:
            self.assertIn(label, catalog)
            self.assertGreater(len(catalog[label][0].strokes), 0)
            self.assertGreater(catalog[label][0].width, 0)
            self.assertGreater(catalog[label][0].height, 0)
            self.assertTrue(catalog[label][0].source_fixture_slug)

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
        self.assertGreater(calibration.bands["line_count"].median, 0)

    def test_phase_one_report_describes_committed_fixture_corpus(self):
        report = build_calibration_report(
            audit_log_dir=Path("/path/that/does/not/exist"),
            sample_limit=3,
        )

        self.assertEqual(report["reportKind"], "real-handwriting-calibration")
        self.assertEqual(report["sourceKinds"], ["distilled-real-fixtures"])
        self.assertGreaterEqual(report["recordCount"], 30)
        self.assertIn("stroke_duration_ms", report["calibration"]["bands"])
        self.assertIn("line_gap_px", report["calibration"]["bands"])
        self.assertIn("has_crossout_or_scratch", report["featureKeys"])
        self.assertEqual(len(report["sampleRecords"]), 3)
        for sample in report["sampleRecords"]:
            self.assertIn("sourceId", sample)
            self.assertIn("summary", sample)
            self.assertGreater(sample["summary"]["strokeCount"], 0)

    def test_corpus_records_keep_source_metadata_for_audits_or_fixtures(self):
        records = load_corpus_records(
            audit_log_dir=DEFAULT_AUDIT_LOG_DIR,
            audit_limit=4,
        )
        self.assertGreater(len(records), 0)
        for record in records:
            self.assertTrue(record.source_id)
            self.assertTrue(record.path)
            self.assertTrue(record.payload.get("strokes"))

    def test_fixture_summary_is_compact_phase_one_surface(self):
        fixture = build_realistic_fixture("circled-answer", seed=40)
        summary = fixture_summary(fixture)

        self.assertEqual(summary["fixtureKind"], "realistic-handwriting-trace")
        self.assertEqual(summary["lineCount"], len(fixture["expectedLineGroups"]))
        self.assertGreater(summary["pointCountMedian"], 0)
        self.assertIn("has_circled_answer", summary["features"])

    def test_hybrid_linear_generator_composes_unseen_equation_from_catalog_atoms(self):
        fixture = build_hybrid_linear_equation_fixture(
            a=3,
            b=2,
            x_value=4,
            seed=11,
            include_crossout=True,
        )
        features = classify_features(fixture)

        self.assertEqual(fixture["fixtureKind"], "hybrid-real-stroke-linear-equation")
        self.assertEqual(fixture["problemLatex"], "3x + 2 = 14")
        self.assertEqual(
            fixture["expectedLatexLines"],
            ["3x+2=14", "-2\\quad -2", "3x=12", "/3\\quad /3", "x=4"],
        )
        self.assertEqual(len(fixture["expectedLineGroups"]), 5)
        self.assertTrue(features["has_pressure"])
        self.assertTrue(features["has_visual_only_marks"])
        self.assertTrue(features["has_circled_answer"])
        self.assertTrue(features["has_crossout_or_scratch"])
        self.assertTrue(any(stroke.get("sourceAtomLabel") == "3x" for stroke in fixture["strokes"]))

    def test_hybrid_linear_fixture_segments_into_expected_rows(self):
        fixture = build_hybrid_linear_equation_fixture(a=3, b=2, x_value=4, seed=21)
        completed = subprocess.run(
            ["node", str(SEGMENTER), "line-order"],
            cwd=str(ROOT),
            input=json.dumps(fixture),
            text=True,
            check=True,
            capture_output=True,
        )
        result = json.loads(completed.stdout)
        expected_keys = sorted(stroke_group_key(group["strokeIds"]) for group in fixture["expectedLineGroups"])
        actual_keys = sorted(stroke_group_key(candidate["strokeIds"]) for candidate in result["selected"])

        self.assertEqual(result["strokeCount"], len(fixture["strokes"]))
        self.assertEqual(actual_keys, expected_keys, json.dumps(result["selected"], indent=2))

    def test_hybrid_linear_fixture_validates_against_real_distribution(self):
        calibration = build_calibration_summary(audit_log_dir=Path("/path/that/does/not/exist"))
        fixture = build_hybrid_linear_equation_fixture(a=3, b=2, x_value=4, seed=23)
        report = validate_fixture_against_calibration(
            fixture,
            calibration,
            required_features=[
                "has_pressure",
                "has_multi_stroke_symbols",
                "has_visual_only_marks",
                "has_circled_answer",
            ],
        )

        self.assertTrue(report["ok"], json.dumps(report, indent=2))

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


def stroke_group_key(stroke_ids: list[str]) -> str:
    return "|".join(sorted(map(str, stroke_ids or [])))


if __name__ == "__main__":
    unittest.main()
