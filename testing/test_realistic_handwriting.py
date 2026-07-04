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
    build_harness_dashboard_report,
    build_hybrid_complex_math_fixture,
    build_hybrid_linear_equation_fixture,
    build_log_observed_pack_fixture,
    build_random_hybrid_linear_equation_fixture,
    build_realistic_fixture,
    default_dashboard_fixtures,
    classify_features,
    fixture_metrics,
    fixture_summary,
    load_handwriting_atom_catalog,
    load_audit_input_payloads,
    load_corpus_records,
    sample_linear_equation_parameters,
    summarize_distribution,
    validate_fixture_against_calibration,
)


SEGMENTER = ROOT / "testing" / "segment_fixture_with_js.mjs"


class RealisticHandwritingHarnessTests(unittest.TestCase):
    def test_curated_catalog_extracts_real_stroke_atoms(self):
        catalog = load_handwriting_atom_catalog()

        for label in ["3x", "+", "2", "=", "12", "x=4", "circle", "crossout", "underline"]:
            self.assertIn(label, catalog)
            self.assertGreater(len(catalog[label][0].strokes), 0)
            self.assertGreater(catalog[label][0].width, 0)
            self.assertGreater(catalog[label][0].height, 0)
            self.assertTrue(catalog[label][0].source_fixture_slug)
        minimum_variants = {
            "0": 4,
            "x": 4,
            "=": 3,
            "1": 3,
            "2": 4,
            "3": 3,
            "4": 4,
            "5": 4,
            "6": 3,
            "7": 3,
            "8": 4,
            "9": 4,
            "+": 3,
            "-": 4,
            "/": 3,
            "3x": 2,
            "12": 3,
            "circle": 2,
            "underline": 3,
        }
        for label, minimum in minimum_variants.items():
            self.assertGreaterEqual(len(catalog[label]), minimum, label)
        for digit in "0123456789":
            self.assertIn(digit, catalog)
        complex_labels = [
            "fraction_addition_row",
            "x_squared",
            "frac_9_4",
            "plus_minus",
            "frac_3_2",
            "frac_7_sqrt9",
            "sqrt_x18_over_x",
            "sqrt_25",
            "frac_7_3",
            "rational_factor_row",
            "rational_linear_row",
            "rational_isolated_row",
            "detached_operation_left",
            "detached_operation_right",
        ]
        for label in complex_labels:
            self.assertIn(label, catalog)
            self.assertGreater(len(catalog[label][0].strokes), 0)

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
        self.assertGreaterEqual(
            len({stroke.get("sourceFixtureSlug") for stroke in fixture["strokes"] if stroke.get("sourceAtomLabel") == "="}),
            3,
        )
        self.assertGreaterEqual(
            len({stroke.get("sourceFixtureSlug") for stroke in fixture["strokes"] if stroke.get("sourceAtomLabel") == "2"}),
            2,
        )
        self.assertGreaterEqual(
            len({stroke.get("sourceFixtureSlug") for stroke in fixture["strokes"] if stroke.get("sourceAtomLabel") == "-"}),
            2,
        )
        self.assertGreaterEqual(
            len({stroke.get("sourceFixtureSlug") for stroke in fixture["strokes"] if stroke.get("sourceAtomLabel") == "/"}),
            2,
        )
        underline_strokes = [
            stroke for stroke in fixture["strokes"]
            if stroke.get("sourceAtomLabel") == "underline"
        ]
        self.assertEqual(len(underline_strokes), 4)
        self.assertTrue({stroke["id"] for stroke in underline_strokes} <= set(fixture["visualOnlyStrokeIds"]))

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

    def test_random_hybrid_linear_generator_samples_and_solves_seeded_equations(self):
        fixtures = [
            build_random_hybrid_linear_equation_fixture(seed=seed, include_crossout=(seed % 2 == 0))
            for seed in [3, 11, 19, 27]
        ]
        problems = {fixture["problemLatex"] for fixture in fixtures}

        self.assertGreater(len(problems), 1)
        for fixture in fixtures:
            metadata = fixture["problemMetadata"]
            a = metadata["a"]
            b = metadata["b"]
            c_value = metadata["c"]
            solution = metadata["solution"]
            self.assertEqual(metadata["problemSource"], "seeded-random-family")
            self.assertEqual(c_value, a * solution + b)
            self.assertEqual(fixture["problemLatex"], f"{a}x + {b} = {c_value}")
            self.assertEqual(fixture["expectedLatexLines"][0], f"{a}x+{b}={c_value}")
            self.assertEqual(fixture["expectedLatexLines"][-1], f"x={solution}")
            self.assertEqual(metadata["answerLatex"], f"x={solution}")
            self.assertIn("sampledParameters", metadata)

    def test_random_hybrid_linear_fixture_segments_into_expected_rows(self):
        fixture = build_random_hybrid_linear_equation_fixture(seed=31)
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

    def test_hybrid_complex_math_fixture_covers_structural_math_and_annotations(self):
        fixture = build_hybrid_complex_math_fixture(seed=41, include_crossout=True)
        features = classify_features(fixture)

        self.assertEqual(fixture["fixtureKind"], "hybrid-real-stroke-complex-math")
        self.assertEqual(len(fixture["expectedLineGroups"]), 8)
        self.assertIn("\\frac{1}{2}+\\frac{3}{4}", fixture["expectedLatexLines"])
        self.assertIn("x^2=\\frac{9}{4}", fixture["expectedLatexLines"])
        self.assertIn("x=\\pm\\frac{3}{2}", fixture["expectedLatexLines"])
        self.assertIn("\\sqrt{25}", fixture["expectedLatexLines"])
        self.assertIn("(x-1)(x-1)", fixture["expectedLatexLines"])
        self.assertIn("12=4x-4", fixture["expectedLatexLines"])
        self.assertIn("16=4x", fixture["expectedLatexLines"])
        self.assertIn("detached-operation-annotations", fixture["problemMetadata"]["coveredStructures"])
        self.assertTrue(features["has_pressure"])
        self.assertTrue(features["has_multi_stroke_symbols"])
        self.assertTrue(features["has_visual_only_marks"])
        self.assertTrue(features["has_circled_answer"])
        self.assertTrue(features["has_crossout_or_scratch"])
        detached_marks = [
            mark for mark in fixture["visualMarks"]
            if mark.get("type") == "detached_operation_annotation"
        ]
        self.assertEqual(len(detached_marks), 2)
        detached_strokes = [
            stroke for stroke in fixture["strokes"]
            if str(stroke.get("sourceAtomLabel") or "").startswith("detached_operation_")
        ]
        self.assertGreater(len(detached_strokes), 0)
        self.assertTrue({stroke["id"] for stroke in detached_strokes} <= set(fixture["visualOnlyStrokeIds"]))

    def test_hybrid_complex_math_fixture_segments_into_expected_rows(self):
        fixture = build_hybrid_complex_math_fixture(seed=41, include_crossout=True)
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

    def test_log_observed_harness_packs_expose_contracts_and_discrepancy_focus(self):
        expectations = {
            "failure-modes": "audit",
            "problem-input": "problem-input-recognition",
            "ambiguous-fractions": "segmentation",
            "visual-intent": "visual-intent",
            "non-sequential": "grading",
            "bad-handwriting-valid-math": "semantic",
        }
        for pack_name, expected_contract in expectations.items():
            fixture = build_log_observed_pack_fixture(pack_name, seed=13)
            contract_names = {contract["name"] for contract in fixture["oracleContracts"]}

            self.assertEqual(fixture["fixtureKind"], "hybrid-real-stroke-log-observed-pack")
            self.assertEqual(fixture["problemMetadata"]["packName"], pack_name)
            self.assertIn(expected_contract, contract_names)
            self.assertGreater(len(fixture["expectedLineGroups"]), 0)
            self.assertGreater(len(fixture["knownDiscrepancyTypes"]), 0)
            self.assertTrue(fixture["problemMetadata"]["focus"])
            self.assertTrue(classify_features(fixture)["has_pressure"])

    def test_problem_input_pack_marks_problem_input_contract_separately_from_answer_grading(self):
        fixture = build_log_observed_pack_fixture("problem-input", seed=15)

        self.assertTrue(fixture["problemMetadata"]["problemInputMode"])
        contracts = {contract["name"]: contract for contract in fixture["oracleContracts"]}
        self.assertIn("problem-input-recognition", contracts)
        self.assertTrue(contracts["problem-input-recognition"]["problemInput"])
        self.assertTrue(any("problem-input" in slug for slug in fixture["problemMetadata"]["sourceSlugs"]))

    def test_visual_intent_pack_attaches_policy_taxonomy(self):
        fixture = build_log_observed_pack_fixture("visual-intent", seed=17)
        policy_types = {policy["type"] for policy in fixture.get("visualIntentPolicies") or []}

        self.assertIn("circled_answer", policy_types)
        self.assertIn("crossed_out", policy_types)
        self.assertIn("detached_operation_annotation", policy_types)
        for policy in fixture["visualIntentPolicies"]:
            self.assertIn("segmentationPolicy", policy)
            self.assertIn("gradingPolicy", policy)
            self.assertIn("auditPolicy", policy)

    def test_non_sequential_pack_has_temporal_inversions(self):
        fixture = build_log_observed_pack_fixture("non-sequential", seed=19)
        metrics = fixture_metrics(fixture)

        self.assertGreater(metrics["non_sequential_inversions"], 0)
        self.assertTrue(classify_features(fixture)["has_non_sequential_writing"])

    def test_generator_dashboard_compares_generated_fixtures_to_real_distribution(self):
        fixtures = default_dashboard_fixtures(seed=2)
        report = build_harness_dashboard_report(
            fixtures,
            audit_log_dir=Path("/path/that/does/not/exist"),
        )

        self.assertEqual(report["reportKind"], "real-handwriting-generator-dashboard")
        self.assertGreaterEqual(report["generatedCount"], 8)
        self.assertIn("problem-input", report["packNames"])
        self.assertIn("detached_operation_annotation", report["visualIntentPolicyKinds"])
        self.assertEqual(len(report["generatedSummaries"]), report["generatedCount"])
        self.assertEqual(len(report["driftChecks"]), report["generatedCount"])
        self.assertTrue(any(summary["oracleContracts"] for summary in report["generatedSummaries"]))

    def test_log_pack_and_dashboard_cli_surfaces_emit_json(self):
        pack = subprocess.run(
            [
                "python3",
                "testing/realistic_handwriting.py",
                "--hybrid-pack",
                "problem-input",
                "--seed",
                "5",
                "--include-fixture",
            ],
            cwd=str(ROOT),
            text=True,
            check=True,
            capture_output=True,
        )
        pack_payload = json.loads(pack.stdout)
        self.assertEqual(pack_payload["fixture"]["problemMetadata"]["packName"], "problem-input")
        self.assertTrue(pack_payload["fixture"]["oracleContracts"])

        dashboard = subprocess.run(
            [
                "python3",
                "testing/realistic_handwriting.py",
                "--dashboard-only",
                "--seed",
                "5",
                "--audit-log-dir",
                "/path/that/does/not/exist",
            ],
            cwd=str(ROOT),
            text=True,
            check=True,
            capture_output=True,
        )
        dashboard_payload = json.loads(dashboard.stdout)
        self.assertEqual(dashboard_payload["reportKind"], "real-handwriting-generator-dashboard")

    def test_linear_sampler_is_reproducible_and_respects_bounds(self):
        first = sample_linear_equation_parameters(seed=77, max_coefficient=5, max_constant=4, max_solution=6)
        second = sample_linear_equation_parameters(seed=77, max_coefficient=5, max_constant=4, max_solution=6)

        self.assertEqual(first, second)
        self.assertGreaterEqual(first.a, 2)
        self.assertLessEqual(first.a, 5)
        self.assertGreaterEqual(first.b, 1)
        self.assertLessEqual(first.b, 4)
        self.assertGreaterEqual(first.x_value, 1)
        self.assertLessEqual(first.x_value, 6)
        self.assertEqual(first.c_value, first.a * first.x_value + first.b)

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
