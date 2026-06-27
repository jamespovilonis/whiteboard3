import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

import testing.run_live_recognition_matrix as matrix
from testing.run_live_recognition_matrix import (
    RenderedFixture,
    apply_contextual_semantic_scores,
    candidate_recognition_timeout,
    candidate_has_local_fraction_structure,
    clamp_comer_timeout_args,
    fixture_layout_label,
    fixture_run_label,
    fixture_run_slug,
    infer_empty_operation_annotation,
    merge_recognition_attempts,
    ordered_recognizable_candidates,
    payload_needs_retry,
    prepare_initial_recognition_crop,
    recognize_candidate_alternatives,
    recognize_fraction_stroke_chunk,
    recognize_stroke_chunked_candidate,
    repair_operation_annotation_from_previous,
    repair_standalone_operation_latex,
    recognize_crop_with_retries,
    recognize_selected_lines,
    selected_records_from_alternatives,
    selected_problems,
    semantic_needs_retry,
    should_recognize_candidate,
    split_candidate_into_stroke_chunks,
    split_fraction_stroke_chunk,
    compare_latex,
    trusted_semantic_latex,
    parse_rate,
    summarize_totals,
    summary_failures,
    threshold_gate_failures,
)
from testing.fixture_catalog import get_problem


def comparison(match=False, strict=False, loose=False):
    return {
        "match": match,
        "strictMatch": strict,
        "looseMatch": loose,
    }


def line(top_match=False, semantic_match=False, accepted_match=None, index=0):
    if accepted_match is None:
        accepted_match = top_match or semantic_match
    return {
        "lineIndex": index,
        "expectedLatex": "x = 4",
        "topLatex": "x = 4" if top_match else "x = 5",
        "semanticBestLatex": "x = 4" if semantic_match else "x = 5",
        "acceptedLatex": "x = 4" if accepted_match else "x = 5",
        "topComparison": comparison(match=top_match, strict=top_match),
        "semanticComparison": comparison(match=semantic_match, loose=semantic_match),
        "acceptedComparison": comparison(match=accepted_match, strict=accepted_match, loose=accepted_match),
    }


def fake_stroke(stroke_id, x_min, y_min, x_max, y_max):
    return {
        "id": stroke_id,
        "canvasBbox": {"xMin": x_min, "yMin": y_min, "xMax": x_max, "yMax": y_max},
        "rawPoints": [
            {"x": x_min, "y": y_min},
            {"x": x_max, "y": y_min},
            {"x": x_max, "y": y_max},
            {"x": x_min, "y": y_max},
        ],
    }


def record(lines, *, ocr_enabled=True, exact=True):
    return {
        "fixture": {
            "problem": "algebra_steps",
            "spacing": "standard",
            "order": "line-order",
            "expectedLines": 1,
        },
        "segmentation": {
            "exactLineCover": exact,
        },
        "pipelineSelection": {
            "exactLineCover": exact,
            "failures": [] if exact else ["expected 1 selected lines, got 0"],
        },
        "ocr": {
            "enabled": ocr_enabled,
            "lines": list(lines),
        },
    }


class LiveRecognitionMatrixSummaryTests(unittest.TestCase):
    def test_accepts_top_or_semantic_match(self):
        summary = {
            "records": [
                record([line(top_match=True, semantic_match=False)]),
                record([line(top_match=False, semantic_match=True)]),
            ],
        }

        self.assertEqual(summary_failures(summary, require_ocr=True), [])
        totals = summarize_totals(summary["records"])
        self.assertEqual(totals["ocrAcceptedMatches"], 2)
        self.assertEqual(totals["ocrAcceptedStrictMatches"], 2)
        self.assertEqual(totals["ocrMisses"], 0)

    def test_strict_comparison_counts_as_loose_for_operation_rows(self):
        comparison_result = compare_latex(r"\times 6 \times 6", r"\times 6 \times 6")

        self.assertTrue(comparison_result["strictMatch"])
        self.assertTrue(comparison_result["looseMatch"])

    def test_reports_ocr_miss_when_comer_is_required(self):
        summary = {
            "records": [
                record([line(top_match=False, semantic_match=False)]),
            ],
        }

        failures = summary_failures(summary, require_ocr=True)
        self.assertEqual(len(failures), 1)
        self.assertIn("OCR miss", failures[0])
        totals = summarize_totals(summary["records"])
        self.assertEqual(totals["ocrAcceptedMatches"], 0)
        self.assertEqual(totals["ocrMisses"], 1)

    def test_reports_segmentation_failure_even_without_ocr_gate(self):
        failed = record([], ocr_enabled=False, exact=False)
        failed["segmentation"]["exactLineCover"] = False
        failed["segmentation"]["failures"] = ["expected 2 selected lines, got 1"]

        failures = summary_failures({"records": [failed]}, require_ocr=False)

        self.assertTrue(any("expected 2 selected lines" in failure for failure in failures))

    def test_threshold_gates_fail_below_required_rates(self):
        summary = {
            "totals": {
                "fixtures": 2,
                "segmentationExact": 2,
                "pipelineSelectionExact": 1,
                "ocrLines": 4,
                "ocrAcceptedStrictMatches": 3,
                "ocrAcceptedMatches": 4,
            },
        }

        failures = threshold_gate_failures(
            summary,
            min_segmentation_exact_rate=1.0,
            min_pipeline_exact_rate=1.0,
            min_accepted_strict_rate=1.0,
            min_accepted_match_rate=1.0,
        )

        self.assertEqual(len(failures), 2)
        self.assertTrue(any("pipeline selection exact rate" in failure for failure in failures))
        self.assertTrue(any("accepted strict OCR rate" in failure for failure in failures))

    def test_parse_rate_rejects_values_outside_unit_interval(self):
        self.assertEqual(parse_rate("0.75"), 0.75)
        with self.assertRaises(Exception):
            parse_rate("1.5")

    def test_skips_ocr_gate_when_comer_is_disabled(self):
        summary = {
            "records": [
                record([line(top_match=False, semantic_match=False)], ocr_enabled=False),
            ],
        }

        self.assertEqual(summary_failures(summary, require_ocr=True), [])

    def test_pipeline_selection_failure_is_always_reported(self):
        summary = {
            "records": [
                record([], ocr_enabled=False, exact=False),
            ],
        }

        failures = summary_failures(summary, require_ocr=False)
        self.assertEqual(len(failures), 2)
        self.assertTrue(any("segmentation failed" in failure for failure in failures))
        self.assertTrue(any("expected 1 selected lines" in failure for failure in failures))

    def test_comer_timeouts_are_clamped_to_api_limit(self):
        args = Namespace(
            timeout_seconds=30,
            extra_candidate_timeout_seconds=25,
            structural_timeout_seconds=24,
        )

        clamped = clamp_comer_timeout_args(args)

        self.assertEqual(clamped.timeout_seconds, 20)
        self.assertEqual(clamped.extra_candidate_timeout_seconds, 20)
        self.assertEqual(clamped.structural_timeout_seconds, 20)

    def test_default_live_problem_set_includes_symbol_context_fixture(self):
        names = [problem.name for problem in selected_problems([], [], False)]

        self.assertIn("symbol_context_eta", names)

    def test_zero_extra_candidate_limit_recognizes_only_required_candidates(self):
        fixture = RenderedFixture(
            problem=get_problem("algebra_simple"),
            spacing="dense",
            ink_style="messy",
            board=None,
            payload={},
            png_path=Path("unused.png"),
            json_path=Path("unused.json"),
        )
        candidates = [
            {
                "candidateId": "required",
                "profiles": ["row-line"],
                "syntheticLineSets": [0],
                "bbox": {"xMin": 0, "yMin": 0, "xMax": 10, "yMax": 10},
            },
            {
                "candidateId": "extra",
                "profiles": ["row-line"],
                "syntheticLineSets": [1],
                "bbox": {"xMin": 0, "yMin": 20, "xMax": 10, "yMax": 30},
            },
        ]
        seen: list[str] = []
        original_crop = matrix.crop_selected_candidate
        original_recognize = matrix.recognize_crop_with_retries

        def fake_crop(_fixture, _candidate, output_path):
            return output_path

        def fake_recognize(_url, _crop_path, _timeout, **kwargs):
            seen.append(kwargs["candidate"]["candidateId"])
            return {
                "_httpStatus": 200,
                "elapsedSeconds": 0.01,
                "candidates": [{"latex": "2 x + 3 = 11", "score": 0}],
                "top": {"latex": "2 x + 3 = 11"},
            }

        try:
            matrix.crop_selected_candidate = fake_crop
            matrix.recognize_crop_with_retries = fake_recognize
            records = recognize_candidate_alternatives(
                "http://example.test",
                fixture,
                candidates,
                order="interleaved-lines",
                timeout_seconds=1,
                retry_candidate_ids={"required"},
                max_extra_candidates=0,
            )
        finally:
            matrix.crop_selected_candidate = original_crop
            matrix.recognize_crop_with_retries = original_recognize

        self.assertEqual(seen, ["required"])
        self.assertEqual([record["candidateId"] for record in records], ["required"])

    def test_retry_merge_preserves_candidates_from_successful_variants(self):
        self.assertTrue(payload_needs_retry({
            "_httpStatus": 408,
            "timedOut": True,
            "candidates": [],
        }))

        merged = merge_recognition_attempts([
            {
                "_httpStatus": 408,
                "timedOut": True,
                "candidates": [],
            },
            {
                "_httpStatus": 200,
                "_retryTargetPixelHeight": 88,
                "top": {"latex": "f ( x )", "score": 1},
                "candidates": [{"latex": "f ( x )", "score": 1}],
            },
            {
                "_httpStatus": 200,
                "_retryTargetPixelHeight": 104,
                "top": {"latex": "f ^ { \\prime } ( x )", "score": 1},
                "candidates": [{"latex": "f ^ { \\prime } ( x )", "score": 1}],
            },
        ])

        self.assertTrue(merged["retryUsed"])
        self.assertFalse(merged["timedOut"])
        self.assertEqual(
            [candidate["latex"] for candidate in merged["candidates"]],
            ["f ( x )", "f ^ { \\prime } ( x )"],
        )
        self.assertEqual(merged["candidates"][1]["retryTargetPixelHeight"], 104)

    def test_suspicious_operation_annotation_prefers_retry_candidate(self):
        self.assertTrue(payload_needs_retry({
            "_httpStatus": 200,
            "top": {"latex": "\\times 6 9 \\times 6 9"},
            "candidates": [{"latex": "\\times 6 9 \\times 6 9"}],
        }))

        merged = merge_recognition_attempts([
            {
                "_httpStatus": 200,
                "top": {"latex": "\\times 6 9 \\times 6 9"},
                "candidates": [{"latex": "\\times 6 9 \\times 6 9"}],
            },
            {
                "_httpStatus": 200,
                "_retryTargetPixelHeight": 88,
                "top": {"latex": "\\times 6 \\times 6"},
                "candidates": [{"latex": "\\times 6 \\times 6"}],
            },
        ])

        self.assertEqual(merged["top"]["latex"], "\\times 6 \\times 6")
        self.assertEqual(merged["candidates"][1]["retryTargetPixelHeight"], 88)

    def test_ellipsis_truncated_payload_needs_retry(self):
        self.assertTrue(payload_needs_retry({
            "_httpStatus": 200,
            "top": {"latex": "\\cdots + 1 ) - 3 ( x - \\ldots"},
            "candidates": [{"latex": "\\cdots + 1 ) - 3 ( x - \\ldots"}],
        }))

    def test_semantic_retry_targets_short_numeric_mismatches(self):
        candidate = {"tightBbox": {"xMin": 0, "xMax": 120}}

        self.assertTrue(semantic_needs_retry(
            "= 2 0 0",
            {
                "semanticScore": 0.8,
                "equivalentToProblem": False,
                "equivalentToPrevious": False,
            },
            candidate,
        ))
        self.assertFalse(semantic_needs_retry(
            "= 1 0",
            {
                "semanticScore": 4,
                "equivalentToProblem": True,
                "equivalentToPrevious": False,
            },
            candidate,
        ))
        self.assertTrue(semantic_needs_retry(
            "x + 1 = 6",
            {
                "semanticScore": 0.8,
                "equivalentToProblem": False,
                "equivalentToPrevious": False,
            },
            candidate,
        ))
        self.assertFalse(semantic_needs_retry(
            "x + 1 = 6",
            {
                "semanticScore": 3,
                "equivalentToProblem": True,
                "equivalentToPrevious": False,
            },
            candidate,
        ))

    def test_semantic_retry_targets_weak_function_equations(self):
        candidate = {"tightBbox": {"xMin": 0, "xMax": 820}}

        self.assertTrue(semantic_needs_retry(
            "v ( x ) = 2 x ( x + 3 ) + n 2",
            {
                "semanticScore": 0.1,
                "equivalentToProblem": False,
                "equivalentToPrevious": False,
            },
            candidate,
        ))
        self.assertFalse(semantic_needs_retry(
            "f ^ { \\prime } ( x ) = 2 x ( x + 3 ) + x ^ { 2 }",
            {
                "semanticScore": 3.0,
                "equivalentToProblem": True,
                "equivalentToPrevious": False,
            },
            candidate,
        ))

    def test_repairs_standalone_operation_x_as_multiplication(self):
        semantic = {
            "semanticScore": 0.2,
            "equivalentToProblem": False,
            "equivalentToPrevious": False,
        }

        self.assertEqual(
            repair_standalone_operation_latex("x 6 9 \\times 6", semantic),
            "\\times 6 \\times 6",
        )
        self.assertEqual(
            repair_standalone_operation_latex("\\times 1 2 X 1 2", semantic),
            "\\times 12 \\times 12",
        )
        self.assertEqual(
            repair_standalone_operation_latex("x 1 2 x 1 2", semantic),
            "\\times 12 \\times 12",
        )
        self.assertEqual(
            repair_standalone_operation_latex("x + 1 = 6", semantic),
            "",
        )
        self.assertEqual(
            repair_standalone_operation_latex("x 6 \\times 7", semantic),
            "",
        )

    def test_infers_empty_subtraction_annotation_from_previous_equation_geometry(self):
        candidate = {
            "bbox": {"xMin": 220, "yMin": 640, "xMax": 505, "yMax": 700},
            "strokes": [
                fake_stroke("left-minus", 223, 661, 249, 670),
                fake_stroke("left-one", 267, 644, 306, 689),
                fake_stroke("left-zero", 308, 644, 353, 691),
                fake_stroke("right-minus", 374, 669, 400, 678),
                fake_stroke("right-one", 419, 652, 459, 697),
                fake_stroke("right-zero", 461, 652, 505, 698),
            ],
        }

        self.assertEqual(
            infer_empty_operation_annotation(candidate, ["5 x + 10 = 60"]),
            "- 10 - 10",
        )

    def test_empty_operation_inference_requires_previous_additive_constant(self):
        candidate = {
            "bbox": {"xMin": 220, "yMin": 640, "xMax": 505, "yMax": 700},
            "strokes": [
                fake_stroke("left-minus", 223, 661, 249, 670),
                fake_stroke("left-one", 267, 644, 306, 689),
                fake_stroke("right-minus", 374, 669, 400, 678),
                fake_stroke("right-one", 419, 652, 459, 697),
            ],
        }

        self.assertEqual(infer_empty_operation_annotation(candidate, ["5 x = 50"]), "")

    def test_repairs_wrong_subtraction_annotation_from_previous_equation(self):
        self.assertEqual(
            repair_operation_annotation_from_previous("- 2 0 - 2 0", ["5 x + 10 = 60"]),
            "- 10 - 10",
        )
        self.assertEqual(
            repair_operation_annotation_from_previous("- 1 0 - 1 0", ["5 x + 10 = 60"]),
            "",
        )
        self.assertEqual(
            repair_operation_annotation_from_previous("- 2 0 - 2 0", ["5 x = 50"]),
            "",
        )

    def test_trusted_semantic_latex_preserves_operation_annotation_shape(self):
        semantic = {
            "bestLatex": "3 x + 3 = 1 0",
            "semanticScore": 3.2,
            "equivalentToProblem": True,
            "equivalentToPrevious": True,
            "sound": True,
        }

        self.assertEqual(trusted_semantic_latex("- 3 - 3", semantic), "- 3 - 3")

    def test_trusted_semantic_latex_preserves_sound_equivalent_current_row(self):
        semantic = {
            "bestLatex": r"\frac { 1 } { 2 } x + \frac { 1 } { 2 } = \frac { 5 } { 3 }",
            "semanticScore": 3.8,
            "equivalentToProblem": True,
            "equivalentToPrevious": True,
            "sound": True,
            "candidateScores": [
                {
                    "latex": "3 ( x + 1 ) = 10",
                    "sound": True,
                    "equivalentToProblem": True,
                    "equivalentToPrevious": True,
                }
            ],
        }

        self.assertEqual(trusted_semantic_latex("3 ( x + 1 ) = 10", semantic), "3 ( x + 1 ) = 10")

    def test_trusted_semantic_latex_uses_better_contextual_same_kind_candidate(self):
        semantic = {
            "bestLatex": "4 + y = 7",
            "semanticScore": 1.0882,
            "equivalentToProblem": False,
            "equivalentToPrevious": False,
            "sound": True,
            "candidateScores": [
                {
                    "latex": "4 + y = 7",
                    "score": 1.0882,
                    "sound": True,
                    "equivalentToProblem": False,
                    "equivalentToPrevious": False,
                    "detail": {"characterOverlap": 0.461},
                },
                {
                    "latex": "q + y = 7",
                    "score": 1.0829,
                    "sound": True,
                    "equivalentToProblem": False,
                    "equivalentToPrevious": False,
                    "detail": {"characterOverlap": 0.293},
                },
            ],
        }

        self.assertEqual(trusted_semantic_latex("q + y = 7", semantic), "4 + y = 7")

    def test_trusted_semantic_latex_rejects_duplicate_previous_contextual_best(self):
        semantic = {
            "bestLatex": "x = 2",
            "semanticScore": 1.2,
            "equivalentToProblem": False,
            "equivalentToPrevious": False,
            "sound": True,
            "candidateScores": [
                {
                    "latex": "x = 2",
                    "score": 1.2,
                    "sound": True,
                    "equivalentToProblem": False,
                    "equivalentToPrevious": False,
                    "detail": {"characterOverlap": 0.6, "duplicatePreviousLatex": True},
                },
                {
                    "latex": "x = 3",
                    "score": 1.0,
                    "sound": True,
                    "equivalentToProblem": False,
                    "equivalentToPrevious": False,
                    "detail": {"characterOverlap": 0.4},
                },
            ],
        }

        self.assertEqual(trusted_semantic_latex("x = 3", semantic), "x = 3")

    def test_prepare_initial_recognition_crop_normalizes_tall_images(self):
        with TemporaryDirectory() as directory:
            crop_path = Path(directory) / "line.png"
            Image.new("RGB", (240, 180), "white").save(crop_path)

            prepared, target_height = prepare_initial_recognition_crop(crop_path, 104, 128)

            self.assertEqual(target_height, 104)
            self.assertNotEqual(prepared, crop_path)
            self.assertTrue(prepared.name.endswith("_h104.png"))
            with Image.open(prepared) as image:
                self.assertEqual(image.height, 104)

            small_path = Path(directory) / "small.png"
            Image.new("RGB", (120, 90), "white").save(small_path)
            prepared_small, target_small = prepare_initial_recognition_crop(small_path, 104, 128)

            self.assertEqual(prepared_small, small_path)
            self.assertIsNone(target_small)

    def test_wide_crop_tries_chunk_fallback_before_height_retries(self):
        with TemporaryDirectory() as directory:
            crop_path = Path(directory) / "wide.png"
            Image.new("RGB", (900, 140), "white").save(crop_path)
            calls = []
            original_post = matrix.post_image
            original_chunked = matrix.recognize_chunked_crop

            def fake_post(_url, image_path, _timeout_seconds):
                calls.append(("post", Path(image_path).name))
                return {
                    "_httpStatus": 408,
                    "timedOut": True,
                    "top": None,
                    "candidates": [],
                }

            def fake_chunked(_url, image_path, _timeout_seconds, *, target_height, only_if_wide=False):
                calls.append(("chunk", Path(image_path).name, target_height, only_if_wide))
                return {
                    "_httpStatus": 200,
                    "timedOut": False,
                    "top": {"latex": "x = 10"},
                    "candidates": [{"latex": "x = 10"}],
                    "chunkFallback": True,
                }

            try:
                matrix.post_image = fake_post
                matrix.recognize_chunked_crop = fake_chunked
                payload = recognize_crop_with_retries(
                    "http://127.0.0.1:8010/recognize",
                    crop_path,
                    20,
                    retry_on_failure=True,
                    initial_raster_height=104,
                    initial_raster_min_height=1,
                )
            finally:
                matrix.post_image = original_post
                matrix.recognize_chunked_crop = original_chunked

        self.assertTrue(payload["chunkFallback"])
        self.assertTrue(payload["chunkFallbackBeforeHeightRetries"])
        self.assertEqual(calls[0][0], "post")
        self.assertEqual(calls[1][0], "chunk")
        self.assertEqual(len(calls), 2)

    def test_structural_crop_gets_longer_timeout_after_short_retries_fail(self):
        with TemporaryDirectory() as directory:
            crop_path = Path(directory) / "fraction.png"
            Image.new("RGB", (320, 130), "white").save(crop_path)
            calls = []
            original_post = matrix.post_image
            original_chunked = matrix.recognize_chunked_crop

            def fake_post(url, image_path, timeout_seconds):
                calls.append((Path(image_path).name, timeout_seconds, url))
                if timeout_seconds > 6:
                    return {
                        "_httpStatus": 200,
                        "timedOut": False,
                        "top": {"latex": r"\frac { x } { 2 } = 1"},
                        "candidates": [{"latex": r"\frac { x } { 2 } = 1"}],
                    }
                return {
                    "_httpStatus": 408,
                    "timedOut": True,
                    "top": None,
                    "candidates": [],
                }

            try:
                matrix.post_image = fake_post
                matrix.recognize_chunked_crop = lambda *_args, **_kwargs: None
                payload = recognize_crop_with_retries(
                    "http://127.0.0.1:8010/recognize?model=comer&timeout_seconds=6",
                    crop_path,
                    6,
                    retry_on_failure=True,
                    initial_raster_height=88,
                    initial_raster_min_height=1,
                    candidate={
                        "profiles": ["row-line"],
                        "bbox": {"xMin": 0, "yMin": 0, "xMax": 300, "yMax": 105},
                        "strokes": [fake_stroke(f"s{i}", i * 20, 0, i * 20 + 12, 90) for i in range(10)],
                    },
                    extended_timeout_seconds=12,
                )
            finally:
                matrix.post_image = original_post
                matrix.recognize_chunked_crop = original_chunked

        self.assertEqual(payload["top"]["latex"], r"\frac { x } { 2 } = 1")
        self.assertTrue(any(timeout == 12 for _name, timeout, _url in calls))
        self.assertTrue(any("timeout_seconds=12" in url for _name, _timeout, url in calls))
        self.assertEqual(payload["retryAttempts"][-1]["extendedTimeoutSeconds"], 12)

    def test_failed_stroke_chunk_fallback_is_not_repeated_after_height_retries(self):
        with TemporaryDirectory() as directory:
            crop_path = Path(directory) / "fraction.png"
            Image.new("RGB", (360, 140), "white").save(crop_path)
            chunk_calls = []
            original_post = matrix.post_image
            original_stroke_chunked = matrix.recognize_stroke_chunked_candidate
            original_image_chunked = matrix.recognize_chunked_crop

            def fake_post(_url, _image_path, _timeout_seconds):
                return {
                    "_httpStatus": 408,
                    "timedOut": True,
                    "top": None,
                    "candidates": [],
                }

            def fake_stroke_chunked(*_args, **_kwargs):
                chunk_calls.append("stroke")
                return {
                    "_httpStatus": 408,
                    "timedOut": True,
                    "failed": True,
                    "chunkFallback": True,
                    "chunkFallbackSource": "stroke",
                    "candidates": [],
                    "top": None,
                }

            try:
                matrix.post_image = fake_post
                matrix.recognize_stroke_chunked_candidate = fake_stroke_chunked
                matrix.recognize_chunked_crop = lambda *_args, **_kwargs: None
                payload = recognize_crop_with_retries(
                    "http://127.0.0.1:8010/recognize",
                    crop_path,
                    20,
                    retry_on_failure=True,
                    initial_raster_height=104,
                    initial_raster_min_height=1,
                    candidate={
                        "profiles": ["row-line"],
                        "bbox": {"xMin": 0, "yMin": 0, "xMax": 340, "yMax": 120},
                        "strokes": [fake_stroke(f"s{i}", i * 20, 0, i * 20 + 12, 90) for i in range(10)],
                    },
                )
            finally:
                matrix.post_image = original_post
                matrix.recognize_stroke_chunked_candidate = original_stroke_chunked
                matrix.recognize_chunked_crop = original_image_chunked

        self.assertTrue(payload["timedOut"])
        self.assertEqual(chunk_calls, ["stroke"])

    def test_stroke_chunk_fallback_infers_context_variable_before_equals(self):
        candidate = {
            "candidateId": "solve-line",
            "profiles": ["row-line"],
            "bbox": {"xMin": 0, "yMin": 0, "xMax": 520, "yMax": 90},
            "strokes": [
                fake_stroke("x", 0, 20, 28, 58),
                fake_stroke("eq-top", 48, 28, 92, 34),
                fake_stroke("eq-bottom", 48, 50, 92, 56),
                fake_stroke("rhs", 125, 8, 520, 86),
            ],
        }
        fixture = RenderedFixture(
            problem=get_problem("quadratic_formula_positive_root"),
            spacing="dense",
            ink_style="messy",
            board=None,
            payload={},
            png_path=Path("fixture.png"),
            json_path=Path("fixture.json"),
        )
        calls = []
        original_post = matrix.post_image

        def fake_post(_url, image_path, _timeout_seconds):
            calls.append(Path(image_path).name)
            return {
                "_httpStatus": 200,
                "top": {"latex": r"\frac { 1 } { 2 }"},
                "candidates": [{"latex": r"\frac { 1 } { 2 }"}],
            }

        try:
            matrix.post_image = fake_post
            payload = recognize_stroke_chunked_candidate(
                "http://127.0.0.1:8010/recognize",
                fixture,
                candidate,
                20,
                target_height=104,
            )
        finally:
            matrix.post_image = original_post

        self.assertEqual(payload["top"]["latex"], r"x = \frac { 1 } { 2 }")
        self.assertEqual(len(calls), 1)
        self.assertTrue(payload["chunkAttempts"][0]["inferredLiteral"])
        self.assertEqual(payload["chunkAttempts"][0]["literalLatex"], "x")

    def test_stroke_chunker_splits_wide_candidate_and_preserves_equals(self):
        candidate = {
            "candidateId": "wide-line",
            "bbox": {"xMin": 0, "yMin": 0, "xMax": 760, "yMax": 80},
            "strokes": [
                fake_stroke("left", 0, 0, 180, 70),
                fake_stroke("eq-top", 210, 24, 260, 30),
                fake_stroke("eq-bottom", 210, 48, 260, 54),
                fake_stroke("right-a", 300, 0, 520, 70),
                fake_stroke("right-b", 560, 0, 760, 70),
            ],
        }

        chunks = split_candidate_into_stroke_chunks(candidate, max_chunk_width=280)

        self.assertEqual([chunk.get("literalLatex") for chunk in chunks], [None, "=", None, None])
        self.assertEqual(chunks[0]["strokeIds"], ["left"])
        self.assertEqual(chunks[2]["strokeIds"], ["right-a"])
        self.assertEqual(chunks[3]["strokeIds"], ["right-b"])

    def test_fraction_stroke_chunk_splitter_finds_numerator_and_denominator(self):
        chunk = {
            "strokeIds": ["num-x", "minus", "one", "bar", "den-x", "den-two"],
            "strokes": [
                fake_stroke("num-x", 10, 0, 38, 34),
                fake_stroke("minus", 48, 18, 84, 24),
                fake_stroke("one", 96, 0, 120, 34),
                fake_stroke("bar", 0, 48, 150, 56),
                fake_stroke("den-x", 44, 72, 76, 108),
                fake_stroke("den-two", 92, 72, 126, 108),
            ],
            "bbox": {"xMin": 0, "yMin": 0, "xMax": 150, "yMax": 108},
        }

        split = split_fraction_stroke_chunk(chunk)

        self.assertIsNotNone(split)
        self.assertEqual(split["bar"]["id"], "bar")
        self.assertEqual(split["numerator"]["strokeIds"], ["num-x", "minus", "one"])
        self.assertEqual(split["denominator"]["strokeIds"], ["den-x", "den-two"])

    def test_stroke_chunker_keeps_compact_neighboring_fractions_separate(self):
        candidate = {
            "candidateId": "compact-fractions",
            "bbox": {"xMin": 123, "yMin": 68, "xMax": 353, "yMax": 157},
            "profiles": ["loose", "strict", "row-line"],
            "strokes": [
                fake_stroke("left-num", 125, 76, 147, 99),
                fake_stroke("plus", 170, 84, 210, 132),
                fake_stroke("right-num", 230, 68, 244, 100),
                fake_stroke("equals-top", 269, 101, 309, 105),
                fake_stroke("left-bar", 123, 112, 149, 117),
                fake_stroke("right-bar", 226, 113, 248, 118),
                fake_stroke("equals-bottom", 269, 115, 309, 119),
                fake_stroke("left-den", 127, 121, 144, 154),
                fake_stroke("right-den", 227, 123, 245, 157),
                fake_stroke("rhs", 327, 81, 353, 129),
            ],
        }

        chunks = split_candidate_into_stroke_chunks(candidate, min_gap=12, max_chunk_width=140)

        self.assertEqual(
            [chunk.get("strokeIds") for chunk in chunks],
            [
                ["left-bar", "left-num", "left-den"],
                ["plus"],
                ["right-bar", "right-den", "right-num"],
                ["equals-top", "equals-bottom"],
                ["rhs"],
            ],
        )
        self.assertEqual(chunks[3].get("literalLatex"), "=")
        self.assertIsNotNone(split_fraction_stroke_chunk(chunks[0]))
        self.assertIsNotNone(split_fraction_stroke_chunk(chunks[2]))

    def test_tall_linear_candidate_is_not_compact_fraction_structure(self):
        linear_candidate = {
            "candidateId": "linear-row",
            "bbox": {"xMin": 160, "yMin": 168, "xMax": 770, "yMax": 252},
            "strokes": [
                fake_stroke("x", 160, 168, 207, 204),
                fake_stroke("minus", 229, 182, 255, 191),
                fake_stroke("one", 273, 169, 313, 215),
                fake_stroke("eq-top", 348, 181, 397, 195),
                fake_stroke("eq-bottom", 345, 196, 394, 210),
                fake_stroke("five", 418, 187, 481, 234),
                fake_stroke("left-paren", 500, 182, 530, 252),
                fake_stroke("rhs-x", 531, 207, 578, 252),
                fake_stroke("plus", 608, 203, 657, 252),
                fake_stroke("rhs-one", 684, 222, 716, 252),
                fake_stroke("right-paren", 755, 213, 770, 252),
            ],
        }

        self.assertFalse(candidate_has_local_fraction_structure(linear_candidate))

    def test_fraction_stroke_chunk_recognizes_parts_when_whole_chunk_times_out(self):
        chunk = {
            "strokeIds": ["num", "bar", "den"],
            "strokes": [
                fake_stroke("num", 20, 0, 70, 32),
                fake_stroke("bar", 0, 46, 120, 52),
                fake_stroke("den", 24, 70, 76, 104),
            ],
            "bbox": {"xMin": 0, "yMin": 0, "xMax": 120, "yMax": 104},
        }
        original_post = matrix.post_image

        def fake_post(_url, image_path, _timeout_seconds):
            name = Path(image_path).name
            latex = "x ^ { 2 } - 1" if "numerator" in name else "x ^ { 2 }"
            return {
                "_httpStatus": 200,
                "timedOut": False,
                "top": {"latex": latex},
                "candidates": [{"latex": latex}],
            }

        try:
            matrix.post_image = fake_post
            latex, attempt = recognize_fraction_stroke_chunk(
                "http://127.0.0.1:8010/recognize",
                chunk,
                6,
                target_height=88,
            )
        finally:
            matrix.post_image = original_post

        self.assertEqual(latex, r"\frac { x ^ { 2 } - 1 } { x ^ { 2 } }")
        self.assertTrue(attempt["fractionSubchunk"])
        self.assertEqual(len(attempt["parts"]), 2)

    def test_wide_crop_prefers_stroke_chunk_fallback_before_image_chunking(self):
        with TemporaryDirectory() as directory:
            crop_path = Path(directory) / "wide.png"
            Image.new("RGB", (900, 140), "white").save(crop_path)
            candidate = {
                "candidateId": "wide-line",
                "bbox": {"xMin": 0, "yMin": 0, "xMax": 760, "yMax": 80},
                "strokes": [
                    fake_stroke("left", 0, 0, 180, 70),
                    fake_stroke("eq-top", 210, 24, 260, 30),
                    fake_stroke("eq-bottom", 210, 48, 260, 54),
                    fake_stroke("right", 300, 0, 760, 70),
                ],
            }
            calls = []
            original_post = matrix.post_image
            original_image_chunked = matrix.recognize_chunked_crop

            def fake_post(_url, image_path, _timeout_seconds):
                name = Path(image_path).name
                calls.append(("post", name))
                if "_chunk_" in name:
                    latex = "x" if "chunk_1" in name else "10"
                    return {
                        "_httpStatus": 200,
                        "top": {"latex": latex},
                        "candidates": [{"latex": latex}],
                    }
                return {
                    "_httpStatus": 408,
                    "timedOut": True,
                    "top": None,
                    "candidates": [],
                }

            def fake_image_chunked(*_args, **_kwargs):
                calls.append(("image-chunk",))
                return None

            try:
                matrix.post_image = fake_post
                matrix.recognize_chunked_crop = fake_image_chunked
                payload = recognize_crop_with_retries(
                    "http://127.0.0.1:8010/recognize",
                    crop_path,
                    20,
                    retry_on_failure=True,
                    initial_raster_height=104,
                    initial_raster_min_height=1,
                    fixture=RenderedFixture(
                        problem=get_problem("algebra_simple"),
                        spacing="dense",
                        ink_style="normal",
                        board=None,
                        payload={},
                        png_path=Path("fixture.png"),
                        json_path=Path("fixture.json"),
                    ),
                    candidate=candidate,
                )
            finally:
                matrix.post_image = original_post
                matrix.recognize_chunked_crop = original_image_chunked

        self.assertTrue(payload["chunkFallback"])
        self.assertEqual(payload["chunkFallbackSource"], "stroke")
        self.assertTrue(payload["chunkFallbackBeforeHeightRetries"])
        self.assertNotIn(("image-chunk",), calls)
        self.assertEqual(payload["top"]["latex"], "x = 10")

    def test_compact_structural_candidate_can_use_stroke_chunk_fallback(self):
        with TemporaryDirectory() as directory:
            crop_path = Path(directory) / "compact.png"
            Image.new("RGB", (320, 140), "white").save(crop_path)
            candidate = {
                "candidateId": "compact-fraction",
                "profiles": ["row-line"],
                "bbox": {"xMin": 0, "yMin": 0, "xMax": 310, "yMax": 105},
                "strokes": [
                    fake_stroke("left-num", 0, 10, 30, 45),
                    fake_stroke("left-bar", 0, 58, 90, 64),
                    fake_stroke("left-den", 0, 75, 30, 105),
                    fake_stroke("plus", 105, 42, 130, 70),
                    fake_stroke("right-num", 155, 10, 185, 45),
                    fake_stroke("right-bar", 150, 58, 230, 64),
                    fake_stroke("right-den", 155, 75, 185, 105),
                    fake_stroke("eq-top", 245, 44, 270, 50),
                    fake_stroke("eq-bottom", 245, 62, 270, 68),
                    fake_stroke("rhs", 285, 20, 310, 85),
                ],
            }
            original_post = matrix.post_image
            original_image_chunked = matrix.recognize_chunked_crop

            def fake_post(_url, image_path, _timeout_seconds):
                name = Path(image_path).name
                if "_chunk_" in name:
                    return {
                        "_httpStatus": 200,
                        "top": {"latex": "x"},
                        "candidates": [{"latex": "x"}],
                    }
                return {
                    "_httpStatus": 408,
                    "timedOut": True,
                    "top": None,
                    "candidates": [],
                }

            try:
                matrix.post_image = fake_post
                matrix.recognize_chunked_crop = lambda *_args, **_kwargs: None
                payload = recognize_crop_with_retries(
                    "http://127.0.0.1:8010/recognize",
                    crop_path,
                    20,
                    retry_on_failure=True,
                    initial_raster_height=88,
                    initial_raster_min_height=1,
                    fixture=RenderedFixture(
                        problem=get_problem("rational_two_fraction_solve"),
                        spacing="tight-steps",
                        ink_style="messy",
                        board=None,
                        payload={},
                        png_path=Path("fixture.png"),
                        json_path=Path("fixture.json"),
                    ),
                    candidate=candidate,
                )
            finally:
                matrix.post_image = original_post
                matrix.recognize_chunked_crop = original_image_chunked

        self.assertTrue(payload["chunkFallback"])
        self.assertEqual(payload["chunkFallbackSource"], "stroke")
        self.assertGreaterEqual(len(payload["chunkAttempts"]), 2)

    def test_candidate_alternative_order_prioritizes_selected_rows(self):
        candidates = [
            {"candidateId": "projection", "profiles": ["projection-line"], "bbox": {"yMin": 0}},
            {"candidateId": "fraction", "profiles": ["fraction-stack-line"], "bbox": {"yMin": 10}},
            {"candidateId": "row", "profiles": ["raw-row-line"], "bbox": {"yMin": 20}},
            {"candidateId": "selected", "profiles": ["parent"], "bbox": {"yMin": 80}},
            {"candidateId": "fallback", "profiles": ["fallback-stroke"], "bbox": {"yMin": 40}},
        ]

        ordered = ordered_recognizable_candidates(candidates, {"selected"})
        self.assertEqual([candidate["candidateId"] for _, candidate in ordered], [
            "selected",
            "fraction",
            "row",
            "projection",
        ])

    def test_extra_candidate_timeout_never_shortens_selected_lines(self):
        self.assertEqual(
            candidate_recognition_timeout(
                20,
                required=True,
                extra_candidate_timeout_seconds=3,
                remaining_extra_budget_seconds=1,
            ),
            20,
        )
        self.assertEqual(
            candidate_recognition_timeout(
                20,
                required=False,
                extra_candidate_timeout_seconds=3,
                remaining_extra_budget_seconds=10,
            ),
            3,
        )
        self.assertEqual(
            candidate_recognition_timeout(
                20,
                required=False,
                extra_candidate_timeout_seconds=8,
                remaining_extra_budget_seconds=2,
            ),
            2,
        )

    def test_contextual_semantic_scores_use_prior_answer_lines(self):
        alternatives = [
            {
                "candidateId": "derivative-line",
                "profiles": ["row-line"],
                "tightBbox": {"xMin": 0, "xMax": 300, "yMin": 0, "yMax": 24},
                "topLatex": r"f ^ { \prime } ( x ) = 2 x",
                "semanticBestLatex": r"f ^ { \prime } ( x ) = 2 x",
                "semanticScore": 1.0,
                "semantic": {"bestLatex": r"f ^ { \prime } ( x ) = 2 x"},
                "topCandidates": [{"latex": r"f ^ { \prime } ( x ) = 2 x", "score": -0.2}],
                "expectedLatex": r"f ^ { \prime } ( x ) = 2 x",
            },
            {
                "candidateId": "evaluation-line",
                "profiles": ["row-line"],
                "tightBbox": {"xMin": 0, "xMax": 300, "yMin": 42, "yMax": 66},
                "topLatex": r"f ^ { \prime } ( 2 ) = 4",
                "semanticBestLatex": r"f ^ { \prime } ( 2 ) = 4",
                "semanticScore": 0.5,
                "semantic": {"bestLatex": r"f ^ { \prime } ( 2 ) = 4"},
                "topCandidates": [{"latex": r"f ^ { \prime } ( 2 ) = 4", "score": -0.3}],
                "expectedLatex": r"f ^ { \prime } ( 2 ) = 4",
            },
        ]

        rescored = apply_contextual_semantic_scores(
            alternatives,
            problem_latex=r"f ( x ) = x ^ { 2 }",
        )
        evaluation = next(record for record in rescored if record["candidateId"] == "evaluation-line")

        self.assertGreater(evaluation["semanticScore"], evaluation["baseSemanticScore"])
        self.assertTrue(evaluation["contextualSemantic"]["equivalentToPrevious"])
        self.assertEqual(
            evaluation["contextualSemantic"]["sameAnswerContext"],
            [r"f ^ { \prime } ( x ) = 2 x"],
        )

    def test_selected_alternative_records_apply_semantic_retry(self):
        with TemporaryDirectory() as directory:
            crop_path = Path(directory) / "line.png"
            Image.new("RGB", (300, 80), "white").save(crop_path)
            fixture = RenderedFixture(
                problem=get_problem("derivative_product_evaluate"),
                spacing="dense",
                ink_style="messy",
                board=None,
                payload={},
                png_path=Path("fixture.png"),
                json_path=Path("fixture.json"),
            )
            selected = [{
                "candidateId": "line-2",
                "profiles": ["row-line"],
                "syntheticLineSets": [1],
                "syntheticLatex": [r"f ^ { \prime } ( x ) = 2 x ( x + 3 ) + x ^ { 2 }"],
                "tightBbox": {"xMin": 0, "xMax": 820},
            }]
            alternatives = [{
                "candidateId": "line-2",
                "profiles": ["row-line"],
                "sourceLineIndexes": [1],
                "expectedLatex": r"f ^ { \prime } ( x ) = 2 x ( x + 3 ) + x ^ { 2 }",
                "crop": str(crop_path),
                "topLatex": r"v ( x ) = 2 x ( x + 3 ) + n 2",
                "semanticBestLatex": r"v ( x ) = 2 x ( x + 3 ) + n 2",
                "topCandidates": [{"latex": r"v ( x ) = 2 x ( x + 3 ) + n 2", "score": -0.2}],
                "semantic": {
                    "bestLatex": r"v ( x ) = 2 x ( x + 3 ) + n 2",
                    "semanticScore": 0.1,
                    "equivalentToProblem": False,
                    "equivalentToPrevious": False,
                },
                "topComparison": comparison(loose=True),
                "semanticComparison": comparison(loose=True),
            }]
            calls = []
            original_retry = matrix.recognize_crop_with_semantic_retries
            original_needs_retry = matrix.semantic_needs_retry

            def fake_retry(*args, **kwargs):
                calls.append((args, kwargs))
                return {
                    "_httpStatus": 200,
                    "top": {"latex": r"f ^ { \prime } ( x ) = 2 x ( x + 3 ) + x ^ { 2 }"},
                    "candidates": [{"latex": r"f ^ { \prime } ( x ) = 2 x ( x + 3 ) + x ^ { 2 }"}],
                    "elapsedSeconds": 0.1,
                    "retryUsed": True,
                    "retryAttempts": [{"retryTargetPixelHeight": 48}],
                }

            try:
                matrix.semantic_needs_retry = lambda *args, **kwargs: True
                matrix.recognize_crop_with_semantic_retries = fake_retry
                records = selected_records_from_alternatives(
                    selected,
                    alternatives,
                    api_url="http://127.0.0.1:8010",
                    fixture=fixture,
                    timeout_seconds=20,
                    initial_raster_height=104,
                )
            finally:
                matrix.recognize_crop_with_semantic_retries = original_retry
                matrix.semantic_needs_retry = original_needs_retry

        self.assertEqual(len(calls), 1)
        self.assertTrue(records[0]["semanticRetryUsed"])
        self.assertEqual(
            records[0]["topLatex"],
            r"f ^ { \prime } ( x ) = 2 x ( x + 3 ) + x ^ { 2 }",
        )
        self.assertEqual(
            records[0]["acceptedLatex"],
            r"f ^ { \prime } ( x ) = 2 x ( x + 3 ) + x ^ { 2 }",
        )
        self.assertTrue(records[0]["topComparison"]["strictMatch"])
        self.assertTrue(records[0]["acceptedComparison"]["strictMatch"])

    def test_selected_alternative_empty_ocr_uses_full_retry_recovery(self):
        with TemporaryDirectory() as directory:
            crop_path = Path(directory) / "line.png"
            Image.new("RGB", (360, 120), "white").save(crop_path)
            fixture = RenderedFixture(
                problem=get_problem("rational_two_fraction_solve"),
                spacing="dense",
                ink_style="messy",
                board=None,
                payload={},
                png_path=Path("fixture.png"),
                json_path=Path("fixture.json"),
            )
            selected = [{
                "candidateId": "fraction-line",
                "profiles": ["row-line", "dbnet-line"],
                "syntheticLineSets": [0],
                "syntheticLatex": [r"\frac { x } { 2 } + \frac { 1 } { 3 } = 2"],
                "strokes": [
                    fake_stroke("num", 20, 0, 40, 32),
                    fake_stroke("bar", 0, 46, 90, 52),
                    fake_stroke("den", 22, 70, 44, 104),
                ],
                "bbox": {"xMin": 0, "yMin": 0, "xMax": 90, "yMax": 104},
            }]
            alternatives = [{
                "candidateId": "fraction-line",
                "profiles": ["row-line", "dbnet-line"],
                "sourceLineIndexes": [0],
                "expectedLatex": r"\frac { x } { 2 } + \frac { 1 } { 3 } = 2",
                "crop": str(crop_path),
                "httpStatus": 408,
                "topLatex": "",
                "semanticBestLatex": "",
                "topCandidates": [],
                "semantic": {
                    "bestLatex": "",
                    "semanticScore": -1000,
                    "equivalentToProblem": False,
                    "equivalentToPrevious": False,
                },
                "topComparison": comparison(),
                "semanticComparison": comparison(),
            }]
            calls = []
            original_retry = matrix.recognize_crop_with_retries

            def fake_retry(*args, **kwargs):
                calls.append((args, kwargs))
                self.assertIs(kwargs["fixture"], fixture)
                self.assertIs(kwargs["candidate"], selected[0])
                self.assertEqual(kwargs["extended_timeout_seconds"], 12)
                return {
                    "_httpStatus": 200,
                    "top": {"latex": r"\frac { x } { 2 } + \frac { 1 } { 3 } = 2"},
                    "candidates": [{"latex": r"\frac { x } { 2 } + \frac { 1 } { 3 } = 2"}],
                    "elapsedSeconds": 0.4,
                    "chunkFallback": True,
                    "chunkAttempts": [{"strokeIds": ["num", "bar", "den"]}],
                }

            try:
                matrix.recognize_crop_with_retries = fake_retry
                records = selected_records_from_alternatives(
                    selected,
                    alternatives,
                    api_url="http://127.0.0.1:8010",
                    fixture=fixture,
                    timeout_seconds=6,
                    initial_raster_height=88,
                    structural_timeout_seconds=12,
                )
            finally:
                matrix.recognize_crop_with_retries = original_retry

        self.assertEqual(len(calls), 1)
        self.assertTrue(records[0]["chunkFallback"])
        self.assertEqual(
            records[0]["acceptedLatex"],
            r"\frac { x } { 2 } + \frac { 1 } { 3 } = 2",
        )
        self.assertTrue(records[0]["acceptedComparison"]["strictMatch"])

    def test_selected_alternative_records_accept_subscripted_operation_repair(self):
        with TemporaryDirectory() as directory:
            crop_path = Path(directory) / "line.png"
            Image.new("RGB", (300, 80), "white").save(crop_path)
            fixture = RenderedFixture(
                problem=get_problem("rational_two_fraction_solve"),
                spacing="tight-steps",
                ink_style="messy",
                board=None,
                payload={},
                png_path=Path("fixture.png"),
                json_path=Path("fixture.json"),
            )
            selected = [{
                "candidateId": "operation-line",
                "profiles": ["row-line"],
                "syntheticLineSets": [1],
                "syntheticLatex": [r"\times 6       \times 6"],
                "tightBbox": {"xMin": 0, "xMax": 820},
            }]
            alternatives = [{
                "candidateId": "operation-line",
                "profiles": ["row-line"],
                "sourceLineIndexes": [1],
                "expectedLatex": r"\times 6       \times 6",
                "crop": str(crop_path),
                "topLatex": r"X _ { 6 } \times _ { n }",
                "semanticBestLatex": r"X _ { 6 } \times _ { n }",
                "topCandidates": [{"latex": r"X _ { 6 } \times _ { n }", "score": -0.7}],
                "semantic": {
                    "bestLatex": r"X _ { 6 } \times _ { n }",
                    "semanticScore": 0.4,
                    "equivalentToProblem": False,
                    "equivalentToPrevious": False,
                },
                "topComparison": comparison(),
                "semanticComparison": comparison(),
            }]

            original_needs_retry = matrix.semantic_needs_retry
            try:
                matrix.semantic_needs_retry = lambda *args, **kwargs: False
                records = selected_records_from_alternatives(
                    selected,
                    alternatives,
                    api_url="http://127.0.0.1:8010",
                    fixture=fixture,
                    timeout_seconds=20,
                    initial_raster_height=104,
                )
            finally:
                matrix.semantic_needs_retry = original_needs_retry

        self.assertEqual(records[0]["acceptedLatex"], r"\times 6 \times 6")
        self.assertEqual(records[0]["semanticBestLatex"], r"\times 6 \times 6")
        self.assertEqual(records[0]["ocrRepair"]["source"], "standalone-operation")
        self.assertTrue(records[0]["acceptedComparison"]["strictMatch"])

    def test_structural_chunking_uses_tighter_gaps_after_timeouts(self):
        candidate = {
            "candidateId": "derivative-quotient",
            "profiles": ["strict", "row-line"],
            "bbox": {"xMin": 160, "yMin": 208, "xMax": 517, "yMax": 322},
            "strokes": [
                fake_stroke("f", 160, 208, 190, 294),
                fake_stroke("paren", 202, 212, 230, 292),
                fake_stroke("x", 240, 225, 306, 280),
                fake_stroke("eq-top", 323, 249, 373, 255),
                fake_stroke("eq-bottom", 323, 266, 373, 272),
                fake_stroke("num-left", 384, 212, 424, 244),
                fake_stroke("num-right", 438, 214, 477, 244),
                fake_stroke("bar", 384, 252, 517, 258),
                fake_stroke("den-left", 402, 274, 438, 322),
                fake_stroke("den-right", 452, 276, 493, 322),
            ],
        }

        ordinary_chunks = split_candidate_into_stroke_chunks(candidate, min_gap=18, max_chunk_width=340)
        structural_chunks = split_candidate_into_stroke_chunks(candidate, min_gap=8, max_chunk_width=340)

        self.assertFalse(any(chunk.get("literalLatex") == "=" for chunk in ordinary_chunks))
        self.assertEqual(len(structural_chunks), 3)
        self.assertEqual(structural_chunks[1]["literalLatex"], "=")

    def test_selected_line_recognition_checkpoints_after_each_line(self):
        with TemporaryDirectory() as directory:
            fixture = RenderedFixture(
                problem=get_problem("algebra_simple"),
                spacing="dense",
                ink_style="normal",
                board=None,
                payload={},
                png_path=Path("fixture.png"),
                json_path=Path("fixture.json"),
            )
            selected = [
                {
                    "candidateId": "line-1",
                    "profiles": ["row-line"],
                    "syntheticLineSets": [0],
                    "tightBbox": {"xMin": 0, "xMax": 300},
                },
                {
                    "candidateId": "line-2",
                    "profiles": ["row-line"],
                    "syntheticLineSets": [1],
                    "tightBbox": {"xMin": 0, "xMax": 300},
                },
            ]
            checkpoints = []
            payloads = [
                {"top": {"latex": r"2 x + 3 = 11"}, "candidates": [{"latex": r"2 x + 3 = 11"}]},
                {"top": {"latex": r"2 x = 8"}, "candidates": [{"latex": r"2 x = 8"}]},
            ]
            original_crop = matrix.crop_selected_candidate
            original_recognize = matrix.recognize_crop_with_retries
            original_needs_retry = matrix.semantic_needs_retry

            def fake_crop(_fixture, _candidate, output_path):
                output_path = Path(directory) / Path(output_path).name
                Image.new("RGB", (240, 80), "white").save(output_path)
                return output_path

            def fake_recognize(*_args, **_kwargs):
                return payloads.pop(0)

            try:
                matrix.crop_selected_candidate = fake_crop
                matrix.recognize_crop_with_retries = fake_recognize
                matrix.semantic_needs_retry = lambda *args, **kwargs: False
                records = recognize_selected_lines(
                    "http://127.0.0.1:8010",
                    fixture,
                    selected,
                    order="line-order",
                    timeout_seconds=20,
                    checkpoint_callback=lambda lines: checkpoints.append([line["lineIndex"] for line in lines]),
                )
            finally:
                matrix.crop_selected_candidate = original_crop
                matrix.recognize_crop_with_retries = original_recognize
                matrix.semantic_needs_retry = original_needs_retry

        self.assertEqual([record["lineIndex"] for record in records], [0, 1])
        self.assertEqual(checkpoints, [[0], [0, 1]])

    def test_live_runner_recognizes_new_segmentation_profiles(self):
        self.assertTrue(should_recognize_candidate({"profiles": ["raw-row-line"]}))
        self.assertTrue(should_recognize_candidate({"profiles": ["fraction-stack-line"]}))
        self.assertTrue(should_recognize_candidate({"profiles": ["projection-line"]}))
        self.assertFalse(should_recognize_candidate({"profiles": ["fallback-stroke"]}))

    def test_fixture_run_names_include_ink_style_and_order(self):
        fixture = RenderedFixture(
            problem=get_problem("algebra_steps"),
            spacing="mixed",
            ink_style="compact",
            board=None,
            payload={},
            png_path=Path("fixture.png"),
            json_path=Path("fixture.json"),
        )

        self.assertEqual(
            fixture_run_label(fixture, "interleaved-lines"),
            "algebra_steps/mixed/compact/interleaved-lines",
        )
        self.assertEqual(
            fixture_run_slug(fixture, "interleaved-lines"),
            "algebra-steps_mixed_compact_interleaved-lines",
        )

    def test_fixture_run_names_include_gap_pattern_when_present(self):
        fixture = RenderedFixture(
            problem=get_problem("algebra_steps"),
            spacing="tight-steps",
            ink_style="messy",
            board=None,
            payload={},
            png_path=Path("fixture.png"),
            json_path=Path("fixture.json"),
            gap_pattern="accordion",
            line_gaps=(0, 72, 6, 54),
        )

        self.assertEqual(fixture_layout_label(fixture), "tight-steps:accordion")
        self.assertEqual(
            fixture_run_label(fixture, "reverse-lines"),
            "algebra_steps/tight-steps:accordion/messy/reverse-lines",
        )
        self.assertEqual(
            fixture_run_slug(fixture, "reverse-lines"),
            "algebra-steps_tight-steps-accordion_messy_reverse-lines",
        )


if __name__ == "__main__":
    unittest.main()
