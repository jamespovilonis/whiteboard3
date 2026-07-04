#!/usr/bin/env python3
"""Real-user calibrated stroke-level handwriting fixtures.

This module complements ``synthetic_handwriting.py``. Instead of rendering
LaTeX to pixels and extracting contours, it builds fixtures by transforming
real distilled audit strokes: raw point trajectories, pressure, timing, visual
marks, and reviewed semantic groups are preserved and lightly jittered.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Optional, Sequence


TESTING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTING_DIR.parent
DEFAULT_REAL_FIXTURE_DIR = TESTING_DIR / "fixtures" / "real_handwriting"
DEFAULT_HANDWRITING_CATALOG = TESTING_DIR / "fixtures" / "handwriting_catalog" / "linear_equation_atoms.json"
DEFAULT_AUDIT_LOG_DIR = Path(
    os.environ.get("WHITEBOARD_AUDIT_LOG_DIR", "/Users/jpovj/Documents/dev/log_whiteboard_3")
).expanduser()


REALISTIC_SCENARIOS = (
    "mixed-marks",
    "circled-answer",
    "crossout-scratch",
    "non-sequential",
)

SCENARIO_SLUGS: dict[str, tuple[str, ...]] = {
    "mixed-marks": ("crossout-scratch-division", "circled-x-equals-four", "rational-detached-parenthetical-annotations"),
    "circled-answer": ("circled-final-answer", "circled-x-equals-four", "circled-intermediate-result"),
    "crossout-scratch": ("crossout-scratch-division", "detached-circled-zero-annotation"),
    "non-sequential": ("rational-detached-parenthetical-annotations", "radical-fraction-simplification"),
}

HARNESS_PACKS: dict[str, dict[str, Any]] = {
    "failure-modes": {
        "description": "Log-observed recognition/grading failure modes with known discrepancy labels.",
        "slugs": (
            "fragmented-invalid-expression",
            "fraction-subtraction-oversegmented",
            "readable-expansion-empty-ocr",
            "circled-intermediate-result",
        ),
        "contracts": ("segmentation", "ocr", "grading", "audit"),
        "focus": ("line_count_mismatch", "line_latex_mismatch", "problem_status_mismatch", "line_segmentation_empty"),
    },
    "problem-input": {
        "description": "Handwritten problem input recognition stressors, separate from answer-work grading.",
        "slugs": (
            "problem-input-fraction-sqrt-numerator-equation",
            "problem-input-fraction-with-decimal-denominator",
            "problem-input-indexed-root",
        ),
        "contracts": ("problem-input-recognition", "segmentation", "ocr"),
        "focus": ("problem_input_ocr_mismatch", "line_latex_mismatch"),
    },
    "ambiguous-fractions": {
        "description": "Neighboring, tall, and equation-tail fraction rows that tend to split or merge.",
        "slugs": (
            "problem-input-two-fraction-denominator-18",
            "problem-input-two-fraction-equation-tail",
            "problem-input-fraction-3x-over-x-plus-1-equals-8",
        ),
        "contracts": ("segmentation", "ocr"),
        "focus": ("fraction_merge", "fraction_split", "line_latex_mismatch"),
    },
    "visual-intent": {
        "description": "Circled answers, cross-outs, scratch work, underlines, and detached operation annotations.",
        "slugs": (
            "crossout-scratch-division",
            "circled-x-equals-four",
            "rational-detached-parenthetical-annotations",
        ),
        "contracts": ("segmentation", "visual-intent", "grading"),
        "focus": ("visual_intent_observed", "equation_side_operation_annotation_mismatch"),
    },
    "non-sequential": {
        "description": "Rows written out of spatial order, with later annotations inserted above earlier work.",
        "slugs": (
            "rational-detached-parenthetical-annotations",
            "radical-fraction-simplification",
            "readable-expansion-empty-ocr",
        ),
        "contracts": ("segmentation", "ocr", "grading"),
        "focus": ("non_sequential_writing", "line_latex_mismatch"),
        "nonSequential": True,
    },
    "bad-handwriting-valid-math": {
        "description": "Compact or visually ambiguous handwriting that represents mathematically valid work.",
        "slugs": (
            "compact-monomial-exponent-ocr",
            "compact-plus-minus-solution",
            "semantic-alternate-absolute-value-denominator",
        ),
        "contracts": ("ocr", "semantic", "grading"),
        "focus": ("line_latex_mismatch", "solution_set_mismatch", "semantic_candidate_promotion"),
    },
}

VISUAL_INTENT_POLICIES: dict[str, dict[str, str]] = {
    "circled_answer": {
        "segmentationPolicy": "exclude_visual_strokes",
        "gradingPolicy": "preserve_answer_context",
        "auditPolicy": "record_final_answer_emphasis",
    },
    "circled_intermediate_result": {
        "segmentationPolicy": "exclude_visual_strokes",
        "gradingPolicy": "grade_enclosed_math_if_grouped",
        "auditPolicy": "record_intermediate_emphasis",
    },
    "crossed_out": {
        "segmentationPolicy": "exclude_visual_strokes",
        "gradingPolicy": "ignore_crossed_out_work",
        "auditPolicy": "record_discarded_attempt",
    },
    "scratch_work": {
        "segmentationPolicy": "candidate_only",
        "gradingPolicy": "ignore_unselected_scratch",
        "auditPolicy": "record_scratch_context",
    },
    "detached_operation_annotation": {
        "segmentationPolicy": "exclude_visual_strokes",
        "gradingPolicy": "contextual_operation_only",
        "auditPolicy": "record_side_operation_intent",
    },
    "underline": {
        "segmentationPolicy": "exclude_visual_strokes",
        "gradingPolicy": "ignore_emphasis_line",
        "auditPolicy": "record_operation_alignment",
    },
    "boxed_answer": {
        "segmentationPolicy": "preserve_enclosed_expression_context",
        "gradingPolicy": "grade_enclosed_problem_or_answer",
        "auditPolicy": "record_boxed_expression_emphasis",
    },
}

METRIC_KEYS = (
    "stroke_count",
    "line_count",
    "visual_only_count",
    "point_count",
    "stroke_duration_ms",
    "inter_stroke_gap_ms",
    "stroke_width_px",
    "stroke_height_px",
    "path_length_px",
    "pressure_mean",
    "line_width_px",
    "line_height_px",
    "line_gap_px",
    "multi_stroke_group_size",
    "non_sequential_inversions",
    "answer_width_px",
    "answer_height_px",
    "board_width_px",
    "board_height_px",
)

FEATURE_KEYS = (
    "has_pressure",
    "has_multi_stroke_symbols",
    "has_visual_only_marks",
    "has_circled_answer",
    "has_crossout_or_scratch",
    "has_non_sequential_writing",
)


@dataclass(frozen=True)
class CorpusRecord:
    source_kind: str
    source_id: str
    path: str
    payload: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "sourceKind": self.source_kind,
            "sourceId": self.source_id,
            "path": self.path,
        }


@dataclass(frozen=True)
class DistributionBand:
    minimum: float
    q05: float
    median: float
    q95: float
    maximum: float
    count: int

    @property
    def padded_min(self) -> float:
        spread = max(1.0, self.q95 - self.q05)
        return self.q05 - spread * 0.35

    @property
    def padded_max(self) -> float:
        spread = max(1.0, self.q95 - self.q05)
        return self.q95 + spread * 0.35

    def to_json(self) -> dict[str, float | int]:
        return {
            "min": round_float(self.minimum, 4),
            "q05": round_float(self.q05, 4),
            "median": round_float(self.median, 4),
            "q95": round_float(self.q95, 4),
            "max": round_float(self.maximum, 4),
            "count": self.count,
        }


@dataclass(frozen=True)
class CalibrationSummary:
    source_kind: str
    source_count: int
    bands: dict[str, DistributionBand]
    feature_rates: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        return {
            "sourceKind": self.source_kind,
            "sourceCount": self.source_count,
            "bands": {name: band.to_json() for name, band in sorted(self.bands.items())},
            "featureRates": {name: round_float(value, 4) for name, value in sorted(self.feature_rates.items())},
        }


@dataclass(frozen=True)
class HandwritingAtom:
    label: str
    kind: str
    source_fixture_slug: str
    source_stroke_ids: tuple[str, ...]
    width: float
    height: float
    strokes: tuple[dict[str, Any], ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind,
            "sourceFixtureSlug": self.source_fixture_slug,
            "sourceStrokeIds": list(self.source_stroke_ids),
            "width": round_float(self.width, 2),
            "height": round_float(self.height, 2),
            "strokeCount": len(self.strokes),
        }


@dataclass(frozen=True)
class LinearEquationParameters:
    a: int
    b: int
    x_value: int
    c_value: int
    seed: int
    constraints: dict[str, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c_value,
            "solution": self.x_value,
            "seed": self.seed,
            "constraints": dict(self.constraints),
        }


def load_real_handwriting_fixtures(fixture_dir: Path = DEFAULT_REAL_FIXTURE_DIR) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(Path(fixture_dir).glob("*.json"))
    ]


def load_real_handwriting_records(fixture_dir: Path = DEFAULT_REAL_FIXTURE_DIR) -> list[CorpusRecord]:
    records: list[CorpusRecord] = []
    for path in sorted(Path(fixture_dir).glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("strokes"), list):
            records.append(CorpusRecord(
                source_kind="distilled-real-fixtures",
                source_id=str(payload.get("slug") or path.stem),
                path=relative_path_for_report(path),
                payload=payload,
            ))
    return records


def load_audit_input_payloads(audit_log_dir: Path = DEFAULT_AUDIT_LOG_DIR, limit: Optional[int] = None) -> list[dict[str, Any]]:
    return [record.payload for record in load_audit_input_records(audit_log_dir, limit=limit)]


def load_audit_input_records(audit_log_dir: Path = DEFAULT_AUDIT_LOG_DIR, limit: Optional[int] = None) -> list[CorpusRecord]:
    records: list[CorpusRecord] = []
    paths = sorted(Path(audit_log_dir).glob("**/input.json"))
    if limit is not None:
        paths = paths[: int(limit)]
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("strokes"), list):
            records.append(CorpusRecord(
                source_kind="audit-input-json",
                source_id=path.parent.name,
                path=relative_path_for_report(path),
                payload=payload,
            ))
    return records


def load_corpus_records(
    *,
    audit_log_dir: Path = DEFAULT_AUDIT_LOG_DIR,
    fixture_dir: Path = DEFAULT_REAL_FIXTURE_DIR,
    audit_limit: Optional[int] = None,
    include_fixtures_with_audits: bool = False,
) -> list[CorpusRecord]:
    audit_records = load_audit_input_records(audit_log_dir, limit=audit_limit)
    fixture_records = load_real_handwriting_records(fixture_dir)
    if audit_records and not include_fixtures_with_audits:
        return audit_records
    if audit_records:
        return [*audit_records, *fixture_records]
    return fixture_records


def load_handwriting_atom_catalog(
    catalog_path: Path = DEFAULT_HANDWRITING_CATALOG,
    *,
    fixture_dir: Path = DEFAULT_REAL_FIXTURE_DIR,
) -> dict[str, list[HandwritingAtom]]:
    manifest = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    fixtures_by_slug = {fixture["slug"]: fixture for fixture in load_real_handwriting_fixtures(fixture_dir)}
    catalog: dict[str, list[HandwritingAtom]] = {}
    for entry in manifest.get("atoms") or []:
        atom = atom_from_manifest_entry(entry, fixtures_by_slug)
        catalog.setdefault(atom.label, []).append(atom)
    return catalog


def atom_from_manifest_entry(entry: dict[str, Any], fixtures_by_slug: dict[str, dict[str, Any]]) -> HandwritingAtom:
    label = str(entry["label"])
    source_slug = str(entry["sourceFixtureSlug"])
    source = fixtures_by_slug[source_slug]
    stroke_ids = [str(stroke_id) for stroke_id in entry.get("strokeIds") or []]
    strokes_by_id = {str(stroke.get("id")): stroke for stroke in source.get("strokes") or []}
    source_strokes = [strokes_by_id[stroke_id] for stroke_id in stroke_ids]
    source_box = bbox_for_strokes(source_strokes) or {"xMin": 0, "yMin": 0, "xMax": 1, "yMax": 1}
    transform = entry.get("transform") or {}
    scale = float(transform.get("scale") or 1.0)
    x_origin = source_box["xMin"]
    y_origin = source_box["yMin"]
    time_base = min((number_or_zero(stroke.get("startTime")) for stroke in source_strokes), default=0.0)
    normalized_strokes: list[dict[str, Any]] = []
    for index, stroke in enumerate(source_strokes, start=1):
        raw_points = normalize_atom_points(stroke.get("rawPoints") or [], x_origin, y_origin, scale)
        outline_points = normalize_atom_points(stroke.get("outlinePoints") or stroke.get("rawPoints") or [], x_origin, y_origin, scale)
        normalized_strokes.append({
            "id": f"a{index:03d}",
            "rawPoints": raw_points,
            "outlinePoints": outline_points or raw_points,
            "startOffset": round_float(number_or_zero(stroke.get("startTime")) - time_base, 1),
            "duration": round_float(max(1.0, number_or_zero(stroke.get("endTime")) - number_or_zero(stroke.get("startTime"))), 1),
            "color": stroke.get("color") or "#000000",
            "sourceStrokeId": str(stroke.get("id") or index),
        })
    atom_box = bbox_for_strokes([
        {"canvasBbox": bbox_for_points(stroke["rawPoints"])}
        for stroke in normalized_strokes
    ]) or {"xMin": 0, "yMin": 0, "xMax": 1, "yMax": 1}
    return HandwritingAtom(
        label=label,
        kind=str(entry.get("kind") or "symbol"),
        source_fixture_slug=source_slug,
        source_stroke_ids=tuple(stroke_ids),
        width=max(1.0, atom_box["xMax"] - atom_box["xMin"]),
        height=max(1.0, atom_box["yMax"] - atom_box["yMin"]),
        strokes=tuple(normalized_strokes),
    )


def normalize_atom_points(points: Sequence[Any], x_origin: float, y_origin: float, scale: float) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for point in points:
        if not isinstance(point, dict) or not is_finite(point.get("x")) or not is_finite(point.get("y")):
            continue
        pressure = float(point.get("pressure")) if is_finite(point.get("pressure")) else 0.5
        out.append({
            "x": round_float((float(point["x"]) - x_origin) * scale, 2),
            "y": round_float((float(point["y"]) - y_origin) * scale, 2),
            "pressure": round_float(pressure, 3),
        })
    return out


def build_calibration_summary(
    *,
    audit_log_dir: Path = DEFAULT_AUDIT_LOG_DIR,
    fixture_dir: Path = DEFAULT_REAL_FIXTURE_DIR,
    audit_limit: Optional[int] = None,
) -> CalibrationSummary:
    records = load_corpus_records(
        audit_log_dir=audit_log_dir,
        fixture_dir=fixture_dir,
        audit_limit=audit_limit,
    )
    source_kind = records[0].source_kind if records else "empty-corpus"
    if len({record.source_kind for record in records}) > 1:
        source_kind = "mixed-real-corpus"
    return summarize_distribution([record.payload for record in records], source_kind=source_kind)


def build_calibration_report(
    *,
    audit_log_dir: Path = DEFAULT_AUDIT_LOG_DIR,
    fixture_dir: Path = DEFAULT_REAL_FIXTURE_DIR,
    audit_limit: Optional[int] = None,
    include_fixtures_with_audits: bool = False,
    sample_limit: int = 12,
) -> dict[str, Any]:
    records = load_corpus_records(
        audit_log_dir=audit_log_dir,
        fixture_dir=fixture_dir,
        audit_limit=audit_limit,
        include_fixtures_with_audits=include_fixtures_with_audits,
    )
    source_kinds = sorted({record.source_kind for record in records})
    source_kind = source_kinds[0] if len(source_kinds) == 1 else "mixed-real-corpus"
    calibration = summarize_distribution([record.payload for record in records], source_kind=source_kind)
    summaries = [
        {
            **record.to_json(),
            "summary": fixture_summary(record.payload),
        }
        for record in records[:sample_limit]
    ]
    return {
        "schemaVersion": 1,
        "reportKind": "real-handwriting-calibration",
        "sourceKinds": source_kinds,
        "recordCount": len(records),
        "calibration": calibration.to_json(),
        "metricKeys": list(METRIC_KEYS),
        "featureKeys": list(FEATURE_KEYS),
        "sampleRecords": summaries,
    }


def build_harness_dashboard_report(
    generated_fixtures: Sequence[dict[str, Any]],
    *,
    audit_log_dir: Path = DEFAULT_AUDIT_LOG_DIR,
    fixture_dir: Path = DEFAULT_REAL_FIXTURE_DIR,
    audit_limit: Optional[int] = None,
) -> dict[str, Any]:
    calibration_report = build_calibration_report(
        audit_log_dir=audit_log_dir,
        fixture_dir=fixture_dir,
        audit_limit=audit_limit,
    )
    calibration = summarize_distribution(
        [
            record.payload
            for record in load_corpus_records(
                audit_log_dir=audit_log_dir,
                fixture_dir=fixture_dir,
                audit_limit=audit_limit,
            )
        ],
        source_kind=calibration_report["calibration"]["sourceKind"],
    )
    generated_summaries = []
    drift_checks = []
    for fixture in generated_fixtures:
        summary = fixture_summary(fixture)
        validation = validate_fixture_against_calibration(fixture, calibration)
        generated_summaries.append({
            **summary,
            "oracleContracts": fixture.get("oracleContracts") or [],
            "visualIntentPolicies": fixture.get("visualIntentPolicies") or [],
        })
        drift_checks.append({
            "slug": fixture.get("slug"),
            "fixtureKind": fixture.get("fixtureKind"),
            "ok": validation["ok"],
            "checks": validation["checks"],
        })
    return {
        "schemaVersion": 1,
        "reportKind": "real-handwriting-generator-dashboard",
        "calibration": calibration.to_json(),
        "generatedCount": len(generated_fixtures),
        "generatedSummaries": generated_summaries,
        "driftChecks": drift_checks,
        "packNames": sorted(HARNESS_PACKS),
        "visualIntentPolicyKinds": sorted(VISUAL_INTENT_POLICIES),
    }


def summarize_distribution(payloads: Sequence[dict[str, Any]], *, source_kind: str) -> CalibrationSummary:
    metric_values: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
    feature_counts = {key: 0 for key in FEATURE_KEYS}

    usable_payloads = [payload for payload in payloads if payload.get("strokes")]
    for payload in usable_payloads:
        metrics = fixture_metrics(payload)
        for key, value in metrics.items():
            if key in metric_values:
                if isinstance(value, list):
                    metric_values[key].extend(float(item) for item in value if is_finite(item))
                elif is_finite(value):
                    metric_values[key].append(float(value))
        features = classify_features(payload)
        for key in feature_counts:
            if features.get(key):
                feature_counts[key] += 1

    bands = {
        key: quantile_band(values)
        for key, values in metric_values.items()
        if values
    }
    denominator = max(1, len(usable_payloads))
    return CalibrationSummary(
        source_kind=source_kind,
        source_count=len(usable_payloads),
        bands=bands,
        feature_rates={key: count / denominator for key, count in feature_counts.items()},
    )


def sample_linear_equation_parameters(
    *,
    seed: int = 0,
    min_coefficient: int = 2,
    max_coefficient: int = 9,
    min_constant: int = 1,
    max_constant: int = 9,
    min_solution: int = 1,
    max_solution: int = 9,
    max_result: int = 99,
) -> LinearEquationParameters:
    constraints = {
        "minCoefficient": int(min_coefficient),
        "maxCoefficient": int(max_coefficient),
        "minConstant": int(min_constant),
        "maxConstant": int(max_constant),
        "minSolution": int(min_solution),
        "maxSolution": int(max_solution),
        "maxResult": int(max_result),
    }
    if constraints["minCoefficient"] <= 0 or constraints["minCoefficient"] > constraints["maxCoefficient"]:
        raise ValueError(f"Invalid coefficient bounds: {constraints}")
    if constraints["minConstant"] < 0 or constraints["minConstant"] > constraints["maxConstant"]:
        raise ValueError(f"Invalid constant bounds: {constraints}")
    if constraints["minSolution"] < 0 or constraints["minSolution"] > constraints["maxSolution"]:
        raise ValueError(f"Invalid solution bounds: {constraints}")

    rng = random.Random(seed)
    for _ in range(500):
        a = rng.randint(constraints["minCoefficient"], constraints["maxCoefficient"])
        b = rng.randint(constraints["minConstant"], constraints["maxConstant"])
        x_value = rng.randint(constraints["minSolution"], constraints["maxSolution"])
        c_value = a * x_value + b
        if c_value <= constraints["maxResult"]:
            return LinearEquationParameters(
                a=a,
                b=b,
                x_value=x_value,
                c_value=c_value,
                seed=seed,
                constraints=constraints,
            )
    raise ValueError(f"Could not sample a linear equation within constraints: {constraints}")


def build_hybrid_linear_equation_fixture(
    *,
    a: int = 3,
    b: int = 2,
    x_value: int = 4,
    seed: int = 0,
    problem_source: str = "explicit-parameters",
    generation_constraints: Optional[dict[str, int]] = None,
    catalog_path: Path = DEFAULT_HANDWRITING_CATALOG,
    fixture_dir: Path = DEFAULT_REAL_FIXTURE_DIR,
    include_circle: bool = True,
    include_crossout: bool = False,
    non_sequential_final: bool = False,
) -> dict[str, Any]:
    catalog = load_handwriting_atom_catalog(catalog_path, fixture_dir=fixture_dir)
    c_value = a * x_value + b
    problem_latex = f"{a}x + {b} = {c_value}"
    rows = linear_equation_rows(a, b, c_value, x_value)
    ensure_catalog_labels(catalog, sorted({token for row in rows for token in row["tokens"]}))

    rng = random.Random(seed)
    fixture = empty_generated_fixture("hybrid-linear-equation", seed)
    fixture["fixtureKind"] = "hybrid-real-stroke-linear-equation"
    fixture["slug"] = f"hybrid-linear-{a}x-plus-{b}-equals-{c_value}-seed-{seed}"
    fixture["description"] = "Hybrid fixture composed from curated real handwriting atoms for a generated linear equation."
    fixture["sourceAuditId"] = "generated-from-real-stroke-atom-catalog"
    fixture["problemLatex"] = problem_latex
    fixture["problemMetadata"] = {
        "problemType": "equation-solving",
        "solveVariable": "x",
        "scenario": "hybrid-linear-equation",
        "problemFamily": "linear-equation-positive-integer-one-variable",
        "problemSource": problem_source,
        "seed": seed,
        "a": a,
        "b": b,
        "c": c_value,
        "solution": x_value,
        "answerLatex": f"x={x_value}",
        "generationConstraints": generation_constraints or {},
    }
    fixture["expectedLatexLines"] = [row["latex"] for row in rows]

    base_x = 116.0
    base_y = 96.0
    line_gap = 116.0
    row_indexes = list(range(len(rows)))
    if non_sequential_final and len(row_indexes) > 2:
        row_indexes = [*row_indexes[:-1], row_indexes[-1]]

    time_cursor = 0.0
    row_boxes: dict[int, dict[str, float]] = {}
    atom_cycle_state: dict[str, int] = {}
    for row_index in row_indexes:
        row = rows[row_index]
        row_x = base_x + row.get("xOffset", 0.0) + rng.uniform(-8.0, 8.0)
        row_y = base_y + row_index * line_gap + rng.uniform(-5.0, 5.0)
        row_stroke_ids, row_box, time_cursor = append_token_row(
            fixture,
            catalog,
            row["tokens"],
            row_index=row_index,
            x=row_x,
            y=row_y,
            start_time=time_cursor,
            rng=rng,
            wide_gap_after=set(row.get("wideGapAfter") or []),
            underline_groups=row.get("underlineGroups") or [],
            atom_cycle_state=atom_cycle_state,
        )
        row_boxes[row_index] = row_box
        fixture["expectedLineGroups"].append({
            "lineIndex": row_index,
            "latex": row["latex"],
            "strokeIds": row_stroke_ids,
            "source": "hybrid-real-stroke-atom-catalog",
            "sourceCandidateId": f"hybrid-linear-row-{row_index + 1}",
        })
        time_cursor += rng.uniform(520.0, 980.0)

    if include_circle and rows:
        final_index = len(rows) - 1
        final_box = row_boxes.get(final_index)
        if final_box:
            visual_ids, _, time_cursor = append_visual_atom_around_box(
                fixture,
                catalog,
                "circle",
                final_box,
                start_time=time_cursor + rng.uniform(180.0, 420.0),
                rng=rng,
            )
            fixture["visualOnlyStrokeIds"].extend(visual_ids)
            fixture["visualMarks"].append({
                "type": "circled_answer",
                "latex": rows[-1]["latex"],
                "lineIndex": final_index,
                "notes": "Hybrid generator circled the final answer using a real visual mark.",
                "confidence": 1.0,
            })

    if include_crossout and row_boxes.get(0):
        visual_ids, _, _ = append_visual_atom_around_box(
            fixture,
            catalog,
            "crossout",
            row_boxes[0],
            start_time=time_cursor + rng.uniform(200.0, 480.0),
            rng=rng,
            fit_mode="diagonal",
        )
        fixture["visualOnlyStrokeIds"].extend(visual_ids)
        fixture["visualMarks"].append({
            "type": "crossed_out",
            "latex": rows[0]["latex"],
            "lineIndex": 0,
            "notes": "Hybrid generator added a real crossed-out visual mark.",
            "confidence": 1.0,
        })

    finish_generated_fixture(fixture)
    fixture["oracleContracts"] = build_oracle_contracts(("segmentation", "ocr", "semantic", "grading"))
    attach_visual_intent_policies(fixture)
    fixture["sourceStats"] = {
        **fixture.get("sourceStats", {}),
        "catalogPath": relative_path_for_report(catalog_path),
        "catalogAtomCount": sum(len(values) for values in catalog.values()),
        "generatedProblemLatex": problem_latex,
    }
    return fixture


def build_random_hybrid_linear_equation_fixture(
    *,
    seed: int = 0,
    min_coefficient: int = 2,
    max_coefficient: int = 9,
    min_constant: int = 1,
    max_constant: int = 9,
    min_solution: int = 1,
    max_solution: int = 9,
    max_result: int = 99,
    catalog_path: Path = DEFAULT_HANDWRITING_CATALOG,
    fixture_dir: Path = DEFAULT_REAL_FIXTURE_DIR,
    include_circle: bool = True,
    include_crossout: bool = False,
    non_sequential_final: bool = False,
) -> dict[str, Any]:
    params = sample_linear_equation_parameters(
        seed=seed,
        min_coefficient=min_coefficient,
        max_coefficient=max_coefficient,
        min_constant=min_constant,
        max_constant=max_constant,
        min_solution=min_solution,
        max_solution=max_solution,
        max_result=max_result,
    )
    fixture = build_hybrid_linear_equation_fixture(
        a=params.a,
        b=params.b,
        x_value=params.x_value,
        seed=seed,
        problem_source="seeded-random-family",
        generation_constraints=params.constraints,
        catalog_path=catalog_path,
        fixture_dir=fixture_dir,
        include_circle=include_circle,
        include_crossout=include_crossout,
        non_sequential_final=non_sequential_final,
    )
    fixture["problemMetadata"]["sampledParameters"] = params.to_json()
    fixture["sourceStats"]["randomLinearSampler"] = params.to_json()
    return fixture


def build_oracle_contracts(
    contract_names: Sequence[str],
    *,
    expected_failure_modes: Sequence[str] = (),
    problem_input: bool = False,
) -> list[dict[str, Any]]:
    descriptions = {
        "segmentation": "Selected stroke groups should match reviewed expectedLineGroups.",
        "ocr": "Recognized LaTeX should preserve the intended handwritten line content.",
        "semantic": "Semantic candidate scoring may promote a better valid candidate over the top OCR candidate.",
        "grading": "Grading should use accepted lines and visual policies to decide problem status.",
        "audit": "Audit payload should preserve discrepancy and visual-intent context.",
        "visual-intent": "Visual-only marks should inform context without becoming graded math lines.",
        "problem-input-recognition": "Problem input handwriting should be recognized as source problem context.",
    }
    contracts = []
    for name in contract_names:
        contracts.append({
            "name": name,
            "description": descriptions.get(name, name),
            "expectedFailureModes": list(expected_failure_modes),
            "problemInput": problem_input,
        })
    return contracts


def attach_visual_intent_policies(fixture: dict[str, Any]) -> None:
    seen: set[str] = set()
    policies: list[dict[str, Any]] = []
    for mark in fixture.get("visualMarks") or []:
        mark_type = str(mark.get("type") or "")
        if mark_type in VISUAL_INTENT_POLICIES and mark_type not in seen:
            seen.add(mark_type)
            policies.append({
                "type": mark_type,
                **VISUAL_INTENT_POLICIES[mark_type],
            })
    source_labels = {str(stroke.get("sourceAtomLabel") or "") for stroke in fixture.get("strokes") or []}
    if "underline" in source_labels and "underline" not in seen:
        policies.append({"type": "underline", **VISUAL_INTENT_POLICIES["underline"]})
    fixture["visualIntentPolicies"] = policies


def apply_nonsequential_timing(fixture: dict[str, Any], *, seed: int) -> None:
    rng = random.Random(seed)
    groups = fixture.get("expectedLineGroups") or []
    if len(groups) < 2:
        return
    offsets = [index * 1600.0 for index in reversed(range(len(groups)))]
    strokes_by_id = {str(stroke.get("id")): stroke for stroke in fixture.get("strokes") or []}
    for group, offset in zip(groups, offsets):
        line_strokes = [strokes_by_id[str(stroke_id)] for stroke_id in group.get("strokeIds") or [] if str(stroke_id) in strokes_by_id]
        if not line_strokes:
            continue
        base = min(number_or_zero(stroke.get("startTime")) for stroke in line_strokes)
        for stroke in line_strokes:
            duration = max(1.0, number_or_zero(stroke.get("endTime")) - number_or_zero(stroke.get("startTime")))
            start = offset + max(0.0, number_or_zero(stroke.get("startTime")) - base) + rng.uniform(0.0, 20.0)
            stroke["startTime"] = round_float(start, 1)
            stroke["endTime"] = round_float(start + duration, 1)


def build_log_observed_pack_fixture(
    pack_name: str,
    *,
    seed: int = 0,
    fixture_dir: Path = DEFAULT_REAL_FIXTURE_DIR,
) -> dict[str, Any]:
    if pack_name not in HARNESS_PACKS:
        known = ", ".join(sorted(HARNESS_PACKS))
        raise KeyError(f"Unknown real-handwriting harness pack {pack_name!r}. Known packs: {known}")
    pack = HARNESS_PACKS[pack_name]
    rng = random.Random(seed)
    fixtures_by_slug = {fixture["slug"]: fixture for fixture in load_real_handwriting_fixtures(fixture_dir)}
    selected = [fixtures_by_slug[slug] for slug in pack["slugs"] if slug in fixtures_by_slug]
    if not selected:
        raise FileNotFoundError(f"No source fixtures found for harness pack {pack_name!r}")

    composite = empty_generated_fixture(f"log-observed-{pack_name}", seed)
    composite["fixtureKind"] = "hybrid-real-stroke-log-observed-pack"
    composite["slug"] = f"hybrid-log-pack-{pack_name}-seed-{seed}"
    composite["description"] = str(pack["description"])
    composite["sourceAuditId"] = "generated-from-log-observed-real-fixture-pack"
    composite["problemMetadata"] = {
        "scenario": "log-observed-pack",
        "packName": pack_name,
        "seed": seed,
        "focus": list(pack.get("focus") or []),
        "sourceSlugs": [fixture["slug"] for fixture in selected],
        "problemInputMode": "problem-input-recognition" in pack.get("contracts", ()),
    }
    composite["oracleContracts"] = build_oracle_contracts(
        pack.get("contracts") or (),
        expected_failure_modes=pack.get("focus") or (),
        problem_input="problem-input-recognition" in pack.get("contracts", ()),
    )

    x_start = 86.0
    y_cursor = 78.0
    time_offsets = scenario_time_offsets("non-sequential" if pack.get("nonSequential") else pack_name, len(selected))
    for source_index, source in enumerate(selected):
        scale = rng.uniform(0.72, 0.92)
        origin = bbox_origin(source.get("answerBox")) or bbox_origin(bbox_for_strokes(source.get("strokes") or [])) or {"x": 0, "y": 0}
        dx = x_start - origin["x"] + rng.uniform(-5.0, 5.0)
        dy = y_cursor - origin["y"] + rng.uniform(-4.0, 4.0)
        append_transformed_source(
            composite,
            source,
            source_index=source_index,
            dx=dx,
            dy=dy,
            scale=scale,
            time_offset=time_offsets[source_index],
            rng=rng,
        )
        source_box = bbox_for_strokes(source.get("strokes") or []) or {"xMin": 0, "yMin": 0, "xMax": 260, "yMax": 160}
        y_cursor += max(180.0, (source_box["yMax"] - source_box["yMin"]) * scale + 90.0)

    if pack.get("nonSequential"):
        apply_nonsequential_timing(composite, seed=seed + 991)
    finish_generated_fixture(composite)
    attach_visual_intent_policies(composite)
    return composite


def default_dashboard_fixtures(
    *,
    seed: int = 0,
    fixture_dir: Path = DEFAULT_REAL_FIXTURE_DIR,
    catalog_path: Path = DEFAULT_HANDWRITING_CATALOG,
) -> list[dict[str, Any]]:
    fixtures = [
        build_random_hybrid_linear_equation_fixture(seed=seed + 31, fixture_dir=fixture_dir, catalog_path=catalog_path),
        build_hybrid_complex_math_fixture(seed=seed + 41, fixture_dir=fixture_dir, catalog_path=catalog_path, include_crossout=True),
    ]
    fixtures.extend(
        build_log_observed_pack_fixture(pack_name, seed=seed + index + 101, fixture_dir=fixture_dir)
        for index, pack_name in enumerate(sorted(HARNESS_PACKS))
    )
    return fixtures


def build_hybrid_complex_math_fixture(
    *,
    seed: int = 0,
    catalog_path: Path = DEFAULT_HANDWRITING_CATALOG,
    fixture_dir: Path = DEFAULT_REAL_FIXTURE_DIR,
    include_circle: bool = True,
    include_detached_annotations: bool = True,
    include_crossout: bool = False,
) -> dict[str, Any]:
    catalog = load_handwriting_atom_catalog(catalog_path, fixture_dir=fixture_dir)
    rows = complex_math_rows()
    ensure_catalog_labels(catalog, sorted({token for row in rows for token in row["tokens"]}))
    if include_detached_annotations:
        ensure_catalog_labels(catalog, ["detached_operation_left", "detached_operation_right"])

    rng = random.Random(seed)
    fixture = empty_generated_fixture("hybrid-complex-math", seed)
    fixture["fixtureKind"] = "hybrid-real-stroke-complex-math"
    fixture["slug"] = f"hybrid-complex-math-seed-{seed}"
    fixture["description"] = (
        "Hybrid complex-math fixture composed from curated real handwriting atoms, "
        "covering fractions, exponents, radicals, plus-minus answers, rational solving, "
        "and detached operation annotations."
    )
    fixture["sourceAuditId"] = "generated-from-real-stroke-atom-catalog"
    fixture["problemLatex"] = "\\frac{x - 1}{x} = \\frac{3}{4}"
    fixture["problemMetadata"] = {
        "problemType": "equation-solving",
        "solveVariable": "x",
        "scenario": "hybrid-complex-math",
        "problemFamily": "complex-expression-and-rational-equation-corpus",
        "problemSource": "seeded-complex-template-family",
        "seed": seed,
        "answerLatex": "x=4",
        "coveredStructures": [
            "fractions",
            "exponents",
            "radicals",
            "plus-minus",
            "multi-line-rational-equation-solving",
            "detached-operation-annotations",
        ],
    }
    fixture["expectedLatexLines"] = [row["latex"] for row in rows]

    base_x = 92.0
    base_y = 78.0
    line_gap = 216.0
    time_cursor = 0.0
    row_boxes: dict[int, dict[str, float]] = {}
    atom_cycle_state: dict[str, int] = {}
    for row_index, row in enumerate(rows):
        row_x = base_x + row.get("xOffset", 0.0) + rng.uniform(-7.0, 7.0)
        row_y = base_y + row_index * line_gap + rng.uniform(-4.0, 4.0)
        row_stroke_ids, row_box, time_cursor = append_token_row(
            fixture,
            catalog,
            row["tokens"],
            row_index=row_index,
            x=row_x,
            y=row_y,
            start_time=time_cursor,
            rng=rng,
            wide_gap_after=set(row.get("wideGapAfter") or []),
            underline_groups=row.get("underlineGroups") or [],
            atom_cycle_state=atom_cycle_state,
        )
        row_boxes[row_index] = row_box
        fixture["expectedLineGroups"].append({
            "lineIndex": row_index,
            "latex": row["latex"],
            "strokeIds": row_stroke_ids,
            "source": "hybrid-real-stroke-atom-catalog",
            "sourceCandidateId": f"hybrid-complex-row-{row_index + 1}",
        })
        time_cursor += rng.uniform(620.0, 1080.0)

    if include_detached_annotations:
        annotation_specs = [
            ("detached_operation_left", 4, -72.0, 68.0, "left-side rational multiplier"),
            ("detached_operation_right", 5, 430.0, -76.0, "right-side rational multiplier"),
        ]
        for label, row_index, dx, dy, note in annotation_specs:
            target_box = row_boxes.get(row_index)
            if not target_box:
                continue
            visual_ids, _, time_cursor = append_detached_visual_atom(
                fixture,
                catalog,
                label,
                target_box,
                x_offset=dx,
                y_offset=dy,
                start_time=time_cursor + rng.uniform(120.0, 320.0),
                rng=rng,
            )
            fixture["visualOnlyStrokeIds"].extend(visual_ids)
            fixture["visualMarks"].append({
                "type": "detached_operation_annotation",
                "latex": "(x-1)",
                "lineIndex": row_index,
                "notes": note,
                "confidence": 0.9,
            })

    if include_circle and row_boxes.get(len(rows) - 1):
        visual_ids, _, time_cursor = append_visual_atom_around_box(
            fixture,
            catalog,
            "circle",
            row_boxes[len(rows) - 1],
            start_time=time_cursor + rng.uniform(180.0, 420.0),
            rng=rng,
        )
        fixture["visualOnlyStrokeIds"].extend(visual_ids)
        fixture["visualMarks"].append({
            "type": "circled_answer",
            "latex": rows[-1]["latex"],
            "lineIndex": len(rows) - 1,
            "notes": "Hybrid complex generator circled the final rational-equation answer.",
            "confidence": 1.0,
        })

    if include_crossout and row_boxes.get(0):
        visual_ids, _, _ = append_visual_atom_around_box(
            fixture,
            catalog,
            "crossout",
            row_boxes[0],
            start_time=time_cursor + rng.uniform(200.0, 480.0),
            rng=rng,
            fit_mode="diagonal",
        )
        fixture["visualOnlyStrokeIds"].extend(visual_ids)
        fixture["visualMarks"].append({
            "type": "crossed_out",
            "latex": rows[0]["latex"],
            "lineIndex": 0,
            "notes": "Hybrid complex generator added a crossed-out scratch row.",
            "confidence": 1.0,
        })

    finish_generated_fixture(fixture)
    fixture["oracleContracts"] = build_oracle_contracts(
        ("segmentation", "ocr", "semantic", "grading", "visual-intent"),
    )
    attach_visual_intent_policies(fixture)
    fixture["sourceStats"] = {
        **fixture.get("sourceStats", {}),
        "catalogPath": relative_path_for_report(catalog_path),
        "catalogAtomCount": sum(len(values) for values in catalog.values()),
        "generatedProblemLatex": fixture["problemLatex"],
    }
    return fixture


def complex_math_rows() -> list[dict[str, Any]]:
    return [
        {
            "latex": "\\frac{1}{2}+\\frac{3}{4}",
            "tokens": ["fraction_addition_row"],
        },
        {
            "latex": "x^2=\\frac{9}{4}",
            "tokens": ["x_squared", "=", "frac_9_4"],
            "xOffset": 34.0,
        },
        {
            "latex": "x=\\pm\\frac{3}{2}",
            "tokens": ["x", "=", "plus_minus", "frac_3_2"],
            "xOffset": 42.0,
        },
        {
            "latex": "\\sqrt{25}",
            "tokens": ["sqrt_25"],
            "xOffset": 24.0,
        },
        {
            "latex": "(x-1)(x-1)",
            "tokens": ["rational_factor_row"],
        },
        {
            "latex": "12=4x-4",
            "tokens": ["rational_linear_row"],
            "xOffset": 122.0,
        },
        {
            "latex": "16=4x",
            "tokens": ["rational_isolated_row"],
            "xOffset": 144.0,
            "underlineGroups": [[0]],
        },
        {
            "latex": "x=4",
            "tokens": ["x=4"],
            "xOffset": 178.0,
        },
    ]


def linear_equation_rows(a: int, b: int, c_value: int, x_value: int) -> list[dict[str, Any]]:
    return [
        {
            "latex": f"{a}x+{b}={c_value}",
            "tokens": compact_coefficient_tokens(a) + ["+", *digits_for(b), "=", *digits_for(c_value)],
        },
        {
            "latex": f"-{b}\\quad -{b}",
            "tokens": ["-", *digits_for(b), "-", *digits_for(b)],
            "xOffset": 56.0,
            "wideGapAfter": [1],
            "underlineGroups": [[0, 1], [2, 3]],
        },
        {
            "latex": f"{a}x={c_value - b}",
            "tokens": compact_coefficient_tokens(a) + ["=", *digits_for(c_value - b)],
            "xOffset": 18.0,
        },
        {
            "latex": f"/{a}\\quad /{a}",
            "tokens": ["/", *digits_for(a), "/", *digits_for(a)],
            "xOffset": 78.0,
            "wideGapAfter": [1],
            "underlineGroups": [[0, 1], [2, 3]],
        },
        {
            "latex": f"x={x_value}",
            "tokens": ["x", "=", *digits_for(x_value)],
            "xOffset": 48.0,
        },
    ]


def compact_coefficient_tokens(value: int) -> list[str]:
    compact = f"{value}x"
    if compact == "3x":
        return [compact]
    return [*digits_for(value), "x"]


def digits_for(value: int) -> list[str]:
    return list(str(abs(int(value))))


def ensure_catalog_labels(catalog: dict[str, list[HandwritingAtom]], labels: Sequence[str]) -> None:
    missing = [label for label in labels if label not in catalog]
    if missing:
        known = ", ".join(sorted(catalog))
        raise KeyError(f"Missing handwriting atoms for labels {missing}. Known labels: {known}")


def append_token_row(
    fixture: dict[str, Any],
    catalog: dict[str, list[HandwritingAtom]],
    tokens: Sequence[str],
    *,
    row_index: int,
    x: float,
    y: float,
    start_time: float,
    rng: random.Random,
    wide_gap_after: set[int],
    underline_groups: Sequence[Sequence[int]],
    atom_cycle_state: dict[str, int],
) -> tuple[list[str], dict[str, float], float]:
    stroke_ids: list[str] = []
    boxes: list[dict[str, float]] = []
    cursor_x = x
    time_cursor = start_time
    row_scale = rng.uniform(0.94, 1.06)
    token_boxes: dict[int, dict[str, float]] = {}
    for token_index, token in enumerate(tokens):
        atom = choose_atom(catalog, token, rng, cycle_state=atom_cycle_state)
        token_scale = row_scale * rng.uniform(0.92, 1.08)
        baseline_y = y + max(0.0, 82.0 - atom.height * token_scale)
        placed_ids, box, time_cursor = append_atom(
            fixture,
            atom,
            prefix=f"r{row_index + 1:02d}t{token_index + 1:02d}",
            x=cursor_x,
            y=baseline_y,
            scale=token_scale,
            start_time=time_cursor,
            rng=rng,
        )
        stroke_ids.extend(placed_ids)
        boxes.append(box)
        token_boxes[token_index] = box
        gap = rng.uniform(13.0, 24.0)
        if token_index in wide_gap_after:
            gap += rng.uniform(0.0, 6.0)
        cursor_x += (atom.width * token_scale) + gap
        time_cursor += rng.uniform(46.0, 135.0)
    for group in underline_groups:
        group_boxes = [token_boxes[index] for index in group if index in token_boxes]
        if not group_boxes:
            continue
        underline_target = union_boxes(group_boxes)
        underline_ids, underline_box, time_cursor = append_underline_for_box(
            fixture,
            catalog,
            underline_target,
            start_time=time_cursor + rng.uniform(30.0, 110.0),
            rng=rng,
        )
        fixture["visualOnlyStrokeIds"].extend(underline_ids)
        boxes.append(underline_box)
    return stroke_ids, union_boxes(boxes), time_cursor


def choose_atom(
    catalog: dict[str, list[HandwritingAtom]],
    label: str,
    rng: random.Random,
    *,
    cycle_state: Optional[dict[str, int]] = None,
) -> HandwritingAtom:
    choices = catalog[label]
    if cycle_state is not None:
        index = cycle_state.get(label, 0)
        cycle_state[label] = index + 1
        return choices[index % len(choices)]
    return choices[rng.randrange(len(choices))]


def append_atom(
    fixture: dict[str, Any],
    atom: HandwritingAtom,
    *,
    prefix: str,
    x: float,
    y: float,
    scale: float,
    start_time: float,
    rng: random.Random,
) -> tuple[list[str], dict[str, float], float]:
    stroke_ids: list[str] = []
    boxes: list[dict[str, float]] = []
    last_end = start_time
    for index, stroke in enumerate(atom.strokes, start=1):
        stroke_id = f"{prefix}s{index:03d}"
        raw_points = place_atom_points(stroke.get("rawPoints") or [], x, y, scale, rng)
        outline_points = place_atom_points(stroke.get("outlinePoints") or stroke.get("rawPoints") or [], x, y, scale, rng)
        box = bbox_for_points(raw_points)
        start = start_time + number_or_zero(stroke.get("startOffset")) + rng.uniform(0.0, 18.0)
        end = start + max(1.0, number_or_zero(stroke.get("duration"))) * rng.uniform(0.88, 1.15)
        fixture["strokes"].append({
            "id": stroke_id,
            "startTime": round_float(start, 1),
            "endTime": round_float(end, 1),
            "rawPoints": raw_points,
            "outlinePoints": outline_points or raw_points,
            "color": stroke.get("color") or "#000000",
            "canvasBbox": box,
            "sourceFixtureSlug": atom.source_fixture_slug,
            "sourceStrokeId": stroke.get("sourceStrokeId"),
            "sourceAtomLabel": atom.label,
            "sourceAtomKind": atom.kind,
        })
        stroke_ids.append(stroke_id)
        boxes.append(box)
        last_end = max(last_end, end)
    return stroke_ids, union_boxes(boxes), last_end


def place_atom_points(points: Sequence[dict[str, Any]], x: float, y: float, scale: float, rng: random.Random) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for point in points:
        if not is_finite(point.get("x")) or not is_finite(point.get("y")):
            continue
        pressure = float(point.get("pressure")) if is_finite(point.get("pressure")) else 0.5
        out.append({
            "x": round_float(x + float(point["x"]) * scale + rng.uniform(-0.9, 0.9), 2),
            "y": round_float(y + float(point["y"]) * scale + rng.uniform(-0.9, 0.9), 2),
            "pressure": round_float(min(1.0, max(0.0, pressure + rng.uniform(-0.018, 0.018))), 3),
        })
    return out


def append_visual_atom_around_box(
    fixture: dict[str, Any],
    catalog: dict[str, list[HandwritingAtom]],
    label: str,
    target_box: dict[str, float],
    *,
    start_time: float,
    rng: random.Random,
    fit_mode: str = "around",
) -> tuple[list[str], dict[str, float], float]:
    atom = choose_atom(catalog, label, rng)
    target_width = max(1.0, target_box["xMax"] - target_box["xMin"])
    target_height = max(1.0, target_box["yMax"] - target_box["yMin"])
    if fit_mode == "diagonal":
        scale = min(
            (target_width * 1.12) / max(1.0, atom.width),
            (target_height * 1.45) / max(1.0, atom.height),
        )
        x = target_box["xMin"] - target_width * 0.06
        y = center_y(target_box) - (atom.height * scale) / 2
    else:
        scale = max((target_width + 48.0) / max(1.0, atom.width), (target_height + 34.0) / max(1.0, atom.height))
        x = target_box["xMin"] - 24.0
        y = target_box["yMin"] - 18.0
    return append_atom(
        fixture,
        atom,
        prefix=f"v{len(fixture.get('visualOnlyStrokeIds') or []) + 1:02d}",
        x=x,
        y=y,
        scale=scale,
        start_time=start_time,
        rng=rng,
    )


def append_detached_visual_atom(
    fixture: dict[str, Any],
    catalog: dict[str, list[HandwritingAtom]],
    label: str,
    target_box: dict[str, float],
    *,
    x_offset: float,
    y_offset: float,
    start_time: float,
    rng: random.Random,
) -> tuple[list[str], dict[str, float], float]:
    atom = choose_atom(catalog, label, rng)
    target_height = max(1.0, target_box["yMax"] - target_box["yMin"])
    scale = min(1.0, (target_height * 0.92) / max(1.0, atom.height))
    x = target_box["xMin"] + x_offset
    y = target_box["yMin"] + y_offset
    return append_atom(
        fixture,
        atom,
        prefix=f"d{len(fixture.get('visualOnlyStrokeIds') or []) + 1:02d}",
        x=x,
        y=y,
        scale=scale,
        start_time=start_time,
        rng=rng,
    )


def append_underline_for_box(
    fixture: dict[str, Any],
    catalog: dict[str, list[HandwritingAtom]],
    target_box: dict[str, float],
    *,
    start_time: float,
    rng: random.Random,
) -> tuple[list[str], dict[str, float], float]:
    atom = choose_atom(catalog, "underline", rng)
    target_width = max(1.0, target_box["xMax"] - target_box["xMin"])
    scale = (target_width * 1.18) / max(1.0, atom.width)
    x = target_box["xMin"] - target_width * 0.08
    y = target_box["yMax"] + 9.0 + rng.uniform(-1.5, 2.5)
    return append_atom(
        fixture,
        atom,
        prefix=f"u{len(fixture.get('visualOnlyStrokeIds') or []) + 1:02d}",
        x=x,
        y=y,
        scale=scale,
        start_time=start_time,
        rng=rng,
    )


def build_realistic_fixture(
    scenario: str = "mixed-marks",
    *,
    seed: int = 0,
    fixture_dir: Path = DEFAULT_REAL_FIXTURE_DIR,
    target_x: float = 88.0,
    target_y: float = 72.0,
) -> dict[str, Any]:
    if scenario not in REALISTIC_SCENARIOS:
        known = ", ".join(REALISTIC_SCENARIOS)
        raise KeyError(f"Unknown realistic handwriting scenario {scenario!r}. Known scenarios: {known}")

    rng = random.Random(seed)
    fixtures_by_slug = {fixture["slug"]: fixture for fixture in load_real_handwriting_fixtures(fixture_dir)}
    source_fixtures = [
        fixtures_by_slug[slug]
        for slug in SCENARIO_SLUGS[scenario]
        if slug in fixtures_by_slug
    ]
    if not source_fixtures:
        raise FileNotFoundError(f"No source real-handwriting fixtures found for scenario {scenario!r}")

    selected = source_fixtures[:1] if scenario != "mixed-marks" else source_fixtures
    composite = empty_generated_fixture(scenario, seed)
    x_cursor = target_x
    y_cursor = target_y
    time_offsets = scenario_time_offsets(scenario, len(selected))

    for source_index, source in enumerate(selected):
        scale = rng.uniform(0.92, 1.08)
        jitter = rng.uniform(-2.5, 2.5)
        origin = bbox_origin(source.get("answerBox")) or bbox_origin(bbox_for_strokes(source.get("strokes") or []))
        dx = x_cursor - origin["x"] + jitter
        dy = y_cursor - origin["y"] - jitter
        append_transformed_source(
            composite,
            source,
            source_index=source_index,
            dx=dx,
            dy=dy,
            scale=scale,
            time_offset=time_offsets[source_index],
            rng=rng,
        )
        source_box = bbox_for_strokes(source.get("strokes") or []) or {"xMin": 0, "yMin": 0, "xMax": 260, "yMax": 120}
        x_cursor += max(260.0, (source_box["xMax"] - source_box["xMin"]) * scale + 74.0)
        if x_cursor > 1180:
            x_cursor = target_x + rng.uniform(10, 50)
            y_cursor += max(190.0, (source_box["yMax"] - source_box["yMin"]) * scale + 70.0)

    finish_generated_fixture(composite)
    return composite


def validate_fixture_against_calibration(
    fixture: dict[str, Any],
    calibration: CalibrationSummary,
    *,
    required_features: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    metrics = fixture_metrics(fixture)
    checks: list[dict[str, Any]] = []
    source_fixture_count = int((fixture.get("sourceStats") or {}).get("sourceFixtureCount") or 1)
    for key in ("stroke_count", "line_count", "point_count", "stroke_duration_ms", "inter_stroke_gap_ms", "pressure_mean"):
        band = calibration.bands.get(key)
        if not band or key not in metrics:
            continue
        informational = key in {"stroke_count", "line_count"} and (source_fixture_count > 1 or band.q95 <= 0)
        values = metrics[key] if isinstance(metrics[key], list) else [metrics[key]]
        if not values:
            continue
        value = mean(float(item) for item in values)
        checks.append({
            "metric": key,
            "value": round_float(value, 4),
            "band": band.to_json(),
            "withinBand": band.padded_min <= value <= band.padded_max,
            "informational": informational,
        })

    features = classify_features(fixture)
    for feature in required_features or []:
        checks.append({
            "feature": feature,
            "present": bool(features.get(feature)),
            "required": True,
        })

    return {
        "ok": all(item.get("withinBand", item.get("present", True)) or item.get("informational", False) for item in checks),
        "sourceKind": calibration.source_kind,
        "sourceCount": calibration.source_count,
        "checks": checks,
        "features": features,
    }


def fixture_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    strokes = [stroke for stroke in payload.get("strokes") or [] if isinstance(stroke, dict)]
    ordered = sorted(strokes, key=lambda stroke: number_or_zero(stroke.get("startTime")))
    strokes_by_id = {str(stroke.get("id")): stroke for stroke in strokes}
    point_counts: list[float] = []
    durations: list[float] = []
    gaps: list[float] = []
    widths: list[float] = []
    heights: list[float] = []
    path_lengths: list[float] = []
    pressure_means: list[float] = []

    previous_end: Optional[float] = None
    for stroke in ordered:
        raw_points = [point for point in stroke.get("rawPoints") or [] if isinstance(point, dict)]
        point_counts.append(float(len(raw_points)))
        start = number_or_zero(stroke.get("startTime"))
        end = number_or_zero(stroke.get("endTime"))
        durations.append(max(0.0, end - start))
        if previous_end is not None:
            gaps.append(max(0.0, start - previous_end))
        previous_end = end

        box = bbox_or_none(stroke.get("canvasBbox")) or bbox_for_points(raw_points)
        if box:
            widths.append(box["xMax"] - box["xMin"])
            heights.append(box["yMax"] - box["yMin"])
        path_lengths.append(path_length(raw_points))
        pressures = [float(point.get("pressure")) for point in raw_points if is_finite(point.get("pressure"))]
        if pressures:
            pressure_means.append(mean(pressures))

    groups = [group for group in payload.get("expectedLineGroups") or [] if isinstance(group, dict)]
    if not groups:
        groups = line_groups_from_fast_result(payload)
    group_sizes = [len(group.get("strokeIds") or []) for group in groups if group.get("strokeIds")]
    line_boxes = [
        box
        for group in groups
        for box in [bbox_for_strokes([strokes_by_id[str(stroke_id)] for stroke_id in group.get("strokeIds") or [] if str(stroke_id) in strokes_by_id])]
        if box
    ]
    line_boxes.sort(key=lambda box: center_y(box))
    line_gaps = [
        max(0.0, line_boxes[index + 1]["yMin"] - line_boxes[index]["yMax"])
        for index in range(len(line_boxes) - 1)
    ]
    answer_box = bbox_or_none(payload.get("answerBox")) or bbox_for_strokes(strokes)
    board_size = board_size_for_payload(payload, answer_box)
    visual_only = payload.get("visualOnlyStrokeIds") or [
        stroke.get("id") for stroke in strokes if stroke.get("visualOnly")
    ]

    return {
        "stroke_count": float(len(strokes)),
        "line_count": float(len(groups) or len(payload.get("expectedLatexLines") or []) or len((payload.get("fastResult") or {}).get("latexLines") or [])),
        "visual_only_count": float(len(visual_only)),
        "point_count": point_counts,
        "stroke_duration_ms": durations,
        "inter_stroke_gap_ms": gaps,
        "stroke_width_px": widths,
        "stroke_height_px": heights,
        "path_length_px": path_lengths,
        "pressure_mean": pressure_means,
        "line_width_px": [box["xMax"] - box["xMin"] for box in line_boxes],
        "line_height_px": [box["yMax"] - box["yMin"] for box in line_boxes],
        "line_gap_px": line_gaps,
        "multi_stroke_group_size": group_sizes,
        "non_sequential_inversions": float(non_sequential_inversions(strokes)),
        "answer_width_px": (answer_box["xMax"] - answer_box["xMin"]) if answer_box else 0.0,
        "answer_height_px": (answer_box["yMax"] - answer_box["yMin"]) if answer_box else 0.0,
        "board_width_px": board_size["width"],
        "board_height_px": board_size["height"],
    }


def fixture_summary(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = fixture_metrics(payload)
    features = classify_features(payload)
    return {
        "fixtureKind": str(payload.get("fixtureKind") or "audit-input-json"),
        "slug": payload.get("slug"),
        "problemLatex": payload.get("problemLatex") or "",
        "strokeCount": int(metrics["stroke_count"]),
        "lineCount": int(metrics["line_count"]),
        "visualOnlyCount": int(metrics["visual_only_count"]),
        "pointCountMedian": round_float(quantile(sorted(metrics["point_count"]), 0.5), 4) if metrics["point_count"] else 0,
        "strokeDurationMedianMs": round_float(quantile(sorted(metrics["stroke_duration_ms"]), 0.5), 4) if metrics["stroke_duration_ms"] else 0,
        "interStrokeGapMedianMs": round_float(quantile(sorted(metrics["inter_stroke_gap_ms"]), 0.5), 4) if metrics["inter_stroke_gap_ms"] else 0,
        "features": sorted(key for key, present in features.items() if present),
    }


def classify_features(payload: dict[str, Any]) -> dict[str, bool]:
    strokes = [stroke for stroke in payload.get("strokes") or [] if isinstance(stroke, dict)]
    groups = [group for group in payload.get("expectedLineGroups") or [] if isinstance(group, dict)]
    if not groups:
        groups = line_groups_from_fast_result(payload)
    group_sizes = [
        len(group.get("strokeIds") or [])
        for group in groups
    ]
    text = json.dumps({
        "slug": payload.get("slug"),
        "description": payload.get("description"),
        "visualMarks": payload.get("visualMarks") or [],
        "knownDiscrepancyTypes": payload.get("knownDiscrepancyTypes") or [],
        "realisticFeatures": payload.get("realisticFeatures") or [],
    }).lower()
    return {
        "has_pressure": any(
            is_finite(point.get("pressure"))
            for stroke in strokes
            for point in stroke.get("rawPoints") or []
            if isinstance(point, dict)
        ),
        "has_multi_stroke_symbols": any(size > 1 for size in group_sizes),
        "has_visual_only_marks": bool(payload.get("visualOnlyStrokeIds")) or any(stroke.get("visualOnly") for stroke in strokes),
        "has_circled_answer": "circle" in text or "circled" in text or "oval" in text,
        "has_crossout_or_scratch": "cross" in text or "scratch" in text or "discard" in text,
        "has_non_sequential_writing": non_sequential_inversions(strokes) > 0,
    }


def line_groups_from_fast_result(payload: dict[str, Any]) -> list[dict[str, Any]]:
    fast_result = payload.get("fastResult") or {}
    groups: list[dict[str, Any]] = []
    for index, line in enumerate(fast_result.get("lines") or []):
        if not isinstance(line, dict):
            continue
        stroke_ids = [str(stroke_id) for stroke_id in line.get("strokeIds") or []]
        if not stroke_ids:
            continue
        groups.append({
            "lineIndex": index,
            "latex": str(line.get("acceptedLatex") or line.get("latex") or line.get("bestLatex") or ""),
            "strokeIds": stroke_ids,
            "source": "fastResult.lines",
        })
    return groups


def board_size_for_payload(payload: dict[str, Any], fallback_box: Optional[dict[str, float]]) -> dict[str, float]:
    board_size = payload.get("boardSize")
    if isinstance(board_size, dict) and is_finite(board_size.get("width")) and is_finite(board_size.get("height")):
        return {
            "width": float(board_size["width"]),
            "height": float(board_size["height"]),
        }
    box = fallback_box or bbox_for_strokes(payload.get("strokes") or [])
    return {
        "width": round_float(max(1.0, float(box["xMax"]) + 24.0), 2) if box else 1.0,
        "height": round_float(max(1.0, float(box["yMax"]) + 24.0), 2) if box else 1.0,
    }


def append_transformed_source(
    target: dict[str, Any],
    source: dict[str, Any],
    *,
    source_index: int,
    dx: float,
    dy: float,
    scale: float,
    time_offset: float,
    rng: random.Random,
) -> None:
    id_map: dict[str, str] = {}
    source_strokes = source.get("strokes") or []
    source_time_base = min((number_or_zero(stroke.get("startTime")) for stroke in source_strokes), default=0.0)
    for stroke_index, stroke in enumerate(source_strokes, start=1):
        old_id = str(stroke.get("id") or stroke_index)
        new_id = f"r{source_index + 1:02d}s{stroke_index:03d}"
        id_map[old_id] = new_id
        start = time_offset + max(0.0, number_or_zero(stroke.get("startTime")) - source_time_base) + rng.uniform(0, 18)
        end = time_offset + max(0.0, number_or_zero(stroke.get("endTime")) - source_time_base) + rng.uniform(0, 18)
        if end < start:
            end = start
        raw_points = transform_points(stroke.get("rawPoints") or [], dx, dy, scale, rng)
        outline_points = transform_points(stroke.get("outlinePoints") or stroke.get("rawPoints") or [], dx, dy, scale, rng)
        box = bbox_for_points(raw_points)
        target["strokes"].append({
            "id": new_id,
            "startTime": round_float(start, 1),
            "endTime": round_float(end, 1),
            "rawPoints": raw_points,
            "outlinePoints": outline_points or raw_points,
            "color": stroke.get("color") or "#000000",
            "canvasBbox": box,
            "sourceStrokeId": old_id,
            "sourceFixtureSlug": source.get("slug"),
        })

    for group in source.get("expectedLineGroups") or []:
        mapped = [id_map[str(stroke_id)] for stroke_id in group.get("strokeIds") or [] if str(stroke_id) in id_map]
        if not mapped:
            continue
        line_index = len(target["expectedLineGroups"])
        target["expectedLineGroups"].append({
            "lineIndex": line_index,
            "latex": str(group.get("latex") or "").strip(),
            "strokeIds": mapped,
            "source": "realistic-transformed-real-trace",
            "sourceFixtureSlug": source.get("slug"),
            "sourceCandidateId": group.get("sourceCandidateId"),
        })
        if str(group.get("latex") or "").strip():
            target["expectedLatexLines"].append(str(group.get("latex")).strip())

    for stroke_id in source.get("visualOnlyStrokeIds") or []:
        if str(stroke_id) in id_map:
            target["visualOnlyStrokeIds"].append(id_map[str(stroke_id)])

    for mark in source.get("visualMarks") or []:
        target["visualMarks"].append(mark)
    for kind in source.get("knownDiscrepancyTypes") or []:
        if kind not in target["knownDiscrepancyTypes"]:
            target["knownDiscrepancyTypes"].append(kind)


def finish_generated_fixture(fixture: dict[str, Any]) -> None:
    strokes = fixture["strokes"]
    board_box = bbox_for_strokes(strokes) or {"xMin": 0, "yMin": 0, "xMax": 1, "yMax": 1}
    board_size = {
        "width": round_float(max(1.0, board_box["xMax"] + 48.0), 2),
        "height": round_float(max(1.0, board_box["yMax"] + 48.0), 2),
    }
    fixture["boardSize"] = board_size
    fixture["answerBox"] = {
        "xMin": round_float(max(0.0, board_box["xMin"] - 24.0), 2),
        "yMin": round_float(max(0.0, board_box["yMin"] - 24.0), 2),
        "xMax": round_float(board_box["xMax"] + 24.0, 2),
        "yMax": round_float(board_box["yMax"] + 24.0, 2),
    }
    visual_ids = set(fixture.get("visualOnlyStrokeIds") or [])
    line_labels = {
        str(stroke_id): (int(group["lineIndex"]), str(group.get("latex") or ""))
        for group in fixture.get("expectedLineGroups") or []
        for stroke_id in group.get("strokeIds") or []
    }

    for stroke in strokes:
        stroke["bbox"] = normalize_box(stroke["canvasBbox"], board_size)
        stroke["points"] = normalized_points_for(stroke, board_size)
        if stroke["id"] in visual_ids:
            stroke["visualOnly"] = True
        label = line_labels.get(stroke["id"])
        if label:
            stroke["expectedLineIndex"] = label[0]
            stroke["expectedLatex"] = label[1]

    strokes.sort(key=lambda stroke: number_or_zero(stroke.get("startTime")))
    previous = None
    for stroke in strokes:
        stroke["relationsToPrev"] = relation_to_previous(stroke, previous) if previous else {
            "dx": 0,
            "dy": 0,
            "dt": 0,
            "overlapRatio": 0,
        }
        previous = stroke

    feature_map = classify_features(fixture)
    fixture["realisticFeatures"] = sorted(key for key, present in feature_map.items() if present)
    fixture["sourceStats"] = {
        "sourceFixtureCount": len(set(stroke.get("sourceFixtureSlug") for stroke in strokes)),
        "generatedStrokeCount": len(strokes),
        "generatedLineCount": len(fixture.get("expectedLineGroups") or []),
    }


def empty_generated_fixture(scenario: str, seed: int) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "fixtureKind": "realistic-handwriting-trace",
        "slug": f"generated-{scenario}-{seed}",
        "description": f"Generated from real audit strokes for the {scenario} scenario.",
        "sourceAuditId": "generated-from-distilled-real-fixtures",
        "problemLatex": "",
        "problemMetadata": {"scenario": scenario, "seed": seed},
        "triggerReasons": ["realistic-handwriting-harness"],
        "knownDiscrepancyTypes": [],
        "fastLatexLines": [],
        "expectedLatexLines": [],
        "visualMarks": [],
        "answerBox": None,
        "problemBox": None,
        "boardSize": None,
        "detections": [],
        "strokes": [],
        "expectedLineGroups": [],
        "visualOnlyStrokeIds": [],
    }


def scenario_time_offsets(scenario: str, count: int) -> list[float]:
    if scenario in {"mixed-marks", "non-sequential"} and count >= 2:
        return [1600.0 * index for index in reversed(range(count))]
    return [1600.0 * index for index in range(count)]


def transform_points(points: Sequence[Any], dx: float, dy: float, scale: float, rng: random.Random) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for point in points:
        if not isinstance(point, dict) or not is_finite(point.get("x")) or not is_finite(point.get("y")):
            continue
        pressure = float(point.get("pressure")) if is_finite(point.get("pressure")) else 0.5
        out.append({
            "x": round_float(float(point["x"]) * scale + dx + rng.uniform(-0.65, 0.65), 2),
            "y": round_float(float(point["y"]) * scale + dy + rng.uniform(-0.65, 0.65), 2),
            "pressure": round_float(min(1.0, max(0.0, pressure + rng.uniform(-0.025, 0.025))), 3),
        })
    return out


def normalized_points_for(stroke: dict[str, Any], board_size: dict[str, float]) -> list[dict[str, float]]:
    raw = stroke.get("rawPoints") or []
    if not raw:
        return []
    duration = max(1.0, number_or_zero(stroke.get("endTime")) - number_or_zero(stroke.get("startTime")))
    last = max(1, len(raw) - 1)
    return [
        {
            "x": round_float(float(point["x"]) / board_size["width"], 4),
            "y": round_float(float(point["y"]) / board_size["height"], 4),
            "t": round_float((index / last) * duration, 1),
            "pressure": round_float(float(point.get("pressure", 0.5)), 3),
        }
        for index, point in enumerate(raw)
    ]


def relation_to_previous(stroke: dict[str, Any], previous: Optional[dict[str, Any]]) -> dict[str, float]:
    if previous is None:
        return {"dx": 0, "dy": 0, "dt": 0, "overlapRatio": 0}
    curr = stroke["bbox"]
    prev = previous["bbox"]
    return {
        "dx": round_float(center_x(curr) - center_x(prev), 4),
        "dy": round_float(center_y(curr) - center_y(prev), 4),
        "dt": round_float(number_or_zero(stroke.get("startTime")) - number_or_zero(previous.get("startTime")), 1),
        "overlapRatio": round_float(iou(curr, prev), 4),
    }


def non_sequential_inversions(strokes: Sequence[dict[str, Any]]) -> int:
    timed = [
        (number_or_zero(stroke.get("startTime")), center_y(bbox_or_none(stroke.get("canvasBbox")) or bbox_for_points(stroke.get("rawPoints") or [])))
        for stroke in strokes
        if stroke.get("canvasBbox") or stroke.get("rawPoints")
    ]
    inversions = 0
    for index, (_, y_left) in enumerate(timed):
        for _, y_right in timed[index + 1 :]:
            if y_left > y_right + 24.0:
                inversions += 1
    return inversions


def quantile_band(values: Sequence[float]) -> DistributionBand:
    cleaned = sorted(float(value) for value in values if is_finite(value))
    if not cleaned:
        return DistributionBand(0, 0, 0, 0, 0, 0)
    return DistributionBand(
        minimum=cleaned[0],
        q05=quantile(cleaned, 0.05),
        median=quantile(cleaned, 0.5),
        q95=quantile(cleaned, 0.95),
        maximum=cleaned[-1],
        count=len(cleaned),
    )


def quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return float(sorted_values[lower])
    weight = pos - lower
    return float(sorted_values[lower]) * (1 - weight) + float(sorted_values[upper]) * weight


def bbox_origin(box: Optional[dict[str, float]]) -> Optional[dict[str, float]]:
    if not box:
        return None
    return {"x": float(box["xMin"]), "y": float(box["yMin"])}


def bbox_for_strokes(strokes: Sequence[dict[str, Any]]) -> Optional[dict[str, float]]:
    boxes = [bbox_or_none(stroke.get("canvasBbox")) for stroke in strokes]
    boxes = [box for box in boxes if box]
    if not boxes:
        return None
    return union_boxes(boxes)


def union_boxes(boxes: Sequence[dict[str, float]]) -> dict[str, float]:
    if not boxes:
        return {"xMin": 0, "yMin": 0, "xMax": 1, "yMax": 1}
    return {
        "xMin": min(box["xMin"] for box in boxes),
        "yMin": min(box["yMin"] for box in boxes),
        "xMax": max(box["xMax"] for box in boxes),
        "yMax": max(box["yMax"] for box in boxes),
    }


def bbox_for_points(points: Sequence[dict[str, Any]]) -> dict[str, float]:
    usable = [point for point in points if is_finite(point.get("x")) and is_finite(point.get("y"))]
    if not usable:
        return {"xMin": 0, "yMin": 0, "xMax": 1, "yMax": 1}
    x_min = min(float(point["x"]) for point in usable)
    y_min = min(float(point["y"]) for point in usable)
    x_max = max(float(point["x"]) for point in usable)
    y_max = max(float(point["y"]) for point in usable)
    if x_max <= x_min:
        x_max = x_min + 0.25
    if y_max <= y_min:
        y_max = y_min + 0.25
    return {
        "xMin": round_float(x_min, 2),
        "yMin": round_float(y_min, 2),
        "xMax": round_float(x_max, 2),
        "yMax": round_float(y_max, 2),
    }


def bbox_or_none(value: Any) -> Optional[dict[str, float]]:
    if not isinstance(value, dict):
        return None
    try:
        box = {
            "xMin": float(value["xMin"]),
            "yMin": float(value["yMin"]),
            "xMax": float(value["xMax"]),
            "yMax": float(value["yMax"]),
        }
    except (KeyError, TypeError, ValueError):
        return None
    if box["xMax"] <= box["xMin"] or box["yMax"] <= box["yMin"]:
        return None
    return box


def normalize_box(box: dict[str, float], board_size: dict[str, float]) -> dict[str, float]:
    return {
        "xMin": round_float(box["xMin"] / board_size["width"], 4),
        "yMin": round_float(box["yMin"] / board_size["height"], 4),
        "xMax": round_float(box["xMax"] / board_size["width"], 4),
        "yMax": round_float(box["yMax"] / board_size["height"], 4),
    }


def path_length(points: Sequence[dict[str, Any]]) -> float:
    length = 0.0
    previous = None
    for point in points:
        if not is_finite(point.get("x")) or not is_finite(point.get("y")):
            continue
        current = (float(point["x"]), float(point["y"]))
        if previous is not None:
            length += math.dist(previous, current)
        previous = current
    return length


def iou(left: dict[str, float], right: dict[str, float]) -> float:
    x_min = max(left["xMin"], right["xMin"])
    y_min = max(left["yMin"], right["yMin"])
    x_max = min(left["xMax"], right["xMax"])
    y_max = min(left["yMax"], right["yMax"])
    inter = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    area_left = max(0.0, left["xMax"] - left["xMin"]) * max(0.0, left["yMax"] - left["yMin"])
    area_right = max(0.0, right["xMax"] - right["xMin"]) * max(0.0, right["yMax"] - right["yMin"])
    union = area_left + area_right - inter
    return inter / union if union > 0 else 0.0


def center_x(box: dict[str, float]) -> float:
    return (float(box["xMin"]) + float(box["xMax"])) / 2


def center_y(box: dict[str, float]) -> float:
    return (float(box["yMin"]) + float(box["yMax"])) / 2


def number_or_zero(value: Any) -> float:
    return float(value) if is_finite(value) else 0.0


def is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def round_float(value: float, digits: int) -> float:
    return round(float(value), digits)


def relative_path_for_report(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=REALISTIC_SCENARIOS, default="mixed-marks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_REAL_FIXTURE_DIR)
    parser.add_argument("--catalog-path", type=Path, default=DEFAULT_HANDWRITING_CATALOG)
    parser.add_argument("--audit-log-dir", type=Path, default=DEFAULT_AUDIT_LOG_DIR)
    parser.add_argument("--audit-limit", type=int, default=None)
    parser.add_argument("--hybrid-linear", action="store_true", help="Generate a hybrid real-stroke linear-equation fixture.")
    parser.add_argument("--hybrid-complex-math", action="store_true", help="Generate a hybrid real-stroke complex-math fixture.")
    parser.add_argument("--hybrid-pack", choices=sorted(HARNESS_PACKS), default=None, help="Generate a log-observed real-handwriting harness pack.")
    parser.add_argument("--random-linear", action="store_true", help="Sample a new seeded linear equation instead of using explicit coefficients.")
    parser.add_argument("--linear-a", type=int, default=3)
    parser.add_argument("--linear-b", type=int, default=2)
    parser.add_argument("--linear-x", type=int, default=4)
    parser.add_argument("--linear-min-coefficient", type=int, default=2)
    parser.add_argument("--linear-max-coefficient", type=int, default=9)
    parser.add_argument("--linear-min-constant", type=int, default=1)
    parser.add_argument("--linear-max-constant", type=int, default=9)
    parser.add_argument("--linear-min-solution", type=int, default=1)
    parser.add_argument("--linear-max-solution", type=int, default=9)
    parser.add_argument("--linear-max-result", type=int, default=99)
    parser.add_argument("--include-crossout", action="store_true")
    parser.add_argument("--no-circle", action="store_true")
    parser.add_argument("--no-detached-annotations", action="store_true")
    parser.add_argument("--non-sequential-final", action="store_true")
    parser.add_argument(
        "--include-fixtures-with-audits",
        action="store_true",
        help="Include committed distilled fixtures even when live audit input.json records are available.",
    )
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Print only the Phase 1 corpus and distribution report; do not generate a fixture.",
    )
    parser.add_argument(
        "--calibration-output",
        type=Path,
        default=None,
        help="Write the Phase 1 corpus and distribution report to this JSON path.",
    )
    parser.add_argument(
        "--dashboard-output",
        type=Path,
        default=None,
        help="Write a generated-vs-real handwriting dashboard report to this JSON path.",
    )
    parser.add_argument("--dashboard-only", action="store_true", help="Print only the generated-vs-real dashboard report.")
    parser.add_argument("--output", type=Path, default=None, help="Write generated fixture JSON to this path.")
    parser.add_argument("--include-fixture", action="store_true", help="Print the full generated fixture JSON.")
    parser.add_argument("--print-calibration", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    calibration_report = build_calibration_report(
        audit_log_dir=args.audit_log_dir,
        fixture_dir=args.fixture_dir,
        audit_limit=args.audit_limit,
        include_fixtures_with_audits=args.include_fixtures_with_audits,
    )
    if args.calibration_output:
        args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_output.write_text(
            json.dumps(calibration_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.calibration_only:
        print(json.dumps(calibration_report, indent=2, sort_keys=True))
        return 0
    if args.dashboard_output or args.dashboard_only:
        dashboard = build_harness_dashboard_report(
            default_dashboard_fixtures(seed=args.seed, fixture_dir=args.fixture_dir, catalog_path=args.catalog_path),
            audit_log_dir=args.audit_log_dir,
            fixture_dir=args.fixture_dir,
            audit_limit=args.audit_limit,
        )
        if args.dashboard_output:
            args.dashboard_output.parent.mkdir(parents=True, exist_ok=True)
            args.dashboard_output.write_text(json.dumps(dashboard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.dashboard_only:
            print(json.dumps(dashboard, indent=2, sort_keys=True))
            return 0

    if args.hybrid_complex_math:
        fixture = build_hybrid_complex_math_fixture(
            seed=args.seed,
            catalog_path=args.catalog_path,
            fixture_dir=args.fixture_dir,
            include_circle=not args.no_circle,
            include_detached_annotations=not args.no_detached_annotations,
            include_crossout=args.include_crossout,
        )
    elif args.hybrid_pack:
        fixture = build_log_observed_pack_fixture(
            args.hybrid_pack,
            seed=args.seed,
            fixture_dir=args.fixture_dir,
        )
    elif args.hybrid_linear:
        if args.random_linear:
            fixture = build_random_hybrid_linear_equation_fixture(
                seed=args.seed,
                min_coefficient=args.linear_min_coefficient,
                max_coefficient=args.linear_max_coefficient,
                min_constant=args.linear_min_constant,
                max_constant=args.linear_max_constant,
                min_solution=args.linear_min_solution,
                max_solution=args.linear_max_solution,
                max_result=args.linear_max_result,
                catalog_path=args.catalog_path,
                fixture_dir=args.fixture_dir,
                include_circle=not args.no_circle,
                include_crossout=args.include_crossout,
                non_sequential_final=args.non_sequential_final,
            )
        else:
            fixture = build_hybrid_linear_equation_fixture(
                a=args.linear_a,
                b=args.linear_b,
                x_value=args.linear_x,
                seed=args.seed,
                catalog_path=args.catalog_path,
                fixture_dir=args.fixture_dir,
                include_circle=not args.no_circle,
                include_crossout=args.include_crossout,
                non_sequential_final=args.non_sequential_final,
            )
    else:
        fixture = build_realistic_fixture(args.scenario, seed=args.seed, fixture_dir=args.fixture_dir)
    calibration = summarize_distribution(
        [
            record.payload
            for record in load_corpus_records(
                audit_log_dir=args.audit_log_dir,
                fixture_dir=args.fixture_dir,
                audit_limit=args.audit_limit,
                include_fixtures_with_audits=args.include_fixtures_with_audits,
            )
        ],
        source_kind=calibration_report["calibration"]["sourceKind"],
    )
    required_features = ["has_pressure", "has_multi_stroke_symbols"]
    if fixture.get("visualOnlyStrokeIds") or any(stroke.get("visualOnly") for stroke in fixture.get("strokes") or []):
        required_features.append("has_visual_only_marks")
    report = validate_fixture_against_calibration(
        fixture,
        calibration,
        required_features=required_features,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fixture_summary = {
        "slug": fixture["slug"],
        "scenario": fixture.get("problemMetadata", {}).get("scenario") or args.scenario,
        "seed": args.seed,
        "problemLatex": fixture.get("problemLatex") or "",
        "strokeCount": len(fixture.get("strokes") or []),
        "lineCount": len(fixture.get("expectedLineGroups") or []),
        "visualOnlyStrokeCount": len(fixture.get("visualOnlyStrokeIds") or []),
        "realisticFeatures": fixture.get("realisticFeatures") or [],
    }
    if args.output:
        fixture_summary["path"] = str(args.output)
    payload = {
        "fixture": fixture if args.include_fixture else fixture_summary,
        "calibration": calibration.to_json() if args.print_calibration else {
            "sourceKind": calibration.source_kind,
            "sourceCount": calibration.source_count,
        },
        "calibrationReportPath": str(args.calibration_output) if args.calibration_output else None,
        "validation": report,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
