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
from typing import Any, Iterable, Optional, Sequence


TESTING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTING_DIR.parent
DEFAULT_REAL_FIXTURE_DIR = TESTING_DIR / "fixtures" / "real_handwriting"
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


def load_real_handwriting_fixtures(fixture_dir: Path = DEFAULT_REAL_FIXTURE_DIR) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(Path(fixture_dir).glob("*.json"))
    ]


def load_audit_input_payloads(audit_log_dir: Path = DEFAULT_AUDIT_LOG_DIR, limit: Optional[int] = None) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    paths = sorted(Path(audit_log_dir).glob("**/input.json"))
    if limit is not None:
        paths = paths[: int(limit)]
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("strokes"), list):
            payloads.append(payload)
    return payloads


def build_calibration_summary(
    *,
    audit_log_dir: Path = DEFAULT_AUDIT_LOG_DIR,
    fixture_dir: Path = DEFAULT_REAL_FIXTURE_DIR,
    audit_limit: Optional[int] = None,
) -> CalibrationSummary:
    audit_payloads = load_audit_input_payloads(audit_log_dir, limit=audit_limit)
    if audit_payloads:
        return summarize_distribution(audit_payloads, source_kind="audit-input-json")
    return summarize_distribution(load_real_handwriting_fixtures(fixture_dir), source_kind="distilled-real-fixtures")


def summarize_distribution(payloads: Sequence[dict[str, Any]], *, source_kind: str) -> CalibrationSummary:
    metric_values: dict[str, list[float]] = {
        "stroke_count": [],
        "line_count": [],
        "visual_only_count": [],
        "point_count": [],
        "stroke_duration_ms": [],
        "inter_stroke_gap_ms": [],
        "stroke_width_px": [],
        "stroke_height_px": [],
        "path_length_px": [],
        "pressure_mean": [],
        "multi_stroke_group_size": [],
        "non_sequential_inversions": [],
    }
    feature_counts = {
        "has_pressure": 0,
        "has_multi_stroke_symbols": 0,
        "has_visual_only_marks": 0,
        "has_circled_answer": 0,
        "has_crossout_or_scratch": 0,
        "has_non_sequential_writing": 0,
    }

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
    group_sizes = [len(group.get("strokeIds") or []) for group in groups if group.get("strokeIds")]
    visual_only = payload.get("visualOnlyStrokeIds") or [
        stroke.get("id") for stroke in strokes if stroke.get("visualOnly")
    ]

    return {
        "stroke_count": float(len(strokes)),
        "line_count": float(len(groups) or len(payload.get("expectedLatexLines") or [])),
        "visual_only_count": float(len(visual_only)),
        "point_count": point_counts,
        "stroke_duration_ms": durations,
        "inter_stroke_gap_ms": gaps,
        "stroke_width_px": widths,
        "stroke_height_px": heights,
        "path_length_px": path_lengths,
        "pressure_mean": pressure_means,
        "multi_stroke_group_size": group_sizes,
        "non_sequential_inversions": float(non_sequential_inversions(strokes)),
    }


def classify_features(payload: dict[str, Any]) -> dict[str, bool]:
    strokes = [stroke for stroke in payload.get("strokes") or [] if isinstance(stroke, dict)]
    group_sizes = [
        len(group.get("strokeIds") or [])
        for group in payload.get("expectedLineGroups") or []
        if isinstance(group, dict)
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=REALISTIC_SCENARIOS, default="mixed-marks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_REAL_FIXTURE_DIR)
    parser.add_argument("--audit-log-dir", type=Path, default=DEFAULT_AUDIT_LOG_DIR)
    parser.add_argument("--audit-limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Write generated fixture JSON to this path.")
    parser.add_argument("--include-fixture", action="store_true", help="Print the full generated fixture JSON.")
    parser.add_argument("--print-calibration", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    fixture = build_realistic_fixture(args.scenario, seed=args.seed, fixture_dir=args.fixture_dir)
    calibration = build_calibration_summary(
        audit_log_dir=args.audit_log_dir,
        fixture_dir=args.fixture_dir,
        audit_limit=args.audit_limit,
    )
    report = validate_fixture_against_calibration(
        fixture,
        calibration,
        required_features=[
            "has_pressure",
            "has_multi_stroke_symbols",
            "has_visual_only_marks",
        ],
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fixture_summary = {
        "slug": fixture["slug"],
        "scenario": args.scenario,
        "seed": args.seed,
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
        "validation": report,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
