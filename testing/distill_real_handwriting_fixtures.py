#!/usr/bin/env python3
"""Distill local VLM audit logs into deterministic real-handwriting fixtures."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_LOG_DIR = Path(
    os.environ.get("WHITEBOARD_AUDIT_LOG_DIR", "/Users/jpovj/Documents/dev/log_whiteboard_3")
).expanduser()
DEFAULT_OUTPUT_DIR = REPO_ROOT / "testing" / "fixtures" / "real_handwriting"

DEFAULT_FIXTURES: Sequence[dict[str, str]] = (
    {
        "audit_id": "audit_20260630T190746242118Z_6345194572",
        "slug": "crossout-scratch-division",
        "description": "Crossed-out and scratch operation work; fast four lines vs VLM two.",
    },
    {
        "audit_id": "audit_20260630T201700533650Z_b7f7e07016",
        "slug": "circled-final-answer",
        "description": "Circled final answer with line-count disagreement.",
    },
    {
        "audit_id": "audit_20260701T130355711645Z_7123ff1932",
        "slug": "circled-intermediate-result",
        "description": "Circled intermediate result; fast four lines vs VLM two.",
    },
    {
        "audit_id": "audit_20260701T211642649058Z_042676e831",
        "slug": "fraction-subtraction-oversegmented",
        "description": "Fraction subtraction over-segmentation.",
    },
    {
        "audit_id": "audit_20260701T220336197762Z_92f670ddce",
        "slug": "detached-operation-radical-fraction",
        "description": "Detached operation and radical fraction behavior.",
    },
    {
        "audit_id": "audit_20260630T204322495004Z_f13be89ef3",
        "slug": "compact-plus-minus-solution",
        "description": "Compact plus-minus solution-set mismatch.",
    },
    {
        "audit_id": "audit_20260630T144334954639Z_7784e96969",
        "slug": "circled-x-equals-four",
        "description": "Large oval around final x=4 answer causing OCR to read parentheses.",
    },
    {
        "audit_id": "audit_20260701T154037909609Z_cfcdad97a1",
        "slug": "wide-fraction-answer-setup",
        "description": "Wide handwritten fraction answer split by timing into numerator and denominator rows.",
    },
    {
        "audit_id": "audit_20260701T220426951173Z_2eafc3f6f8",
        "slug": "radical-fraction-simplification",
        "description": "Radical fraction simplification with late detached equals strokes and stacked fraction bodies.",
    },
    {
        "audit_id": "audit_20260701T153744697859Z_cc6a30ef8a",
        "slug": "plus-minus-fraction-solution",
        "description": "Quadratic solve with a plus-minus fractional final answer.",
    },
    {
        "audit_id": "audit_20260630T192728021251Z_d70549b67a",
        "slug": "fragmented-invalid-expression",
        "description": "Invalid expression answer that must stay one semantic line instead of fragmenting into correct-looking single digits.",
    },
    {
        "audit_id": "audit_20260701T220344096316Z_166bcd217a",
        "slug": "radical-fraction-simplification-no-leading-equals",
        "description": "Radical fraction simplification without leading equals strokes; stale OCR split numerator and radical body.",
    },
    {
        "audit_id": "audit_20260701T220432397511Z_d626c2d905",
        "slug": "radical-fraction-simplification-leading-minus",
        "description": "Radical fraction simplification with leading equals/minus-like strokes and stale OCR fragmentation.",
    },
)

REVIEWED_LINE_GROUP_OVERRIDES: dict[str, list[dict[str, Any]]] = {
    "audit_20260701T154037909609Z_cfcdad97a1": [
        {
            "lineIndex": 0,
            "latex": "x = \\frac{11+9}{2}",
            "strokeIds": ["s001", "s002", "s003", "s004", "s005", "s006", "s007", "s008", "s009", "s010", "s011"],
            "source": "reviewed-vlm-semantic-lines",
            "sourceCandidateId": "parent_s001|s002|s003|s004|s005|s006|s007|s008|s009|s010|s011",
        },
    ],
    "audit_20260630T144334954639Z_7784e96969": [
        {
            "lineIndex": 0,
            "latex": "x=4",
            "strokeIds": ["s001", "s002", "s003", "s004", "s005", "s006"],
            "source": "reviewed-vlm-semantic-lines",
            "sourceCandidateId": "strict_s001|s002|s003|s004|s005|s006",
        },
    ],
    "audit_20260630T190746242118Z_6345194572": [
        {
            "lineIndex": 0,
            "latex": "\\frac{2x}{2} = \\frac{10}{2}",
            "strokeIds": ["s007", "s008", "s010", "s011", "s013", "s014", "s015", "s016", "s017", "s018", "s019"],
            "source": "reviewed-vlm-semantic-lines",
            "sourceCandidateId": "fraction-stack-line_s007|s008|s010|s011|s013|s014|s015|s016|s017|s018|s019",
        },
        {
            "lineIndex": 1,
            "latex": "x=5",
            "strokeIds": ["s020", "s021", "s022", "s023", "s024"],
            "source": "reviewed-vlm-semantic-lines",
            "sourceCandidateId": "raw-row-line_s020|s021|s022|s023|s024",
        },
    ],
    "audit_20260701T130355711645Z_7123ff1932": [
        {
            "lineIndex": 0,
            "latex": "x-1=10",
            "strokeIds": ["s005", "s006", "s007", "s008", "s009", "s010", "s011", "s012"],
            "source": "reviewed-vlm-semantic-lines",
            "sourceCandidateId": "loose_s005|s006|s007|s008|s009|s010|s011|s012",
        },
        {
            "lineIndex": 1,
            "latex": "x=9",
            "strokeIds": ["s013", "s014", "s015", "s016", "s017", "s018"],
            "source": "reviewed-vlm-semantic-lines",
            "sourceCandidateId": "loose_s013|s014|s015|s016|s017|s018",
        },
    ],
    "audit_20260701T211642649058Z_042676e831": [
        {
            "lineIndex": 0,
            "latex": "\\frac{1}{2} - \\frac{1}{2}",
            "strokeIds": ["s001", "s002", "s003", "s004", "s005", "s006", "s007"],
            "source": "reviewed-vlm-semantic-lines",
            "sourceCandidateId": "loose_s001|s002|s003|s004|s005|s006|s007",
        },
        {
            "lineIndex": 1,
            "latex": "0",
            "strokeIds": ["s008"],
            "source": "reviewed-vlm-semantic-lines",
            "sourceCandidateId": "loose_s008",
        },
    ],
    "audit_20260701T220336197762Z_92f670ddce": [
        {
            "lineIndex": 0,
            "latex": "\\frac{7}{\\sqrt{9}}",
            "strokeIds": ["s001", "s002", "s003", "s004", "s005"],
            "source": "reviewed-vlm-semantic-lines",
            "sourceCandidateId": "parent_s001|s002|s003|s004|s005",
        },
    ],
    "audit_20260701T220426951173Z_2eafc3f6f8": [
        {
            "lineIndex": 0,
            "latex": "= \\frac{7}{\\sqrt{9}}",
            "strokeIds": ["s010", "s001", "s011", "s002", "s003", "s004", "s005"],
            "source": "reviewed-vlm-semantic-lines",
            "sourceCandidateId": "reviewed_s010|s001|s011|s002|s003|s004|s005",
        },
        {
            "lineIndex": 1,
            "latex": "= \\frac{7}{3}",
            "strokeIds": ["s012", "s006", "s013", "s007", "s008", "s009"],
            "source": "reviewed-vlm-semantic-lines",
            "sourceCandidateId": "reviewed_s012|s006|s013|s007|s008|s009",
        },
    ],
    "audit_20260701T220344096316Z_166bcd217a": [
        {
            "lineIndex": 0,
            "latex": "\\frac{7}{\\sqrt{9}}",
            "strokeIds": ["s001", "s002", "s003", "s004", "s005"],
            "source": "reviewed-vlm-semantic-lines",
            "sourceCandidateId": "fraction-stack-line_s001|s002|s003|s004|s005",
        },
        {
            "lineIndex": 1,
            "latex": "\\frac{7}{3}",
            "strokeIds": ["s006", "s007", "s008", "s009"],
            "source": "reviewed-vlm-semantic-lines",
            "sourceCandidateId": "fraction-stack-line_s006|s007|s008|s009",
        },
    ],
    "audit_20260701T220432397511Z_d626c2d905": [
        {
            "lineIndex": 0,
            "latex": "= \\frac{7}{\\sqrt{9}}",
            "strokeIds": ["s001", "s002", "s003", "s004", "s005", "s010", "s011", "s014", "s015"],
            "source": "reviewed-vlm-semantic-lines",
            "sourceCandidateId": "fraction-stack-line_s001|s002|s003|s004|s005|s010|s011|s014|s015",
        },
        {
            "lineIndex": 1,
            "latex": "= \\frac{7}{3}",
            "strokeIds": ["s006", "s007", "s008", "s009", "s012", "s013"],
            "source": "reviewed-vlm-semantic-lines",
            "sourceCandidateId": "fraction-stack-line_s006|s007|s008|s009|s012|s013",
        },
    ],
    "audit_20260630T204322495004Z_f13be89ef3": [
        {
            "lineIndex": 0,
            "latex": "x = \\pm 2",
            "strokeIds": ["s001", "s002", "s003", "s004", "s005", "s006", "s007", "s008"],
            "source": "reviewed-vlm-semantic-lines",
            "sourceCandidateId": "parent_s001|s002|s003|s004|s005|s006|s007|s008",
        },
    ],
    "audit_20260630T192728021251Z_d70549b67a": [
        {
            "lineIndex": 0,
            "latex": "1+1=2",
            "strokeIds": ["s001", "s002", "s003", "s004", "s005", "s006", "s007"],
            "source": "reviewed-vlm-semantic-lines",
            "sourceCandidateId": "parent_s001|s002|s003|s004|s005|s006|s007",
        },
    ],
}

REVIEWED_VISUAL_ONLY_STROKES: dict[str, list[str]] = {
    "audit_20260630T144334954639Z_7784e96969": ["s007"],
    "audit_20260630T190746242118Z_6345194572": [
        "s001", "s002", "s003", "s004", "s005", "s006", "s009", "s012",
    ],
    "audit_20260701T130355711645Z_7123ff1932": [
        "s001", "s002", "s003", "s004",
    ],
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    selected = selected_fixture_specs(args.audit_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for spec in selected:
        audit_dir = find_audit_dir(args.audit_log_dir, spec["audit_id"])
        fixture = build_fixture(audit_dir, spec)
        output_path = args.output_dir / f"{spec['slug']}.json"
        if args.dry_run:
            print(json.dumps(fixture, indent=2, sort_keys=True))
        else:
            output_path.write_text(
                json.dumps(fixture, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            written.append(output_path)

    if written:
        for path in written:
            print(path)
    return 0


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-log-dir",
        type=Path,
        default=DEFAULT_AUDIT_LOG_DIR,
        help="Root audit log directory. Defaults to WHITEBOARD_AUDIT_LOG_DIR or the local whiteboard log path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for distilled fixture JSON files.",
    )
    parser.add_argument(
        "--audit-id",
        action="append",
        default=[],
        help="Audit id to distill. Defaults to the curated seed fixture set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print fixture JSON instead of writing files.",
    )
    return parser.parse_args(argv)


def selected_fixture_specs(audit_ids: Sequence[str]) -> list[dict[str, str]]:
    if not audit_ids:
        return [dict(item) for item in DEFAULT_FIXTURES]

    defaults_by_id = {item["audit_id"]: item for item in DEFAULT_FIXTURES}
    selected: list[dict[str, str]] = []
    for audit_id in audit_ids:
        if audit_id in defaults_by_id:
            selected.append(dict(defaults_by_id[audit_id]))
        else:
            selected.append({
                "audit_id": audit_id,
                "slug": slugify(audit_id.replace("audit_", "")),
                "description": "Distilled local audit trace.",
            })
    return selected


def find_audit_dir(root: Path, audit_id: str) -> Path:
    candidates = sorted(root.glob(f"*/{audit_id}"))
    if not candidates:
        raise FileNotFoundError(f"Could not find audit id {audit_id!r} under {root}")
    return candidates[-1]


def build_fixture(audit_dir: Path, spec: dict[str, str]) -> dict[str, Any]:
    input_payload = read_json(audit_dir / "input.json")
    fast_result = read_json(audit_dir / "fast_result.json", default=input_payload.get("fastResult") or {})
    comparison = read_json(audit_dir / "comparison.json", default={})
    vlm = read_json(audit_dir / "vlm_normalized.json", default={})

    source_strokes = [stroke for stroke in input_payload.get("strokes") or [] if isinstance(stroke, dict)]
    answer_box = bbox_or_none(input_payload.get("answerBox")) or bbox_for_strokes(source_strokes)
    eligible_strokes = [
        stroke for stroke in source_strokes
        if stroke_belongs_to_answer_box(stroke, answer_box)
    ]
    if not eligible_strokes:
        eligible_strokes = source_strokes

    origin = {
        "x": float(answer_box["xMin"]) if answer_box else min_box_value(eligible_strokes, "xMin"),
        "y": float(answer_box["yMin"]) if answer_box else min_box_value(eligible_strokes, "yMin"),
    }
    relative_answer_box = shift_box(answer_box, origin) if answer_box else bbox_for_strokes(eligible_strokes, origin)
    relative_problem_box = shift_box(bbox_or_none(input_payload.get("problemBox")), origin)
    time_base = min(
        (float(stroke.get("startTime")) for stroke in eligible_strokes if is_finite(stroke.get("startTime"))),
        default=0.0,
    )

    id_map: dict[str, str] = {}
    distilled_strokes: list[dict[str, Any]] = []
    for index, stroke in enumerate(eligible_strokes, start=1):
        distilled_id = f"s{index:03d}"
        id_map[str(stroke.get("id") or index)] = distilled_id
        distilled_strokes.append(distill_stroke(stroke, distilled_id, origin, time_base))

    board_size = board_size_for(relative_answer_box, distilled_strokes)
    for stroke in distilled_strokes:
        stroke["bbox"] = normalize_box(stroke["canvasBbox"], board_size)
        stroke["points"] = normalized_points_for(stroke, board_size)
    detections = distill_detections(
        fast_result.get("detection", {}).get("detections") or [],
        origin,
        answer_box,
        board_size,
    )

    assign_relations(distilled_strokes)
    line_groups = reviewed_line_groups(spec["audit_id"]) or build_line_groups(fast_result, id_map)
    visual_only_stroke_ids = reviewed_visual_only_stroke_ids(spec["audit_id"])
    annotate_strokes(distilled_strokes, line_groups)
    annotate_visual_only_strokes(distilled_strokes, visual_only_stroke_ids)

    return {
        "schemaVersion": 1,
        "fixtureKind": "real-handwriting-trace",
        "slug": spec["slug"],
        "description": spec.get("description") or "",
        "sourceAuditId": spec["audit_id"],
        "problemLatex": str(input_payload.get("problemLatex") or ""),
        "problemMetadata": sanitize_json(input_payload.get("problemMetadata") or {}),
        "triggerReasons": [str(item) for item in input_payload.get("triggerReasons") or []],
        "knownDiscrepancyTypes": discrepancy_types(comparison),
        "fastLatexLines": [
            str(item or "").strip()
            for item in (fast_result.get("latexLines") or input_payload.get("fastResult", {}).get("latexLines") or [])
            if str(item or "").strip()
        ],
        "expectedLatexLines": [
            str(item or "").strip()
            for item in (vlm.get("latexLines") or fast_result.get("latexLines") or [])
            if str(item or "").strip()
        ],
        "visualMarks": visual_marks(comparison, vlm),
        "answerBox": relative_answer_box,
        "problemBox": relative_problem_box,
        "boardSize": board_size,
        "detections": detections,
        "strokes": distilled_strokes,
        "expectedLineGroups": line_groups,
        "visualOnlyStrokeIds": visual_only_stroke_ids,
        "sourceStats": {
            "originalStrokeCount": len(source_strokes),
            "eligibleStrokeCount": len(eligible_strokes),
        },
    }


def distill_detections(
    detections: Sequence[Any],
    origin: dict[str, float],
    answer_box: Optional[dict[str, float]],
    board_size: dict[str, float],
) -> list[dict[str, Any]]:
    distilled: list[dict[str, Any]] = []
    for item in detections:
        if not isinstance(item, dict):
            continue
        source_box = bbox_or_none(item.get("bbox"))
        if not source_box:
            continue
        if answer_box and not boxes_overlap(source_box, answer_box):
            continue
        shifted = shift_box(source_box, origin)
        if not shifted:
            continue
        detection: dict[str, Any] = {
            "bbox": shifted,
            "normalizedBbox": normalize_box(shifted, board_size),
        }
        score = item.get("score")
        if is_finite(score):
            detection["score"] = round_float(float(score), 6)
        polygon = []
        for point in item.get("polygon") or []:
            if (
                isinstance(point, (list, tuple)) and
                len(point) >= 2 and
                is_finite(point[0]) and
                is_finite(point[1])
            ):
                polygon.append([
                    round_float(float(point[0]) - origin["x"], 2),
                    round_float(float(point[1]) - origin["y"], 2),
                ])
        if polygon:
            detection["polygon"] = polygon
        distilled.append(detection)
    return distilled


def distill_stroke(stroke: dict[str, Any], distilled_id: str, origin: dict[str, float], time_base: float) -> dict[str, Any]:
    start = relative_time(stroke.get("startTime"), time_base)
    end = relative_time(stroke.get("endTime"), time_base)
    if end < start:
        end = start
    raw_points = compact_points(stroke.get("rawPoints") or [], origin)
    outline_points = compact_points(stroke.get("outlinePoints") or [], origin)
    canvas_bbox = shift_box(bbox_or_none(stroke.get("canvasBbox")) or bbox_for_points(raw_points), origin)

    return {
        "id": distilled_id,
        "startTime": start,
        "endTime": end,
        "rawPoints": raw_points,
        "outlinePoints": outline_points,
        "color": str(stroke.get("color") or "#000000"),
        "canvasBbox": canvas_bbox,
    }


def compact_points(points: Sequence[Any], origin: dict[str, float]) -> list[dict[str, float]]:
    compacted: list[dict[str, float]] = []
    for point in downsample([point for point in points if isinstance(point, dict)], 96):
        if not is_finite(point.get("x")) or not is_finite(point.get("y")):
            continue
        pressure = point.get("pressure")
        compacted.append({
            "x": round_float(float(point.get("x", 0)) - origin["x"], 2),
            "y": round_float(float(point.get("y", 0)) - origin["y"], 2),
            "pressure": round_float(float(pressure), 3) if is_finite(pressure) else 0.5,
        })
    return compacted


def normalized_points_for(stroke: dict[str, Any], board_size: dict[str, float]) -> list[dict[str, float]]:
    raw = stroke.get("rawPoints") or []
    if not raw:
        return []
    duration = max(1.0, float(stroke.get("endTime", 0)) - float(stroke.get("startTime", 0)))
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


def build_line_groups(fast_result: dict[str, Any], id_map: dict[str, str]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for line in fast_result.get("lines") or []:
        stroke_ids = [
            id_map[str(stroke_id)]
            for stroke_id in line.get("strokeIds") or []
            if str(stroke_id) in id_map
        ]
        if not stroke_ids:
            continue
        groups.append({
            "lineIndex": len(groups),
            "latex": str(line.get("acceptedLatex") or line.get("latex") or "").strip(),
            "strokeIds": stroke_ids,
            "source": "fastResult.lines",
            "sourceCandidateId": f"fast-line-{len(groups) + 1}",
        })
    return groups


def reviewed_line_groups(audit_id: str) -> list[dict[str, Any]]:
    return [dict(item) for item in REVIEWED_LINE_GROUP_OVERRIDES.get(audit_id, [])]


def reviewed_visual_only_stroke_ids(audit_id: str) -> list[str]:
    return [str(item) for item in REVIEWED_VISUAL_ONLY_STROKES.get(audit_id, [])]


def annotate_strokes(strokes: list[dict[str, Any]], line_groups: Sequence[dict[str, Any]]) -> None:
    labels: dict[str, tuple[int, str]] = {}
    for group in line_groups:
        for stroke_id in group.get("strokeIds") or []:
            labels[str(stroke_id)] = (int(group["lineIndex"]), str(group.get("latex") or ""))

    for stroke in strokes:
        label = labels.get(stroke["id"])
        if label:
            stroke["expectedLineIndex"] = label[0]
            stroke["expectedLatex"] = label[1]


def annotate_visual_only_strokes(strokes: list[dict[str, Any]], visual_only_stroke_ids: Sequence[str]) -> None:
    visual_ids = {str(stroke_id) for stroke_id in visual_only_stroke_ids}
    for stroke in strokes:
        if stroke["id"] in visual_ids:
            stroke["visualOnly"] = True


def assign_relations(strokes: list[dict[str, Any]]) -> None:
    previous: dict[str, Any] | None = None
    for stroke in strokes:
        if previous is None:
            stroke["relationsToPrev"] = {"dx": 0, "dy": 0, "dt": 0, "overlapRatio": 0}
        else:
            stroke["relationsToPrev"] = relation_to_previous(stroke, previous)
        previous = stroke


def relation_to_previous(stroke: dict[str, Any], previous: dict[str, Any]) -> dict[str, float]:
    curr = stroke["bbox"]
    prev = previous["bbox"]
    return {
        "dx": round_float(center_x(curr) - center_x(prev), 4),
        "dy": round_float(center_y(curr) - center_y(prev), 4),
        "dt": round_float(float(stroke["startTime"]) - float(previous["startTime"]), 1),
        "overlapRatio": round_float(iou(curr, prev), 4),
    }


def visual_marks(comparison: dict[str, Any], vlm: dict[str, Any]) -> list[dict[str, Any]]:
    marks: list[dict[str, Any]] = []
    for item in vlm.get("visualMarks") or []:
        if isinstance(item, dict):
            marks.append(sanitize_json(item))

    for container in ("discrepancies", "observations"):
        for item in comparison.get(container) or []:
            if not isinstance(item, dict) or item.get("type") != "visual_intent_observed":
                continue
            for mark in item.get("visualMarks") or []:
                if isinstance(mark, dict):
                    normalized = sanitize_json(mark)
                    if normalized not in marks:
                        marks.append(normalized)
    return marks


def discrepancy_types(comparison: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for container in ("discrepancies", "observations"):
        for item in comparison.get(container) or []:
            if isinstance(item, dict) and item.get("type"):
                out.append(str(item["type"]))
    return sorted(set(out))


def stroke_belongs_to_answer_box(stroke: dict[str, Any], answer_box: Optional[dict[str, float]]) -> bool:
    box = bbox_or_none(stroke.get("canvasBbox"))
    if not box:
        return False
    if not answer_box:
        return True
    if not boxes_overlap(box, answer_box):
        return False
    if (
        answer_box["xMin"] <= center_x(box) <= answer_box["xMax"]
        and answer_box["yMin"] <= center_y(box) <= answer_box["yMax"]
    ):
        return True
    overlap_width = max(0.0, min(box["xMax"], answer_box["xMax"]) - max(box["xMin"], answer_box["xMin"]))
    overlap_height = max(0.0, min(box["yMax"], answer_box["yMax"]) - max(box["yMin"], answer_box["yMin"]))
    overlap_area = overlap_width * overlap_height
    stroke_area = max(1.0, (box["xMax"] - box["xMin"]) * (box["yMax"] - box["yMin"]))
    return overlap_area / stroke_area >= 0.35


def board_size_for(answer_box: Optional[dict[str, float]], strokes: Sequence[dict[str, Any]]) -> dict[str, float]:
    boxes = [stroke.get("canvasBbox") for stroke in strokes if stroke.get("canvasBbox")]
    if answer_box:
        boxes.append(answer_box)
    x_max = max((float(box["xMax"]) for box in boxes if box), default=1.0)
    y_max = max((float(box["yMax"]) for box in boxes if box), default=1.0)
    return {
        "width": round_float(max(1.0, x_max + 24.0), 2),
        "height": round_float(max(1.0, y_max + 24.0), 2),
    }


def normalize_box(box: dict[str, float], board_size: dict[str, float]) -> dict[str, float]:
    return {
        "xMin": round_float(box["xMin"] / board_size["width"], 4),
        "yMin": round_float(box["yMin"] / board_size["height"], 4),
        "xMax": round_float(box["xMax"] / board_size["width"], 4),
        "yMax": round_float(box["yMax"] / board_size["height"], 4),
    }


def shift_box(box: Optional[dict[str, float]], origin: dict[str, float]) -> Optional[dict[str, float]]:
    if not box:
        return None
    return {
        "xMin": round_float(float(box["xMin"]) - origin["x"], 2),
        "yMin": round_float(float(box["yMin"]) - origin["y"], 2),
        "xMax": round_float(float(box["xMax"]) - origin["x"], 2),
        "yMax": round_float(float(box["yMax"]) - origin["y"], 2),
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


def bbox_for_strokes(strokes: Sequence[dict[str, Any]], origin: Optional[dict[str, float]] = None) -> Optional[dict[str, float]]:
    boxes = [bbox_or_none(stroke.get("canvasBbox")) for stroke in strokes]
    boxes = [box for box in boxes if box]
    if not boxes:
        return None
    box = {
        "xMin": min(item["xMin"] for item in boxes),
        "yMin": min(item["yMin"] for item in boxes),
        "xMax": max(item["xMax"] for item in boxes),
        "yMax": max(item["yMax"] for item in boxes),
    }
    return shift_box(box, origin) if origin else box


def bbox_for_points(points: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not points:
        return {"xMin": 0, "yMin": 0, "xMax": 1, "yMax": 1}
    return {
        "xMin": min(float(point["x"]) for point in points),
        "yMin": min(float(point["y"]) for point in points),
        "xMax": max(float(point["x"]) for point in points),
        "yMax": max(float(point["y"]) for point in points),
    }


def boxes_overlap(left: dict[str, float], right: dict[str, float]) -> bool:
    return not (
        left["xMax"] < right["xMin"]
        or left["xMin"] > right["xMax"]
        or left["yMax"] < right["yMin"]
        or left["yMin"] > right["yMax"]
    )


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


def min_box_value(strokes: Sequence[dict[str, Any]], key: str) -> float:
    values = [
        float(box[key])
        for stroke in strokes
        for box in [bbox_or_none(stroke.get("canvasBbox"))]
        if box
    ]
    return min(values) if values else 0.0


def relative_time(value: Any, base: float) -> float:
    if not is_finite(value):
        return 0.0
    return round_float(float(value) - base, 1)


def downsample(values: Sequence[Any], max_count: int) -> list[Any]:
    if len(values) <= max_count:
        return list(values)
    if max_count <= 2:
        return list(values[:max_count])
    last = len(values) - 1
    return [values[round((index / (max_count - 1)) * last)] for index in range(max_count)]


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json(child) for key, child in value.items()}
    if isinstance(value, list):
        return [sanitize_json(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def round_float(value: float, digits: int) -> float:
    return round(float(value), digits)


def slugify(value: str) -> str:
    out = []
    previous_dash = False
    for char in value.lower():
        if char.isalnum():
            out.append(char)
            previous_dash = False
        elif not previous_dash:
            out.append("-")
            previous_dash = True
    return "".join(out).strip("-") or "real-handwriting-fixture"


if __name__ == "__main__":
    raise SystemExit(main())
