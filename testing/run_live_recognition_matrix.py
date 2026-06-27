#!/usr/bin/env python3
"""Live DBNet + CoMER evaluation for the whiteboard_3 recognition pipeline.

This runner intentionally keeps ground-truth labels in the test harness only.
The pipeline under test receives the rendered ink, DBNet detections, CoMER
candidate lists, and the problem equation context; expected labels are used
only to report whether segmentation and OCR landed on the intended line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import socket
import subprocess
import sys
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from PIL import Image, ImageDraw

try:
    from .fixture_catalog import (
        PROBLEMS,
        RESULTS_DIR,
        SPACING_VARIANTS,
        MathProblem,
        build_board,
        fixture_payload,
        gap_pattern_names,
        get_problem,
        line_gaps_for_pattern,
        placements_for,
        parse_line_gaps,
        slug,
    )
    from .latex_semantics import (
        ParseFailure,
        contextual_variable_names,
        parse_math,
        score_candidate_group,
        sympy_value_to_latex,
    )
    from .synthetic_handwriting import save_board_png
    from .synthetic_handwriting import available_ink_styles
except ImportError:
    from fixture_catalog import (
        PROBLEMS,
        RESULTS_DIR,
        SPACING_VARIANTS,
        MathProblem,
        build_board,
        fixture_payload,
        gap_pattern_names,
        get_problem,
        line_gaps_for_pattern,
        placements_for,
        parse_line_gaps,
        slug,
    )
    from latex_semantics import ParseFailure, contextual_variable_names, parse_math, score_candidate_group, sympy_value_to_latex
    from synthetic_handwriting import save_board_png
    from synthetic_handwriting import available_ink_styles


ROOT = Path(__file__).resolve().parents[1]
SEGMENTER = ROOT / "testing" / "segment_fixture_with_js.mjs"
LIVE_RESULTS = RESULTS_DIR / "live_recognition"
ORDERS = ("line-order", "reverse-lines", "interleaved-lines")
DEFAULT_PROBLEMS = (
    "algebra_steps",
    "symbol_context_eta",
    "rational_two_fraction_solve",
    "derivative_quotient_evaluate",
)
DEFAULT_SPACINGS = ("standard", "tight-steps")
DEFAULT_INITIAL_RASTER_HEIGHT = 104
DEFAULT_INITIAL_RASTER_MIN_HEIGHT = 1
RETRY_RASTER_HEIGHTS = (88, 104, 72)
SEMANTIC_RETRY_RASTER_HEIGHTS = (48, 64, 72, 88, 104)
MAX_COMER_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class RenderedFixture:
    problem: MathProblem
    spacing: str
    ink_style: str
    board: Any
    payload: dict[str, Any]
    png_path: Path
    json_path: Path
    gap_pattern: str = ""
    line_gaps: Sequence[float] = ()


def post_image(url: str, image_path: Path, timeout_seconds: float) -> dict[str, Any]:
    boundary = "----whiteboard3-" + uuid.uuid4().hex
    image_bytes = image_path.read_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\n".encode("ascii"),
            b'Content-Disposition: form-data; name="file"; filename="image.png"\r\n',
            b"Content-Type: image/png\r\n\r\n",
            image_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("ascii"),
        ]
    )
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds + 5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            payload["_httpStatus"] = response.status
            return payload
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"detail": body}
        payload["_httpStatus"] = exc.code
        return payload
    except (TimeoutError, socket.timeout) as exc:
        return {
            "_httpStatus": 408,
            "timedOut": True,
            "elapsedSeconds": round(time.monotonic() - started, 3),
            "selectionPenalty": -1000,
            "candidates": [],
            "top": None,
            "detail": str(exc) or "client-side request timeout",
        }
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            return {
                "_httpStatus": 408,
                "timedOut": True,
                "elapsedSeconds": round(time.monotonic() - started, 3),
                "selectionPenalty": -1000,
                "candidates": [],
                "top": None,
                "detail": str(exc.reason) or "client-side request timeout",
            }
        raise


def recognize_crop_with_retries(
    recognize_url: str,
    crop_path: Path,
    timeout_seconds: float,
    *,
    retry_on_failure: bool,
    initial_raster_height: int = DEFAULT_INITIAL_RASTER_HEIGHT,
    initial_raster_min_height: int = DEFAULT_INITIAL_RASTER_MIN_HEIGHT,
    fixture: Optional[RenderedFixture] = None,
    candidate: Optional[dict[str, Any]] = None,
    extended_timeout_seconds: float = 0,
) -> dict[str, Any]:
    initial_crop, initial_target_height = prepare_initial_recognition_crop(
        crop_path,
        initial_raster_height,
        initial_raster_min_height,
    )
    initial = post_image(recognize_url, initial_crop, timeout_seconds)
    if initial_target_height:
        initial["_initialTargetPixelHeight"] = initial_target_height
        initial["_initialCrop"] = str(initial_crop)
    if not retry_on_failure or not payload_needs_retry(initial):
        return initial

    stroke_chunk_fallback_attempted = False
    early_stroke_chunked = recognize_stroke_chunked_candidate(
        recognize_url,
        fixture,
        candidate,
        timeout_seconds,
        target_height=initial_raster_height if initial_raster_height > 0 else DEFAULT_INITIAL_RASTER_HEIGHT,
    )
    if early_stroke_chunked is not None:
        stroke_chunk_fallback_attempted = True
    if early_stroke_chunked and not payload_needs_retry(early_stroke_chunked):
        early_stroke_chunked["chunkFallbackBeforeHeightRetries"] = True
        return early_stroke_chunked

    early_chunked = recognize_chunked_crop(
        recognize_url,
        crop_path,
        timeout_seconds,
        target_height=initial_raster_height if initial_raster_height > 0 else DEFAULT_INITIAL_RASTER_HEIGHT,
        only_if_wide=True,
    )
    if early_chunked and not payload_needs_retry(early_chunked):
        early_chunked["chunkFallbackBeforeHeightRetries"] = True
        return early_chunked

    attempts = [initial]
    for height in RETRY_RASTER_HEIGHTS:
        if initial_target_height and int(height) == int(initial_target_height):
            continue
        variant_path = normalized_crop_variant(crop_path, height)
        payload = post_image(recognize_url, variant_path, timeout_seconds)
        payload["_retryTargetPixelHeight"] = height
        payload["_retryCrop"] = str(variant_path)
        attempts.append(payload)

    merged = merge_recognition_attempts(attempts)
    if payload_needs_retry(merged):
        extended = recognize_with_extended_timeout(
            recognize_url,
            crop_path,
            timeout_seconds,
            extended_timeout_seconds,
            initial_raster_height=initial_raster_height,
            candidate=candidate,
        )
        if extended and not payload_needs_retry(extended):
            attempts.append(extended)
            return merge_recognition_attempts(attempts)
        if extended:
            attempts.append(extended)
            merged = merge_recognition_attempts(attempts)
    if payload_needs_retry(merged):
        stroke_chunked = None
        if not stroke_chunk_fallback_attempted:
            stroke_chunked = recognize_stroke_chunked_candidate(
                recognize_url,
                fixture,
                candidate,
                timeout_seconds,
                target_height=initial_raster_height if initial_raster_height > 0 else DEFAULT_INITIAL_RASTER_HEIGHT,
            )
        if stroke_chunked and not payload_needs_retry(stroke_chunked):
            return stroke_chunked
        chunked = recognize_chunked_crop(
            recognize_url,
            crop_path,
            timeout_seconds,
            target_height=initial_raster_height if initial_raster_height > 0 else DEFAULT_INITIAL_RASTER_HEIGHT,
        )
        if chunked and not payload_needs_retry(chunked):
            return chunked
    return merged


def recognize_crop_with_semantic_retries(
    recognize_url: str,
    crop_path: Path,
    timeout_seconds: float,
    initial_payload: dict[str, Any],
    *,
    initial_raster_height: int = DEFAULT_INITIAL_RASTER_HEIGHT,
    retry_heights: Sequence[int] = SEMANTIC_RETRY_RASTER_HEIGHTS,
    fixture: Optional[RenderedFixture] = None,
    candidate: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    skip_heights = {
        int(height)
        for height in (
            initial_payload.get("_initialTargetPixelHeight"),
            initial_payload.get("initialTargetPixelHeight"),
        )
        if isinstance(height, (int, float)) and height > 0
    }
    for attempt in initial_payload.get("retryAttempts") or []:
        height = attempt.get("targetPixelHeight")
        if isinstance(height, (int, float)) and height > 0:
            skip_heights.add(int(height))

    attempts = [initial_payload]
    for height in retry_heights:
        if int(height) in skip_heights:
            continue
        variant_path = normalized_crop_variant(crop_path, int(height))
        payload = post_image(recognize_url, variant_path, timeout_seconds)
        payload["_retryTargetPixelHeight"] = int(height)
        payload["_retryCrop"] = str(variant_path)
        attempts.append(payload)

    if len(attempts) == 1:
        return initial_payload
    merged = merge_recognition_attempts(attempts)
    merged["semanticRetryUsed"] = True
    if payload_needs_retry(merged):
        stroke_chunked = recognize_stroke_chunked_candidate(
            recognize_url,
            fixture,
            candidate,
            timeout_seconds,
            target_height=initial_raster_height if initial_raster_height > 0 else DEFAULT_INITIAL_RASTER_HEIGHT,
        )
        if stroke_chunked and not payload_needs_retry(stroke_chunked):
            stroke_chunked["semanticRetryUsed"] = True
            stroke_chunked["chunkFallbackAfterSemanticRetries"] = True
            return stroke_chunked
        chunked = recognize_chunked_crop(
            recognize_url,
            crop_path,
            timeout_seconds,
            target_height=initial_raster_height if initial_raster_height > 0 else DEFAULT_INITIAL_RASTER_HEIGHT,
        )
        if chunked and not payload_needs_retry(chunked):
            chunked["semanticRetryUsed"] = True
            chunked["chunkFallbackAfterSemanticRetries"] = True
            return chunked
    return merged


def recognize_with_extended_timeout(
    recognize_url: str,
    crop_path: Path,
    base_timeout_seconds: float,
    extended_timeout_seconds: float,
    *,
    initial_raster_height: int,
    candidate: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not candidate_needs_extended_timeout(candidate):
        return None
    extended_timeout = max(float(base_timeout_seconds), float(extended_timeout_seconds or 0))
    if extended_timeout <= float(base_timeout_seconds) + 0.01:
        return None
    target_height = int(initial_raster_height) if int(initial_raster_height or 0) > 0 else DEFAULT_INITIAL_RASTER_HEIGHT
    variant_path = normalized_crop_variant(crop_path, target_height)
    payload = post_image(url_with_timeout(recognize_url, extended_timeout), variant_path, extended_timeout)
    payload["_retryTargetPixelHeight"] = target_height
    payload["_retryCrop"] = str(variant_path)
    payload["_extendedTimeoutSeconds"] = extended_timeout
    return payload


def url_with_timeout(url: str, timeout_seconds: float) -> str:
    timeout = f"timeout_seconds={float(timeout_seconds):g}"
    if "timeout_seconds=" in url:
        return re.sub(r"timeout_seconds=[^&]+", timeout, url)
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{timeout}"


def prepare_initial_recognition_crop(
    crop_path: Path,
    target_height: int,
    min_height: int,
) -> tuple[Path, Optional[int]]:
    if target_height <= 0 or min_height <= 0:
        return crop_path, None
    with Image.open(crop_path) as image:
        image_height = image.height
    if image_height < min_height or image_height == target_height:
        return crop_path, None
    return normalized_crop_variant(crop_path, target_height), target_height


def normalized_crop_variant(crop_path: Path, target_height: int) -> Path:
    image = Image.open(crop_path).convert("RGB")
    width = max(1, round(image.width * (target_height / image.height)))
    resized = image.resize((width, target_height), Image.Resampling.LANCZOS)
    variant_path = crop_path.with_name(f"{crop_path.stem}_h{target_height}{crop_path.suffix}")
    resized.save(variant_path)
    return variant_path


def recognize_chunked_crop(
    recognize_url: str,
    crop_path: Path,
    timeout_seconds: float,
    *,
    target_height: int,
    only_if_wide: bool = False,
) -> Optional[dict[str, Any]]:
    if only_if_wide and not crop_is_wide_for_chunking(crop_path):
        return None
    chunks = split_crop_into_horizontal_chunks(crop_path)
    if len(chunks) < 2:
        return None

    with Image.open(crop_path).convert("RGB") as source:
        parts: list[str] = []
        attempts: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            if chunk.get("literalLatex"):
                parts.append(str(chunk["literalLatex"]))
                attempts.append({
                    "literalLatex": chunk["literalLatex"],
                    "xRange": chunk["xRange"],
                })
                continue

            x0, x1 = chunk["xRange"]
            chunk_path = crop_path.with_name(f"{crop_path.stem}_chunk_{index + 1}.png")
            source.crop((max(0, x0 - 8), 0, min(source.width, x1 + 8), source.height)).save(chunk_path)
            variant_path = normalized_crop_variant(chunk_path, target_height)
            payload = post_image(recognize_url, variant_path, timeout_seconds)
            latex = choose_chunk_latex(payload)
            attempts.append({
                "xRange": chunk["xRange"],
                "crop": str(variant_path),
                "httpStatus": payload.get("_httpStatus", 200),
                "timedOut": bool(payload.get("timedOut")),
                "elapsedSeconds": payload.get("elapsedSeconds"),
                "topLatex": latex,
            })
            if not latex or payload.get("timedOut") or payload.get("_httpStatus", 200) >= 400:
                return {
                    "_httpStatus": payload.get("_httpStatus", 500),
                    "timedOut": bool(payload.get("timedOut")),
                    "failed": True,
                    "chunkFallback": True,
                    "chunkAttempts": attempts,
                    "candidates": [],
                    "top": None,
                }
            parts.append(latex)

    latex = normalize_chunked_latex(" ".join(parts))
    if not latex:
        return None
    return {
        "_httpStatus": 200,
        "timedOut": False,
        "failed": False,
        "chunkFallback": True,
        "chunkAttempts": attempts,
        "candidates": [{"latex": latex, "score": 0, "source": "chunk-fallback"}],
        "top": {"latex": latex, "score": 0, "source": "chunk-fallback"},
        "elapsedSeconds": round(sum(float(item.get("elapsedSeconds") or 0) for item in attempts), 3),
    }


def recognize_stroke_chunked_candidate(
    recognize_url: str,
    fixture: Optional[RenderedFixture],
    candidate: Optional[dict[str, Any]],
    timeout_seconds: float,
    *,
    target_height: int,
    min_width: float = 460,
) -> Optional[dict[str, Any]]:
    if fixture is None or candidate is None:
        return None
    bbox = candidate.get("bbox") or {}
    structural = candidate_needs_extended_timeout(candidate)
    compact_fraction_structure = candidate_has_local_fraction_structure(candidate)
    effective_min_width = 220 if structural else min_width
    if bbox_width(bbox) < effective_min_width:
        return None
    chunks = split_candidate_into_stroke_chunks(
        candidate,
        min_gap=8 if structural else (12 if compact_fraction_structure else 18),
        max_chunk_width=140 if compact_fraction_structure else 340,
    )
    if len(chunks) < 2:
        return None

    parts: list[str] = []
    attempts: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        if chunk.get("literalLatex"):
            parts.append(str(chunk["literalLatex"]))
            attempts.append({
                "literalLatex": chunk["literalLatex"],
                "strokeIds": chunk.get("strokeIds") or [],
            })
            continue

        inferred_literal = infer_contextual_chunk_literal(chunk, chunks, index, fixture)
        if inferred_literal:
            parts.append(inferred_literal)
            attempts.append({
                "literalLatex": inferred_literal,
                "inferredLiteral": True,
                "strokeIds": chunk.get("strokeIds") or [],
            })
            continue

        chunk_path = LIVE_RESULTS / "stroke_chunks" / (
            f"{safe_filename_slug(str(candidate.get('candidateId') or 'candidate'))}_chunk_{index + 1}.png"
        )
        crop_stroke_chunk(chunk, chunk_path)
        variant_path = normalized_crop_variant(chunk_path, target_height)
        payload = post_image(recognize_url, variant_path, timeout_seconds)
        latex = choose_chunk_latex(payload)
        attempts.append({
            "strokeIds": chunk.get("strokeIds") or [],
            "crop": str(variant_path),
            "httpStatus": payload.get("_httpStatus", 200),
            "timedOut": bool(payload.get("timedOut")),
            "elapsedSeconds": payload.get("elapsedSeconds"),
            "topLatex": latex,
        })
        if not latex or payload.get("timedOut") or payload.get("_httpStatus", 200) >= 400:
            fraction_latex, fraction_attempt = recognize_fraction_stroke_chunk(
                recognize_url,
                chunk,
                timeout_seconds,
                target_height=target_height,
            )
            if fraction_latex:
                attempts.append(fraction_attempt)
                parts.append(fraction_latex)
                continue
            if fraction_attempt:
                attempts.append(fraction_attempt)
            return {
                "_httpStatus": payload.get("_httpStatus", 500),
                "timedOut": bool(payload.get("timedOut")),
                "failed": True,
                "chunkFallback": True,
                "chunkFallbackSource": "stroke",
                "chunkAttempts": attempts,
                "candidates": [],
                "top": None,
            }
        parts.append(latex)

    latex = normalize_chunked_latex(" ".join(parts))
    if not latex:
        return None
    return {
        "_httpStatus": 200,
        "timedOut": False,
        "failed": False,
        "chunkFallback": True,
        "chunkFallbackSource": "stroke",
        "chunkAttempts": attempts,
        "candidates": [{"latex": latex, "score": 0, "source": "stroke-chunk-fallback"}],
        "top": {"latex": latex, "score": 0, "source": "stroke-chunk-fallback"},
        "elapsedSeconds": round(sum(float(item.get("elapsedSeconds") or 0) for item in attempts), 3),
    }


def split_candidate_into_stroke_chunks(
    candidate: dict[str, Any],
    *,
    min_gap: float = 18,
    max_chunk_width: float = 340,
) -> list[dict[str, Any]]:
    strokes = sorted(
        [stroke for stroke in candidate.get("strokes") or [] if stroke.get("canvasBbox")],
        key=lambda stroke: (
            float(stroke["canvasBbox"].get("xMin", 0)),
            float(stroke["canvasBbox"].get("yMin", 0)),
        ),
    )
    if len(strokes) < 2:
        return []

    atoms = make_fraction_aware_chunk_atoms(strokes, min_gap=min_gap)

    chunks: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush_pending() -> None:
        nonlocal pending
        if pending:
            chunks.append(make_stroke_chunk([stroke for atom in pending for stroke in atom["strokes"]]))
        pending = []

    for atom in atoms:
        if atom.get("atomicChunk"):
            flush_pending()
            chunks.append(make_stroke_chunk(atom["strokes"]))
            continue

        if is_equals_stroke_atom(atom):
            flush_pending()
            chunk = make_stroke_chunk(atom["strokes"])
            chunk["literalLatex"] = "="
            chunks.append(chunk)
            continue

        proposed = [*pending, atom]
        proposed_box = bbox_for_strokes([stroke for item in proposed for stroke in item["strokes"]])
        if pending and bbox_width(proposed_box) > max_chunk_width:
            flush_pending()
        pending.append(atom)
    flush_pending()
    return [chunk for chunk in chunks if chunk.get("strokeIds")]


def make_fraction_aware_chunk_atoms(
    strokes: Sequence[dict[str, Any]],
    *,
    min_gap: float,
) -> list[dict[str, Any]]:
    fraction_groups = find_local_fraction_groups(strokes)
    assigned: set[int] = {
        id(stroke)
        for group in fraction_groups
        for stroke in group
    }
    entries: list[dict[str, Any]] = [
        {**make_stroke_chunk_atom(group), "atomicChunk": True}
        for group in fraction_groups
    ]

    remaining = [stroke for stroke in strokes if id(stroke) not in assigned]
    if remaining:
        current = [remaining[0]]
        current_box = dict(remaining[0]["canvasBbox"])
        for stroke in remaining[1:]:
            box = stroke["canvasBbox"]
            gap = float(box.get("xMin", 0)) - float(current_box.get("xMax", 0))
            if gap >= min_gap:
                entries.append(make_stroke_chunk_atom(current))
                current = [stroke]
                current_box = dict(box)
            else:
                current.append(stroke)
                current_box = bbox_union(current_box, box)
        entries.append(make_stroke_chunk_atom(current))

    return sorted(
        entries,
        key=lambda atom: (
            float((atom.get("bbox") or {}).get("xMin", 0)),
            float((atom.get("bbox") or {}).get("yMin", 0)),
        ),
    )


def find_local_fraction_groups(strokes: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    available = [stroke for stroke in strokes if stroke.get("canvasBbox")]
    groups: list[list[dict[str, Any]]] = []
    used: set[int] = set()
    bars = sorted(
        [stroke for stroke in available if stroke_is_horizontal_bar_like(stroke.get("canvasBbox") or {})],
        key=lambda stroke: bbox_width(stroke.get("canvasBbox") or {}),
        reverse=True,
    )
    for bar in bars:
        if id(bar) in used:
            continue
        bar_box = bar.get("canvasBbox") or {}
        numerator = nearest_fraction_side_strokes(available, bar, side="above", used=used)
        denominator = nearest_fraction_side_strokes(available, bar, side="below", used=used)
        if not numerator or not denominator:
            continue
        group = [*numerator, bar, *denominator]
        for stroke in group:
            used.add(id(stroke))
        groups.append(sorted(
            group,
            key=lambda stroke: (
                float((stroke.get("canvasBbox") or {}).get("xMin", 0)),
                float((stroke.get("canvasBbox") or {}).get("yMin", 0)),
            ),
        ))
    return groups


def nearest_fraction_side_strokes(
    strokes: Sequence[dict[str, Any]],
    bar: dict[str, Any],
    *,
    side: str,
    used: set[int],
) -> list[dict[str, Any]]:
    bar_box = bar.get("canvasBbox") or {}
    bar_center = bbox_x_center(bar_box)
    matches: list[tuple[float, dict[str, Any]]] = []
    for stroke in strokes:
        if stroke is bar or id(stroke) in used:
            continue
        box = stroke.get("canvasBbox") or {}
        if side == "above":
            vertical_gap = float(bar_box.get("yMin", 0)) - float(box.get("yMax", 0))
        else:
            vertical_gap = float(box.get("yMin", 0)) - float(bar_box.get("yMax", 0))
        if vertical_gap < -2 or vertical_gap > 44:
            continue
        overlap = horizontal_overlap_ratio(box, bar_box)
        center_distance = abs(bbox_x_center(box) - bar_center)
        if overlap < 0.25 and center_distance > max(18.0, bbox_width(bar_box) * 0.75):
            continue
        matches.append((center_distance + max(0.0, vertical_gap) * 0.15, stroke))
    if not matches:
        return []
    matches.sort(key=lambda item: item[0])
    best_score = matches[0][0]
    cluster_tolerance = max(16.0, bbox_width(bar_box) * 0.4)
    return [stroke for score, stroke in matches if score <= best_score + cluster_tolerance]


def make_stroke_chunk_atom(strokes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "strokes": list(strokes),
        "bbox": bbox_for_strokes(strokes),
    }


def make_stroke_chunk(strokes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "strokes": list(strokes),
        "strokeIds": [str(stroke.get("id")) for stroke in strokes],
        "bbox": bbox_for_strokes(strokes),
    }


def infer_contextual_chunk_literal(
    chunk: dict[str, Any],
    chunks: Sequence[dict[str, Any]],
    index: int,
    fixture: Optional[RenderedFixture],
) -> str:
    if fixture is None or index < 0 or index >= len(chunks) - 1:
        return ""
    if (chunks[index + 1] or {}).get("literalLatex") != "=":
        return ""
    strokes = [stroke for stroke in chunk.get("strokes") or [] if stroke.get("canvasBbox")]
    if not 1 <= len(strokes) <= 2:
        return ""
    box = chunk.get("bbox") or bbox_for_strokes(strokes)
    width = bbox_width(box)
    height = bbox_height(box)
    if width < 8 or height < 10:
        return ""
    if width >= height * 2.2 or height >= width * 2.8:
        return ""
    problem_context = getattr(fixture.problem, "context_latex", "") if fixture.problem else ""
    variables = [
        variable for variable in contextual_variable_names([problem_context])
        if re.fullmatch(r"[a-z]", variable)
    ]
    if len(variables) != 1:
        return ""
    return variables[0]


def crop_stroke_chunk(chunk: dict[str, Any], output_path: Path, padding: int = 24) -> Path:
    bbox = chunk.get("bbox") or {}
    x_min = math.floor(float(bbox["xMin"])) - padding
    y_min = math.floor(float(bbox["yMin"])) - padding
    x_max = math.ceil(float(bbox["xMax"])) + padding
    y_max = math.ceil(float(bbox["yMax"])) + padding
    image = Image.new("RGB", (max(1, x_max - x_min), max(1, y_max - y_min)), "white")
    draw = ImageDraw.Draw(image)
    for stroke in chunk.get("strokes") or []:
        points = [
            (float(point["x"]) - x_min, float(point["y"]) - y_min)
            for point in stroke.get("rawPoints") or []
        ]
        if len(points) >= 3:
            draw.polygon(points, fill="black")
        elif len(points) >= 2:
            draw.line(points, fill="black", width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def recognize_fraction_stroke_chunk(
    recognize_url: str,
    chunk: dict[str, Any],
    timeout_seconds: float,
    *,
    target_height: int,
) -> tuple[str, Optional[dict[str, Any]]]:
    split = split_fraction_stroke_chunk(chunk)
    if not split:
        return "", None

    attempt: dict[str, Any] = {
        "fractionSubchunk": True,
        "strokeIds": chunk.get("strokeIds") or [],
        "barStrokeId": split["bar"].get("id"),
        "parts": [],
    }
    latex_parts: dict[str, str] = {}
    for role in ("numerator", "denominator"):
        part = split[role]
        part_path = LIVE_RESULTS / "stroke_chunks" / (
            f"{safe_filename_slug(str(chunk.get('strokeIds') or ['fraction']))}_{role}.png"
        )
        crop_stroke_chunk(part, part_path)
        variant_path = normalized_crop_variant(part_path, target_height)
        payload = post_image(recognize_url, variant_path, timeout_seconds)
        latex = choose_chunk_latex(payload)
        attempt["parts"].append({
            "role": role,
            "strokeIds": part.get("strokeIds") or [],
            "crop": str(variant_path),
            "httpStatus": payload.get("_httpStatus", 200),
            "timedOut": bool(payload.get("timedOut")),
            "elapsedSeconds": payload.get("elapsedSeconds"),
            "topLatex": latex,
        })
        if not latex or payload.get("timedOut") or payload.get("_httpStatus", 200) >= 400:
            return "", attempt
        latex_parts[role] = latex

    latex = rf"\frac {{ {latex_parts['numerator']} }} {{ {latex_parts['denominator']} }}"
    attempt["topLatex"] = latex
    return latex, attempt


def split_fraction_stroke_chunk(chunk: dict[str, Any]) -> Optional[dict[str, Any]]:
    strokes = [stroke for stroke in chunk.get("strokes") or [] if stroke.get("canvasBbox")]
    if len(strokes) < 3:
        return None
    bbox = chunk.get("bbox") or bbox_for_strokes(strokes)
    bars = [
        stroke for stroke in strokes
        if stroke_looks_like_fraction_bar(stroke.get("canvasBbox") or {}, bbox)
    ]
    if not bars:
        return None
    bar = max(bars, key=lambda stroke: bbox_width(stroke.get("canvasBbox") or {}))
    bar_box = bar.get("canvasBbox") or {}
    bar_mid = (float(bar_box.get("yMin", 0)) + float(bar_box.get("yMax", 0))) / 2
    numerator = [
        stroke for stroke in strokes
        if stroke is not bar and float((stroke.get("canvasBbox") or {}).get("yMax", 0)) <= bar_mid
    ]
    denominator = [
        stroke for stroke in strokes
        if stroke is not bar and float((stroke.get("canvasBbox") or {}).get("yMin", 0)) >= bar_mid
    ]
    if not numerator or not denominator:
        return None
    return {
        "bar": bar,
        "numerator": make_stroke_chunk(numerator),
        "denominator": make_stroke_chunk(denominator),
    }


def stroke_looks_like_fraction_bar(box: dict[str, Any], parent_box: dict[str, Any]) -> bool:
    width = bbox_width(box)
    height = bbox_height(box)
    parent_width = max(1.0, bbox_width(parent_box))
    if width < max(18.0, parent_width * 0.45):
        return False
    if height > 18 or width < height * 3.5:
        return False
    return True


def stroke_is_horizontal_bar_like(box: dict[str, Any]) -> bool:
    width = bbox_width(box)
    height = bbox_height(box)
    return width >= 18 and height <= 12 and width >= height * 3.5


def is_equals_stroke_atom(atom: dict[str, Any]) -> bool:
    strokes = atom.get("strokes") or []
    if len(strokes) != 2:
        return False
    box = atom.get("bbox") or {}
    if bbox_width(box) < bbox_height(box) * 1.35:
        return False
    if not all(bbox_width(stroke.get("canvasBbox") or {}) >= bbox_height(stroke.get("canvasBbox") or {}) * 2.5 for stroke in strokes):
        return False
    overlap = horizontal_overlap_ratio(strokes[0].get("canvasBbox") or {}, strokes[1].get("canvasBbox") or {})
    vertical_gap = max(
        float(strokes[0]["canvasBbox"].get("yMin", 0)),
        float(strokes[1]["canvasBbox"].get("yMin", 0)),
    ) - min(
        float(strokes[0]["canvasBbox"].get("yMax", 0)),
        float(strokes[1]["canvasBbox"].get("yMax", 0)),
    )
    return overlap >= 0.45 and vertical_gap > 0


def bbox_for_strokes(strokes: Sequence[dict[str, Any]]) -> dict[str, float]:
    boxes = [stroke.get("canvasBbox") or {} for stroke in strokes]
    if not boxes:
        return {"xMin": 0.0, "yMin": 0.0, "xMax": 0.0, "yMax": 0.0}
    box = dict(boxes[0])
    for next_box in boxes[1:]:
        box = bbox_union(box, next_box)
    return {key: float(value) for key, value in box.items()}


def bbox_union(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    return {
        "xMin": min(float(a.get("xMin", 0)), float(b.get("xMin", 0))),
        "yMin": min(float(a.get("yMin", 0)), float(b.get("yMin", 0))),
        "xMax": max(float(a.get("xMax", 0)), float(b.get("xMax", 0))),
        "yMax": max(float(a.get("yMax", 0)), float(b.get("yMax", 0))),
    }


def bbox_width(box: dict[str, Any]) -> float:
    return max(0.0, float(box.get("xMax", 0)) - float(box.get("xMin", 0)))


def bbox_height(box: dict[str, Any]) -> float:
    return max(0.0, float(box.get("yMax", 0)) - float(box.get("yMin", 0)))


def bbox_x_center(box: dict[str, Any]) -> float:
    return (float(box.get("xMin", 0)) + float(box.get("xMax", 0))) / 2


def horizontal_overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    overlap = max(
        0.0,
        min(float(a.get("xMax", 0)), float(b.get("xMax", 0))) -
        max(float(a.get("xMin", 0)), float(b.get("xMin", 0))),
    )
    narrower = min(bbox_width(a), bbox_width(b))
    return overlap / narrower if narrower > 0 else 0.0


def crop_is_wide_for_chunking(crop_path: Path, min_width: int = 460) -> bool:
    with Image.open(crop_path) as image:
        return image.width >= min_width


def candidate_needs_extended_timeout(candidate: Optional[dict[str, Any]]) -> bool:
    if not candidate:
        return False
    profiles = set(candidate.get("profiles") or [])
    if "fraction-stack-line" in profiles:
        return True
    strokes = [stroke for stroke in candidate.get("strokes") or [] if stroke.get("canvasBbox")]
    bbox = candidate.get("bbox") or candidate.get("tightBbox") or (bbox_for_strokes(strokes) if strokes else {})
    width = bbox_width(bbox)
    height = bbox_height(bbox)
    if len(strokes) >= 10 and height >= 80:
        return True
    if len(strokes) >= 8 and width >= 240 and height >= 90:
        return True
    return False


def candidate_has_local_fraction_structure(candidate: Optional[dict[str, Any]]) -> bool:
    if not candidate:
        return False
    strokes = [stroke for stroke in candidate.get("strokes") or [] if stroke.get("canvasBbox")]
    return bool(find_local_fraction_groups(strokes))


def split_crop_into_horizontal_chunks(
    crop_path: Path,
    *,
    min_gap: int = 16,
    max_chunk_width: int = 340,
    blank_ink_max: int = 2,
) -> list[dict[str, Any]]:
    with Image.open(crop_path).convert("L") as image:
        ink_counts = [
            sum(1 for y in range(image.height) if image.getpixel((x, y)) < 200)
            for x in range(image.width)
        ]
        image_copy = image.copy()
    ink_columns = [index for index, count in enumerate(ink_counts) if count > blank_ink_max]
    if not ink_columns:
        return []
    content_start = min(ink_columns)
    content_end = max(ink_columns) + 1

    blank_runs: list[tuple[int, int]] = []
    run_start: Optional[int] = None
    for x in range(content_start, content_end):
        if ink_counts[x] <= blank_ink_max:
            if run_start is None:
                run_start = x
        elif run_start is not None:
            if x - run_start >= min_gap:
                blank_runs.append((run_start, x))
            run_start = None
    if run_start is not None and content_end - run_start >= min_gap:
        blank_runs.append((run_start, content_end))

    atoms: list[tuple[int, int]] = []
    start = content_start
    for gap_start, gap_end in blank_runs:
        if gap_start > start:
            atoms.append((start, gap_start))
        start = gap_end
    if start < content_end:
        atoms.append((start, content_end))

    chunks: list[dict[str, Any]] = []
    pending: Optional[tuple[int, int]] = None

    def flush_pending() -> None:
        nonlocal pending
        if pending and pending[1] - pending[0] > 4:
            chunks.append({"xRange": pending})
        pending = None

    for atom in atoms:
        if is_equals_crop_atom(image_copy, atom):
            flush_pending()
            chunks.append({"xRange": atom, "literalLatex": "="})
            continue
        if pending is None:
            pending = atom
            continue
        proposed = (pending[0], atom[1])
        if proposed[1] - proposed[0] > max_chunk_width:
            flush_pending()
            pending = atom
        else:
            pending = proposed
    flush_pending()
    return chunks


def is_equals_crop_atom(image: Image.Image, atom: tuple[int, int]) -> bool:
    x0, x1 = atom
    pixels = [
        (x, y)
        for x in range(x0, x1)
        for y in range(image.height)
        if image.getpixel((x, y)) < 200
    ]
    if not pixels:
        return False
    min_x = min(x for x, _ in pixels)
    max_x = max(x for x, _ in pixels) + 1
    min_y = min(y for _, y in pixels)
    max_y = max(y for _, y in pixels) + 1
    width = max_x - min_x
    height = max_y - min_y
    if height <= 0 or width < height * 1.35:
        return False

    row_counts = [
        sum(1 for x in range(min_x, max_x) if image.getpixel((x, y)) < 200)
        for y in range(min_y, max_y)
    ]
    active_rows = [index for index, count in enumerate(row_counts) if count >= max(2, width * 0.35)]
    bands = 0
    previous = None
    for row in active_rows:
        if previous is None or row > previous + 1:
            bands += 1
        previous = row
    return bands == 2 and count_ink_components(image, min_x, min_y, max_x, max_y) == 2


def count_ink_components(image: Image.Image, min_x: int, min_y: int, max_x: int, max_y: int) -> int:
    ink = {
        (x, y)
        for x in range(min_x, max_x)
        for y in range(min_y, max_y)
        if image.getpixel((x, y)) < 200
    }
    components = 0
    while ink:
        components += 1
        stack = [ink.pop()]
        while stack:
            x, y = stack.pop()
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if (nx, ny) in ink:
                    ink.remove((nx, ny))
                    stack.append((nx, ny))
    return components


def normalize_chunked_latex(latex: str) -> str:
    return re.sub(r"\s+", " ", str(latex or "")).strip()


def choose_chunk_latex(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    top = payload_top_latex(payload)
    if not top:
        return ""

    if re.search(r"(\w|\))\s*\^\s*\{\s*f\s*\}", top):
        for candidate in candidates:
            latex = str(candidate.get("latex") or "").strip()
            if r"\prime" in latex:
                return latex

    if re.match(r"^t\s+[a-zA-Z\\]", top):
        suffix = re.sub(r"^t\s+", "", top).strip()
        for candidate in candidates:
            latex = str(candidate.get("latex") or "").strip()
            if latex.startswith("+") and latex[1:].strip() == suffix:
                return latex

    return top


def payload_needs_retry(payload: dict[str, Any]) -> bool:
    if not payload or payload.get("timedOut") or payload.get("failed"):
        return True
    if payload.get("_httpStatus", 200) >= 400:
        return True
    latex = payload_top_latex(payload)
    return not latex or is_suspicious_operation_latex(latex)


def semantic_needs_retry(latex: str, semantic: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if not latex:
        return False
    if semantic.get("equivalentToProblem") or semantic.get("equivalentToPrevious"):
        return False
    try:
        semantic_score = float(semantic.get("semanticScore", -1000))
    except (TypeError, ValueError):
        semantic_score = -1000
    if semantic_score >= 2:
        return False
    weak_function_equation = looks_like_weak_function_equation(latex, semantic)
    weak_variable_equation = looks_like_weak_variable_equation(latex, semantic)
    if (
        not looks_like_short_numeric_equation(latex) and
        not looks_malformed_for_semantic_retry(latex) and
        not weak_function_equation and
        not weak_variable_equation
    ):
        return False

    bbox = candidate.get("tightBbox") or {}
    try:
        width = float(bbox.get("xMax", 0)) - float(bbox.get("xMin", 0))
    except (TypeError, ValueError):
        width = 0
    return width <= (980 if weak_function_equation else 620)


def looks_like_short_numeric_equation(latex: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(latex or "")).strip()
    if not normalized or "=" not in normalized or not re.search(r"\d", normalized):
        return False
    without_latex_ops = re.sub(r"\\(?:cdot|times|div|pm)", "", normalized)
    if re.search(r"[A-Za-z]", without_latex_ops):
        return False
    compact = re.sub(r"[\s{}()[\].,+\-*/=^_]", "", without_latex_ops)
    return bool(re.fullmatch(r"\d{1,8}", compact))


def looks_malformed_for_semantic_retry(latex: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(latex or "")).strip()
    if not normalized or not re.search(r"\d", normalized):
        return False

    without_commands = re.sub(r"\\[A-Za-z]+", "", normalized)
    letters = re.findall(r"[A-Za-z]", without_commands)
    has_math_syntax = bool(re.search(r"[=+\-*/()]|\\(?:times|div|frac)", normalized))
    if not has_math_syntax:
        return False

    if grouping_looks_unbalanced(normalized):
        return True
    if re.search(r"\\(?:times|div)\b", normalized) and re.match(r"^[A-Za-z]\s+\d", without_commands):
        return True
    if letters and all(re.match(r"[oln]", letter, re.IGNORECASE) for letter in letters) and re.search(r"[()+\-]", normalized):
        return True
    return False


def looks_like_weak_function_equation(latex: str, semantic: dict[str, Any]) -> bool:
    normalized = re.sub(r"\s+", " ", str(latex or "")).strip()
    if "=" not in normalized or not re.search(r"[A-Za-z]", normalized):
        return False
    if not re.search(r"[A-Za-z]\s*(?:\^\s*\{[^}]+\}\s*)?\(", normalized):
        return False
    try:
        semantic_score = float(semantic.get("semanticScore", -1000))
    except (TypeError, ValueError):
        semantic_score = -1000
    if semantic_score >= 1.2:
        return False
    return not (semantic.get("equivalentToProblem") or semantic.get("equivalentToPrevious"))


def looks_like_weak_variable_equation(latex: str, semantic: dict[str, Any]) -> bool:
    normalized = re.sub(r"\s+", " ", str(latex or "")).strip()
    if "=" not in normalized or not re.search(r"[A-Za-z]", normalized) or not re.search(r"\d", normalized):
        return False
    if re.search(r"\\(?:frac|sqrt|log|ln|int|sum|prod)\b", normalized):
        return False
    if re.search(r"[A-Za-z]\s*(?:\^\s*\{[^}]+\}\s*)?\(", normalized):
        return False
    without_commands = re.sub(r"\\[A-Za-z]+", "", normalized)
    if not re.search(r"[+\-]|\b\d+\s*[A-Za-z]\b|\b[A-Za-z]\s*\^\s*\{", without_commands):
        return False
    try:
        semantic_score = float(semantic.get("semanticScore", -1000))
    except (TypeError, ValueError):
        semantic_score = -1000
    if semantic_score >= 2:
        return False
    return not (semantic.get("equivalentToProblem") or semantic.get("equivalentToPrevious"))


def grouping_looks_unbalanced(latex: str) -> bool:
    pairs = {"(": ")", "[": "]", "{": "}"}
    closers = set(pairs.values())
    stack: list[str] = []
    for char in str(latex or ""):
        if char in pairs:
            stack.append(pairs[char])
        elif char in closers:
            if not stack or stack.pop() != char:
                return True
    return bool(stack)


def payload_top_latex(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if payload.get("top"):
        return str(payload["top"].get("latex") or "").strip()
    if candidates:
        return str(candidates[0].get("latex") or "").strip()
    return ""


def merge_recognition_attempts(attempts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    merged_candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for attempt in attempts:
        for candidate in attempt.get("candidates") or []:
            latex = str(candidate.get("latex") or "").strip()
            if not latex or latex in seen:
                continue
            seen.add(latex)
            merged = dict(candidate)
            if attempt.get("_retryTargetPixelHeight"):
                merged["retryTargetPixelHeight"] = attempt["_retryTargetPixelHeight"]
            merged_candidates.append(merged)

    successful = [
        attempt for attempt in attempts
        if not attempt.get("timedOut")
        and attempt.get("_httpStatus", 200) < 400
        and payload_top_latex(attempt)
        and not is_suspicious_operation_latex(payload_top_latex(attempt))
    ] or [
        attempt for attempt in attempts
        if not attempt.get("timedOut")
        and attempt.get("_httpStatus", 200) < 400
        and payload_top_latex(attempt)
    ]
    best = successful[0] if successful else attempts[0]
    top = best.get("top") or (merged_candidates[0] if merged_candidates else None)
    return {
        **best,
        "timedOut": False if successful else bool(best.get("timedOut")),
        "retryUsed": len(attempts) > 1,
        "initialTargetPixelHeight": best.get("_initialTargetPixelHeight"),
        "initialCrop": best.get("_initialCrop"),
        "retryAttempts": [
            {
                "targetPixelHeight": attempt.get("_retryTargetPixelHeight"),
                "crop": attempt.get("_retryCrop"),
                "extendedTimeoutSeconds": attempt.get("_extendedTimeoutSeconds"),
                "httpStatus": attempt.get("_httpStatus", 200),
                "timedOut": bool(attempt.get("timedOut")),
                "elapsedSeconds": attempt.get("elapsedSeconds"),
                "topLatex": payload_top_latex(attempt),
            }
            for attempt in attempts[1:]
        ],
        "candidates": merged_candidates or best.get("candidates") or [],
        "top": top,
    }


def is_suspicious_operation_latex(latex: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(latex or "")).strip()
    if re.search(r"\\(?:ldots|cdots)\b", normalized):
        return True
    return bool(re.fullmatch(r"\\(?:times|div)\s+\d+\s+\d+\s+\\(?:times|div)\s+\d+\s+\d+", normalized))


def get_json(url: str, timeout_seconds: float = 8.0) -> Optional[dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def api_available(api_url: str, *, require_comer: bool) -> tuple[bool, dict[str, Any]]:
    api = api_url.rstrip("/")
    comer = get_json(f"{api}/health?model=comer")
    detector = get_json(f"{api}/segment-lines/health")
    available = comer is not None and detector is not None
    if require_comer:
        available = available and bool(comer.get("loaded")) if comer else False
    return available, {"comer": comer, "detector": detector}


def render_fixture(
    problem: MathProblem,
    spacing: str,
    seed: int,
    ink_style: str = "normal",
    *,
    gap_pattern: str = "",
    line_gaps: Optional[Sequence[float]] = None,
) -> RenderedFixture:
    _, _, _, gaps = placements_for(problem, spacing, line_gaps)
    board = build_board(problem.name, spacing=spacing, line_gaps=gaps, seed=seed, ink_style=ink_style)
    payload = fixture_payload(problem, spacing, gaps, board, ink_style=ink_style)
    if gap_pattern:
        payload["fixture"]["gapPattern"] = gap_pattern
    layout = fixture_layout_slug(spacing, gap_pattern)
    fixture_name = slug(problem.name, layout, ink_style) if ink_style != "normal" else slug(problem.name, layout)
    png_path = LIVE_RESULTS / f"{fixture_name}.png"
    json_path = LIVE_RESULTS / f"{fixture_name}.json"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    save_board_png(board, str(png_path))
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return RenderedFixture(problem, spacing, ink_style, board, payload, png_path, json_path, gap_pattern, tuple(gaps))


def fixture_layout_slug(spacing: str, gap_pattern: str = "") -> str:
    return f"{spacing}-{gap_pattern}" if gap_pattern else spacing


def fixture_layout_label(fixture: RenderedFixture) -> str:
    return f"{fixture.spacing}:{fixture.gap_pattern}" if fixture.gap_pattern else fixture.spacing


def fixture_run_slug(fixture: RenderedFixture, order: str) -> str:
    return slug(fixture.problem.name, fixture_layout_slug(fixture.spacing, fixture.gap_pattern), fixture.ink_style, order)


def safe_filename_slug(value: str, max_length: int = 96) -> str:
    text = slug(value or "item")
    if len(text) <= max_length:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    prefix = text[:max(1, max_length - len(digest) - 1)].rstrip("-")
    return f"{prefix}-{digest}"


def fixture_run_label(fixture: RenderedFixture, order: Optional[str] = None) -> str:
    parts = [fixture.problem.name, fixture_layout_label(fixture), fixture.ink_style]
    if order:
        parts.append(order)
    return "/".join(parts)


def run_js_segmenter(
    board_payload: dict[str, Any],
    detections: Sequence[dict[str, Any]],
    order: str,
    score_by_candidate_id: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    segment_payload = {
        **board_payload,
        "detections": list(detections),
    }
    if score_by_candidate_id:
        segment_payload["scoreByCandidateId"] = score_by_candidate_id
    completed = subprocess.run(
        ["node", str(SEGMENTER), order],
        cwd=str(ROOT),
        input=json.dumps(segment_payload),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def validate_selected_lines(selected: Sequence[dict[str, Any]], expected_count: int) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if len(selected) != expected_count:
        failures.append(f"expected {expected_count} selected lines, got {len(selected)}")

    seen: list[int] = []
    for index, candidate in enumerate(selected):
        line_set = candidate.get("syntheticLineSets") or []
        if len(line_set) != 1:
            failures.append(f"candidate {index + 1} mixes source lines {line_set}")
            continue
        seen.append(int(line_set[0]))

    missing = sorted(set(range(expected_count)) - set(seen))
    duplicates = sorted(index for index in set(seen) if seen.count(index) > 1)
    if missing:
        failures.append(f"missing source lines {missing}")
    if duplicates:
        failures.append(f"duplicate source lines {duplicates}")
    return not failures, failures


def run_detector(api_url: str, png_path: Path, timeout_seconds: float) -> dict[str, Any]:
    payload = post_image(f"{api_url.rstrip('/')}/segment-lines", png_path, timeout_seconds)
    if payload.get("_httpStatus", 500) >= 400:
        return {
            "failed": True,
            "error": payload.get("detail") or f"HTTP {payload.get('_httpStatus')}",
            "payload": payload,
            "detections": [],
        }
    return {
        "failed": False,
        "detections": payload.get("detections") or [],
        "rawDetections": payload.get("detections") or [],
        "elapsedSeconds": payload.get("elapsedSeconds"),
        "model": payload.get("model"),
        "imageWidth": payload.get("imageWidth"),
        "imageHeight": payload.get("imageHeight"),
    }


def crop_selected_candidate(
    fixture: RenderedFixture,
    candidate: dict[str, Any],
    output_path: Path,
    padding: int = 24,
) -> Path:
    bbox = candidate.get("bbox") or {}
    line_indexes = [int(index) for index in candidate.get("syntheticLineSets") or []]
    if not bbox or not line_indexes:
        return crop_board_bbox(fixture.png_path, bbox, output_path, padding)

    x_min = math.floor(float(bbox["xMin"])) - padding
    y_min = math.floor(float(bbox["yMin"])) - padding
    x_max = math.ceil(float(bbox["xMax"])) + padding
    y_max = math.ceil(float(bbox["yMax"])) + padding
    width = max(1, x_max - x_min)
    height = max(1, y_max - y_min)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    lines = fixture.payload.get("lines") or []
    for line_index in line_indexes:
        if line_index < 0 or line_index >= len(lines):
            continue
        for contour in lines[line_index].get("contours") or []:
            points = [(float(point["x"]) - x_min, float(point["y"]) - y_min) for point in contour]
            if len(points) >= 3:
                draw.polygon(points, fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def crop_board_bbox(board_png: Path, bbox: dict[str, Any], output_path: Path, padding: int = 24) -> Path:
    image = Image.open(board_png).convert("RGB")
    if not bbox:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return output_path

    x_min = max(0, math.floor(float(bbox["xMin"])) - padding)
    y_min = max(0, math.floor(float(bbox["yMin"])) - padding)
    x_max = min(image.width, math.ceil(float(bbox["xMax"])) + padding)
    y_max = min(image.height, math.ceil(float(bbox["yMax"])) + padding)
    crop = image.crop((x_min, y_min, x_max, y_max))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)
    return output_path


def recognize_selected_lines(
    api_url: str,
    fixture: RenderedFixture,
    selected: Sequence[dict[str, Any]],
    *,
    order: str,
    timeout_seconds: float,
    initial_raster_height: int = DEFAULT_INITIAL_RASTER_HEIGHT,
    initial_raster_min_height: int = DEFAULT_INITIAL_RASTER_MIN_HEIGHT,
    structural_timeout_seconds: float = 0,
    progress: bool = False,
    checkpoint_callback: Optional[Callable[[list[dict[str, Any]]], None]] = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    previous_semantic_latex: list[str] = []
    problem_latex = fixture.problem.context_latex
    recognize_url = f"{api_url.rstrip('/')}/recognize?model=comer&timeout_seconds={timeout_seconds}"
    label = fixture_run_label(fixture, order)

    for line_index, candidate in enumerate(selected):
        if progress:
            source_lines = ",".join(str(index) for index in candidate.get("syntheticLineSets") or [])
            print(f"[progress] OCR selected {label} line {line_index + 1}/{len(selected)} source={source_lines}", flush=True)
        crop_path = LIVE_RESULTS / "crops" / (
            f"{fixture_run_slug(fixture, order)}_line_{line_index + 1}.png"
        )
        crop_selected_candidate(fixture, candidate, crop_path)
        started = time.monotonic()
        payload = recognize_crop_with_retries(
            recognize_url,
            crop_path,
            timeout_seconds,
            retry_on_failure=True,
            initial_raster_height=initial_raster_height,
            initial_raster_min_height=initial_raster_min_height,
            fixture=fixture,
            candidate=candidate,
            extended_timeout_seconds=structural_timeout_seconds,
        )
        elapsed = round(time.monotonic() - started, 3)
        candidates = payload.get("candidates") or []
        top_latex = ""
        if payload.get("top"):
            top_latex = str(payload["top"].get("latex") or "")
        elif candidates:
            top_latex = str(candidates[0].get("latex") or "")

        source_lines = [int(index) for index in candidate.get("syntheticLineSets") or []]
        expected_lines = [
            fixture.problem.lines[index]
            for index in source_lines
            if 0 <= index < len(fixture.problem.lines)
        ]
        expected_latex = r" \\ ".join(expected_lines)
        semantic = score_candidate_group(
            {
                "candidateId": candidate.get("candidateId"),
                "latex": top_latex,
                "candidates": candidates,
                "elapsedSeconds": payload.get("elapsedSeconds"),
            },
            problem_latex=problem_latex,
            previous_latex=previous_semantic_latex,
        )
        semantic_retry_used = False
        if semantic_needs_retry(top_latex, semantic, candidate):
            payload = recognize_crop_with_semantic_retries(
                recognize_url,
                crop_path,
                timeout_seconds,
                payload,
                initial_raster_height=initial_raster_height,
                fixture=fixture,
                candidate=candidate,
            )
            candidates = payload.get("candidates") or []
            top_latex = payload_top_latex(payload)
            semantic = score_candidate_group(
                {
                    "candidateId": candidate.get("candidateId"),
                    "latex": top_latex,
                    "candidates": candidates,
                    "elapsedSeconds": payload.get("elapsedSeconds"),
                },
                problem_latex=problem_latex,
                previous_latex=previous_semantic_latex,
            )
            semantic_retry_used = True
        semantic_latex = str(semantic.get("bestLatex") or "")
        trusted_latex = trusted_semantic_latex(top_latex, semantic)
        operation_repair = repair_standalone_operation_latex(trusted_latex, semantic)
        contextual_operation_repair = ""
        operation_inference = ""
        if operation_repair:
            trusted_latex = operation_repair
            semantic_latex = operation_repair
        else:
            contextual_operation_repair = repair_operation_annotation_from_previous(
                trusted_latex,
                previous_semantic_latex,
            )
            if contextual_operation_repair:
                trusted_latex = contextual_operation_repair
                semantic_latex = contextual_operation_repair
        if not trusted_latex:
            operation_inference = infer_empty_operation_annotation(candidate, previous_semantic_latex)
            if operation_inference:
                trusted_latex = operation_inference
                semantic_latex = operation_inference
        if trusted_latex:
            previous_semantic_latex.append(trusted_latex)

        line_record = {
            "lineIndex": line_index,
            "candidateId": candidate.get("candidateId"),
            "profiles": candidate.get("profiles") or [],
            "sourceLineIndexes": source_lines,
            "expectedLatex": expected_latex,
            "crop": str(crop_path),
            "httpStatus": payload.get("_httpStatus", 200),
            "elapsedSeconds": payload.get("elapsedSeconds", elapsed),
            "timedOut": bool(payload.get("timedOut")),
            "initialTargetPixelHeight": payload.get("initialTargetPixelHeight") or payload.get("_initialTargetPixelHeight"),
            "initialCrop": payload.get("initialCrop") or payload.get("_initialCrop"),
            "retryUsed": bool(payload.get("retryUsed")),
            "retryAttempts": payload.get("retryAttempts") or [],
            "semanticRetryUsed": semantic_retry_used or bool(payload.get("semanticRetryUsed")),
            "chunkFallback": bool(payload.get("chunkFallback")),
            "chunkAttempts": payload.get("chunkAttempts") or [],
            "topLatex": top_latex,
            "semanticBestLatex": semantic_latex,
            "acceptedLatex": trusted_latex,
            "ocrRepair": {
                "source": "standalone-operation",
                "repairedLatex": operation_repair,
            } if operation_repair else ({
                "source": "contextual-operation",
                "repairedLatex": contextual_operation_repair,
            } if contextual_operation_repair else ({
                "source": "inferred-empty-operation",
                "repairedLatex": operation_inference,
            } if operation_inference else None)),
            "topComparison": compare_latex(top_latex, expected_latex),
            "semanticComparison": compare_latex(semantic_latex, expected_latex),
            "acceptedComparison": compare_latex(trusted_latex, expected_latex),
            "semantic": semantic,
            "topCandidates": candidates[:5],
            "error": payload.get("detail") if payload.get("_httpStatus", 200) >= 400 else None,
        }
        records.append(line_record)
        if checkpoint_callback:
            checkpoint_callback(records)
    return records


def recognize_candidate_alternatives(
    api_url: str,
    fixture: RenderedFixture,
    candidates: Sequence[dict[str, Any]],
    *,
    order: str,
    timeout_seconds: float,
    retry_candidate_ids: Optional[set[str]] = None,
    max_extra_candidates: int = 0,
    candidate_time_budget_seconds: float = 0,
    extra_candidate_timeout_seconds: float = 0,
    initial_raster_height: int = DEFAULT_INITIAL_RASTER_HEIGHT,
    initial_raster_min_height: int = DEFAULT_INITIAL_RASTER_MIN_HEIGHT,
    structural_timeout_seconds: float = 0,
    progress: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    problem_latex = fixture.problem.context_latex
    recognize_url = f"{api_url.rstrip('/')}/recognize?model=comer&timeout_seconds={timeout_seconds}"
    label = fixture_run_label(fixture, order)
    required_ids = {str(candidate_id) for candidate_id in (retry_candidate_ids or set())}
    ordered_candidates = ordered_recognizable_candidates(candidates, required_ids)
    budget_started = time.monotonic()
    extra_count = 0

    for index, candidate in ordered_candidates:
        candidate_id = str(candidate.get("candidateId") or "")
        required = candidate_id in required_ids
        if not required and max_extra_candidates <= 0:
            continue
        if not required and max_extra_candidates > 0 and extra_count >= max_extra_candidates:
            continue
        if (
            not required and
            candidate_time_budget_seconds > 0 and
            time.monotonic() - budget_started >= candidate_time_budget_seconds
        ):
            continue
        candidate_timeout = candidate_recognition_timeout(
            timeout_seconds,
            required=required,
            extra_candidate_timeout_seconds=extra_candidate_timeout_seconds,
            remaining_extra_budget_seconds=(
                candidate_time_budget_seconds - (time.monotonic() - budget_started)
                if not required and candidate_time_budget_seconds > 0
                else 0
            ),
        )
        if progress:
            source_lines = ",".join(str(line_index) for line_index in candidate.get("syntheticLineSets") or [])
            profiles = ",".join(candidate.get("profiles") or [])
            marker = "selected" if required else "extra"
            print(
                f"[progress] OCR candidate {label} {index + 1}/{len(candidates)} "
                f"{marker} timeout={candidate_timeout:g}s source={source_lines} profiles={profiles}",
                flush=True,
            )
        crop_path = LIVE_RESULTS / "candidate_crops" / (
            f"{fixture_run_slug(fixture, order)}_{index + 1}.png"
        )
        crop_selected_candidate(fixture, candidate, crop_path)
        started = time.monotonic()
        payload = recognize_crop_with_retries(
            recognize_url,
            crop_path,
            candidate_timeout,
            retry_on_failure=required,
            initial_raster_height=initial_raster_height,
            initial_raster_min_height=initial_raster_min_height,
            fixture=fixture,
            candidate=candidate,
            extended_timeout_seconds=structural_timeout_seconds,
        )
        if not required:
            extra_count += 1
        elapsed = round(time.monotonic() - started, 3)
        candidates_payload = payload.get("candidates") or []
        top_latex = ""
        if payload.get("top"):
            top_latex = str(payload["top"].get("latex") or "")
        elif candidates_payload:
            top_latex = str(candidates_payload[0].get("latex") or "")

        source_lines = [int(line_index) for line_index in candidate.get("syntheticLineSets") or []]
        expected_lines = [
            fixture.problem.lines[line_index]
            for line_index in source_lines
            if 0 <= line_index < len(fixture.problem.lines)
        ]
        expected_latex = r" \\ ".join(expected_lines)
        semantic = score_candidate_group(
            {
                "candidateId": candidate.get("candidateId"),
                "latex": top_latex,
                "candidates": candidates_payload,
                "elapsedSeconds": payload.get("elapsedSeconds"),
            },
            problem_latex=problem_latex,
            previous_latex=[],
        )
        semantic_latex = str(semantic.get("bestLatex") or "")
        accepted_latex = trusted_semantic_latex(top_latex, semantic)

        records.append({
            "candidateId": candidate.get("candidateId"),
            "profiles": candidate.get("profiles") or [],
            "bbox": candidate.get("bbox") or {},
            "tightBbox": candidate.get("tightBbox") or candidate.get("bbox") or {},
            "sourceLineIndexes": source_lines,
            "expectedLatex": expected_latex,
            "crop": str(crop_path),
            "httpStatus": payload.get("_httpStatus", 200),
            "elapsedSeconds": payload.get("elapsedSeconds", elapsed),
            "timeoutSeconds": candidate_timeout,
            "timedOut": bool(payload.get("timedOut")),
            "initialTargetPixelHeight": payload.get("initialTargetPixelHeight") or payload.get("_initialTargetPixelHeight"),
            "initialCrop": payload.get("initialCrop") or payload.get("_initialCrop"),
            "retryUsed": bool(payload.get("retryUsed")),
            "retryAttempts": payload.get("retryAttempts") or [],
            "chunkFallback": bool(payload.get("chunkFallback")),
            "chunkAttempts": payload.get("chunkAttempts") or [],
            "topLatex": top_latex,
            "semanticBestLatex": semantic_latex,
            "acceptedLatex": accepted_latex,
            "topComparison": compare_latex(top_latex, expected_latex),
            "semanticComparison": compare_latex(semantic_latex, expected_latex),
            "acceptedComparison": compare_latex(accepted_latex, expected_latex),
            "semantic": semantic,
            "semanticScore": semantic.get("semanticScore", -1000),
            "topCandidates": candidates_payload[:5],
            "error": payload.get("detail") if payload.get("_httpStatus", 200) >= 400 else None,
        })
    return records


def candidate_recognition_timeout(
    timeout_seconds: float,
    *,
    required: bool,
    extra_candidate_timeout_seconds: float = 0,
    remaining_extra_budget_seconds: float = 0,
) -> float:
    timeout = max(0.1, float(timeout_seconds))
    if required:
        return timeout
    extra_timeout = float(extra_candidate_timeout_seconds or 0)
    if extra_timeout > 0:
        timeout = min(timeout, max(0.1, extra_timeout))
    remaining_budget = float(remaining_extra_budget_seconds or 0)
    if remaining_budget > 0:
        timeout = min(timeout, max(0.1, remaining_budget))
    return timeout


def ordered_recognizable_candidates(
    candidates: Sequence[dict[str, Any]],
    required_candidate_ids: set[str],
) -> list[tuple[int, dict[str, Any]]]:
    ordered = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if should_recognize_candidate(candidate)
    ]
    return sorted(
        ordered,
        key=lambda item: (
            candidate_priority(item[1], required_candidate_ids),
            candidate_top(item[1]),
            item[0],
        ),
    )


def candidate_priority(candidate: dict[str, Any], required_candidate_ids: set[str]) -> int:
    candidate_id = str(candidate.get("candidateId") or "")
    if candidate_id in required_candidate_ids:
        return 0

    profiles = set(candidate.get("profiles") or [])
    if profiles & {"dbnet-line", "fraction-stack-line", "row-line", "raw-row-line", "strict", "loose"}:
        return 1
    if profiles & {"dbnet-parent", "temporal", "parent"}:
        return 2
    if profiles & {"projection-line"}:
        return 3
    return 4


def candidate_top(candidate: dict[str, Any]) -> float:
    bbox = candidate.get("bbox") or candidate.get("tightBbox") or {}
    try:
        return float(bbox.get("yMin", 0))
    except (TypeError, ValueError):
        return 0.0


def should_recognize_candidate(candidate: dict[str, Any]) -> bool:
    profiles = set(candidate.get("profiles") or [])
    return bool(profiles & {
        "parent",
        "loose",
        "strict",
        "temporal",
        "row-line",
        "raw-row-line",
        "fraction-stack-line",
        "projection-line",
        "dbnet-parent",
        "dbnet-line",
    })


def selected_records_from_alternatives(
    selected: Sequence[dict[str, Any]],
    alternatives: Sequence[dict[str, Any]],
    *,
    api_url: str = "",
    fixture: Optional[RenderedFixture] = None,
    order: str = "line-order",
    timeout_seconds: float = 20.0,
    initial_raster_height: int = DEFAULT_INITIAL_RASTER_HEIGHT,
    structural_timeout_seconds: float = 0,
) -> list[dict[str, Any]]:
    by_id = {record["candidateId"]: record for record in alternatives}
    records: list[dict[str, Any]] = []
    previous_semantic_latex: list[str] = []
    problem_latex = fixture.problem.context_latex if fixture else ""
    recognize_url = f"{api_url.rstrip('/')}/recognize?model=comer&timeout_seconds={timeout_seconds}" if api_url else ""

    for index, candidate in enumerate(selected):
        record = dict(by_id.get(candidate.get("candidateId")) or {
            "candidateId": candidate.get("candidateId"),
            "profiles": candidate.get("profiles") or [],
            "sourceLineIndexes": candidate.get("syntheticLineSets") or [],
            "expectedLatex": r" \\ ".join(candidate.get("syntheticLatex") or []),
            "topLatex": "",
            "semanticBestLatex": "",
            "acceptedLatex": "",
            "topComparison": compare_latex("", r" \\ ".join(candidate.get("syntheticLatex") or [])),
            "semanticComparison": compare_latex("", r" \\ ".join(candidate.get("syntheticLatex") or [])),
            "acceptedComparison": compare_latex("", r" \\ ".join(candidate.get("syntheticLatex") or [])),
            "topCandidates": [],
            "error": "candidate was not recognized as an alternative",
        })
        record["lineIndex"] = index
        if fixture and recognize_url and record.get("crop"):
            record = refine_selected_alternative_record(
                record,
                candidate,
                fixture=fixture,
                recognize_url=recognize_url,
                timeout_seconds=timeout_seconds,
                initial_raster_height=initial_raster_height,
                structural_timeout_seconds=structural_timeout_seconds,
                previous_semantic_latex=previous_semantic_latex,
                problem_latex=problem_latex,
            )
        trusted = (
            str(record.get("acceptedLatex") or "").strip() or
            trusted_semantic_latex(record.get("topLatex") or "", record.get("semantic") or {})
        )
        if trusted:
            previous_semantic_latex.append(trusted)
        records.append(record)
    return records


def refine_selected_alternative_record(
    record: dict[str, Any],
    candidate: dict[str, Any],
    *,
    fixture: RenderedFixture,
    recognize_url: str,
    timeout_seconds: float,
    initial_raster_height: int,
    structural_timeout_seconds: float,
    previous_semantic_latex: Sequence[str],
    problem_latex: str,
) -> dict[str, Any]:
    crop_path = Path(str(record.get("crop") or ""))
    if not crop_path.exists():
        return record

    top_latex = str(record.get("topLatex") or "")
    candidates = list(record.get("topCandidates") or [])
    payload: dict[str, Any] = {
        "_httpStatus": record.get("httpStatus", 200),
        "_initialTargetPixelHeight": record.get("initialTargetPixelHeight"),
        "_initialCrop": record.get("initialCrop"),
        "top": {"latex": top_latex} if top_latex else None,
        "candidates": candidates,
        "elapsedSeconds": record.get("elapsedSeconds"),
        "retryAttempts": record.get("retryAttempts") or [],
        "retryUsed": bool(record.get("retryUsed")),
    }

    if payload_needs_retry(payload) and not record.get("chunkFallback"):
        payload = recognize_crop_with_retries(
            recognize_url,
            crop_path,
            timeout_seconds,
            retry_on_failure=True,
            initial_raster_height=initial_raster_height,
            fixture=fixture,
            candidate=candidate,
            extended_timeout_seconds=structural_timeout_seconds,
        )
        top_latex = payload_top_latex(payload)
        candidates = list(payload.get("candidates") or [])

    semantic = score_candidate_group(
        {
            "candidateId": candidate.get("candidateId"),
            "latex": top_latex,
            "candidates": candidates,
            "elapsedSeconds": payload.get("elapsedSeconds"),
        },
        problem_latex=problem_latex,
        previous_latex=previous_semantic_latex,
    )
    semantic_retry_used = False
    if semantic_needs_retry(top_latex, semantic, candidate):
        payload = recognize_crop_with_semantic_retries(
            recognize_url,
            crop_path,
            timeout_seconds,
            payload,
            initial_raster_height=initial_raster_height,
            fixture=fixture,
            candidate=candidate,
        )
        candidates = payload.get("candidates") or []
        top_latex = payload_top_latex(payload)
        semantic = score_candidate_group(
            {
                "candidateId": candidate.get("candidateId"),
                "latex": top_latex,
                "candidates": candidates,
                "elapsedSeconds": payload.get("elapsedSeconds"),
            },
            problem_latex=problem_latex,
            previous_latex=previous_semantic_latex,
        )
        semantic_retry_used = True

    semantic_latex = str(semantic.get("bestLatex") or "")
    trusted_latex = trusted_semantic_latex(top_latex, semantic)
    operation_repair = repair_standalone_operation_latex(trusted_latex, semantic)
    contextual_operation_repair = ""
    if operation_repair:
        semantic_latex = operation_repair
        trusted_latex = operation_repair
    else:
        contextual_operation_repair = repair_operation_annotation_from_previous(
            trusted_latex,
            previous_semantic_latex,
        )
        if contextual_operation_repair:
            semantic_latex = contextual_operation_repair
            trusted_latex = contextual_operation_repair

    expected_latex = record.get("expectedLatex") or r" \\ ".join(candidate.get("syntheticLatex") or [])
    return {
        **record,
        "httpStatus": payload.get("_httpStatus", record.get("httpStatus", 200)),
        "elapsedSeconds": payload.get("elapsedSeconds", record.get("elapsedSeconds")),
        "timedOut": bool(payload.get("timedOut")),
        "initialTargetPixelHeight": payload.get("initialTargetPixelHeight") or payload.get("_initialTargetPixelHeight"),
        "initialCrop": payload.get("initialCrop") or payload.get("_initialCrop"),
        "retryUsed": bool(payload.get("retryUsed")),
        "retryAttempts": payload.get("retryAttempts") or [],
        "semanticRetryUsed": semantic_retry_used or bool(payload.get("semanticRetryUsed")),
        "chunkFallback": bool(payload.get("chunkFallback", record.get("chunkFallback"))),
        "chunkAttempts": payload.get("chunkAttempts") or record.get("chunkAttempts") or [],
        "topLatex": top_latex,
        "semanticBestLatex": semantic_latex,
        "acceptedLatex": trusted_latex,
        "ocrRepair": {
            "source": "standalone-operation",
            "repairedLatex": operation_repair,
        } if operation_repair else ({
            "source": "contextual-operation",
            "repairedLatex": contextual_operation_repair,
        } if contextual_operation_repair else record.get("ocrRepair")),
        "topComparison": compare_latex(top_latex, expected_latex),
        "semanticComparison": compare_latex(semantic_latex, expected_latex),
        "acceptedComparison": compare_latex(trusted_latex, expected_latex),
        "semantic": semantic,
        "topCandidates": candidates[:5],
        "error": payload.get("detail") if payload.get("_httpStatus", 200) >= 400 else record.get("error"),
    }


def trusted_semantic_latex(current_latex: str, semantic: dict[str, Any]) -> str:
    best_latex = str(semantic.get("bestLatex") or "").strip()
    current = str(current_latex or "").strip()
    if not best_latex:
        return current
    if not current or current == best_latex:
        return best_latex
    if latex_kind(current) == "operation" and latex_kind(best_latex) != "operation":
        return current
    current_score = semantic_score_for_latex(semantic, current)
    if (
        current_score and
        current_score.get("sound") is True and
        (current_score.get("equivalentToProblem") or current_score.get("equivalentToPrevious"))
    ):
        return current
    if semantic.get("equivalentToProblem") or semantic.get("equivalentToPrevious"):
        return best_latex
    try:
        if float(semantic.get("semanticScore", -1000)) >= 3:
            return best_latex
    except (TypeError, ValueError):
        pass

    for candidate in semantic.get("candidateScores") or []:
        if str(candidate.get("latex") or "").strip() == current and candidate.get("sound") is False and semantic.get("sound"):
            return best_latex
    if should_trust_contextual_semantic_best(current, best_latex, semantic, current_score):
        return best_latex
    return current


def should_trust_contextual_semantic_best(
    current_latex: str,
    best_latex: str,
    semantic: dict[str, Any],
    current_score: Optional[dict[str, Any]],
) -> bool:
    if not current_score or current_score.get("equivalentToProblem") or current_score.get("equivalentToPrevious"):
        return False
    best_score = semantic_score_for_latex(semantic, best_latex)
    if not best_score or best_score.get("sound") is not True:
        return False
    if best_score.get("detail", {}).get("duplicatePreviousLatex"):
        return False
    if latex_kind(current_latex) != latex_kind(best_latex):
        return False
    try:
        best_value = float(best_score.get("score"))
        current_value = float(current_score.get("score"))
    except (TypeError, ValueError):
        return False
    if best_value <= current_value:
        return False
    best_overlap = safe_float(best_score.get("detail", {}).get("characterOverlap"), -1.0)
    current_overlap = safe_float(current_score.get("detail", {}).get("characterOverlap"), -1.0)
    return best_overlap >= current_overlap + 0.08


def semantic_score_for_latex(semantic: dict[str, Any], latex: str) -> Optional[dict[str, Any]]:
    current = str(latex or "").strip()
    for candidate in semantic.get("candidateScores") or []:
        if str(candidate.get("latex") or "").strip() == current:
            return candidate
    return None


def latex_kind(latex: str) -> str:
    try:
        return parse_math(latex).kind
    except ParseFailure:
        return ""


def repair_standalone_operation_latex(latex: str, semantic: dict[str, Any]) -> str:
    if semantic.get("equivalentToProblem") or semantic.get("equivalentToPrevious"):
        return ""
    try:
        if float(semantic.get("semanticScore", -1000)) >= 2:
            return ""
    except (TypeError, ValueError):
        pass

    normalized = re.sub(r"\s+", " ", str(latex or "")).strip()
    if not normalized or re.search(r"[=<>]", normalized):
        return ""
    if re.search(r"\\(?:frac|sqrt|log|ln|int|sum|prod)\b", normalized):
        return ""

    subscripted_times = re.fullmatch(
        r"(?:x|X|\\times)\s*_\s*\{\s*(-?\d+)\s*\}\s*"
        r"\\times\s*(?:_|(?:x|X|\\times)\s*_)\s*\{\s*([A-Za-z0-9-]+)\s*\}",
        normalized,
    )
    if subscripted_times:
        left_operand, right_operand = subscripted_times.groups()
        if not re.fullmatch(r"-?\d+", right_operand):
            right_operand = left_operand
        if left_operand == right_operand:
            return rf"\times {left_operand} \times {right_operand}"

    spaced_operands = re.fullmatch(
        r"(\\(?:times|div))\s+((?:\d\s*){1,5})\s+(?:x|X|\\times|\\div)\s+((?:\d\s*){1,5})",
        normalized,
    )
    if spaced_operands:
        operator, left_raw, right_raw = spaced_operands.groups()
        left_operand = "".join(re.findall(r"\d", left_raw))
        right_operand = "".join(re.findall(r"\d", right_raw))
        if left_operand and left_operand == right_operand:
            return f"{operator} {left_operand} {operator} {right_operand}"

    plain_x_operands = re.fullmatch(
        r"(?:x|X)\s+((?:\d\s*){1,5})\s+(?:x|X)\s+((?:\d\s*){1,5})",
        normalized,
    )
    if plain_x_operands:
        left_raw, right_raw = plain_x_operands.groups()
        left_operand = "".join(re.findall(r"\d", left_raw))
        right_operand = "".join(re.findall(r"\d", right_raw))
        if left_operand and left_operand == right_operand:
            return rf"\times {left_operand} \times {right_operand}"

    without_commands = re.sub(r"\\(?:times|div|cdot|pm)\b", "", normalized)
    letters = re.findall(r"[A-Za-z]", without_commands)
    if not letters or any(letter.lower() != "x" for letter in letters):
        return ""

    explicit = re.search(r"\\(times|div)\s+(-?\d+)", normalized)
    explicit_operator = f"\\{explicit.group(1)}" if explicit else ""
    operator = explicit_operator or r"\times"
    operator_pattern = re.escape(explicit_operator) if explicit_operator else r"(?:x|X|\\times)"
    match = re.fullmatch(
        rf"(?:x|X|\\times|\\div)\s+(-?\d+)(?:\s+\d+)?\s+{operator_pattern}\s+(-?\d+)",
        normalized,
    )
    if not match:
        return ""
    left_operand, right_operand = match.groups()
    if left_operand != right_operand:
        return ""
    return f"{operator} {left_operand} {operator} {right_operand}"


def infer_empty_operation_annotation(candidate: dict[str, Any], previous_latex: Sequence[str]) -> str:
    if not candidate_looks_like_subtraction_annotation(candidate):
        return ""
    operand = additive_constant_to_remove(previous_latex)
    if not operand:
        return ""
    return f"- {operand} - {operand}"


def repair_operation_annotation_from_previous(latex: str, previous_latex: Sequence[str]) -> str:
    operand = additive_constant_to_remove(previous_latex)
    if not operand:
        return ""

    normalized = re.sub(r"\s+", " ", str(latex or "")).strip()
    if not normalized or re.search(r"[=<>]", normalized):
        return ""
    match = re.fullmatch(r"-\s+((?:\d\s*){1,5})\s+-\s+((?:\d\s*){1,5})", normalized)
    if not match:
        return ""
    left = "".join(re.findall(r"\d", match.group(1)))
    right = "".join(re.findall(r"\d", match.group(2)))
    if not left or left != right or left == operand:
        return ""
    return f"- {operand} - {operand}"


def candidate_looks_like_subtraction_annotation(candidate: dict[str, Any]) -> bool:
    strokes = [stroke for stroke in candidate.get("strokes") or [] if stroke.get("canvasBbox")]
    if len(strokes) < 4:
        return False
    bbox = candidate.get("bbox") or candidate.get("tightBbox") or bbox_for_strokes(strokes)
    if bbox_width(bbox) < 120 or bbox_height(bbox) > 90:
        return False

    horizontal_marks = [
        stroke for stroke in strokes
        if stroke_looks_like_horizontal_operator(stroke.get("canvasBbox") or {})
    ]
    if len(horizontal_marks) < 2:
        return False

    centers = [
        (
            float((stroke.get("canvasBbox") or {}).get("xMin", 0)) +
            float((stroke.get("canvasBbox") or {}).get("xMax", 0))
        ) / 2
        for stroke in horizontal_marks
    ]
    return max(centers) - min(centers) >= 70


def stroke_looks_like_horizontal_operator(box: dict[str, Any]) -> bool:
    width = bbox_width(box)
    height = bbox_height(box)
    return 16 <= width <= 90 and height <= 16 and width >= height * 2.1


def additive_constant_to_remove(previous_latex: Sequence[str]) -> str:
    for latex in reversed(previous_latex or []):
        try:
            parsed = parse_math(latex)
        except ParseFailure:
            continue
        if parsed.kind != "equation" or parsed.right is None:
            continue
        symbols = sorted(parsed.left.free_symbols | parsed.right.free_symbols, key=lambda item: item.name)
        if len(symbols) != 1:
            continue
        variable = symbols[0]
        if parsed.right.free_symbols:
            continue
        constant = parsed.left.subs(variable, 0)
        if constant == 0 or constant.is_number is not True or constant.is_positive is not True:
            continue
        return sympy_value_to_latex(constant)
    return ""


def line_ocr_matches(line: dict[str, Any]) -> bool:
    accepted = line.get("acceptedComparison")
    if accepted:
        return bool(accepted.get("match"))
    return bool(
        line.get("topComparison", {}).get("match") or
        line.get("semanticComparison", {}).get("match")
    )


def ocr_failures_for_record(record: dict[str, Any]) -> list[str]:
    ocr = record.get("ocr") or {}
    if not ocr.get("enabled"):
        return []
    if not record.get("pipelineSelection", {}).get("exactLineCover"):
        return []

    expected_count = int(record.get("fixture", {}).get("expectedLines") or 0)
    lines = ocr.get("lines") or []
    failures: list[str] = []
    if len(lines) != expected_count:
        failures.append(f"expected {expected_count} OCR lines, got {len(lines)}")

    for index, line in enumerate(lines):
        if line_ocr_matches(line):
            continue
        line_number = int(line.get("lineIndex", index)) + 1
        expected = line.get("expectedLatex") or ""
        top = line.get("topLatex") or ""
        semantic = line.get("semanticBestLatex") or ""
        accepted = line.get("acceptedLatex") or top or semantic
        failures.append(
            f"line {line_number} OCR miss: expected {expected!r}, "
            f"accepted {accepted!r}, top {top!r}, semantic {semantic!r}"
        )
    return failures


def summary_failures(summary: dict[str, Any], *, require_ocr: bool = True) -> list[str]:
    failures: list[str] = []
    for record in summary.get("records") or []:
        fixture = record.get("fixture") or {}
        label = "/".join(str(fixture.get(key) or "") for key in ("problem", "spacing", "order"))
        segmentation = record.get("segmentation") or {}
        if not segmentation.get("exactLineCover"):
            details = "; ".join(str(item) for item in segmentation.get("failures") or ["segmentation failed"])
            failures.append(f"{label}: {details}")

        pipeline = record.get("pipelineSelection") or {}
        if not pipeline.get("exactLineCover"):
            details = "; ".join(str(item) for item in pipeline.get("failures") or ["pipeline selection failed"])
            failures.append(f"{label}: {details}")
            continue

        if require_ocr:
            for failure in ocr_failures_for_record(record):
                failures.append(f"{label}: {failure}")
    return failures


def threshold_gate_failures(
    summary: dict[str, Any],
    *,
    min_segmentation_exact_rate: Optional[float] = None,
    min_pipeline_exact_rate: Optional[float] = None,
    min_accepted_strict_rate: Optional[float] = None,
    min_accepted_match_rate: Optional[float] = None,
) -> list[str]:
    totals = summary.get("totals") or {}
    return [
        failure
        for failure in [
            threshold_failure(
                "segmentation exact rate",
                totals.get("segmentationExact", 0),
                totals.get("fixtures", 0),
                min_segmentation_exact_rate,
            ),
            threshold_failure(
                "pipeline selection exact rate",
                totals.get("pipelineSelectionExact", 0),
                totals.get("fixtures", 0),
                min_pipeline_exact_rate,
            ),
            threshold_failure(
                "accepted strict OCR rate",
                totals.get("ocrAcceptedStrictMatches", 0),
                totals.get("ocrLines", 0),
                min_accepted_strict_rate,
            ),
            threshold_failure(
                "accepted OCR match rate",
                totals.get("ocrAcceptedMatches", 0),
                totals.get("ocrLines", 0),
                min_accepted_match_rate,
            ),
        ]
        if failure
    ]


def threshold_failure(label: str, numerator: Any, denominator: Any, minimum: Optional[float]) -> str:
    if minimum is None:
        return ""
    count = int(numerator or 0)
    total = int(denominator or 0)
    actual = 1.0 if total <= 0 else count / total
    required = float(minimum)
    if actual + 1e-12 >= required:
        return ""
    return f"{label} {actual:.3f} below required {required:.3f} ({count}/{total})"


def score_map_from_alternatives(alternatives: Sequence[dict[str, Any]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for record in alternatives:
        candidate_id = str(record.get("candidateId") or "")
        if not candidate_id:
            continue
        try:
            scores[candidate_id] = float(record.get("semanticScore", -1000))
        except (TypeError, ValueError):
            scores[candidate_id] = -1000
    return scores


def apply_contextual_semantic_scores(
    alternatives: Sequence[dict[str, Any]],
    *,
    problem_latex: str,
    max_delta: float = 6.0,
    min_delta: float = -3.0,
) -> list[dict[str, Any]]:
    records = [dict(record) for record in alternatives]
    for record in records:
        context = prior_line_context_latex(record, records)
        if not context:
            continue
        base_score = safe_float(record.get("semanticScore"), -1000.0)
        contextual = score_candidate_group(
            {
                "candidateId": record.get("candidateId"),
                "latex": record.get("topLatex") or record.get("semanticBestLatex") or "",
                "candidates": record.get("topCandidates") or [],
                "elapsedSeconds": record.get("elapsedSeconds"),
            },
            problem_latex=problem_latex,
            previous_latex=context,
        )
        contextual_score = safe_float(contextual.get("semanticScore"), -1000.0)
        delta = clamp(contextual_score - base_score, min_delta, max_delta)
        record["baseSemanticScore"] = base_score
        record["semanticScore"] = round(base_score + delta, 4)
        record["contextualSemantic"] = {
            **contextual,
            "sameAnswerContext": context,
            "evidenceDelta": round(delta, 4),
        }
        if should_replace_with_contextual_latex(record, contextual):
            record["semanticBestLatex"] = str(contextual.get("bestLatex") or "")
            record["acceptedLatex"] = record["semanticBestLatex"]
            record["semanticComparison"] = compare_latex(
                record["semanticBestLatex"],
                str(record.get("expectedLatex") or ""),
            )
            record["acceptedComparison"] = compare_latex(
                record["acceptedLatex"],
                str(record.get("expectedLatex") or ""),
            )
    return records


def prior_line_context_latex(record: dict[str, Any], records: Sequence[dict[str, Any]], limit: int = 3) -> list[str]:
    box = record_box(record)
    if not box:
        return []
    chosen: list[dict[str, Any]] = []
    candidates = [
        candidate
        for candidate in records
        if candidate is not record and is_line_context_record(candidate)
    ]
    candidates = [
        candidate
        for candidate in candidates
        if (record_box(candidate) or {}).get("yMax", math.inf) <= box.get("yMin", -math.inf) + 2
    ]
    candidates.sort(key=lambda item: (
        -float((record_box(item) or {}).get("yMin", 0)),
        -safe_float(item.get("semanticScore"), -1000.0),
    ))

    for candidate in candidates:
        candidate_box = record_box(candidate)
        if not candidate_box:
            continue
        if any(vertical_overlap_ratio(record_box(selected) or {}, candidate_box) >= 0.45 for selected in chosen):
            continue
        chosen.append(candidate)
        if len(chosen) >= limit:
            break

    chosen.sort(key=lambda item: float((record_box(item) or {}).get("yMin", 0)))
    return unique_strings(record_latex_for_context(candidate) for candidate in chosen)


def is_line_context_record(record: dict[str, Any]) -> bool:
    if not record_latex_for_context(record) or not record_box(record):
        return False
    profiles = set(record.get("profiles") or [])
    if any("parent" in profile or profile == "temporal" for profile in profiles):
        return False
    return bool(profiles & {"row-line", "raw-row-line", "fraction-stack-line", "dbnet-line", "strict", "loose"})


def record_latex_for_context(record: dict[str, Any]) -> str:
    accepted = str(record.get("acceptedLatex") or "").strip()
    if accepted:
        return accepted
    return trusted_semantic_latex(str(record.get("topLatex") or ""), record.get("semantic") or {}).strip()


def should_replace_with_contextual_latex(record: dict[str, Any], contextual: dict[str, Any]) -> bool:
    current = str(record.get("acceptedLatex") or record.get("semanticBestLatex") or record.get("topLatex") or "")
    replacement = trusted_semantic_latex(current, contextual)
    return bool(replacement and replacement != current)


def record_box(record: dict[str, Any]) -> dict[str, float]:
    box = record.get("tightBbox") or record.get("bbox") or {}
    try:
        y_min = float(box["yMin"])
        y_max = float(box["yMax"])
        x_min = float(box.get("xMin", 0))
        x_max = float(box.get("xMax", 0))
    except (KeyError, TypeError, ValueError):
        return {}
    return {"xMin": x_min, "xMax": x_max, "yMin": y_min, "yMax": y_max}


def vertical_overlap_ratio(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    overlap = max(0.0, min(a["yMax"], b["yMax"]) - max(a["yMin"], b["yMin"]))
    smaller = min(max(1.0, a["yMax"] - a["yMin"]), max(1.0, b["yMax"] - b["yMin"]))
    return overlap / smaller


def unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def canonical_latex(text: str) -> str:
    out = text or ""
    replacements = {
        r"\left": "",
        r"\right": "",
        r"\,": "",
        r"\!": "",
        r"\limits": "",
        r"\prime": "'",
    }
    for old, new in replacements.items():
        out = out.replace(old, new)
    out = out.replace("X", "x")
    out = re.sub(r"\^\s*\{\s*'\s*\}", "'", out)
    out = re.sub(r"\^\s*'", "'", out)
    out = re.sub(r"\s+", "", out)
    return out


def latex_key_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in [
        r"\frac",
        r"\log",
        r"\int",
        r"\sqrt",
        "^",
        "_",
        "=",
        "+",
        "-",
        "x",
        "y",
        "d",
        "f",
        "1",
        "2",
        "3",
        "4",
        "5",
        "7",
        "8",
        "10",
        "16",
        "20",
    ]:
        if token in text:
            tokens.append(token)
    return tokens


def loose_latex_match(predicted: str, expected: str) -> dict[str, Any]:
    expected_tokens = latex_key_tokens(expected)
    if not expected_tokens:
        return {"match": False, "matchedTokens": 0, "expectedTokens": 0, "missingTokens": []}
    missing = [token for token in expected_tokens if token not in predicted]
    matched = len(expected_tokens) - len(missing)
    required = max(1, int(len(expected_tokens) * 0.7))
    match = matched >= required
    if "=" in expected:
        match = match and "=" in predicted
    return {
        "match": match,
        "matchedTokens": matched,
        "expectedTokens": len(expected_tokens),
        "missingTokens": missing,
    }


def compare_latex(predicted: str, expected: str) -> dict[str, Any]:
    predicted_key = canonical_latex(predicted)
    expected_key = canonical_latex(expected)
    loose = loose_latex_match(predicted, expected)
    strict = bool(predicted_key and predicted_key == expected_key)
    return {
        "match": strict or bool(loose["match"]),
        "strictMatch": strict,
        "looseMatch": strict or bool(loose["match"]),
        "canonicalPredicted": predicted_key,
        "canonicalExpected": expected_key,
        "loose": loose,
    }


def selected_problems(problem_names: Sequence[str], families: Sequence[str], include_all: bool) -> list[MathProblem]:
    if include_all:
        return list(PROBLEMS)
    chosen: list[MathProblem] = []
    for name in problem_names:
        chosen.append(get_problem(name))
    for family in families:
        chosen.extend(problem for problem in PROBLEMS if problem.family == family)
    if not chosen:
        chosen = [get_problem(name) for name in DEFAULT_PROBLEMS]

    unique: dict[str, MathProblem] = {}
    for problem in chosen:
        unique[problem.name] = problem
    return list(unique.values())


def layout_gap_variants(problem: MathProblem, args: argparse.Namespace) -> list[tuple[str, Optional[list[float]]]]:
    variants: list[tuple[str, Optional[list[float]]]] = [("", None)]
    for pattern in args.gap_pattern or []:
        variants.append((pattern, line_gaps_for_pattern(len(problem.lines), pattern)))
    for index, gaps in enumerate(args.line_gaps or []):
        variants.append((f"custom-{index + 1}", gaps))
    return variants


def summarize_totals(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ocr_lines = [
        line
        for record in records
        for line in record.get("ocr", {}).get("lines", [])
    ]
    ocr_misses = [line for line in ocr_lines if not line_ocr_matches(line)]
    return {
        "fixtures": len(records),
        "segmentationExact": sum(1 for record in records if record["segmentation"]["exactLineCover"]),
        "segmentationFailures": sum(1 for record in records if not record["segmentation"]["exactLineCover"]),
        "pipelineSelectionExact": sum(1 for record in records if record["pipelineSelection"]["exactLineCover"]),
        "pipelineSelectionFailures": sum(1 for record in records if not record["pipelineSelection"]["exactLineCover"]),
        "ocrLines": len(ocr_lines),
        "ocrTopStrictMatches": sum(1 for line in ocr_lines if line["topComparison"]["strictMatch"]),
        "ocrTopLooseMatches": sum(1 for line in ocr_lines if line["topComparison"]["looseMatch"]),
        "ocrSemanticStrictMatches": sum(1 for line in ocr_lines if line["semanticComparison"]["strictMatch"]),
        "ocrSemanticLooseMatches": sum(1 for line in ocr_lines if line["semanticComparison"]["looseMatch"]),
        "ocrAcceptedStrictMatches": sum(1 for line in ocr_lines if line.get("acceptedComparison", {}).get("strictMatch")),
        "ocrAcceptedLooseMatches": sum(1 for line in ocr_lines if line.get("acceptedComparison", {}).get("looseMatch")),
        "ocrAcceptedMatches": sum(1 for line in ocr_lines if line_ocr_matches(line)),
        "ocrMisses": len(ocr_misses),
    }


def build_live_summary(args: argparse.Namespace, records: Sequence[dict[str, Any]], *, status: str) -> dict[str, Any]:
    return {
        "status": status,
        "apiUrl": args.api_url,
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": list(records),
        "totals": summarize_totals(records),
    }


def write_live_checkpoint(args: argparse.Namespace, records: Sequence[dict[str, Any]]) -> None:
    if getattr(args, "no_checkpoints", False):
        return
    summary_path = getattr(args, "write_summary", "")
    if not summary_path:
        return
    summary = build_live_summary(args, records, status="running")
    summary["checkpoint"] = True
    write_summary(Path(summary_path), summary)


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    spacings = args.spacing or list(DEFAULT_SPACINGS)
    orders = args.order or ["line-order"]
    ink_styles = args.ink_style or ["normal"]
    problems = selected_problems(args.problem or [], args.family or [], args.all)
    records: list[dict[str, Any]] = []
    seed = args.seed

    for problem in problems:
        for spacing in spacings:
            for gap_pattern, line_gaps in layout_gap_variants(problem, args):
                for ink_style in ink_styles:
                    seed += 17
                    fixture = render_fixture(
                        problem,
                        spacing,
                        seed,
                        ink_style=ink_style,
                        gap_pattern=gap_pattern,
                        line_gaps=line_gaps,
                    )
                    detector = run_detector(args.api_url, fixture.png_path, args.timeout_seconds)
                    for order in orders:
                        if detector["failed"]:
                            segment = {
                                "order": order,
                                "selected": [],
                                "selectedCount": 0,
                                "expectedLines": len(problem.lines),
                            }
                            exact, failures = False, [str(detector["error"])]
                        else:
                            segment = run_js_segmenter(fixture.payload, detector["detections"], order)
                            exact, failures = validate_selected_lines(segment.get("selected") or [], len(problem.lines))

                        selected_for_ocr = segment.get("selected") or []
                        pipeline_selection = {
                            "enabled": False,
                            "exactLineCover": exact,
                            "failures": failures,
                            "selectedCount": len(selected_for_ocr),
                            "selected": selected_for_ocr,
                        }
                        alternative_records: list[dict[str, Any]] = []

                        record: dict[str, Any] = {
                            "fixture": {
                                "problem": problem.name,
                                "family": problem.family,
                                "spacing": spacing,
                                "gapPattern": gap_pattern,
                                "lineGaps": list(fixture.line_gaps),
                                "inkStyle": ink_style,
                                "order": order,
                                "expectedLines": len(problem.lines),
                                "problemLatex": problem.context_latex,
                            },
                            "paths": {
                                "png": str(fixture.png_path),
                                "json": str(fixture.json_path),
                            },
                            "detector": detector,
                            "segmentation": {
                                "exactLineCover": exact,
                                "failures": failures,
                                "selectedCount": len(segment.get("selected") or []),
                                "candidateCount": segment.get("candidateCount", 0),
                                "partitions": segment.get("partitions", {}),
                                "selected": segment.get("selected") or [],
                            },
                            "ocr": {
                                "enabled": not args.skip_comer,
                                "lines": [],
                                "candidateAlternatives": [],
                            }
                        }
                        if not args.skip_comer and not detector["failed"]:
                            if args.geometry_only_selection:
                                def checkpoint_selected_lines(lines: list[dict[str, Any]]) -> None:
                                    record["ocr"]["lines"] = list(lines)
                                    record["pipelineSelection"] = pipeline_selection
                                    write_live_checkpoint(args, [*records, record])

                                record["ocr"]["lines"] = recognize_selected_lines(
                                    args.api_url,
                                    fixture,
                                    selected_for_ocr,
                                    order=order,
                                    timeout_seconds=args.timeout_seconds,
                                    initial_raster_height=args.initial_raster_height,
                                    initial_raster_min_height=args.initial_raster_min_height,
                                    structural_timeout_seconds=args.structural_timeout_seconds,
                                    progress=args.progress,
                                    checkpoint_callback=checkpoint_selected_lines,
                                )
                            else:
                                alternative_records = recognize_candidate_alternatives(
                                    args.api_url,
                                    fixture,
                                    segment.get("candidates") or [],
                                    order=order,
                                    timeout_seconds=args.timeout_seconds,
                                    retry_candidate_ids={
                                        str(candidate.get("candidateId") or "")
                                        for candidate in selected_for_ocr
                                    },
                                    max_extra_candidates=args.max_candidate_alternatives,
                                    candidate_time_budget_seconds=args.candidate_time_budget_seconds,
                                    extra_candidate_timeout_seconds=args.extra_candidate_timeout_seconds,
                                    initial_raster_height=args.initial_raster_height,
                                    initial_raster_min_height=args.initial_raster_min_height,
                                    structural_timeout_seconds=args.structural_timeout_seconds,
                                    progress=args.progress,
                                )
                                alternative_records = apply_contextual_semantic_scores(
                                    alternative_records,
                                    problem_latex=problem.context_latex,
                                )
                                score_map = score_map_from_alternatives(alternative_records)
                                rescored = run_js_segmenter(
                                    fixture.payload,
                                    detector["detections"],
                                    order,
                                    score_by_candidate_id=score_map,
                                )
                                selected_for_ocr = rescored.get("rescoredSelected") or selected_for_ocr
                                pipeline_exact, pipeline_failures = validate_selected_lines(
                                    selected_for_ocr,
                                    len(problem.lines),
                                )
                                pipeline_selection = {
                                    "enabled": True,
                                    "exactLineCover": pipeline_exact,
                                    "failures": pipeline_failures,
                                    "selectedCount": len(selected_for_ocr),
                                    "selected": selected_for_ocr,
                                    "recognizedCandidateCount": len(alternative_records),
                                    "maxExtraCandidateAlternatives": args.max_candidate_alternatives,
                                    "candidateTimeBudgetSeconds": args.candidate_time_budget_seconds,
                                    "extraCandidateTimeoutSeconds": args.extra_candidate_timeout_seconds,
                                }
                                record["ocr"]["candidateAlternatives"] = alternative_records
                                record["ocr"]["lines"] = selected_records_from_alternatives(
                                    selected_for_ocr,
                                    alternative_records,
                                    api_url=args.api_url,
                                    fixture=fixture,
                                    order=order,
                                    timeout_seconds=args.timeout_seconds,
                                    initial_raster_height=args.initial_raster_height,
                                    structural_timeout_seconds=args.structural_timeout_seconds,
                                )
                        record["pipelineSelection"] = pipeline_selection
                        records.append(record)
                        write_live_checkpoint(args, records)

    return build_live_summary(args, records, status="complete")


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def print_human_summary(summary: dict[str, Any], summary_path: Path) -> None:
    totals = summary.get("totals", {})
    print(json.dumps({"status": summary.get("status"), "totals": totals, "summary": str(summary_path)}, indent=2))
    for record in summary.get("records", []):
        fixture = record["fixture"]
        failures = record["segmentation"]["failures"]
        layout = fixture["spacing"]
        if fixture.get("gapPattern"):
            layout = f"{layout}:{fixture['gapPattern']}"
        label = f"{fixture['problem']}/{layout}/{fixture.get('inkStyle', 'normal')}/{fixture['order']}"
        if failures:
            print(f"[segmentation] {label}: FAIL - {'; '.join(failures)}")
        else:
            print(f"[segmentation] {label}: ok")

        for line in record.get("ocr", {}).get("lines", []):
            top = "strict" if line["topComparison"]["strictMatch"] else (
                "loose" if line["topComparison"]["looseMatch"] else "miss"
            )
            semantic = "strict" if line["semanticComparison"]["strictMatch"] else (
                "loose" if line["semanticComparison"]["looseMatch"] else "miss"
            )
            accepted_comparison = line.get("acceptedComparison") or line.get("topComparison") or {}
            accepted = "strict" if accepted_comparison.get("strictMatch") else (
                "loose" if accepted_comparison.get("looseMatch") else "miss"
            )
            print(
                f"  [ocr] line {line['lineIndex'] + 1}: top={top}, semantic={semantic}, accepted={accepted}, "
                f"acceptedLatex={line.get('acceptedLatex')!r}, "
                f"topLatex={line['topLatex']!r}, semanticBest={line['semanticBestLatex']!r}"
            )
        for failure in ocr_failures_for_record(record):
            print(f"  [ocr] FAIL - {failure}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument(
        "--initial-raster-height",
        type=int,
        default=DEFAULT_INITIAL_RASTER_HEIGHT,
        help="Normalize initial OCR crops to this pixel height when they exceed --initial-raster-min-height; use 0 to disable.",
    )
    parser.add_argument(
        "--initial-raster-min-height",
        type=int,
        default=DEFAULT_INITIAL_RASTER_MIN_HEIGHT,
        help="Only pre-normalize initial OCR crops at or above this pixel height; use 0 to disable.",
    )
    parser.add_argument("--problem", action="append", default=[], help="Problem fixture name. Repeatable.")
    parser.add_argument("--family", action="append", default=[], help="Problem family. Repeatable.")
    parser.add_argument("--all", action="store_true", help="Run every fixture problem.")
    parser.add_argument("--spacing", action="append", choices=sorted(SPACING_VARIANTS), default=[])
    parser.add_argument(
        "--gap-pattern",
        action="append",
        choices=gap_pattern_names(),
        default=[],
        help="Add a named non-uniform line-gap pattern alongside the selected spacing profile.",
    )
    parser.add_argument(
        "--line-gaps",
        action="append",
        type=parse_line_gaps,
        default=[],
        help="Add an explicit comma-separated gap list. It must match each selected problem's line count minus one.",
    )
    parser.add_argument("--ink-style", action="append", choices=available_ink_styles(), default=[])
    parser.add_argument("--order", action="append", choices=ORDERS, default=[])
    parser.add_argument("--skip-comer", action="store_true", help="Only test DBNet plus JS segmentation.")
    parser.add_argument(
        "--geometry-only-selection",
        action="store_true",
        help="Recognize only the initial geometry-selected lines instead of rescoring candidate alternatives.",
    )
    parser.add_argument(
        "--max-candidate-alternatives",
        type=int,
        default=0,
        help=(
            "Maximum number of extra non-selected candidates to OCR during alternative rescoring. "
            "Geometry-selected candidates are always recognized first. Use 0 for unlimited."
        ),
    )
    parser.add_argument(
        "--candidate-time-budget-seconds",
        type=float,
        default=0.0,
        help=(
            "Optional wall-clock budget for extra candidate OCR during alternative rescoring. "
            "Geometry-selected candidates are always recognized even when this budget is exhausted. "
            "Use 0 for unlimited."
        ),
    )
    parser.add_argument(
        "--extra-candidate-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Optional per-request CoMER timeout for non-selected candidate alternatives. "
            "Selected geometry candidates still use --timeout-seconds. Use 0 to reuse --timeout-seconds."
        ),
    )
    parser.add_argument(
        "--structural-timeout-seconds",
        type=float,
        default=12.0,
        help=(
            "Longer final CoMER timeout for selected structural fraction/function rows after short "
            "timeout retries fail. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="Return success with a skipped summary when the live API is not available.",
    )
    parser.add_argument(
        "--allow-ocr-misses",
        action="store_true",
        help="Report OCR misses without failing the live matrix process.",
    )
    parser.add_argument(
        "--min-segmentation-exact-rate",
        type=parse_rate,
        default=None,
        help="Fail if exact DBNet/JS segmentation coverage falls below this 0..1 rate.",
    )
    parser.add_argument(
        "--min-pipeline-exact-rate",
        type=parse_rate,
        default=None,
        help="Fail if the final OCR-aware selected-line cover falls below this 0..1 exact rate.",
    )
    parser.add_argument(
        "--min-accepted-strict-rate",
        type=parse_rate,
        default=None,
        help="Fail if accepted LaTeX strict-match rate falls below this 0..1 rate.",
    )
    parser.add_argument(
        "--min-accepted-match-rate",
        type=parse_rate,
        default=None,
        help="Fail if accepted LaTeX match rate falls below this 0..1 rate.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print each OCR crop before it is sent to the live model.",
    )
    parser.add_argument("--seed", type=int, default=700)
    parser.add_argument(
        "--write-summary",
        default=str(LIVE_RESULTS / "live_recognition_matrix_summary.json"),
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable record-by-record summary checkpoints during long live runs.",
    )
    return parser.parse_args(argv)


def parse_rate(value: str) -> float:
    try:
        rate = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rate must be a number between 0 and 1") from exc
    if rate < 0 or rate > 1:
        raise argparse.ArgumentTypeError("rate must be between 0 and 1")
    return rate


def clamp_comer_timeout_args(args: argparse.Namespace) -> argparse.Namespace:
    args.timeout_seconds = min(MAX_COMER_TIMEOUT_SECONDS, max(0.1, float(args.timeout_seconds)))
    if args.extra_candidate_timeout_seconds:
        args.extra_candidate_timeout_seconds = min(
            MAX_COMER_TIMEOUT_SECONDS,
            max(0.1, float(args.extra_candidate_timeout_seconds)),
        )
    if args.structural_timeout_seconds:
        args.structural_timeout_seconds = min(
            MAX_COMER_TIMEOUT_SECONDS,
            max(0.1, float(args.structural_timeout_seconds)),
        )
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = clamp_comer_timeout_args(parse_args(argv))
    summary_path = Path(args.write_summary)
    available, health = api_available(args.api_url, require_comer=not args.skip_comer)
    if not available:
        summary = {
            "status": "skipped",
            "reason": "live API unavailable",
            "apiUrl": args.api_url,
            "health": health,
            "totals": {},
            "records": [],
        }
        write_summary(summary_path, summary)
        print_human_summary(summary, summary_path)
        return 0 if args.allow_unavailable else 2

    summary = run_matrix(args)
    summary["health"] = health
    write_summary(summary_path, summary)
    print_human_summary(summary, summary_path)

    failures = summary_failures(summary, require_ocr=not args.skip_comer and not args.allow_ocr_misses)
    failures.extend(threshold_gate_failures(
        summary,
        min_segmentation_exact_rate=args.min_segmentation_exact_rate,
        min_pipeline_exact_rate=args.min_pipeline_exact_rate,
        min_accepted_strict_rate=args.min_accepted_strict_rate,
        min_accepted_match_rate=args.min_accepted_match_rate,
    ))
    for failure in failures:
        print(f"[gate] FAIL - {failure}")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
