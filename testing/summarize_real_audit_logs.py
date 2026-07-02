#!/usr/bin/env python3
"""Summarize local VLM audit logs against the current segmenter and grader."""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_LOG_DIR = Path(
    os.environ.get("WHITEBOARD_AUDIT_LOG_DIR", "/Users/jpovj/Documents/dev/log_whiteboard_3")
).expanduser()
SEGMENTER = ROOT / "testing" / "segment_fixture_with_js.mjs"

sys.path.insert(0, str(ROOT))
from src.grading import grade_math_payload  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = [record for record in iter_records(args.audit_log_dir) if record]
    print(f"records {len(records)}")
    print_counter("logged fast status -> current logged-OCR status", (
        (record["loggedStatus"], record["loggedOcrStatus"]) for record in records
    ))
    print_counter("current logged-OCR status vs VLM status", (
        (record["loggedOcrStatus"], record["vlmStatus"]) for record in records
        if record["vlmStatus"] is not None
    ))

    uncovered = [
        record for record in records
        if record["vlmStatus"] == "correct" and record["loggedOcrStatus"] != "correct"
    ]
    print_records(
        "VLM correct but current regrade of logged OCR is not correct",
        (classify_uncovered(record) for record in uncovered)
    )

    line_mismatches = [
        record for record in records
        if record["vlmLineCount"] and record["currentSegmentCount"] is not None and
        record["currentSegmentCount"] != record["vlmLineCount"]
    ]
    print_records(
        "Current raw-stroke segmentation line-count mismatches",
        (classify_line_mismatch(record) for record in line_mismatches)
    )
    return 0


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-log-dir",
        type=Path,
        default=DEFAULT_AUDIT_LOG_DIR,
        help="Root audit log directory. Defaults to WHITEBOARD_AUDIT_LOG_DIR or the local whiteboard log path.",
    )
    return parser.parse_args(argv)


def iter_records(log_dir: Path) -> list[dict[str, Any]]:
    records = []
    for audit_dir in sorted(log_dir.glob("2026-*/*")):
        if not audit_dir.is_dir():
            continue
        input_payload = read_json(audit_dir / "input.json")
        fast_result = read_json(audit_dir / "fast_result.json")
        if not input_payload or not fast_result:
            continue
        comparison = read_json(audit_dir / "comparison.json", {})
        vlm = read_json(audit_dir / "vlm_normalized.json", {})
        current_grading = regrade_logged_ocr(input_payload, fast_result)
        segment_count = current_segment_count(input_payload, fast_result)
        records.append({
            "auditId": audit_dir.name,
            "problemLatex": input_payload.get("problemLatex"),
            "problemMetadata": input_payload.get("problemMetadata") or {},
            "loggedStatus": ((fast_result.get("grading") or {}).get("result") or {}).get("problemStatus"),
            "loggedOcrStatus": ((current_grading.get("result") or {}).get("problemStatus")),
            "vlmStatus": comparison.get("vlmProblemStatus"),
            "discrepancyTypes": [item.get("type") for item in comparison.get("discrepancies") or []],
            "fastLines": [
                line.get("latex") or line.get("bestLatex") or ""
                for line in fast_result.get("lines") or []
            ],
            "vlmLines": vlm.get("latexLines") or [],
            "foundSolutions": (current_grading.get("result") or {}).get("foundSolutions"),
            "missingSolutions": (current_grading.get("result") or {}).get("missingSolutions"),
            "currentSegmentCount": segment_count,
            "vlmLineCount": len(vlm.get("latexLines") or []),
        })
    return records


def regrade_logged_ocr(input_payload: dict[str, Any], fast_result: dict[str, Any]) -> dict[str, Any]:
    lines = []
    for index, line in enumerate(fast_result.get("lines") or []):
        candidates = []
        for candidate in line.get("candidates") or []:
            if isinstance(candidate, dict) and candidate.get("latex") is not None:
                candidates.append({"latex": candidate.get("latex")})
            elif isinstance(candidate, str):
                candidates.append({"latex": candidate})
        lines.append({
            "lineIndex": index,
            "latex": line.get("latex") or line.get("bestLatex") or "",
            "candidates": candidates,
        })
    return grade_math_payload({
        "problemLatex": input_payload.get("problemLatex"),
        "problemMetadata": input_payload.get("problemMetadata") or {},
        "lines": lines,
    })


def current_segment_count(input_payload: dict[str, Any], fast_result: dict[str, Any]) -> int | None:
    payload = {
        "strokes": input_payload.get("strokes") or [],
        "answerBox": input_payload.get("answerBox"),
        "detections": ((fast_result.get("detection") or {}).get("detections") or []),
    }
    process = subprocess.run(
        ["node", str(SEGMENTER), "line-order"],
        cwd=ROOT,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    if process.returncode != 0:
        return None
    return len((json.loads(process.stdout).get("selected") or []))


def classify_uncovered(record: dict[str, Any]) -> dict[str, Any]:
    reason = "logged-ocr-disagreement"
    if record["currentSegmentCount"] == record["vlmLineCount"]:
        reason = "stale-logged-ocr-or-ocr-candidate"
    if is_unsimplified_policy_case(record):
        reason = "unsimplified-final-policy"
    return {**record, "classification": reason}


def classify_line_mismatch(record: dict[str, Any]) -> dict[str, Any]:
    reason = "current-segmentation-disagreement"
    if record["loggedOcrStatus"] == record["vlmStatus"]:
        reason = "line-count-only-status-agrees"
    if record["vlmStatus"] in {"not_started", "incomplete"} and record["loggedOcrStatus"] == "correct":
        reason = "vlm-status-disagreement-current-correct"
    return {**record, "classification": reason}


def is_unsimplified_policy_case(record: dict[str, Any]) -> bool:
    return (
        record["loggedOcrStatus"] == "incomplete" and
        record["vlmStatus"] == "correct" and
        record.get("foundSolutions") == [] and
        bool(record.get("missingSolutions"))
    )


def print_counter(title: str, values: Any) -> None:
    print(f"\n{title}")
    for key, count in collections.Counter(values).most_common():
        print(f"  {count:>3} {key}")


def print_records(title: str, records: Any) -> None:
    records = list(records)
    print(f"\n{title}: {len(records)}")
    for record in records:
        print(
            f"- {record['auditId']} [{record['classification']}] "
            f"seg={record['currentSegmentCount']} vlmLines={record['vlmLineCount']} "
            f"loggedOcr={record['loggedOcrStatus']} vlm={record['vlmStatus']}"
        )
        print(f"  problem: {record['problemLatex']} metadata={record['problemMetadata']}")
        print(f"  fast: {record['fastLines']}")
        print(f"  vlm:  {record['vlmLines']}")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


if __name__ == "__main__":
    raise SystemExit(main())
