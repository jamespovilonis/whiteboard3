#!/usr/bin/env python3
"""Offline segmentation matrix for synthetic math handwriting fixtures."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    from .fixture_catalog import (
        PROBLEMS,
        SPACING_VARIANTS,
        build_board,
        family_names,
        gap_pattern_names,
        get_problem,
        line_gaps_for_pattern,
        parse_line_gaps,
    )
    from .synthetic_handwriting import available_ink_styles
except ImportError:
    from fixture_catalog import (
        PROBLEMS,
        SPACING_VARIANTS,
        build_board,
        family_names,
        gap_pattern_names,
        get_problem,
        line_gaps_for_pattern,
        parse_line_gaps,
    )
    from synthetic_handwriting import available_ink_styles


ROOT = Path(__file__).resolve().parents[1]
SEGMENTER = ROOT / "testing" / "segment_fixture_with_js.mjs"
ORDERS = ("line-order", "reverse-lines", "interleaved-lines")


def selected_problems(problem_names: Sequence[str], families: Sequence[str], include_all: bool):
    if include_all:
        return list(PROBLEMS)
    selected = []
    for name in problem_names:
        selected.append(get_problem(name))
    for family in families:
        selected.extend(problem for problem in PROBLEMS if problem.family == family)
    if not selected:
        return list(PROBLEMS)
    unique = {}
    for problem in selected:
        unique[problem.name] = problem
    return list(unique.values())


def run_segmentation(board: dict[str, Any], order: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["node", str(SEGMENTER), order],
        cwd=str(ROOT),
        input=json.dumps(board),
        text=True,
        check=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def validate_result(result: dict[str, Any], expected_count: int) -> tuple[bool, list[str]]:
    failures: list[str] = []
    selected = result.get("selected") or []
    if len(selected) != expected_count:
        failures.append(f"expected {expected_count} selected lines, got {len(selected)}")

    covered: list[int] = []
    for index, candidate in enumerate(selected):
        line_set = candidate.get("syntheticLineSets") or []
        if len(line_set) != 1:
            failures.append(f"candidate {index + 1} mixes fixture lines {line_set}")
        covered.extend(int(item) for item in line_set)

    missing = sorted(set(range(expected_count)) - set(covered))
    duplicates = sorted(line for line in set(covered) if covered.count(line) > 1)
    if missing:
        failures.append(f"missing fixture lines {missing}")
    if duplicates:
        failures.append(f"duplicated fixture lines {duplicates}")
    return not failures, failures


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    problems = selected_problems(args.problem or [], args.family or [], args.all)
    spacings = args.spacing or sorted(SPACING_VARIANTS)
    orders = args.order or list(ORDERS)
    ink_styles = args.ink_style or ["normal"]
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for problem in problems:
        for spacing in spacings:
            for gap_pattern, line_gaps in layout_gap_variants(problem, args):
                for ink_style in ink_styles:
                    for order in orders:
                        for seed in args.seed:
                            board = build_board(
                                problem.name,
                                spacing=spacing,
                                line_gaps=line_gaps,
                                seed=seed,
                                ink_style=ink_style,
                            )
                            result = run_segmentation(board.to_json(), order)
                            ok, messages = validate_result(result, len(problem.lines))
                            record = {
                                "problem": problem.name,
                                "family": problem.family,
                                "spacing": spacing,
                                "gapPattern": gap_pattern,
                                "lineGaps": line_gaps,
                                "inkStyle": ink_style,
                                "order": order,
                                "seed": seed,
                                "expectedLines": len(problem.lines),
                                "selectedCount": result.get("selectedCount"),
                                "candidateCount": result.get("candidateCount"),
                                "ok": ok,
                                "failures": messages,
                            }
                            records.append(record)
                            if not ok:
                                record["selected"] = result.get("selected") or []
                                failures.append(record)

    return {
        "total": len(records),
        "passed": len(records) - len(failures),
        "failed": len(failures),
        "failures": failures,
        "records": records if args.verbose else [],
    }


def layout_gap_variants(problem: Any, args: argparse.Namespace) -> list[tuple[Optional[str], Optional[list[float]]]]:
    variants: list[tuple[Optional[str], Optional[list[float]]]] = [(None, None)]
    for pattern in args.gap_pattern or []:
        variants.append((pattern, line_gaps_for_pattern(len(problem.lines), pattern)))
    for index, gaps in enumerate(args.line_gaps or []):
        variants.append((f"custom-{index + 1}", gaps))
    return variants


def parse_seeds(raw_values: Sequence[str]) -> list[int]:
    seeds: list[int] = []
    for raw in raw_values:
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            seeds.append(int(chunk))
    return seeds or [101, 102, 103]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", action="append", default=[], help="Problem fixture name. Repeatable.")
    parser.add_argument("--family", action="append", choices=family_names(), default=[])
    parser.add_argument("--all", action="store_true", help="Run all fixture problems.")
    parser.add_argument("--spacing", action="append", choices=sorted(SPACING_VARIANTS), default=[])
    parser.add_argument("--ink-style", action="append", choices=available_ink_styles(), default=[])
    parser.add_argument("--order", action="append", choices=ORDERS, default=[])
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
    parser.add_argument("--seed", action="append", default=[], help="Seed or comma list. Repeatable.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    args.seed = parse_seeds(args.seed)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    summary = run_matrix(parse_args(argv))
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
