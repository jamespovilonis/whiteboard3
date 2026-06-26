#!/usr/bin/env python3
"""Catalog of standalone synthetic math board fixtures."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import cycle, islice
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .synthetic_handwriting import BoardFixture, Placement, place_handwriting_lines, save_board_png
except ImportError:
    from synthetic_handwriting import BoardFixture, Placement, place_handwriting_lines, save_board_png


TESTING_DIR = Path(__file__).resolve().parent
RESULTS_DIR = TESTING_DIR / "results"


@dataclass(frozen=True)
class MathProblem:
    name: str
    family: str
    lines: Sequence[str]
    max_line_width: int = 1120
    max_line_height: int = 96
    board_width: int = 1400
    margin_x: int = 118


PROBLEMS: Sequence[MathProblem] = [
    MathProblem(
        name="algebra_simple",
        family="algebra",
        lines=[
            r"2 x + 3 = 11",
            r"2 x = 8",
            r"x = 4",
        ],
        max_line_width=850,
        max_line_height=126,
        board_width=1200,
        margin_x=126,
    ),
    MathProblem(
        name="algebra_steps",
        family="algebra",
        lines=[
            r"2 x + 3 = 11",
            r"- 3      - 3",
            r"2 x = 8",
            r"/ 2      / 2",
            r"x = 4",
        ],
        max_line_width=850,
        max_line_height=104,
        board_width=1200,
        margin_x=126,
    ),
    MathProblem(
        name="rational_solve",
        family="rational",
        lines=[
            r"\frac { x - 1 } { x + 1 } = 5",
            r"x - 1 = 5 ( x + 1 )",
            r"x - 1 = 5 x + 5",
            r"- x       - x",
            r"- 1 = 4 x + 5",
            r"- 5       - 5",
            r"- 6 = 4 x",
            r"/ 4       / 4",
            r"x = - \frac { 3 } { 2 }",
        ],
        max_line_width=1180,
        max_line_height=86,
    ),
    MathProblem(
        name="rational_quadratic_solve",
        family="rational",
        lines=[
            r"\frac { x ^ { 2 } - 1 } { x - 1 } = 4",
            r"x + 1 = 4",
            r"- 1       - 1",
            r"x = 3",
        ],
        max_line_width=1180,
        max_line_height=100,
    ),
    MathProblem(
        name="rational_two_fraction_solve",
        family="rational",
        lines=[
            r"\frac { x } { 2 } + \frac { 1 } { 3 } = 2",
            r"\times 6       \times 6",
            r"3 x + 2 = 12",
            r"- 2       - 2",
            r"3 x = 10",
            r"/ 3       / 3",
            r"x = \frac { 10 } { 3 }",
        ],
        max_line_width=1240,
        max_line_height=104,
    ),
    MathProblem(
        name="logarithmic_solve",
        family="logarithmic",
        lines=[
            r"\log _ { 2 } ( x ) + 3 = 7",
            r"- 3       - 3",
            r"\log _ { 2 } ( x ) = 4",
            r"2 ^ { 4 } = x",
            r"x = 16",
        ],
        max_line_width=1080,
        max_line_height=94,
    ),
    MathProblem(
        name="logarithmic_shift_solve",
        family="logarithmic",
        lines=[
            r"\log _ { 3 } ( x + 1 ) = 2",
            r"3 ^ { 2 } = x + 1",
            r"9 = x + 1",
            r"- 1       - 1",
            r"8 = x",
        ],
        max_line_width=1120,
        max_line_height=96,
    ),
    MathProblem(
        name="logarithmic_product_solve",
        family="logarithmic",
        lines=[
            r"\log _ { 2 } ( x ) + \log _ { 2 } ( 4 ) = 5",
            r"\log _ { 2 } ( 4 x ) = 5",
            r"2 ^ { 5 } = 4 x",
            r"32 = 4 x",
            r"/ 4       / 4",
            r"8 = x",
        ],
        max_line_width=1260,
        max_line_height=96,
    ),
    MathProblem(
        name="derivative_evaluate",
        family="derivative",
        lines=[
            r"f ( x ) = x ^ { 3 } + 2 x ^ { 2 }",
            r"f ' ( x ) = 3 x ^ { 2 } + 4 x",
            r"f ' ( 2 ) = 3 ( 2 ) ^ { 2 } + 4 ( 2 )",
            r"f ' ( 2 ) = 20",
        ],
        max_line_width=1180,
        max_line_height=96,
    ),
    MathProblem(
        name="derivative_product_evaluate",
        family="derivative",
        lines=[
            r"f ( x ) = x ^ { 2 } ( x + 3 )",
            r"f ' ( x ) = 2 x ( x + 3 ) + x ^ { 2 }",
            r"f ' ( 2 ) = 4 ( 5 ) + 4",
            r"f ' ( 2 ) = 24",
        ],
        max_line_width=1240,
        max_line_height=100,
    ),
    MathProblem(
        name="derivative_quotient_evaluate",
        family="derivative",
        lines=[
            r"f ( x ) = \frac { x ^ { 2 } + 1 } { x }",
            r"f ' ( x ) = \frac { x ^ { 2 } - 1 } { x ^ { 2 } }",
            r"f ' ( 2 ) = \frac { 3 } { 4 }",
        ],
        max_line_width=1240,
        max_line_height=118,
    ),
    MathProblem(
        name="integral_evaluate",
        family="integral",
        lines=[
            r"\int _ { 0 } ^ { 2 } ( 3 x ^ { 2 } + 1 ) d x",
            r"= [ x ^ { 3 } + x ] _ { 0 } ^ { 2 }",
            r"= ( 8 + 2 ) - ( 0 + 0 )",
            r"= 10",
        ],
        max_line_width=1180,
        max_line_height=100,
    ),
    MathProblem(
        name="integral_power_evaluate",
        family="integral",
        lines=[
            r"\int _ { 1 } ^ { 3 } 2 x d x",
            r"= [ x ^ { 2 } ] _ { 1 } ^ { 3 }",
            r"= 9 - 1",
            r"= 8",
        ],
        max_line_width=1120,
        max_line_height=100,
    ),
    MathProblem(
        name="integral_fraction_antiderivative",
        family="integral",
        lines=[
            r"\int _ { 0 } ^ { 2 } ( x + 1 ) d x",
            r"= [ \frac { x ^ { 2 } } { 2 } + x ] _ { 0 } ^ { 2 }",
            r"= ( 2 + 2 ) - ( 0 + 0 )",
            r"= 4",
        ],
        max_line_width=1240,
        max_line_height=116,
    ),
]


SPACING_VARIANTS: Dict[str, Dict[str, object]] = {
    "standard": {"margin_y": 76, "line_gap": 54},
    "dense": {"margin_y": 62, "line_gap": 20},
    "tight-steps": {"margin_y": 58, "line_gap": 4},
    "mixed": {"margin_y": 64, "line_gaps": [76, 14, 52, 8, 38, 18, 62, 10]},
}


def slug(*parts: str) -> str:
    return "_".join(re.sub(r"[^a-zA-Z0-9]+", "-", part).strip("-").lower() for part in parts)


def problem_by_name() -> Dict[str, MathProblem]:
    return {problem.name: problem for problem in PROBLEMS}


def family_names() -> List[str]:
    return sorted({problem.family for problem in PROBLEMS})


def get_problem(problem_name: str) -> MathProblem:
    problems = problem_by_name()
    try:
        return problems[problem_name]
    except KeyError as exc:
        known = ", ".join(sorted(problems))
        raise KeyError(f"Unknown problem {problem_name!r}. Known problems: {known}") from exc


def iter_selected_problems(
    *,
    problem_name: Optional[str] = None,
    family: Optional[str] = None,
    include_all: bool = False,
) -> Iterable[MathProblem]:
    if problem_name:
        yield get_problem(problem_name)
        return
    if family:
        matches = [problem for problem in PROBLEMS if problem.family == family]
        if not matches:
            known = ", ".join(family_names())
            raise KeyError(f"Unknown family {family!r}. Known families: {known}")
        yield from matches
        return
    if include_all or not problem_name:
        yield from PROBLEMS


def validate_line_gaps(line_count: int, line_gaps: Optional[Sequence[float]]) -> Optional[List[float]]:
    if line_gaps is None:
        return None
    gaps = [float(gap) for gap in line_gaps]
    expected = max(0, line_count - 1)
    if len(gaps) != expected:
        raise ValueError(f"line_gaps must contain {expected} values for {line_count} lines")
    if any(gap < 0 for gap in gaps):
        raise ValueError("line_gaps values must be non-negative")
    return gaps


def gaps_for_spacing(
    line_count: int,
    spacing: str,
    line_gaps: Optional[Sequence[float]] = None,
) -> List[float]:
    custom_gaps = validate_line_gaps(line_count, line_gaps)
    if custom_gaps is not None:
        return custom_gaps
    if spacing not in SPACING_VARIANTS:
        known = ", ".join(sorted(SPACING_VARIANTS))
        raise KeyError(f"Unknown spacing {spacing!r}. Known spacing variants: {known}")
    profile = SPACING_VARIANTS[spacing]
    gap_count = max(0, line_count - 1)
    if "line_gaps" in profile:
        return list(islice(cycle(profile["line_gaps"]), gap_count))
    return [float(profile["line_gap"])] * gap_count


def placements_for(
    problem: MathProblem,
    spacing: str = "standard",
    line_gaps: Optional[Sequence[float]] = None,
) -> Tuple[List[Placement], int, int, List[float]]:
    if spacing not in SPACING_VARIANTS:
        known = ", ".join(sorted(SPACING_VARIANTS))
        raise KeyError(f"Unknown spacing {spacing!r}. Known spacing variants: {known}")
    profile = SPACING_VARIANTS[spacing]
    gaps = gaps_for_spacing(len(problem.lines), spacing, line_gaps)
    margin_y = int(profile["margin_y"])
    line_height = problem.max_line_height
    board_width = problem.board_width
    board_height = max(760, int(margin_y * 2 + len(problem.lines) * line_height + sum(gaps) + 80))
    offsets = [0, 42, 24, 92, 40, 100, 34, 108, 64]
    placements: List[Placement] = []
    y = float(margin_y)
    for index in range(len(problem.lines)):
        placements.append({
            "x": problem.margin_x + offsets[index % len(offsets)],
            "y": y,
        })
        if index < len(gaps):
            y += line_height + gaps[index]
    return placements, board_width, board_height, gaps


def build_board(
    problem_name: str,
    spacing: str = "standard",
    line_gaps: Optional[Sequence[float]] = None,
    seed: int = 0,
) -> BoardFixture:
    problem = get_problem(problem_name)
    placements, board_width, board_height, _ = placements_for(problem, spacing, line_gaps)
    return place_handwriting_lines(
        problem.lines,
        board_width=board_width,
        board_height=board_height,
        placements=placements,
        seed=seed,
        max_line_width=problem.max_line_width,
        max_line_height=problem.max_line_height,
    )


def fixture_payload(
    problem: MathProblem,
    spacing: str,
    line_gaps: Sequence[float],
    board: BoardFixture,
) -> Dict[str, object]:
    payload = board.to_json()
    payload["fixture"] = {
        "problem": problem.name,
        "family": problem.family,
        "spacing": spacing,
        "lineGaps": list(line_gaps),
        "expectedLatexLines": list(problem.lines),
    }
    return payload
