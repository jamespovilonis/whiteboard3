#!/usr/bin/env python3
"""Render synthetic math board fixtures into testing/results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

try:
    from .fixture_catalog import (
        RESULTS_DIR,
        SPACING_VARIANTS,
        build_board,
        family_names,
        fixture_payload,
        iter_selected_problems,
        placements_for,
        slug,
    )
    from .synthetic_handwriting import save_board_png
except ImportError:
    from fixture_catalog import (
        RESULTS_DIR,
        SPACING_VARIANTS,
        build_board,
        family_names,
        fixture_payload,
        iter_selected_problems,
        placements_for,
        slug,
    )
    from synthetic_handwriting import save_board_png


def parse_line_gaps(value: Optional[str]) -> Optional[List[float]]:
    if value is None:
        return None
    chunks = [chunk.strip() for chunk in value.split(",")]
    if any(not chunk for chunk in chunks):
        raise argparse.ArgumentTypeError("--line-gaps must be a comma-separated number list")
    try:
        return [float(chunk) for chunk in chunks]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--line-gaps values must be numbers") from exc


def assert_results_child(path: Path) -> Path:
    resolved = path.resolve()
    results_root = RESULTS_DIR.resolve()
    if resolved != results_root and results_root not in resolved.parents:
        raise ValueError(f"Output path must be inside {results_root}")
    return resolved


def selected_spacings(spacing: str, all_spacings: bool) -> Sequence[str]:
    if all_spacings:
        return sorted(SPACING_VARIANTS)
    return [spacing]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", help="Problem fixture name to render")
    parser.add_argument("--family", choices=family_names(), help="Problem family to render")
    parser.add_argument("--all", action="store_true", help="Render every problem fixture")
    parser.add_argument("--spacing", choices=sorted(SPACING_VARIANTS), default="standard")
    parser.add_argument("--all-spacings", action="store_true", help="Render every spacing variant")
    parser.add_argument("--line-gaps", type=parse_line_gaps, help="Comma-separated custom gaps between lines")
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--output-dir", default=str(RESULTS_DIR), help="Directory under testing/results")
    args = parser.parse_args(argv)

    output_dir = assert_results_child(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    problems = list(
        iter_selected_problems(
            problem_name=args.problem,
            family=args.family,
            include_all=args.all or not (args.problem or args.family),
        )
    )
    spacings = selected_spacings(args.spacing, args.all_spacings)
    wrote = []
    seed = args.seed

    for problem in problems:
        for spacing in spacings:
            seed += 17
            _, _, _, gaps = placements_for(problem, spacing, args.line_gaps)
            board = build_board(problem.name, spacing=spacing, line_gaps=args.line_gaps, seed=seed)
            name = slug(problem.name, spacing)
            png_path = output_dir / f"{name}.png"
            json_path = output_dir / f"{name}.json"
            save_board_png(board, str(png_path))
            json_path.write_text(
                json.dumps(fixture_payload(problem, spacing, gaps, board), indent=2),
                encoding="utf-8",
            )
            wrote.append((png_path, json_path))

    for png_path, json_path in wrote:
        print(f"Wrote {png_path}")
        print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
