#!/usr/bin/env python3
"""Export equation-solving problems from the synthetic fixture catalog as JSON."""

from __future__ import annotations

import json
import sys
from typing import Iterable

try:
    from .fixture_catalog import PROBLEMS, MathProblem
except ImportError:
    from fixture_catalog import PROBLEMS, MathProblem


EQUATION_SOLVING_FAMILIES = {
    "algebra",
    "quadratic",
    "radical",
    "symbol-context",
    "rational",
    "logarithmic",
}


def main(argv: list[str]) -> int:
    selected_names = set(argv)
    selected_problems = [
        problem
        for problem in PROBLEMS
        if problem.family in EQUATION_SOLVING_FAMILIES
        and (not selected_names or problem.name in selected_names)
    ]
    if not selected_names:
        selected_problems = dedupe_by_context_latex(selected_problems)

    problems = [to_problem_definition(problem) for problem in selected_problems]
    json.dump({"problems": problems}, sys.stdout)
    return 0


def dedupe_by_context_latex(problems: Iterable[MathProblem]) -> list[MathProblem]:
    by_latex: dict[str, MathProblem] = {}
    for problem in problems:
        current = by_latex.get(problem.context_latex)
        if current is None or should_prefer_problem(problem, current):
            by_latex[problem.context_latex] = problem
    return list(by_latex.values())


def should_prefer_problem(candidate: MathProblem, current: MathProblem) -> bool:
    if candidate.problem_latex and not current.problem_latex:
        return True
    return False


def to_problem_definition(problem: MathProblem) -> dict[str, object]:
    return {
        "id": problem.name,
        "name": problem.name,
        "kind": "equation-solving",
        "family": problem.family,
        "latex": problem.context_latex,
        "modelResponse": {
            "before": f"Solve this {family_label(problem.family)} equation.",
            "latex": first_solution_line(problem),
            "after": "Submit your work when you are ready.",
        },
        "expectedLatexLines": list(problem.lines),
        "source": "testing/fixture_catalog.py",
    }


def first_solution_line(problem: MathProblem) -> str:
    for line in problem.lines:
        if line != problem.context_latex:
            return line
    return problem.lines[0] if problem.lines else problem.context_latex


def family_label(family: str) -> str:
    return family.replace("-", " ")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
