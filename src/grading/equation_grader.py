"""V1 symbolic grading for single-variable equation solving.

The grader has two layers:

* ``create_answer_manifest`` turns a user-provided equation into a small answer
  manifest backed by SymPy ``solveset``.
* ``grade_equation_work`` classifies OCR line candidates against that manifest.

V1 intentionally focuses on real-valued equations solved for one variable. It
handles finite, empty, all-real, and integer-parameterized infinite solution
families directly, and returns an ``unsupported`` manifest for symbolic sets
that need a richer future grader.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import math
import re
import signal
import threading
from typing import Any, Iterable, Optional, Sequence

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from .answer_grader import (
    AnswerFinality,
    candidate_rank,
    final_answer,
    invalid_format,
    not_answer,
    unsimplified_answer,
)


DEFAULT_TOLERANCE = 0.005
MAX_OCR_CANDIDATES = 5
SYMPY_GRADER_BUDGET_SECONDS = 0.5
EVALUATE_PROBLEM_TYPES = {"evaluate-expression", "expression-evaluation", "numeric-expression"}
SIMPLIFY_PROBLEM_TYPES = {"simplify-expression", "expression-simplification", "simplifying-expression"}
EQUATION_PROBLEM_TYPE = "equation-solving"
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

LATEX_COMMAND_REPLACEMENTS = {
    r"\cdot": "*",
    r"\times": "*",
    r"\div": "/",
    r"\pi": "pi",
    r"\mathrm{e}": "e",
    "\u2212": "-",
    "\u00d7": "*",
    "\u00f7": "/",
}

LATEX_VARIABLE_COMMANDS = {
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma",
    r"\delta": "delta",
    r"\epsilon": "epsilon",
    r"\varepsilon": "varepsilon",
    r"\theta": "theta",
    r"\lambda": "lambda",
    r"\mu": "mu",
    r"\rho": "rho",
    r"\sigma": "sigma",
    r"\tau": "tau",
    r"\phi": "phi",
    r"\varphi": "varphi",
    r"\omega": "omega",
}


def sympy_log10(arg: sympy.Expr, base: Optional[sympy.Expr] = None, **_kwargs: Any) -> sympy.Expr:
    if base is None:
        return sympy.log(arg, 10)
    return sympy.log(arg, base)


KNOWN_FUNCTIONS = {
    "sqrt": sympy.sqrt,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "sec": sympy.sec,
    "csc": sympy.csc,
    "cot": sympy.cot,
    "asin": sympy.asin,
    "acos": sympy.acos,
    "atan": sympy.atan,
    "arcsin": sympy.asin,
    "arccos": sympy.acos,
    "arctan": sympy.atan,
    "log": sympy_log10,
    "ln": sympy.log,
    "exp": sympy.exp,
    "abs": sympy.Abs,
    "Abs": sympy.Abs,
}
KNOWN_CONSTANTS = {"pi": sympy.pi, "e": sympy.E}

LATEX_FUNCTION_REPLACEMENTS = {
    r"\sin": "sin",
    r"\cos": "cos",
    r"\tan": "tan",
    r"\sec": "sec",
    r"\csc": "csc",
    r"\cot": "cot",
    r"\arcsin": "arcsin",
    r"\arccos": "arccos",
    r"\arctan": "arctan",
    r"\log": "log",
    r"\ln": "ln",
    r"\exp": "exp",
}

NO_SOLUTION_STRINGS = {
    "nosolution",
    "norealsolution",
    "norealsolutions",
    "none",
    "emptyset",
    "varnothing",
    "\u2205",
}
INFINITE_SOLUTION_STRINGS = {
    "allrealnumbers",
    "allreals",
    "allreal",
    "allx",
    "infinitelymanysolutions",
    "infinite",
    "r",
    "reals",
    "realnumbers",
}


class GradingParseFailure(ValueError):
    """Raised when a math string is outside the V1 parser's support."""


class SympyGraderBudgetExceeded(TimeoutError):
    """Raised when a symbolic grading check exceeds the per-operation budget."""


@contextlib.contextmanager
def sympy_grader_budget(seconds: float = SYMPY_GRADER_BUDGET_SECONDS):
    """Context manager that raises SympyGraderBudgetExceeded after *seconds*.

    On the main thread this uses SIGALRM / setitimer. In any other thread
    it is a no-op (signal-based timers only work on the main thread).
    """
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        yield
        return

    def raise_timeout(_signum, _frame):
        raise SympyGraderBudgetExceeded("symbolic grading check exceeded budget")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _simplify_with_budget(expr: sympy.Expr, budget: float = SYMPY_GRADER_BUDGET_SECONDS) -> Optional[sympy.Expr]:
    """Return sympy.simplify(expr) or None if timeout / exception."""
    try:
        with sympy_grader_budget(budget):
            return sympy.simplify(expr)
    except (Exception, SympyGraderBudgetExceeded):
        return None


def _solveset_with_budget(
    residual: sympy.Expr,
    symbol: sympy.Symbol,
    budget: float = SYMPY_GRADER_BUDGET_SECONDS,
) -> Optional[sympy.Set]:
    """Return sympy.solveset(...) or None if timeout / exception."""
    try:
        with sympy_grader_budget(budget):
            return sympy.solveset(residual, symbol, domain=sympy.S.Reals)
    except (Exception, SympyGraderBudgetExceeded):
        return None


@dataclass(frozen=True)
class ParsedMath:
    kind: str
    left: sympy.Expr
    right: Optional[sympy.Expr]
    standardized: str

    @property
    def residual(self) -> sympy.Expr:
        if self.kind != "equation" or self.right is None:
            raise GradingParseFailure("not an equation")
        return self.left - self.right


@dataclass(frozen=True)
class SolutionMatch:
    matched_indices: tuple[int, ...]
    coverage: str
    accepted_special: Optional[str] = None
    finality: AnswerFinality = not_answer()


def create_answer_manifest(
    problem_raw: str,
    variable: Optional[str] = None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Create an answer manifest for a user-provided equation."""

    raw = str(problem_raw or "").strip()
    manifest: dict[str, Any] = {
        "problem_raw": raw,
        "responseKind": "solution_set",
        "finalityPolicy": "pragmatic",
        "variable": variable,
        "cardinality": "unsupported",
        "exact_set": [],
        "decimal_set": [],
        "tolerance": float(tolerance),
        "acceptable_strings": [],
    }
    if not raw:
        return {**manifest, "error": "empty problem"}

    try:
        parsed = parse_math(raw)
    except GradingParseFailure as exc:
        return {**manifest, "error": str(exc)}

    if parsed.kind != "equation":
        return {
            **manifest,
            "problem_standardized": parsed.standardized,
            "error": "problem must be an equation",
        }

    symbol = resolve_variable(
        parsed.residual,
        variable,
        fallback_symbols=parsed.left.free_symbols | (parsed.right.free_symbols if parsed.right is not None else set()),
    )
    if symbol is None:
        return {
            **manifest,
            "problem_standardized": parsed.standardized,
            "error": "could not infer a single solve variable",
        }

    manifest["variable"] = symbol.name
    manifest["problem_standardized"] = parsed.standardized

    solution_set = _solveset_with_budget(parsed.residual, symbol)
    if solution_set is None:
        return {**manifest, "error": "solveset failed or exceeded budget"}

    if solution_set == sympy.S.EmptySet:
        return {
            **manifest,
            "cardinality": "none",
            "acceptable_strings": sorted(NO_SOLUTION_STRINGS),
        }

    if solution_set == sympy.S.Reals:
        return {
            **manifest,
            "cardinality": "infinite",
            "acceptable_strings": sorted(INFINITE_SOLUTION_STRINGS | {
                f"{symbol.name}inR",
                f"{symbol.name}inmathbbR",
            }),
        }

    if isinstance(solution_set, sympy.FiniteSet):
        values = sorted(
            list(solution_set),
            key=lambda item: (safe_float(item) is None, safe_float(item) or 0.0, sympy.sstr(item)),
        )
        exact_set = [sympy.sstr(sympy.simplify(value)) for value in values]
        decimal_set = [
            round(float_value, 3)
            for float_value in (safe_float(value) for value in values)
            if float_value is not None
        ]
        return {
            **manifest,
            "cardinality": "finite",
            "exact_set": exact_set,
            "decimal_set": decimal_set,
            "acceptable_strings": acceptable_strings_for_finite(symbol.name, values),
        }

    if isinstance(solution_set, sympy.Set) and solution_set.is_FiniteSet is False:
        return {
            **manifest,
            "cardinality": "infinite_family",
            "solution_set_repr": sympy.sstr(solution_set),
            "acceptable_strings": [],
        }

    return {
        **manifest,
        "solution_set_repr": sympy.sstr(solution_set),
        "error": "solution set is not finite, empty, or all real numbers",
    }


def grade_equation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Grade a JSON-style payload from the gateway/client boundary."""

    problem_raw = payload_problem_raw(payload)
    manifest = payload.get("manifest")
    manifest_source = "payload"
    if not manifest_is_usable_for_problem_type(
        manifest,
        problem_type=EQUATION_PROBLEM_TYPE,
        problem_raw=problem_raw,
    ):
        variable = payload_solve_variable(payload)
        manifest = create_answer_manifest(str(problem_raw), variable=variable)
        manifest_source = "generated"

    lines = payload.get("lines") or payload.get("studentLines") or []
    result = grade_equation_work(manifest, lines)
    return annotate_grading_problem_metadata(
        result,
        resolved_problem_type=EQUATION_PROBLEM_TYPE,
        manifest_source=manifest_source,
        manifest=manifest,
    )


def grade_math_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Grade a generic math-work payload.

    V1 supports equation solving, numeric expression evaluation, and symbolic
    expression simplification. Numeric expression prompts without an equals
    sign are treated as evaluate mode; variable expression prompts without an
    equals sign are treated as simplify mode.
    """

    problem_type = resolve_math_problem_type(payload)
    if problem_type == "evaluate-expression":
        return grade_expression_payload({**payload, "problemType": problem_type})
    if problem_type == "simplify-expression":
        return grade_simplification_payload({**payload, "problemType": problem_type})

    equation_payload = {
        **payload,
        "problemType": EQUATION_PROBLEM_TYPE,
    }
    return grade_equation_payload(equation_payload)


def payload_problem_raw(payload: dict[str, Any]) -> str:
    return str(
        payload.get("problem_raw") or
        payload.get("problemRaw") or
        payload.get("problemLatex") or
        payload.get("problem") or
        ""
    )


def payload_solve_variable(payload: dict[str, Any]) -> Any:
    variable = payload.get("variable")
    if variable is None and isinstance(payload.get("problemMetadata"), dict):
        variable = payload["problemMetadata"].get("solveVariable")
    return variable


def requested_problem_type(payload: dict[str, Any]) -> str:
    problem_metadata = payload.get("problemMetadata") if isinstance(payload.get("problemMetadata"), dict) else {}
    return str(
        payload.get("problemType") or
        problem_metadata.get("problemType") or
        problem_metadata.get("kind") or
        EQUATION_PROBLEM_TYPE
    ).strip()


def resolve_math_problem_type(payload: dict[str, Any]) -> str:
    """Resolve the authoritative grader mode for a boundary payload."""

    requested = requested_problem_type(payload)
    if requested in EVALUATE_PROBLEM_TYPES:
        return "evaluate-expression"
    if requested in SIMPLIFY_PROBLEM_TYPES:
        return "simplify-expression"

    problem_raw = payload_problem_raw(payload).strip()
    if problem_raw and "=" not in problem_raw:
        expression_manifest = create_expression_manifest(problem_raw)
        if not expression_manifest.get("error"):
            return "evaluate-expression"
        simplification_manifest = create_simplification_manifest(problem_raw)
        if not simplification_manifest.get("error"):
            return "simplify-expression"

    return EQUATION_PROBLEM_TYPE


def expected_manifest_response_kind(problem_type: str) -> str:
    if problem_type in EVALUATE_PROBLEM_TYPES:
        return "numeric_value"
    if problem_type in SIMPLIFY_PROBLEM_TYPES:
        return "simplified_expression"
    return "solution_set"


def manifest_response_kind(manifest: Any) -> str:
    if not isinstance(manifest, dict):
        return ""
    return str(manifest.get("responseKind") or manifest.get("response_kind") or "")


def manifest_is_usable_for_problem_type(
    manifest: Any,
    *,
    problem_type: str,
    problem_raw: str,
) -> bool:
    if not isinstance(manifest, dict):
        return False
    if manifest_response_kind(manifest) != expected_manifest_response_kind(problem_type):
        return False
    if str(manifest.get("problem_raw") or "").strip() != str(problem_raw or "").strip():
        return False
    if manifest.get("error"):
        return False
    return True


def annotate_grading_problem_metadata(
    result: dict[str, Any],
    *,
    resolved_problem_type: str,
    manifest_source: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    problem = result.get("problem")
    if isinstance(problem, dict):
        problem["resolvedProblemType"] = resolved_problem_type
        problem["manifestSource"] = manifest_source
        problem["manifestResponseKind"] = manifest_response_kind(manifest)
        problem["manifestError"] = manifest.get("error") if isinstance(manifest, dict) else None
    return result


def create_expression_manifest(
    problem_raw: str,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Create an answer manifest for a numeric-only expression prompt."""

    raw = str(problem_raw or "").strip()
    manifest: dict[str, Any] = {
        "problem_raw": raw,
        "responseKind": "numeric_value",
        "finalityPolicy": "pragmatic",
        "variable": None,
        "cardinality": "finite",
        "exact_set": [],
        "decimal_set": [],
        "tolerance": float(tolerance),
        "acceptable_strings": [],
    }
    if not raw:
        return {**manifest, "error": "empty problem"}

    try:
        parsed = parse_math(raw)
    except GradingParseFailure as exc:
        return {**manifest, "error": str(exc)}

    if parsed.kind != "expression":
        return {
            **manifest,
            "problem_standardized": parsed.standardized,
            "error": "problem must be a numeric expression",
        }
    if parsed.left.free_symbols:
        return {
            **manifest,
            "problem_standardized": parsed.standardized,
            "error": "expression prompts must be numeric only",
        }

    exact = _simplify_with_budget(parsed.left)
    if exact is None:
        exact = parsed.left
    exact = sympy.simplify(exact)
    exact_set = [expression_answer_string(exact)]
    decimal = safe_float(exact)
    return {
        **manifest,
        "problem_standardized": parsed.standardized,
        "exact_set": exact_set,
        "decimal_set": [round(decimal, 3)] if decimal is not None else [],
        "acceptable_strings": [compact_answer_text(exact_set[0])],
    }


def grade_expression_payload(payload: dict[str, Any]) -> dict[str, Any]:
    problem_raw = payload_problem_raw(payload)
    manifest = payload.get("manifest")
    manifest_source = "payload"
    if not manifest_is_usable_for_problem_type(
        manifest,
        problem_type="evaluate-expression",
        problem_raw=problem_raw,
    ):
        manifest = create_expression_manifest(str(problem_raw))
        manifest_source = "generated"

    lines = payload.get("lines") or payload.get("studentLines") or []
    result = grade_expression_work(manifest, lines)
    return annotate_grading_problem_metadata(
        result,
        resolved_problem_type="evaluate-expression",
        manifest_source=manifest_source,
        manifest=manifest,
    )


def create_simplification_manifest(
    problem_raw: str,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Create an answer manifest for a symbolic expression simplification prompt."""

    raw = str(problem_raw or "").strip()
    manifest: dict[str, Any] = {
        "problem_raw": raw,
        "responseKind": "simplified_expression",
        "finalityPolicy": "pragmatic",
        "variable": None,
        "variables": [],
        "cardinality": "finite",
        "exact_set": [],
        "decimal_set": [],
        "tolerance": float(tolerance),
        "acceptable_strings": [],
    }
    if not raw:
        return {**manifest, "error": "empty problem"}

    try:
        parsed = parse_math(raw)
    except GradingParseFailure as exc:
        return {**manifest, "error": str(exc)}

    if parsed.kind != "expression":
        return {
            **manifest,
            "problem_standardized": parsed.standardized,
            "error": "problem must be an expression",
        }
    if not parsed.left.free_symbols:
        return {
            **manifest,
            "problem_standardized": parsed.standardized,
            "error": "simplification prompts must contain at least one variable",
        }

    exact = _simplify_with_budget(parsed.left)
    if exact is None:
        return {
            **manifest,
            "problem_standardized": parsed.standardized,
            "variables": sorted(symbol.name for symbol in parsed.left.free_symbols),
            "error": "simplify failed or exceeded budget",
        }
    exact_set = [expression_answer_string(exact)]
    decimal = safe_float(exact)
    return {
        **manifest,
        "problem_standardized": parsed.standardized,
        "variables": sorted(symbol.name for symbol in parsed.left.free_symbols),
        "exact_set": exact_set,
        "decimal_set": [round(decimal, 3)] if decimal is not None else [],
        "acceptable_strings": [compact_answer_text(exact_set[0])],
    }


def grade_simplification_payload(payload: dict[str, Any]) -> dict[str, Any]:
    problem_raw = payload_problem_raw(payload)
    manifest = payload.get("manifest")
    manifest_source = "payload"
    if not manifest_is_usable_for_problem_type(
        manifest,
        problem_type="simplify-expression",
        problem_raw=problem_raw,
    ):
        manifest = create_simplification_manifest(str(problem_raw))
        manifest_source = "generated"

    lines = payload.get("lines") or payload.get("studentLines") or []
    result = grade_simplification_work(manifest, lines)
    return annotate_grading_problem_metadata(
        result,
        resolved_problem_type="simplify-expression",
        manifest_source=manifest_source,
        manifest=manifest,
    )


def grade_equation_work(
    manifest: dict[str, Any],
    lines: Sequence[dict[str, Any] | str],
) -> dict[str, Any]:
    """Classify OCR line candidates and the whole problem against a manifest."""

    normalized_lines = list(lines or [])
    variable = str(manifest.get("variable") or "x")
    exact_values = manifest_exact_values(manifest)
    found_indices: set[int] = set()
    steps: list[dict[str, Any]] = []
    reference_latex = str(manifest.get("problem_raw") or "")
    saw_valid = False
    first_invalid_index: Optional[int] = None

    for fallback_index, line in enumerate(normalized_lines):
        line_index = line.get("lineIndex", fallback_index) if isinstance(line, dict) else fallback_index
        candidates = line_candidate_latex(line)
        selected = classify_line_candidates(
            candidates,
            manifest,
            exact_values,
            variable,
            reference_latex,
        )
        selected = public_line_selection(selected, manifest)

        if selected["classification"] == "valid_step":
            saw_valid = True
            matched_indices = selected.get("_matched_indices", ())
            if selected.get("countsTowardCompletion", True) is not False:
                found_indices.update(matched_indices)
            if selected.get("studentLatex"):
                reference_latex = selected["studentLatex"]
        elif selected["classification"] == "invalid_step" and first_invalid_index is None:
            first_invalid_index = int(line_index)

        steps.append({
            "lineIndex": line_index,
            **strip_private_selection_fields(selected),
        })

    cardinality = manifest.get("cardinality")
    complete = problem_complete(manifest, exact_values, found_indices)
    if complete:
        status = "correct"
        breakdown_line_index = None
    elif first_invalid_index is not None:
        status = "incorrect"
        breakdown_line_index = first_invalid_index
    elif saw_valid:
        status = "incomplete"
        breakdown_line_index = None
    else:
        status = "not_started"
        breakdown_line_index = None

    if cardinality == "unsupported" and status == "correct":
        status = "incomplete"

    found_solutions = [
        str(manifest.get("exact_set", [])[index])
        for index in sorted(found_indices)
        if index >= 0 and index < len(manifest.get("exact_set", []))
    ]
    missing_solutions = [
        str(value)
        for index, value in enumerate(manifest.get("exact_set", []))
        if index not in found_indices
    ] if cardinality == "finite" else []

    return {
        "problem": {
            "latex": manifest.get("problem_raw", ""),
            "standardized": manifest.get("problem_standardized", ""),
            "solveVariable": manifest.get("variable"),
            "cardinality": cardinality,
            "solutionSet": list(manifest.get("exact_set") or []),
            "decimalSet": list(manifest.get("decimal_set") or []),
            "tolerance": manifest.get("tolerance", DEFAULT_TOLERANCE),
            "manifest": manifest,
        },
        "steps": steps,
        "result": {
            "problemStatus": status,
            "breakdownLineIndex": breakdown_line_index,
            "foundSolutions": found_solutions,
            "missingSolutions": missing_solutions,
        },
    }


def grade_expression_work(
    manifest: dict[str, Any],
    lines: Sequence[dict[str, Any] | str],
) -> dict[str, Any]:
    """Grade numeric expression-evaluation answers against a manifest.

    Evaluate-mode work is graded as one ordered expression sequence. A student
    may write a horizontal chain, vertical chain, leading-equals continuation
    lines, or a mixture of those forms.
    """

    normalized_lines = list(lines or [])
    exact_values = manifest_exact_values(manifest)
    selected_by_line: list[dict[str, Any]] = []
    expression_elements: list[dict[str, Any]] = []

    for fallback_index, line in enumerate(normalized_lines):
        line_index = line.get("lineIndex", fallback_index) if isinstance(line, dict) else fallback_index
        candidates = line_candidate_latex(line)
        selected = classify_expression_candidates(candidates, manifest, exact_values)
        for raw_part, value in selected.get("_expression_elements", ()):
            expression_elements.append({
                "lineIndex": int(line_index),
                "raw": raw_part,
                "value": value,
            })
        selected_by_line.append({
            "lineIndex": line_index,
            **selected,
        })

    first_invalid_index: Optional[int] = next(
        (
            int(step["lineIndex"])
            for step in selected_by_line
            if step.get("classification") == "invalid_step"
        ),
        None,
    )
    saw_valid = any(step.get("classification") == "valid_step" for step in selected_by_line)
    found_indices: set[int] = set()

    if first_invalid_index is None and expression_elements:
        clear_expression_completion_credit(selected_by_line)
        first_invalid_index = first_broken_expression_link_index(expression_elements, manifest)
        if first_invalid_index is not None:
            mark_expression_line_invalid(
                selected_by_line,
                first_invalid_index,
                "adjacent expressions are not equivalent",
            )

    if first_invalid_index is None and expression_elements and exact_values:
        final_element = expression_elements[-1]
        matched = tuple(
            index
            for index, exact in enumerate(exact_values)
            if solution_values_equivalent(final_element["value"], exact, manifest)
        )
        if matched:
            finality = expression_value_finality(final_element["raw"], final_element["value"])
            mark_expression_line_final(
                selected_by_line,
                int(final_element["lineIndex"]),
                matched,
                finality,
            )
            saw_valid = True
            if finality.counts_toward_completion:
                found_indices.update(matched)
        else:
            first_invalid_index = int(final_element["lineIndex"])
            mark_expression_line_invalid(
                selected_by_line,
                first_invalid_index,
                "final value does not match the prompt",
            )

    manifest_failed = bool(manifest.get("error"))
    if manifest_failed and first_invalid_index is not None and not found_indices:
        first_invalid_index = None

    if found_indices and len(found_indices) == len(exact_values):
        status = "correct"
        breakdown_line_index = None
    elif first_invalid_index is not None:
        status = "incorrect"
        breakdown_line_index = first_invalid_index
    elif saw_valid:
        status = "incomplete"
        breakdown_line_index = None
    else:
        status = "not_started"
        breakdown_line_index = None

    exact_set = list(manifest.get("exact_set") or [])
    steps = [
        {
            **strip_private_selection_fields(public_line_selection(step, manifest)),
        }
        for step in selected_by_line
    ]
    return {
        "problem": {
            "latex": manifest.get("problem_raw", ""),
            "standardized": manifest.get("problem_standardized", ""),
            "solveVariable": None,
            "cardinality": manifest.get("cardinality"),
            "solutionSet": exact_set,
            "decimalSet": list(manifest.get("decimal_set") or []),
            "tolerance": manifest.get("tolerance", DEFAULT_TOLERANCE),
            "manifest": manifest,
        },
        "steps": steps,
        "result": {
            "problemStatus": status,
            "breakdownLineIndex": breakdown_line_index,
            "foundSolutions": [
                str(exact_set[index])
                for index in sorted(found_indices)
                if index >= 0 and index < len(exact_set)
            ],
            "missingSolutions": [
                str(value)
                for index, value in enumerate(exact_set)
                if index not in found_indices
            ],
        },
    }


def grade_simplification_work(
    manifest: dict[str, Any],
    lines: Sequence[dict[str, Any] | str],
) -> dict[str, Any]:
    """Grade symbolic expression-simplification work against a manifest."""

    normalized_lines = list(lines or [])
    exact_values = manifest_exact_values(manifest)
    selected_by_line: list[dict[str, Any]] = []
    expression_elements: list[dict[str, Any]] = []

    for fallback_index, line in enumerate(normalized_lines):
        line_index = line.get("lineIndex", fallback_index) if isinstance(line, dict) else fallback_index
        candidates = line_candidate_latex(line)
        selected = classify_simplification_candidates(candidates, manifest, exact_values)
        for raw_part, value in selected.get("_expression_elements", ()):
            expression_elements.append({
                "lineIndex": int(line_index),
                "raw": raw_part,
                "value": value,
            })
        selected_by_line.append({
            "lineIndex": line_index,
            **selected,
        })

    first_invalid_index: Optional[int] = next(
        (
            int(step["lineIndex"])
            for step in selected_by_line
            if step.get("classification") == "invalid_step"
        ),
        None,
    )
    saw_valid = any(step.get("classification") == "valid_step" for step in selected_by_line)
    found_indices: set[int] = set()

    if first_invalid_index is None and expression_elements:
        clear_expression_completion_credit(selected_by_line)
        first_invalid_index = first_broken_expression_link_index(expression_elements, manifest)
        if first_invalid_index is not None:
            mark_expression_line_invalid(
                selected_by_line,
                first_invalid_index,
                "adjacent expressions are not equivalent",
            )

    if first_invalid_index is None and expression_elements and exact_values:
        final_element = expression_elements[-1]
        matched = tuple(
            index
            for index, exact in enumerate(exact_values)
            if solution_values_equivalent(final_element["value"], exact, manifest)
        )
        if matched:
            target = exact_values[matched[0]]
            finality = simplification_value_finality(
                final_element["raw"],
                final_element["value"],
                target,
                manifest,
            )
            mark_expression_line_final(
                selected_by_line,
                int(final_element["lineIndex"]),
                matched,
                finality,
            )
            saw_valid = True
            if finality.counts_toward_completion:
                found_indices.update(matched)
        else:
            first_invalid_index = int(final_element["lineIndex"])
            mark_expression_line_invalid(
                selected_by_line,
                first_invalid_index,
                "final expression does not match the simplified target",
            )

    manifest_failed = bool(manifest.get("error"))
    if manifest_failed and first_invalid_index is not None and not found_indices:
        first_invalid_index = None

    if found_indices and len(found_indices) == len(exact_values):
        status = "correct"
        breakdown_line_index = None
    elif first_invalid_index is not None:
        status = "incorrect"
        breakdown_line_index = first_invalid_index
    elif saw_valid:
        status = "incomplete"
        breakdown_line_index = None
    else:
        status = "not_started"
        breakdown_line_index = None

    exact_set = list(manifest.get("exact_set") or [])
    steps = [
        {
            **strip_private_selection_fields(public_line_selection(step, manifest)),
        }
        for step in selected_by_line
    ]
    return {
        "problem": {
            "latex": manifest.get("problem_raw", ""),
            "standardized": manifest.get("problem_standardized", ""),
            "solveVariable": None,
            "variables": list(manifest.get("variables") or []),
            "cardinality": manifest.get("cardinality"),
            "solutionSet": exact_set,
            "decimalSet": list(manifest.get("decimal_set") or []),
            "tolerance": manifest.get("tolerance", DEFAULT_TOLERANCE),
            "manifest": manifest,
        },
        "steps": steps,
        "result": {
            "problemStatus": status,
            "breakdownLineIndex": breakdown_line_index,
            "foundSolutions": [
                str(exact_set[index])
                for index in sorted(found_indices)
                if index >= 0 and index < len(exact_set)
            ],
            "missingSolutions": [
                str(value)
                for index, value in enumerate(exact_set)
                if index not in found_indices
            ],
        },
    }


def grade_candidate_group(
    manifest: dict[str, Any],
    group: dict[str, Any] | str,
    *,
    previous_latex: Sequence[str] = (),
    problem_latex: str = "",
) -> dict[str, Any]:
    """Grade one OCR candidate group and choose the semantically supported item.

    Caches equivalence results so each candidate's relation to the reference
    and solution set is computed only once, then reused for the group-level
    verdict and per-candidate verdicts.
    """

    if manifest.get("responseKind") == "numeric_value":
        return grade_expression_candidate_group(manifest, group)
    if manifest.get("responseKind") == "simplified_expression":
        return grade_simplification_candidate_group(manifest, group)

    variable = str(manifest.get("variable") or "x")
    exact_values = manifest_exact_values(manifest)
    candidates = line_candidate_latex(group)
    reference_latex = last_reference_latex(previous_latex, problem_latex, manifest)

    # Compute all candidate verdicts once, caching equivalence results.
    candidate_verdicts_raw: list[dict[str, Any]] = []
    for index, latex in enumerate(candidates[:MAX_OCR_CANDIDATES]):
        verdict = classify_line_candidates(
            [latex],
            manifest,
            exact_values,
            variable,
            reference_latex,
        )
        verdict = public_line_selection(verdict, manifest)
        verdict["_original_candidate_index"] = index
        candidate_verdicts_raw.append(verdict)

    best, best_original_index = _pick_best_candidate_verdict(candidate_verdicts_raw)
    if best is not None:
        selected = best
        if selected.get("selectedCandidateIndex") is not None:
            selected["selectedCandidateIndex"] = best_original_index
    elif candidates:
        selected = classify_line_candidates(candidates, manifest, exact_values, variable, reference_latex)
    else:
        selected = selected_line("", "other", None, "none", ())
    if selected.get("studentLatex"):
        selected = public_line_selection(selected, manifest)

    candidate_verdicts = [
        {
            "candidateIndex": index,
            "latex": verdict.get("studentLatex", ""),
            **strip_private_selection_fields(verdict),
        }
        for index, verdict in enumerate(candidate_verdicts_raw)
    ]

    return {
        **strip_private_selection_fields(selected),
        "candidateVerdicts": candidate_verdicts,
    }


def grade_expression_candidate_group(
    manifest: dict[str, Any],
    group: dict[str, Any] | str,
) -> dict[str, Any]:
    exact_values = manifest_exact_values(manifest)
    candidates = line_candidate_latex(group)

    candidate_verdicts_raw: list[dict[str, Any]] = []
    for index, latex in enumerate(candidates[:MAX_OCR_CANDIDATES]):
        verdict = classify_expression_candidates([latex], manifest, exact_values)
        verdict = public_line_selection(verdict, manifest)
        verdict["_original_candidate_index"] = index
        candidate_verdicts_raw.append(verdict)

    best, best_original_index = _pick_best_candidate_verdict(candidate_verdicts_raw)
    if best is not None:
        selected = best
        if selected.get("selectedCandidateIndex") is not None:
            selected["selectedCandidateIndex"] = best_original_index
    elif candidates:
        selected = classify_expression_candidates(candidates, manifest, exact_values)
    else:
        selected = selected_line("", "other", None, "none", ())

    candidate_verdicts = [
        {
            "candidateIndex": index,
            "latex": verdict.get("studentLatex", ""),
            **strip_private_selection_fields(verdict),
        }
        for index, verdict in enumerate(candidate_verdicts_raw)
    ]

    return {
        **strip_private_selection_fields(selected),
        "candidateVerdicts": candidate_verdicts,
    }


def grade_simplification_candidate_group(
    manifest: dict[str, Any],
    group: dict[str, Any] | str,
) -> dict[str, Any]:
    exact_values = manifest_exact_values(manifest)
    candidates = line_candidate_latex(group)

    candidate_verdicts_raw: list[dict[str, Any]] = []
    for index, latex in enumerate(candidates[:MAX_OCR_CANDIDATES]):
        verdict = classify_simplification_candidates([latex], manifest, exact_values)
        verdict = public_line_selection(verdict, manifest)
        verdict["_original_candidate_index"] = index
        candidate_verdicts_raw.append(verdict)

    best, best_original_index = _pick_best_candidate_verdict(candidate_verdicts_raw)
    if best is not None:
        selected = best
        if selected.get("selectedCandidateIndex") is not None:
            selected["selectedCandidateIndex"] = best_original_index
    elif candidates:
        selected = classify_simplification_candidates(candidates, manifest, exact_values)
    else:
        selected = selected_line("", "other", None, "none", ())

    candidate_verdicts = [
        {
            "candidateIndex": index,
            "latex": verdict.get("studentLatex", ""),
            **strip_private_selection_fields(verdict),
        }
        for index, verdict in enumerate(candidate_verdicts_raw)
    ]

    return {
        **strip_private_selection_fields(selected),
        "candidateVerdicts": candidate_verdicts,
    }


def _pick_best_candidate_verdict(
    verdicts: Sequence[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], int]:
    """Select the most supported verdict from pre-computed per-candidate results.

    Priority: best classification (valid_step > invalid_step > other), then
    best solution_coverage when both are valid.

    Returns (best_verdict, original_candidate_index).
    """
    best: Optional[dict[str, Any]] = None
    best_original_index = 0
    best_rank = (999, 999)

    for verdict in verdicts:
        rank = candidate_rank(verdict)
        if rank < best_rank:
            best_rank = rank
            best = verdict
            best_original_index = verdict.get("_original_candidate_index", 0)

    return best, best_original_index


def classify_line_candidates(
    candidates: Sequence[str],
    manifest: dict[str, Any],
    exact_values: Sequence[sympy.Expr],
    variable: str,
    reference_latex: str,
) -> dict[str, Any]:
    """Choose and classify the best supported candidate among one OCR line."""

    first_latex = candidates[0] if candidates else ""
    verdicts: list[dict[str, Any]] = []

    for index, latex in enumerate(candidates[:MAX_OCR_CANDIDATES]):
        text = str(latex or "").strip()
        if not text:
            continue
        verdicts.append(classify_equation_candidate(
            text,
            index,
            manifest,
            exact_values,
            variable,
            reference_latex,
        ))

    if verdicts:
        best, _best_original_index = _pick_best_candidate_verdict(verdicts)
        if (
            best is not None and
            best.get("classification") != "other" and
            best.get("solutionCoverage") == "full"
        ):
            return best

    repaired = repeated_assignment_equals_repair_verdict(
        candidates,
        manifest,
        exact_values,
        variable,
        reference_latex,
    )
    if repaired is not None:
        return repaired

    if verdicts:
        best, _best_original_index = _pick_best_candidate_verdict(verdicts)
        if best is not None and best.get("classification") != "other":
            return best

    # Distinguish between lines with no parseable content at all
    # (unrecognized) and lines with some non-equation content (other).
    has_any_text = any(
        str(latex or "").strip()
        for latex in candidates[:MAX_OCR_CANDIDATES]
    )
    if not has_any_text:
        return selected_line(first_latex, "unrecognized", 0 if candidates else None, "none", ())

    return selected_line(first_latex, "other", 0 if candidates else None, "none", ())


def classify_equation_candidate(
    text: str,
    index: int,
    manifest: dict[str, Any],
    exact_values: Sequence[sympy.Expr],
    variable: str,
    reference_latex: str,
) -> dict[str, Any]:
    special = special_solution_match(text, manifest)
    if special is not None:
        return selected_line(
            text,
            "valid_step",
            index,
            special.coverage,
            special.matched_indices,
            accepted_special=special.accepted_special,
            finality=special.finality,
        )

    solution_match = finite_solution_match(text, manifest, exact_values, variable)
    if solution_match is not None:
        return selected_line(
            text,
            "valid_step",
            index,
            solution_match.coverage,
            solution_match.matched_indices,
            finality=solution_match.finality,
        )

    general_match = general_solution_match(text, manifest, variable)
    if general_match is not None:
        return selected_line(
            text,
            "valid_step",
            index,
            general_match.coverage,
            general_match.matched_indices,
            finality=general_match.finality,
        )

    if equation_equivalent(reference_latex, text, variable):
        if manifest.get("cardinality") == "none" and equation_contradiction(text):
            return selected_line(
                text,
                "valid_step",
                index,
                "full",
                (-1,),
                accepted_special="none",
                finality=final_answer(),
            )
        return selected_line(text, "valid_step", index, "none", ())

    if is_parseable_equation(text):
        return selected_line(text, "invalid_step", index, "none", ())

    return selected_line(text, "other", index, "none", ())


def repeated_assignment_equals_repair_verdict(
    candidates: Sequence[str],
    manifest: dict[str, Any],
    exact_values: Sequence[sympy.Expr],
    variable: str,
    reference_latex: str,
) -> Optional[dict[str, Any]]:
    if manifest.get("cardinality") != "finite" or len(exact_values) < 2:
        return None

    for index, latex in enumerate(candidates[:MAX_OCR_CANDIDATES]):
        original = str(latex or "").strip()
        if not original:
            continue
        repaired = repeated_assignment_equals_repair_candidate(original, variable)
        if not repaired:
            continue
        verdict = classify_equation_candidate(
            repaired,
            index,
            manifest,
            exact_values,
            variable,
            reference_latex,
        )
        if verdict.get("classification") != "valid_step" or verdict.get("solutionCoverage") != "full":
            continue
        return {
            **verdict,
            "studentLatex": original,
            "repairedLatex": repaired,
            "ocrRepair": {
                "source": "repeated-assignment-equals-repair",
                "originalLatex": original,
                "repairedLatex": repaired,
            },
        }
    return None


def repeated_assignment_equals_repair_candidate(text: str, variable: str) -> Optional[str]:
    raw = str(text or "").strip()
    variable_pattern = re.escape(str(variable or "x"))
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    match = re.fullmatch(
        rf"\s*({variable_pattern})\s*=\s*({number})\s+\1\s*[-+]\s*({number})\s*",
        raw,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return f"{match.group(1)} = {match.group(2)} {match.group(1)} = {match.group(3)}"


def classify_expression_candidates(
    candidates: Sequence[str],
    manifest: dict[str, Any],
    exact_values: Sequence[sympy.Expr],
) -> dict[str, Any]:
    first_latex = candidates[0] if candidates else ""
    verdicts: list[dict[str, Any]] = []

    for index, latex in enumerate(candidates[:MAX_OCR_CANDIDATES]):
        text = str(latex or "").strip()
        if not text:
            continue
        verdicts.append(classify_expression_candidate(text, index, manifest, exact_values))

    if verdicts:
        best, _best_original_index = _pick_best_candidate_verdict(verdicts)
        if best is not None and best.get("solutionCoverage") == "full":
            return best

    repaired = expression_equals_repair_verdict(candidates, manifest, exact_values)
    if repaired is not None:
        return repaired

    if verdicts:
        best, _best_original_index = _pick_best_candidate_verdict(verdicts)
        if best is not None:
            return best

    has_any_text = any(str(latex or "").strip() for latex in candidates[:MAX_OCR_CANDIDATES])
    if not has_any_text:
        return selected_line(first_latex, "unrecognized", 0 if candidates else None, "none", ())
    return selected_line(first_latex, "other", 0 if candidates else None, "none", ())


def classify_expression_candidate(
    text: str,
    index: int,
    manifest: dict[str, Any],
    exact_values: Sequence[sympy.Expr],
) -> dict[str, Any]:
    try:
        expression_parts = parse_expression_sequence(text)
    except GradingParseFailure:
        return selected_line(text, "other", index, "none", ())

    if not expression_parts:
        return selected_line(text, "other", index, "none", ())

    if any(value.free_symbols for _raw, value in expression_parts):
        return selected_line(
            text,
            "invalid_step",
            index,
            "none",
            (),
            finality=invalid_format("expression answers must not contain variables"),
        )

    broken = first_broken_expression_parts_index(expression_parts, manifest)
    if broken is not None:
        return selected_line(
            text,
            "invalid_step",
            index,
            "none",
            (),
            finality=invalid_format("adjacent expressions are not equivalent"),
        )

    final_raw, final_value = expression_parts[-1]
    matched: set[int] = set()
    for exact_index, exact in enumerate(exact_values):
        if solution_values_equivalent(final_value, exact, manifest):
            matched.add(exact_index)

    if not matched:
        return selected_line(text, "invalid_step", index, "none", (), finality=invalid_format("numeric value does not match"))

    finality = expression_value_finality(final_raw, final_value)
    verdict = selected_line(
        text,
        "valid_step",
        index,
        "full",
        tuple(sorted(matched)),
        finality=finality,
    )
    verdict["_expression_elements"] = tuple(expression_parts)
    return verdict


def classify_simplification_candidates(
    candidates: Sequence[str],
    manifest: dict[str, Any],
    exact_values: Sequence[sympy.Expr],
) -> dict[str, Any]:
    first_latex = candidates[0] if candidates else ""
    verdicts: list[dict[str, Any]] = []

    for index, latex in enumerate(candidates[:MAX_OCR_CANDIDATES]):
        text = str(latex or "").strip()
        if not text:
            continue
        verdicts.append(classify_simplification_candidate(text, index, manifest, exact_values))

    if verdicts:
        best, _best_original_index = _pick_best_candidate_verdict(verdicts)
        if best is not None and best.get("solutionCoverage") == "full":
            return best

    repaired = simplification_equals_repair_verdict(candidates, manifest, exact_values)
    if repaired is not None:
        return repaired

    if verdicts:
        best, _best_original_index = _pick_best_candidate_verdict(verdicts)
        if best is not None:
            return best

    has_any_text = any(str(latex or "").strip() for latex in candidates[:MAX_OCR_CANDIDATES])
    if not has_any_text:
        return selected_line(first_latex, "unrecognized", 0 if candidates else None, "none", ())
    return selected_line(first_latex, "other", 0 if candidates else None, "none", ())


def classify_simplification_candidate(
    text: str,
    index: int,
    manifest: dict[str, Any],
    exact_values: Sequence[sympy.Expr],
) -> dict[str, Any]:
    try:
        expression_parts = parse_expression_sequence(text)
    except GradingParseFailure:
        return selected_line(text, "other", index, "none", ())

    if not expression_parts:
        return selected_line(text, "other", index, "none", ())

    broken = first_broken_expression_parts_index(expression_parts, manifest)
    if broken is not None:
        return selected_line(
            text,
            "invalid_step",
            index,
            "none",
            (),
            finality=invalid_format("adjacent expressions are not equivalent"),
        )

    final_raw, final_value = expression_parts[-1]
    matched: set[int] = set()
    for exact_index, exact in enumerate(exact_values):
        if solution_values_equivalent(final_value, exact, manifest):
            matched.add(exact_index)

    if not matched:
        return selected_line(
            text,
            "invalid_step",
            index,
            "none",
            (),
            finality=invalid_format("expression does not match the simplified target"),
        )

    target = exact_values[sorted(matched)[0]]
    finality = simplification_value_finality(final_raw, final_value, target, manifest)
    verdict = selected_line(
        text,
        "valid_step",
        index,
        "full",
        tuple(sorted(matched)),
        finality=finality,
    )
    verdict["_expression_elements"] = tuple(expression_parts)
    return verdict


def simplification_equals_repair_verdict(
    candidates: Sequence[str],
    manifest: dict[str, Any],
    exact_values: Sequence[sympy.Expr],
) -> Optional[dict[str, Any]]:
    for index, latex in enumerate(candidates[:MAX_OCR_CANDIDATES]):
        original = str(latex or "").strip()
        if not original:
            continue
        for repaired in expression_equals_repair_candidates(original):
            verdict = classify_simplification_candidate(repaired, index, manifest, exact_values)
            if verdict.get("classification") != "valid_step" or verdict.get("solutionCoverage") != "full":
                continue
            return {
                **verdict,
                "studentLatex": original,
                "repairedLatex": repaired,
                "ocrRepair": {
                    "source": "simplification-equals-repair",
                    "originalLatex": original,
                    "repairedLatex": repaired,
                },
            }
    return None


def expression_equals_repair_verdict(
    candidates: Sequence[str],
    manifest: dict[str, Any],
    exact_values: Sequence[sympy.Expr],
) -> Optional[dict[str, Any]]:
    for index, latex in enumerate(candidates[:MAX_OCR_CANDIDATES]):
        original = str(latex or "").strip()
        if not original:
            continue
        for repaired in expression_equals_repair_candidates(original):
            verdict = classify_expression_candidate(repaired, index, manifest, exact_values)
            if verdict.get("classification") != "valid_step" or verdict.get("solutionCoverage") != "full":
                continue
            verdict = {
                **verdict,
                "studentLatex": original,
                "repairedLatex": repaired,
                "ocrRepair": {
                    "source": "expression-equals-repair",
                    "originalLatex": original,
                    "repairedLatex": repaired,
                },
            }
            return verdict
    return None


def expression_equals_repair_candidates(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []

    repairs: list[str] = []
    for index, char in enumerate(raw):
        if char not in "+-":
            continue
        if not is_binary_additive_operator(raw, index):
            continue
        candidate = f"{raw[:index]}={raw[index + 1:]}"
        if candidate.count("=") < 1:
            continue
        repairs.append(candidate)
    return list(dict.fromkeys(repairs))


def is_binary_additive_operator(text: str, index: int) -> bool:
    left = previous_non_space(text, index)
    right = next_non_space(text, index)
    if left is None or right is None:
        return False
    if left in "=+-*/^({[,":  # unary sign or operator sequence
        return False
    if right in "=+*/^)}],":
        return False
    return True


def previous_non_space(text: str, index: int) -> Optional[str]:
    for cursor in range(index - 1, -1, -1):
        if not text[cursor].isspace():
            return text[cursor]
    return None


def next_non_space(text: str, index: int) -> Optional[str]:
    for cursor in range(index + 1, len(text)):
        if not text[cursor].isspace():
            return text[cursor]
    return None


def first_broken_expression_parts_index(
    expression_parts: Sequence[tuple[str, sympy.Expr]],
    manifest: dict[str, Any],
) -> Optional[int]:
    for index in range(1, len(expression_parts)):
        if not solution_values_equivalent(expression_parts[index - 1][1], expression_parts[index][1], manifest):
            return index
    return None


def first_broken_expression_link_index(
    expression_elements: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
) -> Optional[int]:
    for index in range(1, len(expression_elements)):
        previous = expression_elements[index - 1]["value"]
        current = expression_elements[index]["value"]
        if not solution_values_equivalent(previous, current, manifest):
            return int(expression_elements[index]["lineIndex"])
    return None


def mark_expression_line_invalid(
    selected_by_line: list[dict[str, Any]],
    line_index: int,
    reason: str,
) -> None:
    for step in selected_by_line:
        if int(step.get("lineIndex", -1)) != int(line_index):
            continue
        step.update({
            "classification": "invalid_step",
            "solutionCoverage": "none",
            "_matched_indices": (),
            **invalid_format(reason).public_fields(),
        })
        return


def clear_expression_completion_credit(selected_by_line: list[dict[str, Any]]) -> None:
    for step in selected_by_line:
        if step.get("classification") != "valid_step":
            continue
        step.update({
            "solutionCoverage": "none",
            "_matched_indices": (),
            **not_answer().public_fields(),
        })


def mark_expression_line_final(
    selected_by_line: list[dict[str, Any]],
    line_index: int,
    matched_indices: Sequence[int],
    finality: AnswerFinality,
) -> None:
    for step in reversed(selected_by_line):
        if int(step.get("lineIndex", -1)) != int(line_index):
            continue
        step.update({
            "classification": "valid_step",
            "solutionCoverage": "full",
            "_matched_indices": tuple(sorted(set(matched_indices))),
            **finality.public_fields(),
        })
        return


def public_line_selection(selected: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    matched_indices = tuple(selected.get("_matched_indices") or ())
    exact_set = list(manifest.get("exact_set") or [])
    matched_solutions = [
        str(exact_set[index])
        for index in matched_indices
        if index >= 0 and index < len(exact_set)
    ]
    if not matched_solutions and -1 in matched_indices:
        matched_solutions = [str(selected.get("acceptedSpecialAnswer") or manifest.get("cardinality") or "")]
    return {
        **selected,
        "matchedSolutions": [item for item in matched_solutions if item],
    }


def strip_private_selection_fields(selected: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in selected.items()
        if not key.startswith("_")
    }


def last_reference_latex(
    previous_latex: Sequence[str],
    problem_latex: str,
    manifest: dict[str, Any],
) -> str:
    for latex in reversed(list(previous_latex or [])):
        if str(latex or "").strip():
            return str(latex)
    return str(problem_latex or manifest.get("problem_raw") or "")


def selected_line(
    latex: str,
    classification: str,
    selected_candidate_index: Optional[int],
    coverage: str,
    matched_indices: Sequence[int],
    *,
    accepted_special: Optional[str] = None,
    finality: Optional[AnswerFinality] = None,
) -> dict[str, Any]:
    finality = finality or (final_answer() if coverage in {"full", "partial"} else not_answer())
    result = {
        "studentLatex": latex,
        "classification": classification,
        "selectedCandidateIndex": selected_candidate_index,
        "solutionCoverage": coverage,
        "matchedSolutions": [],
        "_matched_indices": tuple(sorted(set(matched_indices))),
        **finality.public_fields(),
    }
    if accepted_special:
        result["acceptedSpecialAnswer"] = accepted_special
    return result


def problem_complete(
    manifest: dict[str, Any],
    exact_values: Sequence[sympy.Expr],
    found_indices: set[int],
) -> bool:
    cardinality = manifest.get("cardinality")
    if cardinality == "finite":
        return bool(exact_values) and len(found_indices) == len(exact_values)
    if cardinality in {"none", "infinite"}:
        return -1 in found_indices
    if cardinality == "infinite_family":
        return -1 in found_indices
    return False


def finite_solution_match(
    latex: str,
    manifest: dict[str, Any],
    exact_values: Sequence[sympy.Expr],
    variable: str,
) -> Optional[SolutionMatch]:
    if manifest.get("cardinality") != "finite" or not exact_values:
        return None

    entries = candidate_solution_entries(latex, variable)
    if not entries:
        return None

    isolated_assignment = has_isolated_variable_assignment(latex, variable)
    matched: set[int] = set()
    all_matched_final = True
    unsimplified_reasons: list[str] = []
    for _raw_part, candidate_value, finality in entries:
        for index, exact in enumerate(exact_values):
            if solution_values_equivalent(candidate_value, exact, manifest):
                matched.add(index)
                if finality.status != "final" and not isolated_assignment:
                    all_matched_final = False
                    if finality.reason:
                        unsimplified_reasons.append(finality.reason)

    if not matched:
        return None
    coverage = "full" if len(matched) == len(exact_values) else "partial"
    answer_finality = final_answer() if all_matched_final else unsimplified_answer(
        unsimplified_reasons[0] if unsimplified_reasons else "answer is equivalent but not fully simplified"
    )
    return SolutionMatch(tuple(sorted(matched)), coverage, finality=answer_finality)


def general_solution_match(latex: str, manifest: dict[str, Any], variable: str) -> Optional[SolutionMatch]:
    if manifest.get("cardinality") != "infinite_family":
        return None

    expected = manifest_solution_set(manifest, variable)
    if expected is None:
        return None
    candidate = candidate_general_solution_set(latex, variable)
    if candidate is None:
        return None
    if solution_sets_equal(expected, candidate):
        return SolutionMatch((-1,), "full", finality=final_answer())
    return None


def manifest_solution_set(manifest: dict[str, Any], variable: str) -> Optional[sympy.Set]:
    raw = str(manifest.get("problem_raw") or "").strip()
    if not raw:
        return None
    try:
        parsed = parse_math(raw)
    except GradingParseFailure:
        return None
    if parsed.kind != "equation":
        return None
    symbol = sympy.Symbol(variable, real=True)
    return _solveset_with_budget(parsed.residual, symbol)


def candidate_general_solution_set(latex: str, variable: str) -> Optional[sympy.Set]:
    parts = split_solution_parts(latex, variable)
    if not parts:
        return None

    solution_sets: list[sympy.Set] = []
    solve_symbol = sympy.Symbol(variable, real=True)
    for part in parts:
        try:
            expression = parse_expression(part)
        except GradingParseFailure:
            continue
        parameter_symbols = sorted(expression.free_symbols - {solve_symbol}, key=lambda item: item.name)
        if len(parameter_symbols) != 1 or solve_symbol in expression.free_symbols:
            continue
        parameter = parameter_symbols[0]
        integer_parameter = sympy.Symbol(parameter.name, integer=True)
        integer_expression = expression.xreplace({parameter: integer_parameter})
        solution_sets.append(sympy.imageset(integer_parameter, integer_expression, sympy.S.Integers))

    if not solution_sets:
        return None
    if len(solution_sets) == 1:
        return solution_sets[0]
    return sympy.Union(*solution_sets)


def special_solution_match(latex: str, manifest: dict[str, Any]) -> Optional[SolutionMatch]:
    cardinality = manifest.get("cardinality")
    compact = compact_answer_text(latex)
    acceptable = {
        compact_answer_text(item)
        for item in manifest.get("acceptable_strings") or []
    }
    if compact in acceptable:
        if cardinality in {"none", "infinite"}:
            return SolutionMatch((-1,), "full", accepted_special=cardinality, finality=final_answer())
        if cardinality == "finite":
            return None

    if cardinality == "none" and compact in NO_SOLUTION_STRINGS:
        return SolutionMatch((-1,), "full", accepted_special="none", finality=final_answer())

    if cardinality == "infinite":
        if compact in INFINITE_SOLUTION_STRINGS or re.fullmatch(r"[a-z]+in(?:mathbb)?r", compact):
            return SolutionMatch((-1,), "full", accepted_special="infinite", finality=final_answer())
        if equation_identity(latex):
            return SolutionMatch((-1,), "full", accepted_special="identity", finality=final_answer())

    return None


def solution_values_equivalent(
    candidate_value: sympy.Expr,
    exact_value: sympy.Expr,
    manifest: dict[str, Any],
) -> bool:
    simplified = _simplify_with_budget(candidate_value - exact_value)
    if simplified is not None:
        try:
            if simplified == 0:
                return True
        except Exception:
            pass

    candidate_float = safe_float(candidate_value)
    exact_float = safe_float(exact_value)
    if candidate_float is None or exact_float is None:
        return False
    tolerance = float(manifest.get("tolerance", DEFAULT_TOLERANCE))
    return abs(candidate_float - exact_float) <= tolerance + 1e-12


def equation_equivalent(reference_latex: str, candidate_latex: str, variable: str) -> bool:
    try:
        reference = parse_math(reference_latex)
        candidate = parse_math(candidate_latex)
    except GradingParseFailure:
        return False
    if reference.kind != "equation" or candidate.kind != "equation":
        return False

    symbol = sympy.Symbol(variable, real=True)
    reference_set = _solveset_with_budget(reference.residual, symbol)
    if reference_set is None:
        return False
    candidate_set = _solveset_with_budget(candidate.residual, symbol)
    if candidate_set is None:
        return False
    return solution_sets_equal(reference_set, candidate_set)


def equation_identity(latex: str) -> bool:
    try:
        parsed = parse_math(latex)
    except GradingParseFailure:
        return False
    if parsed.kind != "equation":
        return False
    simplified = _simplify_with_budget(parsed.residual)
    if simplified is None:
        return False
    try:
        return bool(simplified == 0)
    except Exception:
        return False


def equation_contradiction(latex: str) -> bool:
    try:
        parsed = parse_math(latex)
    except GradingParseFailure:
        return False
    if parsed.kind != "equation":
        return False
    simplified = _simplify_with_budget(parsed.residual)
    if simplified is None or getattr(simplified, "free_symbols", set()):
        return False
    try:
        return bool(simplified != 0)
    except Exception:
        return False


def solution_sets_equal(a: sympy.Set, b: sympy.Set) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    if isinstance(a, sympy.FiniteSet) and isinstance(b, sympy.FiniteSet) and len(a) == len(b):
        remaining = list(b)
        for left in a:
            match_index = next(
                (
                    index for index, right in enumerate(remaining)
                    if solution_values_equivalent(left, right, {"tolerance": DEFAULT_TOLERANCE})
                ),
                None,
            )
            if match_index is None:
                return False
            remaining.pop(match_index)
        return True
    if symmetric_difference_empty(a, b):
        return True
    canonical_a = canonical_periodic_set(a)
    canonical_b = canonical_periodic_set(b)
    if canonical_a is not None and canonical_a == canonical_b:
        return True
    if sampled_sets_match(a, b):
        return True
    return False


def symmetric_difference_empty(a: sympy.Set, b: sympy.Set) -> bool:
    try:
        with sympy_grader_budget():
            symmetric_difference = sympy.simplify(a.symmetric_difference(b))
    except (Exception, SympyGraderBudgetExceeded):
        return False
    return symmetric_difference == sympy.S.EmptySet


def canonical_periodic_set(solution_set: sympy.Set) -> Optional[tuple[Any, ...]]:
    if isinstance(solution_set, sympy.ImageSet):
        lambda_expr, base_set = solution_set.args
        if base_set != sympy.S.Integers or not isinstance(lambda_expr, sympy.Lambda):
            return None
        parameter = lambda_expr.variables[0]
        canonical_parameter = sympy.Symbol("_k", integer=True)
        canonical_expr = sympy.simplify(lambda_expr.expr.xreplace({parameter: canonical_parameter}))
        return ("image", sympy.sstr(canonical_expr))
    if isinstance(solution_set, sympy.Union):
        parts = [canonical_periodic_set(part) for part in solution_set.args]
        if any(part is None for part in parts):
            return None
        return ("union", tuple(sorted(parts)))
    return None


def sampled_sets_match(a: sympy.Set, b: sympy.Set) -> bool:
    a_samples = sample_solution_set_values(a)
    b_samples = sample_solution_set_values(b)
    if not a_samples or not b_samples:
        return False
    for value in a_samples:
        if not set_contains_value(b, value):
            return False
    for value in b_samples:
        if not set_contains_value(a, value):
            return False
    return True


def sample_solution_set_values(solution_set: sympy.Set) -> list[sympy.Expr]:
    if isinstance(solution_set, sympy.FiniteSet):
        return list(solution_set)
    if isinstance(solution_set, sympy.ImageSet):
        lambda_expr, base_set = solution_set.args
        if base_set != sympy.S.Integers or not isinstance(lambda_expr, sympy.Lambda):
            return []
        parameter = lambda_expr.variables[0]
        return [
            sympy.simplify(lambda_expr.expr.subs(parameter, value))
            for value in range(-3, 4)
        ]
    if isinstance(solution_set, sympy.Union):
        samples: list[sympy.Expr] = []
        for part in solution_set.args:
            samples.extend(sample_solution_set_values(part))
        return samples
    return []


def set_contains_value(solution_set: sympy.Set, value: sympy.Expr) -> bool:
    try:
        contained = solution_set.contains(value)
        if contained == sympy.true:
            return True
        if contained == sympy.false:
            return False
    except Exception:
        pass
    if isinstance(solution_set, sympy.FiniteSet):
        return any(
            solution_values_equivalent(value, exact, {"tolerance": DEFAULT_TOLERANCE})
            for exact in solution_set
        )
    return False


def is_parseable_equation(latex: str) -> bool:
    try:
        return parse_math(latex).kind == "equation"
    except GradingParseFailure:
        return False


def candidate_solution_values(latex: str, variable: str) -> list[sympy.Expr]:
    return [entry[1] for entry in candidate_solution_entries(latex, variable)]


def candidate_solution_entries(latex: str, variable: str) -> list[tuple[str, sympy.Expr, AnswerFinality]]:
    text = str(latex or "").strip()
    if not text:
        return []

    values: list[tuple[str, sympy.Expr, AnswerFinality]] = []
    for part in split_solution_parts(text, variable):
        for expanded in expand_pm(part):
            try:
                parsed = parse_expression(expanded)
            except GradingParseFailure:
                continue
            if parsed.free_symbols:
                continue
            values.append((expanded, parsed, expression_value_finality(expanded, parsed)))
    return values


def split_solution_parts(latex: str, variable: str) -> list[str]:
    text = str(latex or "").strip()
    if not text:
        return []

    membership_parts = split_set_membership_solution(text, variable)
    if membership_parts:
        return membership_parts

    repeated_assignments = split_repeated_solution_assignments(text, variable)
    if repeated_assignments:
        return repeated_assignments

    equation_match = split_equation_text(text)
    if equation_match is not None:
        left_text, right_text = equation_match
        left_compact = compact_answer_text(left_text)
        right_compact = compact_answer_text(right_text)
        if left_compact == compact_answer_text(variable):
            return split_solution_list(right_text)
        if right_compact == compact_answer_text(variable):
            return split_solution_list(left_text)

    return split_solution_list(text)


def has_isolated_variable_assignment(latex: str, variable: str) -> bool:
    equation_match = split_equation_text(str(latex or "").strip())
    if equation_match is None:
        return False
    left_text, right_text = equation_match
    variable_text = compact_answer_text(variable)
    return (
        compact_answer_text(left_text) == variable_text or
        compact_answer_text(right_text) == variable_text
    )


def split_repeated_solution_assignments(text: str, variable: str) -> list[str]:
    variable_pattern = re.escape(str(variable or "x").strip())
    if not variable_pattern:
        return []
    pattern = re.compile(rf"(?:^|\s){variable_pattern}\s*=\s*(.+?)(?=(?:\s*{variable_pattern}\s*=)|$)")
    matches = [match.group(1).strip() for match in pattern.finditer(str(text or "").strip())]
    matches = [match for match in matches if match and "=" not in match]
    if len(matches) < 2:
        return []
    return [part for match in matches for part in split_solution_list(match)]


def split_set_membership_solution(text: str, variable: str) -> list[str]:
    variable_compact = compact_answer_text(variable)
    for token in (r"\in", "\u2208", " in "):
        if token not in text:
            continue
        left, right = text.split(token, 1)
        if compact_answer_text(left) == variable_compact:
            return split_solution_list(right)
    return []


def split_solution_list(text: str) -> list[str]:
    return split_top_level_commas(strip_solution_set_wrapper(text))


def strip_solution_set_wrapper(text: str) -> str:
    output = str(text or "").strip()
    for left, right in ((r"\left\{", r"\right\}"), (r"\{", r"\}"), ("{", "}")):
        if output.startswith(left) and output.endswith(right):
            return output[len(left):len(output) - len(right)].strip()
    return output


def expand_pm(text: str) -> list[str]:
    if r"\pm" not in text and "\u00b1" not in text:
        return [text]
    plus = text.replace(r"\pm", "+", 1).replace("\u00b1", "+", 1)
    minus = text.replace(r"\pm", "-", 1).replace("\u00b1", "-", 1)
    return [item for branch in (plus, minus) for item in expand_pm(branch)]


def line_candidate_latex(line: dict[str, Any] | str) -> list[str]:
    if isinstance(line, str):
        return [line]

    values: list[str] = []
    for key in ("acceptedLatex", "latex", "studentLatex"):
        value = str(line.get(key) or "").strip()
        if value:
            values.append(value)

    for candidate in line.get("candidates") or []:
        if isinstance(candidate, str):
            value = candidate.strip()
        else:
            value = str(candidate.get("latex") or "").strip()
        if value:
            values.append(value)

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        compact = compact_answer_text(value)
        if compact in seen:
            continue
        seen.add(compact)
        deduped.append(value)
        if len(deduped) >= MAX_OCR_CANDIDATES:
            break
    return deduped


def manifest_exact_values(manifest: dict[str, Any]) -> list[sympy.Expr]:
    values = []
    for item in manifest.get("exact_set") or []:
        try:
            values.append(parse_expression(str(item)))
        except GradingParseFailure:
            continue
    return values


def resolve_variable(
    expression: sympy.Expr,
    variable: Optional[str],
    *,
    fallback_symbols: Iterable[sympy.Symbol] = (),
) -> Optional[sympy.Symbol]:
    if variable:
        try:
            parsed = parse_expression(str(variable))
        except GradingParseFailure:
            parsed = sympy.Symbol(str(variable).strip(), real=True)
        if isinstance(parsed, sympy.Symbol):
            return parsed
        return sympy.Symbol(str(variable).strip(), real=True)

    symbols = sorted(expression.free_symbols or set(fallback_symbols), key=lambda item: item.name)
    if len(symbols) != 1:
        return None
    return symbols[0]


def acceptable_strings_for_finite(variable: str, values: Sequence[sympy.Expr]) -> list[str]:
    strings: set[str] = set()
    exact = [sympy.sstr(sympy.simplify(value)) for value in values]
    for item in exact:
        compact = item.replace(" ", "")
        strings.add(compact)
        strings.add(f"{variable}={compact}")
    if len(exact) > 1:
        joined = ",".join(item.replace(" ", "") for item in exact)
        strings.add(joined)
        strings.add(f"{variable}={joined}")
    return sorted(strings)


def parse_math(latex: str) -> ParsedMath:
    equation = split_equation_text(latex)
    if equation is None:
        expression = parse_expression(latex)
        return ParsedMath("expression", expression, None, sympy.sstr(expression))

    left_text, right_text = equation
    left = parse_expression(left_text)
    right = parse_expression(right_text)
    return ParsedMath("equation", left, right, f"{sympy.sstr(left)} = {sympy.sstr(right)}")


def parse_expression_sequence(text: str) -> list[tuple[str, sympy.Expr]]:
    raw = str(text or "").strip()
    if not raw:
        return []

    parts = split_top_level_equals(raw)
    if not parts:
        parts = [raw]

    if parts[0].strip() == "":
        parts = parts[1:]
    if not parts or any(not part.strip() for part in parts):
        raise GradingParseFailure("expression chain contains a missing expression")

    expressions: list[tuple[str, sympy.Expr]] = []
    for part in parts:
        parsed = parse_expression(part)
        expressions.append((part.strip(), parsed))
    return expressions


def parse_expression(text: str, *, evaluate: bool = True) -> sympy.Expr:
    normalized = normalize_math_text(text)
    if not normalized:
        raise GradingParseFailure("empty expression")
    identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", normalized))
    local_dict: dict[str, Any] = {**KNOWN_FUNCTIONS, **KNOWN_CONSTANTS}
    for name in identifiers:
        if name not in local_dict:
            local_dict[name] = sympy.Symbol(name, real=True)
    try:
        parsed = parse_expr(
            normalized,
            local_dict=local_dict,
            transformations=TRANSFORMATIONS,
            evaluate=evaluate,
        )
    except Exception as exc:
        raise GradingParseFailure(f"could not parse expression: {exc}") from exc
    if not isinstance(parsed, sympy.Expr):
        raise GradingParseFailure("not an expression")
    if parsed.has(sympy.zoo, sympy.oo, -sympy.oo, sympy.nan):
        raise GradingParseFailure("expression contains a non-finite value")
    return parsed


def expression_value_finality(raw_text: str, evaluated: sympy.Expr) -> AnswerFinality:
    try:
        normalized = normalize_math_text(raw_text)
        unevaluated = parse_expression(raw_text, evaluate=False)
    except GradingParseFailure:
        return invalid_format("could not parse final answer")

    if unevaluated.free_symbols:
        return invalid_format("final numeric answer contains variables")

    finality_normalized = normalized[1:] if normalized.startswith("+") else normalized
    compact_normalized = compact_math_form(finality_normalized)
    if is_decimal_or_integer_literal(finality_normalized):
        return final_answer()
    if is_reduced_fraction_literal(finality_normalized, evaluated):
        return final_answer()

    compact_canonical = compact_math_form(sympy.sstr(evaluated))
    if compact_normalized == compact_canonical:
        return final_answer()

    if compact_math_form(normalize_e_power_text(finality_normalized)) == compact_canonical:
        return final_answer()

    simplified = _simplify_with_budget(unevaluated - evaluated)
    if simplified == 0:
        return unsimplified_answer("answer is equivalent but not fully simplified")

    return final_answer()


def simplification_value_finality(
    raw_text: str,
    evaluated: sympy.Expr,
    target: sympy.Expr,
    manifest: dict[str, Any],
) -> AnswerFinality:
    try:
        unevaluated = parse_expression(raw_text, evaluate=False)
    except GradingParseFailure:
        return invalid_format("could not parse final expression")

    if not solution_values_equivalent(evaluated, target, manifest):
        return invalid_format("final expression is not equivalent to the simplified target")

    if expressions_match_commutative_structure(unevaluated, target):
        return final_answer()

    return unsimplified_answer("expression is equivalent but not fully simplified")


def expressions_match_commutative_structure(left: sympy.Expr, right: sympy.Expr) -> bool:
    return commutative_structure_key(left) == commutative_structure_key(right)


def commutative_structure_key(expression: sympy.Expr) -> tuple[Any, ...]:
    if isinstance(expression, sympy.Add):
        return (
            "Add",
            tuple(sorted(
                (commutative_structure_key(arg) for arg in expression.args),
                key=repr,
            )),
        )
    if isinstance(expression, sympy.Mul):
        return (
            "Mul",
            tuple(sorted(
                (commutative_structure_key(arg) for arg in expression.args),
                key=repr,
            )),
        )
    if getattr(expression, "args", None):
        return (
            expression.func.__name__,
            tuple(commutative_structure_key(arg) for arg in expression.args),
        )
    return (expression.func.__name__, sympy.sstr(expression))


def expression_answer_string(value: sympy.Expr) -> str:
    if isinstance(value, sympy.Float):
        try:
            return format(float(value), ".12g")
        except Exception:
            pass
    return sympy.sstr(value)


def compact_math_form(text: str) -> str:
    compact = str(text or "").replace(" ", "")
    compact = compact.replace("**", "^")
    compact = compact.replace("*", "")
    return compact


def normalize_e_power_text(text: str) -> str:
    return re.sub(r"\be\^\(?([^()]+)\)?", r"exp(\1)", str(text or ""))


def is_decimal_or_integer_literal(text: str) -> bool:
    return bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", str(text or "").strip()))


def is_reduced_fraction_literal(text: str, evaluated: sympy.Expr) -> bool:
    cleaned = str(text or "").strip()
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = re.sub(r"\((-?\d+)\)", r"\1", cleaned)
        cleaned = strip_outer_parentheses(cleaned)

    match = re.fullmatch(r"(-?)\(?(-?\d+)/(-?\d+)\)?", cleaned)
    if not match:
        return False
    numerator = int(match.group(2))
    denominator = int(match.group(3))
    if match.group(1):
        numerator = -numerator
    if denominator == 0 or math.gcd(numerator, denominator) != 1:
        return False
    try:
        return sympy.Rational(numerator, denominator) == evaluated
    except Exception:
        return False


def strip_outer_parentheses(text: str) -> str:
    output = str(text or "").strip()
    while output.startswith("(") and output.endswith(")") and outer_parentheses_wrap(output):
        output = output[1:-1].strip()
    return output


def outer_parentheses_wrap(text: str) -> bool:
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index < len(text) - 1:
                return False
    return depth == 0


def normalize_math_text(text: str) -> str:
    output = str(text or "").strip()
    if not output:
        return ""

    output = output.replace("$", "")
    for token in (r"\left", r"\right", r"\limits", r"\!", r"\,", r"\;", r"\:"):
        output = output.replace(token, "")
    output = output.replace(r"\lvert", "|").replace(r"\rvert", "|")
    output = output.replace(r"\operatorname", "")
    output = re.sub(r"\\text\s*\{[^{}]*\}", "", output)
    output = re.sub(r"\\mathrm\s*\{\s*([A-Za-z])\s*\}", r"\1", output)
    output = re.sub(r"\\mathbb\s*\{\s*R\s*\}", "R", output)

    for old, new in LATEX_COMMAND_REPLACEMENTS.items():
        output = output.replace(old, new)
    output = normalize_inverse_trig_notation(output)
    for old, new in sorted(LATEX_FUNCTION_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True):
        output = output.replace(old, new)
    for old, new in sorted(LATEX_VARIABLE_COMMANDS.items(), key=lambda item: len(item[0]), reverse=True):
        output = output.replace(old, new)

    output = replace_structural_latex(output)
    output = output.replace("{", "(").replace("}", ")")
    output = normalize_absolute_value_notation(output)
    output = normalize_log_base_application(output)
    output = normalize_bare_function_application(output)
    output = normalize_spaced_implicit_multiplication(output)
    output = re.sub(r"\s+", "", output)
    if "\\" in output:
        raise GradingParseFailure("unsupported LaTeX command")
    return output


def normalize_absolute_value_notation(text: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != "|":
            result.append(text[index])
            index += 1
            continue

        closing = find_matching_absolute_value_bar(text, index + 1)
        if closing is None:
            result.append(text[index])
            index += 1
            continue

        inner = text[index + 1:closing].strip()
        if not inner:
            result.append(text[index])
            index += 1
            continue
        result.append(f"abs({inner})")
        index = closing + 1

    return "".join(result)


def find_matching_absolute_value_bar(text: str, start: int) -> Optional[int]:
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "|" and depth == 0:
            return index
    return None


def normalize_inverse_trig_notation(text: str) -> str:
    """Treat sin^{-1}(x) as inverse trig, not reciprocal exponent syntax."""

    def replace(match: re.Match[str]) -> str:
        return f"arc{match.group(1)}"

    return re.sub(
        r"\\?(sin|cos|tan)\s*\^\s*(?:\{\s*-\s*1\s*\}|\(\s*-\s*1\s*\)|-\s*1)",
        replace,
        text,
    )


def normalize_log_base_application(text: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(text):
        if not starts_word_at(text, "log", index):
            result.append(text[index])
            index += 1
            continue

        after_name = skip_spaces(text, index + 3)
        if after_name >= len(text) or text[after_name] != "_":
            result.append(text[index])
            index += 1
            continue

        try:
            base, after_base = extract_log_base_token(text, after_name + 1)
            argument, after_argument = extract_log_argument_token(text, after_base)
        except GradingParseFailure:
            result.append(text[index])
            index += 1
            continue

        result.append(f"log({argument},{base})")
        index = after_argument
    return "".join(result)


def starts_word_at(text: str, word: str, index: int) -> bool:
    if not text.startswith(word, index):
        return False
    before = text[index - 1] if index > 0 else ""
    after_index = index + len(word)
    after = text[after_index] if after_index < len(text) else ""
    return not before.isalnum() and before != "_" and not after.isalnum()


def extract_log_base_token(text: str, index: int) -> tuple[str, int]:
    start = skip_spaces(text, index)
    if start >= len(text):
        raise GradingParseFailure("expected logarithm base")
    if text[start] == "(":
        return extract_parenthetical_token(text, start)

    end = start
    while end < len(text) and not text[end].isspace() and text[end] not in "()+-*/=,":
        end += 1
    if end == start:
        raise GradingParseFailure("expected logarithm base")
    return text[start:end], end


def extract_log_argument_token(text: str, index: int) -> tuple[str, int]:
    start = skip_spaces(text, index)
    if start >= len(text):
        raise GradingParseFailure("expected logarithm argument")
    if text[start] == "(":
        return extract_parenthetical_token(text, start)

    end = start
    while end < len(text) and not text[end].isspace() and text[end] not in "+-*/=,":
        end += 1
    if end == start:
        raise GradingParseFailure("expected logarithm argument")
    return text[start:end], end


def extract_parenthetical_token(text: str, opening_index: int) -> tuple[str, int]:
    if opening_index >= len(text) or text[opening_index] != "(":
        raise GradingParseFailure("expected a parenthetical group")
    depth = 0
    for index in range(opening_index, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[opening_index + 1:index], index + 1
    raise GradingParseFailure("unbalanced parentheses")


def normalize_bare_function_application(text: str) -> str:
    function_names = sorted(KNOWN_FUNCTIONS, key=len, reverse=True)
    function_pattern = "|".join(re.escape(name) for name in function_names)
    return re.sub(
        rf"\b({function_pattern})\s+([A-Za-z][A-Za-z0-9_]*|\d+(?:\.\d+)?)\b",
        r"\1(\2)",
        text,
    )


def normalize_spaced_implicit_multiplication(text: str) -> str:
    output: list[str] = []
    index = 0
    function_names = tuple(sorted(KNOWN_FUNCTIONS, key=len, reverse=True))
    while index < len(text):
        char = text[index]
        if not char.isspace():
            output.append(char)
            index += 1
            continue

        next_index = index
        while next_index < len(text) and text[next_index].isspace():
            next_index += 1

        previous = output[-1] if output else ""
        next_char = text[next_index] if next_index < len(text) else ""
        prefix = "".join(output)
        follows_function = next_char == "(" and any(
            re.search(rf"\b{re.escape(name)}$", prefix)
            for name in function_names
        )
        if (
            previous
            and next_char
            and re.match(r"[A-Za-z0-9)]", previous)
            and re.match(r"[A-Za-z(]", next_char)
            and not follows_function
        ):
            output.append("*")

        index = next_index
    return "".join(output)


def replace_structural_latex(text: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(text):
        if text.startswith(r"\frac", index):
            after_command = skip_spaces(text, index + 5)
            numerator, after_numerator = extract_group_or_compact_token(text, after_command)
            after_numerator = skip_spaces(text, after_numerator)
            denominator, after_denominator = extract_group_or_compact_token(text, after_numerator)
            result.append(
                "((" + replace_structural_latex(numerator) + ")/(" +
                replace_structural_latex(denominator) + "))"
            )
            index = after_denominator
            continue
        if text.startswith(r"\sqrt", index):
            after_command = skip_spaces(text, index + 5)
            if after_command < len(text) and text[after_command] == "[":
                _degree, after_degree = extract_bracket(text, after_command)
                after_command = skip_spaces(text, after_degree)
            radicand, after_radicand = extract_group(text, after_command)
            result.append("sqrt(" + replace_structural_latex(radicand) + ")")
            index = after_radicand
            continue
        result.append(text[index])
        index += 1
    return "".join(result)


def split_equation_text(text: str) -> Optional[tuple[str, str]]:
    raw = str(text or "").strip()
    if raw.count("=") != 1:
        return None
    left, right = raw.split("=", 1)
    if not left.strip() or not right.strip():
        return None
    return left, right


def split_top_level_equals(text: str) -> list[str]:
    raw = str(text or "").strip()
    if "=" not in raw:
        return []

    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(raw):
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth = max(0, depth - 1)
        elif char == "=" and depth == 0:
            parts.append(raw[start:index].strip())
            start = index + 1
    parts.append(raw[start:].strip())
    return parts


def split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def extract_group(text: str, opening_index: int) -> tuple[str, int]:
    if opening_index >= len(text) or text[opening_index] != "{":
        raise GradingParseFailure("expected a braced LaTeX group")
    depth = 0
    for index in range(opening_index, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening_index + 1:index], index + 1
    raise GradingParseFailure("unbalanced LaTeX braces")


def extract_group_or_compact_token(text: str, opening_index: int) -> tuple[str, int]:
    if opening_index < len(text) and text[opening_index] == "{":
        return extract_group(text, opening_index)
    if opening_index >= len(text):
        raise GradingParseFailure("expected a braced LaTeX group")

    if text[opening_index] == "\\":
        command = re.match(r"\\[A-Za-z]+", text[opening_index:])
        if command:
            end = opening_index + len(command.group(0))
            return text[opening_index:end], end
        if opening_index + 1 < len(text):
            return text[opening_index:opening_index + 2], opening_index + 2

    return text[opening_index], opening_index + 1


def extract_bracket(text: str, opening_index: int) -> tuple[str, int]:
    if opening_index >= len(text) or text[opening_index] != "[":
        raise GradingParseFailure("expected a bracketed LaTeX group")
    depth = 0
    for index in range(opening_index, len(text)):
        if text[index] == "[":
            depth += 1
        elif text[index] == "]":
            depth -= 1
            if depth == 0:
                return text[opening_index + 1:index], index + 1
    raise GradingParseFailure("unbalanced LaTeX brackets")


def skip_spaces(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def compact_answer_text(text: str) -> str:
    compact = str(text or "").strip().lower()
    compact = compact.replace("$", "")
    compact = compact.replace("\u2205", "emptyset")
    compact = compact.replace("\u221a", "sqrt")
    compact = compact.replace(r"\emptyset", "emptyset")
    compact = compact.replace(r"\varnothing", "varnothing")
    compact = re.sub(r"\\mathbb\s*\{\s*r\s*\}", "r", compact, flags=re.IGNORECASE)
    compact = compact.replace(r"\in", "in")
    compact = compact.replace("\u2208", "in")
    compact = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", compact)
    compact = re.sub(r"[^a-z0-9=+\-*/().,^]+", "", compact)
    return compact


def safe_float(value: sympy.Expr) -> Optional[float]:
    try:
        numeric = sympy.N(value, 16)
        if getattr(numeric, "is_real", None) is False:
            return None
        result = float(numeric)
    except Exception:
        return None
    if not math.isfinite(result):
        return None
    return result
