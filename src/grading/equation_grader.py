"""V1 symbolic grading for single-variable equation solving.

The grader has two layers:

* ``create_answer_manifest`` turns a user-provided equation into a small answer
  manifest backed by SymPy ``solveset``.
* ``grade_equation_work`` classifies OCR line candidates against that manifest.

V1 intentionally focuses on real-valued equations solved for one variable. It
handles finite, empty, and all-real solution sets directly, and returns an
``unsupported`` manifest for symbolic sets that need a richer future grader.
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


DEFAULT_TOLERANCE = 0.005
MAX_OCR_CANDIDATES = 5
SYMPY_GRADER_BUDGET_SECONDS = 0.5
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

KNOWN_FUNCTIONS = {
    "sqrt": sympy.sqrt,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "log": sympy.log,
    "ln": sympy.log,
    "abs": sympy.Abs,
}
KNOWN_CONSTANTS = {"pi": sympy.pi, "e": sympy.E}

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

    try:
        solution_set = sympy.solveset(parsed.residual, symbol, domain=sympy.S.Reals)
    except Exception as exc:
        return {**manifest, "error": f"solveset failed: {exc}"}

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

    return {
        **manifest,
        "solution_set_repr": sympy.sstr(solution_set),
        "error": "solution set is not finite, empty, or all real numbers",
    }


def grade_equation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Grade a JSON-style payload from the gateway/client boundary."""

    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        problem_raw = (
            payload.get("problem_raw") or
            payload.get("problemRaw") or
            payload.get("problemLatex") or
            payload.get("problem") or
            ""
        )
        variable = payload.get("variable")
        if variable is None and isinstance(payload.get("problemMetadata"), dict):
            variable = payload["problemMetadata"].get("solveVariable")
        manifest = create_answer_manifest(str(problem_raw), variable=variable)

    lines = payload.get("lines") or payload.get("studentLines") or []
    return grade_equation_work(manifest, lines)


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


def _pick_best_candidate_verdict(
    verdicts: Sequence[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], int]:
    """Select the most supported verdict from pre-computed per-candidate results.

    Priority: best classification (valid_step > invalid_step > other), then
    best solution_coverage when both are valid.

    Returns (best_verdict, original_candidate_index).
    """
    classification_rank = {"valid_step": 0, "invalid_step": 1, "other": 2}
    coverage_rank = {"full": 0, "partial": 1, "none": 2}
    best: Optional[dict[str, Any]] = None
    best_original_index = 0
    best_rank = (999, 999)

    for verdict in verdicts:
        cls = verdict.get("classification", "other")
        cov = verdict.get("solutionCoverage", "none")
        rank = (classification_rank.get(cls, 99), coverage_rank.get(cov, 99))
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

    first_parseable_equation: Optional[tuple[int, str]] = None
    first_latex = candidates[0] if candidates else ""

    for index, latex in enumerate(candidates[:MAX_OCR_CANDIDATES]):
        text = str(latex or "").strip()
        if not text:
            continue

        special = special_solution_match(text, manifest)
        if special is not None:
            return selected_line(
                text,
                "valid_step",
                index,
                special.coverage,
                special.matched_indices,
                accepted_special=special.accepted_special,
            )

        solution_match = finite_solution_match(text, manifest, exact_values, variable)
        if solution_match is not None:
            return selected_line(
                text,
                "valid_step",
                index,
                solution_match.coverage,
                solution_match.matched_indices,
            )

        if equation_equivalent(reference_latex, text, variable):
            return selected_line(text, "valid_step", index, "none", ())

        if first_parseable_equation is None and is_parseable_equation(text):
            first_parseable_equation = (index, text)

    if first_parseable_equation is not None:
        index, text = first_parseable_equation
        return selected_line(text, "invalid_step", index, "none", ())

    return selected_line(first_latex, "other", 0 if candidates else None, "none", ())


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
) -> dict[str, Any]:
    result = {
        "studentLatex": latex,
        "classification": classification,
        "selectedCandidateIndex": selected_candidate_index,
        "solutionCoverage": coverage,
        "matchedSolutions": [],
        "_matched_indices": tuple(sorted(set(matched_indices))),
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
    return False


def finite_solution_match(
    latex: str,
    manifest: dict[str, Any],
    exact_values: Sequence[sympy.Expr],
    variable: str,
) -> Optional[SolutionMatch]:
    if manifest.get("cardinality") != "finite" or not exact_values:
        return None

    values = candidate_solution_values(latex, variable)
    if not values:
        return None

    matched: set[int] = set()
    for candidate_value in values:
        for index, exact in enumerate(exact_values):
            if solution_values_equivalent(candidate_value, exact, manifest):
                matched.add(index)

    if not matched:
        return None
    coverage = "full" if len(matched) == len(exact_values) else "partial"
    return SolutionMatch(tuple(sorted(matched)), coverage)


def special_solution_match(latex: str, manifest: dict[str, Any]) -> Optional[SolutionMatch]:
    cardinality = manifest.get("cardinality")
    compact = compact_answer_text(latex)
    acceptable = {
        compact_answer_text(item)
        for item in manifest.get("acceptable_strings") or []
    }
    if compact in acceptable:
        if cardinality in {"none", "infinite"}:
            return SolutionMatch((-1,), "full", accepted_special=cardinality)
        if cardinality == "finite":
            return None

    if cardinality == "none" and compact in NO_SOLUTION_STRINGS:
        return SolutionMatch((-1,), "full", accepted_special="none")

    if cardinality == "infinite":
        if compact in INFINITE_SOLUTION_STRINGS or re.fullmatch(r"[a-z]+in(?:mathbb)?r", compact):
            return SolutionMatch((-1,), "full", accepted_special="infinite")
        if equation_identity(latex):
            return SolutionMatch((-1,), "full", accepted_special="identity")

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


def solution_sets_equal(a: sympy.Set, b: sympy.Set) -> bool:
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
    return False


def is_parseable_equation(latex: str) -> bool:
    try:
        return parse_math(latex).kind == "equation"
    except GradingParseFailure:
        return False


def candidate_solution_values(latex: str, variable: str) -> list[sympy.Expr]:
    text = str(latex or "").strip()
    if not text:
        return []

    values: list[sympy.Expr] = []
    for part in split_solution_parts(text, variable):
        for expanded in expand_pm(part):
            try:
                parsed = parse_expression(expanded)
            except GradingParseFailure:
                continue
            if parsed.free_symbols:
                continue
            values.append(parsed)
    return values


def split_solution_parts(latex: str, variable: str) -> list[str]:
    text = str(latex or "").strip()
    if not text:
        return []

    equation_match = split_equation_text(text)
    if equation_match is not None:
        left_text, right_text = equation_match
        left_compact = compact_answer_text(left_text)
        right_compact = compact_answer_text(right_text)
        if left_compact == compact_answer_text(variable):
            return split_top_level_commas(right_text)
        if right_compact == compact_answer_text(variable):
            return split_top_level_commas(left_text)

    return split_top_level_commas(text)


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


def parse_expression(text: str) -> sympy.Expr:
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
            evaluate=True,
        )
    except Exception as exc:
        raise GradingParseFailure(f"could not parse expression: {exc}") from exc
    if not isinstance(parsed, sympy.Expr):
        raise GradingParseFailure("not an expression")
    if parsed.has(sympy.zoo, sympy.oo, -sympy.oo, sympy.nan):
        raise GradingParseFailure("expression contains a non-finite value")
    return parsed


def normalize_math_text(text: str) -> str:
    output = str(text or "").strip()
    if not output:
        return ""

    output = output.replace("$", "")
    for token in (r"\left", r"\right", r"\limits", r"\!", r"\,", r"\;", r"\:"):
        output = output.replace(token, "")
    output = output.replace(r"\operatorname", "")
    output = re.sub(r"\\text\s*\{[^{}]*\}", "", output)
    output = re.sub(r"\\mathrm\s*\{\s*([A-Za-z])\s*\}", r"\1", output)
    output = re.sub(r"\\mathbb\s*\{\s*R\s*\}", "R", output)

    for old, new in LATEX_COMMAND_REPLACEMENTS.items():
        output = output.replace(old, new)
    for old, new in sorted(LATEX_VARIABLE_COMMANDS.items(), key=lambda item: len(item[0]), reverse=True):
        output = output.replace(old, new)

    output = replace_structural_latex(output)
    output = output.replace("{", "(").replace("}", ")")
    output = re.sub(r"\s+", "", output)
    if "\\" in output:
        raise GradingParseFailure("unsupported LaTeX command")
    return output


def replace_structural_latex(text: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(text):
        if text.startswith(r"\frac", index):
            after_command = skip_spaces(text, index + 5)
            numerator, after_numerator = extract_group(text, after_command)
            after_numerator = skip_spaces(text, after_numerator)
            denominator, after_denominator = extract_group(text, after_numerator)
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