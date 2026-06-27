"""Problem-aware LaTeX scoring helpers for OCR candidates.

This module intentionally does not know any worked solutions. It scores an OCR
candidate against the problem statement and previously recognized student lines
using parse soundness, character continuity, and symbolic equivalence when
SymPy can determine it.
"""

from __future__ import annotations

import contextlib
import re
import signal
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

ALLOWED_TEXT = re.compile(r"^[A-Za-z0-9_+*/^=().,{}\\-]+$")
MAX_EQUIVALENCE_OPS = 120
MAX_EQUIVALENCE_CHARS = 900
SYMPY_TIMEOUT_SECONDS = 0.35


class ParseFailure(ValueError):
    """Raised when a LaTeX candidate cannot be safely parsed."""


class SympyBudgetExceeded(TimeoutError):
    """Raised when a symbolic check exceeds the scoring budget."""


@dataclass(frozen=True)
class MathParse:
    kind: str
    left: sympy.Expr
    right: Optional[sympy.Expr] = None


@dataclass(frozen=True)
class EquivalenceResult:
    equivalent: Optional[bool]
    method: str
    detail: str = ""


@dataclass(frozen=True)
class CandidateScore:
    latex: str
    score: float
    sound: bool
    equivalent_to_problem: bool
    equivalent_to_previous: bool
    detail: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "latex": self.latex,
            "score": self.score,
            "sound": self.sound,
            "equivalentToProblem": self.equivalent_to_problem,
            "equivalentToPrevious": self.equivalent_to_previous,
            "detail": self.detail,
        }


def sympy_pm(a, b):
    return sympy.FiniteSet(a - b, a + b)


def sympy_eval_at(expr, lower, upper):
    symbols = sorted(expr.free_symbols, key=lambda item: item.name)
    variable = symbols[0] if symbols else sympy.Symbol("x", real=True)
    return sympy.simplify(expr.subs(variable, upper) - expr.subs(variable, lower))


KNOWN_FUNCTIONS = {
    "Integral": sympy.Integral,
    "evalat": sympy_eval_at,
    "sqrt": sympy.sqrt,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "log": sympy.log,
    "ln": sympy.log,
    "exp": sympy.exp,
    "abs": sympy.Abs,
    "pm": sympy_pm,
}
KNOWN_CONSTANTS = {"pi": sympy.pi, "e": sympy.E}
LATEX_FUNCTION_COMMANDS = {
    r"\log": "log",
    r"\ln": "ln",
    r"\sin": "sin",
    r"\cos": "cos",
    r"\tan": "tan",
    r"\exp": "exp",
}
LATEX_VARIABLE_COMMANDS = {
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma",
    r"\delta": "delta",
    r"\epsilon": "epsilon",
    r"\varepsilon": "varepsilon",
    r"\zeta": "zeta",
    r"\eta": "eta",
    r"\theta": "theta",
    r"\vartheta": "vartheta",
    r"\iota": "iota",
    r"\kappa": "kappa",
    r"\lambda": "lambda",
    r"\mu": "mu",
    r"\nu": "nu",
    r"\xi": "xi",
    r"\rho": "rho",
    r"\varrho": "varrho",
    r"\sigma": "sigma",
    r"\varsigma": "varsigma",
    r"\tau": "tau",
    r"\upsilon": "upsilon",
    r"\phi": "phi",
    r"\varphi": "varphi",
    r"\chi": "chi",
    r"\psi": "psi",
    r"\omega": "omega",
}


def _extract_group(text: str, opening_index: int) -> tuple[str, int]:
    if opening_index >= len(text) or text[opening_index] != "{":
        raise ParseFailure("expected a braced LaTeX group")
    depth = 0
    for index in range(opening_index, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening_index + 1:index], index + 1
    raise ParseFailure("unbalanced LaTeX braces")


def _extract_parenthesized(text: str, opening_index: int) -> tuple[str, int]:
    if opening_index >= len(text) or text[opening_index] != "(":
        raise ParseFailure("expected a parenthesized group")
    depth = 0
    for index in range(opening_index, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[opening_index + 1:index], index + 1
    raise ParseFailure("unbalanced parentheses")


def _extract_bracketed(text: str, opening_index: int) -> tuple[str, int]:
    if opening_index >= len(text) or text[opening_index] != "[":
        raise ParseFailure("expected a bracketed group")
    depth = 0
    for index in range(opening_index, len(text)):
        if text[index] == "[":
            depth += 1
        elif text[index] == "]":
            depth -= 1
            if depth == 0:
                return text[opening_index + 1:index], index + 1
    raise ParseFailure("unbalanced brackets")


def _read_script_value(text: str, index: int) -> tuple[Optional[str], int]:
    if index >= len(text) or text[index] != "_":
        return None, index
    index += 1
    if index < len(text) and text[index] == "{":
        value, after_value = _extract_group(text, index)
        return _replace_structural_latex(value), after_value
    if index < len(text):
        return _replace_structural_latex(text[index]), index + 1
    raise ParseFailure("missing LaTeX subscript value")


def _read_superscript_value(text: str, index: int) -> tuple[Optional[str], int]:
    if index >= len(text) or text[index] != "^":
        return None, index
    index += 1
    if index < len(text) and text[index] == "{":
        value, after_value = _extract_group(text, index)
        return _replace_structural_latex(value), after_value
    if index < len(text):
        return _replace_structural_latex(text[index]), index + 1
    raise ParseFailure("missing LaTeX superscript value")


def _read_function_argument(text: str, index: int) -> tuple[Optional[str], int]:
    if index >= len(text):
        return None, index
    if text[index] == "(":
        value, after_value = _extract_parenthesized(text, index)
        return _replace_structural_latex(value), after_value
    if text[index] == "{":
        value, after_value = _extract_group(text, index)
        return _replace_structural_latex(value), after_value
    return None, index


def _read_integral_body(text: str, index: int) -> tuple[str, str, int]:
    depth = 0
    for position in range(index, len(text) - 1):
        char = text[position]
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth = max(0, depth - 1)
        if depth == 0 and char == "d" and text[position + 1].isalpha():
            variable_start = position + 1
            variable_end = variable_start + 1
            while variable_end < len(text) and text[variable_end].isalpha():
                variable_end += 1
            if variable_end < len(text) and text[variable_end] not in "+-=,)":
                continue
            integrand = text[index:position]
            if not integrand:
                raise ParseFailure("integral missing integrand")
            return (
                _replace_structural_latex(integrand),
                text[variable_start:variable_end],
                variable_end,
            )
    raise ParseFailure("integral missing differential")


def _replace_structural_latex(text: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(text):
        matched_function = next(
            (
                (command, name)
                for command, name in LATEX_FUNCTION_COMMANDS.items()
                if text.startswith(command, index)
            ),
            None,
        )
        if matched_function is not None:
            command, name = matched_function
            after_command = index + len(command)
            base, after_base = _read_script_value(text, after_command)
            argument, after_argument = _read_function_argument(text, after_base)
            if argument is None:
                output.append(name)
                index = after_base
            elif base is not None and name == "log":
                output.append(f"log({_replace_structural_latex(argument)},{base})")
                index = after_argument
            else:
                output.append(f"{name}({_replace_structural_latex(argument)})")
                index = after_argument
            continue
        if text.startswith(r"\int", index):
            after_command = index + 4
            lower, after_lower = _read_script_value(text, after_command)
            upper, after_upper = _read_superscript_value(text, after_lower)
            integrand, variable, after_body = _read_integral_body(text, after_upper)
            if lower is not None and upper is not None:
                output.append(f"Integral({integrand},({variable},{lower},{upper}))")
            else:
                output.append(f"Integral({integrand},{variable})")
            index = after_body
            continue
        if text.startswith(r"\frac", index):
            numerator, after_numerator = _extract_group(text, index + 5)
            denominator, after_denominator = _extract_group(text, after_numerator)
            output.append(
                "((" + _replace_structural_latex(numerator) + ")/(" +
                _replace_structural_latex(denominator) + "))"
            )
            index = after_denominator
            continue
        if text.startswith(r"\sqrt", index):
            radicand, after_radicand = _extract_group(text, index + 5)
            output.append("sqrt(" + _replace_structural_latex(radicand) + ")")
            index = after_radicand
            continue
        if text.startswith("^{", index):
            exponent, after_exponent = _extract_group(text, index + 1)
            output.append("^(" + _replace_structural_latex(exponent) + ")")
            index = after_exponent
            continue
        if text[index] == "[":
            expression, after_expression = _extract_bracketed(text, index)
            lower, after_lower = _read_script_value(text, after_expression)
            upper, after_upper = _read_superscript_value(text, after_lower)
            if lower is not None and upper is not None:
                output.append(f"evalat({_replace_structural_latex(expression)},{lower},{upper})")
                index = after_upper
                continue
            output.append("(" + _replace_structural_latex(expression) + ")")
            index = after_expression
            continue
        output.append(text[index])
        index += 1
    return "".join(output)


def _insert_pm_commas(text: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(text):
        if text.startswith("pm(", index):
            start = index + 3
            depth = 1
            for j in range(start, len(text)):
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                    if depth == 0:
                        inner = text[start:j]
                        break
            else:
                raise ParseFailure("unbalanced parentheses in pm(...)")
            parts = inner.rsplit("-", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                result.append(f"pm({parts[0]},{parts[1]})")
                index = j + 1
                continue
        result.append(text[index])
        index += 1
    return "".join(result)


def normalize_latex(latex: str) -> str:
    if not isinstance(latex, str) or not latex.strip():
        raise ParseFailure("empty candidate")
    text = latex.strip()
    text = re.sub(r"([A-Za-z])\s*(?:'\s*|\^\s*\{\s*\\prime\s*\})", r"\1prime", text)
    for token in (r"\left", r"\right", r"\!", r"\,", r"\;", r"\:"):
        text = text.replace(token, "")
    replacements = {
        r"\cdot": "*",
        r"\times": "*",
        r"\div": "/",
        r"\pi": "pi",
        r"\mathrm{e}": "e",
        r"\operatorname": "",
        r"\pm": "pm",
        "−": "-",
        "×": "*",
        "÷": "/",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    for old, new in sorted(LATEX_VARIABLE_COMMANDS.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(old, new)
    text = re.sub(r"\s+", "", text)
    text = _replace_structural_latex(text)
    text = text.replace("{", "(").replace("}", ")")
    text = _insert_pm_commas(text)
    if "\\" in text or not ALLOWED_TEXT.fullmatch(text):
        raise ParseFailure("unsupported LaTeX remains after normalization")
    return text


def parse_math(latex: str) -> MathParse:
    normalized = normalize_latex(latex)
    operation = parse_operation_annotation(normalized)
    if operation is not None:
        return operation
    if normalized.startswith("="):
        if normalized.count("=") > 1 or len(normalized) == 1:
            raise ParseFailure("equation missing one side")
        return MathParse("expression", _parse_expression(normalized[1:]), None)
    if normalized.count("=") > 1:
        raise ParseFailure("multiple equals signs")
    if "=" in normalized:
        left, right = normalized.split("=", 1)
        if not left or not right:
            raise ParseFailure("equation missing one side")
        return MathParse("equation", _parse_expression(left), _parse_expression(right))
    return MathParse("expression", _parse_expression(normalized), None)


def parse_operation_annotation(normalized: str) -> Optional[MathParse]:
    match = re.fullmatch(r"([+\-*/])(.+)\1(.+)", normalized)
    if not match:
        return None
    operator, left_operand, right_operand = match.groups()
    if not left_operand or not right_operand:
        raise ParseFailure("operation annotation missing operand")
    left = _parse_expression(left_operand)
    right = _parse_expression(right_operand)
    if operator in {"/", "*"} and (_zero(left) or _zero(right)):
        raise ParseFailure("operation annotation divides or multiplies by zero")
    return MathParse("operation", left, right)


def _parse_expression(text: str) -> sympy.Expr:
    identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    local_dict: dict[str, Any] = {**KNOWN_FUNCTIONS, **KNOWN_CONSTANTS}
    for name in identifiers:
        if name not in local_dict:
            local_dict[name] = sympy.Symbol(name, real=True)
    try:
        parsed = parse_expr(
            text,
            local_dict=local_dict,
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as exc:
        raise ParseFailure(f"could not parse expression: {exc}") from exc
    if not isinstance(parsed, sympy.Expr):
        raise ParseFailure("not an expression")
    _reject_nonfinite_expression(parsed)
    return parsed


def _reject_nonfinite_expression(expression: sympy.Expr) -> None:
    if expression.has(sympy.zoo, sympy.oo, -sympy.oo, sympy.nan):
        raise ParseFailure("expression contains a non-finite value")
    if expression.is_finite is False:
        raise ParseFailure("expression is not finite")


def is_sound_latex(latex: str) -> bool:
    try:
        parse_math(latex)
        return True
    except ParseFailure:
        return False


def check_equivalence(reference_latex: str, candidate_latex: str) -> EquivalenceResult:
    evaluation_result = check_function_evaluation_equivalence(reference_latex, candidate_latex)
    if evaluation_result.equivalent is not None:
        return evaluation_result

    try:
        reference = parse_math(reference_latex)
        candidate = parse_math(candidate_latex)
    except ParseFailure as exc:
        return EquivalenceResult(None, "parse_failure", str(exc))

    if reference.kind != candidate.kind:
        return EquivalenceResult(False, "kind_mismatch")

    if reference.kind == "expression":
        difference = _zero_or_unknown(reference.left - candidate.left)
        if difference is None:
            return EquivalenceResult(None, "expression_difference_budget")
        return EquivalenceResult(difference, "expression_difference")

    assert reference.right is not None and candidate.right is not None
    return _equivalent_equations(reference.left - reference.right, candidate.left - candidate.right)


def _zero(expr: sympy.Expr) -> bool:
    return _zero_or_unknown(expr) is True


def _zero_or_unknown(expr: sympy.Expr) -> Optional[bool]:
    if expr == 0 or expr.is_zero is True:
        return True
    if expression_exceeds_budget(expr):
        return None
    try:
        with sympy_budget(SYMPY_TIMEOUT_SECONDS):
            simplified = sympy.simplify(expr.doit())
    except (Exception, SympyBudgetExceeded):
        return None
    return simplified == 0 or simplified.is_zero is True


def _equivalent_equations(reference_residual: sympy.Expr, candidate_residual: sympy.Expr) -> EquivalenceResult:
    difference = _zero_or_unknown(reference_residual - candidate_residual)
    if difference is True:
        return EquivalenceResult(True, "residual_difference")
    sum_difference = _zero_or_unknown(reference_residual + candidate_residual)
    if sum_difference is True:
        return EquivalenceResult(True, "residual_difference")
    if difference is None or sum_difference is None:
        return EquivalenceResult(None, "residual_difference_budget")

    try:
        if expressions_exceed_budget(reference_residual, candidate_residual):
            return EquivalenceResult(None, "constant_residual_factor_budget")
        with sympy_budget(SYMPY_TIMEOUT_SECONDS):
            ratio = sympy.simplify(reference_residual / candidate_residual)
        if ratio != 0 and not ratio.free_symbols and ratio.is_finite is not False:
            return EquivalenceResult(True, "constant_residual_factor")
    except (Exception, SympyBudgetExceeded):
        pass

    symbols = sorted(reference_residual.free_symbols | candidate_residual.free_symbols, key=lambda item: item.name)
    if len(symbols) == 1:
        try:
            if expressions_exceed_budget(reference_residual, candidate_residual):
                return EquivalenceResult(None, "real_solution_set_budget")
            with sympy_budget(SYMPY_TIMEOUT_SECONDS):
                reference_set = sympy.solveset(reference_residual, symbols[0], domain=sympy.S.Reals)
                candidate_set = sympy.solveset(candidate_residual, symbols[0], domain=sympy.S.Reals)
            if reference_set == candidate_set:
                return EquivalenceResult(True, "real_solution_set")
            if not isinstance(reference_set, sympy.ConditionSet) and not isinstance(candidate_set, sympy.ConditionSet):
                return EquivalenceResult(False, "real_solution_set")
        except (Exception, SympyBudgetExceeded):
            pass

    return EquivalenceResult(False, "symbolic_equation")


def expressions_exceed_budget(*expressions: sympy.Expr) -> bool:
    return any(expression_exceeds_budget(expression) for expression in expressions)


def expression_exceeds_budget(expression: sympy.Expr) -> bool:
    try:
        if sympy.count_ops(expression, visual=False) > MAX_EQUIVALENCE_OPS:
            return True
        if len(str(expression)) > MAX_EQUIVALENCE_CHARS:
            return True
    except Exception:
        return True
    return False


@contextlib.contextmanager
def sympy_budget(seconds: float):
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        yield
        return

    def raise_timeout(_signum, _frame):
        raise SympyBudgetExceeded("symbolic check timed out")

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


def check_function_evaluation_equivalence(reference_latex: str, candidate_latex: str) -> EquivalenceResult:
    try:
        reference = normalize_latex(reference_latex)
        candidate = normalize_latex(candidate_latex)
    except ParseFailure as exc:
        return EquivalenceResult(None, "evaluation_parse_failure", str(exc))

    pattern = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\(([^()]+)\)=(.+)")
    reference_match = pattern.fullmatch(reference)
    candidate_match = pattern.fullmatch(candidate)
    if not reference_match or not candidate_match:
        return check_function_derivative_equivalence(reference, candidate)

    reference_name, reference_arg, reference_rhs = reference_match.groups()
    candidate_name, candidate_arg, candidate_rhs = candidate_match.groups()
    if reference_name != candidate_name:
        derivative_result = check_function_derivative_equivalence(reference, candidate)
        if derivative_result.equivalent is not None:
            return derivative_result
        return EquivalenceResult(False, "function_name_mismatch")

    try:
        variable_expr = _parse_expression(reference_arg)
        if not isinstance(variable_expr, sympy.Symbol):
            return EquivalenceResult(None, "reference_argument_not_symbol")
        value = _parse_expression(candidate_arg)
        expected_rhs = _parse_expression(reference_rhs).subs(variable_expr, value)
        actual_rhs = _parse_expression(candidate_rhs)
    except ParseFailure as exc:
        return EquivalenceResult(None, "evaluation_parse_failure", str(exc))

    return EquivalenceResult(
        _zero(expected_rhs - actual_rhs),
        "function_evaluation_substitution",
    )


def check_function_derivative_equivalence(reference: str, candidate: str) -> EquivalenceResult:
    pattern = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\(([^()]+)\)=(.+)")
    reference_match = pattern.fullmatch(reference)
    candidate_match = pattern.fullmatch(candidate)
    if not reference_match or not candidate_match:
        return EquivalenceResult(None, "not_function_relation")

    reference_name, reference_arg, reference_rhs = reference_match.groups()
    candidate_name, candidate_arg, candidate_rhs = candidate_match.groups()
    if candidate_name != f"{reference_name}prime":
        return EquivalenceResult(None, "not_function_derivative")

    try:
        variable_expr = _parse_expression(reference_arg)
        candidate_variable = _parse_expression(candidate_arg)
        if not isinstance(variable_expr, sympy.Symbol) or candidate_variable != variable_expr:
            return EquivalenceResult(None, "derivative_variable_mismatch")
        expected_rhs = sympy.diff(_parse_expression(reference_rhs), variable_expr)
        actual_rhs = _parse_expression(candidate_rhs)
    except ParseFailure as exc:
        return EquivalenceResult(None, "derivative_parse_failure", str(exc))

    return EquivalenceResult(
        _zero(expected_rhs - actual_rhs),
        "function_derivative",
    )


def score_latex_candidate(
    latex: str,
    *,
    problem_latex: str,
    previous_latex: Sequence[str] = (),
    model_score: Optional[float] = None,
    elapsed_seconds: Optional[float] = None,
) -> CandidateScore:
    score = 0.0
    detail: dict[str, Any] = {}

    if model_score is not None:
        score += max(-20.0, min(2.0, float(model_score))) * 0.35
        detail["modelScore"] = model_score

    try:
        parsed = parse_math(latex)
        sound = True
        parsed_kind = parsed.kind
        score += 1.2
        if parsed.kind == "operation":
            score += 0.45
            detail["operationAnnotation"] = True
    except ParseFailure as exc:
        sound = False
        parsed_kind = ""
        score -= 2.5
        detail["parseFailure"] = str(exc)

    problem_result = check_equivalence(problem_latex, latex)
    equivalent_to_problem = problem_result.equivalent is True
    if equivalent_to_problem:
        score += 2.2
    elif problem_result.equivalent is None:
        score -= 0.25
    detail["problemEquivalence"] = problem_result.method

    equivalent_to_previous = False
    previous_equivalence_method = None
    for previous in previous_latex:
        previous_result = check_equivalence(previous, latex)
        if previous_result.equivalent is True:
            equivalent_to_previous = True
            previous_equivalence_method = previous_result.method
            score += 2.4 if previous_result.method in {
                "function_evaluation_substitution",
                "function_derivative",
            } else 1.1
            break
    detail["equivalentToPrevious"] = equivalent_to_previous
    if previous_equivalence_method:
        detail["previousEquivalence"] = previous_equivalence_method
    if parsed_kind in {"equation", "expression"} and exact_previous_latex_match(latex, previous_latex):
        score -= 1.25
        detail["duplicatePreviousLatex"] = True

    overlap = character_overlap(problem_latex, *previous_latex, candidate=latex)
    score += overlap * 0.9
    detail["characterOverlap"] = round(overlap, 3)

    if re.search(r"\\frac\s*\{\s*[0-9+\-*/\s]+[+\-*/][0-9+\-*/\s]*\s*\}", latex):
        score -= 0.35
        detail["unreducedNumericFractionArithmetic"] = True

    if elapsed_seconds is not None:
        elapsed = float(elapsed_seconds)
        detail["elapsedSeconds"] = elapsed
        if elapsed > 3.5:
            score -= min(2.0, (elapsed - 3.5) * 0.18)

    return CandidateScore(
        latex=latex,
        score=round(score, 4),
        sound=sound,
        equivalent_to_problem=equivalent_to_problem,
        equivalent_to_previous=equivalent_to_previous,
        detail=detail,
    )


def exact_previous_latex_match(latex: str, previous_latex: Sequence[str]) -> bool:
    current = compact_latex_text(latex)
    if not current:
        return False
    return any(compact_latex_text(previous) == current for previous in previous_latex)


def compact_latex_text(latex: str) -> str:
    return re.sub(r"\s+", "", str(latex or "")).strip()


def score_ocr_predictions(
    predictions: Iterable[dict[str, Any] | str],
    *,
    problem_latex: str,
    previous_latex: Sequence[str] = (),
) -> list[CandidateScore]:
    scored: list[CandidateScore] = []
    variants_by_latex: dict[str, dict[str, Any] | str] = {}
    for prediction in predictions:
        for variant in prediction_variants(
            prediction,
            problem_latex=problem_latex,
            previous_latex=previous_latex,
        ):
            latex = variant_latex(variant)
            if not latex:
                continue
            existing = variants_by_latex.get(latex)
            if existing is None or variant_preferred_over(variant, existing):
                variants_by_latex[latex] = variant

    for variant in variants_by_latex.values():
        if isinstance(variant, str):
            scored.append(score_latex_candidate(variant, problem_latex=problem_latex, previous_latex=previous_latex))
            continue
        candidate_score = score_latex_candidate(
            str(variant.get("latex", "")),
            problem_latex=problem_latex,
            previous_latex=previous_latex,
            model_score=variant.get("score"),
            elapsed_seconds=variant.get("elapsedSeconds"),
        )
        if variant.get("repairedFrom"):
            repair_adjustment = 0.0 if variant.get("repair") == "parenthesized_unit_product" else -0.15
            candidate_score = CandidateScore(
                latex=candidate_score.latex,
                score=round(candidate_score.score + repair_adjustment, 4),
                sound=candidate_score.sound,
                equivalent_to_problem=candidate_score.equivalent_to_problem,
                equivalent_to_previous=candidate_score.equivalent_to_previous,
                detail={
                    **candidate_score.detail,
                    "repairedFrom": variant["repairedFrom"],
                    "repair": variant.get("repair"),
                },
            )
        scored.append(candidate_score)
    return sorted(scored, key=lambda item: item.score, reverse=True)


def variant_latex(variant: dict[str, Any] | str) -> str:
    if isinstance(variant, str):
        return variant.strip()
    return str(variant.get("latex") or "").strip()


def variant_preferred_over(candidate: dict[str, Any] | str, existing: dict[str, Any] | str) -> bool:
    candidate_score = variant_model_score(candidate)
    existing_score = variant_model_score(existing)
    if candidate_score != existing_score:
        return candidate_score > existing_score
    if variant_is_repaired(candidate) != variant_is_repaired(existing):
        return not variant_is_repaired(candidate)
    return False


def variant_model_score(variant: dict[str, Any] | str) -> float:
    if isinstance(variant, str):
        return 0.0
    try:
        return float(variant.get("score"))
    except (TypeError, ValueError):
        return 0.0


def variant_is_repaired(variant: dict[str, Any] | str) -> bool:
    return isinstance(variant, dict) and bool(variant.get("repairedFrom"))


def prediction_variants(
    prediction: dict[str, Any] | str,
    *,
    problem_latex: str = "",
    previous_latex: Sequence[str] = (),
) -> list[dict[str, Any] | str]:
    variants: list[dict[str, Any] | str] = [prediction]
    latex = prediction if isinstance(prediction, str) else str(prediction.get("latex", ""))
    repairs = [
        (repair_redundant_trailing_braces(latex), "redundant_trailing_brace", 0.2),
        (repair_parenthesized_unit_products(latex), "parenthesized_unit_product", -0.05),
        (repair_trailing_one_as_parenthesis(latex), "trailing_one_to_parenthesis", 0.6),
        (repair_malformed_arithmetic_continuation(latex), "malformed_arithmetic_continuation", 0.8),
        (repair_symbolic_numeric_continuation(latex, previous_latex), "symbolic_numeric_continuation", 0.95),
    ]
    for repaired in repair_numeric_fraction_arithmetic(latex, problem_latex, previous_latex):
        repairs.append((repaired, "numeric_fraction_arithmetic", 0.05))
    for repaired in repair_contextual_greek_variables(latex, problem_latex, previous_latex):
        repairs.append((repaired, "contextual_greek_variable", 0.9))
    for repaired in repair_contextual_numeric_lookalikes(latex, problem_latex, previous_latex):
        repairs.append((repaired, "contextual_numeric_lookalike", 0.85))
    for repaired in repair_contextual_latex_numeric_equivalence(latex, problem_latex, previous_latex):
        repairs.append((repaired, "contextual_latex_numeric_equivalence", 0.55))
    for repaired in repair_contextual_malformed_log_bases(latex, problem_latex, previous_latex):
        repairs.append((repaired, "contextual_malformed_log_base", 0.35))
    for repaired in repair_contextual_log_numeric_arguments(latex, problem_latex, previous_latex):
        repairs.append((repaired, "contextual_log_numeric_argument", 0.45))
    for repaired in repair_contextual_exponent_equals_confusion(
        re.sub(r"\s+", " ", str(latex or "")).strip(),
        contextual_variable_names([problem_latex, *(previous_latex or [])]),
    ):
        if numeric_repair_has_semantic_support(repaired, problem_latex, previous_latex):
            repairs.append((repaired, "contextual_exponent_confusion", 0.1))
    for repaired in repair_from_contextual_denominator_clear_step(latex, problem_latex, previous_latex):
        repairs.append((repaired, "contextual_denominator_clear", 0.1))
        normalized_repaired = repair_parenthesized_unit_products(repaired)
        if normalized_repaired:
            repairs.append((normalized_repaired, "contextual_denominator_clear", 0.05))
    for repaired in repair_from_previous_linear_simplification(latex, problem_latex, previous_latex):
        repairs.append((repaired, "contextual_linear_simplification", 0.2))
    for repaired in repair_contextual_symbol_confusions(latex, problem_latex, previous_latex):
        repairs.append((repaired, "contextual_symbol_confusion", 1.05))
    for repaired, repair_name, score_penalty in repairs:
        if not repaired or repaired == latex:
            continue
        if isinstance(prediction, str):
            variants.append(repaired)
        else:
            variants.append({
                **prediction,
                "latex": repaired,
                "score": float(prediction.get("score") or 0) - score_penalty,
                "repairedFrom": latex,
                "repair": repair_name,
            })
    return variants


def repair_symbolic_numeric_continuation(latex: str, previous_latex: Sequence[str]) -> Optional[str]:
    text = re.sub(r"\s+", "", str(latex or ""))
    if not text or "\\" in text.replace(r"\infty", ""):
        return None
    if len(text) > 8 or not re.search(r"[A-Za-z]|\\infty", text):
        return None
    if re.search(r"[+\-*/^(){}\[\]]", text):
        return None
    if "=" in text and text.count("=") > 1:
        return None

    value = previous_numeric_value(previous_latex)
    if value is None:
        return None
    repaired = f"= {sympy_value_to_latex(value)}"
    if check_equivalence(previous_latex[-1], repaired).equivalent is not True:
        return None
    return repaired


def repair_redundant_trailing_braces(latex: str) -> Optional[str]:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    if not text.endswith("}"):
        return None
    if is_sound_latex(text):
        return None
    candidate = text
    for _ in range(3):
        candidate = candidate.rstrip()
        if not candidate.endswith("}"):
            return None
        candidate = candidate[:-1].rstrip()
        if candidate == text:
            return None
        if is_sound_latex(candidate):
            return candidate
    return None


def repair_parenthesized_unit_products(latex: str) -> Optional[str]:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    if not text or "(" not in text:
        return None
    candidate = text
    candidate = re.sub(
        r"(?<![A-Za-z])(\d+)\s*\(\s*([A-Za-z])\s*\)",
        lambda match: f"{match.group(1)} {match.group(2)}",
        candidate,
    )
    candidate = re.sub(
        r"(?<![A-Za-z])(\d+)\s*\(\s*1\s*\)",
        lambda match: match.group(1),
        candidate,
    )
    candidate = normalize_algebra_spacing(candidate)
    if candidate == text:
        return None
    try:
        parse_math(candidate)
    except ParseFailure:
        return None
    return candidate


def repair_contextual_greek_variables(
    latex: str,
    problem_latex: str,
    previous_latex: Sequence[str],
) -> list[str]:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    if not text or "\\" in text:
        return []

    commands = contextual_greek_variable_commands([problem_latex, *(previous_latex or [])])
    if not commands:
        return []

    variants: list[str] = []
    for command in commands[:2]:
        for lookalike in greek_variable_lookalikes(command):
            candidate = replace_latin_variable_token(text, lookalike, command)
            if candidate != text:
                variants.append(candidate)
    return unique_preserving_order(variants)


def contextual_greek_variable_commands(latex_values: Sequence[str]) -> list[str]:
    commands: list[str] = []
    for latex in latex_values:
        for command in LATEX_VARIABLE_COMMANDS:
            if command in str(latex or "") and command not in commands:
                commands.append(command)
    return commands


def greek_variable_lookalikes(command: str) -> list[str]:
    return {
        r"\eta": ["n", "z"],
        r"\mu": ["u"],
        r"\nu": ["v"],
        r"\rho": ["p"],
        r"\theta": ["o"],
        r"\alpha": ["a"],
        r"\beta": ["b"],
    }.get(command, [])


def replace_latin_variable_token(latex: str, token: str, replacement: str) -> str:
    return re.sub(rf"(?<!\\)\b{re.escape(token)}\b", lambda _match: replacement, latex)


def repair_contextual_numeric_lookalikes(
    latex: str,
    problem_latex: str,
    previous_latex: Sequence[str],
) -> list[str]:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    if not text or "=" not in text:
        return []

    context = [problem_latex, *(previous_latex or [])]
    replacements = contextual_numeric_replacement_options(text, context)
    if not replacements:
        return []

    variants = {text}
    for pattern, replacement in replacements[:8]:
        next_variants = set(variants)
        for variant in variants:
            candidate = pattern.sub(lambda _match, value=replacement: value, variant)
            if candidate != variant:
                next_variants.add(candidate)
        variants = set(list(next_variants)[:24])

    repaired: list[str] = []
    for candidate in variants:
        if candidate == text:
            continue
        if not numeric_repair_has_semantic_support(candidate, problem_latex, previous_latex):
            continue
        repaired.append(candidate)
    return unique_preserving_order(repaired)


def contextual_numeric_replacement_options(
    latex: str,
    context_latex: Sequence[str],
) -> list[tuple[re.Pattern[str], str]]:
    numbers = contextual_number_tokens(context_latex)
    if not numbers:
        return []

    variables = set(contextual_variable_names(context_latex))
    options: list[tuple[re.Pattern[str], str]] = []
    if r"\infty" in latex:
        for number in numbers:
            if number in {"2", "8"}:
                options.append((re.compile(r"\\infty\b"), number))
    if r"\alpha" in latex and not context_contains_command(context_latex, r"\alpha"):
        for number in numbers:
            if number == "2":
                options.append((re.compile(r"\\alpha\b"), number))
                if "c" not in variables:
                    for variable in variables:
                        options.append((re.compile(r"(?<!\\)\bc\s*\\alpha\b"), f"{number} {variable}"))
                        options.append((re.compile(r"\\alpha\s*(?<!\\)\bc\b"), f"{number} {variable}"))
    if "c" not in variables and re.search(r"(?<!\\)\bc\s+[A-Za-z]\b", latex):
        for number in numbers:
            if number == "2":
                for variable in variables:
                    options.append((re.compile(rf"(?<!\\)\bc\s+(?={re.escape(variable)}\b)"), f"{number} "))
    if re.search(r"(?<!\\)\bn\b", latex) and "n" not in variables:
        for number in numbers:
            if re.fullmatch(r"1{2,}", number):
                options.append((re.compile(r"(?<!\\)\bn\b"), number))
    if re.search(r"(?<!\\)\b[oOsS]\b", latex):
        for number in numbers:
            if number == "0":
                options.append((re.compile(r"(?<!\\)\b[oOsS]\b"), number))
    if re.search(r"(?<!\\)\bi\d+\b", latex) and "i" not in variables:
        for number in numbers:
            options.append((re.compile(rf"(?<!\\)\bi(?={re.escape(number)}\b)"), ""))
    return options


def contextual_number_tokens(latex_values: Sequence[str]) -> list[str]:
    numbers: list[str] = []
    for latex in latex_values:
        text = str(latex or "")
        extracted = [
            "".join(re.findall(r"\d", match.group(0)))
            for match in re.finditer(r"(?<![A-Za-z])\d(?:\s+\d){1,}(?![A-Za-z])", text)
        ]
        extracted.extend(re.findall(r"(?<![A-Za-z])\d+(?![A-Za-z])", text))
        for number in extracted:
            if number not in numbers:
                numbers.append(number)
    return numbers[:12]


def context_contains_command(latex_values: Sequence[str], command: str) -> bool:
    return any(command in str(latex or "") for latex in latex_values)


def numeric_repair_has_semantic_support(
    candidate: str,
    problem_latex: str,
    previous_latex: Sequence[str],
) -> bool:
    try:
        parse_math(candidate)
    except ParseFailure:
        return False
    references = [problem_latex, *(previous_latex or [])]
    for reference in references:
        if not str(reference or "").strip():
            continue
        if check_equivalence(reference, candidate).equivalent is True:
            return True
    return False


def repair_contextual_numeric_equivalence(
    latex: str,
    problem_latex: str,
    previous_latex: Sequence[str],
) -> list[str]:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    numbers = contextual_number_tokens([problem_latex, *(previous_latex or [])])
    if not text or not numbers:
        return []
    if len(text) > 96 or "\\" in text or "{" in text or "}" in text:
        return []
    if text.count("=") != 1:
        return []

    matches = list(re.finditer(r"(?<![A-Za-z])\d+(?![A-Za-z])", text))
    variants: list[str] = []
    for match in matches[:8]:
        original = match.group(0)
        for number in numbers:
            if number == original:
                continue
            candidate = f"{text[:match.start()]}{number}{text[match.end():]}"
            if candidate == text or candidate in variants:
                continue
            if numeric_repair_has_semantic_support(candidate, problem_latex, previous_latex):
                variants.append(candidate)
            if len(variants) >= 12:
                return variants
    return variants


def repair_contextual_log_numeric_arguments(
    latex: str,
    problem_latex: str,
    previous_latex: Sequence[str],
) -> list[str]:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    if r"\log" not in text:
        return []
    numbers = contextual_number_tokens([problem_latex, *(previous_latex or [])])
    if not numbers:
        return []

    pattern = re.compile(
        r"(\\log\s*_\s*\{[^{}]+\}\s*\(\s*)(\d(?:\s+\d){0,3})(\s*\))"
    )
    variants: list[str] = []
    for match in pattern.finditer(text):
        original = "".join(re.findall(r"\d", match.group(2)))
        if not original:
            continue
        for number in numbers:
            if number == original:
                continue
            replacement = " ".join(number) if len(number) > 1 else number
            candidate = f"{text[:match.start(2)]}{replacement}{text[match.end(2):]}"
            if candidate == text or candidate in variants:
                continue
            if numeric_repair_has_semantic_support(candidate, problem_latex, previous_latex):
                variants.append(candidate)
                break
    return variants


def repair_contextual_latex_numeric_equivalence(
    latex: str,
    problem_latex: str,
    previous_latex: Sequence[str],
) -> list[str]:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    numbers = contextual_number_tokens([problem_latex, *(previous_latex or [])])
    if not text or not numbers or text.count("=") != 1 or len(text) > 140:
        return []

    variants: list[str] = []
    for match in re.finditer(r"(?<![A-Za-z\\])\d(?:\s+\d)*(?![A-Za-z])", text):
        original = "".join(re.findall(r"\d", match.group(0)))
        if not original:
            continue
        for number in numbers:
            if number == original:
                continue
            replacement = " ".join(number) if len(number) > 1 else number
            candidate = f"{text[:match.start()]}{replacement}{text[match.end():]}"
            if candidate == text or candidate in variants:
                continue
            if numeric_repair_has_semantic_support(candidate, problem_latex, previous_latex):
                variants.append(candidate)
                break
        if len(variants) >= 8:
            break
    return variants


def repair_contextual_malformed_log_bases(
    latex: str,
    problem_latex: str,
    previous_latex: Sequence[str],
) -> list[str]:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    if r"\log" not in text:
        return []
    numbers = contextual_number_tokens([problem_latex, *(previous_latex or [])])
    if not numbers:
        return []

    pattern = re.compile(r"\\log\s*_\s*\{\s*[0oO]\s*\}\s*((?:\d\s*){1,4})\s*\}")
    variants: list[str] = []
    for match in pattern.finditer(text):
        captured = "".join(re.findall(r"\d", match.group(1)))
        candidates = [captured, *numbers] if captured else numbers
        for number in candidates:
            if not number:
                continue
            replacement = rf"\log _ {{ {' '.join(number) if len(number) > 1 else number} }}"
            candidate = f"{text[:match.start()]}{replacement}{text[match.end():]}"
            if candidate == text or candidate in variants:
                continue
            if numeric_repair_has_semantic_support(candidate, problem_latex, previous_latex):
                variants.append(candidate)
                break
    return variants


def repair_unmatched_parenthesis_letter(latex: str, variables: Sequence[str]) -> Optional[str]:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    if not text or text.count("(") <= text.count(")"):
        return None
    match = re.search(r"(?<!\\)\b([A-Za-z])\b\s*$", text)
    if not match or match.group(1) in set(variables):
        return None
    return f"{text[:match.start()].rstrip()} )"


def repair_spaced_digit_equivalence(
    latex: str,
    problem_latex: str,
    previous_latex: Sequence[str],
) -> list[str]:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    if not text or text.count("=") != 1 or "\\" in text:
        return []

    variants: list[str] = []
    for match in re.finditer(r"(?<![A-Za-z])\d(?:\s+\d){2,}(?![A-Za-z])", text):
        digits = re.findall(r"\d", match.group(0))
        if len(digits) > 5:
            continue
        for index in range(len(digits)):
            shortened = digits[:index] + digits[index + 1:]
            if len(shortened) < 2:
                continue
            replacement = " ".join(shortened)
            candidate = f"{text[:match.start()]}{replacement}{text[match.end():]}"
            if candidate == text or candidate in variants:
                continue
            if numeric_repair_has_semantic_support(candidate, problem_latex, previous_latex):
                variants.append(candidate)
    return variants


def repair_contextual_trig_number_lookalikes(
    latex: str,
    problem_latex: str,
    previous_latex: Sequence[str],
    variables: Sequence[str],
) -> list[str]:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    if not text or "n" in set(variables):
        return []
    numbers = contextual_number_tokens([problem_latex, *(previous_latex or [])])
    if not numbers:
        return []

    variants: list[str] = []
    patterns: list[tuple[re.Pattern[str], Optional[str]]] = [
        (re.compile(r"\\(?:cos|sin)\s*n\b"), None),
        (re.compile(r"(?<!\\)\bm\s*n\b"), None),
        (re.compile(r"(?<!\\)\bc\s*n\b"), None),
    ]
    for number in numbers:
        if number:
            patterns.append((re.compile(rf"(?<!\\)\b{re.escape(number[0])}\s*n\b"), number[0]))

    for pattern, required_prefix in patterns:
        for number in numbers:
            if required_prefix and not number.startswith(required_prefix):
                continue
            candidate = pattern.sub(lambda _match, value=number: value, text)
            if candidate == text or candidate in variants:
                continue
            if numeric_repair_has_semantic_support(candidate, problem_latex, previous_latex):
                variants.append(candidate)
    return variants


def repair_contextual_fraction_numeric_denominators(
    latex: str,
    problem_latex: str,
    previous_latex: Sequence[str],
) -> list[str]:
    if r"\frac" not in latex:
        return []
    numbers = contextual_number_tokens([problem_latex, *(previous_latex or [])])
    if not numbers:
        return []

    variants: list[str] = []
    index = 0
    while index < len(latex):
        frac_index = latex.find(r"\frac", index)
        if frac_index < 0:
            break
        try:
            numerator_start = skip_whitespace(latex, frac_index + 5)
            _numerator, after_numerator = _extract_group(latex, numerator_start)
            denominator_start = skip_whitespace(latex, after_numerator)
            denominator, after_denominator = _extract_group(latex, denominator_start)
        except ParseFailure:
            index = frac_index + 5
            continue
        denominator_text = re.sub(r"\s+", " ", denominator).strip()
        if re.fullmatch(r"\d+(?:\s+\d+)*", denominator_text):
            for number in numbers:
                replacement = " ".join(number) if len(number) > 1 else number
                if replacement == denominator_text:
                    continue
                candidate = latex[:denominator_start] + f"{{ {replacement} }}" + latex[after_denominator:]
                if candidate == latex or candidate in variants:
                    continue
                if numeric_repair_has_semantic_support(candidate, problem_latex, previous_latex):
                    variants.append(candidate)
        index = after_denominator
    return variants


def repair_from_previous_distributive_step(
    latex: str,
    problem_latex: str,
    previous_latex: Sequence[str],
) -> list[str]:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    if not text or "=" not in text and r"\cdots" not in text and r"\ldots" not in text:
        return []

    variants: list[str] = []
    for previous in previous_latex[-3:]:
        candidate = distribute_linear_parentheses_latex(previous)
        if not candidate or candidate in variants:
            continue
        overlap = latex_token_overlap(text, candidate)
        if overlap < 0.42:
            continue
        if numeric_repair_has_semantic_support(candidate, problem_latex, previous_latex):
            variants.append(candidate)
    return variants


def repair_from_contextual_denominator_clear_step(
    latex: str,
    problem_latex: str,
    previous_latex: Sequence[str],
) -> list[str]:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    if not text or ("=" not in text and r"\cdots" not in text and r"\ldots" not in text):
        return []
    if looks_like_final_variable_assignment(text):
        return []

    multiplier = contextual_equal_operation_multiplier(previous_latex)
    if multiplier is None:
        return []

    candidate = denominator_cleared_equation_latex(problem_latex, multiplier)
    if not candidate:
        return []
    candidate_compact = compact_latex(candidate)
    if any(compact_latex(previous) == candidate_compact for previous in (previous_latex or [])[-3:]):
        return []
    if latex_token_overlap(text, candidate) < 0.58:
        return []
    if numeric_repair_has_semantic_support(candidate, problem_latex, previous_latex):
        return [candidate]
    return []


def contextual_equal_operation_multiplier(previous_latex: Sequence[str]) -> Optional[int]:
    for latex in reversed(previous_latex or []):
        text = re.sub(r"\s+", " ", str(latex or "")).strip()
        if not text:
            continue
        match = re.fullmatch(
            r"\\(?:times|cdot)\s+((?:\d\s*){1,5})\s+\\(?:times|cdot)\s+((?:\d\s*){1,5})",
            text,
        )
        if not match:
            continue
        left = "".join(re.findall(r"\d", match.group(1)))
        right = "".join(re.findall(r"\d", match.group(2)))
        if left and left == right:
            return int(left)
    return None


def denominator_cleared_equation_latex(problem_latex: str, multiplier: int) -> Optional[str]:
    text = re.sub(r"\s+", " ", str(problem_latex or "")).strip()
    if multiplier <= 0 or text.count("=") != 1 or r"\frac" not in text:
        return None
    left, right = [part.strip() for part in text.split("=", 1)]
    terms = top_level_fraction_terms(left)
    if not terms:
        return None

    pieces: list[str] = []
    for index, (sign, numerator, denominator) in enumerate(terms):
        if denominator == 0 or multiplier % denominator != 0:
            return None
        coefficient = multiplier // denominator
        pieces.append(format_denominator_cleared_term(sign, coefficient, numerator, leading=index > 0))

    right_latex = denominator_cleared_side_latex(right, multiplier)
    if not right_latex:
        return None

    return f"{normalize_algebra_spacing(' '.join(pieces))} = {right_latex}"


def denominator_cleared_side_latex(latex: str, multiplier: int) -> Optional[str]:
    terms = top_level_fraction_terms(latex)
    if terms:
        numeric_value = numeric_cleared_terms_value(terms, multiplier)
        if numeric_value is not None:
            return sympy_value_to_latex(numeric_value)
        pieces: list[str] = []
        for index, (sign, numerator, denominator) in enumerate(terms):
            if denominator == 0 or multiplier % denominator != 0:
                return None
            coefficient = multiplier // denominator
            pieces.append(format_denominator_cleared_term(sign, coefficient, numerator, leading=index > 0))
        return normalize_algebra_spacing(" ".join(pieces))

    try:
        rhs = _parse_expression(normalize_latex(latex))
        rhs_value = sympy.simplify(multiplier * rhs)
    except (ParseFailure, SympyBudgetExceeded, Exception):
        return None
    if rhs_value.free_symbols or rhs_value.is_number is not True:
        return None
    return sympy_value_to_latex(rhs_value)


def numeric_cleared_terms_value(
    terms: Sequence[tuple[int, str, int]],
    multiplier: int,
) -> Optional[sympy.Expr]:
    total = sympy.Integer(0)
    for sign, numerator, denominator in terms:
        if denominator == 0 or multiplier % denominator != 0:
            return None
        try:
            value = _parse_expression(normalize_latex(numerator))
        except ParseFailure:
            return None
        if value.free_symbols or value.is_number is not True:
            return None
        total += sign * (multiplier // denominator) * value
    return sympy.simplify(total)


def top_level_fraction_terms(latex: str) -> list[tuple[int, str, int]]:
    terms: list[tuple[int, str, int]] = []
    index = 0
    while index < len(latex):
        index = skip_whitespace(latex, index)
        sign = 1
        if index < len(latex) and latex[index] in "+-":
            sign = -1 if latex[index] == "-" else 1
            index += 1
            index = skip_whitespace(latex, index)
        if not latex.startswith(r"\frac", index):
            return []
        try:
            numerator_start = skip_whitespace(latex, index + 5)
            numerator, after_numerator = _extract_group(latex, numerator_start)
            denominator_start = skip_whitespace(latex, after_numerator)
            denominator, after_denominator = _extract_group(latex, denominator_start)
        except ParseFailure:
            return []
        denominator_digits = "".join(re.findall(r"\d", denominator))
        if not denominator_digits or not re.fullmatch(r"\s*\d+(?:\s+\d+)*\s*", denominator):
            return []
        terms.append((sign, re.sub(r"\s+", " ", numerator).strip(), int(denominator_digits)))
        index = skip_whitespace(latex, after_denominator)
        if index < len(latex) and latex[index] not in "+-":
            return []
    return terms


def format_denominator_cleared_term(
    sign: int,
    coefficient: int,
    numerator: str,
    *,
    leading: bool,
) -> str:
    prefix = ""
    if sign < 0:
        prefix = "- "
    elif leading:
        prefix = "+ "
    numerator_text = re.sub(r"\s+", " ", numerator).strip()
    if coefficient == 1:
        return f"{prefix}{numerator_text}".strip()
    return f"{prefix}{coefficient} ( {numerator_text} )".strip()


def repair_from_previous_linear_simplification(
    latex: str,
    problem_latex: str,
    previous_latex: Sequence[str],
) -> list[str]:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    if not text or text.count("=") != 1:
        return []

    variants: list[str] = []
    for previous in previous_latex[-3:]:
        if r"\frac" in str(previous or ""):
            continue
        candidate = simplify_linear_equation_latex(previous)
        if not candidate or candidate in variants:
            continue
        if compact_latex(candidate) == compact_latex(previous):
            continue
        if latex_token_overlap(text.lower(), str(previous or "").lower()) >= 0.56:
            continue
        if latex_token_overlap(text, candidate) < 0.38:
            continue
        if numeric_repair_has_semantic_support(candidate, problem_latex, previous_latex):
            variants.append(candidate)
    return variants


def looks_like_final_variable_assignment(latex: str) -> bool:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    if text.count("=") != 1:
        return False
    left, right = [part.strip() for part in text.split("=", 1)]
    if not re.fullmatch(r"[A-Za-z](?:\s*\^\s*\{\s*[A-Za-z0-9]+\s*\})?", left):
        return False
    return bool(re.search(r"\d|\\frac", right))


def simplify_linear_equation_latex(latex: str) -> Optional[str]:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    if text.count("=") != 1:
        return None
    try:
        parsed = parse_math(text)
    except ParseFailure:
        return None
    if parsed.kind != "equation" or parsed.right is None:
        return None
    symbols = sorted(parsed.left.free_symbols | parsed.right.free_symbols, key=lambda item: item.name)
    if len(symbols) != 1 or parsed.right.free_symbols:
        return None
    variable = symbols[0]
    try:
        with sympy_budget(SYMPY_TIMEOUT_SECONDS):
            left = sympy.expand(parsed.left)
            coefficient = sympy.simplify(left.coeff(variable))
            constant = sympy.simplify(left.subs(variable, 0))
            right = sympy.simplify(parsed.right)
    except (Exception, SympyBudgetExceeded):
        return None
    if any(value.free_symbols for value in (coefficient, constant, right)):
        return None
    if any(value.is_number is not True for value in (coefficient, constant, right)):
        return None
    if any(value.is_finite is False for value in (coefficient, constant, right)):
        return None
    if any(value.is_real is False for value in (coefficient, constant, right)):
        return None
    left_latex = format_linear_expression_latex(coefficient, variable.name, constant)
    if not left_latex:
        return None
    return f"{left_latex} = {sympy_value_to_latex(right)}"


def format_linear_expression_latex(coefficient: sympy.Expr, variable: str, constant: sympy.Expr) -> str:
    parts: list[str] = []
    coefficient = sympy.simplify(coefficient)
    constant = sympy.simplify(constant)
    if coefficient != 0:
        if coefficient < 0:
            parts.append("-")
        abs_coefficient = abs(coefficient)
        coefficient_text = "" if abs_coefficient == 1 else f"{sympy_value_to_latex(abs_coefficient)} "
        parts.append(f"{coefficient_text}{variable}".strip())
    if constant != 0:
        sign = "-" if constant < 0 else "+"
        if not parts and sign == "+":
            parts.append(sympy_value_to_latex(constant))
        else:
            parts.append(sign)
            parts.append(sympy_value_to_latex(abs(constant)))
    return " ".join(parts)


def distribute_linear_parentheses_latex(latex: str) -> Optional[str]:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    if text.count("=") != 1:
        return None
    left, right = [part.strip() for part in text.split("=", 1)]
    term_pattern = re.compile(r"([+\-]?)\s*(\d+)\s*\(\s*([^()]+?)\s*\)")
    pieces: list[str] = []
    cursor = 0
    changed = False
    for match in term_pattern.finditer(left):
        pieces.append(left[cursor:match.start()])
        sign_text, outer_text, inner = match.groups()
        expanded = expand_linear_parenthesis_term(sign_text, outer_text, inner)
        if not expanded:
            return None
        pieces.append(expanded)
        cursor = match.end()
        changed = True
    if not changed:
        return None
    pieces.append(left[cursor:])
    distributed_left = normalize_algebra_spacing(" ".join(pieces))
    if not distributed_left:
        return None
    return f"{distributed_left} = {normalize_spaced_number_latex(right)}"


def expand_linear_parenthesis_term(sign_text: str, outer_text: str, inner: str) -> Optional[str]:
    match = re.fullmatch(r"\s*(?:(\d+)\s*)?([A-Za-z])\s*([+\-])\s*(\d+)\s*", inner)
    if not match:
        return None
    inner_coeff_text, variable, inner_sign, constant_text = match.groups()
    outer = int(outer_text)
    sign = -1 if sign_text == "-" else 1
    inner_coeff = int(inner_coeff_text or "1")
    constant = int(constant_text)
    variable_coeff = sign * outer * inner_coeff
    constant_coeff = sign * outer * (constant if inner_sign == "+" else -constant)
    return format_linear_terms(variable_coeff, variable, constant_coeff, leading_sign=bool(sign_text))


def format_linear_terms(
    variable_coeff: int,
    variable: str,
    constant_coeff: int,
    *,
    leading_sign: bool,
) -> str:
    parts: list[str] = []
    if variable_coeff < 0:
        parts.append("-")
    elif leading_sign:
        parts.append("+")
    abs_variable = abs(variable_coeff)
    parts.append(f"{abs_variable} {variable}" if abs_variable != 1 else variable)
    if constant_coeff < 0:
        parts.append("-")
        parts.append(str(abs(constant_coeff)))
    elif constant_coeff > 0:
        parts.append("+")
        parts.append(str(constant_coeff))
    return " ".join(parts)


def normalize_algebra_spacing(latex: str) -> str:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    text = re.sub(r"^\+\s+", "", text)
    text = re.sub(r"\s+([+\-])\s+", r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_spaced_number_latex(latex: str) -> str:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    if re.fullmatch(r"\d(?:\s+\d)+", text):
        return "".join(re.findall(r"\d", text))
    return text


def latex_token_overlap(a: str, b: str) -> float:
    left = re.findall(r"\\[A-Za-z]+|[A-Za-z]+|\d+|[+\-*/=()]", str(a or ""))
    right = re.findall(r"\\[A-Za-z]+|[A-Za-z]+|\d+|[+\-*/=()]", str(b or ""))
    left = [token.lower() for token in left if token not in {r"\cdots", r"\ldots"}]
    right = [token.lower() for token in right if token not in {r"\cdots", r"\ldots"}]
    if not left or not right:
        return 0.0
    remaining = list(right)
    matched = 0
    for token in left:
        if token in remaining:
            remaining.remove(token)
            matched += 1
    return matched / max(1, min(len(left), len(right)))


def compact_latex(latex: str) -> str:
    return re.sub(r"\s+", "", str(latex or "")).strip()


def repair_contextual_symbol_confusions(
    latex: str,
    problem_latex: str,
    previous_latex: Sequence[str],
) -> list[str]:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    if not text:
        return []

    variants: list[str] = []
    variants.extend(repair_from_previous_distributive_step(text, problem_latex, previous_latex))

    context = [problem_latex, *(previous_latex or [])]
    functions = contextual_function_names(context)
    variables = contextual_variable_names(context)
    if not functions and not variables:
        return unique_preserving_order(variants)

    def append_variant(candidate: Optional[str], *, require_support: bool = False) -> None:
        if not candidate or candidate == text:
            return
        if require_support and not numeric_repair_has_semantic_support(candidate, problem_latex, previous_latex):
            return
        variants.append(candidate)
        variants.extend(repair_contextual_numeric_lookalikes(candidate, problem_latex, previous_latex))
        variants.extend(repair_contextual_numeric_equivalence(candidate, problem_latex, previous_latex))
        variants.extend(repair_contextual_fraction_numeric_denominators(candidate, problem_latex, previous_latex))
        variants.extend(repair_numeric_fraction_arithmetic(candidate, problem_latex, previous_latex))

    variants.extend(repair_contextual_numeric_equivalence(text, problem_latex, previous_latex))
    variants.extend(repair_spaced_digit_equivalence(text, problem_latex, previous_latex))
    variants.extend(repair_contextual_trig_number_lookalikes(text, problem_latex, previous_latex, variables))
    variants.extend(repair_contextual_fraction_numeric_denominators(text, problem_latex, previous_latex))
    variants.extend(repair_contextual_function_evaluation_step(text, problem_latex, previous_latex))
    variants.extend(repair_contextual_derivative_equation(text, problem_latex, previous_latex))
    variants.extend(repair_contextual_adjacent_fraction_sum(text, problem_latex, previous_latex, variables))
    parenthesis_repaired = repair_unmatched_parenthesis_letter(text, variables)
    append_variant(parenthesis_repaired, require_support=True)
    if parenthesis_repaired:
        variants.extend(repair_contextual_numeric_equivalence(parenthesis_repaired, problem_latex, previous_latex))
    for repaired_equation in repair_contextual_exponent_equals_confusion(text, variables):
        append_variant(repaired_equation)
    for repaired_noise in repair_stray_letter_before_fraction(text, variables):
        append_variant(repaired_noise)
    if "=" not in text:
        return unique_preserving_order(variants)
    for repaired_variable in repair_contextual_noncontext_variables(text, functions, variables):
        append_variant(repaired_variable)
        for repaired_denominator in repair_contextual_fraction_denominators(repaired_variable, variables):
            append_variant(repaired_denominator)
    for repaired_variable_case in repair_contextual_uppercase_variables(text, context, variables):
        append_variant(repaired_variable_case)
    for repaired_command_variable in repair_contextual_command_variables(text, context, variables):
        append_variant(repaired_command_variable)
    for repaired_subscript in repair_contextual_variable_subscripts(text, variables):
        append_variant(repaired_subscript)
    for repaired_rhs in repair_contextual_power_terms(text, variables):
        append_variant(repaired_rhs)
    for repaired_denominator in repair_contextual_fraction_denominators(text, variables):
        append_variant(repaired_denominator)
    for function_repaired in repair_contextual_function_header(text, functions):
        append_variant(function_repaired)
        for repaired_rhs in repair_contextual_power_terms(function_repaired, variables):
            append_variant(repaired_rhs)
    return unique_preserving_order(variants)


def contextual_function_names(latex_values: Sequence[str]) -> list[str]:
    names: list[str] = []
    for latex in latex_values:
        left_side = str(latex or "").split("=", 1)[0]
        match = re.match(
            r"^\s*([A-Za-z])\s*(?:\^\s*\{\s*\\prime\s*\}|'\s*)?\s*\(",
            left_side,
        )
        if match:
            name = match.group(1)
            if name not in names:
                names.append(name)
    return names


def contextual_variable_names(latex_values: Sequence[str]) -> list[str]:
    variables: list[str] = []
    function_names = set(contextual_function_names(latex_values))
    for latex in latex_values:
        normalized = str(latex or "")
        normalized = re.sub(r"\\[A-Za-z]+", " ", normalized)
        for token in re.findall(r"\b[A-Za-z]\b", normalized):
            if token in function_names:
                continue
            if token not in variables:
                variables.append(token)
    return variables


def repair_contextual_function_evaluation_step(
    latex: str,
    problem_latex: str,
    previous_latex: Sequence[str],
) -> list[str]:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    if text.count("=") != 1:
        return []
    match = re.match(
        r"^\s*([A-Za-z])\s*(?:\^\s*\{\s*(?:\\prime|[A-Za-z])\s*\}|'\s*)?\s*\(\s*(-?\d+)\s*\)\s*=",
        text,
    )
    if not match:
        return []
    function_name, value_text = match.groups()
    value = int(value_text)

    variants: list[str] = []
    for previous in reversed(previous_latex or []):
        previous_match = re.match(
            rf"^\s*{re.escape(function_name)}\s*(?:\^\s*\{{\s*\\prime\s*\}}|'\s*)\s*\(\s*x\s*\)\s*=\s*(.+)$",
            re.sub(r"\s+", " ", str(previous or "")).strip(),
        )
        if not previous_match:
            continue
        rhs = derivative_substitution_latex(previous_match.group(1), value)
        if not rhs:
            continue
        candidate = f"{function_name} ^ {{ \\prime }} ( {value} ) = {rhs}"
        if numeric_repair_has_semantic_support(candidate, problem_latex, previous_latex):
            variants.append(candidate)
            break
    return variants


def derivative_substitution_latex(rhs: str, value: int) -> Optional[str]:
    text = re.sub(r"\s+", " ", str(rhs or "")).strip()
    if not text:
        return None

    def product_replacement(match: re.Match[str]) -> str:
        coefficient = int(match.group(1))
        sign = match.group(2)
        constant = int(match.group(3))
        inner = value + constant if sign == "+" else value - constant
        return f"{coefficient * value} ( {inner} )"

    text = re.sub(
        r"\b(\d+)\s*x\s*\(\s*x\s*([+\-])\s*(\d+)\s*\)",
        product_replacement,
        text,
    )

    def powered_replacement(match: re.Match[str]) -> str:
        coefficient_text, power_text = match.groups()
        power = int(power_text)
        if coefficient_text:
            return f"{int(coefficient_text)} ( {value} ) ^ {{ {power} }}"
        return sympy_value_to_latex(sympy.Integer(value) ** power)

    text = re.sub(
        r"\b(?:(\d+)\s*)?x\s*\^\s*\{\s*(\d+)\s*\}",
        powered_replacement,
        text,
    )
    text = re.sub(r"\b(\d+)\s*x\b", lambda match: f"{match.group(1)} ( {value} )", text)
    text = re.sub(r"(?<![A-Za-z])x(?![A-Za-z])", str(value), text)
    return normalize_algebra_spacing(text)


def repair_contextual_derivative_equation(
    latex: str,
    problem_latex: str,
    previous_latex: Sequence[str],
) -> list[str]:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    if r"\frac" not in text:
        return []
    try:
        reference = normalize_latex(problem_latex)
    except ParseFailure:
        return []
    match = re.fullmatch(r"([A-Za-z])\(([^()]+)\)=(.+)", reference)
    if not match:
        return []
    function_name, argument, rhs = match.groups()
    try:
        variable = _parse_expression(argument)
        if not isinstance(variable, sympy.Symbol):
            return []
        derivative = sympy.together(sympy.simplify(sympy.diff(_parse_expression(rhs), variable)))
    except (ParseFailure, SympyBudgetExceeded, Exception):
        return []
    candidate = f"{function_name} ^ {{ \\prime }} ( {variable.name} ) = {sympy_expression_to_latex(derivative)}"
    if "=" in text:
        if not re.match(
            rf"^\s*{re.escape(function_name)}\s*(?:\^\s*\{{\s*(?:\\prime|[A-Za-z])\s*\}}|'\s*)?\s*\(",
            text,
        ):
            return []
        _lhs, rhs_text = text.split("=", 1)
        derivative_latex = sympy_expression_to_latex(derivative)
        for rhs_variant in derivative_rhs_context_variants(rhs_text, variable.name, function_name):
            if check_equivalence(f"= {derivative_latex}", f"= {rhs_variant}").equivalent is True:
                if numeric_repair_has_semantic_support(candidate, problem_latex, previous_latex):
                    return [candidate]
        return []

    if not re.match(r"^\s*[A-Za-z]\s*(?:\^\s*\{\s*\\prime\s*\}|'\s*)", text):
        return []
    if numeric_repair_has_semantic_support(candidate, problem_latex, previous_latex):
        return [candidate]
    return []


def derivative_rhs_context_variants(rhs_latex: str, variable_name: str, function_name: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(rhs_latex or "")).strip()
    if not text:
        return []
    variants = [text]
    for token in sorted(set(re.findall(r"(?<!\\)\b[A-Za-z]\b", text))):
        if token in {variable_name, function_name}:
            continue
        variants.append(re.sub(rf"(?<!\\)\b{re.escape(token)}\b", variable_name, text))
    return unique_preserving_order(variants)


def repair_contextual_function_header(latex: str, functions: Sequence[str]) -> list[str]:
    if not functions:
        return []
    match = re.match(
        r"^\s*([A-Za-z/])\s*(?:\^\s*\{\s*([^}]+)\s*\}|'\s*)?\s*\(\s*([^)]{1,16})\s*\)\s*=",
        latex,
    )
    if not match:
        return []
    seen_name, superscript, argument = match.groups()

    suffix = latex[match.end():].strip()
    variants: list[str] = []
    for name in functions[:2]:
        if seen_name != name or superscript != r"\prime":
            variants.append(f"{name} ^ {{ \\prime }} ( {argument.strip()} ) = {suffix}")
        if seen_name != name or superscript:
            variants.append(f"{name} ( {argument.strip()} ) = {suffix}")
    return variants


def repair_contextual_exponent_equals_confusion(latex: str, variables: Sequence[str]) -> list[str]:
    variants: list[str] = []
    if not variables:
        return variants
    for variable in variables[:2]:
        pattern = re.compile(rf"^\s*{re.escape(variable)}\s*\^\s*\{{\s*(?:[nN]|\d+)\s*\}}\s*=\s*(.+)$")
        match = pattern.match(latex)
        if match:
            rhs = re.sub(r"(?<!\\)\bi\s*(?=\d+\b)", "", match.group(1).strip())
            variants.append(f"{variable} = {rhs}")
    return variants


def repair_stray_letter_before_fraction(latex: str, variables: Sequence[str]) -> list[str]:
    if r"\frac" not in latex:
        return []
    variants: list[str] = []
    variable_set = set(variables)
    adjacent_fraction_pattern = re.compile(
        r"(\\frac\s*\{\s*[^{}]+\s*\}\s*\{\s*[^{}]+\s*\})\s+([A-Za-z])\s+(?=\\frac\b)"
    )
    for match in adjacent_fraction_pattern.finditer(latex):
        if match.group(2) in variable_set:
            continue
        candidate = (latex[:match.start()] + match.group(1) + " + " + latex[match.end():]).strip()
        candidate = re.sub(r"\s+", " ", candidate)
        if candidate != latex:
            variants.append(candidate)
    for match in re.finditer(r"(?<!\\)\b([A-Za-z])\s+(?=\\frac\b)", latex):
        if match.group(1) in variable_set:
            continue
        candidate = (latex[:match.start()] + latex[match.end():]).strip()
        candidate = re.sub(r"\s+", " ", candidate)
        if candidate != latex:
            variants.append(candidate)
    return variants


def repair_contextual_adjacent_fraction_sum(
    latex: str,
    problem_latex: str,
    previous_latex: Sequence[str],
    variables: Sequence[str],
) -> list[str]:
    if latex.count(r"\frac") < 2 or "=" not in latex:
        return []
    context = [problem_latex, *(previous_latex or [])]
    if r"\pi" not in latex and not re.search(r"\}\s*(?:[A-Za-z]|[+]\s*[A-Za-z])\s*\\frac\b", latex):
        return []

    variants = {re.sub(r"\s+", " ", str(latex or "")).strip()}
    context_has_pi = context_contains_command(context, r"\pi")
    if not context_has_pi:
        replacement_variables = [variable for variable in variables if re.fullmatch(r"[a-z]", variable)]
        if replacement_variables:
            next_variants = set(variants)
            for variant in variants:
                next_variants.add(re.sub(r"\\pi\b", replacement_variables[0], variant))
            variants = next_variants

    next_variants = set(variants)
    for variant in variants:
        next_variants.add(re.sub(r"\}\s*[A-Za-z]\s*(?=\\frac\b)", r"} + ", variant))
        next_variants.add(re.sub(r"\}\s*\+\s*[A-Za-z]\s*(?=\\frac\b)", r"} + ", variant))
    variants = next_variants

    repaired: list[str] = []
    for candidate in variants:
        candidate = normalize_algebra_spacing(candidate)
        if candidate == latex:
            continue
        if numeric_repair_has_semantic_support(candidate, problem_latex, previous_latex):
            repaired.append(candidate)
    return unique_preserving_order(repaired)


def repair_contextual_noncontext_variables(
    latex: str,
    functions: Sequence[str],
    variables: Sequence[str],
) -> list[str]:
    if not variables:
        return []
    if r"\frac" not in latex and not re.match(r"^\s*[A-Za-z]\s*(?:\^\s*\{[^}]+\}\s*)?\(", latex):
        return []
    function_set = set(functions)
    variable_set = set(variables)
    variants: list[str] = []
    for variable in variables[:2]:
        candidate = re.sub(
            r"(?<!\\)\b([A-Za-z])\b",
            lambda match: (
                match.group(1)
                if match.group(1) in variable_set or match.group(1) in function_set
                else variable
            ),
            latex,
        )
        if candidate != latex:
            variants.append(candidate)
    return variants


def repair_contextual_uppercase_variables(
    latex: str,
    context: Sequence[str],
    variables: Sequence[str],
) -> list[str]:
    lowercase_variables = [variable for variable in variables if re.fullmatch(r"[a-z]", variable)]
    if not lowercase_variables:
        return []
    context_tokens = set(contextual_variable_names(context))
    variants: list[str] = []
    for variable in lowercase_variables[:2]:
        uppercase = variable.upper()
        if uppercase in context_tokens or not re.search(rf"(?<!\\)\b{re.escape(uppercase)}\b", latex):
            continue
        candidate = re.sub(rf"(?<!\\)\b{re.escape(uppercase)}\b", variable, latex)
        if candidate != latex:
            variants.append(candidate)
    return variants


def repair_contextual_command_variables(
    latex: str,
    context: Sequence[str],
    variables: Sequence[str],
) -> list[str]:
    if not variables:
        return []
    command_lookalikes = {
        r"\alpha": variables[:2],
    }
    variants: list[str] = []
    for command, replacements in command_lookalikes.items():
        if command not in latex or context_contains_command(context, command):
            continue
        for variable in replacements:
            candidate = re.sub(rf"{re.escape(command)}\b", variable, latex)
            if candidate != latex:
                variants.append(candidate)
    return variants


def repair_contextual_variable_subscripts(latex: str, variables: Sequence[str]) -> list[str]:
    variants: list[str] = []
    for variable in variables[:2]:
        candidate = re.sub(
            rf"(?<!\\)\b{re.escape(variable)}\s*_\s*\{{\s*\d+\s*\}}",
            variable,
            latex,
        )
        if candidate != latex:
            variants.append(candidate)
    return variants


def repair_contextual_fraction_denominators(latex: str, variables: Sequence[str]) -> list[str]:
    if r"\frac" not in latex or not variables:
        return []
    variants: list[str] = []
    index = 0
    while index < len(latex):
        frac_index = latex.find(r"\frac", index)
        if frac_index < 0:
            break
        try:
            numerator_start = skip_whitespace(latex, frac_index + 5)
            _numerator, after_numerator = _extract_group(latex, numerator_start)
            denominator_start = skip_whitespace(latex, after_numerator)
            denominator, after_denominator = _extract_group(latex, denominator_start)
        except ParseFailure:
            index = frac_index + 5
            continue
        if not denominator_contains_context_variable(denominator, variables):
            for variable in variables[:2]:
                candidate = latex[:denominator_start] + f"{{ {variable} }}" + latex[after_denominator:]
                if candidate != latex:
                    variants.append(candidate)
        index = after_denominator
    return variants


def skip_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def denominator_contains_context_variable(denominator: str, variables: Sequence[str]) -> bool:
    tokens = set(re.findall(r"(?<!\\)\b[A-Za-z]\b", denominator))
    return bool(tokens & set(variables))


def repair_contextual_power_terms(latex: str, variables: Sequence[str]) -> list[str]:
    if not variables:
        return []
    variants: list[str] = []
    for variable in variables[:2]:
        candidate = re.sub(
            r"(?<=\+ )([mn])\s*2\b",
            rf"{variable} ^ {{ 2 }}",
            latex,
        )
        candidate = re.sub(
            r"(?<== )([mn])\s*2\b",
            rf"{variable} ^ {{ 2 }}",
            candidate,
        )
        candidate = re.sub(
            r"\\(?:ldots|cdots)\s*\+?\s*2\b",
            rf"{variable} ^ {{ 2 }}",
            candidate,
        )
        if candidate != latex:
            variants.append(candidate)
    return variants


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def previous_numeric_value(previous_latex: Sequence[str]) -> Optional[sympy.Expr]:
    for previous in reversed(previous_latex or []):
        try:
            parsed = parse_math(previous)
        except ParseFailure:
            continue
        if parsed.kind != "expression":
            continue
        try:
            with sympy_budget(SYMPY_TIMEOUT_SECONDS):
                value = sympy.simplify(parsed.left.doit())
        except (Exception, SympyBudgetExceeded):
            continue
        if value.free_symbols or value.is_number is not True:
            continue
        if expression_exceeds_budget(value):
            continue
        return value
    return None


def sympy_value_to_latex(value: sympy.Expr) -> str:
    value = sympy.simplify(value)
    if value.is_Integer:
        return str(value)
    if value.is_Rational:
        numerator, denominator = value.as_numer_denom()
        if denominator == 1:
            return str(numerator)
        return rf"\frac {{ {numerator} }} {{ {denominator} }}"
    return str(value)


def sympy_expression_to_latex(value: sympy.Expr) -> str:
    raw = sympy.latex(value)

    def normalize_fraction(match: re.Match[str]) -> str:
        numerator = normalize_sympy_latex_fragment(match.group(1))
        denominator = normalize_sympy_latex_fragment(match.group(2))
        return rf"\frac {{ {numerator} }} {{ {denominator} }}"

    previous = None
    text = raw
    while previous != text:
        previous = text
        text = re.sub(r"\\frac\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", normalize_fraction, text)
    return normalize_sympy_latex_fragment(text)


def repair_numeric_fraction_arithmetic(
    latex: str,
    problem_latex: str,
    previous_latex: Sequence[str],
) -> list[str]:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    if r"\frac" not in text:
        return []

    variants: list[str] = []
    pattern = re.compile(r"\\frac\s*\{\s*([0-9+\-*/\s]+)\s*\}\s*\{\s*((?:\d+\s*){1,5})\s*\}")
    for match in pattern.finditer(text):
        numerator_text = match.group(1).strip()
        denominator_text = "".join(re.findall(r"\d", match.group(2)))
        if not denominator_text or not re.search(r"[+\-*/]", numerator_text):
            continue
        try:
            numerator_value = sympy.simplify(_parse_expression(numerator_text))
        except (ParseFailure, SympyBudgetExceeded, Exception):
            continue
        if numerator_value.free_symbols or numerator_value.is_number is not True:
            continue
        replacement = rf"\frac {{ {sympy_value_to_latex(numerator_value)} }} {{ {denominator_text} }}"
        candidate = f"{text[:match.start()]}{replacement}{text[match.end():]}"
        if candidate == text or candidate in variants:
            continue
        if numeric_repair_has_semantic_support(candidate, problem_latex, previous_latex):
            variants.append(candidate)
    return variants


def normalize_sympy_latex_fragment(latex: str) -> str:
    text = str(latex or "")
    text = re.sub(r"([A-Za-z])\s*\^\s*\{\s*([^{}]+)\s*\}", r"\1 ^ { \2 }", text)
    text = re.sub(r"([A-Za-z])\s*\^\s*(\d+)", r"\1 ^ { \2 }", text)
    text = re.sub(r"([+\-=()])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def repair_trailing_one_as_parenthesis(latex: str) -> Optional[str]:
    text = str(latex or "").rstrip()
    if not text.endswith("1"):
        return None
    candidate = text[:-1].rstrip() + " )"
    try:
        parse_math(text)
        return None
    except ParseFailure:
        pass
    try:
        parse_math(candidate)
    except ParseFailure:
        return None
    return candidate


def repair_malformed_arithmetic_continuation(latex: str) -> Optional[str]:
    text = re.sub(r"\s+", "", str(latex or ""))
    if not text or "\\" in text:
        return None
    if not re.search(r"\d", text) or not re.search(r"[()+\-*/]", text):
        return None

    letters = re.findall(r"[A-Za-z]", text)
    if letters and any(letter.lower() not in {"o", "n"} for letter in letters):
        return None
    if not letters and text.count("(") == text.count(")") and not text.startswith("-1"):
        return None

    arithmetic = re.sub(r"[oOnN]", "0", text)
    candidate: Optional[str] = None
    if arithmetic.startswith("-1") and len(arithmetic) > 2:
        candidate = "=(" + arithmetic[2:]
    elif arithmetic.startswith("-("):
        candidate = "=" + arithmetic[1:]
    elif arithmetic.startswith("-") and arithmetic.count(")") > arithmetic.count("("):
        candidate = "=(" + arithmetic[1:]

    if candidate is None:
        return None
    candidate = _balance_trailing_parentheses(candidate)
    if any(re.fullmatch(r"0\d+", token) for token in re.findall(r"\d+", candidate)):
        return None
    pretty = _space_arithmetic_latex(candidate)
    try:
        parsed = parse_math(pretty)
    except ParseFailure:
        return None
    if parsed.kind != "expression":
        return None
    return pretty


def _balance_trailing_parentheses(text: str) -> str:
    balance = 0
    output: list[str] = []
    for char in text:
        if char == "(":
            balance += 1
        elif char == ")":
            if balance == 0:
                output.append("(")
                balance += 1
            balance -= 1
        output.append(char)
    if balance > 0:
        output.extend(")" for _ in range(balance))
    return "".join(output)


def _space_arithmetic_latex(text: str) -> str:
    tokens = re.findall(r"\d+|[=()+\-*/^]", text)
    if not tokens or "".join(tokens) != text:
        return text
    return " ".join(tokens)


def score_candidate_group(
    group: dict[str, Any],
    *,
    problem_latex: str,
    previous_latex: Sequence[str] = (),
) -> dict[str, Any]:
    elapsed_seconds = group.get("elapsedSeconds")
    predictions = list(group.get("candidates") or [])
    if not predictions and group.get("latex"):
        predictions = [{"latex": group.get("latex"), "score": group.get("score")}]
    if elapsed_seconds is not None:
        predictions = [
            prediction_with_elapsed(prediction, elapsed_seconds)
            for prediction in predictions
        ]

    scored = score_ocr_predictions(
        predictions,
        problem_latex=problem_latex,
        previous_latex=previous_latex,
    )
    best = scored[0] if scored else None

    return {
        "candidateId": group.get("candidateId"),
        "semanticScore": best.score if best else -1000,
        "bestLatex": best.latex if best else "",
        "sound": best.sound if best else False,
        "equivalentToProblem": best.equivalent_to_problem if best else False,
        "equivalentToPrevious": best.equivalent_to_previous if best else False,
        "candidateScores": [item.to_json() for item in scored],
    }


def prediction_with_elapsed(prediction: dict[str, Any] | str, elapsed_seconds: Any) -> dict[str, Any] | str:
    if isinstance(prediction, str):
        return {"latex": prediction, "elapsedSeconds": elapsed_seconds}
    if "elapsedSeconds" in prediction:
        return prediction
    return {**prediction, "elapsedSeconds": elapsed_seconds}


def score_semantic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    problem_latex = str(payload.get("problemLatex") or "")
    previous_latex = [str(item) for item in payload.get("previousLatex") or []]
    groups = payload.get("candidateGroups") or payload.get("candidates") or []

    return {
        "candidateScores": [
            score_candidate_group(
                group,
                problem_latex=problem_latex,
                previous_latex=previous_latex,
            )
            for group in groups
        ]
    }


def character_overlap(*references: str, candidate: str) -> float:
    reference_weights = math_symbol_weights(" ".join(references))
    candidate_weights = math_symbol_weights(candidate)
    if not reference_weights or not candidate_weights:
        return 0.0
    all_symbols = set(reference_weights) | set(candidate_weights)
    union_weight = sum(max(reference_weights.get(symbol, 0.0), candidate_weights.get(symbol, 0.0)) for symbol in all_symbols)
    if union_weight <= 0:
        return 0.0
    matched_weight = sum(
        min(reference_weights[symbol], candidate_weights[symbol])
        for symbol in set(reference_weights) & set(candidate_weights)
    )
    return matched_weight / union_weight


def math_symbol_weights(latex: str) -> dict[str, float]:
    return {
        symbol: symbol_weight(symbol)
        for symbol in math_symbols(latex)
    }


def symbol_weight(symbol: str) -> float:
    if symbol.isdigit():
        return 0.35
    if len(symbol) == 1 and symbol.isalpha():
        return 1.35
    return 1.6


def math_symbols(latex: str) -> set[str]:
    ignore = {"left", "right", "frac", "sqrt", "cdot", "times", "log", "ln"}
    symbols: set[str] = set()
    for match in re.finditer(r"\\[A-Za-z]+|[A-Za-z0-9]", latex or ""):
        token = match.group(0).lstrip("\\")
        if token not in ignore:
            symbols.add(token)
    return symbols
