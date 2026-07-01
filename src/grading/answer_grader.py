"""Shared final-answer grading primitives.

This module deliberately stays small: problem-specific graders still own
parsing and mathematical matching, while this layer provides the common result
shape used to distinguish equivalence from final-answer form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import sympy


FINALITY_FINAL = "final"
FINALITY_UNSIMPLIFIED = "unsimplified"
FINALITY_INVALID_FORMAT = "invalid_format"
FINALITY_NOT_ANSWER = "not_answer"


@dataclass(frozen=True)
class AnswerSpec:
    response_kind: str
    cardinality: str
    exact_values: tuple[sympy.Expr, ...] = ()
    solution_set: Optional[sympy.Set] = None
    variable: Optional[str] = None
    tolerance: float = 0.005
    acceptable_strings: tuple[str, ...] = ()
    finality_policy: str = "pragmatic"

    def to_manifest_fields(self) -> dict[str, Any]:
        return {
            "responseKind": self.response_kind,
            "cardinality": self.cardinality,
            "finalityPolicy": self.finality_policy,
        }


@dataclass(frozen=True)
class AnswerFinality:
    status: str = FINALITY_NOT_ANSWER
    counts_toward_completion: bool = False
    reason: Optional[str] = None

    def public_fields(self) -> dict[str, Any]:
        fields = {
            "answerFinality": self.status,
            "countsTowardCompletion": bool(self.counts_toward_completion),
        }
        if self.reason:
            fields["finalityReason"] = self.reason
        return fields


def final_answer() -> AnswerFinality:
    return AnswerFinality(FINALITY_FINAL, True)


def unsimplified_answer(reason: str = "answer is equivalent but not fully simplified") -> AnswerFinality:
    return AnswerFinality(FINALITY_UNSIMPLIFIED, False, reason)


def invalid_format(reason: str = "answer format is not supported") -> AnswerFinality:
    return AnswerFinality(FINALITY_INVALID_FORMAT, False, reason)


def not_answer() -> AnswerFinality:
    return AnswerFinality(FINALITY_NOT_ANSWER, False)


def answer_spec_from_manifest(manifest: dict[str, Any], exact_values: Sequence[sympy.Expr] = ()) -> AnswerSpec:
    return AnswerSpec(
        response_kind=str(manifest.get("responseKind") or manifest.get("response_kind") or "solution_set"),
        cardinality=str(manifest.get("cardinality") or "unsupported"),
        exact_values=tuple(exact_values),
        variable=str(manifest.get("variable")) if manifest.get("variable") else None,
        tolerance=float(manifest.get("tolerance", 0.005)),
        acceptable_strings=tuple(str(item) for item in (manifest.get("acceptable_strings") or ())),
        finality_policy=str(manifest.get("finalityPolicy") or manifest.get("finality_policy") or "pragmatic"),
    )


def candidate_rank(verdict: dict[str, Any]) -> tuple[int, int]:
    """Return a stable ranking tuple for candidate verdict selection."""

    classification = verdict.get("classification", "other")
    finality = verdict.get("answerFinality", FINALITY_NOT_ANSWER)
    coverage = verdict.get("solutionCoverage", "none")
    counts = verdict.get("countsTowardCompletion")

    if classification == "valid_step" and finality == FINALITY_FINAL and counts is not False:
        primary = 0
    elif classification == "valid_step" and finality == FINALITY_UNSIMPLIFIED:
        primary = 1
    elif classification == "valid_step":
        primary = 2
    elif classification == "invalid_step":
        primary = 3
    elif classification == "other":
        primary = 4
    else:
        primary = 5

    coverage_rank = {"full": 0, "partial": 1, "none": 2}
    return primary, coverage_rank.get(coverage, 9)
