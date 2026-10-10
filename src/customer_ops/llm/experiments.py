"""Score recorded LLM experiment outputs; this module is not part of an agent workflow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from customer_ops.domain.schemas import DecisionProposal, IntentResult, ToolRequest


@dataclass(frozen=True)
class FailureExample:
    case_id: str
    expected: Any
    observed: Any


@dataclass(frozen=True)
class EvaluationResult:
    metric: str
    correct: int
    total: int
    failures: list[FailureExample] = field(default_factory=list)

    @property
    def rate(self) -> float | None:
        return self.correct / self.total if self.total else None

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "rate": self.rate}


def evaluate_schema_validity(
    schema: type[BaseModel], cases: list[tuple[str, object]]
) -> EvaluationResult:
    failures: list[FailureExample] = []
    for case_id, payload in cases:
        try:
            # Every generated payload is expected to satisfy the requested response schema.
            schema.model_validate(payload)
        except ValidationError as error:
            failures.append(FailureExample(case_id, schema.__name__, error.errors(include_url=False)))
    return EvaluationResult("schema_validity", len(cases) - len(failures), len(cases), failures)


def evaluate_intents(cases: list[tuple[str, IntentResult, IntentResult]]) -> EvaluationResult:
    return _evaluate_fields("intent_accuracy", cases, ("intent",))


def evaluate_entities(cases: list[tuple[str, IntentResult, IntentResult]]) -> EvaluationResult:
    return _evaluate_fields("entity_accuracy", cases, ("order_id", "proposed_address"))


def evaluate_decisions(cases: list[tuple[str, DecisionProposal, DecisionProposal]]) -> EvaluationResult:
    return _evaluate_fields("decision_correctness", cases, ("action", "order_id", "proposed_address", "rejected_action", "policy_references"))


def evaluate_requests(cases: list[tuple[str, BaseModel, BaseModel]]) -> EvaluationResult:
    """Score request identification (IdentifiedRequest) on what was asked and supplied, apart from any decision."""
    return _evaluate_fields("request_identification", cases, ("capabilities", "order_ids", "new_address"))


def evaluate_tool_sequences(
    cases: list[tuple[str, list[ToolRequest], list[list[ToolRequest]]]]
) -> EvaluationResult:
    failures: list[FailureExample] = []
    for case_id, observed, valid_sequences in cases:
        observed_value = [request.model_dump() for request in observed]
        valid_values = [[request.model_dump() for request in sequence] for sequence in valid_sequences]
        # A scenario can allow several equally correct tool orders and argument combinations.
        if observed_value not in valid_values:
            failures.append(FailureExample(case_id, valid_values, observed_value))
    return EvaluationResult("tool_argument_correctness", len(cases) - len(failures), len(cases), failures)


def write_evaluation_results(path: Path, results: list[EvaluationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([result.as_dict() for result in results], indent=2), encoding="utf-8")


def _evaluate_fields(
    metric: str, cases: list[tuple[str, BaseModel, BaseModel]], fields: tuple[str, ...]
) -> EvaluationResult:
    failures: list[FailureExample] = []
    for case_id, expected, observed in cases:
        # Score only the fields relevant to this metric, not incidental model output.
        expected_value = {name: getattr(expected, name) for name in fields}
        observed_value = {name: getattr(observed, name) for name in fields}
        if expected_value != observed_value:
            failures.append(FailureExample(case_id, expected_value, observed_value))
    return EvaluationResult(metric, len(cases) - len(failures), len(cases), failures)