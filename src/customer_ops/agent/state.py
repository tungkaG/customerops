from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypedDict

from sqlalchemy.orm import Session

from customer_ops.agent.capabilities import IdentifiedRequest
from customer_ops.domain.schemas import DecisionProposal, DemoContext
from customer_ops.llm.client import LLMProvider
from customer_ops.llm.schemas import ChatMessage
from customer_ops.retrieval.service import PolicyRetriever


class AgentOutcome(StrEnum):
    # Outcomes of a run.
    RESPONSE_READY = "response_ready"
    PENDING_APPROVAL = "pending_approval"
    DECLINED = "declined"
    CLARIFICATION_NEEDED = "clarification_needed"
    ESCALATED = "escalated"
    BLOCKED = "blocked"
    INVALID_TOOL_REQUEST = "invalid_tool_request"
    STEP_LIMIT_REACHED = "step_limit_reached"
    INVALID_DECISION = "invalid_decision"
    MODEL_ERROR = "model_error"
    # Outcomes of continuing after an operator handled a pending action.
    ACTION_EXECUTED = "action_executed"
    ACTION_REJECTED = "action_rejected"
    ACTION_EXPIRED = "action_expired"
    ACTION_FAILED = "action_failed"


@dataclass(frozen=True)
class AgentRuntime:
    """Trusted dependencies. The customer context comes from the caller and is never stored in model-visible state."""

    session: Session
    context: DemoContext
    provider: LLMProvider
    retriever: PolicyRetriever | None = None
    # Two separate budgets. max_required_calls bounds the workflow's required evidence collection, which comes first.
    # max_steps bounds only the model's optional tool rounds afterwards; required calls never use it up.
    max_steps: int = 4
    max_required_calls: int = 8


class ToolCallRecord(TypedDict):
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    source: str  # "workflow_required" or "model_selected"


class AgentState(TypedDict, total=False):
    """Everything a run produces. Identity, roles, and the database session are deliberately absent."""

    ticket_id: str
    run_id: str
    customer_message: str
    messages: list[ChatMessage]
    facts: dict[str, Any]
    identified_request: IdentifiedRequest  # What the customer asked for; separate from the final decision.
    required_done: bool
    resolved_order_ids: list[str]  # Orders this request is about, fixed before the model's optional tools can add others.
    required_calls: int
    steps: int
    gather_done: bool
    tool_calls: list[ToolCallRecord]
    policy_evidence: list[dict[str, Any]]
    retrieval_source: str
    decision: DecisionProposal
    validation: dict[str, str]
    pending_action_id: str
    outcome: AgentOutcome
    outcome_reason: str
    response_draft: str
