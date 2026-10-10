from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from customer_ops.agent.state import AgentState
from customer_ops.database.models import AuditEvent


def audit_trail(session: Session, run_id: str) -> list[tuple[str, dict[str, object]]]:
    """The structured audit events recorded for one run, oldest first."""
    events = session.scalars(select(AuditEvent).where(AuditEvent.run_id == run_id).order_by(AuditEvent.created_at))
    return [(event.event_type, event.payload) for event in events]


def format_run(state: AgentState) -> str:
    """A compact, readable trace: tool calls, evidence, decision, validation, and result. No hidden reasoning."""
    lines = [f"Run {state['run_id']} on {state['ticket_id']} -> outcome: {state['outcome'].value}"]
    lines.append(f"Customer message: {state['customer_message']}")
    identified = state.get("identified_request")
    if identified is not None:
        address = f", new_address={identified.new_address.model_dump()}" if identified.new_address else ""
        lines.append(
            f"Identified request: capabilities={[c.value for c in identified.capabilities]}, "
            f"order_ids={identified.order_ids}{address}, missing_details={identified.missing_details}"
        )
    required = sum(call["source"] == "workflow_required" for call in state["tool_calls"])
    lines.append(
        f"Tool calls ({required} workflow-required, {len(state['tool_calls']) - required} model-selected, "
        f"{state['steps']} optional model round(s) executed):"
    )
    for number, call in enumerate(state["tool_calls"], start=1):
        tag = "required" if call["source"] == "workflow_required" else "model"
        lines.append(f"  {number}. [{tag}] {call['tool_name']}({json.dumps(call['arguments'])}) -> {_clip(json.dumps(call['result'], default=str))}")
    if "policy_evidence" in state:
        if state["retrieval_source"] == "not_requested":
            lines.append("Policy evidence: none (retrieval not requested: no policy search was required or made)")
        else:
            lines.append(f"Policy evidence (source: {state['retrieval_source']}):")
            lines.extend(f"  {hit['score']:.3f} {hit['citation']}" for hit in state["policy_evidence"])
    decision = state.get("decision")
    if decision is not None:
        address = f", address={decision.proposed_address.model_dump()}" if decision.proposed_address else ""
        refused = f", rejected_action={decision.rejected_action.value}" if decision.rejected_action else ""
        lines.append(f"Decision: {decision.action.value}, order={decision.order_id}, policies={decision.policy_references}{address}{refused}")
        lines.append(f"  justification: {decision.justification}")
    if "validation" in state:
        lines.append(f"Validation: {state['validation']['status']} - {state['validation']['reason']}")
    lines.append(f"Result: {state['outcome'].value} - {state['outcome_reason']}")
    if "pending_action_id" in state:
        lines.append(f"  pending action: {state['pending_action_id']} (not executed)")
    lines.append(f"Response draft: {state['response_draft']}")
    return "\n".join(lines)


def _clip(text: str, limit: int = 220) -> str:
    return text if len(text) <= limit else text[:limit] + "..."
