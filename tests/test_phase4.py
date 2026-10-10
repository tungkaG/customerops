from __future__ import annotations

import itertools
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from customer_ops.agent.capabilities import CAPABILITIES, IDENTIFY_SYSTEM_PROMPT, CapabilityId, IdentifiedRequest
from customer_ops.agent.graph import run_agent_graph
from customer_ops.agent.operations import OPERATIONS
from customer_ops.agent.state import AgentOutcome, AgentRuntime, AgentState
from customer_ops.agent.tools import READ_TOOLS, NoArguments, ReadTool, tool_definitions
from customer_ops.agent.trace import audit_trail, format_run
from customer_ops.agent.workflow import DECISION_SYSTEM_PROMPT, act_step, agent_result, continue_after_approval, run_agent, start_step
from customer_ops.database.models import Customer, Order, PendingAction, Refund, Ticket
from customer_ops.database.seed import seed_demo
from customer_ops.database.session import initialize_database, make_engine, make_session_factory
from customer_ops.domain.schemas import (
    OPERATION_ACTIONS, ActionType, Address, DecisionAction, DecisionProposal, DemoContext, OperationAction, Role, ToolRequest,
)
from customer_ops.llm.schemas import ChatMessage
from customer_ops.retrieval.embeddings import HashingEmbedder
from customer_ops.retrieval.service import PolicyRetriever
from customer_ops.services.actions import ActionService
from customer_ops.services.eligibility import check_eligibility
from scripted_provider import ScriptedProvider

ROOT = Path(__file__).parents[1]
GOLD = DemoContext(actor_id="cust-gold-10-day", role=Role.CUSTOMER, customer_id="cust-gold-10-day")
STANDARD = DemoContext(actor_id="cust-standard-10-day", role=Role.CUSTOMER, customer_id="cust-standard-10-day")
OPERATOR = DemoContext(actor_id="operator-1", role=Role.OPERATOR)
ADDRESS = {"line1": "5 New Street", "city": "Hamburg", "postal_code": "20095", "country": "DE"}
REFUND_POLICY = "POL-REFUND-DELAYED"


def _retriever() -> PolicyRetriever:
    return PolicyRetriever.from_directory(
        ROOT / "data" / "policies", ROOT / "data" / "policy_registry.json", HashingEmbedder(), date(2026, 1, 20)
    )


def _runtime(session: Session, provider: ScriptedProvider, context: DemoContext = GOLD, **kwargs) -> AgentRuntime:
    return AgentRuntime(session=session, context=context, provider=provider, retriever=_retriever(), **kwargs)


def _ticket(session: Session, ticket_id: str, context: DemoContext, message: str) -> str:
    session.add(Ticket(id=ticket_id, customer_id=context.customer_id, original_message=message))
    session.commit()
    return ticket_id


def _get(order_id: str) -> ToolRequest:
    return ToolRequest(tool_name="get_order", arguments={"order_id": order_id})


def _search(query: str, category: str | None = None) -> ToolRequest:
    return ToolRequest(tool_name="search_policy", arguments={"query": query, **({"category": category} if category else {})})


REFUND_SEARCH = _search("delayed order refund eligibility", "refund")


def _decision(action: DecisionAction, order_id: str | None, refs: list[str] | None = None, **extra) -> DecisionProposal:
    return DecisionProposal(action=action, order_id=order_id, policy_references=refs or [], justification="Scripted test decision.", **extra)


def _reject(order_id: str | None, refused: OperationAction, refs: list[str] | None = None, **extra) -> DecisionProposal:
    return DecisionProposal(
        action=DecisionAction.REJECT_REQUEST, rejected_action=refused, order_id=order_id,
        policy_references=[REFUND_POLICY] if refs is None else refs, justification="MODEL CLAIM: this is not allowed.", **extra,
    )


def _pending_count(session: Session) -> int:
    return session.scalar(select(func.count(PendingAction.id)))


@pytest.fixture(params=[run_agent, run_agent_graph], ids=["python", "langgraph"])
def run(request):
    """Every scenario runs through the plain-function workflow and through the LangGraph graph."""
    return request.param


# 1. A status inquiry is answered from the database record, and the tool result is fed back to the model.
def test_status_inquiry_produces_an_accurate_draft(session, run) -> None:
    ticket = _ticket(session, "t-status", GOLD, "Where is order-gold-10-day?")
    provider = ScriptedProvider(identification=_identify("status", order_ids=("order-gold-10-day",)), decision=_decision(DecisionAction.RESPOND_WITH_STATUS, "order-gold-10-day"))

    state = run(_runtime(session, provider), ticket)

    assert state["outcome"] is AgentOutcome.RESPONSE_READY
    assert "currently delayed" in state["response_draft"] and "199.00 EUR" in state["response_draft"]
    assert state["policy_evidence"] == [] and state["retrieval_source"] == "not_requested"
    assert "policy_retrieval_not_requested" in [event for event, _ in audit_trail(session, state["run_id"])]
    assert "Policy evidence" not in provider.decision_conversations[0][1].content
    assert _pending_count(session) == 0
    assert [m.role for m in provider.tool_conversations[0]][-2:] == ["assistant", "tool"]
    assert "199.00 EUR" in provider.tool_conversations[0][-1].content
    assert session.get(Ticket, ticket).workflow_status == "resolved"


# 2. An eligible Gold refund becomes a pending action for the recorded amount and does not execute.
def test_gold_delayed_refund_becomes_a_pending_action(session, run) -> None:
    ticket = _ticket(session, "t-refund", GOLD, "My delayed order order-gold-10-day needs a refund.")
    provider = ScriptedProvider(identification=_identify("refund", order_ids=("order-gold-10-day",)), decision=_decision(DecisionAction.PROPOSE_REFUND, "order-gold-10-day", [REFUND_POLICY]))

    state = run(_runtime(session, provider), ticket)

    action = session.get(PendingAction, state["pending_action_id"])
    assert state["outcome"] is AgentOutcome.PENDING_APPROVAL and state["validation"]["status"] == "accepted"
    assert (action.status, action.action_type, action.normalized_arguments["order_id"]) == ("pending", "refund", "order-gold-10-day")
    assert state["retrieval_source"] == "workflow_required"
    assert "199.00 EUR" in state["response_draft"] and "Nothing is refunded until" in state["response_draft"]
    assert session.scalar(select(func.count(Refund.id))) == 0
    assert session.get(Ticket, ticket).workflow_status == "pending_approval"
    assert session.get(Ticket, ticket).final_response_draft is None


# 3. The same delay for a Standard customer fails the deterministic rule even when the model proposes a refund.
def test_standard_customer_cannot_receive_the_same_refund(session, run) -> None:
    ticket = _ticket(session, "t-std", STANDARD, "Refund order-standard-10-day please.")
    provider = ScriptedProvider(identification=_identify("refund", order_ids=("order-standard-10-day",)), decision=_decision(DecisionAction.PROPOSE_REFUND, "order-standard-10-day", [REFUND_POLICY]))

    state = run(_runtime(session, provider, STANDARD), ticket)

    assert state["outcome"] is AgentOutcome.DECLINED and state["validation"]["status"] == "rejected"
    assert "strictly greater than 14 days" in state["outcome_reason"]
    assert _pending_count(session) == 0 and session.scalar(select(func.count(Refund.id))) == 0


# 4. A processing order cancellation becomes a pending proposal with the cancellation policy collected by the workflow.
def test_processing_cancellation_becomes_a_pending_proposal(session, run) -> None:
    ticket = _ticket(session, "t-cancel", GOLD, "Please cancel order-processing-123.")
    provider = ScriptedProvider(identification=_identify("cancellation", order_ids=("order-processing-123",)), decision=_decision(DecisionAction.PROPOSE_CANCELLATION, "order-processing-123", ["POL-CANCELLATION"]))

    state = run(_runtime(session, provider), ticket)

    assert state["outcome"] is AgentOutcome.PENDING_APPROVAL and state["retrieval_source"] == "workflow_required"
    assert [hit["policy_id"] for hit in state["policy_evidence"]] == ["POL-CANCELLATION"]
    assert session.get(PendingAction, state["pending_action_id"]).action_type == "cancellation"
    assert session.get(Order, "order-processing-123").status == "processing"


# 5. A processing order address change keeps the structured address in the pending action.
def test_processing_address_change_becomes_a_pending_proposal(session, run) -> None:
    ticket = _ticket(session, "t-address", GOLD, "Ship order-processing-123 to 5 New Street, 20095 Hamburg, DE.")
    provider = ScriptedProvider(identification=_identify("address_change", order_ids=("order-processing-123",), address=True), decision=_decision(DecisionAction.PROPOSE_ADDRESS_CHANGE, "order-processing-123", ["POL-ADDRESS-CHANGE"], proposed_address=Address(**ADDRESS)))

    state = run(_runtime(session, provider), ticket)

    action = session.get(PendingAction, state["pending_action_id"])
    assert state["outcome"] is AgentOutcome.PENDING_APPROVAL
    assert action.normalized_arguments["new_address"] == ADDRESS
    assert session.get(Order, "order-processing-123").shipping_address["city"] == "Berlin"


# 6. Another customer's order is not readable, leaks nothing, and cannot be used in a proposal.
def test_another_customers_order_is_not_accessible(session, run) -> None:
    ticket = _ticket(session, "t-foreign", STANDARD, "Refund order-gold-10-day.")
    provider = ScriptedProvider(identification=_identify("refund", order_ids=("order-gold-10-day",)), decision=_decision(DecisionAction.PROPOSE_REFUND, "order-gold-10-day", [REFUND_POLICY]))

    state = run(_runtime(session, provider, STANDARD), ticket)

    assert state["tool_calls"][0]["result"] == {"found": False, "message": "Order not found or inaccessible."}
    assert state["facts"]["orders"] == {}
    assert state["outcome"] is AgentOutcome.BLOCKED
    assert _pending_count(session) == 0


# 7. Missing or ambiguous order information ends with a request for clarification, not a guess.
def test_missing_order_information_requests_clarification(session, run) -> None:
    ticket = _ticket(session, "t-vague", GOLD, "I want a refund.")
    listing = ToolRequest(tool_name="get_customer_orders", arguments={})
    provider = ScriptedProvider(tool_rounds=[[listing]], decision=_decision(DecisionAction.ASK_CLARIFICATION, None, missing_information=["which order you mean"]))

    state = run(_runtime(session, provider), ticket)

    assert state["outcome"] is AgentOutcome.CLARIFICATION_NEEDED
    assert "which order you mean" in state["response_draft"]
    assert len(state["facts"]["orders"]) >= 4 and _pending_count(session) == 0


# 7b. A mutation decision that names no order is not guessed into an order; it asks the customer.
def test_mutation_decision_without_an_order_asks_for_clarification(session, run) -> None:
    ticket = _ticket(session, "t-noorder", GOLD, "Refund please.")
    provider = ScriptedProvider(decision=_decision(DecisionAction.PROPOSE_REFUND, None, [REFUND_POLICY]))

    state = run(_runtime(session, provider), ticket)

    assert state["outcome"] is AgentOutcome.CLARIFICATION_NEEDED and _pending_count(session) == 0


# 8. After approval, continuation reports the persisted result. Before that, it leaves the action paused.
def test_continuation_follows_the_persisted_outcome(session) -> None:
    ticket = _ticket(session, "t-cont", GOLD, "Refund order-gold-10-day.")
    provider = ScriptedProvider(identification=_identify("refund", order_ids=("order-gold-10-day",)), decision=_decision(DecisionAction.PROPOSE_REFUND, "order-gold-10-day", [REFUND_POLICY]))
    action_id = run_agent(_runtime(session, provider), ticket)["pending_action_id"]

    paused = continue_after_approval(session, GOLD, action_id)
    assert paused.outcome == "pending_approval" and session.get(PendingAction, action_id).status == "pending"
    assert session.scalar(select(func.count(Refund.id))) == 0 and session.get(Ticket, ticket).final_response_draft is None

    ActionService(session).approve(action_id, OPERATOR)
    done = continue_after_approval(session, GOLD, action_id)

    assert done.outcome == "action_executed"
    assert "199.00 EUR" in done.response_draft and "No real money was transferred" in done.response_draft
    assert session.scalar(select(Refund.amount_cents).where(Refund.order_id == "order-gold-10-day")) == 19_900
    assert session.get(Ticket, ticket).workflow_status == "resolved"


# 9. A rejected proposal leaves the order unchanged and continuation says so.
def test_rejection_leaves_business_records_unchanged(session) -> None:
    ticket = _ticket(session, "t-reject", GOLD, "Cancel order-processing-123.")
    provider = ScriptedProvider(identification=_identify("cancellation", order_ids=("order-processing-123",)), decision=_decision(DecisionAction.PROPOSE_CANCELLATION, "order-processing-123", ["POL-CANCELLATION"]))
    action_id = run_agent(_runtime(session, provider), ticket)["pending_action_id"]

    ActionService(session).reject(action_id, OPERATOR)
    result = continue_after_approval(session, GOLD, action_id)

    assert result.outcome == "action_rejected" and "Nothing was changed" in result.response_draft
    assert session.get(Order, "order-processing-123").status == "processing"
    assert session.scalar(select(func.count(Refund.id))) == 0


# 9b. Continuation reports an expired action honestly, and another customer cannot read it.
def test_continuation_reports_expiry_and_is_scoped_to_the_customer(session) -> None:
    ticket = _ticket(session, "t-expire", GOLD, "Cancel order-processing-123.")
    provider = ScriptedProvider(identification=_identify("cancellation", order_ids=("order-processing-123",)), decision=_decision(DecisionAction.PROPOSE_CANCELLATION, "order-processing-123", ["POL-CANCELLATION"]))
    action_id = run_agent(_runtime(session, provider), ticket)["pending_action_id"]
    order = session.get(Order, "order-processing-123")
    order.status, order.version = "shipped", order.version + 1
    session.commit()
    ActionService(session).approve(action_id, OPERATOR)

    assert continue_after_approval(session, GOLD, action_id).outcome == "action_expired"
    with pytest.raises(LookupError):
        continue_after_approval(session, STANDARD, action_id)


# 10a. Unknown tools and invalid or identity-injecting arguments end the run before any decision or mutation.
@pytest.mark.parametrize("bad_request", [
    ToolRequest(tool_name="execute_sql", arguments={"query": "DROP TABLE orders"}),
    ToolRequest(tool_name="approve_action", arguments={"action_id": "x"}),
    ToolRequest(tool_name="get_order", arguments={"order_id": 7}),
    ToolRequest(tool_name="get_order", arguments={"order_id": "order-gold-10-day", "customer_id": "cust-standard-10-day"}),
    ToolRequest(tool_name="get_order", arguments={}),
], ids=["sql", "approval", "wrong-type", "identity-injection", "missing-argument"])
def test_invalid_tool_requests_terminate_without_mutation(session, run, bad_request) -> None:
    ticket = _ticket(session, "t-bad", GOLD, "Refund order-gold-10-day.")
    provider = ScriptedProvider(tool_rounds=[[bad_request]])

    state = run(_runtime(session, provider), ticket)

    assert state["outcome"] is AgentOutcome.INVALID_TOOL_REQUEST and "decision" not in state
    assert state["tool_calls"] == [] and not provider.decision_conversations
    assert _pending_count(session) == 0 and session.scalar(select(func.count(Refund.id))) == 0
    assert ("tool_request_rejected" in [event for event, _ in audit_trail(session, state["run_id"])])


# 10b. A valid request in the same batch is not executed once another request in it is invalid.
def test_one_invalid_request_prevents_the_whole_batch(session, run) -> None:
    ticket = _ticket(session, "t-batch", GOLD, "Refund order-gold-10-day.")
    provider = ScriptedProvider(tool_rounds=[[_get("order-gold-10-day"), ToolRequest(tool_name="drop", arguments={})]])

    state = run(_runtime(session, provider), ticket)

    assert state["outcome"] is AgentOutcome.INVALID_TOOL_REQUEST and state["tool_calls"] == []


# 10c. A model that keeps requesting tools stops at the step limit without any mutation.
def test_step_exhaustion_terminates_without_mutation(session, run) -> None:
    ticket = _ticket(session, "t-loop", GOLD, "Refund order-gold-10-day.")
    provider = ScriptedProvider(tool_rounds=itertools.repeat([ToolRequest(tool_name="get_customer_orders", arguments={})]))

    state = run(_runtime(session, provider, max_steps=2), ticket)

    assert state["outcome"] is AgentOutcome.STEP_LIMIT_REACHED and state["steps"] == 2
    assert len(provider.tool_conversations) == 3 and not provider.decision_conversations
    assert _pending_count(session) == 0


# Model failures stay visible: an altered ID is not in the resolved orders, so it is rejected and never repaired.
def test_altered_order_id_is_not_repaired(session, run) -> None:
    ticket = _ticket(session, "t-alter", GOLD, "Refund order-gold-10-day.")
    provider = ScriptedProvider(identification=_identify("refund", order_ids=("order-gold-10-day",)), decision=_decision(DecisionAction.PROPOSE_REFUND, "gold-10-day", [REFUND_POLICY]))

    state = run(_runtime(session, provider), ticket)

    assert state["outcome"] is AgentOutcome.INVALID_DECISION and "not resolved" in state["outcome_reason"] and _pending_count(session) == 0


# Policy references must be real, retrieved evidence that includes the applicable policy.
@pytest.mark.parametrize("refs", [["POL-MADE-UP"], [], ["POL-CANCELLATION"]], ids=["invented", "none", "wrong-policy"])
def test_unsupported_policy_references_are_rejected(session, run, refs) -> None:
    ticket = _ticket(session, "t-refs", GOLD, "My delayed order needs a refund.")
    provider = ScriptedProvider(identification=_identify("refund", order_ids=("order-gold-10-day",)), decision=_decision(DecisionAction.PROPOSE_REFUND, "order-gold-10-day", refs))

    state = run(_runtime(session, provider), ticket)

    assert state["outcome"] is AgentOutcome.INVALID_DECISION and "policies" in state["outcome_reason"] and _pending_count(session) == 0


# The workflow always collects policy for an operation, so the no-evidence guard is exercised on the act step directly.
@pytest.mark.parametrize("decision", [
    _decision(DecisionAction.PROPOSE_REFUND, "order-gold-10-day", [REFUND_POLICY]),
    _reject("order-gold-10-day", OperationAction.PROPOSE_REFUND),
], ids=["proposal", "rejection"])
def test_a_decision_without_policy_evidence_never_proposes(session, decision) -> None:
    runtime = _runtime(session, ScriptedProvider())
    state = start_step({"ticket_id": _ticket(session, "t-noevidence", GOLD, "My delayed order needs a refund.")}, runtime)
    state.update(identified_request=_identify("refund", order_ids=("order-gold-10-day",)), resolved_order_ids=["order-gold-10-day"],
                 policy_evidence=[], decision=decision)

    outcome = act_step(state, runtime)["outcome"]

    assert outcome in {AgentOutcome.ESCALATED, AgentOutcome.INVALID_DECISION} and _pending_count(session) == 0


# Provider failures and schema-invalid model output end the run visibly and change nothing.
@pytest.mark.parametrize("failure", [TimeoutError("provider timed out"), ValueError("not json")], ids=["outage", "bad-output"])
def test_model_failures_terminate_without_mutation(session, run, failure) -> None:
    ticket = _ticket(session, "t-fail", GOLD, "My delayed order needs a refund.")
    provider = ScriptedProvider(decision=failure)

    state = run(_runtime(session, provider), ticket)

    assert state["outcome"] is AgentOutcome.MODEL_ERROR and type(failure).__name__ in state["outcome_reason"]
    assert _pending_count(session) == 0


# The audit trail records tool use, evidence, the decision, its validation, and the pending action ID.
def test_audit_events_capture_the_run(session) -> None:
    ticket = _ticket(session, "t-audit", GOLD, "My delayed order order-gold-10-day needs a refund.")
    provider = ScriptedProvider(identification=_identify("refund", order_ids=("order-gold-10-day",)), decision=_decision(DecisionAction.PROPOSE_REFUND, "order-gold-10-day", [REFUND_POLICY]))

    state = run_agent(_runtime(session, provider), ticket)
    events = audit_trail(session, state["run_id"])
    by_type = {event: payload for event, payload in events}

    assert [event for event, _ in events][0] == "agent_run_started" and events[-1][0] == "agent_run_completed"
    assert by_type["tool_requested"]["tool_name"] == "search_policy"  # The last request wins in this dict.
    assert [payload["tool_name"] for event, payload in events if event == "tool_requested"] == ["get_order", "search_policy"]
    assert [payload["summary"] for event, payload in events if event == "tool_result"][0] == {"found": True, "order_id": "order-gold-10-day"}
    assert by_type["policy_retrieved"]["source"] == "workflow_required"
    assert by_type["policy_retrieved"]["searches"][0]["tool_name"] == "search_policy"
    assert any(citation.startswith("POL-REFUND-DELAYED#") for citation in by_type["policy_retrieved"]["citations"])
    assert by_type["decision_proposed"]["action"] == "propose_refund"
    assert by_type["decision_validated"]["pending_action_id"] == state["pending_action_id"]
    assert by_type["agent_run_completed"]["pending_action_id"] == state["pending_action_id"]
    assert "action_proposed" in by_type
    assert "Run " + state["run_id"] in format_run(state)


# The model sees verified facts in EUR and policy evidence with citations, never raw cents or the trusted context.
def test_decision_prompt_contains_verified_facts_and_evidence(session) -> None:
    ticket = _ticket(session, "t-prompt", GOLD, "My delayed order order-gold-10-day needs a refund.")
    provider = ScriptedProvider(tool_rounds=[[_get("order-gold-10-day"), REFUND_SEARCH]], decision=_decision(DecisionAction.ESCALATE, None))

    run_agent(_runtime(session, provider), ticket)
    prompt = provider.decision_conversations[0][1].content

    assert "199.00 EUR" in prompt and "19900" not in prompt and '"tier": "gold"' in prompt
    assert "[POL-REFUND-DELAYED] (POL-REFUND-DELAYED#" in prompt and "strictly greater than 7 days" in prompt
    assert "cust-gold-10-day" not in prompt


# The model can only read: no proposal, approval, SQL, or identity arguments exist in its tool schemas.
def test_model_tools_are_read_only_and_carry_no_identity() -> None:
    assert set(READ_TOOLS) == {"get_order", "get_customer_orders", "search_policy"}
    for definition in tool_definitions():
        assert not {"customer_id", "ticket_id", "role", "actor_id"} & set(definition.parameters.get("properties", {}))
        assert definition.parameters["additionalProperties"] is False


# Trusted context stays out of model-writable state.
def test_trusted_context_is_not_part_of_agent_state(session) -> None:
    ticket = _ticket(session, "t-state", GOLD, "Where is order-gold-10-day?")
    provider = ScriptedProvider(decision=_decision(DecisionAction.ESCALATE, None))

    run_agent(_runtime(session, provider), ticket)

    assert not {"context", "customer_id", "role", "session", "actor_id"} & set(AgentState.__annotations__)


# A ticket that belongs to another customer cannot be run.
def test_running_another_customers_ticket_is_refused(session, run) -> None:
    with pytest.raises(LookupError):
        run(_runtime(session, ScriptedProvider(), STANDARD), "ticket-gold-refund")


# The two runners produce the same observable results on identical, fresh databases.
def test_python_and_langgraph_runners_agree() -> None:
    results = []
    for runner in (run_agent, run_agent_graph):
        engine = make_engine("sqlite://")
        initialize_database(engine)
        with make_session_factory(engine)() as database_session:
            seed_demo(database_session)
            ticket = _ticket(database_session, "t-same", GOLD, "My delayed order order-gold-10-day needs a refund.")
            provider = ScriptedProvider(identification=_identify("refund", order_ids=("order-gold-10-day",)), decision=_decision(DecisionAction.PROPOSE_REFUND, "order-gold-10-day", [REFUND_POLICY]))
            state = runner(_runtime(database_session, provider), ticket, "run-fixed")
            results.append((state["outcome"], state["policy_evidence"], state["decision"], state["response_draft"], state["tool_calls"],
                            [event for event, _ in audit_trail(database_session, "run-fixed")]))
    assert results[0] == results[1]


# agent_result summarizes a run with the existing AgentResult contract.
def test_agent_result_uses_the_existing_contract(session) -> None:
    ticket = _ticket(session, "t-result", GOLD, "Refund order-gold-10-day.")
    provider = ScriptedProvider(identification=_identify("refund", order_ids=("order-gold-10-day",)), decision=_decision(DecisionAction.PROPOSE_REFUND, "order-gold-10-day", [REFUND_POLICY]))

    result = agent_result(run_agent(_runtime(session, provider), ticket))

    assert (result.outcome, result.cited_policy_references) == ("pending_approval", [REFUND_POLICY]) and result.pending_action_id


# An eligible refund that the model rejects is not accepted, and nothing is persisted by checking it.
def test_eligible_refund_incorrectly_rejected_is_an_invalid_decision(session, run) -> None:
    ticket = _ticket(session, "t-wrong-reject", GOLD, "My delayed order order-gold-10-day needs a refund.")
    provider = ScriptedProvider(identification=_identify("refund", order_ids=("order-gold-10-day",)), decision=_reject("order-gold-10-day", OperationAction.PROPOSE_REFUND))

    state = run(_runtime(session, provider), ticket)

    assert state["outcome"] is AgentOutcome.INVALID_DECISION and state["validation"]["status"] == "rejected"
    assert "contradicts the eligibility rules" in state["outcome_reason"]
    assert _pending_count(session) == 0 and session.scalar(select(func.count(Refund.id))) == 0
    assert "MODEL CLAIM" not in state["response_draft"]


# A correct rejection is accepted, and the refusal is built from the rule's own reason, not the model's wording.
@pytest.mark.parametrize(("context", "order_id", "refused", "rule_reason"), [
    (STANDARD, "order-standard-10-day", OperationAction.PROPOSE_REFUND, "strictly greater than 14 days"),
    (GOLD, "order-gold-10-day", OperationAction.PROPOSE_CANCELLATION, "Only processing orders can be cancelled"),
], ids=["refund-delay", "cancel-delayed-order"])
def test_correct_rejection_uses_the_verified_eligibility_reason(session, run, context, order_id, refused, rule_reason) -> None:
    ticket = _ticket(session, "t-right-reject", context, "Please handle my order.")
    policy = "POL-CANCELLATION" if refused is OperationAction.PROPOSE_CANCELLATION else REFUND_POLICY
    capability = "cancellation" if refused is OperationAction.PROPOSE_CANCELLATION else "refund"
    provider = ScriptedProvider(identification=_identify(capability, order_ids=(order_id,)), decision=_reject(order_id, refused, [policy]))

    state = run(_runtime(session, provider, context), ticket)

    assert state["outcome"] is AgentOutcome.DECLINED and state["validation"]["status"] == "accepted"
    assert rule_reason in state["outcome_reason"] and rule_reason in state["response_draft"]
    assert "MODEL CLAIM" not in state["response_draft"] and _pending_count(session) == 0


# The schema requires rejected_action exactly when the action is reject_request.
def test_rejected_action_is_required_with_a_rejection_and_forbidden_otherwise() -> None:
    with pytest.raises(ValidationError, match="rejected_action"):
        DecisionProposal(action=DecisionAction.REJECT_REQUEST, justification="No operation named.")
    with pytest.raises(ValidationError, match="rejected_action"):
        DecisionProposal(action=DecisionAction.ESCALATE, rejected_action=OperationAction.PROPOSE_REFUND, justification="Misplaced.")
    with pytest.raises(ValidationError):
        DecisionProposal.model_validate_json('{"action": "reject_request", "rejected_action": "escalate", "justification": "x"}')


# Each rejection precondition fails closed: unidentified order, unresolved order, missing or wrong policy evidence.
@pytest.mark.parametrize(("decision", "outcome"), [
    (_reject(None, OperationAction.PROPOSE_REFUND), AgentOutcome.INVALID_DECISION),
    (_reject("order-gold-10-day", OperationAction.PROPOSE_REFUND, []), AgentOutcome.INVALID_DECISION),
    (_reject("order-gold-10-day", OperationAction.PROPOSE_REFUND, ["POL-MADE-UP"]), AgentOutcome.INVALID_DECISION),
    (_reject("order-gold-10-day", OperationAction.PROPOSE_CANCELLATION, [REFUND_POLICY]), AgentOutcome.INVALID_DECISION),
    (_reject("gold-10-day", OperationAction.PROPOSE_REFUND), AgentOutcome.INVALID_DECISION),
], ids=["no-order", "no-refs", "invented-ref", "wrong-policy", "altered-id"])
def test_rejection_preconditions_fail_closed(session, run, decision, outcome) -> None:
    ticket = _ticket(session, "t-reject-pre", GOLD, "Refund order-gold-10-day or cancel it.")
    provider = ScriptedProvider(identification=_identify("refund", "cancellation", order_ids=("order-gold-10-day",)), decision=decision)

    state = run(_runtime(session, provider), ticket)

    assert state["outcome"] is outcome and _pending_count(session) == 0


# A customer cannot get a rejection verified against another customer's order.
def test_rejection_of_another_customers_order_is_blocked(session, run) -> None:
    ticket = _ticket(session, "t-reject-foreign", STANDARD, "Refund order-gold-10-day.")
    provider = ScriptedProvider(identification=_identify("refund", order_ids=("order-gold-10-day",)), decision=_reject("order-gold-10-day", OperationAction.PROPOSE_REFUND))

    assert run(_runtime(session, provider, STANDARD), ticket)["outcome"] is AgentOutcome.BLOCKED


# A proposal or rejection must concern a capability the customer asked for.
@pytest.mark.parametrize(("identified", "decision"), [
    ("refund", _decision(DecisionAction.PROPOSE_CANCELLATION, "order-gold-10-day", ["POL-CANCELLATION"])),
    ("cancellation", _reject("order-gold-10-day", OperationAction.PROPOSE_REFUND)),
    ("status", _decision(DecisionAction.PROPOSE_REFUND, "order-gold-10-day", [REFUND_POLICY])),
], ids=["proposal-of-another-operation", "rejection-of-another-operation", "operation-from-a-status-request"])
def test_decisions_must_concern_a_requested_capability(session, run, identified, decision) -> None:
    ticket = _ticket(session, "t-scope-capability", GOLD, "About order-gold-10-day.")
    provider = ScriptedProvider(identification=_identify(identified, order_ids=("order-gold-10-day",)), decision=decision)

    state = run(_runtime(session, provider), ticket)

    assert state["outcome"] is AgentOutcome.INVALID_DECISION and "did not request" in state["outcome_reason"]
    assert _pending_count(session) == 0 and session.scalar(select(func.count(Refund.id))) == 0


# The decision's order must be one resolved for this request, not merely any order the customer owns.
@pytest.mark.parametrize("decision", [
    _decision(DecisionAction.PROPOSE_REFUND, "order-processing-123", [REFUND_POLICY]),
    _reject("order-processing-123", OperationAction.PROPOSE_REFUND),
    _decision(DecisionAction.RESPOND_WITH_STATUS, "order-processing-123"),
], ids=["proposal", "rejection", "status"])
def test_decision_order_must_be_resolved_for_the_request(session, run, decision) -> None:
    ticket = _ticket(session, "t-scope-order", GOLD, "Refund order-gold-10-day.")
    identified = _identify("status" if decision.action is DecisionAction.RESPOND_WITH_STATUS else "refund", order_ids=("order-gold-10-day",))
    provider = ScriptedProvider(identification=identified, decision=decision)

    state = run(_runtime(session, provider), ticket)

    assert state["resolved_order_ids"] == ["order-gold-10-day"]
    assert state["outcome"] is AgentOutcome.INVALID_DECISION and "not resolved" in state["outcome_reason"]
    assert _pending_count(session) == 0


# An order the model fetched for itself does not become resolved for the request.
def test_a_model_selected_order_does_not_become_resolved(session, run) -> None:
    ticket = _ticket(session, "t-scope-model-order", GOLD, "Refund order-gold-10-day.")
    provider = ScriptedProvider(
        identification=_identify("refund", order_ids=("order-gold-10-day",)), tool_rounds=[[_get("order-processing-123")]],
        decision=_decision(DecisionAction.PROPOSE_REFUND, "order-processing-123", [REFUND_POLICY]),
    )

    state = run(_runtime(session, provider), ticket)

    assert "order-processing-123" in state["facts"]["orders"] and state["resolved_order_ids"] == ["order-gold-10-day"]
    assert state["outcome"] is AgentOutcome.INVALID_DECISION and _pending_count(session) == 0


# With no order written, the customer's only order is the resolved one.
def test_the_only_order_is_resolved_when_no_id_is_written(session, run) -> None:
    solo = DemoContext(actor_id="cust-solo", role=Role.CUSTOMER, customer_id="cust-solo")
    session.add(Customer(id="cust-solo", name="Solo", email="solo@example.test", tier="gold", default_address=ADDRESS))
    session.add(Order(id="order-solo", customer_id="cust-solo", amount_cents=5_000, currency="EUR", status="delayed",
                      expected_delivery_date=date(2026, 1, 1), shipping_address=ADDRESS))
    session.commit()
    ticket = _ticket(session, "t-scope-solo", solo, "Refund my order.")
    provider = ScriptedProvider(identification=_identify("refund"), decision=_decision(DecisionAction.PROPOSE_REFUND, "order-solo", [REFUND_POLICY]))

    state = run(_runtime(session, provider, solo), ticket)

    assert state["resolved_order_ids"] == ["order-solo"] and state["outcome"] is AgentOutcome.PENDING_APPROVAL


# Status alone never completes an identified operation request, even when the order is right.
@pytest.mark.parametrize("capabilities", [("refund",), ("status", "refund"), ("cancellation",)], ids=["refund", "status-and-refund", "cancellation"])
def test_status_alone_does_not_complete_an_operation_request(session, run, capabilities) -> None:
    ticket = _ticket(session, "t-scope-status", GOLD, "Status and an operation on order-gold-10-day.")
    provider = ScriptedProvider(identification=_identify(*capabilities, order_ids=("order-gold-10-day",)),
                                decision=_decision(DecisionAction.RESPOND_WITH_STATUS, "order-gold-10-day"))

    state = run(_runtime(session, provider), ticket)

    assert state["outcome"] is AgentOutcome.INVALID_DECISION and "status alone" in state["outcome_reason"].lower()
    assert state["validation"]["status"] == "rejected" and "delayed" not in state["response_draft"]
    assert session.get(Ticket, ticket).workflow_status == "needs_review"


# Clarification and escalation stay valid answers to an operation request.
@pytest.mark.parametrize(("decision", "outcome"), [
    (_decision(DecisionAction.ASK_CLARIFICATION, None, missing_information=["the reason"]), AgentOutcome.CLARIFICATION_NEEDED),
    (_decision(DecisionAction.ESCALATE, None), AgentOutcome.ESCALATED),
], ids=["clarification", "escalation"])
def test_clarification_and_escalation_remain_valid_for_an_operation_request(session, run, decision, outcome) -> None:
    ticket = _ticket(session, "t-scope-alternatives", GOLD, "Refund order-gold-10-day.")
    provider = ScriptedProvider(identification=_identify("refund", order_ids=("order-gold-10-day",)), decision=decision)

    state = run(_runtime(session, provider), ticket)

    assert state["outcome"] is outcome and state["validation"]["status"] == "accepted" and _pending_count(session) == 0


# Proposal and rejection validation read one registry, and the model-facing text is generated from it.
def test_operation_registry_is_the_single_source_of_operations() -> None:
    assert set(OPERATIONS) == set(OPERATION_ACTIONS) == {DecisionAction(member.value) for member in OperationAction}
    assert {operation.action_type for operation in OPERATIONS.values()} == set(ActionType)
    for operation in OPERATIONS.values():
        assert operation.action.value in DECISION_SYSTEM_PROMPT and operation.meaning in DECISION_SYSTEM_PROMPT
    schema = DecisionProposal.model_json_schema()
    assert set(schema["$defs"]["OperationAction"]["enum"]) == {action.value for action in OPERATIONS}


# Rejection and proposal checks use the one shared, read-only eligibility function.
def test_shared_eligibility_check_is_side_effect_free_and_matches_the_rules(session) -> None:
    gold_delayed, processing, standard_delayed = (session.get(Order, order) for order in ("order-gold-10-day", "order-processing-123", "order-standard-10-day"))

    assert check_eligibility(session, ActionType.REFUND, gold_delayed).allowed
    assert not check_eligibility(session, ActionType.REFUND, standard_delayed).allowed
    assert check_eligibility(session, ActionType.CANCELLATION, processing).allowed
    assert not check_eligibility(session, ActionType.CANCELLATION, gold_delayed).allowed
    assert check_eligibility(session, ActionType.ADDRESS_CHANGE, processing, ADDRESS).allowed
    assert not check_eligibility(session, ActionType.ADDRESS_CHANGE, processing, {}).allowed
    assert _pending_count(session) == 0 and session.scalar(select(func.count(Refund.id))) == 0


# A new read tool needs no workflow change: its own fact merging, audit summary, and evidence hooks are used.
def test_a_registered_read_tool_works_without_workflow_changes(session, monkeypatch) -> None:
    ping = ReadTool(
        "ping_notes", "A hypothetical read tool.", NoArguments, lambda runtime, arguments: {"notes": ["gift wrap"]},
        merge_facts=lambda facts, result: facts.update(notes=result["notes"]),
        summarize=lambda result: {"note_count": len(result["notes"])},
        evidence=lambda result: [{"policy_id": "POL-NOTE", "chunk_id": "POL-NOTE:1", "citation": "POL-NOTE#1", "text": "note", "score": 1.0}],
    )
    monkeypatch.setitem(READ_TOOLS, "ping_notes", ping)
    ticket = _ticket(session, "t-ping", GOLD, "Anything on my account?")
    provider = ScriptedProvider(tool_rounds=[[ToolRequest(tool_name="ping_notes", arguments={})]], decision=_decision(DecisionAction.ESCALATE, None))

    state = run_agent(_runtime(session, provider), ticket)

    assert state["facts"]["notes"] == ["gift wrap"]
    assert [citation for hit in state["policy_evidence"] for citation in [hit["citation"]]] == ["POL-NOTE#1"]
    assert dict(audit_trail(session, state["run_id"]))["tool_result"]["summary"] == {"note_count": 1}


def _identify(*capabilities: str, order_ids: tuple[str, ...] = (), address: bool = False) -> IdentifiedRequest:
    return IdentifiedRequest(
        capabilities=[CapabilityId(capability) for capability in capabilities], order_ids=list(order_ids),
        new_address=Address(**ADDRESS) if address else None,
    )


def _sources(state: AgentState) -> list[tuple[str, str]]:
    return [(call["source"], call["tool_name"]) for call in state["tool_calls"]]


# Status needs order facts only, so the workflow collects them and never searches policy.
def test_status_collects_order_facts_without_searching_policy(session, run) -> None:
    ticket = _ticket(session, "t-req-status", GOLD, "Where is order-gold-10-day?")
    provider = ScriptedProvider(identification=_identify("status", order_ids=("order-gold-10-day",)), decision=_decision(DecisionAction.RESPOND_WITH_STATUS, "order-gold-10-day"))

    state = run(_runtime(session, provider), ticket)

    assert _sources(state) == [("workflow_required", "get_order")]
    assert state["retrieval_source"] == "not_requested" and state["policy_evidence"] == []
    assert state["outcome"] is AgentOutcome.RESPONSE_READY
    # The required call is in the conversation as a well-formed assistant tool call plus its result.
    conversation = provider.tool_conversations[0]
    assert [message.role for message in conversation] == ["system", "user", "assistant", "tool"]
    assert conversation[2].tool_requests == [_get("order-gold-10-day")] and conversation[3].tool_name == "get_order"


# A refund needs order facts and refund policy, and the workflow gets both even though the model never searches.
def test_refund_collects_order_facts_and_refund_policy_when_the_model_does_not_search(session, run) -> None:
    ticket = _ticket(session, "t-req-refund", GOLD, "My order order-gold-10-day is late. Refund please.")
    provider = ScriptedProvider(identification=_identify("refund", order_ids=("order-gold-10-day",)), decision=_decision(DecisionAction.PROPOSE_REFUND, "order-gold-10-day", [REFUND_POLICY]))

    state = run(_runtime(session, provider), ticket)

    assert _sources(state) == [("workflow_required", "get_order"), ("workflow_required", "search_policy")]
    assert state["tool_calls"][1]["arguments"]["category"] == "refund"
    assert [hit["policy_id"] for hit in state["policy_evidence"]] == [REFUND_POLICY]
    assert state["retrieval_source"] == "workflow_required" and state["steps"] == 0
    assert state["outcome"] is AgentOutcome.PENDING_APPROVAL


# Cancellation and address change each collect their own applicable policy.
@pytest.mark.parametrize(("capability", "category", "policy", "action", "address"), [
    ("cancellation", "cancellation", "POL-CANCELLATION", DecisionAction.PROPOSE_CANCELLATION, False),
    ("address_change", "address", "POL-ADDRESS-CHANGE", DecisionAction.PROPOSE_ADDRESS_CHANGE, True),
], ids=["cancellation", "address-change"])
def test_cancellation_and_address_change_collect_their_applicable_policy(session, run, capability, category, policy, action, address) -> None:
    ticket = _ticket(session, "t-req-op", GOLD, "Please handle order-processing-123.")
    decision = _decision(action, "order-processing-123", [policy], **({"proposed_address": Address(**ADDRESS)} if address else {}))
    provider = ScriptedProvider(identification=_identify(capability, order_ids=("order-processing-123",), address=address), decision=decision)

    state = run(_runtime(session, provider), ticket)

    assert _sources(state) == [("workflow_required", "get_order"), ("workflow_required", "search_policy")]
    assert state["tool_calls"][1]["arguments"]["category"] == category
    assert [hit["policy_id"] for hit in state["policy_evidence"]] == [policy]
    assert state["outcome"] is AgentOutcome.PENDING_APPROVAL


# An address change without a supplied address asks for it before any decision is requested.
def test_address_change_without_an_address_asks_for_it(session, run) -> None:
    ticket = _ticket(session, "t-req-noaddr", GOLD, "Please change the address of order-processing-123.")
    provider = ScriptedProvider(identification=_identify("address_change", order_ids=("order-processing-123",)))

    state = run(_runtime(session, provider), ticket)

    assert state["outcome"] is AgentOutcome.CLARIFICATION_NEEDED and "new shipping address" in state["response_draft"]
    assert not provider.decision_conversations and _pending_count(session) == 0


# Status plus refund share one order requirement, so the order is fetched once and the policy once.
def test_mixed_status_and_refund_collect_shared_order_evidence_once(session, run) -> None:
    ticket = _ticket(session, "t-req-mixed", GOLD, "Where is order-gold-10-day, and I want a refund.")
    provider = ScriptedProvider(identification=_identify("status", "refund", order_ids=("order-gold-10-day",)), decision=_decision(DecisionAction.PROPOSE_REFUND, "order-gold-10-day", [REFUND_POLICY]))

    state = run(_runtime(session, provider), ticket)

    assert _sources(state) == [("workflow_required", "get_order"), ("workflow_required", "search_policy")]
    requested = [payload for event, payload in audit_trail(session, state["run_id"]) if event == "tool_requested"]
    assert [payload["tool_name"] for payload in requested].count("get_order") == 1
    prompt = provider.decision_conversations[0][1].content
    assert '"status"' in prompt and '"refund"' in prompt and "Identified request" in prompt
    assert state["outcome"] is AgentOutcome.PENDING_APPROVAL


# With no order named and several orders, the workflow lists them and asks, instead of picking one.
def test_ambiguous_orders_lead_to_clarification_without_guessing(session, run) -> None:
    ticket = _ticket(session, "t-req-ambiguous", GOLD, "I want a refund.")
    provider = ScriptedProvider(identification=_identify("refund"))

    state = run(_runtime(session, provider), ticket)

    assert ("workflow_required", "get_customer_orders") in _sources(state)
    assert state["outcome"] is AgentOutcome.CLARIFICATION_NEEDED and "which order you mean" in state["response_draft"]
    assert len(state["facts"]["orders"]) > 1 and not provider.decision_conversations and _pending_count(session) == 0


# A customer with exactly one order has an unambiguous order, so no ID is needed.
def test_a_single_order_is_resolved_without_an_id(session, run) -> None:
    solo = DemoContext(actor_id="cust-solo", role=Role.CUSTOMER, customer_id="cust-solo")
    session.add(Customer(id="cust-solo", name="Solo", email="solo@example.test", tier="gold", default_address=ADDRESS))
    session.add(Order(id="order-solo", customer_id="cust-solo", amount_cents=5_000, currency="EUR", status="delayed",
                      expected_delivery_date=date(2026, 1, 1), shipping_address=ADDRESS))
    session.commit()
    ticket = _ticket(session, "t-req-solo", solo, "Where is my order?")
    provider = ScriptedProvider(identification=_identify("status"), decision=_decision(DecisionAction.RESPOND_WITH_STATUS, "order-solo"))

    state = run(_runtime(session, provider, solo), ticket)

    assert list(state["facts"]["orders"]) == ["order-solo"] and state["outcome"] is AgentOutcome.RESPONSE_READY


# Missing or inaccessible orders keep the generic answer, reveal nothing, and request no decision.
@pytest.mark.parametrize(("context", "order_id"), [(STANDARD, "order-gold-10-day"), (GOLD, "ORD-12345")], ids=["foreign", "invented"])
def test_missing_or_inaccessible_orders_keep_the_generic_response(session, run, context, order_id) -> None:
    ticket = _ticket(session, "t-req-blocked", context, f"Refund {order_id}.")
    provider = ScriptedProvider(identification=_identify("refund", order_ids=(order_id,)))

    state = run(_runtime(session, provider, context), ticket)

    assert state["outcome"] is AgentOutcome.BLOCKED and state["tool_calls"][0]["result"]["found"] is False
    assert state["facts"]["orders"] == {} and "gold" not in state["response_draft"].lower()
    assert not provider.decision_conversations and _pending_count(session) == 0


# Required collection has its own budget and does not use up the model's optional rounds.
def test_required_collection_has_its_own_budget(session, run) -> None:
    ticket = _ticket(session, "t-req-budget", GOLD, "Refund order-gold-10-day.")
    over_budget = ScriptedProvider(identification=_identify("refund", order_ids=("order-gold-10-day",)))

    state = run(_runtime(session, over_budget, max_required_calls=1), ticket)

    assert state["outcome"] is AgentOutcome.STEP_LIMIT_REACHED and "more than 1" in state["outcome_reason"]
    assert state["tool_calls"] == [] and not over_budget.decision_conversations and _pending_count(session) == 0

    ticket = _ticket(session, "t-req-optional", GOLD, "Refund order-gold-10-day.")
    optional = ScriptedProvider(
        identification=_identify("refund", order_ids=("order-gold-10-day",)),
        tool_rounds=[[ToolRequest(tool_name="get_customer_orders", arguments={})]],
        decision=_decision(DecisionAction.PROPOSE_REFUND, "order-gold-10-day", [REFUND_POLICY]),
    )
    state = run(_runtime(session, optional, max_steps=1), ticket)

    assert state["required_calls"] == 2 and state["steps"] == 1 and state["outcome"] is AgentOutcome.PENDING_APPROVAL
    assert [source for source, _ in _sources(state)] == ["workflow_required", "workflow_required", "model_selected"]


# A failed identification ends the run visibly, with no evidence collected and nothing changed.
def test_identification_failure_is_a_model_error(session, run) -> None:
    ticket = _ticket(session, "t-req-error", GOLD, "Refund order-gold-10-day.")
    provider = ScriptedProvider(identification=ValueError("not json"))

    state = run(_runtime(session, provider), ticket)

    assert state["outcome"] is AgentOutcome.MODEL_ERROR and "ValueError" in state["outcome_reason"]
    assert state["tool_calls"] == [] and _pending_count(session) == 0


# Workflow-required and model-selected calls are told apart in state, audit events, and the trace.
def test_required_and_model_selected_calls_are_distinguished(session, run) -> None:
    ticket = _ticket(session, "t-req-trace", GOLD, "Refund order-gold-10-day.")
    provider = ScriptedProvider(
        identification=_identify("refund", order_ids=("order-gold-10-day",)),
        tool_rounds=[[ToolRequest(tool_name="get_customer_orders", arguments={})]],
        decision=_decision(DecisionAction.PROPOSE_REFUND, "order-gold-10-day", [REFUND_POLICY]),
    )

    state = run(_runtime(session, provider), ticket)
    events = audit_trail(session, state["run_id"])

    assert [payload["source"] for event, payload in events if event == "tool_requested"] == ["workflow_required", "workflow_required", "model_selected"]
    assert [payload["source"] for event, payload in events if event == "tool_result"] == ["workflow_required", "workflow_required", "model_selected"]
    trace = format_run(state)
    assert "[required] get_order" in trace and "[model] get_customer_orders" in trace and "Identified request" in trace


# The identification step only sees the customer's words, and its result is stored apart from the decision.
def test_identification_is_separate_from_the_decision(session) -> None:
    ticket = _ticket(session, "t-req-split", GOLD, "Where is order-gold-10-day?")
    provider = ScriptedProvider(identification=_identify("status", order_ids=("order-gold-10-day",)), decision=_decision(DecisionAction.ESCALATE, None))

    state = run_agent(_runtime(session, provider), ticket)

    conversation = provider.identification_conversations[0]
    assert [message.role for message in conversation] == ["system", "user"]
    assert conversation[1].content == "Where is order-gold-10-day?" and "Verified facts" not in conversation[1].content
    assert isinstance(state["identified_request"], IdentifiedRequest) and state["decision"].action is DecisionAction.ESCALATE
    assert "Reporting status alone does not fulfil" in DECISION_SYSTEM_PROMPT


# Capability IDs and descriptions come from one registry that includes status next to the operations.
def test_capabilities_are_generated_from_the_registry() -> None:
    assert set(CAPABILITIES) == {"status"} | {operation.action_type.value for operation in OPERATIONS.values()}
    assert {member.value for member in CapabilityId} == set(CAPABILITIES)
    assert set(IdentifiedRequest.model_json_schema()["$defs"]["CapabilityId"]["enum"]) == set(CAPABILITIES)
    assert all(capability.id in IDENTIFY_SYSTEM_PROMPT and capability.meaning in IDENTIFY_SYSTEM_PROMPT for capability in CAPABILITIES.values())
    request = IdentifiedRequest.model_validate_json('{"capabilities": ["refund", "status", "refund"], "order_ids": ["a", "a", "b"]}')
    assert [capability.value for capability in request.capabilities] == ["status", "refund"] and request.order_ids == ["a", "b"]


# Both runners behave identically when the workflow collects required evidence.
def test_runners_agree_with_required_evidence() -> None:
    results = []
    for runner in (run_agent, run_agent_graph):
        engine = make_engine("sqlite://")
        initialize_database(engine)
        with make_session_factory(engine)() as database_session:
            seed_demo(database_session)
            ticket = _ticket(database_session, "t-req-same", GOLD, "Refund order-gold-10-day.")
            provider = ScriptedProvider(
                identification=_identify("status", "refund", order_ids=("order-gold-10-day",)),
                tool_rounds=[[ToolRequest(tool_name="get_customer_orders", arguments={})]],
                decision=_decision(DecisionAction.PROPOSE_REFUND, "order-gold-10-day", [REFUND_POLICY]),
            )
            state = runner(_runtime(database_session, provider), ticket, "run-fixed")
            results.append((state["outcome"], state["identified_request"], state["tool_calls"], state["policy_evidence"],
                            state["retrieval_source"], state["decision"], [event for event, _ in audit_trail(database_session, "run-fixed")]))
    assert results[0] == results[1]


# Messages: tool results and assistant tool calls are valid turns; malformed combinations are not.
def test_chat_messages_support_tool_turns() -> None:
    call = ToolRequest(tool_name="get_order", arguments={"order_id": "o"})
    assert ChatMessage(role="assistant", content="", tool_requests=[call]).tool_requests == [call]
    assert ChatMessage(role="tool", content="{}", tool_name="get_order").tool_name == "get_order"
    for invalid in (dict(role="tool", content="x"), dict(role="user", content="x", tool_name="t"), dict(role="user", content=""),
                    dict(role="user", content="x", tool_requests=[call])):
        with pytest.raises(ValidationError):
            ChatMessage(**invalid)
