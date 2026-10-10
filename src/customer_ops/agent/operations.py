"""The operations a customer can request, registered explicitly in one place.

Proposal validation and rejection validation both read this registry, and the model-facing prompt text is generated
from it. See the README section "Extending the agent" for what a new operation still needs elsewhere.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from customer_ops.agent.requirements import ADDRESS_PROVIDED, Requirement
from customer_ops.database.models import Order, PendingAction
from customer_ops.domain.rules import ADDRESS_POLICY_ID, CANCELLATION_POLICY_ID, REFUND_POLICY_ID, Eligibility
from customer_ops.domain.schemas import OPERATION_ACTIONS, ActionType, DecisionAction, DecisionProposal
from customer_ops.services.eligibility import check_eligibility
from customer_ops.tools.proposals import propose_address_change, propose_cancellation, propose_refund

if TYPE_CHECKING:
    from customer_ops.agent.state import AgentRuntime


def _nothing_missing(_decision: DecisionProposal) -> str | None:
    return None


@dataclass(frozen=True)
class Operation:
    action: DecisionAction  # The decision action that proposes this operation.
    action_type: ActionType  # How the persisted pending action is typed.
    policy_id: str  # The policy a proposal or rejection must cite and have retrieved.
    policy_category: str  # Restricts the evidence search to this operation's policies.
    policy_query: str  # The targeted search the workflow runs when the customer asks for this operation.
    label: str
    meaning: str  # Model-facing: when the customer is asking for this operation.
    propose: Callable[[AgentRuntime, str, Order, DecisionProposal, list[str]], PendingAction]
    pending_draft: Callable[[Order, DecisionProposal], str]
    executed_draft: Callable[[str, dict[str, Any]], str]
    # Returns what is missing for the operation, or None when the decision has everything the proposal needs.
    missing_input: Callable[[DecisionProposal], str | None] = _nothing_missing
    # Evidence needed beyond order facts and this operation's policy, such as a requested address.
    extra_requirements: tuple[Requirement, ...] = ()

    def check(self, session: Session, order: Order, decision: DecisionProposal) -> Eligibility:
        """Side-effect-free eligibility, shared by proposal validation, rejection validation, and approval."""
        address = decision.proposed_address.model_dump() if decision.proposed_address else None
        return check_eligibility(session, self.action_type, order, address)

    def refusal_draft(self, order: Order, reason: str) -> str:
        """Built from the verified eligibility reason, never from a model explanation."""
        return f"I'm sorry, but I can't request a {self.label} for order {order.id}: {reason}"


def _money(order: Order) -> str:
    return f"{Decimal(order.amount_cents) / 100:.2f} {order.currency}"


def _propose_refund(runtime: AgentRuntime, ticket_id: str, order: Order, _decision: DecisionProposal, refs: list[str]) -> PendingAction:
    return propose_refund(runtime.session, runtime.context, ticket_id, order.id, refs)


def _propose_cancellation(runtime: AgentRuntime, ticket_id: str, order: Order, _decision: DecisionProposal, refs: list[str]) -> PendingAction:
    return propose_cancellation(runtime.session, runtime.context, ticket_id, order.id, refs)


def _propose_address_change(runtime: AgentRuntime, ticket_id: str, order: Order, decision: DecisionProposal, refs: list[str]) -> PendingAction:
    return propose_address_change(runtime.session, runtime.context, ticket_id, order.id, decision.proposed_address.model_dump(), refs)


def _refund_pending(order: Order, _decision: DecisionProposal) -> str:
    return (f"I asked an operator to approve a full refund of {_money(order)} for order {order.id}. "
            "Nothing is refunded until it is approved, and demo refunds are simulated.")


def _cancellation_pending(order: Order, _decision: DecisionProposal) -> str:
    return f"I asked an operator to approve cancelling order {order.id}. The order stays unchanged until it is approved."


def _address_pending(order: Order, decision: DecisionProposal) -> str:
    address = decision.proposed_address
    return (f"I asked an operator to approve changing the shipping address of order {order.id} to {address.line1}, "
            f"{address.postal_code} {address.city}, {address.country}. The address stays unchanged until it is approved.")


def _refund_executed(order_id: str, result: dict[str, Any]) -> str:
    amount = f"{Decimal(result['refund_amount_cents']) / 100:.2f} {result['currency']}"
    return f"A refund of {amount} for order {order_id} has been recorded in the demo system. No real money was transferred."


def _cancellation_executed(order_id: str, _result: dict[str, Any]) -> str:
    return f"Order {order_id} has been cancelled."


def _address_executed(order_id: str, _result: dict[str, Any]) -> str:
    return f"The shipping address of order {order_id} has been updated."


def _address_missing(decision: DecisionProposal) -> str | None:
    return None if decision.proposed_address is not None else "the new shipping address"


OPERATIONS: dict[DecisionAction, Operation] = {
    operation.action: operation
    for operation in (
        Operation(
            DecisionAction.PROPOSE_REFUND, ActionType.REFUND, REFUND_POLICY_ID, "refund", "delayed order refund eligibility",
            "refund", "The customer asks for a refund of a delayed order.",
            _propose_refund, _refund_pending, _refund_executed,
        ),
        Operation(
            DecisionAction.PROPOSE_CANCELLATION, ActionType.CANCELLATION, CANCELLATION_POLICY_ID, "cancellation",
            "cancel a processing order", "cancellation", "The customer asks to cancel an order.",
            _propose_cancellation, _cancellation_pending, _cancellation_executed,
        ),
        Operation(
            DecisionAction.PROPOSE_ADDRESS_CHANGE, ActionType.ADDRESS_CHANGE, ADDRESS_POLICY_ID, "address",
            "change the shipping address of an order", "address change",
            "The customer asks to change an order's shipping address.",
            _propose_address_change, _address_pending, _address_executed, missing_input=_address_missing,
            extra_requirements=(ADDRESS_PROVIDED,),
        ),
    )
}
OPERATION_BY_ACTION_TYPE: dict[ActionType, Operation] = {operation.action_type: operation for operation in OPERATIONS.values()}

# A propose_* action without an entry here, or the reverse, must fail at import rather than at decision time.
if set(OPERATIONS) != set(OPERATION_ACTIONS):
    raise RuntimeError("Every propose_* DecisionAction needs exactly one registered Operation.")
