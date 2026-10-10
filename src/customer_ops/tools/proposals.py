from __future__ import annotations

from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from customer_ops.config import POLICY_VERSION
from customer_ops.database.models import Order, PendingAction, Ticket
from customer_ops.domain.rules import ADDRESS_POLICY_ID, CANCELLATION_POLICY_ID, REFUND_POLICY_ID
from customer_ops.domain.schemas import ActionType, Address, DemoContext
from customer_ops.services.audit import record_event
from customer_ops.services.eligibility import check_eligibility


def propose_refund(session: Session, context: DemoContext, ticket_id: str, order_id: str, policy_refs: list[str]) -> PendingAction:
    return _propose(session, context, ticket_id, order_id, policy_refs, ActionType.REFUND, None)


def propose_cancellation(session: Session, context: DemoContext, ticket_id: str, order_id: str, policy_refs: list[str]) -> PendingAction:
    return _propose(session, context, ticket_id, order_id, policy_refs, ActionType.CANCELLATION, None)


def propose_address_change(session: Session, context: DemoContext, ticket_id: str, order_id: str, new_address: dict[str, str], policy_refs: list[str]) -> PendingAction:
    address = Address.model_validate(new_address).model_dump()
    return _propose(session, context, ticket_id, order_id, policy_refs, ActionType.ADDRESS_CHANGE, address)


def _propose(session: Session, context: DemoContext, ticket_id: str, order_id: str, policy_refs: list[str], action_type: ActionType, new_address: dict[str, str] | None) -> PendingAction:
    # Scope both ticket and order access to the customer established by trusted context.
    customer_id = context.require_customer()
    ticket = session.get(Ticket, ticket_id)
    order = session.scalar(select(Order).where(Order.id == order_id, Order.customer_id == customer_id))
    if ticket is None or ticket.customer_id != customer_id or order is None:
        raise LookupError("Requested record is inaccessible or not found.")

    # A proposal must cite the policy that applies to its requested action.
    expected_ref = {ActionType.REFUND: REFUND_POLICY_ID, ActionType.CANCELLATION: CANCELLATION_POLICY_ID, ActionType.ADDRESS_CHANGE: ADDRESS_POLICY_ID}[action_type]
    if expected_ref not in policy_refs:
        raise ValueError("Applicable policy reference is required.")

    # Evaluate eligibility from database facts; policy text and caller input cannot authorize a write.
    eligibility = check_eligibility(session, action_type, order, new_address)
    if not eligibility.allowed:
        raise ValueError(eligibility.reason)

    # Store normalized arguments so an identical request in the same run can be reused.
    arguments: dict[str, object] = {"order_id": order.id, "policy_refs": sorted(set(policy_refs))}
    if new_address is not None:
        arguments["new_address"] = new_address
    existing = session.scalar(select(PendingAction).where(PendingAction.ticket_id == ticket_id, PendingAction.run_id == (ticket.agent_run_id or "manual"), PendingAction.action_type == action_type.value, PendingAction.status == "pending"))
    if existing is not None and existing.normalized_arguments == arguments:
        return existing

    # Persist a proposal only; business mutations wait for a separate operator approval.
    action = PendingAction(id=str(uuid4()), ticket_id=ticket_id, run_id=ticket.agent_run_id or "manual", customer_id=customer_id,
        action_type=action_type.value, normalized_arguments=arguments, order_version=order.version, policy_version=POLICY_VERSION,
        status="pending", proposer=context.actor_id)
    session.add(action)

    # Record the proposal in the audit trail in the same database transaction.
    record_event(session, actor_type=context.role.value, actor_id=context.actor_id, event_type="action_proposed",
        payload={"action_id": action.id, "action_type": action_type.value}, run_id=action.run_id, ticket_id=ticket_id)
    session.commit()
    return action