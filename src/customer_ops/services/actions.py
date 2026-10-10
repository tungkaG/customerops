from __future__ import annotations

from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from customer_ops.database.models import Order, PendingAction, Refund
from customer_ops.domain.schemas import ActionStatus, ActionType, DemoContext, OrderStatus, Role
from customer_ops.services.audit import record_event
from customer_ops.services.eligibility import check_eligibility


class ActionService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def reject(self, action_id: str, context: DemoContext) -> PendingAction:
        # Only a trusted operator or manager can reject a proposed business action.
        if context.role not in {Role.OPERATOR, Role.MANAGER}:
            raise PermissionError("An operator context is required.")
        action = self._get_action(action_id)

        # Reject only pending actions; completed actions retain their recorded result.
        if action.status == ActionStatus.PENDING:
            action.status = ActionStatus.REJECTED
            action.approver = context.actor_id

            # Record who rejected the proposal without changing the order or refund records.
            record_event(self.session, actor_type=context.role.value, actor_id=context.actor_id,
                         event_type="action_rejected", payload={"action_id": action.id}, run_id=action.run_id, ticket_id=action.ticket_id)
            self.session.commit()
        return action

    def approve(self, action_id: str, context: DemoContext) -> PendingAction:
        # Only a trusted operator or manager can approve a proposed business action.
        if context.role not in {Role.OPERATOR, Role.MANAGER}:
            raise PermissionError("An operator context is required.")
        action = self._get_action(action_id)

        # Repeated approval is idempotent: an executed action returns its original result.
        if action.status == ActionStatus.EXECUTED:
            return action
        if action.status != ActionStatus.PENDING:
            return action

        # Expire the proposal if its order changed after the proposal captured its version.
        order = self.session.get(Order, action.normalized_arguments["order_id"])
        if order is None or order.customer_id != action.customer_id or order.version != action.order_version:
            action.status = ActionStatus.EXPIRED
            action.failure_reason = "Order state changed before approval."
            self.session.commit()
            return action

        # Re-run deterministic eligibility checks using the current database state.
        eligibility = self._revalidate(action, order)
        if not eligibility.allowed:
            action.status = ActionStatus.EXPIRED
            action.failure_reason = eligibility.reason
            self.session.commit()
            return action

        # Apply role escalation at approval time, such as requiring a manager for large refunds.
        if eligibility.required_role == Role.MANAGER.value and context.role is not Role.MANAGER:
            raise PermissionError("A manager is required for this action.")
        action.approver = context.actor_id

        # Execute exactly one permitted mutation for the approved action type.
        if action.action_type == ActionType.REFUND:
            existing_refund = self.session.scalar(select(Refund).where(Refund.order_id == order.id))
            if existing_refund is not None:
                action.status = ActionStatus.EXPIRED
                action.failure_reason = "The order already has a recorded refund."
                self.session.commit()
                return action
            self.session.add(Refund(id=str(uuid4()), order_id=order.id, executed_action_id=action.id,
                                    amount_cents=order.amount_cents, currency=order.currency, status="recorded"))
            result: dict[str, object] = {"refund_amount_cents": order.amount_cents, "currency": order.currency, "simulated": True}
        elif action.action_type == ActionType.CANCELLATION:
            order.status = OrderStatus.CANCELLED.value
            order.version += 1
            result = {"order_status": order.status}
        else:
            order.shipping_address = action.normalized_arguments["new_address"]
            order.version += 1
            result = {"shipping_address": order.shipping_address}
        action.status = ActionStatus.EXECUTED
        action.execution_result = result

        # Persist the outcome and its audit trail with the resulting business change.
        record_event(self.session, actor_type=context.role.value, actor_id=context.actor_id,
                     event_type="action_executed", payload={"action_id": action.id, "result": result}, run_id=action.run_id, ticket_id=action.ticket_id)
        self.session.commit()
        return action

    def _get_action(self, action_id: str) -> PendingAction:
        action = self.session.get(PendingAction, action_id)
        if action is None:
            raise LookupError("Pending action not found.")
        return action

    def _revalidate(self, action: PendingAction, order: Order):
        # Recompute eligibility from current database facts before executing an approved action.
        return check_eligibility(
            self.session, ActionType(action.action_type), order, action.normalized_arguments.get("new_address")
        )