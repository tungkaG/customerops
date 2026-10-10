from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from customer_ops.config import REFERENCE_DATE
from customer_ops.database.models import Customer, Order, Refund
from customer_ops.domain.rules import (
    Eligibility,
    address_change_eligibility,
    cancellation_eligibility,
    delayed_refund_eligibility,
)
from customer_ops.domain.schemas import ActionType, CustomerTier, OrderStatus


def check_eligibility(
    session: Session, action_type: ActionType, order: Order, new_address: dict[str, str] | None = None
) -> Eligibility:
    """Evaluate the business rules from current database facts. It reads only and never creates a proposal."""
    if action_type is ActionType.REFUND:
        customer = session.get(Customer, order.customer_id)
        already_refunded = session.scalar(select(Refund).where(Refund.order_id == order.id)) is not None
        return delayed_refund_eligibility(
            tier=CustomerTier(customer.tier), status=OrderStatus(order.status),
            expected_delivery_date=order.expected_delivery_date, reference_date=REFERENCE_DATE,
            already_refunded=already_refunded, amount_cents=order.amount_cents, currency=order.currency,
        )
    if action_type is ActionType.CANCELLATION:
        return cancellation_eligibility(OrderStatus(order.status))
    return address_change_eligibility(OrderStatus(order.status), new_address or {})
