from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from customer_ops.config import REFERENCE_DATE
from customer_ops.database.models import Order, Refund
from customer_ops.domain.rules import delayed_refund_eligibility
from customer_ops.domain.schemas import CustomerTier, DemoContext, OrderStatus, Role
from customer_ops.services.actions import ActionService
from customer_ops.tools.proposals import propose_cancellation, propose_refund
from customer_ops.tools.reads import get_order


GOLD = DemoContext(actor_id="cust-gold-10-day", role=Role.CUSTOMER, customer_id="cust-gold-10-day")
STANDARD = DemoContext(actor_id="cust-standard-10-day", role=Role.CUSTOMER, customer_id="cust-standard-10-day")
OPERATOR = DemoContext(actor_id="operator-1", role=Role.OPERATOR)
MANAGER = DemoContext(actor_id="manager-1", role=Role.MANAGER)


# Tests the strict delay boundaries: the threshold itself is ineligible, while one day beyond it is eligible.
@pytest.mark.parametrize(("tier", "delay", "allowed"), [
    (CustomerTier.GOLD, 7, False), (CustomerTier.GOLD, 8, True),
    (CustomerTier.STANDARD, 14, False), (CustomerTier.STANDARD, 15, True),
])
def test_refund_thresholds(tier: CustomerTier, delay: int, allowed: bool) -> None:
    result = delayed_refund_eligibility(tier=tier, status=OrderStatus.DELAYED,
        expected_delivery_date=REFERENCE_DATE - timedelta(days=delay), reference_date=REFERENCE_DATE,
        already_refunded=False, amount_cents=19_900, currency="EUR")
    assert result.allowed is allowed


# Tests that an eligible refund becomes a pending proposal for the stored order amount, without creating a refund yet.
def test_gold_ten_day_refund_is_pending_for_exact_order_amount(session: Session) -> None:
    action = propose_refund(session, GOLD, "ticket-gold-refund", "order-gold-10-day", ["POL-REFUND-DELAYED"])
    order = session.get(Order, "order-gold-10-day")
    assert action.status == "pending"
    assert action.normalized_arguments["order_id"] == order.id
    assert session.scalar(select(func.count(Refund.id))) == 0
    assert order.amount_cents == 19_900


# Tests that a Standard customer's 10-day delay fails the 14-day eligibility requirement.
def test_standard_ten_day_refund_is_rejected(session: Session) -> None:
    with pytest.raises(ValueError, match="strictly greater"):
        propose_refund(session, STANDARD, "ticket-standard-refund", "order-standard-10-day", ["POL-REFUND-DELAYED"])


# Tests that another customer's order is neither readable nor usable in a proposal, without revealing why it is unavailable.
def test_cross_customer_order_access_is_generic(session: Session) -> None:
    assert get_order(session, STANDARD, "order-gold-10-day") is None
    with pytest.raises(LookupError, match="inaccessible or not found"):
        propose_refund(session, STANDARD, "ticket-standard-refund", "order-gold-10-day", ["POL-REFUND-DELAYED"])


# Tests that operator approval records one refund for the exact order amount and repeated approval is idempotent.
def test_authorized_approval_executes_once_and_records_exact_amount(session: Session) -> None:
    action = propose_refund(session, GOLD, "ticket-gold-refund", "order-gold-10-day", ["POL-REFUND-DELAYED"])
    approved = ActionService(session).approve(action.id, OPERATOR)
    repeated = ActionService(session).approve(action.id, OPERATOR)
    refund = session.scalar(select(Refund).where(Refund.order_id == "order-gold-10-day"))
    assert approved.status == repeated.status == "executed"
    assert refund.amount_cents == 19_900
    assert session.scalar(select(func.count(Refund.id))) == 1


# Tests that rejecting a pending cancellation leaves the order's business state unchanged.
def test_rejection_leaves_business_records_unchanged(session: Session) -> None:
    action = propose_cancellation(session, GOLD, "ticket-processing", "order-processing", ["POL-CANCELLATION"])
    rejected = ActionService(session).reject(action.id, OPERATOR)
    assert rejected.status == "rejected"
    assert session.get(Order, "order-processing").status == "processing"


# Tests that approval detects an order changed after proposal and expires the action without applying another write.
def test_stale_proposal_expires_without_write(session: Session) -> None:
    action = propose_cancellation(session, GOLD, "ticket-processing", "order-processing", ["POL-CANCELLATION"])
    order = session.get(Order, "order-processing")
    order.status = "shipped"
    order.version += 1
    session.commit()
    expired = ActionService(session).approve(action.id, OPERATOR)
    assert expired.status == "expired"
    assert session.get(Order, "order-processing").status == "shipped"


# Tests that refunds over EUR 500 cannot be approved by an operator but can be executed by a manager.
def test_large_refund_requires_manager(session: Session) -> None:
    action = propose_refund(session, GOLD, "ticket-gold-refund", "order-gold-large", ["POL-REFUND-DELAYED"])
    with pytest.raises(PermissionError, match="manager"):
        ActionService(session).approve(action.id, OPERATOR)
    assert ActionService(session).approve(action.id, MANAGER).status == "executed"


# Tests that the documented Gold 10-day policy example agrees with the deterministic eligibility evaluator.
def test_policy_example_matches_evaluator() -> None:
    text = open("data/policies/POL-REFUND-DELAYED_refund.md", encoding="utf-8").read()
    assert "Gold customer with a 10-day" in text
    assert delayed_refund_eligibility(tier="gold", status="delayed", expected_delivery_date=REFERENCE_DATE - timedelta(days=10),
        reference_date=REFERENCE_DATE, already_refunded=False, amount_cents=19_900, currency="EUR").allowed