from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from customer_ops.domain.schemas import CustomerTier, OrderStatus


POLICY_VERSION = "2026-01"
REFUND_POLICY_ID = "POL-REFUND-DELAYED"
CANCELLATION_POLICY_ID = "POL-CANCELLATION"
ADDRESS_POLICY_ID = "POL-ADDRESS-CHANGE"


@dataclass(frozen=True)
class Eligibility:
    allowed: bool
    reason: str
    required_role: str | None = None


def delayed_refund_eligibility(
    *,
    tier: CustomerTier,
    status: OrderStatus,
    expected_delivery_date: date,
    reference_date: date,
    already_refunded: bool,
    amount_cents: int,
    currency: str,
) -> Eligibility:
    tier = CustomerTier(tier)
    status = OrderStatus(status)
    if status is not OrderStatus.DELAYED:
        return Eligibility(False, "Delayed-delivery refunds require a delayed order.")
    if already_refunded:
        return Eligibility(False, "The order already has a recorded refund.")
    if currency != "EUR":
        return Eligibility(False, "Only EUR demo orders are supported.")
    delay_days = (reference_date - expected_delivery_date).days
    threshold = 7 if tier is CustomerTier.GOLD else 14
    if delay_days <= threshold:
        return Eligibility(False, f"Delay must be strictly greater than {threshold} days.")
    required_role = "manager" if amount_cents > 50_000 else "operator"
    return Eligibility(True, "Eligible for a full simulated delayed-delivery refund.", required_role)


def cancellation_eligibility(status: OrderStatus) -> Eligibility:
    status = OrderStatus(status)
    if status is not OrderStatus.PROCESSING:
        return Eligibility(False, "Only processing orders can be cancelled.")
    return Eligibility(True, "Eligible for cancellation with operator approval.", "operator")


def address_change_eligibility(status: OrderStatus, address: dict[str, str]) -> Eligibility:
    status = OrderStatus(status)
    required_keys = {"line1", "city", "postal_code", "country"}
    if status is not OrderStatus.PROCESSING:
        return Eligibility(False, "Only processing orders can have their address changed.")
    if set(address) != required_keys or not all(address.values()):
        return Eligibility(False, "A complete structured address is required.")
    return Eligibility(True, "Eligible for address change with operator approval.", "operator")