from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from customer_ops.config import POLICY_VERSION
from customer_ops.database.models import Customer, Order, Ticket
from customer_ops.domain.schemas import DemoContext, PolicyHit


def get_customer(session: Session, context: DemoContext) -> Customer:
    customer = session.get(Customer, context.require_customer())
    if customer is None:
        raise LookupError("Customer not found.")
    return customer


def get_order(session: Session, context: DemoContext, order_id: str) -> Order | None:
    return session.scalar(select(Order).where(Order.id == order_id, Order.customer_id == context.require_customer()))


def get_customer_orders(session: Session, context: DemoContext) -> list[Order]:
    return list(session.scalars(select(Order).where(Order.customer_id == context.require_customer()).order_by(Order.id)))


def get_ticket(session: Session, context: DemoContext, ticket_id: str) -> Ticket | None:
    return session.scalar(select(Ticket).where(Ticket.id == ticket_id, Ticket.customer_id == context.require_customer()))


def search_policy(_context: DemoContext, query: str, category: str | None = None) -> list[PolicyHit]:
    policy_dir = Path(__file__).resolve().parents[3] / "data" / "policies"
    hits: list[PolicyHit] = []
    for path in policy_dir.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        policy_id, policy_category = path.stem.split("_", maxsplit=1)
        if (category is None or category == policy_category) and query.lower() in text.lower():
            hits.append(PolicyHit(policy_id=policy_id, category=policy_category, version=POLICY_VERSION, text=text))
    return hits