from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from customer_ops.database.models import Customer, Order, Ticket
from customer_ops.domain.schemas import DemoContext, PolicyHit
from customer_ops.retrieval.service import PolicyRetriever, default_policy_retriever


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


def search_policy(
    _context: DemoContext, query: str, category: str | None = None, *, retriever: PolicyRetriever | None = None
) -> list[PolicyHit]:
    hits = (retriever or default_policy_retriever()).search(query, category=category)
    return [
        PolicyHit(
            policy_id=hit.chunk.metadata.document_id,
            chunk_id=hit.chunk.chunk_id,
            category=hit.chunk.metadata.category,
            version=hit.chunk.metadata.version,
            text=hit.chunk.text,
            citation=hit.citation,
            score=hit.score,
        )
        for hit in hits
    ]