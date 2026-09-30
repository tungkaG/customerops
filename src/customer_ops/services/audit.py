from __future__ import annotations

from uuid import uuid4

from sqlalchemy.orm import Session

from customer_ops.database.models import AuditEvent


def record_event(
    session: Session,
    *,
    actor_type: str,
    actor_id: str,
    event_type: str,
    payload: dict[str, object],
    run_id: str | None = None,
    ticket_id: str | None = None,
) -> None:
    session.add(AuditEvent(
        id=str(uuid4()), run_id=run_id, ticket_id=ticket_id, actor_type=actor_type,
        actor_id=actor_id, event_type=event_type, payload=payload,
    ))