from __future__ import annotations

import argparse
import random
from datetime import timedelta

from faker import Faker
from sqlalchemy import select
from sqlalchemy.orm import Session

from customer_ops.config import REFERENCE_DATE, settings
from customer_ops.database.models import Customer, Order, Ticket
from customer_ops.database.session import initialize_database, make_engine, make_session_factory


DEFAULT_ADDRESS = {"line1": "1 Demo Street", "city": "Berlin", "postal_code": "10115", "country": "DE"}


def seed_demo(session: Session, *, seed: int = 20260120) -> None:
    if session.scalar(select(Customer.id).limit(1)) is not None:
        raise RuntimeError("Database is not empty. Use the explicit --reset option to replace demo data.")
    random.seed(seed)
    fake = Faker("en_US")
    Faker.seed(seed)
    customers = [
        Customer(id="cust-gold-10-day", name="Gold Ten Day", email="gold10@example.test", tier="gold", default_address=DEFAULT_ADDRESS),
        Customer(id="cust-standard-10-day", name="Standard Ten Day", email="standard10@example.test", tier="standard", default_address=DEFAULT_ADDRESS),
    ]
    for index in range(98):
        customers.append(Customer(id=f"cust-{index:03d}", name=fake.name(), email=fake.unique.email(), tier="gold" if index % 4 == 0 else "standard", default_address={"line1": fake.street_address(), "city": fake.city(), "postal_code": fake.postcode(), "country": "DE"}))
    session.add_all(customers)
    orders = [
        Order(id="order-gold-10-day", customer_id="cust-gold-10-day", amount_cents=19_900, currency="EUR", status="delayed", expected_delivery_date=REFERENCE_DATE - timedelta(days=10), shipping_address=DEFAULT_ADDRESS),
        Order(id="order-standard-10-day", customer_id="cust-standard-10-day", amount_cents=19_900, currency="EUR", status="delayed", expected_delivery_date=REFERENCE_DATE - timedelta(days=10), shipping_address=DEFAULT_ADDRESS),
        Order(id="order-gold-large", customer_id="cust-gold-10-day", amount_cents=50_001, currency="EUR", status="delayed", expected_delivery_date=REFERENCE_DATE - timedelta(days=10), shipping_address=DEFAULT_ADDRESS),
        Order(id="order-processing", customer_id="cust-gold-10-day", amount_cents=8_500, currency="EUR", status="processing", expected_delivery_date=REFERENCE_DATE + timedelta(days=2), shipping_address=DEFAULT_ADDRESS),
    ]
    statuses = ["processing", "shipped", "delayed", "delivered"]
    for index in range(296):
        customer = customers[index % len(customers)]
        orders.append(Order(id=f"order-{index:03d}", customer_id=customer.id, amount_cents=random.randint(1_000, 60_000), currency="EUR", status=statuses[index % len(statuses)], expected_delivery_date=REFERENCE_DATE + timedelta(days=(index % 31) - 15), shipping_address=customer.default_address))
    session.add_all(orders)
    session.add_all([
        Ticket(id="ticket-gold-refund", customer_id="cust-gold-10-day", original_message="My delayed order needs a refund.", agent_run_id="run-gold-refund"),
        Ticket(id="ticket-standard-refund", customer_id="cust-standard-10-day", original_message="My delayed order needs a refund.", agent_run_id="run-standard-refund"),
        Ticket(id="ticket-processing", customer_id="cust-gold-10-day", original_message="Please update my processing order.", agent_run_id="run-processing"),
    ])
    session.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize deterministic CustomerOps demo data.")
    parser.add_argument("--database-url", default=settings.database_url)
    parser.add_argument("--reset", action="store_true", help="Explicitly replace existing database data.")
    args = parser.parse_args()
    engine = make_engine(args.database_url)
    initialize_database(engine, reset=args.reset)
    with make_session_factory(engine)() as session:
        seed_demo(session)
    print(f"Initialized seeded database at {args.database_url}")


if __name__ == "__main__":
    main()