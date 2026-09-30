from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from customer_ops.database.seed import seed_demo
from customer_ops.database.session import initialize_database, make_engine, make_session_factory


@pytest.fixture
def session() -> Session:
    engine = make_engine("sqlite://")
    initialize_database(engine)
    factory = make_session_factory(engine)
    with factory() as database_session:
        seed_demo(database_session)
        yield database_session