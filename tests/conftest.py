import os
from pathlib import Path

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from customer_ops.retrieval import PolicyIndex  # noqa: E402
from customer_ops.tools import Toolbox  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def policies() -> PolicyIndex:
    return PolicyIndex(DATA / "policies")


@pytest.fixture
def toolbox(policies) -> Toolbox:
    return Toolbox(DATA / "orders.json", policies)
