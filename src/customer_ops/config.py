from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date


POLICY_VERSION = "2026-01"
REFERENCE_DATE = date(2026, 1, 20)


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("CUSTOMER_OPS_DATABASE_URL", "sqlite:///./customer_ops.db")
    reference_date: date = REFERENCE_DATE


settings = Settings()