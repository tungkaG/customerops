from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date


POLICY_VERSION = "2026-01"
REFERENCE_DATE = date(2026, 1, 20)


@dataclass(frozen=True)
class ProviderSettings:
    selected_provider: str = os.getenv("CUSTOMER_OPS_LLM_PROVIDER", "gemini")
    gemini_api_key: str | None = os.getenv("CUSTOMER_OPS_GEMINI_API_KEY")
    gemini_model: str = os.getenv("CUSTOMER_OPS_GEMINI_MODEL", "gemini-2.5-flash")
    groq_api_key: str | None = os.getenv("CUSTOMER_OPS_GROQ_API_KEY")
    groq_model: str = os.getenv("CUSTOMER_OPS_GROQ_MODEL", "llama-3.3-70b-versatile")
    # A hub model ID or a local directory; no API key is needed.
    huggingface_model: str = os.getenv("CUSTOMER_OPS_HUGGINGFACE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("CUSTOMER_OPS_DATABASE_URL", "sqlite:///./customer_ops.db")
    reference_date: date = REFERENCE_DATE
    embedding_model: str = os.getenv("CUSTOMER_OPS_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    providers: ProviderSettings = ProviderSettings()


settings = Settings()