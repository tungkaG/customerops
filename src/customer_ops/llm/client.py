from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TypeVar

from pydantic import BaseModel

from customer_ops.llm.schemas import ChatMessage, ProviderResponse, ToolDefinition


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class MissingProviderCredentialError(RuntimeError):
    """Raised when a configured hosted provider has no usable credential."""


def configured_provider() -> LLMProvider:
    """Create the hosted provider selected by the trusted process environment."""
    from customer_ops.config import settings
    from customer_ops.llm.providers.gemini import GeminiProvider
    from customer_ops.llm.providers.groq import GroqProvider

    if settings.providers.selected_provider == "gemini":
        return GeminiProvider.from_environment()
    if settings.providers.selected_provider == "groq":
        return GroqProvider.from_environment()
    raise ValueError("CUSTOMER_OPS_LLM_PROVIDER must be either 'gemini' or 'groq'.")


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def chat(self, messages: Iterable[ChatMessage]) -> ProviderResponse:
        """Produce a plain chat response."""

    @abstractmethod
    def structured_output(self, messages: Iterable[ChatMessage], schema: type[SchemaT]) -> SchemaT:
        """Produce a response validated against the requested Pydantic schema."""

    @abstractmethod
    def request_tools(
        self, messages: Iterable[ChatMessage], tools: Iterable[ToolDefinition]
    ) -> ProviderResponse:
        """Produce native tool requests when that provider capability is available."""