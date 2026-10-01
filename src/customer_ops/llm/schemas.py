from __future__ import annotations

from pydantic import Field

from customer_ops.domain.schemas import StrictModel, ToolRequest


class ChatMessage(StrictModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1)


class ToolDefinition(StrictModel):
    name: str = Field(pattern="^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    parameters: dict[str, object]


class ProviderResponse(StrictModel):
    content: str
    model: str
    provider: str
    tool_requests: list[ToolRequest] = Field(default_factory=list)
    usage: dict[str, int] | None = None