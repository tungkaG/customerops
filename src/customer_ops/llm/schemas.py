from __future__ import annotations

from pydantic import Field, model_validator

from customer_ops.domain.schemas import StrictModel, ToolRequest


class ChatMessage(StrictModel):
    """One turn. Assistant turns may carry the tool requests they made; tool turns carry a tool's result."""

    role: str = Field(pattern="^(system|user|assistant|tool)$")
    content: str
    tool_requests: list[ToolRequest] = Field(default_factory=list)
    tool_name: str | None = None

    @model_validator(mode="after")
    def _check_tool_fields(self) -> ChatMessage:
        if (self.role == "tool") != (self.tool_name is not None):
            raise ValueError("Only tool messages carry a tool_name, and every tool message needs one.")
        if self.tool_requests and self.role != "assistant":
            raise ValueError("Only assistant messages can carry tool requests.")
        if not self.content and not (self.role == "assistant" and self.tool_requests):
            raise ValueError("Message content must not be empty.")
        return self


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