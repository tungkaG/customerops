from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TypeVar

from pydantic import BaseModel

from customer_ops.config import settings
from customer_ops.llm.client import LLMProvider, MissingProviderCredentialError
from customer_ops.llm.schemas import ChatMessage, ProviderResponse, ToolDefinition


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class GroqProvider(LLMProvider):
    """Optional Groq adapter for isolated chat and native-tool experiments."""

    name = "groq"

    def __init__(self, *, api_key: str, model: str) -> None:
        if not api_key:
            raise MissingProviderCredentialError(
                "Set CUSTOMER_OPS_GROQ_API_KEY before running a hosted Groq experiment."
            )
        try:
            from groq import Groq
        except ImportError as error:
            raise RuntimeError("Install Groq extras with: py -m pip install '.[groq]'") from error
        self._client = Groq(api_key=api_key)
        self.model = model

    @classmethod
    def from_environment(cls) -> GroqProvider:
        return cls(
            api_key=settings.providers.groq_api_key or "",
            model=settings.providers.groq_model,
        )

    def chat(self, messages: Iterable[ChatMessage]) -> ProviderResponse:
        response = self._client.chat.completions.create(model=self.model, messages=_messages(messages))
        return ProviderResponse(
            content=response.choices[0].message.content or "",
            model=self.model,
            provider=self.name,
            usage=_usage(response),
        )

    def structured_output(self, messages: Iterable[ChatMessage], schema: type[SchemaT]) -> SchemaT:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=_messages(messages),
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            },
        )
        return schema.model_validate_json(response.choices[0].message.content)

    def request_tools(
        self, messages: Iterable[ChatMessage], tools: Iterable[ToolDefinition]
    ) -> ProviderResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=_messages(messages),
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ],
        )
        tool_calls = response.choices[0].message.tool_calls or []
        requests = [
            {"tool_name": call.function.name, "arguments": _string_arguments(call.function.arguments)}
            for call in tool_calls
        ]
        return ProviderResponse(
            content=response.choices[0].message.content or "",
            model=self.model,
            provider=self.name,
            tool_requests=requests,
            usage=_usage(response),
        )


def _messages(messages: Iterable[ChatMessage]) -> list[dict[str, str]]:
    return [message.model_dump() for message in messages]


def _string_arguments(arguments: str) -> dict[str, str]:
    parsed = json.loads(arguments)
    if not isinstance(parsed, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in parsed.items()
    ):
        raise ValueError("Groq function call arguments must be a string-to-string object.")
    return parsed


def _usage(response: object) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }