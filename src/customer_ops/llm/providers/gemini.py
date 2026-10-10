from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TypeVar

from pydantic import BaseModel

from customer_ops.config import settings
from customer_ops.llm.client import LLMProvider, MissingProviderCredentialError
from customer_ops.llm.schemas import ChatMessage, ProviderResponse, ToolDefinition


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class GeminiProvider(LLMProvider):
    """Optional Google GenAI adapter for isolated Gemini experiments."""

    name = "gemini"

    def __init__(self, *, api_key: str, model: str) -> None:
        if not api_key:
            raise MissingProviderCredentialError(
                "Set CUSTOMER_OPS_GEMINI_API_KEY before running a hosted Gemini experiment."
            )
        try:
            from google import genai
        except ImportError as error:
            raise RuntimeError("Install Gemini extras with: py -m pip install '.[gemini]'") from error
        self._client = genai.Client(api_key=api_key)
        self.model = model

    @classmethod
    def from_environment(cls) -> GeminiProvider:
        return cls(
            api_key=settings.providers.gemini_api_key or "",
            model=settings.providers.gemini_model,
        )

    def chat(self, messages: Iterable[ChatMessage]) -> ProviderResponse:
        response = self._client.interactions.create(model=self.model, input=_prompt(messages))
        return ProviderResponse(content=response.output_text, model=self.model, provider=self.name)

    def structured_output(self, messages: Iterable[ChatMessage], schema: type[SchemaT]) -> SchemaT:
        from google.genai import types

        prompt = _prompt(messages)

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=None,
                response_mime_type="application/json",
                response_json_schema=schema.model_json_schema(),
            ),
        )

        # print("Raw response:", repr(response.text))

        return schema.model_validate_json(response.text)

    def request_tools(
        self, messages: Iterable[ChatMessage], tools: Iterable[ToolDefinition]
    ) -> ProviderResponse:
        response = self._client.interactions.create(
            model=self.model,
            input=_prompt(messages),
            tools=[
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in tools
            ],
        )
        requests = [
            {"tool_name": step.name, "arguments": _object_arguments(step.arguments)}
            for step in response.steps
            if step.type == "function_call"
        ]
        return ProviderResponse(
            content=response.output_text,
            model=self.model,
            provider=self.name,
            tool_requests=requests,
        )


def _prompt(messages: Iterable[ChatMessage]) -> str:
    # Tool calls and results are flattened into text, like every other turn in this adapter's single-prompt format.
    parts = []
    for message in messages:
        if message.tool_requests:
            calls = "; ".join(f"{request.tool_name}({json.dumps(request.arguments)})" for request in message.tool_requests)
            parts.append(f"ASSISTANT TOOL CALLS: {calls}")
        elif message.role == "tool":
            parts.append(f"TOOL RESULT ({message.tool_name}): {message.content}")
        else:
            parts.append(f"{message.role.upper()}: {message.content}")
    return "\n\n".join(parts)


def _object_arguments(arguments: object) -> dict[str, object]:
    if not isinstance(arguments, dict) or not all(isinstance(key, str) for key in arguments):
        raise ValueError("Gemini function call arguments must be a JSON object.")
    return arguments