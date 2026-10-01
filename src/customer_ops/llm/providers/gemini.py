from __future__ import annotations

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
            {"tool_name": step.name, "arguments": _string_arguments(step.arguments)}
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
    return "\n\n".join(f"{message.role.upper()}: {message.content}" for message in messages)


def _string_arguments(arguments: object) -> dict[str, str]:
    if not isinstance(arguments, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in arguments.items()
    ):
        raise ValueError("Gemini function call arguments must be a string-to-string object.")
    return arguments