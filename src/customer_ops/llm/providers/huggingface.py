from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any, TypeVar

from pydantic import BaseModel

from customer_ops.config import settings
from customer_ops.llm.client import LLMProvider
from customer_ops.llm.schemas import ChatMessage, ProviderResponse, ToolDefinition


SchemaT = TypeVar("SchemaT", bound=BaseModel)

_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


class HuggingFaceProvider(LLMProvider):
    """Runs a local Hugging Face instruct model (default Qwen2.5-1.5B-Instruct) with transformers."""

    name = "huggingface"

    def __init__(self, *, model: str, max_new_tokens: int = 512) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install local-model extras with: py -m pip install '.[huggingface]'") from error
        self.model = model
        self._max_new_tokens = max_new_tokens
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        use_cuda = torch.cuda.is_available()
        # float32 on CPU: bfloat16 matrix multiplies are very slow on most CPUs.
        self._model = AutoModelForCausalLM.from_pretrained(model, dtype="auto" if use_cuda else torch.float32)
        if use_cuda:
            self._model.to("cuda")
        self._model.eval()

    @classmethod
    def from_environment(cls) -> HuggingFaceProvider:
        return cls(model=settings.providers.huggingface_model)

    def chat(self, messages: Iterable[ChatMessage]) -> ProviderResponse:
        text, usage = self._generate(messages)
        return ProviderResponse(content=text, model=self.model, provider=self.name, usage=usage)

    def structured_output(self, messages: Iterable[ChatMessage], schema: type[SchemaT]) -> SchemaT:
        # The model is only prompted for JSON, not constrained to it, so invalid output fails validation visibly.
        instruction = (
            "Respond with exactly one JSON object and nothing else. It must conform to this JSON schema:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
        text, _ = self._generate(_with_instruction(messages, instruction))
        return schema.model_validate_json(_extract_json_text(text))

    def request_tools(
        self, messages: Iterable[ChatMessage], tools: Iterable[ToolDefinition]
    ) -> ProviderResponse:
        tool_schemas = [
            {
                "type": "function",
                "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters},
            }
            for tool in tools
        ]
        text, usage = self._generate(messages, tools=tool_schemas)
        return ProviderResponse(
            content=_TOOL_CALL.sub("", text).strip(),
            model=self.model,
            provider=self.name,
            tool_requests=_parse_tool_calls(text),
            usage=usage,
        )

    def _generate(
        self, messages: Iterable[ChatMessage], tools: list[dict[str, Any]] | None = None
    ) -> tuple[str, dict[str, int]]:
        prompt = self._tokenizer.apply_chat_template(
            _template_messages(messages), tools=tools, add_generation_prompt=True, tokenize=False
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        # Greedy decoding keeps experiment runs repeatable.
        output = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens, do_sample=False)
        generated = output[0][inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        input_tokens, output_tokens = int(inputs["input_ids"].shape[1]), int(generated.shape[0])
        return text, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }


def _template_messages(messages: Iterable[ChatMessage]) -> list[dict[str, Any]]:
    # Qwen's chat template renders assistant tool_calls and tool-role results natively.
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.tool_requests:
            calls = [
                {"type": "function", "function": {"name": request.tool_name, "arguments": request.arguments}}
                for request in message.tool_requests
            ]
            converted.append({"role": "assistant", "content": message.content, "tool_calls": calls})
        else:
            converted.append({"role": message.role, "content": message.content})
    return converted


def _with_instruction(messages: Iterable[ChatMessage], instruction: str) -> list[ChatMessage]:
    items = list(messages)
    if items and items[0].role == "system":
        return [ChatMessage(role="system", content=f"{items[0].content}\n\n{instruction}"), *items[1:]]
    return [ChatMessage(role="system", content=instruction), *items]


def _extract_json_text(text: str) -> str:
    # Small models often wrap JSON in prose or code fences, so decode from the first brace.
    # Returning text lets strict models validate JSON strings, such as enum values.
    start = text.find("{")
    if start == -1:
        raise ValueError(f"Model output contains no JSON object: {text!r}")
    _, end = json.JSONDecoder().raw_decode(text[start:])
    return text[start : start + end]


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    requests = []
    for block in _TOOL_CALL.findall(text):
        call = json.loads(block)
        arguments = call["arguments"]
        if not isinstance(arguments, dict):
            raise ValueError("Hugging Face tool call arguments must be a JSON object.")
        requests.append({"tool_name": call["name"], "arguments": arguments})
    return requests
