from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Any


@dataclass
class ModelResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] | None = None
    prompt: str | None = None  # exact text fed to the model, after chat template rendering
    raw_output: str | None = None  # exact decoded model output, before tool call parsing


class ModelOutputError(Exception):
    """The model produced output that could not be parsed; the raw text is kept for the trace."""

    def __init__(self, message: str, raw_output: str = "") -> None:
        super().__init__(message)
        self.raw_output = raw_output


class HuggingFaceModel:
    """Local Hugging Face chat model. Qwen-style `<tool_call>` JSON blocks are parsed into tool calls."""

    def __init__(self, model_name: str = DEFAULT_MODEL, max_new_tokens: int = 512) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._calls_seen = 0
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        use_cuda = torch.cuda.is_available()
        # float32 on CPU: bfloat16 matrix multiplies are very slow on most CPUs.
        self._model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto" if use_cuda else torch.float32)
        if use_cuda:
            self._model.to("cuda")
        self._model.eval()

    def __call__(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        prompt = self._tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        # Greedy decoding keeps runs repeatable.
        output = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens, do_sample=False)
        generated = output[0][inputs["input_ids"].shape[1] :]
        raw = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        input_tokens, output_tokens = int(inputs["input_ids"].shape[1]), int(generated.shape[0])
        usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        calls = parse_tool_calls(raw, first_index=self._calls_seen)
        self._calls_seen += len(calls)
        return ModelResponse(
            text=_TOOL_CALL.sub("", raw).strip(), tool_calls=calls, usage=usage, prompt=prompt, raw_output=raw
        )


def parse_tool_calls(raw: str, first_index: int = 0) -> list[ToolCall]:
    blocks = _TOOL_CALL.findall(raw)
    if len(blocks) != raw.count("<tool_call>"):
        raise ModelOutputError("Unterminated <tool_call> block (output may have been cut off).", raw)
    calls = []
    for index, block in enumerate(blocks):
        try:
            payload = json.loads(block)
            calls.append(ToolCall(id=f"call_{first_index + index}", name=payload["name"], arguments=payload["arguments"]))
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ModelOutputError(f"Could not parse tool call: {error}", raw) from error
    return calls
