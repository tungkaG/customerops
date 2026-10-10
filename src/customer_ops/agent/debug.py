"""TEMPORARY debugging aid: print what the model receives and returns when the agent takes a decision.

Delete this file when it is no longer needed; nothing else imports it.

    from customer_ops.agent.debug import trace_decision_io
    provider = trace_decision_io(configured_provider())
"""

from __future__ import annotations

from typing import Any

from customer_ops.llm.client import LLMProvider

_RULE = "=" * 78


def trace_decision_io(provider: LLMProvider) -> LLMProvider:
    """Patch `provider` in place so every structured_output call (the decision) prints its input and output."""
    original = provider.structured_output

    def traced(messages: Any, schema: Any) -> Any:
        messages = list(messages)
        print(f"{_RULE}\nLLM STRUCTURED-OUTPUT INPUT  (provider={provider.name}, model={provider.model})\n{_RULE}")
        for message in messages:
            print(f"--- message role={message.role} ---\n{message.content}\n")
        print(f"--- requested schema: {schema.__name__} ---")

        generate = getattr(provider, "_generate", None)
        if generate is not None and hasattr(provider, "_tokenizer"):
            provider._generate = _traced_generate(provider, generate)
        try:
            result = original(messages, schema)
        except Exception as error:
            print(f"{_RULE}\nLLM STRUCTURED-OUTPUT (failed)\n{_RULE}\n{type(error).__name__}: {error}")
            raise
        finally:
            provider.__dict__.pop("_generate", None)
        print(f"{_RULE}\nLLM STRUCTURED-OUTPUT (parsed {schema.__name__})\n{_RULE}\n{result.model_dump_json(indent=2)}")
        return result

    provider.structured_output = traced
    return provider


def _traced_generate(provider: Any, generate: Any) -> Any:
    # Local models only: show the final chat-template text and the raw reply, before any JSON parsing.
    from customer_ops.llm.providers.huggingface import _template_messages

    def traced_generate(messages: Any, tools: Any = None) -> Any:
        messages = list(messages)
        rendered = provider._tokenizer.apply_chat_template(
            _template_messages(messages), tools=tools, add_generation_prompt=True, tokenize=False
        )
        print(f"{_RULE}\nEXACT TEXT SENT TO THE MODEL (after the chat template and JSON-schema instruction)\n{_RULE}\n{rendered}")
        text, usage = generate(messages, tools)
        print(f"{_RULE}\nRAW MODEL REPLY (before JSON parsing) {usage}\n{_RULE}\n{text}")
        return text, usage

    return traced_generate
