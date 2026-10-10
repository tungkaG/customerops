"""A scripted stand-in for an LLM, used only by offline tests. It is never imported by the package or notebooks."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from customer_ops.agent.capabilities import IdentifiedRequest
from customer_ops.domain.schemas import DecisionProposal, ToolRequest
from customer_ops.llm.client import LLMProvider
from customer_ops.llm.schemas import ChatMessage, ProviderResponse, ToolDefinition


class ScriptedProvider(LLMProvider):
    """Returns pre-written tool rounds, one identified request, and one decision; records every conversation.

    A test that does not script an identification gets an empty one (no capabilities, so no required evidence).
    """

    name = "scripted-test-double"
    model = "scripted"

    def __init__(
        self,
        *,
        tool_rounds: Iterable[list[ToolRequest]] = (),
        decision: DecisionProposal | Exception | None = None,
        identification: IdentifiedRequest | Exception | None = None,
    ) -> None:
        self._tool_rounds = iter(tool_rounds)
        self._decision = decision
        self._identification = IdentifiedRequest() if identification is None else identification
        self.tool_conversations: list[list[ChatMessage]] = []
        self.decision_conversations: list[list[ChatMessage]] = []
        self.identification_conversations: list[list[ChatMessage]] = []

    def request_tools(self, messages: Iterable[ChatMessage], tools: Iterable[ToolDefinition]) -> ProviderResponse:
        self.tool_conversations.append(list(messages))
        requests = next(self._tool_rounds, [])
        return ProviderResponse(content="", model=self.model, provider=self.name, tool_requests=requests)

    def structured_output(self, messages: Iterable[ChatMessage], schema: type[Any]) -> Any:
        if schema is IdentifiedRequest:
            self.identification_conversations.append(list(messages))
            if isinstance(self._identification, Exception):
                raise self._identification
            return self._identification.model_copy(deep=True)
        self.decision_conversations.append(list(messages))
        if self._decision is None:
            raise AssertionError("The workflow asked for a decision that this test did not script.")
        if isinstance(self._decision, Exception):
            raise self._decision
        return self._decision.model_copy(deep=True)

    def chat(self, messages: Iterable[ChatMessage]) -> ProviderResponse:
        raise AssertionError("The agent workflow must not use plain chat.")
