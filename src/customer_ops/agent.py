from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from customer_ops.model import ModelOutputError, ModelResponse
from customer_ops.tools import Toolbox

SYSTEM_PROMPT = """You are a customer support assistant. Help the customer using the
available tools and the information they return.

Our company has policies that must be followed when handling requests
that involve a business action. Obtain and consider the applicable
policies before preparing an action for review.

Choose the tools and their order yourself. If the information returned
is insufficient, gather more information. If necessary details are
missing or ambiguous, ask the customer.

Use verified order facts and retrieved policy text. Do not invent
identifiers, facts, policies, or tool results.
"""

# In general make sure you obtain the order details accurately before taking any action.
# Then check for policies if applicable before taking any action.
# Then take the appropriate action based on the gathered information and policies.

ModelFn = Callable[[list[dict], list[dict]], ModelResponse]


@dataclass
class RunResult:
    final_text: str | None
    termination: str  # "final_response", "max_turns" or "model_error"
    trace: list[dict]
    model_calls: int


class Agent:
    """One conversation. The model picks every tool call; this class only executes them and bounds the loop."""

    def __init__(
        self, model: ModelFn, toolbox: Toolbox, system_prompt: str = SYSTEM_PROMPT, max_turns: int = 8
    ) -> None:
        self.model = model
        self.toolbox = toolbox
        self.max_turns = max_turns
        self.history: list[dict] = [{"role": "system", "content": system_prompt}]

    def send(self, customer_message: str) -> RunResult:
        """Append a customer message and run until a final response, an error, or the turn budget (per call)."""
        trace: list[dict] = [{"type": "customer_message", "text": customer_message}]
        self.history.append({"role": "user", "content": customer_message})
        tools = self.toolbox.definitions()
        calls_made = 0

        while True:
            if calls_made >= self.max_turns:
                return self._finish(trace, None, "max_turns", calls_made, f"Model call budget of {self.max_turns} reached.")
            calls_made += 1
            try:
                response = self.model(self.history, tools)
            except ModelOutputError as error:
                trace.append({"type": "model_error", "error": str(error), "raw_output": error.raw_output})
                return self._finish(trace, None, "model_error", calls_made, str(error))

            trace.append(
                {
                    "type": "model_response",
                    "text": response.text,
                    "usage": response.usage,
                    "prompt": response.prompt,
                    "raw_output": response.raw_output,
                }
            )
            if not response.tool_calls:
                if not response.text:
                    trace.append({"type": "model_error", "error": "Model returned no text and no tool calls."})
                    return self._finish(trace, None, "model_error", calls_made, "Empty model response.")
                self.history.append({"role": "assistant", "content": response.text})
                return self._finish(trace, response.text, "final_response", calls_made, "Model returned a final response.")

            self.history.append(
                {
                    "role": "assistant",
                    "content": response.text,
                    "tool_calls": [
                        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                        for c in response.tool_calls
                    ],
                }
            )
            for call in response.tool_calls:
                trace.append({"type": "tool_call", "id": call.id, "name": call.name, "arguments": call.arguments})
                result = self.toolbox.run(call.name, call.arguments)
                trace.append({"type": "tool_result", "id": call.id, "name": call.name, "result": result})
                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

    @staticmethod
    def _finish(trace: list[dict], text: str | None, termination: str, calls: int, reason: str) -> RunResult:
        if text is not None:
            trace.append({"type": "final_response", "text": text})
        trace.append({"type": "termination", "reason": termination, "detail": reason})
        return RunResult(final_text=text, termination=termination, trace=trace, model_calls=calls)
