from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from customer_ops import tracing
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
    model_calls: int
    trace_id: str | None = None  # set only when tracing is enabled


class Agent:
    """One conversation. The model picks every tool call; this class only executes them and bounds the loop."""

    def __init__(
        self,
        model: ModelFn,
        toolbox: Toolbox,
        system_prompt: str = SYSTEM_PROMPT,
        max_turns: int = 8,
        scenario_id: str | None = None,
    ) -> None:
        self.model = model
        self.toolbox = toolbox
        self.max_turns = max_turns
        self.scenario_id = scenario_id
        self.session_id = f"{scenario_id or 'chat'}-{uuid.uuid4().hex[:8]}"  # groups all turns of this conversation
        self.history: list[dict] = [{"role": "system", "content": system_prompt}]

    def send(self, customer_message: str) -> RunResult:
        """Append a customer message and run until a final response, an error, or the turn budget (per call)."""
        metadata = {"max_turns": self.max_turns}
        if self.scenario_id:
            metadata["scenario_id"] = self.scenario_id
        with tracing.session(self.session_id), tracing.observation(
            "agent", "agent.send", input={"customer_message": customer_message}, metadata=metadata
        ) as root:
            final_text, termination, detail, calls = self._run(customer_message)
            output = {"final_text": final_text, "termination": termination, "model_calls": calls}
            if termination == "final_response":
                root.update(output=output)
            else:
                level = "ERROR" if termination == "model_error" else "WARNING"
                root.update(output=output, level=level, status_message=detail)
            trace_id = root.trace_id
        return RunResult(final_text, termination, calls, trace_id)

    def _run(self, customer_message: str) -> tuple[str | None, str, str, int]:
        """The loop. Returns (final text, termination reason, detail, model calls)."""
        self.history.append({"role": "user", "content": customer_message})
        tools = self.toolbox.definitions()
        calls_made = 0

        while True:
            if calls_made >= self.max_turns:
                return None, "max_turns", f"Model call budget of {self.max_turns} reached.", calls_made
            calls_made += 1
            try:
                response = self.model(self.history, tools)
            except ModelOutputError as error:
                return None, "model_error", str(error), calls_made

            if not response.tool_calls:
                if not response.text:
                    return None, "model_error", "Model returned no text and no tool calls.", calls_made
                self.history.append({"role": "assistant", "content": response.text})
                return response.text, "final_response", "Model returned a final response.", calls_made

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
                result = self.toolbox.run(call.name, call.arguments, call.id)
                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
