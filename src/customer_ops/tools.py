from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from customer_ops.retrieval import PolicyIndex

DRAFT_PURPOSE = (
    "The text is recorded as a draft for a human operator to review; it should read like an internal email or "
    "support ticket. This does not execute anything: no order or money is changed and nothing is sent."
)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., dict]


def _single_string(name: str, description: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {name: {"type": "string", "description": description}},
        "required": [name],
        "additionalProperties": False,
    }


class Toolbox:
    """The five tools plus the only place where model supplied arguments are validated."""

    def __init__(self, orders_path: Path, policies: PolicyIndex, demo_date: str | None = None) -> None:
        data = json.loads(Path(orders_path).read_text(encoding="utf-8"))
        self.demo_date = demo_date or data["demo_date"]
        self._orders = {order["order_id"]: order for order in data["orders"]}
        self._policies = policies
        self.drafts: list[dict] = []
        self.tools = {tool.name: tool for tool in self._build_tools()}

    def definitions(self) -> list[dict]:
        return [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in self.tools.values()
        ]

    def run(self, name: str, arguments: Any) -> dict:
        """Execute one model requested call. Invalid calls return an error and are never corrected."""
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool '{name}'. Available tools: {', '.join(self.tools)}."}
        if not isinstance(arguments, dict):
            return {"error": f"Arguments for '{name}' must be a JSON object."}
        (param,) = tool.parameters["required"]
        unexpected = sorted(set(arguments) - {param})
        if unexpected:
            return {"error": f"Unexpected argument(s) for '{name}': {', '.join(unexpected)}. Only '{param}' is accepted."}
        value = arguments.get(param)
        if not isinstance(value, str) or not value.strip():
            return {"error": f"Argument '{param}' for '{name}' is required and must be a non-empty string."}
        return tool.handler(value)

    def _get_order(self, order_id: str) -> dict:
        order = self._orders.get(order_id)
        if order is None:
            return {"found": False, "error": f"No order with ID '{order_id}' exists."}
        return {"found": True, "as_of_date": self.demo_date, "order": order}

    def _draft(self, kind: str, text: str) -> dict:
        self.drafts.append({"kind": kind, "text": text})
        return {
            "status": "recorded_locally",
            "draft_type": kind,
            "draft_text": text,
            "sent": False,
            "order_changed": False,
            "money_moved": False,
            "message": "Draft recorded locally for human review. Nothing was sent. No order or money was changed.",
        }

    def _build_tools(self) -> list[Tool]:
        return [
            Tool(
                "get_order",
                "Look up one customer order by its exact order ID and return the verified order record "
                "(status, customer tier, dates, amount, shipping address) together with today's date. "
                "Returns an explicit not-found result if the ID does not exist. Use only an ID the customer gave.",
                _single_string("order_id", "The exact order ID, for example ORD-1234."),
                self._get_order,
            ),
            Tool(
                "search_policy",
                "Search the company policy documents with a free text query and return the single best matching "
                "policy document in full, with its document ID and a similarity score. The match may be "
                "irrelevant or incomplete; search again with a different query if it does not give what you need.",
                _single_string("query", "Free text describing the policy information you need."),
                self._policies.search,
            ),
            Tool(
                "draft_refund",
                "Prepare a refund proposal for a human operator to review. The draft should explain the customer "
                f"request, the order, the relevant facts, the policy basis, and the requested next step. {DRAFT_PURPOSE}",
                _single_string("text", "The full refund proposal text."),
                lambda text: self._draft("refund", text),
            ),
            Tool(
                "draft_cancellation",
                "Prepare an order cancellation proposal for a human operator to review. The draft should explain "
                f"the customer request, the order, the relevant facts, the policy basis, and the requested next step. {DRAFT_PURPOSE}",
                _single_string("text", "The full cancellation proposal text."),
                lambda text: self._draft("cancellation", text),
            ),
            Tool(
                "draft_address_change",
                "Prepare a shipping address change proposal for a human operator to review. The draft should "
                "explain the customer request, the order, the current and requested address, the relevant facts, "
                f"the policy basis, and the requested next step. {DRAFT_PURPOSE}",
                _single_string("text", "The full address change proposal text."),
                lambda text: self._draft("address_change", text),
            ),
        ]
