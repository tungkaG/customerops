from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from pydantic import Field, ValidationError

from customer_ops.agent.state import AgentRuntime
from customer_ops.config import REFERENCE_DATE
from customer_ops.database.models import Order
from customer_ops.domain.schemas import StrictModel, ToolRequest
from customer_ops.llm.schemas import ToolDefinition
from customer_ops.tools import reads


class InvalidToolRequest(ValueError):
    """The model asked for an unknown tool or supplied arguments that do not match its schema."""


# Argument models never contain customer, ticket, or role fields: those come from the trusted runtime.
class GetOrderArguments(StrictModel):
    order_id: str = Field(min_length=1, max_length=64)


class NoArguments(StrictModel):
    pass


class SearchPolicyArguments(StrictModel):
    query: str = Field(min_length=1, max_length=300)
    category: Literal["refund", "cancellation", "address"] | None = None


def _no_facts(_facts: dict[str, Any], _result: dict[str, Any]) -> None:
    return None


def _no_summary(_result: dict[str, Any]) -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class ReadTool:
    name: str
    description: str
    arguments: type[StrictModel]
    handler: Callable[[AgentRuntime, Any], dict[str, Any]]
    # Optional behavior the generic workflow calls, so a new tool never needs a tool-name branch there.
    merge_facts: Callable[[dict[str, Any], dict[str, Any]], None] = _no_facts  # Adds verified facts from a result.
    summarize: Callable[[dict[str, Any]], dict[str, Any]] = _no_summary  # The compact form recorded in audit events.
    # Set only by tools that return policy evidence (items carry policy_id, chunk_id, citation, text, score).
    evidence: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None


def order_facts(order: Order) -> dict[str, Any]:
    """Describe an order with money in EUR and the delay computed from database values."""
    return {
        "order_id": order.id,
        "status": order.status,
        "amount": f"{Decimal(order.amount_cents) / 100:.2f} {order.currency}",
        "expected_delivery_date": order.expected_delivery_date.isoformat(),
        "delay_days": (REFERENCE_DATE - order.expected_delivery_date).days,
    }


def _get_order(runtime: AgentRuntime, arguments: GetOrderArguments) -> dict[str, Any]:
    order = reads.get_order(runtime.session, runtime.context, arguments.order_id)
    if order is None:
        # One generic answer for missing and foreign orders, so nothing leaks about other customers.
        return {"found": False, "message": "Order not found or inaccessible."}
    return {"found": True, "order": order_facts(order)}


def _get_customer_orders(runtime: AgentRuntime, _arguments: NoArguments) -> dict[str, Any]:
    return {"orders": [order_facts(order) for order in reads.get_customer_orders(runtime.session, runtime.context)]}


def _search_policy(runtime: AgentRuntime, arguments: SearchPolicyArguments) -> dict[str, Any]:
    hits = reads.search_policy(runtime.context, arguments.query, arguments.category, retriever=runtime.retriever)
    return {"hits": [hit.model_dump() for hit in hits]}


def _merge_order(facts: dict[str, Any], result: dict[str, Any]) -> None:
    if result["found"]:
        facts["orders"][result["order"]["order_id"]] = result["order"]


def _merge_orders(facts: dict[str, Any], result: dict[str, Any]) -> None:
    facts["orders"].update({order["order_id"]: order for order in result["orders"]})


def _summarize_order(result: dict[str, Any]) -> dict[str, Any]:
    return {"found": result["found"], **({"order_id": result["order"]["order_id"]} if result["found"] else {})}


def _summarize_orders(result: dict[str, Any]) -> dict[str, Any]:
    return {"order_ids": [order["order_id"] for order in result["orders"]]}


def _summarize_policy_hits(result: dict[str, Any]) -> dict[str, Any]:
    return {"citations": [hit["citation"] for hit in result["hits"]]}


def _policy_evidence(result: dict[str, Any]) -> list[dict[str, Any]]:
    return result["hits"]


READ_TOOLS: dict[str, ReadTool] = {
    tool.name: tool
    for tool in (
        ReadTool(
            "get_order", "Read one of the customer's orders by its exact order ID. Reports the known facts, including the status.",
            GetOrderArguments, _get_order, merge_facts=_merge_order, summarize=_summarize_order,
        ),
        ReadTool(
            "get_customer_orders",
            "List all of the customer's orders. Use it when the customer did not say which order they mean.",
            NoArguments, _get_customer_orders, merge_facts=_merge_orders, summarize=_summarize_orders,
        ),
        ReadTool(
            "search_policy",
            "Search company policy documents. Call it before proposing or rejecting a refund, cancellation, or address "
            "change, because those decisions depend on policy. It is not needed to report an order's status.",
            SearchPolicyArguments, _search_policy, summarize=_summarize_policy_hits, evidence=_policy_evidence,
        ),
    )
}


def tool_definitions() -> list[ToolDefinition]:
    return [
        ToolDefinition(name=tool.name, description=tool.description, parameters=tool.arguments.model_json_schema())
        for tool in READ_TOOLS.values()
    ]


def parse_tool_request(request: ToolRequest) -> tuple[ReadTool, StrictModel]:
    """Check a model-generated request against the allowlist and the tool's argument schema."""
    tool = READ_TOOLS.get(request.tool_name)
    if tool is None:
        raise InvalidToolRequest(f"Unknown tool: {request.tool_name!r}.")
    try:
        # Validate from JSON text so the strict models accept JSON strings such as enum values.
        arguments = tool.arguments.model_validate_json(json.dumps(request.arguments))
    except ValidationError as error:
        details = "; ".join(f"{'.'.join(map(str, item['loc'])) or 'arguments'}: {item['msg']}" for item in error.errors())
        raise InvalidToolRequest(f"Invalid arguments for {tool.name}: {details}") from error
    return tool, arguments


def execute_tool_request(runtime: AgentRuntime, request: ToolRequest) -> dict[str, Any]:
    """Validate a request, then run it with the trusted runtime context."""
    tool, arguments = parse_tool_request(request)
    return tool.handler(runtime, arguments)
