"""What a capability needs before the agent can decide: small registered collectors, not a dependency framework.

A requirement says which read-tool calls to run (`collect`) and, once those ran, what still blocks the request
(`unmet`). The workflow runs them generically and knows nothing about refunds, cancellations, or addresses.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RequiredCall:
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Unmet:
    # blocked: the order is missing or inaccessible, answered generically. missing: details to ask the customer for.
    blocked: bool = False
    missing: tuple[str, ...] = ()


def _no_calls(_state: Mapping[str, Any]) -> list[RequiredCall]:
    return []


def _satisfied(_state: Mapping[str, Any]) -> Unmet | None:
    return None


@dataclass(frozen=True)
class Requirement:
    collect: Callable[[Mapping[str, Any]], list[RequiredCall]] = _no_calls
    unmet: Callable[[Mapping[str, Any]], Unmet | None] = _satisfied


def _collect_orders(state: Mapping[str, Any]) -> list[RequiredCall]:
    order_ids = state["identified_request"].order_ids
    if order_ids:
        return [RequiredCall("get_order", {"order_id": order_id}) for order_id in order_ids]
    # No order was named, so list the customer's orders and resolve only an unambiguous one.
    return [RequiredCall("get_customer_orders", {})]


def _orders_unmet(state: Mapping[str, Any]) -> Unmet | None:
    order_ids = state["identified_request"].order_ids
    orders = state["facts"]["orders"]
    if order_ids:
        return Unmet(blocked=True) if any(order_id not in orders for order_id in order_ids) else None
    if not orders:
        return Unmet(blocked=True)
    return None if len(orders) == 1 else Unmet(missing=("which order you mean",))


def _address_unmet(state: Mapping[str, Any]) -> Unmet | None:
    return None if state["identified_request"].new_address is not None else Unmet(missing=("the new shipping address",))


def resolved_order_ids(state: Mapping[str, Any]) -> list[str]:
    """The orders this request is about: the IDs the customer wrote, else the customer's only order."""
    order_ids = state["identified_request"].order_ids
    if order_ids:
        return list(order_ids)
    orders = state["facts"]["orders"]
    return list(orders) if len(orders) == 1 else []


# One shared instance, so a request that needs order facts for several capabilities collects them once.
ORDER_FACTS = Requirement(collect=_collect_orders, unmet=_orders_unmet)
ADDRESS_PROVIDED = Requirement(unmet=_address_unmet)


def policy_requirement(category: str, query: str) -> Requirement:
    """Policy evidence for one operation: a targeted search restricted to the operation's policy category."""
    return Requirement(collect=lambda _state: [RequiredCall("search_policy", {"query": query, "category": category})])
