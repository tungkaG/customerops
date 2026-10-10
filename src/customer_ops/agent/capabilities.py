"""What a customer can ask for, and the evidence each ask needs. Business capabilities come from the operation registry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator

from customer_ops.agent.operations import OPERATIONS, Operation
from customer_ops.agent.requirements import ORDER_FACTS, RequiredCall, Requirement, Unmet, policy_requirement
from customer_ops.domain.schemas import Address, StrictModel


@dataclass(frozen=True)
class Capability:
    id: str
    meaning: str  # Model-facing description used to identify the request.
    requirements: tuple[Requirement, ...]  # Evidence the agent collects before it decides.
    operation: Operation | None = None  # Set for capabilities that can become a pending action.


def _operation_capability(operation: Operation) -> Capability:
    requirements = (ORDER_FACTS, policy_requirement(operation.policy_category, operation.policy_query), *operation.extra_requirements)
    return Capability(operation.action_type.value, operation.meaning, requirements, operation)


# Status is the only capability declared here; every business operation is derived from OPERATIONS.
CAPABILITIES: dict[str, Capability] = {
    capability.id: capability
    for capability in (
        Capability("status", "The customer asks where an order is or what its status is.", (ORDER_FACTS,)),
        *(_operation_capability(operation) for operation in OPERATIONS.values()),
    )
}

CapabilityId = StrEnum("CapabilityId", {capability_id.upper(): capability_id for capability_id in CAPABILITIES})
_CAPABILITY_ORDER = {member: position for position, member in enumerate(CapabilityId)}


class IdentifiedRequest(StrictModel):
    """What the customer asked for and explicitly supplied. It carries no decision about eligibility or approval."""

    capabilities: list[CapabilityId] = Field(  # pyright: ignore[reportInvalidTypeForm] (enum generated from the registry)
        default_factory=list, description="Every capability the customer asks for. A message can ask for several."
    )
    order_ids: list[str] = Field(
        default_factory=list, description="Order IDs the customer wrote, copied exactly. Empty when none is written."
    )
    new_address: Address | None = Field(default=None, description="A complete new shipping address, only if the customer wrote one.")
    missing_details: list[str] = Field(
        default_factory=list, description="Details the requested capabilities need that the customer did not give."
    )

    @field_validator("capabilities")
    @classmethod
    def _canonical_capabilities(cls, value: list[CapabilityId]) -> list[CapabilityId]:  # pyright: ignore[reportInvalidTypeForm]
        return sorted(set(value), key=_CAPABILITY_ORDER.__getitem__)

    @field_validator("order_ids")
    @classmethod
    def _unique_order_ids(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


_CAPABILITY_LINES = "\n".join(f"- {capability.id}: {capability.meaning}" for capability in CAPABILITIES.values())

IDENTIFY_SYSTEM_PROMPT = (
    "You identify what a customer is asking for. You do not decide anything: do not judge eligibility, and do not "
    "approve, propose, or reject anything. Use these capability IDs exactly, and list every capability the customer "
    "asks for, because one message can ask for several:\n"
    f"{_CAPABILITY_LINES}\n"
    "Set order_ids to the order IDs the customer wrote, copied exactly as written. Never invent, shorten, "
    "reformat, or guess an ID, and leave order_ids empty when the customer wrote none. Set new_address only when the "
    "customer wrote a complete new address. List in missing_details what a requested capability needs but the "
    "customer did not say."
)


def requests_capability(request: IdentifiedRequest, capability_id: str) -> bool:
    return any(requested.value == capability_id for requested in request.capabilities)


def requests_operation(request: IdentifiedRequest) -> bool:
    """Whether the customer asked for something that can become a pending action."""
    return any(CAPABILITIES[requested.value].operation is not None for requested in request.capabilities)


def requirements_for(request: IdentifiedRequest) -> list[Requirement]:
    """The distinct requirements of the identified capabilities, so shared evidence is collected once."""
    unique: list[Requirement] = []
    for capability_id in request.capabilities:
        for requirement in CAPABILITIES[capability_id.value].requirements:
            if requirement not in unique:
                unique.append(requirement)
    return unique


def required_calls(requirements: list[Requirement], state: Mapping[str, Any]) -> list[RequiredCall]:
    return [call for requirement in requirements for call in requirement.collect(state)]


def unmet_requirements(requirements: list[Requirement], state: Mapping[str, Any]) -> Unmet | None:
    """Combine what still blocks the request. A blocked order outranks details to ask for."""
    unmet = [problem for requirement in requirements if (problem := requirement.unmet(state)) is not None]
    if any(problem.blocked for problem in unmet):
        return Unmet(blocked=True)
    missing = tuple(dict.fromkeys(detail for problem in unmet for detail in problem.missing))
    return Unmet(missing=missing) if missing else None
