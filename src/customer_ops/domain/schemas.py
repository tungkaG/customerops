from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CustomerTier(StrEnum):
    GOLD = "gold"
    STANDARD = "standard"


class Role(StrEnum):
    CUSTOMER = "customer"
    OPERATOR = "operator"
    MANAGER = "manager"


class OrderStatus(StrEnum):
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELAYED = "delayed"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class ActionType(StrEnum):
    REFUND = "refund"
    CANCELLATION = "cancellation"
    ADDRESS_CHANGE = "address_change"


class ActionStatus(StrEnum):
    PENDING = "pending"
    EXECUTED = "executed"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class DecisionAction(StrEnum):
    RESPOND_WITH_STATUS = "respond_with_status"
    PROPOSE_REFUND = "propose_refund"
    PROPOSE_CANCELLATION = "propose_cancellation"
    PROPOSE_ADDRESS_CHANGE = "propose_address_change"
    REJECT_REQUEST = "reject_request"
    ASK_CLARIFICATION = "ask_clarification"
    ESCALATE = "escalate"


class IntentResult(StrictModel):
    intent: str
    order_id: str | None = None
    proposed_address: str | None = None
    missing_information: list[str] = Field(default_factory=list)


class ToolRequest(StrictModel):
    tool_name: str
    arguments: dict[str, str]


class DecisionProposal(StrictModel):
    action: DecisionAction
    order_id: str | None = None
    proposed_address: str | None = None
    policy_references: list[str] = Field(default_factory=list)
    justification: str = Field(min_length=1, max_length=500)
    missing_information: list[str] = Field(default_factory=list)


class AgentResult(StrictModel):
    run_id: str
    outcome: str
    pending_action_id: str | None = None
    cited_policy_references: list[str] = Field(default_factory=list)
    response_draft: str


class DemoContext(StrictModel):
    actor_id: str
    role: Role
    customer_id: str | None = None

    def require_customer(self) -> str:
        if self.role is not Role.CUSTOMER or self.customer_id is None:
            raise PermissionError("A trusted customer context is required.")
        return self.customer_id


class Address(StrictModel):
    line1: str = Field(min_length=1, max_length=120)
    city: str = Field(min_length=1, max_length=80)
    postal_code: str = Field(min_length=1, max_length=20)
    country: str = Field(min_length=2, max_length=2)


class PolicyHit(StrictModel):
    policy_id: str
    category: str
    version: str
    text: str