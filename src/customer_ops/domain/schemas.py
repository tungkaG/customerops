from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


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


# Every propose_* action is an operation a customer can request. Rejections name the operation they refuse.
OPERATION_ACTIONS = tuple(action for action in DecisionAction if action.value.startswith("propose_"))
OperationAction = StrEnum("OperationAction", {action.name: action.value for action in OPERATION_ACTIONS})


class Address(StrictModel):
    line1: str = Field(min_length=1, max_length=120)
    city: str = Field(min_length=1, max_length=80)
    postal_code: str = Field(min_length=1, max_length=20)
    country: str = Field(min_length=2, max_length=2)


class IntentResult(StrictModel):
    intent: str
    order_id: str | None = None
    proposed_address: str | None = None
    missing_information: list[str] = Field(default_factory=list)


class ToolRequest(StrictModel):
    tool_name: str
    # Structured values such as addresses are allowed; each tool validates its own argument schema.
    arguments: dict[str, JsonValue]


class DecisionProposal(StrictModel):
    action: DecisionAction
    order_id: str | None = None
    proposed_address: Address | None = None
    rejected_action: OperationAction | None = Field(
        default=None, description="Required with reject_request: the operation that the customer asked for and is refused."
    )
    policy_references: list[str] = Field(default_factory=list)
    justification: str = Field(min_length=1, max_length=500)
    missing_information: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _rejection_names_its_operation(self) -> DecisionProposal:
        if (self.action is DecisionAction.REJECT_REQUEST) != (self.rejected_action is not None):
            raise ValueError("rejected_action is required with reject_request and not allowed with any other action.")
        return self


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


class PolicyHit(StrictModel):
    policy_id: str
    chunk_id: str
    category: str
    version: str
    text: str
    citation: str
    score: float