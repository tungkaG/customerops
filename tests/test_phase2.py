from __future__ import annotations

import json
from dataclasses import replace

import pytest

from customer_ops.config import settings
from customer_ops.domain.schemas import DecisionAction, DecisionProposal, IntentResult, ToolRequest
from customer_ops.llm.client import MissingProviderCredentialError, configured_provider
from customer_ops.llm.experiments import (
    evaluate_decisions,
    evaluate_entities,
    evaluate_intents,
    evaluate_schema_validity,
    evaluate_tool_sequences,
    write_evaluation_results,
)
from customer_ops.llm.providers.gemini import GeminiProvider
from customer_ops.llm.providers.groq import GroqProvider
from customer_ops.llm.providers.huggingface import _extract_json_text, _parse_tool_calls


# An unknown provider name must fail loudly and list the valid choices, including huggingface.
def test_unknown_provider_name_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    configured_settings = replace(settings, providers=replace(settings.providers, selected_provider="openai"))
    monkeypatch.setattr("customer_ops.config.settings", configured_settings)
    with pytest.raises(ValueError, match="huggingface"):
        configured_provider()


# Small local models wrap JSON in prose or code fences; extraction must still find exactly one object.
# Validation runs from JSON text because the project's strict models reject enum strings given as Python values.
def test_huggingface_structured_output_extracts_json_from_surrounding_text() -> None:
    text = 'Sure! ```json\n{"action": "propose_refund", "justification": "Eligible."}\n``` Hope that helps.'

    assert DecisionProposal.model_validate_json(_extract_json_text(text)) == DecisionProposal(
        action=DecisionAction.PROPOSE_REFUND, justification="Eligible."
    )
    with pytest.raises(ValueError, match="no JSON object"):
        _extract_json_text("I cannot answer that.")


# Qwen's native tool-call blocks become ToolRequests, structured arguments are kept, and non-object arguments are rejected.
def test_huggingface_tool_calls_are_parsed_and_validated() -> None:
    text = 'Looking it up.\n<tool_call>\n{"name": "get_order", "arguments": {"order_id": "order-gold-10-day"}}\n</tool_call>'
    nested = '<tool_call>{"name": "x", "arguments": {"new_address": {"city": "Berlin"}}}</tool_call>'

    assert _parse_tool_calls(text) == [{"tool_name": "get_order", "arguments": {"order_id": "order-gold-10-day"}}]
    assert _parse_tool_calls(nested)[0]["arguments"] == {"new_address": {"city": "Berlin"}}
    assert _parse_tool_calls("No tool needed.") == []
    with pytest.raises(ValueError, match="JSON object"):
        _parse_tool_calls('<tool_call>{"name": "get_order", "arguments": "order-1"}</tool_call>')


# Selecting Gemini must not require a Groq credential; the selected provider alone is validated.
def test_selected_gemini_provider_does_not_require_a_groq_key(monkeypatch: pytest.MonkeyPatch) -> None:
    configured_settings = replace(
        settings,
        providers=replace(settings.providers, selected_provider="gemini", gemini_api_key=""),
    )
    monkeypatch.setattr("customer_ops.config.settings", configured_settings)
    with pytest.raises(MissingProviderCredentialError, match="CUSTOMER_OPS_GEMINI_API_KEY"):
        configured_provider()


# A hosted adapter must fail with its own missing-credential error before importing its SDK.
@pytest.mark.parametrize(
    ("provider", "credential"),
    [
        (GeminiProvider, "CUSTOMER_OPS_GEMINI_API_KEY"),
        (GroqProvider, "CUSTOMER_OPS_GROQ_API_KEY"),
    ],
)
def test_hosted_provider_requires_a_credential_before_loading_sdk(
    provider: type[GeminiProvider] | type[GroqProvider], credential: str
) -> None:
    with pytest.raises(MissingProviderCredentialError, match=credential):
        provider(api_key="", model="test")


# Evaluation helpers must count successes and failures correctly, then preserve those results in JSON.
def test_evaluations_report_denominators_failures_and_json(tmp_path) -> None:
    expected_intent = IntentResult(intent="refund", order_id="order-1")
    wrong_intent = IntentResult(intent="status", order_id="order-1")
    expected_decision = DecisionProposal(
        action=DecisionAction.PROPOSE_REFUND,
        order_id="order-1",
        policy_references=["POL-REFUND-DELAYED"],
        justification="The supplied facts show an eligible delayed order.",
    )
    observed_decision = expected_decision.model_copy()
    tool = ToolRequest(tool_name="get_order", arguments={"order_id": "order-1"})
    # Exercise valid/invalid schemas plus matching and mismatching evaluation cases.
    results = [
        evaluate_schema_validity(IntentResult, [("valid", {"intent": "refund"}), ("invalid", {"intent": 2})]),
        evaluate_intents([("match", expected_intent, expected_intent), ("miss", expected_intent, wrong_intent)]),
        evaluate_entities([("match", expected_intent, expected_intent)]),
        evaluate_tool_sequences([("match", [tool], [[tool], []])]),
        evaluate_decisions([("match", expected_decision, observed_decision)]),
    ]
    path = tmp_path / "results.json"
    write_evaluation_results(path, results)
    serialized = json.loads(path.read_text(encoding="utf-8"))
    # Verify both aggregate counts and the exported failure details.
    assert serialized[0]["total"] == 2
    assert serialized[0]["correct"] == 1
    assert serialized[1]["failures"][0]["case_id"] == "miss"
    assert serialized[3]["rate"] == 1