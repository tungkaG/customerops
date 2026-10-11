import json
from pathlib import Path

import pytest

from customer_ops.agent import Agent
from customer_ops.model import ModelOutputError, ModelResponse, ToolCall, parse_tool_calls
from customer_ops.retrieval import PolicyIndex, PolicyTooLongError

DATA = Path(__file__).resolve().parents[1] / "data"

# These tests use scripted model output to check the plumbing. They say nothing about model behavior.


def scripted(*responses):
    """Return a model callable that replays responses and records a copy of the history it was shown."""
    seen = []
    queue = iter(responses)

    def model(messages, tools):
        seen.append(list(messages))
        return next(queue)

    model.seen = seen
    return model


def call(call_id, name, **arguments):
    return ToolCall(call_id, name, arguments)


def test_history_has_assistant_calls_and_matching_results(toolbox):
    model = scripted(
        ModelResponse("", [call("c1", "get_order", order_id="ORD-1001")]),
        ModelResponse("Your order is delayed."),
    )
    agent = Agent(model, toolbox)
    result = agent.send("Where is ORD-1001?")

    assert [m["role"] for m in agent.history] == ["system", "user", "assistant", "tool", "assistant"]
    assert agent.history[2]["tool_calls"][0]["id"] == agent.history[3]["tool_call_id"] == "c1"
    assert json.loads(agent.history[3]["content"])["order"]["status"] == "delayed"
    assert model.seen[1][-1]["role"] == "tool"  # the second model call saw the result
    assert result.termination == "final_response"


def test_multiple_calls_execute_in_order(toolbox):
    model = scripted(
        ModelResponse("", [call("a", "get_order", order_id="ORD-1001"), call("b", "get_order", order_id="ORD-1002")]),
        ModelResponse("done"),
    )
    agent = Agent(model, toolbox)
    agent.send("compare")

    tool_messages = [m for m in agent.history if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["a", "b"]
    assert [json.loads(m["content"])["order"]["order_id"] for m in tool_messages] == ["ORD-1001", "ORD-1002"]


def test_errors_are_returned_to_the_model(toolbox):
    model = scripted(
        ModelResponse(
            "",
            [
                call("1", "get_order", order_id="ORD-9999"),
                call("2", "no_such_tool", x="y"),
                call("3", "get_order", wrong="ORD-1001"),
                call("4", "draft_refund", text=""),
            ],
        ),
        ModelResponse("sorry"),
    )
    agent = Agent(model, toolbox)
    agent.send("hi")

    results = [json.loads(m["content"]) for m in agent.history if m["role"] == "tool"]
    assert all("error" in r for r in results)
    assert results[0]["found"] is False
    assert toolbox.drafts == []


def test_draft_handlers_only_record_text(toolbox):
    receipt = toolbox.run("draft_cancellation", {"text": "Please cancel ORD-1004."})

    assert receipt["draft_text"] == "Please cancel ORD-1004."
    assert (receipt["status"], receipt["sent"], receipt["order_changed"], receipt["money_moved"]) == (
        "recorded_locally",
        False,
        False,
        False,
    )
    assert toolbox.drafts == [{"kind": "cancellation", "text": "Please cancel ORD-1004."}]


def test_search_returns_one_complete_document(toolbox):
    result = toolbox.run("search_policy", {"query": "who approves a large refund"})

    assert set(result) == {"doc_id", "score", "text"}
    assert result["text"] == (DATA / "policies" / result["doc_id"]).read_text(encoding="utf-8")
    assert toolbox.run("search_policy", {"query": "who approves a large refund"}) == result  # same query repeats


def test_empty_policy_dir_is_an_explicit_error(tmp_path, policies):
    empty = PolicyIndex(tmp_path, policies.model_name)
    assert "error" in empty.search("anything")


def test_oversized_policy_is_reported(tmp_path, policies):
    (tmp_path / "long.md").write_text("refund policy " * 2000, encoding="utf-8")
    with pytest.raises(PolicyTooLongError, match="long.md"):
        PolicyIndex(tmp_path, policies.model_name)


def test_model_call_budget_is_enforced(toolbox):
    model = scripted(*[ModelResponse("", [call(f"c{i}", "get_order", order_id="ORD-1001")]) for i in range(10)])
    agent = Agent(model, toolbox, max_turns=3)
    result = agent.send("loop")

    assert len(model.seen) == 3
    assert (result.termination, result.model_calls, result.final_text) == ("max_turns", 3, None)


def test_unparseable_model_output_is_reported(toolbox):
    def broken(messages, tools):
        raise ModelOutputError("bad json", "<tool_call>{oops")

    result = Agent(broken, toolbox).send("hi")
    assert result.termination == "model_error"
    assert result.final_text is None
    assert any(e["type"] == "model_error" and e["raw_output"] for e in result.trace)


def test_parse_tool_calls_rejects_malformed_blocks():
    with pytest.raises(ModelOutputError):
        parse_tool_calls("<tool_call>\n{not json}\n</tool_call>")
    with pytest.raises(ModelOutputError):
        parse_tool_calls('<tool_call>\n{"name": "get_order", "arguments": {')
