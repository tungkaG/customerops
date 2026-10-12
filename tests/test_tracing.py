import json
from pathlib import Path

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import demo
from customer_ops import tracing
from customer_ops.agent import Agent
from customer_ops.model import HuggingFaceModel
from customer_ops.tools import Toolbox

DATA = Path(__file__).resolve().parents[1] / "data"


def tool_call_reply(name: str, arguments: dict) -> str:
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call>"

# The real Langfuse SDK runs here, exporting into memory instead of to the network.
# A client is a per-process singleton for a given public key, so one is shared by all tests.


@pytest.fixture(scope="session")
def langfuse_memory():
    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key="pk-test", secret_key="sk-test", base_url="http://127.0.0.1:9", timeout=1, span_exporter=exporter
    )
    return client, exporter


@pytest.fixture
def spans(langfuse_memory):
    client, exporter = langfuse_memory
    exporter.clear()
    tracing.enable(client)

    def collect() -> list[dict]:
        client.flush()
        finished = sorted(exporter.get_finished_spans(), key=lambda s: s.start_time)
        return [
            {
                "name": s.name,
                "id": s.context.span_id,
                "parent": s.parent.span_id if s.parent else None,
                "type": s.attributes.get("langfuse.observation.type"),
                "level": s.attributes.get("langfuse.observation.level"),
                "session": s.attributes.get("session.id"),
                "attrs": {k: (json.loads(v) if k.endswith(("input", "output", "usage_details", "parameters")) else v)
                          for k, v in s.attributes.items()},
            }
            for s in finished
        ]

    yield collect
    tracing.disable()


class FakeHuggingFace(HuggingFaceModel):
    """The real model class with only the tokenizer/torch primitives replaced, so its tracing code runs."""

    def __init__(self, replies):
        self.model_name = "fake-model"
        self._max_new_tokens = 32
        self._calls_seen = 0
        self._replies = iter(replies)

    def _render(self, messages, tools):
        return "RENDERED " + "|".join(m["role"] for m in messages)

    def _generate(self, prompt):
        return next(self._replies), 11, 7


def test_each_turn_has_a_root_with_ordered_children(toolbox, spans):
    model = FakeHuggingFace([tool_call_reply("get_order", {"order_id": "ORD-1001"}), "It is delayed."])
    agent = Agent(model, toolbox, max_turns=5, scenario_id="demo")
    result = agent.send("Where is ORD-1001?")

    recorded = spans()
    root = next(s for s in recorded if s["name"] == "agent.send")
    children = [s for s in recorded if s is not root]
    assert root["parent"] is None and root["type"] == "agent"
    assert [(s["name"], s["type"]) for s in children] == [
        ("model-call", "generation"),
        ("get_order", "tool"),
        ("model-call", "generation"),
    ]
    assert all(s["parent"] == root["id"] for s in children)
    assert root["attrs"]["langfuse.observation.input"] == {"customer_message": "Where is ORD-1001?"}
    assert root["attrs"]["langfuse.observation.metadata.scenario_id"] == "demo"
    assert root["attrs"]["langfuse.observation.metadata.max_turns"] == 5
    assert root["attrs"]["langfuse.observation.output"] == {
        "final_text": "It is delayed.",
        "termination": "final_response",
        "model_calls": 2,
    }
    assert result.trace_id is not None and result.termination == "final_response"


def test_generation_records_exact_input_output_and_usage(toolbox, spans):
    reply = tool_call_reply("get_order", {"order_id": "ORD-1001"})
    agent = Agent(FakeHuggingFace([reply, "done"]), toolbox)
    agent.send("hi")

    first, second = [s for s in spans() if s["type"] == "generation"]
    attrs = first["attrs"]
    # Recorded input is a snapshot: the history grew afterwards but the first call still shows two messages.
    assert [m["role"] for m in attrs["langfuse.observation.input"]["messages"]] == ["system", "user"]
    assert attrs["langfuse.observation.input"]["rendered_prompt"] == "RENDERED system|user"
    assert len(attrs["langfuse.observation.input"]["tools"]) == 5
    assert attrs["langfuse.observation.output"]["raw_output"] == reply
    assert attrs["langfuse.observation.output"]["parsed_tool_calls"] == [
        {"id": "call_0", "name": "get_order", "arguments": {"order_id": "ORD-1001"}}
    ]
    assert attrs["langfuse.observation.model.name"] == "fake-model"
    assert attrs["langfuse.observation.model.parameters"] == {"do_sample": False, "max_new_tokens": 32}
    assert attrs["langfuse.observation.usage_details"] == {"input_tokens": 11, "output_tokens": 7}
    assert "langfuse.observation.cost_details" not in attrs
    assert [m["role"] for m in second["attrs"]["langfuse.observation.input"]["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]


def test_follow_up_turns_share_a_session(toolbox, spans):
    agent = Agent(FakeHuggingFace(["first", "second"]), toolbox)
    agent.send("one")
    agent.send("two")
    other = Agent(FakeHuggingFace(["third"]), toolbox)
    other.send("three")

    roots = [s for s in spans() if s["name"] == "agent.send"]
    assert len(roots) == 3
    assert roots[0]["session"] == roots[1]["session"] == agent.session_id
    assert roots[2]["session"] == other.session_id != agent.session_id
    assert all(s["session"] for s in spans())  # propagated to children too


def test_raw_reply_survives_a_parsing_error(toolbox, spans):
    agent = Agent(FakeHuggingFace(['<tool_call>\n{"name": "get_order", "arguments": {']), toolbox)
    result = agent.send("hi")

    generation = next(s for s in spans() if s["type"] == "generation")
    root = next(s for s in spans() if s["name"] == "agent.send")
    assert result.termination == "model_error"
    assert generation["level"] == "ERROR"
    assert generation["attrs"]["langfuse.observation.output"]["raw_output"].startswith("<tool_call>")
    assert "parse_error" in generation["attrs"]["langfuse.observation.output"]
    assert root["level"] == "ERROR"


def test_tool_error_results_are_marked_as_errors(toolbox, spans):
    toolbox.run("get_order", {"order_id": "ORD-9999"}, "c1")  # error result without an exception
    toolbox.run("no_such_tool", {"x": "y"}, "c2")
    toolbox.run("draft_refund", {"text": ""}, "c3")
    toolbox.run("get_order", {"order_id": "ORD-1001"}, "c4")

    tools = [s for s in spans() if s["type"] == "tool"]
    assert [s["name"] for s in tools] == ["get_order", "no_such_tool", "draft_refund", "get_order"]
    assert [s["level"] for s in tools] == ["ERROR", "ERROR", "ERROR", None]
    assert tools[0]["attrs"]["langfuse.observation.metadata.call_id"] == "c1"
    assert tools[0]["attrs"]["langfuse.observation.input"]["arguments"] == {"order_id": "ORD-9999"}


def test_search_policy_observation_has_query_and_full_document(toolbox, spans):
    result = toolbox.run("search_policy", {"query": "who approves a large refund"}, "c1")

    (tool,) = [s for s in spans() if s["type"] == "tool"]
    assert tool["attrs"]["langfuse.observation.input"]["arguments"] == {"query": "who approves a large refund"}
    assert tool["attrs"]["langfuse.observation.output"] == result
    assert {"doc_id", "score", "text"} <= set(result)


def test_disabled_tracing_does_not_change_behavior(spans, policies):
    def run_once():
        agent = Agent(
            FakeHuggingFace([tool_call_reply("get_order", {"order_id": "ORD-1001"}), "ok"]),
            Toolbox(DATA / "orders.json", policies),
        )
        return agent, agent.send("hi")

    traced_agent, traced = run_once()
    tracing.disable()
    plain_agent, plain = run_once()

    assert traced_agent.history == plain_agent.history
    assert (plain.final_text, plain.termination, plain.model_calls) == (traced.final_text, "final_response", 2)
    assert plain.trace_id is None


def test_cli_flushes_on_normal_exit_and_on_error(monkeypatch):
    flushed = []
    monkeypatch.setattr(demo.tracing, "flush", lambda: flushed.append(True))
    monkeypatch.setattr(demo, "load_dotenv", lambda *_: None)

    monkeypatch.setattr("sys.argv", ["demo.py", "--list"])
    demo.main()
    monkeypatch.setattr("sys.argv", ["demo.py", "--scenario", "does-not-exist"])
    with pytest.raises(SystemExit):
        demo.main()

    assert flushed == [True, True]
