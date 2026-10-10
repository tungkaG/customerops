from __future__ import annotations

from collections.abc import Callable

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from customer_ops.agent.state import AgentRuntime, AgentState
from customer_ops.agent.workflow import (
    act_step, decide_step, finalize_step, gather_step, identify_request_step, retrieve_step, start_step,
)


def _node(step: Callable[[AgentState, AgentRuntime], AgentState]) -> Callable[[AgentState, RunnableConfig], AgentState]:
    # The trusted runtime arrives through run configuration, so it can never be written into graph state.
    def run(state: AgentState, config: RunnableConfig) -> AgentState:
        return step(state, config["configurable"]["runtime"])

    return run


def _proceed_to(next_node: str) -> Callable[[AgentState], str]:
    # Any step that sets an outcome ends the run, so later steps never see a failed run.
    return lambda state: "finalize" if "outcome" in state else next_node


def _after_gather(state: AgentState) -> str:
    if "outcome" in state:
        return "finalize"
    return "retrieve" if state.get("gather_done") else "gather"


def build_agent_graph():
    graph = StateGraph(AgentState)
    for name, step in (("start", start_step), ("identify", identify_request_step), ("gather", gather_step),
                       ("retrieve", retrieve_step), ("decide", decide_step), ("act", act_step),
                       ("finalize", finalize_step)):
        graph.add_node(name, _node(step))
    graph.add_edge(START, "start")
    graph.add_edge("start", "identify")
    graph.add_conditional_edges("identify", _proceed_to("gather"), ["gather", "finalize"])
    graph.add_conditional_edges("gather", _after_gather, ["gather", "retrieve", "finalize"])
    graph.add_conditional_edges("retrieve", _proceed_to("decide"), ["decide", "finalize"])
    graph.add_conditional_edges("decide", _proceed_to("act"), ["act", "finalize"])
    graph.add_edge("act", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_agent_graph(runtime: AgentRuntime, ticket_id: str, run_id: str | None = None) -> AgentState:
    initial: AgentState = {"ticket_id": ticket_id}
    if run_id:
        initial["run_id"] = run_id
    return build_agent_graph().invoke(initial, config={"configurable": {"runtime": runtime}})
