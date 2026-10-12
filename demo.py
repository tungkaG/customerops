"""Command line demo: runs scenarios against a live model and prints tool progress, the final response and a trace link."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from customer_ops import tracing
from customer_ops.agent import Agent
from customer_ops.model import DEFAULT_MODEL, HuggingFaceModel
from customer_ops.retrieval import DEFAULT_EMBEDDING_MODEL, PolicyIndex
from customer_ops.tools import Toolbox

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--scenario", help="ID of a scenario from data/scenarios.json")
    what.add_argument("--all", action="store_true", help="run every scenario")
    what.add_argument("--message", help="free text customer message")
    what.add_argument("--list", action="store_true", help="list scenario IDs")
    what.add_argument("--check-langfuse", action="store_true", help="verify the Langfuse credentials and exit")
    p.add_argument("--follow-up", action="append", default=[], help="extra customer message sent after the first run; repeatable")
    p.add_argument("--model", default=os.environ.get("CUSTOMEROPS_MODEL", DEFAULT_MODEL))
    p.add_argument("--embedding-model", default=os.environ.get("CUSTOMEROPS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    p.add_argument("--date", default=os.environ.get("CUSTOMEROPS_DEMO_DATE"), help="demo date, YYYY-MM-DD (default: orders.json)")
    p.add_argument("--max-turns", type=int, default=int(os.environ.get("CUSTOMEROPS_MAX_TURNS", "8")))
    p.add_argument("--no-trace", action="store_true", help="do not send traces to Langfuse even if credentials are set")
    p.add_argument("--out", type=Path, help="JSON summary file (default: artifacts/<model name>.json)")
    return p.parse_args()


def shorten(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def print_tool_progress(messages: list[dict]) -> None:
    """Concise progress read from the conversation history; the full detail lives in the trace."""
    for message in messages:
        for call in message.get("tool_calls", []):
            function = call["function"]
            print(f"[tool call] {function['name']}({shorten(json.dumps(function['arguments'], ensure_ascii=False))})")
        if message["role"] == "tool":
            print(f"[tool result] {message['name']}: {shorten(message['content'])}")


def run_conversation(model, policies, args, scenario: dict) -> dict:
    agent = Agent(
        model, Toolbox(DATA / "orders.json", policies, args.date), max_turns=args.max_turns, scenario_id=scenario["id"]
    )
    runs = []
    for message in scenario["messages"]:
        print(f"\n[customer] {message}")
        start = len(agent.history)
        result = agent.send(message)
        print_tool_progress(agent.history[start:])
        if result.final_text is not None:
            print(f"[final response] {result.final_text}")
        print(f"[termination] {result.termination} (model calls: {result.model_calls})")
        if result.trace_id:
            print(f"[trace] {tracing.trace_url(result.trace_id) or result.trace_id}")
        runs.append(
            {
                "message": message,
                "termination": result.termination,
                "final_text": result.final_text,
                "model_calls": result.model_calls,
                "trace_id": result.trace_id,
            }
        )
        if result.termination != "final_response":
            break
    return {"id": scenario["id"], "expect": scenario.get("expect"), "session_id": agent.session_id, "runs": runs}


def setup_tracing(args: argparse.Namespace) -> None:
    if args.no_trace:
        print("Tracing disabled (--no-trace).")
    elif not tracing.credentials_present():
        print("Tracing disabled: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are not set.")
    else:
        tracing.enable()


def check_langfuse() -> None:
    if not tracing.credentials_present():
        sys.exit("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are not set (see .env.example).")
    client = tracing.enable()
    try:
        authenticated = client.auth_check()
    except Exception as error:  # auth_check raises on rejected credentials and unreachable hosts
        sys.exit(f"Langfuse authentication check failed: {error}")
    if not authenticated:
        sys.exit("Langfuse authentication check failed.")
    print("Langfuse authentication OK.")


def run(args: argparse.Namespace) -> None:
    scenarios = json.loads((DATA / "scenarios.json").read_text(encoding="utf-8"))
    if args.list:
        for s in scenarios:
            print(f"{s['id']}: {s['messages'][0]}")
        return
    if args.check_langfuse:
        check_langfuse()
        return

    if args.all:
        selected = scenarios
    elif args.scenario:
        selected = [s for s in scenarios if s["id"] == args.scenario]
        if not selected:
            sys.exit(f"Unknown scenario '{args.scenario}'. Use --list.")
    else:
        selected = [{"id": "ad-hoc", "messages": [args.message], "expect": None}]
    selected = [{**s, "messages": s["messages"] + args.follow_up} for s in selected]

    setup_tracing(args)
    policies = PolicyIndex(DATA / "policies", args.embedding_model)
    model = HuggingFaceModel(args.model)
    output = {"model": args.model, "embedding_model": args.embedding_model, "scenarios": []}
    for scenario in selected:
        print(f"\n{'=' * 70}\nScenario: {scenario['id']}")
        if scenario.get("expect"):
            print(f"Expected (for human review, not enforced): {scenario['expect']}")
        output["scenarios"].append(run_conversation(model, policies, args, scenario))

    out = args.out or ROOT / "artifacts" / f"{args.model.split('/')[-1].lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")  # real environment variables win over .env; must run before the client is created
    args = parse_args()
    try:
        run(args)
    finally:
        tracing.flush()  # short-lived process: send buffered telemetry before exiting


if __name__ == "__main__":
    main()
