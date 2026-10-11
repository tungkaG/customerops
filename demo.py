"""Command line demo: runs scenarios against a live model and prints the trace and final response."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from customer_ops.agent import Agent
from customer_ops.model import DEFAULT_MODEL, HuggingFaceModel
from customer_ops.retrieval import DEFAULT_EMBEDDING_MODEL, PolicyIndex
from customer_ops.tools import Toolbox

DATA = Path(__file__).parent / "data"
load_dotenv(Path(__file__).parent / ".env")  # real environment variables win over .env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--scenario", help="ID of a scenario from data/scenarios.json")
    what.add_argument("--all", action="store_true", help="run every scenario")
    what.add_argument("--message", help="free text customer message")
    what.add_argument("--list", action="store_true", help="list scenario IDs")
    p.add_argument("--follow-up", action="append", default=[], help="extra customer message sent after the first run; repeatable")
    p.add_argument("--model", default=os.environ.get("CUSTOMEROPS_MODEL", DEFAULT_MODEL))
    p.add_argument("--embedding-model", default=os.environ.get("CUSTOMEROPS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    p.add_argument("--date", default=os.environ.get("CUSTOMEROPS_DEMO_DATE"), help="demo date, YYYY-MM-DD (default: orders.json)")
    p.add_argument("--max-turns", type=int, default=int(os.environ.get("CUSTOMEROPS_MAX_TURNS", "8")))
    p.add_argument("--out", type=Path, help="JSON output file (default: artifacts/<model name>.json)")
    return p.parse_args()


def print_trace(trace: list[dict]) -> None:
    for event in trace:
        kind = event["type"]
        if kind == "customer_message":
            print(f"\n[customer] {event['text']}")
        elif kind == "model_response":
            usage = event["usage"]
            tokens = f" (tokens in/out: {usage['input_tokens']}/{usage['output_tokens']})" if usage else ""
            print(f"[model text]{tokens} {event['text'] or '-'}")
        elif kind == "tool_call":
            print(f"[tool call] {event['name']}({json.dumps(event['arguments'], ensure_ascii=False)})")
        elif kind == "tool_result":
            print(f"[tool result] {json.dumps(event['result'], ensure_ascii=False)}")
        elif kind == "model_error":
            print(f"[MODEL ERROR] {event['error']}\n  raw output: {event.get('raw_output', '')!r}")
        elif kind == "final_response":
            print(f"[final response] {event['text']}")
        elif kind == "termination":
            print(f"[termination] {event['reason']}: {event['detail']}")


def run_conversation(model, policies, args, messages: list[str]) -> list[dict]:
    agent = Agent(model, Toolbox(DATA / "orders.json", policies, args.date), max_turns=args.max_turns)
    runs = []
    for message in messages:
        result = agent.send(message)
        print_trace(result.trace)
        runs.append(
            {
                "message": message,
                "termination": result.termination,
                "final_text": result.final_text,
                "model_calls": result.model_calls,
                "trace": result.trace,
            }
        )
        if result.termination != "final_response":
            break
    return runs


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    scenarios = json.loads((DATA / "scenarios.json").read_text(encoding="utf-8"))
    if args.list:
        for s in scenarios:
            print(f"{s['id']}: {s['messages'][0]}")
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

    policies = PolicyIndex(DATA / "policies", args.embedding_model)
    model = HuggingFaceModel(args.model)
    output = {"model": args.model, "embedding_model": args.embedding_model, "scenarios": []}
    for scenario in selected:
        print(f"\n{'=' * 70}\nScenario: {scenario['id']}")
        if scenario.get("expect"):
            print(f"Expected (for human review, not enforced): {scenario['expect']}")
        runs = run_conversation(model, policies, args, scenario["messages"])
        output["scenarios"].append({"id": scenario["id"], "expect": scenario.get("expect"), "runs": runs})

    out = args.out or DATA.parent / "artifacts" / f"{args.model.split('/')[-1].lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
