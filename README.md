# Customer ops: autonomous support experiment

Question under test: can an LLM choose its own tools, gather enough evidence, interpret company
policy, and draft a sensible request for a human operator?

The model picks every tool call and its order. The code only implements the tools, runs valid calls,
keeps the conversation history, and bounds the loop. Nothing is sent and nothing is executed: the
`draft_*` tools record text locally and return a receipt.

The previous workflow implementation is preserved in Git (tag `legacy-workflow`).

## Install

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Python 3.12+. Models are downloaded from Hugging Face on first use and run locally (CPU works, a GPU is used if available).

## Configuration

The demo reads a `.env` file in the repo root (git-ignored). Priority: CLI flag, then real environment variable, then `.env`, then default.

| Setting | CLI flag | Environment variable | Default |
| --- | --- | --- | --- |
| Chat model (must support tool calling in its chat template and emit `<tool_call>` JSON, as Qwen does) | `--model` | `CUSTOMEROPS_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` |
| Embedding model | `--embedding-model` | `CUSTOMEROPS_EMBEDDING_MODEL` | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` |
| Demo date | `--date` | `CUSTOMEROPS_DEMO_DATE` | `demo_date` in `data/orders.json` |
| Max model calls per customer message | `--max-turns` | `CUSTOMEROPS_MAX_TURNS` | `8` |

## Data

- Orders: `data/orders.json` (`demo_date` plus a list of orders). `get_order` returns the record and the demo date, so delays are reproducible.
- Policies: `data/policies/*.md` or `*.txt`. Each file is one retrieval document. A policy longer than the embedding model's input limit stops startup with an error naming the file; nothing is truncated.
- Scenarios: `data/scenarios.json` (customer messages plus a human-readable expectation that is never enforced).

## Tracing with Langfuse (optional)

Tracing is off unless credentials are set, so the demo and `pytest` work without them.

1. Create a Langfuse project (Langfuse Cloud or self-hosted) and an API key pair.
2. Copy `.env.example` to `.env` (git-ignored) and fill in `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL` (EU region by default). `demo.py` loads `.env` before the client is created.
3. Check the credentials once (this is the only authentication check; it is not repeated per model call):

   ```powershell
   python demo.py --check-langfuse
   ```

4. Run a scenario. The demo prints a trace link per customer message; open it in Langfuse. `--no-trace` turns tracing off even when credentials are set.

```powershell
python demo.py --scenario two-policies
```

What is recorded, per `Agent.send()` call (one customer message):

- Root `agent` observation `agent.send`: the customer message, scenario ID, max turns, final text, termination reason, number of model calls.
- `generation` observation `model-call` for every model call: messages, tool definitions and the exact rendered prompt; the raw decoded reply, separately from the parsed text and parsed tool calls; model name, `do_sample`, `max_new_tokens`; token usage. A parse failure keeps the raw reply and is marked as an error. No cost is recorded for the local model.
- `tool` observation per requested tool call, named after the tool (including unknown tools and invalid arguments): call ID, arguments, full result. Results containing an `error` are marked as errors. For `search_policy` the result holds the document ID, score and full text.

All turns of one `Agent` share one Langfuse session ID (`<scenario id>-<random>`); independent scenarios get different sessions. Find a trace via the printed link, or in the Langfuse UI under Sessions or Traces. The `--out` JSON file contains scenario summaries and trace IDs, not the full prompts or tool results.

The client is flushed once before the CLI exits.

## Run

```powershell
python demo.py --list
python demo.py --scenario refund-eligible
python demo.py --scenario refund-eligible --follow-up "Thanks. Can you also change the delivery address to Hauptstrasse 1, 8001 Zurich, CH?"
python demo.py --message "Where is ORD-1005?"
```

`notebooks/demo.ipynb` runs the same loop and prints a summary and the trace link for each message; the exact model input and output are in the Langfuse trace.

`--follow-up` continues the same conversation history with another customer message (repeatable).
The demo prints concise tool progress, the final response, the termination reason and the trace link.

## Compare models

Run the same scenarios with different models. Each run is saved to `artifacts/<model name>.json` (e.g. `qwen2.5-3b-instruct.json`) unless you pass `--out`:

```powershell
python demo.py --all --model Qwen/Qwen2.5-1.5B-Instruct --out artifacts/qwen-1.5b.json
python demo.py --all --model Qwen/Qwen2.5-7B-Instruct --out artifacts/qwen-7b.json
```

Then compare the two JSON files. Judge each run against the scenario's `expect` text yourself: did the
model search, notice missing evidence and search again, draft the right operation, and report only what the receipt confirms?

## Two kinds of checks

- `pytest` runs scripted infrastructure tests: history shape, sequential multi-call execution, visible errors,
  draft receipts, single-document retrieval, call budget, and the Langfuse instrumentation (using the real SDK with an
  in-memory exporter, so nothing is sent anywhere). They use scripted model output (and the real embedding
  model from the local Hugging Face cache) and say nothing about model quality.
- `demo.py` runs live behavior evaluation. Failures are reported as failures; the loop never corrects or enforces behavior.
