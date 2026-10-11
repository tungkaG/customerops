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

## Run

```powershell
python demo.py --list
python demo.py --scenario refund-eligible
python demo.py --scenario refund-eligible --follow-up "Thanks. Can you also change the delivery address to Hauptstrasse 1, 8001 Zurich, CH?"
python demo.py --message "Where is ORD-1005?"
```

`notebooks/demo.ipynb` runs the same loop and shows, per model call, the exact rendered prompt, the raw model output, and each tool call and result.

`--follow-up` continues the same conversation history with another customer message (repeatable).
The demo prints each model text, tool call, tool result, token usage, the final response, and the
termination reason.

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
  draft receipts, single-document retrieval, call budget. They use scripted model output (and the real embedding
  model from the local Hugging Face cache) and say nothing about model quality.
- `demo.py` runs live behavior evaluation. Failures are reported as failures; the loop never corrects or enforces behavior.
