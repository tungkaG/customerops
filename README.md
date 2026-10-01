# CustomerOps AI

CustomerOps is a synthetic ecommerce support operations demo. Phase 1 is fully
deterministic: the system records proposed changes, requires a trusted demo
operator to approve them, and records simulated refunds locally. Phase 2 adds
isolated LLM experiments only; it does not add agent orchestration or allow LLMs
to execute business mutations.

## Setup

Requires Python 3.12 or newer.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m customer_ops.database.seed --database-url sqlite:///./customer_ops.db
.\.venv\Scripts\python.exe -m pytest
```

The initializer refuses to seed a non-empty database. Replacing local demo data
is explicit:

```powershell
.\.venv\Scripts\python.exe -m customer_ops.database.seed --database-url sqlite:///./customer_ops.db --reset
```

Run the health API with `uvicorn customer_ops.api.main:app --reload`; `GET /health`
returns `{"status":"ok"}`. Demo contexts are simulated identity only and are not
production authentication.

The documented policies live under `data/policies`. Rules in
`customer_ops.domain.rules` decide eligibility; policy text cannot authorize a
write. Set `CUSTOMER_OPS_DATABASE_URL` to change the default database location.

## Phase 2 experiments

Evaluation tests require only the `dev` extra. Gemini and Groq experiments
additionally require their optional dependencies and the API key for the
provider currently selected:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade -e '.[dev,notebooks,gemini]'
$env:CUSTOMER_OPS_LLM_PROVIDER = 'gemini'
$env:CUSTOMER_OPS_GEMINI_API_KEY = '...'
.\.venv\Scripts\jupyter.exe lab experiments
```

Gemini notebook experiments require `google-genai` 2.x or newer because they use
the current Interactions API. After changing dependencies, restart the selected
VS Code notebook kernel before rerunning a cell.

Set `CUSTOMER_OPS_GEMINI_MODEL` or `CUSTOMER_OPS_GROQ_MODEL` to record the
model used for an experiment. Set `CUSTOMER_OPS_LLM_PROVIDER=groq` and provide
`CUSTOMER_OPS_GROQ_API_KEY` only when switching to Groq. The included notebooks
use the selected package adapter and stop with a visible error when its
credential is absent; they do not create illustrative provider results.