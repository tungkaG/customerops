# CustomerOps AI

CustomerOps is a synthetic ecommerce support operations demo. Phase 1 is fully
deterministic: the system records proposed changes, requires a trusted demo
operator to approve them, and records simulated refunds locally. It does not send
email, transfer money, or use an LLM.

## Setup

Requires Python 3.12 or newer.

```powershell
py -m pip install -e '.[dev]'
py -m customer_ops.database.seed --database-url sqlite:///./customer_ops.db
py -m pytest
```

The initializer refuses to seed a non-empty database. Replacing local demo data
is explicit:

```powershell
py -m customer_ops.database.seed --database-url sqlite:///./customer_ops.db --reset
```

Run the health API with `uvicorn customer_ops.api.main:app --reload`; `GET /health`
returns `{"status":"ok"}`. Demo contexts are simulated identity only and are not
production authentication.

The documented policies live under `data/policies`. Rules in
`customer_ops.domain.rules` decide eligibility; policy text cannot authorize a
write. Set `CUSTOMER_OPS_DATABASE_URL` to change the default database location.