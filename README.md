# CustomerOps AI

CustomerOps is a synthetic ecommerce support operations demo. Phase 1 is fully
deterministic: the system records proposed changes, requires a trusted demo
operator to approve them, and records simulated refunds locally. Phase 2 adds
isolated LLM experiments, Phase 3 a retrieval laboratory, and Phase 4 one
integrated agent that proposes actions but can never execute a business mutation.

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

Evaluation tests require only the `dev` extra. Gemini, Groq, and local Hugging Face
experiments additionally require their optional dependencies. Select the provider
with `CUSTOMER_OPS_LLM_PROVIDER` (`gemini`, `groq`, or `huggingface`):

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade -e '.[dev,notebooks,gemini]'
$env:CUSTOMER_OPS_LLM_PROVIDER = 'gemini'
$env:CUSTOMER_OPS_GEMINI_API_KEY = '...'
.\.venv\Scripts\jupyter.exe lab experiments
```

To use a local model instead, no API key is needed. The default is
`Qwen/Qwen2.5-1.5B-Instruct`; `CUSTOMER_OPS_HUGGINGFACE_MODEL` accepts a hub ID
or a local directory:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade -e '.[dev,notebooks,huggingface]'
$env:CUSTOMER_OPS_LLM_PROVIDER = 'huggingface'
```

The local adapter prompts for JSON and does not constrain decoding, so a small
model can return invalid or wrong values. The notebooks score and display those
failures rather than hide them.

Gemini notebook experiments require `google-genai` 2.x or newer because they use
the current Interactions API. After changing dependencies, restart the selected
VS Code notebook kernel before rerunning a cell.

Set `CUSTOMER_OPS_GEMINI_MODEL` or `CUSTOMER_OPS_GROQ_MODEL` to record the
model used for an experiment. Provide `CUSTOMER_OPS_GROQ_API_KEY` only when
switching to Groq. The included notebooks use the selected package adapter and
stop with a visible error when its credential is absent; they do not create
illustrative provider results.

## Phase 3 retrieval laboratory

Notebooks 05 and 06 use semantic embeddings (`all-MiniLM-L6-v2`, set with
`CUSTOMER_OPS_EMBEDDING_MODEL`) as the main experiment. The lexical hashing
embedder is an optional baseline (`include_hashing_baseline`). Both notebooks
score the same questions, including one hostile query, against three corpora:

- **clean**: only documents that trusted ingestion marked active and trusted;
- **polluted**: every document in `data/policies`, ranked by similarity alone;
- **polluted_filtered**: the polluted index restricted to active, trusted chunks.

They report Recall@1, Recall@3, MRR, coverage, and contamination@1 (the share of
questions whose top result is not active, trusted evidence). Relevance comes
from the query and similarity ranking, not from filenames or folders.

Provenance (`source_type`, `trusted`, `status`) comes from the trusted ingestion
registry `data/policy_registry.json`, never from text inside a document. A file
that is not listed in the registry is ingested as unclassified and untrusted, so
it cannot pass a trust filter. Add a registry entry when you add a document.

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade -e '.[dev,notebooks,retrieval]'
.\.venv\Scripts\jupyter.exe lab experiments
```

The agent-facing `search_policy()` tool is a thin wrapper over the same retriever
the notebooks evaluate. It only returns active, trusted policies in effect on the
reference date, each with a chunk-level citation and version. Generated indexes
and metric reports are saved under `artifacts/retrieval`. Retrieval only supplies
evidence and citations; it cannot authorize writes, override trusted identity, or
change deterministic business rules. Metadata filtering is not a complete defense
against prompt injection: hostile text inside a trusted, active document would
still be returned.

## Phase 4 integrated agent

`customer_ops.agent` processes one ticket end to end:

```text
start -> identify request -> gather (required evidence, then optional tools) -> retrieve -> decide -> validate and act -> finalize
```

1. **Identify the request.** A structured model call returns an `IdentifiedRequest`:
   which capabilities the customer asks for (`status`, `refund`, `cancellation`,
   `address_change`; several are possible), the order IDs they wrote, any address they
   supplied, and details still missing. It decides nothing about eligibility or approval,
   and the result is stored apart from the final decision. The allowed capability IDs and
   their descriptions are generated from the capability registry. An ID is never invented
   or repaired.
2. **Gather.** Phase one collects the evidence the identified capabilities declare, so
   it does not depend on the model choosing a tool. The workflow runs those read tools
   itself through the same validated path and trusted context as model calls, records
   them in the same state, audit events, and conversation turns, and tags each call
   `workflow_required`. Shared requirements are collected once (status plus refund
   fetches the order once), and a status inquiry never searches policy. With no order
   named, the customer's orders are listed, an unambiguous single order is used, and
   several orders end in a clarification request. An order that is missing or
   inaccessible keeps the generic "not found" response. Phase two is the model's optional
   tool loop (`model_selected` calls). The model picks from `get_order`,
   `get_customer_orders`, and `search_policy`; requests are validated against each tool's
   schema, which has no customer, ticket, or role fields. Unknown tools, extra or invalid
   arguments, and an over-budget loop end the run with an explicit outcome.
   Two budgets apply. `max_required_calls` (default 8) bounds phase one, and
   exceeding it ends the run as `step_limit_reached` before any decision. `max_steps`
   (default 4) bounds only the model's optional rounds. Required calls never use up the
   optional rounds.
3. **Retrieve.** Evidence is collected from the recorded `search_policy` results
   (`PolicyRetriever`: active, trusted policies in effect on the reference date) and
   labelled by who searched. Nothing is searched here. If no search was required or
   made, the evidence is empty and the trace and audit trail record `retrieval not
   requested`.
4. **Decide.** The model returns a strict `DecisionProposal` from the identified
   request, the customer's message, the verified facts (money in EUR), and any retrieved
   evidence. Reporting status alone does not fulfil a refund request. `reject_request`
   must name the refused operation in `rejected_action`. One decision covers the whole
   request; a status answer is not combined with an operation's reply.
5. **Validate and act.** Deterministic code first checks the decision against the identified
   request. A proposal or rejection must concern a capability the customer asked for. The
   decision's order must be one resolved for this request (the IDs the customer wrote, or
   their only order), not merely another order they own, and an order the model fetched
   for itself does not count. `respond_with_status` is not accepted as completion of an
   identified refund, cancellation, or address change. Any of these ends as
   `invalid_decision`. Clarification and escalation stay valid alternatives. Then it checks
   ownership and that the cited policies were retrieved and include the applicable one.
   - A proposal then calls the existing `propose_*` function, which re-checks
     eligibility. A valid mutation ends in `pending_approval`; nothing executes.
   - A rejection is accepted only when the shared, read-only eligibility check
     (`services/eligibility.py`) says the request is ineligible. It never calls a
     `propose_*` function, so checking cannot create a pending action. A rejection
     that contradicts the rules, or lacks the order or evidence, ends in
     `invalid_decision`. The refusal draft is built from the rule's reason, not the
     model's explanation. IDs are never repaired.
6. **Continue.** `continue_after_approval` reports the persisted action status after an
   operator approves or rejects it through `ActionService`. It never approves or
   executes.

Trusted context lives in `AgentRuntime`, not in `AgentState`. Response drafts are
deterministic templates built from database records, so they cannot misstate a
status or an amount. Every step is recorded as a structured audit event
(`audit_trail(session, run_id)`).

```python
from customer_ops.agent.state import AgentRuntime
from customer_ops.agent.workflow import run_agent          # plain functions
from customer_ops.agent.graph import run_agent_graph       # the same steps as a LangGraph graph
state = run_agent(AgentRuntime(session=session, context=customer_context, provider=provider), ticket_id)
```

Notebook 07 runs the scenarios against the live provider selected by
`CUSTOMER_OPS_LLM_PROVIDER`. It prints every model call's identified request, tool calls
(marked required or model-selected), evidence, decision, validation outcome, and result,
and scores request identification and decision correctness separately, so a wrong
identified capability is not hidden by the decision. The expected requests in the notebook
are hand-written for scoring only and are never given to the agent.
`tests/test_phase4.py` runs the same scenarios offline through both runners with a
scripted stand-in for the model. That stand-in lives in `tests/` only: it is not in the
package and the notebooks never use it.

Provider notes: Groq receives tool turns as OpenAI-style `tool_calls`/`tool` messages and
the local Hugging Face adapter uses Qwen's native chat template. The Gemini adapter
flattens tool turns into its single text prompt. The Groq and Gemini tool-turn paths have
not been run against the live services.

### Extending the agent

The identify, gather, retrieve, decide, validate, and finalize steps are generic, and both
runners use the same step functions. Extension points are ordinary Python definitions with
explicit registration; there is no plugin loader.

**A new read tool** is one `ReadTool` entry in `agent/tools.py`: an argument model (which
both builds the schema shown to the model and validates its arguments), a handler, and
optionally `merge_facts` (verified facts to add), `summarize` (the audit summary), and
`evidence` (set only if the tool returns policy evidence). `workflow.py` needs no change.

**New required evidence** is a `Requirement` in `agent/requirements.py`: `collect` returns
the read-tool calls to run, and `unmet` says what still blocks the request (a missing
order, or details to ask the customer for). Attach it to a capability. Capabilities for
business operations are generated from the operation registry, so an operation adds its
policy through `policy_category` and `policy_query` and any other requirement through
`extra_requirements`. Only `status` is declared by hand, in `agent/capabilities.py`.

**A new operation** (a request that can become a pending action) needs:

1. `DecisionAction.PROPOSE_...` and `ActionType....` in `domain/schemas.py`. The new
   `propose_*` action automatically becomes an allowed `rejected_action` value, and
   import fails until step 6 registers it.
2. Its rule in `domain/rules.py`, a policy document in `data/policies`, and an entry in
   `data/policy_registry.json`.
3. A branch in `check_eligibility` (`services/eligibility.py`). Proposals, approval
   revalidation, and rejection validation all call it.
4. A proposal function in `tools/proposals.py` that persists a `PendingAction` and cites
   the policy.
5. An execution branch in `ActionService.approve` (`services/actions.py`) that performs
   the mutation and records `execution_result`. The agent never executes, so nothing in
   the agent does this for you.
6. One `Operation` entry in `agent/operations.py`: policy ID, policy category and search
   query, model-facing meaning, the proposal handler, any missing-input check or extra
   requirement, and the pending and executed drafts. The capability the model can
   identify, its required evidence (order facts and this policy), the decision prompt,
   the `rejected_action` values, proposal validation, rejection validation, and
   continuation drafts are all generated from this entry.
7. Tests: the eligibility cases, the required-evidence collection, the proposal and
   approval paths, and a rejection scenario.
Changes the identification step needs for a new operation: none beyond step 6. If the
operation needs a detail the customer must supply, add a `Requirement` for it.

Also required when relevant: a new table or column (`database/models.py`, `seed.py`)
for data the operation changes, and a new field on `DecisionProposal` if the operation
needs structured input beyond an address. `PendingAction.normalized_arguments` is JSON,
so storing the operation's arguments needs no migration.