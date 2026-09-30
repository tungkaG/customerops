# CustomerOps AI, implementation plan for Copilot

## 1. Instructions for the coding assistant

You are implementing a portfolio project that demonstrates reliable Agentic AI for customer support operations. Read this entire plan before editing code. Treat this document as the project specification.

1. Inspect the repository and its existing instructions first. Preserve unrelated work.
2. Implement only the phase requested by the developer. If no phase is specified, start with Phase 1.
3. Before coding, briefly state the deliverables and acceptance criteria for that phase.
4. Favor the smallest practical implementation. Do not build abstractions for hypothetical future requirements.
5. After implementation, run the checks relevant to the phase and report actual results. Never invent passing tests, evaluation scores, provider capabilities, or notebook outputs.
6. Update the progress section at the bottom of this document. Record completed work, unresolved issues, and the next concrete task.
7. Stop after completing the requested phase. Do not implement later phases without a new instruction.
8. Ask for clarification only when a missing decision materially changes the implementation. Explain routine assumptions and proceed.
9. Use normal Python modules for reusable behavior. Notebooks import those modules and provide experimental inputs, inspection, plots, and evaluation.
10. Do not duplicate application logic inside notebooks.
11. Never embed API keys, provider tokens, or real customer information in code, notebook outputs, fixtures, screenshots, or logs.
12. Unexpected programming errors should fail visibly. Handle expected boundary conditions such as missing orders, invalid input, rejected approvals, provider outages, and unsupported model features explicitly.

## 2. Product objective and scope

Build a fictional ecommerce support system where one agent reads a customer request, gathers order information through explicit tools, retrieves company policies, proposes a decision, and drafts a response. Changes to business records require approval by a human support operator.

The primary portfolio evidence is measurable behavior: correct tool arguments, relevant policy retrieval, valid decisions, controlled business mutations, and traceable execution.

Supported workflows:

1. Order status inquiry.
2. Refund request for delayed delivery.
3. Cancellation of an order that has not shipped.
4. Shipping address change before shipment.
5. Clarification or escalation when information is missing, ambiguous, unsupported, or inconsistent.

Exclude CRM notes. Tickets, pending actions, refunds, and audit events already capture the relevant workflow.

Refunds are simulated database records. No payments are transferred. Customer replies are drafts displayed in the demo. No email is sent.

Initially exclude external CRM integrations, inbox integrations, voice interfaces, multiple agents, Kubernetes, microservices, fine tuning, React, and real payment processing.

## 3. Architecture boundaries

### Deterministic application responsibilities

1. Resolve the customer identity from trusted request context.
2. Enforce record ownership on every customer scoped read and write.
3. Validate tool names and arguments.
4. Load database records and execute transactions.
5. Apply explicit business eligibility rules.
6. Calculate monetary values from database records.
7. Track pending actions and human approvals.
8. Revalidate an approved action against current state before execution.
9. Prevent duplicate execution and repeated refunds.
10. Persist structured audit events.

### Independently experimentable AI responsibilities

1. Intent and entity extraction.
2. Structured output generation.
3. Tool selection and argument generation.
4. Retrieval query formulation.
5. Policy retrieval and optional reranking.
6. Decision proposal using resolved facts and policy evidence.
7. Response drafting.
8. Agent orchestration and model comparison.

The LLM may suggest an action, but application code decides whether that action is permitted. Retrieved text cannot grant permissions or modify the business rule implementation.

A deterministic policy evaluator and an LLM decision experiment intentionally coexist. The experiment measures whether the model understands the evidence. The evaluator protects the actual mutation boundary.

## 4. Technical choices

Use Python 3.12, FastAPI, Pydantic, SQLAlchemy, SQLite, Faker, pytest, Jupyter, and Docker. Add LangGraph when integrating the agent and Streamlit when building the demo.

Use a standard package layout under `src/customer_ops/` and a `pyproject.toml`. Support `pip install .` and `pip install '.[dev]'`. Add optional dependency groups for notebooks, hosted providers, local inference, RAG, and the UI as needed.

Use a local embedding model obtained from Hugging Face and a small persistent local vector index. Prefer Chroma initially. Record the embedding model identifier and version or revision in evaluation metadata. Add a reranker only after measuring the retrieval baseline.

Consider Gemini, Groq, Hugging Face hosted inference, and local Hugging Face model weights. Provider and model identifiers must be configuration values. Verify current SDK interfaces, model availability, structured output support, tool support, and hardware requirements from official documentation during implementation.

Implement one hosted provider first. Add another provider after the initial experiment works. Do not require every provider or local inference to complete the MVP. Do not promise that hosted inference is free or that a local model fits the developer's hardware.

Keep all experiments on synthetic data. Cloud deployment and migration to a hosted PostgreSQL service are optional future work.

## 5. Suggested repository layout

Create directories when their phase needs them, rather than filling the repository with empty placeholders.

```text
customer_ops/
    pyproject.toml
    README.md
    Dockerfile
    .env.example
    .gitignore
    customer_ops_copilot_plan.md
    src/customer_ops/
        config.py
        domain/
            schemas.py
            rules.py
        database/
            models.py
            session.py
            seed.py
        tools/
            registry.py
            reads.py
            proposals.py
        services/
            actions.py
            audit.py
        llm/
            client.py
            schemas.py
            experiments.py
            providers/
        rag/
            ingest.py
            retrieval.py
            evaluation.py
        agent/
            state.py
            nodes.py
            graph.py
        evaluation/
            scenarios.py
            metrics.py
            runner.py
        api/
            main.py
            routes.py
            identity.py
    experiments/
        01_llm_baseline.ipynb
        02_intent_classification.ipynb
        03_tool_selection.ipynb
        04_decision_making.ipynb
        05_rag_retrieval.ipynb
        06_rag_pollution.ipynb
        07_agent_loop.ipynb
        08_model_comparison.ipynb
        09_end_to_end.ipynb
    data/
        policies/
        distractors/
        scenarios/
    tests/
    ui/
        streamlit_app.py
    artifacts/
        evaluations/
```

## 6. Data contracts

### Database entities

1. `customers`: ID, name, unique email, tier, default address, creation timestamp.
2. `orders`: ID, customer ID, amount in integer cents, currency, status, expected delivery date, shipping address, version number, timestamps.
3. `tickets`: ID, customer ID, original message, workflow status, agent run ID, final response draft, timestamps. An order reference extracted from a message is untrusted until validated.
4. `refunds`: ID, order ID, executed action ID, amount in integer cents, currency, status, creation timestamp. This project supports one full refund per order, enforced by a unique constraint on order ID.
5. `pending_actions`: ID, ticket ID, run ID, customer ID, action type, normalized arguments, order version, policy version, status, proposer, approver, timestamps, execution result or failure reason.
6. `audit_events`: ID, run ID, ticket ID, actor type, actor ID, event type, timestamp, and sanitized structured payload.

Do not use floating point values for money. Enable SQLite foreign key enforcement. Use UTC timestamps and inject a fixed reference date for reproducible scenarios.

Treat the audit trace as append only through the application interface. Do not describe a local SQLite database as tamper proof.

### Structured AI outputs

Use strict Pydantic schemas with enumerated values and forbidden extra fields:

1. `IntentResult`: intent, optional order ID, optional proposed address, and missing information.
2. `ToolRequest`: allowlisted tool name and validated arguments.
3. `DecisionProposal`: action, optional order ID, optional proposed address, policy references, concise justification, and missing information.
4. `AgentResult`: run ID, outcome, optional pending action ID, cited policy references, and response draft.

Allowed decisions include `respond_with_status`, `propose_refund`, `propose_cancellation`, `propose_address_change`, `reject_request`, `ask_clarification`, and `escalate`.

Do not trust model supplied monetary amounts, ownership claims, approval claims, or unsupported policy references. Do not use model self reported confidence as authorization or as a calibrated probability.

## 7. Fictional business policies

These rules belong to the fictional demo company. They are not statements about real consumer law.

1. A full delayed delivery refund is eligible only for an order with status `delayed` and a delay strictly greater than 7 days for Gold customers or strictly greater than 14 days for Standard customers.
2. Delay is the fixed reference date minus the expected delivery date. Test the exact threshold as well as the day before and the day after.
3. Delivered, cancelled, or already refunded orders cannot receive this delayed delivery refund.
4. The refund amount equals the order's recorded amount. The LLM cannot choose the amount.
5. All eligible refunds require operator approval. Refunds above EUR 500 require a manager. EUR 500 itself does not require a manager.
6. Cancellation is allowed only for orders with status `processing` and requires operator approval. It changes the status to `cancelled`. Payment handling for cancellations is outside this demo.
7. Shipping address changes are allowed only for orders with status `processing` and require operator approval. The address must be provided and structurally validated. No external address verification is implied.
8. Shipped orders cannot be cancelled or have their address changed through these tools.
9. Unsupported requests, including damaged goods refunds, use clarification or escalation. Do not invent a policy.
10. Orders belonging to another customer are inaccessible. Return a generic inaccessible or not found result without exposing the other customer's details.

Keep these rules in `domain/rules.py`, with equivalent versioned policy documents for retrieval. Add a test for consistency between the documented examples and evaluator outputs.

## 8. Identity and action approval

For the local demo, establish customer and operator identities through an explicit demo context outside the customer message. A customer email inside a request is not authentication. The LLM must not set or replace the trusted customer ID or operator role.

The API and tools receive trusted context separately from model generated arguments. Customer reads are scoped to that context. Approval endpoints require an operator context, with a separate manager role for large refunds.

A clearly labeled demo identity selector is sufficient locally. Document that this is simulated identity, not production authentication. Do not expose the demo publicly without replacing that boundary.

Action lifecycle:

```text
pending → executed
pending → rejected
pending → expired
pending → failed
```

The approval handler records the approver, revalidates the current order and active policy, and executes the mutation in one transaction. If execution fails, roll back business changes and record a safe failure outcome separately. An approval event alone must never permit a later unchecked write.

Bind approval to the exact persisted action and normalized arguments. The approval request cannot replace the order, amount, customer, or address. Changes require a new proposal.

Expire a proposal if the order version or relevant policy version changed. Repeated approval of an executed action returns its existing result. Enforce single execution through conditional status transitions and database constraints, including concurrent approval attempts.

The customer requesting cancellation and the operator authorizing execution are separate events. No cancellation happens merely because the customer asked for it.

## 9. Explicit tool interfaces

Provide these read interfaces:

```python
get_customer(context)
get_order(context, order_id)
get_customer_orders(context)
get_ticket(context, ticket_id)
search_policy(context, query, category=None)
```

Provide these proposal interfaces:

```python
propose_refund(context, ticket_id, order_id, policy_refs)
propose_cancellation(context, ticket_id, order_id, policy_refs)
propose_address_change(context, ticket_id, order_id, new_address, policy_refs)
```

Expose proposals, not mutation execution, to the agent. The action service executes approved mutations behind the operator endpoint. The agent cannot approve actions or invoke arbitrary SQL, arbitrary Python, arbitrary URLs, or unrestricted database access.

Proposal creation validates deterministic eligibility and persists a pending action. It does not modify the order or create a refund. Repeated identical proposals within one run should reuse the existing pending action.

Internal workflow records such as audit events and pending proposals may be persisted automatically. Human approval protects business mutations such as refunds, cancellations, and address changes. Ticket workflow state can update automatically to record analysis progress; final resolution follows the action outcome.

## 10. Synthetic scenario generation

Faker generates names, emails, addresses, and identifiers. Our scenario factory creates meaningful order states, requests, and expected outcomes.

1. Seed Faker and the random generator. Use a fixed reference date and record generator configuration.
2. Generate approximately 100 customers and 300 orders for the demo, including consistent historical refunds.
3. Create explicitly named scenario families and deliberate boundary cases rather than relying solely on random selection.
4. Label each scenario with intent, accessible order, relevant policy IDs, expected action, required approval role, and expected business state after approval or rejection.
5. For a 10 day delay, label a Gold customer as refund eligible and a Standard customer as ineligible. Do not label every delayed order as refundable.
6. Keep ground truth out of agent prompts, tool results, and the policy index.
7. Separate development examples from a held out evaluation set. Vary wording independently of state, and avoid identical templates across both sets.
8. Check critical labels with hand authored reference cases so a shared bug in the generator and evaluator cannot make evaluations falsely pass.

Required cases include status requests, each supported mutation, threshold boundaries, missing order IDs, multiple possible orders, inaccessible orders, already refunded orders, large refunds, missing addresses, unsupported requests, changed order state before approval, repeated approval, and prompt injection attempts.

## 11. Implementation phases

### Phase 1, deterministic foundation

Goal: establish the package, database, synthetic scenarios, business rules, tools, and minimal action service without an LLM.

Deliverables:

1. Package configuration, environment template, README startup instructions, and `GET /health`.
2. Database models and an explicit local initialization command. Resetting demo data must be an explicit operation.
3. Reproducible seed generation and policy documents.
4. Trusted demo identity context and strict domain schemas.
5. Scoped read tools, deterministic eligibility, proposal persistence, and approval service.
6. Tests for ownership, boundary rules, money, duplicate refunds, rejection, stale proposals, repeated execution, and approval roles.

Acceptance criteria:

1. A fresh database can be initialized and seeded reproducibly.
2. The Gold customer with a 10 day delayed order produces a pending refund for the exact recorded amount.
3. The equivalent Standard customer is rejected.
4. Creating a pending action leaves business records unchanged.
5. An authorized approval executes once. Rejection does not mutate business records.
6. Cross customer access is blocked independently of any LLM behavior.
7. All deterministic tests run without provider credentials or internet access.

### Phase 2, isolated LLM experiments

Goal: measure model behavior before introducing agent orchestration.

Deliverables:

1. A small provider interface for chat, structured output, and tool requests. Represent unsupported capabilities explicitly instead of silently changing experiment semantics.
2. One hosted adapter, a deterministic fake adapter for tests, and configuration for additional providers.
3. Notebook 01 for simple response and capability inspection.
4. Notebook 02 for intent and entity extraction.
5. Notebook 03 for tool choice and arguments using stubbed tool responses.
6. Notebook 04 for decisions with supplied facts and policy text, without retrieval.
7. Shared evaluation functions and machine readable result exports.

Acceptance criteria:

1. Each notebook imports the actual package and can execute from a clean kernel when its optional dependencies and credentials exist.
2. Missing credentials are explained visibly. Results are never fabricated or replaced with fake success.
3. Report schema validity, intent accuracy, entity accuracy, tool argument correctness, and decision correctness with denominators and failure examples.
4. Accept multiple valid tool sequences. Searching policy before an order lookup is not automatically incorrect when it can still satisfy the task.
5. Compare native tool calling separately from prompted JSON if provider capabilities differ.
6. Local inference remains optional until hardware and model suitability are established.

### Phase 3, RAG laboratory

Goal: implement genuine embedding retrieval and evaluate noisy and adversarial corpora independently.

Deliverables:

1. Versioned policy ingestion, sensible section based chunks, embeddings, and a persistent local index.
2. Chunk metadata: document ID, chunk ID, policy category, version, status, source type, effective date, and trusted source classification.
3. A naive similarity baseline and a retrieval variant constrained to trusted, active, applicable policies.
4. Notebook 05 for clean retrieval and Notebook 06 for corpus pollution.
5. Distractor corpora containing similar marketing text, obsolete policies, training examples, and malicious instructions embedded in documents.
6. Optional reranking after collecting baseline results.

Acceptance criteria:

1. Report document or chunk relevance explicitly, Recall at 1 and 3, MRR, and retrieval coverage.
2. Evaluate identical queries against clean and polluted corpora and compare baseline, metadata filtering, and optional reranking.
3. Verify citations identify retrieved chunks and applicable policy versions.
4. Distinguish retrieval failures from generation failures by testing the LLM with both retrieved context and known correct context.
5. Abstain or escalate when no valid policy evidence is available.
6. Do not treat metadata filtering as a complete prompt injection defense. Metadata comes from trusted ingestion, not from claims inside document text. Also test malicious text in a document that passes the metadata filter.
7. Retrieved instructions cannot authorize writes, override identity, or alter deterministic business rules.

### Phase 4, integrated agent

Goal: connect proven components into one bounded stateful agent.

Deliverables:

1. Notebook 07 showing the pipeline with observable intermediate values.
2. LangGraph state and nodes using the same modules demonstrated in the notebooks.
3. A tool loop with configurable step limits, provider timeout, bounded retries, and clear terminal outcomes.
4. Structured audit records for tool requests, validated arguments, tool outcomes, retrieved policy references, decisions, and action proposals.
5. A pause at pending approval and continuation based on the stored action outcome.

Acceptance criteria:

1. The four supported workflows complete through the same public entry point.
2. Read only requests produce a response draft without requiring operator approval.
3. Mutations create pending actions and stop before execution.
4. Unknown tools, invalid arguments, unavailable policies, and step exhaustion produce explicit outcomes without business mutations.
5. Store messages and state needed for continuation. A restart must not execute a pending action automatically.
6. The trace shows facts and concise decision justifications. Do not request or display hidden chain of thought.

### Phase 5, API and durable approval workflow

Goal: make the workflow operable and resumable through explicit API operations.

Deliverables:

```text
GET /health
POST /tickets
GET /tickets/{ticket_id}
POST /tickets/{ticket_id}/run
GET /tickets/{ticket_id}/trace
GET /pending_actions
POST /pending_actions/{action_id}/approve
POST /pending_actions/{action_id}/reject
```

Use the action service from Phase 1. Do not create a second approval implementation in the API.

Acceptance criteria:

1. API access is scoped to the trusted demo role and customer context.
2. Operators can inspect the exact action, supporting policy, amount or address, and reason before approving.
3. Approval survives process restart because the proposal and its outcome are persisted.
4. Changed order or policy state invalidates the pending proposal before execution.
5. Rejection leaves orders and refunds unchanged and yields an appropriate draft.
6. A completed simulated refund is described as recorded in the demo. The response must not claim actual money was transferred.
7. Failed execution cannot yield a success response. Response drafting failure after a committed mutation cannot undo or repeat that mutation.
8. Concurrent and repeated approval requests cannot execute an action twice.

### Phase 6, small Streamlit demo

Goal: make the system understandable to a prospective client.

Deliverables:

1. A ticket inbox with seeded scenarios and custom synthetic requests.
2. A customer and order panel.
3. An activity trace with expandable tool inputs and sanitized outputs.
4. Retrieved policies with document names, versions, and citations.
5. A proposal panel with exact action details, Approve and Reject controls, and visible role requirements.
6. Before and after business state plus the final response draft.

Acceptance criteria:

1. The UI calls the backend API and contains no business rule or approval execution logic.
2. Refreshing or rerunning Streamlit does not execute a mutation.
3. Buttons operate on persisted action IDs and display the current backend outcome.
4. Show a successful refund, blocked shipped cancellation, address change, and inaccessible order.
5. The interface clearly labels synthetic data, simulated identities, and simulated refunds.

### Phase 7, evaluations and portfolio delivery

Goal: provide reproducible evidence and an accessible demonstration.

Deliverables:

1. Notebook 08 for provider comparison and Notebook 09 for integrated evaluation.
2. At least 50 held out scenarios spanning normal cases, boundaries, failures, and adversarial inputs.
3. A reusable evaluation runner that resets scenario state and exports JSON or CSV results.
4. A Docker setup for the baseline API and documented optional UI startup.
5. A README with architecture, setup, workflow examples, actual results, limitations, and a short demo recording or screenshots.

Acceptance criteria:

1. Report intent accuracy, tool argument correctness, policy retrieval metrics, correct proposal rate, business state correctness after approval, unauthorized mutation count, duplicate execution count, response factuality, and policy citation validity.
2. Track latency, provider errors, token usage when available, and experiment configuration. Calculate costs only using verified prices; otherwise mark them unavailable.
3. Compare providers on the same data, prompts, tool contracts, and retrieval configuration. Record capability differences and repeat stochastic evaluations when feasible.
4. Keep live provider evaluations separate from ordinary offline tests. Do not make API keys mandatory for the test suite.
5. Required mutation invariants are zero unauthorized and duplicate executions in the tested scenarios. These results are evidence for that suite, not proof of universal security.
6. Show failures and explain their source. Never substitute illustrative percentages for measured outcomes.
7. A fresh checkout can run the deterministic demo following the README. Live LLM use requires the documented optional credentials and dependencies.

## 12. Essential security and failure scenarios

1. Customer asks to refund another customer's order.
2. Customer claims to be an administrator or claims approval already happened.
3. Customer requests a refund larger than the order amount.
4. Customer or retrieved document asks the model to ignore policy.
5. The model proposes an unknown tool or malformed arguments.
6. Retrieval returns an obsolete or irrelevant policy.
7. A relevant trusted document includes hostile embedded instructions.
8. An order ships after an address change was proposed.
9. Two operators approve the same action concurrently.
10. A second request tries to refund an already refunded order.
11. The provider times out before a proposal exists.
12. The provider fails while drafting a response after a mutation succeeded.
13. The agent reaches its step limit or cannot resolve an ambiguous order reference.
14. The database transaction fails during approval.

For each, assert the business state and recorded outcome. Text saying that an action was blocked is insufficient if the database changed.

## 13. Suggested order and effort

First milestone: one Gold customer, one 10 day delayed order, one policy, a proposed refund, and a manually approved simulated execution. Complete this deterministic vertical slice before expanding the dataset or trying every provider.

Then follow Phases 2 through 7. Notebook experiments can begin once the schemas and reference cases exist, but they must use those shared contracts.

Planning estimate: approximately 45 to 70 focused hours for a credible portfolio version. This is a rough planning range, not a delivery commitment. Provider incompatibilities, unfamiliar SDKs, local inference setup, and evaluation failures can increase it.

If scope must shrink, keep one hosted provider, the local embedding model, basic vector retrieval with trusted metadata, the approval boundary, and meaningful evaluations. Defer local generative inference, extra providers, reranking, and cloud hosting.

## 14. Prompt to start Copilot

```text
Read customer_ops_copilot_plan.md in full and inspect this repository.
Implement Phase 1 only, following existing repository instructions.
Start with the smallest deterministic vertical slice, then complete the
remaining Phase 1 deliverables. Do not add an LLM, RAG, LangGraph, or a UI yet.
Keep business rules in reusable Python modules and use strict schemas.
Run meaningful offline tests for the acceptance criteria.
Update the progress section with actual results and remaining work.
Stop after Phase 1 and summarize the changed files, validation, and next task.
```

For subsequent work, replace `Phase 1` with the requested phase and its scope. Attach this plan as repository context in Copilot. Do not assume Copilot automatically loads an arbitrary Markdown filename.

## 15. Progress record

Current phase: Phase 1 complete.

Completed phases: Phase 1 deterministic foundation.

Completed work: installable Python package; FastAPI `GET /health`; SQLite models with foreign keys and a unique per-order refund constraint; explicit seeded database initializer; 100 reproducible synthetic customers and 300 orders; versioned policy documents; strict Pydantic contracts and trusted demo contexts; scoped read and proposal tools; deterministic eligibility rules; transactional approval and rejection service; structured audit events; and offline regression tests.

Last validation results: `py -m pytest` completed with 12 passed in 1.45s. The explicit SQLite initializer was also run against a fresh database and verified 100 customers and 300 orders. `/health` returned `{"status":"ok"}` through FastAPI's test client.

Open decisions: first hosted provider and model; optional local model after hardware review.

Known blockers: no Python 3.12 interpreter was available locally. Validation used Python 3.14; project metadata requires Python 3.12 or newer.

Next task: implement Phase 2 isolated LLM experiments, beginning with a capability-explicit provider interface and deterministic fake adapter.
