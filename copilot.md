Rebuild this project as a minimal autonomous customer support experiment.

Objective

Evaluate whether an LLM can independently select tools, gather sufficient evidence, interpret company policies, and draft an appropriate request for a human operator.

The model chooses every business tool call and its order. Application code only implements the tools, executes valid calls, maintains conversation history, and bounds the loop.

Start with a fresh implementation. Inspect the repository first and preserve the existing work through Git before replacing it. Reuse a working model adapter if useful, but do not carry over the old workflow architecture.

Keep implementation small. Do not create a general agent framework.

Phase 1: Minimal structure and data

Use approximately this structure:

```text
data/
    orders.json
    policies/
        *.md

src/
    agent.py
    tools.py
    retrieval.py
    model.py

tests/

demo.py
README.md
```

Adapt to existing packaging conventions if necessary. Do not create files merely to match this layout.

Policies are authored by me, trusted, and stored under the repository’s data directory. Support Markdown and plain text only.

Use a small local JSON order fixture. Include enough verified facts to evaluate the policies, such as status, customer tier, dates, and amount.

Keep dates reproducible with a configurable demo date. Do not require a database, authentication system, web server, email service, or approval workflow.

Phase 2: Minimal dense policy retrieval

Use Sentence Transformers and its existing semantic search utility.

Use a small configurable embedding model, initially:

```text
sentence-transformers/multi-qa-MiniLM-L6-cos-v1
```

Load the embedding model once. Load policies and compute corpus embeddings once at startup. Keep everything in memory.

Each policy file is one document and one retrieval entry.

The initial policies should be short enough to fit within the embedding model’s input limit. Detect oversized documents and report the limitation rather than silently truncating them.

Implement:

```text
search_policy(query: str)
```

Its only model supplied argument is a free text query string.

Use cosine similarity and return the single highest ranked policy document, including its full text, filename or document ID, and similarity score.

Return one document when the corpus is nonempty. Return an explicit error if policies are unavailable.

Do not add category filters, known policy mappings, reranking, a vector database, hybrid search, or automatic follow up searches.

The model decides whether the returned document is relevant or sufficient. If it needs more evidence, it must call search_policy again with another query.

Do not automatically hide previously returned documents. An identical query may return the same document again.

Phase 3: Tool definitions and placeholder handlers

Provide a small tool registry containing tool names, descriptions, argument schemas, and handlers.

Expose these tools:

```text
get_order(order_id: str)

search_policy(query: str)

draft_refund(text: str)

draft_cancellation(text: str)

draft_address_change(text: str)
```

get_order returns the verified fixture record or an explicit not found result. Never invent or repair an order ID.

Each drafting tool accepts exactly one free text string.

Describe drafting tools as preparing a request for human operator review, with text suitable for an internal email or support ticket.

Descriptions must explain the purpose of each tool clearly. Generic system instructions do not mean generic or ambiguous tool descriptions.

For example, draft_cancellation prepares a cancellation proposal for a human operator to review. The draft should explain the customer request, order, relevant facts, policy basis, and requested next step.

The handlers are placeholders. They return a receipt containing the draft text and a clear indication that it was recorded locally, nothing was sent, and no order or money was changed.

Do not send email, create real tickets, or execute business operations.

Do not add deterministic eligibility checks, mandatory evidence checks, citation checks, or operation routing to these placeholder tools. This experiment measures whether the model follows the instructions itself.

Do retain basic argument validation and unknown tool handling.

Phase 4: One autonomous agent loop

Implement one reusable loop:

1. Start with a system message and the customer message.
2. Call the model with the available tool definitions.
3. Append the assistant response and its tool calls to history.
4. Execute requested tools and append their results.
5. Call the model again with the updated history.
6. Stop when the model returns a final response without tool calls, or a configurable call budget is reached.

Support multiple tool calls in a response. Execute them sequentially and preserve provider specific call identifiers where required.

If a tool fails or arguments are invalid, return a clear tool error to the model so it can decide how to recover. Never secretly correct its arguments.

If model output cannot be parsed, report that failure visibly. Do not fabricate a tool result or final response.

Keep a configurable maximum number of model turns, initially 8. Enforce it before the next model call.

Do not stop automatically after a drafting tool call. Give the model its receipt so it can provide the final customer response.

Preserve full conversation history during a run. Allow the caller to continue that history with a follow up customer message.

Do not add intent classification, required evidence collection, a separate DecisionProposal stage, capability dependency graphs, or a separate validation and action stage.

Do not automatically call any business tool. Every call must originate from model output.

Use the existing working Hugging Face tool calling adapter if suitable. Model choice must be configurable. Do not add multiple provider integrations unless already available and reusable.

Phase 5: Generic system prompt

Use this as the initial prompt:

```text
You are a customer support assistant. Help the customer using the
available tools and the information they return.

Our company has policies that must be followed when handling requests
that involve a business action. Obtain and consider the applicable
policies before preparing an action for review.

Choose the tools and their order yourself. If the information returned
is insufficient, gather more information. If necessary details are
missing or ambiguous, ask the customer.

Use verified order facts and retrieved policy text. Do not invent
identifiers, facts, policies, or tool results.

Preparing a draft does not execute an action. Describe only what the
tools confirm happened.
```

Do not list operation names in the system prompt.

Do not prescribe a fixed tool sequence or insert reminders based on a classified intent.

Specific operation descriptions belong in the tools. Specific eligibility conditions belong in policy documents.

Keep the prompt short. Do not add examples or extra instructions until live evaluation shows a reason to test them.

Phase 6: Trace and verification

Record observable events:

1. Model response text.
2. Requested tool name and arguments.
3. Tool result or error.
4. Final response or termination reason.
5. Token usage when available.

Do not request hidden chain of thought. Do not label model commentary as access to its internal reasoning.

Add focused offline tests for:

1. Conversation history includes assistant calls and matching tool results.
2. Multiple calls are executed and returned correctly.
3. Errors are visible to the model.
4. Draft handlers only record text.
5. Retrieval returns at most one complete document.
6. The model call budget is enforced.

Use scripted model responses for these infrastructure tests. Passing them does not establish model intelligence.

Provide a small live evaluation set with:

1. A status inquiry.
2. An eligible action request.
3. An ineligible action request.
4. A request with no order ID, requiring clarification.
5. An unknown order.
6. A question whose answer requires two separate policy documents.
7. A follow up message making a previously vague request explicit.

For the two document case, author policies whose relevant conditions are split across documents and reference each other naturally.

Observe whether the model searches, notices insufficient evidence, searches again, drafts the correct operation, and accurately reports the placeholder result.

Do not enforce these expected behaviors in the loop. Report failures as failures.

Phase 7: Runnable demo and cleanup

Provide a simple command line demo that prints the trace and final response.

Document:

1. Installation and model configuration.
2. Where I place orders and policies.
3. How to run one scenario and continue with a follow up.
4. How to change the model and compare the same scenarios.
5. The distinction between scripted infrastructure tests and live behavior evaluation.

Remove obsolete workflow modules from the active implementation. Preserve the previous implementation through Git history rather than keeping two competing architectures.

No LangGraph is necessary for this loop. No persistent memory, multi agent orchestration, production services, or speculative abstractions.

Implement and verify one phase at a time. Keep the repository runnable as functionality is added. Report what was actually tested, and do not claim live model success without running the model.