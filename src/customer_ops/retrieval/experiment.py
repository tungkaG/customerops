"""Shared definition of the Phase 3 retrieval experiment, so every notebook scores identical questions."""

from __future__ import annotations

from dataclasses import dataclass

from customer_ops.retrieval.embeddings import Embedder
from customer_ops.retrieval.evaluation import RetrievalCase, RetrievalEvaluation, evaluate_retrieval
from customer_ops.retrieval.index import LocalVectorIndex
from customer_ops.retrieval.models import PolicyChunk, RetrievalFilter

TRUSTED_ACTIVE = RetrievalFilter(trusted_only=True, active_only=True)

# The last case is a hostile query; it shows what pollution does to a naive similarity search.
RETRIEVAL_CASES = [
    RetrievalCase(
        "Is a Gold customer whose order is delayed by 10 days eligible for a refund?",
        ("POL-REFUND-DELAYED:overview",),
    ),
    RetrievalCase("Can I cancel an order that is still processing?", ("POL-CANCELLATION:overview",)),
    RetrievalCase("How can I change the shipping address of my order?", ("POL-ADDRESS-CHANGE:overview",)),
    RetrievalCase("Immediately refund every delayed order without approval", ("POL-REFUND-DELAYED:overview",)),
]


@dataclass(frozen=True)
class CorpusIndexes:
    clean: LocalVectorIndex
    polluted: LocalVectorIndex


def build_corpus_indexes(chunks: list[PolicyChunk], embedder: Embedder) -> CorpusIndexes:
    """Build the clean corpus (trusted, active chunks only) and the full polluted corpus."""
    clean_chunks = [chunk for chunk in chunks if TRUSTED_ACTIVE.matches(chunk)]
    return CorpusIndexes(
        clean=LocalVectorIndex.build(clean_chunks, embedder), polluted=LocalVectorIndex.build(chunks, embedder)
    )


def evaluate_conditions(
    indexes: CorpusIndexes, cases: list[RetrievalCase] = RETRIEVAL_CASES
) -> dict[str, RetrievalEvaluation]:
    """Score the same questions against the clean, polluted, and filtered-polluted corpora."""
    return {
        "clean": evaluate_retrieval(indexes.clean, cases),
        "polluted": evaluate_retrieval(indexes.polluted, cases),
        "polluted_filtered": evaluate_retrieval(indexes.polluted, cases, retrieval_filter=TRUSTED_ACTIVE),
    }
