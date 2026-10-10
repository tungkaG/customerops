from __future__ import annotations

from dataclasses import asdict, dataclass

from customer_ops.retrieval.index import LocalVectorIndex
from customer_ops.retrieval.models import RetrievalFilter, RetrievalHit


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvaluatedHit:
    chunk_id: str
    citation: str
    score: float
    relevant: bool
    source_type: str
    # True when trusted ingestion classified the chunk as active, trusted evidence.
    valid_evidence: bool


@dataclass(frozen=True)
class EvaluatedCase:
    query: str
    relevant_chunk_ids: tuple[str, ...]
    hits: list[EvaluatedHit]


@dataclass(frozen=True)
class RetrievalEvaluation:
    cases: list[EvaluatedCase]

    @property
    def recall_at_1(self) -> float | None:
        return _recall_at(self.cases, 1)

    @property
    def recall_at_3(self) -> float | None:
        return _recall_at(self.cases, 3)

    @property
    def mean_reciprocal_rank(self) -> float | None:
        if not self.cases:
            return None
        return sum(_reciprocal_rank(case) for case in self.cases) / len(self.cases)

    @property
    def coverage(self) -> float | None:
        if not self.cases:
            return None
        return sum(bool(case.hits) for case in self.cases) / len(self.cases)

    @property
    def contamination_at_1(self) -> float | None:
        """Share of questions whose top result is not active, trusted evidence."""
        if not self.cases:
            return None
        return sum(bool(case.hits) and not case.hits[0].valid_evidence for case in self.cases) / len(self.cases)

    def summary(self) -> dict[str, float | int | None]:
        return {
            "questions": len(self.cases),
            "recall_at_1": self.recall_at_1,
            "recall_at_3": self.recall_at_3,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "coverage": self.coverage,
            "contamination_at_1": self.contamination_at_1,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.summary(), "cases": [asdict(case) for case in self.cases]}


def evaluate_retrieval(
    index: LocalVectorIndex,
    cases: list[RetrievalCase],
    *,
    retrieval_filter: RetrievalFilter | None = None,
) -> RetrievalEvaluation:
    """Score top-three retrieval results and retain per-hit relevance evidence."""
    evaluated_cases = []
    for case in cases:
        hits = index.search(case.query, retrieval_filter=retrieval_filter, limit=3)
        evaluated_cases.append(
            EvaluatedCase(
                query=case.query,
                relevant_chunk_ids=case.relevant_chunk_ids,
                hits=[_evaluated_hit(hit, case.relevant_chunk_ids) for hit in hits],
            )
        )
    return RetrievalEvaluation(evaluated_cases)


def _evaluated_hit(hit: RetrievalHit, relevant_chunk_ids: tuple[str, ...]) -> EvaluatedHit:
    metadata = hit.chunk.metadata
    return EvaluatedHit(
        chunk_id=hit.chunk.chunk_id,
        citation=hit.citation,
        score=hit.score,
        relevant=hit.chunk.chunk_id in relevant_chunk_ids,
        source_type=metadata.source_type,
        valid_evidence=metadata.trusted and metadata.status == "active",
    )


def _recall_at(cases: list[EvaluatedCase], limit: int) -> float | None:
    if not cases:
        return None
    return sum(any(hit.relevant for hit in case.hits[:limit]) for case in cases) / len(cases)


def _reciprocal_rank(case: EvaluatedCase) -> float:
    for position, hit in enumerate(case.hits, start=1):
        if hit.relevant:
            return 1 / position
    return 0.0