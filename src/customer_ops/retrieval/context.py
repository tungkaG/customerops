from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from customer_ops.retrieval.models import RetrievalHit


class EvidenceStatus(StrEnum):
    VALID = "valid"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class EvidenceContext:
    status: EvidenceStatus
    text: str
    citations: list[str]


def build_evidence_context(hits: list[RetrievalHit]) -> EvidenceContext:
    """Format retrieved policy evidence, or explicitly abstain when none qualifies."""
    if not hits:
        return EvidenceContext(EvidenceStatus.ABSTAIN, "", [])
    return EvidenceContext(
        EvidenceStatus.VALID,
        "\n\n".join(f"[{hit.citation}]\n{hit.chunk.text}" for hit in hits),
        [hit.citation for hit in hits],
    )