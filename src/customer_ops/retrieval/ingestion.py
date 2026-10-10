from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Literal

from customer_ops.domain.schemas import StrictModel
from customer_ops.retrieval.models import PolicyChunk, PolicyMetadata


class SourceClassification(StrictModel):
    """Provenance assigned by trusted ingestion, never by claims inside a document."""

    source_type: str
    trusted: bool
    status: Literal["active", "obsolete"]


NEUTRAL_SOURCE = SourceClassification(source_type="unclassified", trusted=False, status="active")


def load_source_registry(path: Path) -> dict[str, SourceClassification]:
    """Load the filename-keyed provenance registry maintained outside the document corpus."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    return {filename: SourceClassification.model_validate(entry) for filename, entry in entries.items()}


def ingest_policy_directory(
    directory: Path, registry: dict[str, SourceClassification] | None = None
) -> list[PolicyChunk]:
    """Ingest every Markdown file as a retrieval candidate; unlisted files stay neutral and untrusted."""
    registry = registry or {}
    paths = sorted(directory.glob("*.md"))
    unknown = set(registry) - {path.name for path in paths}
    if unknown:
        raise ValueError(f"Registry lists files that are not in {directory}: {sorted(unknown)}")
    chunks: list[PolicyChunk] = []
    for path in paths:
        source = registry.get(path.name, NEUTRAL_SOURCE)
        chunks.extend(
            ingest_policy_file(path, trusted=source.trusted, source_type=source.source_type, status=source.status)
        )
    return chunks


def ingest_policy_file(
    path: Path,
    *,
    trusted: bool = False,
    source_type: str = "unclassified",
    status: str = "active",
) -> list[PolicyChunk]:
    """Parse one Markdown document into independently citable section chunks."""
    text = path.read_text(encoding="utf-8")
    document_id = _required_match(r"^#\s+(.+)$", text, "document ID", path)
    version = _required_match(r"^Version:\s*(.+)$", text, "version", path)
    category = _required_match(r"^Category:\s*(.+)$", text, "category", path)
    metadata = PolicyMetadata(
        document_id=document_id,
        category=category,
        version=version,
        status=status,
        source_type=source_type,
        effective_date=_version_date(version, path),
        trusted=trusted,
    )
    sections = _sections(text, document_id)
    return [PolicyChunk(chunk_id=chunk_id, metadata=metadata, text=section_text) for chunk_id, section_text in sections]


def _required_match(pattern: str, text: str, label: str, path: Path) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"{path} is missing a {label}.")
    return match.group(1).strip()


def _version_date(version: str, path: Path) -> date:
    try:
        return date.fromisoformat(f"{version}-01")
    except ValueError as error:
        raise ValueError(f"{path} has an invalid YYYY-MM version: {version!r}.") from error


def _sections(text: str, document_id: str) -> list[tuple[str, str]]:
    headings = list(re.finditer(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE))
    if not headings:
        body = _body_without_header(text).strip()
        return [(f"{document_id}:overview", body)] if body else []

    sections: list[tuple[str, str]] = []
    for position, heading in enumerate(headings):
        end = headings[position + 1].start() if position + 1 < len(headings) else len(text)
        section_text = text[heading.end() : end].strip()
        if section_text:
            slug = re.sub(r"[^a-z0-9]+", "-", heading.group(1).lower()).strip("-")
            sections.append((f"{document_id}:{slug}", section_text))
    return sections


def _body_without_header(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(
        line
        for line in lines
        if not line.startswith("# ") and not line.startswith("Version:") and not line.startswith("Category:")
    )