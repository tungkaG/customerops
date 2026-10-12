"""Optional Langfuse tracing. Everything is a no-op until enable() is called, so agent behavior never depends on it."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any

import httpx
from langfuse import Langfuse, get_client, propagate_attributes
from langfuse.api.core.api_error import ApiError

_client: Langfuse | None = None


class _NoOp:
    trace_id = None

    def update(self, **_: Any) -> "_NoOp":
        return self


def credentials_present() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def enable(client: Langfuse | None = None) -> Langfuse:
    """Initialize tracing once; the client comes from the LANGFUSE_* environment variables unless given."""
    global _client
    _client = client or get_client()
    return _client


def disable() -> None:
    global _client
    _client = None


def enabled() -> bool:
    return _client is not None


def observation(as_type: str, name: str, **fields: Any):
    """Context manager yielding an observation that is the active parent for anything created inside it."""
    if _client is None:
        return nullcontext(_NoOp())
    return _client.start_as_current_observation(as_type=as_type, name=name, **fields)


def session(session_id: str):
    return nullcontext() if _client is None else propagate_attributes(session_id=session_id)


def trace_url(trace_id: str | None) -> str | None:
    """Look up the UI link (one API call for the project ID); an unreachable server just means no link."""
    if _client is None or not trace_id:
        return None
    try:
        return _client.get_trace_url(trace_id=trace_id)
    except (httpx.HTTPError, ApiError):
        return None


def flush() -> None:
    if _client is not None:
        _client.flush()
