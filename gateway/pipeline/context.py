"""PhaseContext: carries request data and phase results through the pipeline."""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field


class PhaseTimingEntry(BaseModel):
    """Timing record for a single phase execution."""

    phase_name: str
    started_at: float
    elapsed_ms: float
    skipped: bool = False
    skip_reason: Optional[str] = None


class PhaseResult(BaseModel):
    """Result stored by each phase into ctx.phase_results[phase_name]."""

    phase_name: str
    verdict: str  # "continue" | "short_circuit"
    action: Optional[str] = None  # "block" | "challenge" | "cache_hit" | etc.
    status_code: Optional[int] = None
    response_body: Optional[dict[str, Any]] = None
    response_headers: Optional[dict[str, str]] = None
    mode: str = "enforce"
    # Phase-specific data (typed loosely for extensibility)
    data: dict[str, Any] = Field(default_factory=dict)


class PhaseContext(BaseModel):
    """
    Request context carried across all pipeline phases.

    Populated once by the orchestrator before the pipeline runs.
    Each phase reads from here and writes its PhaseResult.
    """

    # Request identity
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = Field(default_factory=time.perf_counter)

    # Raw request data
    client_ip: str = "unknown"
    method: str = "GET"
    path: str = "/"
    query_string: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    body_bytes: Optional[bytes] = None
    content_type: str = ""

    # Tenant context
    org_id: Optional[str] = None

    # Body is read lazily between pre-body and post-body phases
    body_read: bool = False

    # Phase outputs
    phase_results: dict[str, PhaseResult] = Field(default_factory=dict)
    timings: list[PhaseTimingEntry] = Field(default_factory=list)
    phases_run: list[str] = Field(default_factory=list)

    # Final verdict (set by orchestrator after pipeline completes)
    final_verdict: str = "allow"
    blocking_phase: Optional[str] = None

    # Edge cache context for store-on-miss after forwarding
    cache_ctx: Optional[Any] = None

    model_config = {"arbitrary_types_allowed": True}
