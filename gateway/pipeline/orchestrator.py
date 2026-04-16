"""PipelineOrchestrator: runs phases in strict order with timing, filters, and mode handling."""

from __future__ import annotations

import time
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from loguru import logger

from gateway.pipeline.base import Phase, PhaseMode, PhaseVerdict
from gateway.pipeline.context import PhaseContext, PhaseResult, PhaseTimingEntry
from gateway.pipeline.metrics import PipelineMetrics


class PipelineOrchestrator:
    """
    Runs phases in strict fixed order. Handles:
    - O(1) filter checks (should_run)
    - Lazy body read (before first phase that needs it)
    - Per-phase timing and metrics
    - Mode enforcement (enforce / monitor / disabled)
    - Short-circuit with shadow mode support
    """

    def __init__(self, phases: list[Phase], metrics: PipelineMetrics):
        self.phases = sorted(phases, key=lambda p: p.order)
        self.metrics = metrics

        # Validate no duplicate names
        names = [p.name for p in self.phases]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate phase names: {names}")

    async def run(self, request: Request, ctx: PhaseContext) -> PhaseContext:
        """Execute all phases in order. Returns the (mutated) context."""
        for phase in self.phases:
            # Lazy body read: read just before first phase that needs it
            if phase.requires_body and not ctx.body_read:
                try:
                    ctx.body_bytes = await request.body()
                except Exception:
                    ctx.body_bytes = b""
                ctx.body_read = True

            # Check phase mode
            mode = phase.get_mode(ctx)
            if mode == PhaseMode.DISABLED:
                self.metrics.record_skip(phase.name, "disabled")
                ctx.timings.append(PhaseTimingEntry(
                    phase_name=phase.name,
                    started_at=time.perf_counter(),
                    elapsed_ms=0.0,
                    skipped=True,
                    skip_reason="disabled",
                ))
                continue

            # Check filter
            should_run, skip_reason = phase.should_run(ctx)
            if not should_run:
                self.metrics.record_skip(phase.name, skip_reason or "filter")
                ctx.timings.append(PhaseTimingEntry(
                    phase_name=phase.name,
                    started_at=time.perf_counter(),
                    elapsed_ms=0.0,
                    skipped=True,
                    skip_reason=skip_reason,
                ))
                continue

            # Execute phase with timing
            t0 = time.perf_counter()
            try:
                result = await phase.execute(ctx)
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000
                logger.error(f"Phase {phase.name} error: {exc}")
                self.metrics.record_error(phase.name, elapsed)
                ctx.timings.append(PhaseTimingEntry(
                    phase_name=phase.name,
                    started_at=t0,
                    elapsed_ms=elapsed,
                ))
                # Fail-open by default for individual phase errors
                continue

            elapsed = (time.perf_counter() - t0) * 1000
            result.mode = mode.value

            # Store result and timing
            ctx.phase_results[phase.name] = result
            ctx.phases_run.append(phase.name)
            ctx.timings.append(PhaseTimingEntry(
                phase_name=phase.name,
                started_at=t0,
                elapsed_ms=elapsed,
            ))

            # Record metrics
            is_short_circuit = result.verdict == PhaseVerdict.SHORT_CIRCUIT
            if is_short_circuit:
                self.metrics.record_block(phase.name, elapsed)
            else:
                self.metrics.record_execution(phase.name, elapsed)

            # Handle short circuit
            if is_short_circuit:
                if mode == PhaseMode.MONITOR:
                    # Shadow mode: log the would-block, continue pipeline
                    logger.info(
                        f"Phase {phase.name} MONITOR "
                        f"(would {result.action}): {ctx.method} {ctx.path}"
                    )
                    result.verdict = PhaseVerdict.CONTINUE.value
                    if ctx.final_verdict == "allow":
                        ctx.final_verdict = "monitor_would_block"
                        ctx.blocking_phase = phase.name
                    continue
                else:
                    # Enforce mode: short circuit the pipeline
                    ctx.final_verdict = result.action or "block"
                    ctx.blocking_phase = phase.name
                    return ctx

        return ctx

    def build_block_response(self, ctx: PhaseContext) -> Response:
        """Build HTTP response for a short-circuited pipeline."""
        result = ctx.phase_results.get(ctx.blocking_phase)
        if not result:
            return JSONResponse(
                status_code=403,
                content={"blocked": True, "message": "Blocked by WAF"},
            )
        status = result.status_code or 403
        body = result.response_body or {"blocked": True, "message": "Blocked by WAF"}
        resp = JSONResponse(status_code=status, content=body)
        if result.response_headers:
            for k, v in result.response_headers.items():
                resp.headers[k] = v
        resp.headers["X-Request-ID"] = ctx.request_id
        resp.headers["X-WAF-Phases-Run"] = ",".join(ctx.phases_run)
        return resp

    def add_observability_headers(self, response: Response, ctx: PhaseContext) -> None:
        """Add pipeline observability headers to any response."""
        response.headers["X-Request-ID"] = ctx.request_id
        elapsed_ms = (time.perf_counter() - ctx.start_time) * 1000
        response.headers["X-Gateway-Time-Ms"] = f"{elapsed_ms:.1f}"
        response.headers["X-WAF-Phases-Run"] = ",".join(ctx.phases_run)
        # WAF score from ML phase if it ran
        waf_result = ctx.phase_results.get("waf_ml")
        if waf_result and waf_result.data.get("attack_score") is not None:
            response.headers["X-WAF-Score"] = str(waf_result.data["attack_score"])
