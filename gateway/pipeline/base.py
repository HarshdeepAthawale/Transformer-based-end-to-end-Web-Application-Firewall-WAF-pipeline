"""Phase base class and enums for the WAF pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from gateway.pipeline.context import PhaseContext, PhaseResult


class PhaseVerdict(str, Enum):
    CONTINUE = "continue"
    SHORT_CIRCUIT = "short_circuit"


class PhaseMode(str, Enum):
    ENFORCE = "enforce"
    MONITOR = "monitor"
    DISABLED = "disabled"


class Phase(ABC):
    """
    Base class for all WAF pipeline phases.

    Each phase declares:
    - name: unique string identifier
    - order: integer for strict ordering (10, 20, ... 100)
    - requires_body: whether body_bytes must be populated before this phase
    - should_run(ctx): O(1) filter controlling whether this phase executes
    - execute(ctx): async method that does the actual work
    """

    name: str = ""
    order: int = 0
    requires_body: bool = False

    @abstractmethod
    def should_run(self, ctx: PhaseContext) -> tuple[bool, Optional[str]]:
        """
        Return (should_run, skip_reason).

        If should_run is False, skip_reason explains why (for metrics/logging).
        Must be O(1) or O(log n) -- never slower than the phase itself.
        """
        ...

    @abstractmethod
    async def execute(self, ctx: PhaseContext) -> PhaseResult:
        """
        Execute the phase logic. Must return a PhaseResult.

        To short-circuit the pipeline, set result.verdict = PhaseVerdict.SHORT_CIRCUIT.
        To continue to the next phase, set result.verdict = PhaseVerdict.CONTINUE.
        """
        ...

    def get_mode(self, ctx: PhaseContext) -> PhaseMode:
        """Return the mode for this phase. Override for custom logic."""
        from gateway.pipeline.config import get_phase_mode

        return get_phase_mode(self.name, ctx.org_id)
