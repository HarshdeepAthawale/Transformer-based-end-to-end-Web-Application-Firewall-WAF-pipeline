"""
Per-phase mode configuration.

Priority: per-phase env var > global WAF_MODE > default (enforce).

Env var pattern: WAF_PHASE_{NAME}_MODE=enforce|monitor|disabled
Example: WAF_PHASE_WAF_ML_MODE=monitor
"""

import os
from functools import lru_cache
from typing import Optional

from gateway.pipeline.base import PhaseMode

_GLOBAL_DEFAULT = os.getenv("WAF_MODE", "block")


@lru_cache(maxsize=64)
def _env_phase_mode(phase_name: str) -> Optional[PhaseMode]:
    """Read WAF_PHASE_{NAME}_MODE from env. Cached per process."""
    key = f"WAF_PHASE_{phase_name.upper()}_MODE"
    val = os.getenv(key, "").lower().strip()
    if val in ("enforce", "block"):
        return PhaseMode.ENFORCE
    if val == "monitor":
        return PhaseMode.MONITOR
    if val == "disabled":
        return PhaseMode.DISABLED
    return None


def get_phase_mode(phase_name: str, org_id: Optional[str] = None) -> PhaseMode:
    """
    Resolve phase mode with priority:
    1. Per-phase env var (WAF_PHASE_{NAME}_MODE)
    2. Global WAF_MODE
    3. Default: enforce
    """
    env_mode = _env_phase_mode(phase_name)
    if env_mode is not None:
        return env_mode

    if _GLOBAL_DEFAULT == "monitor":
        return PhaseMode.MONITOR
    return PhaseMode.ENFORCE
