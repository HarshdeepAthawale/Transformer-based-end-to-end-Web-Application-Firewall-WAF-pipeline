"""
Per-phase metrics collection: execution_count, skip_count, block_count, latency distribution.
Thread-safe via threading.Lock.
"""

import threading
from collections import defaultdict
from typing import Any


class PhaseStats:
    """Mutable counters and latency buffer for a single phase."""

    __slots__ = (
        "execution_count", "skip_count", "block_count",
        "error_count", "latencies_ms",
    )

    def __init__(self):
        self.execution_count: int = 0
        self.skip_count: int = 0
        self.block_count: int = 0
        self.error_count: int = 0
        self.latencies_ms: list[float] = []


class PipelineMetrics:
    """
    Collects per-phase metrics. Exposes snapshots for API and Prometheus.
    Latency buffer is capped at 10,000 entries per phase.
    """

    LATENCY_BUFFER_MAX = 10_000

    def __init__(self):
        self._stats: dict[str, PhaseStats] = defaultdict(PhaseStats)
        self._lock = threading.Lock()

    def record_execution(self, phase_name: str, elapsed_ms: float) -> None:
        with self._lock:
            s = self._stats[phase_name]
            s.execution_count += 1
            self._append_latency(s, elapsed_ms)

    def record_skip(self, phase_name: str, reason: str) -> None:
        with self._lock:
            self._stats[phase_name].skip_count += 1

    def record_block(self, phase_name: str, elapsed_ms: float) -> None:
        with self._lock:
            s = self._stats[phase_name]
            s.block_count += 1
            s.execution_count += 1
            self._append_latency(s, elapsed_ms)

    def record_error(self, phase_name: str, elapsed_ms: float) -> None:
        with self._lock:
            s = self._stats[phase_name]
            s.error_count += 1
            self._append_latency(s, elapsed_ms)

    def _append_latency(self, s: PhaseStats, elapsed_ms: float) -> None:
        s.latencies_ms.append(elapsed_ms)
        if len(s.latencies_ms) > self.LATENCY_BUFFER_MAX:
            s.latencies_ms = s.latencies_ms[-self.LATENCY_BUFFER_MAX:]

    def snapshot(self) -> dict[str, Any]:
        """Return serializable metrics dict for API consumption."""
        with self._lock:
            result = {}
            for name, s in sorted(self._stats.items()):
                latencies = sorted(s.latencies_ms) if s.latencies_ms else []
                n = len(latencies)
                result[name] = {
                    "execution_count": s.execution_count,
                    "skip_count": s.skip_count,
                    "block_count": s.block_count,
                    "error_count": s.error_count,
                    "latency_p50_ms": round(latencies[n // 2], 3) if n else 0.0,
                    "latency_p95_ms": round(latencies[int(n * 0.95)], 3) if n else 0.0,
                    "latency_p99_ms": round(latencies[int(n * 0.99)], 3) if n else 0.0,
                    "latency_avg_ms": round(sum(latencies) / n, 3) if n else 0.0,
                }
            return result

    def prometheus_lines(self) -> str:
        """Return Prometheus text exposition lines for per-phase metrics."""
        snap = self.snapshot()
        lines = []
        for phase_name, m in snap.items():
            pfx = f"waf_phase_{phase_name}"
            lines.append(f'{pfx}_executions_total {m["execution_count"]}')
            lines.append(f'{pfx}_skips_total {m["skip_count"]}')
            lines.append(f'{pfx}_blocks_total {m["block_count"]}')
            lines.append(f'{pfx}_errors_total {m["error_count"]}')
            lines.append(f'{pfx}_latency_p50_ms {m["latency_p50_ms"]:.3f}')
            lines.append(f'{pfx}_latency_p99_ms {m["latency_p99_ms"]:.3f}')
        return "\n".join(lines) + "\n" if lines else ""
