"""
Prometheus metrics from security_events and traffic (Feature 10).
Data source: DB aggregates; no mock values.
"""
from datetime import datetime, timedelta
from typing import List, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.models.security_event import SecurityEvent
from backend.config import config


# Block-type event_types (counted as blocks)
BLOCK_EVENT_TYPES = [
    "waf_block", "waf_challenge", "waf",
    "rate_limit",
    "ddos_burst", "ddos_blocked", "ddos_size",
    "bot_block", "bot_challenge",
    "upload_scan_infected",
    "credential_leak_block",
    "firewall_ai_prompt_block", "firewall_ai_pii", "firewall_ai_abuse_rate",
    "blacklist",
]


def _namespace() -> str:
    ns = getattr(config, "PROMETHEUS_METRICS_NAMESPACE", "waf") or "waf"
    return ns.rstrip("_")


def _metric_name(name: str) -> str:
    return f"{_namespace()}_{name}"


def gather_metrics(db: Session, window_minutes: int = 60) -> List[Tuple[str, dict, float]]:
    """
    Query security_events for the last window_minutes and return list of
    (metric_name, labels_dict, value) for counters/gauges.
    """
    start_time = datetime.utcnow() - timedelta(minutes=window_minutes)
    results = []

    # Total events in window (requests)
    total = db.query(func.count(SecurityEvent.id)).filter(
        SecurityEvent.timestamp >= start_time
    ).scalar() or 0

    # Block events in window
    block_count = db.query(func.count(SecurityEvent.id)).filter(
        SecurityEvent.event_type.in_(BLOCK_EVENT_TYPES),
        SecurityEvent.timestamp >= start_time,
    ).scalar() or 0

    allow_count = max(0, total - block_count)

    ns = _metric_name("requests_total")
    results.append((ns, {"outcome": "allow"}, float(allow_count)))
    results.append((ns, {"outcome": "block"}, float(block_count)))

    ns_total = _metric_name("blocks_total")
    results.append((ns_total, {}, float(block_count)))

    # Attack score: avg and count (for histogram-like usage)
    attack_row = db.query(
        func.avg(SecurityEvent.attack_score).label("avg"),
        func.count(SecurityEvent.id).label("count"),
    ).filter(
        SecurityEvent.timestamp >= start_time,
        SecurityEvent.attack_score.isnot(None),
    ).first()
    if attack_row and attack_row.count and attack_row.avg is not None:
        results.append((_metric_name("attack_score_avg"), {}, float(attack_row.avg)))
        results.append((_metric_name("attack_score_count"), {}, float(attack_row.count)))

    return results


def format_prometheus_exposition(metrics: List[Tuple[str, dict, float]]) -> str:
    """Format as Prometheus text exposition format (grouped by metric name)."""
    from collections import defaultdict
    by_name = defaultdict(list)
    for name, labels, value in metrics:
        by_name[name].append((labels, value))
    out = []
    for name in sorted(by_name.keys()):
        type_ = "counter" if (name.endswith("_total") or name.endswith("_count")) else "gauge"
        out.append(f"# HELP {name} WAF metric")
        out.append(f"# TYPE {name} {type_}")
        for labels, value in by_name[name]:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            if label_str:
                out.append(f"{name}{{{label_str}}} {value}")
            else:
                out.append(f"{name} {value}")
    return "\n".join(out) + "\n"


def get_prometheus_text(db: Session, window_minutes: int = 60) -> str:
    """Return full Prometheus exposition format string for GET /metrics."""
    metrics = gather_metrics(db, window_minutes)
    return format_prometheus_exposition(metrics)
