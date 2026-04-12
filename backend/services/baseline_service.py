"""
Behavioral Baseline Service (Cloudflare-style Z-score anomaly detection).

Computes rolling 30-day hourly baselines for key metrics and detects
anomalies when current values exceed 3 standard deviations from baseline.
"""

import math
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models.security_event import SecurityEvent


# Z-score threshold for anomaly detection (Cloudflare uses 3.0)
Z_SCORE_THRESHOLD = 3.0

# Number of days for baseline calculation
BASELINE_WINDOW_DAYS = 30


class BaselineService:
    """Computes behavioral baselines and detects anomalies via Z-scores."""

    def compute_hourly_baseline(
        self, db: Session, metric: str, hour: int, org_id: Optional[int] = None
    ) -> dict:
        """
        Compute mean and std_dev for a metric at a specific hour-of-day
        over the last 30 days.

        Args:
            db: Database session
            metric: One of "request_volume", "block_rate", "bot_volume", "unique_ips"
            hour: Hour of day (0-23)
            org_id: Optional org filter

        Returns:
            Dict with mean, std_dev, sample_count
        """
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=BASELINE_WINDOW_DAYS)

        # Query hourly counts from security_events for the given hour-of-day
        values = []

        for day_offset in range(BASELINE_WINDOW_DAYS):
            day = start_date + timedelta(days=day_offset)
            hour_start = day.replace(hour=hour, minute=0, second=0, microsecond=0)
            hour_end = hour_start + timedelta(hours=1)

            q = db.query(func.count(SecurityEvent.id)).filter(
                SecurityEvent.timestamp >= hour_start,
                SecurityEvent.timestamp < hour_end,
            )
            if org_id is not None:
                q = q.filter(SecurityEvent.org_id == org_id)

            if metric == "bot_volume":
                q = q.filter(SecurityEvent.bot_score < 30)
            elif metric == "block_rate":
                # Count blocked events
                q = q.filter(SecurityEvent.event_type.in_([
                    "waf_block", "rate_limit", "ddos_burst", "bot_block",
                ]))

            count = q.scalar() or 0
            values.append(count)

        if not values:
            return {"mean": 0.0, "std_dev": 0.0, "sample_count": 0}

        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
        std_dev = math.sqrt(variance)

        return {
            "mean": round(mean, 2),
            "std_dev": round(std_dev, 2),
            "sample_count": len(values),
        }

    def compute_z_score(
        self, current_value: float, mean: float, std_dev: float
    ) -> float:
        """
        Compute Z-score: how many standard deviations from the mean.
        |Z| > 3.0 is considered anomalous (Cloudflare's threshold).
        """
        if std_dev == 0:
            return 0.0 if current_value == mean else float("inf")
        return round((current_value - mean) / std_dev, 2)

    def is_anomalous(self, z_score: float) -> bool:
        """Check if Z-score exceeds anomaly threshold."""
        return abs(z_score) > Z_SCORE_THRESHOLD

    def check_current_hour(
        self, db: Session, metric: str, org_id: Optional[int] = None
    ) -> dict:
        """
        Check if the current hour's metric value is anomalous compared to baseline.

        Returns:
            Dict with current_value, baseline (mean, std_dev), z_score, is_anomalous
        """
        now = datetime.now(timezone.utc)
        hour = now.hour
        hour_start = now.replace(minute=0, second=0, microsecond=0)

        # Get current hour's value
        q = db.query(func.count(SecurityEvent.id)).filter(
            SecurityEvent.timestamp >= hour_start,
            SecurityEvent.timestamp < now,
        )
        if org_id is not None:
            q = q.filter(SecurityEvent.org_id == org_id)

        if metric == "bot_volume":
            q = q.filter(SecurityEvent.bot_score < 30)
        elif metric == "block_rate":
            q = q.filter(SecurityEvent.event_type.in_([
                "waf_block", "rate_limit", "ddos_burst", "bot_block",
            ]))

        current_value = q.scalar() or 0

        # Get baseline
        baseline = self.compute_hourly_baseline(db, metric, hour, org_id)
        z_score = self.compute_z_score(current_value, baseline["mean"], baseline["std_dev"])

        return {
            "metric": metric,
            "hour": hour,
            "current_value": current_value,
            "baseline_mean": baseline["mean"],
            "baseline_std_dev": baseline["std_dev"],
            "z_score": z_score,
            "is_anomalous": self.is_anomalous(z_score),
        }
