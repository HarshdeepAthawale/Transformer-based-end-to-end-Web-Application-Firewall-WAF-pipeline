"""
Time range parsing utility for range query params (e.g. 24h, 7d).
"""
from datetime import datetime, timedelta
from typing import Tuple


def parse_time_range(range_str: str) -> Tuple[datetime, datetime]:
    """
    Parse a range string like '1h', '24h', '7d', '30d' into (start_time, end_time).
    end_time is utcnow(); start_time is end_time - range.
    """
    now = datetime.utcnow()
    range_str = (range_str or "24h").strip().lower()
    hours = 24
    if range_str.endswith("h"):
        try:
            hours = int(range_str[:-1])
        except ValueError:
            hours = 24
    elif range_str.endswith("d"):
        try:
            hours = int(range_str[:-1]) * 24
        except ValueError:
            hours = 24
    start_time = now - timedelta(hours=hours)
    return start_time, now
