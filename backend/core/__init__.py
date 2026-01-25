"""
Backend core utilities: WAF factory, time-range parsing, etc.
"""
from backend.core.waf_factory import create_waf_service, get_waf_service
from backend.core.time_range import parse_time_range

__all__ = ["create_waf_service", "get_waf_service", "parse_time_range"]
