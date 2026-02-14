"""
Integration module for standalone WAF service (Nginx, etc.).
"""

from integration.waf_service import app, initialize_waf_service

__all__ = ["app", "initialize_waf_service"]
