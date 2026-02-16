#!/usr/bin/env python3
"""
Production Environment Validation Script

Checks required environment variables and warns on development defaults.
Run before deploying to production: python scripts/validate_production_env.py
"""
import os
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(_project_root / ".env")
except ImportError:
    pass


REQUIRED = [
    ("DATABASE_URL", "PostgreSQL connection string (not SQLite)"),
    ("REDIS_URL", "Redis for rate limit, DDoS, blacklist"),
    ("NEXTAUTH_SECRET", "32+ char random string for auth"),
    ("MONGODB_URI", "Gateway event store"),
    ("UPSTREAM_URL", "Protected application URL (gateway)"),
]

PRODUCTION_WARN = [
    ("CORS_ORIGINS", "localhost", "Restrict to production domains"),
    ("NEXT_PUBLIC_API_URL", "localhost", "Use production API URL"),
    ("NEXT_PUBLIC_WS_URL", "localhost", "Use production WebSocket URL (wss://)"),
    ("NEXTAUTH_SECRET", "your-random-secret", "Generate a secure random secret"),
]


def main() -> int:
    errors = []
    warnings = []

    for var, desc in REQUIRED:
        val = os.getenv(var)
        if not val or not val.strip():
            errors.append(f"  {var}: missing - {desc}")
        elif var == "DATABASE_URL" and "sqlite" in val.lower():
            errors.append(f"  {var}: SQLite not recommended for production - {desc}")

    for var, bad_val, msg in PRODUCTION_WARN:
        val = os.getenv(var)
        if val and bad_val.lower() in val.lower():
            warnings.append(f"  {var}: appears to be dev default - {msg}")

    # WAF model check
    try:
        from backend.core.waf_factory import is_model_available
        if not is_model_available():
            if os.getenv("WAF_ENABLED", "true").lower() == "true":
                warnings.append("  WAF model: models/waf-distilbert not found or incomplete (config.json, tokenizer.json)")
    except Exception:
        warnings.append("  WAF model: could not verify (backend imports may fail)")

    print("Production environment validation")
    print("=" * 50)

    if errors:
        print("\nERRORS (must fix before production):")
        for e in errors:
            print(e)
        print()

    if warnings:
        print("\nWARNINGS (review before production):")
        for w in warnings:
            print(w)
        print()

    if not errors and not warnings:
        print("\nAll checks passed.")

    if errors:
        print("Fix errors above and re-run.")
        return 1

    if warnings:
        print("Review warnings above before deploying.")
        return 0  # Don't fail on warnings

    return 0


if __name__ == "__main__":
    sys.exit(main())
