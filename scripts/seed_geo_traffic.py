#!/usr/bin/env python3
"""
Seed Geo Traffic for WAF Dashboard

Inserts sample TrafficLog and Threat records with country_code set so the
Geo Rules page (globe, bar chart, Top Threat Countries) displays data when
the GeoIP database is not available.

Run from project root:
  python scripts/seed_geo_traffic.py
  python scripts/seed_geo_traffic.py 50   # 50 traffic + 15 threat records (default)
  python scripts/seed_geo_traffic.py 100 20  # 100 traffic, 20 threats
"""
import sys
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

# Country codes for seed data (country_code only - IP is synthetic)
COUNTRIES = ["US", "IN", "CN", "RU", "DE", "BR", "GB", "FR", "JP", "KR", "CA", "AU", "NG", "PL"]

ENDPOINTS = ["/api/products", "/login", "/search", "/admin", "/cart", "/checkout"]
METHODS = ["GET", "POST", "GET", "GET", "POST"]
THREAT_TYPES = ["SQL Injection", "XSS", "Command Injection", "Path Traversal", "NoSQL Injection"]


def seed_geo_traffic(traffic_count: int = 50, threat_count: int = 15):
    """Insert sample traffic and threat records with country attribution."""
    from backend.database import SessionLocal
    from backend.models.threats import Threat, ThreatSeverity
    from backend.models.traffic import TrafficLog

    db = SessionLocal()
    try:

        now = datetime.now(timezone.utc)
        # Spread records over last 24 hours
        base_time = now - timedelta(hours=23)

        print(f"Seeding {traffic_count} traffic logs and {threat_count} threats with country attribution...")

        # Create traffic logs
        for i in range(traffic_count):
            country_code = random.choice(COUNTRIES)
            ip = f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
            ts = base_time + timedelta(
                hours=random.uniform(0, 23),
                minutes=random.uniform(0, 59),
            )
            was_blocked = random.random() < 0.15

            log = TrafficLog(
                ip=ip,
                method=random.choice(METHODS),
                endpoint=random.choice(ENDPOINTS),
                status_code=403 if was_blocked else 200,
                response_size=random.randint(100, 5000),
                user_agent="Mozilla/5.0 (seed)",
                query_string=None,
                request_body=None,
                processing_time_ms=random.randint(5, 50),
                was_blocked=1 if was_blocked else 0,
                anomaly_score=str(round(random.uniform(0.3, 0.9), 4)) if was_blocked else None,
                country_code=country_code,
                threat_type=random.choice(THREAT_TYPES) if was_blocked else None,
                timestamp=ts,
            )
            db.add(log)

        db.commit()

        # Create threats (with country_code)
        base_time_t = now - timedelta(hours=22)
        for i in range(threat_count):
            country_code = random.choice(COUNTRIES)
            ip = f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
            ts = base_time_t + timedelta(
                hours=random.uniform(0, 21),
                minutes=random.uniform(0, 59),
            )
            severity = random.choice(
                [ThreatSeverity.CRITICAL, ThreatSeverity.HIGH, ThreatSeverity.MEDIUM, ThreatSeverity.LOW]
            )
            threat = Threat(
                type=random.choice(THREAT_TYPES),
                severity=severity,
                source_ip=ip,
                endpoint=random.choice(ENDPOINTS),
                method=random.choice(METHODS),
                blocked=True,
                anomaly_score=str(round(random.uniform(0.6, 0.95), 4)),
                details="Seeded for Geo Rules demo",
                payload=f"id=1' OR '1'='1",
                user_agent="Mozilla/5.0 (seed)",
                country_code=country_code,
                processing_time_ms=random.randint(10, 80),
                timestamp=ts,
            )
            db.add(threat)

        db.commit()
        print(f"Done. {traffic_count} traffic logs and {threat_count} threats seeded.")
        print("Geo Rules page at http://localhost:3000/geo-rules should now show globe, charts, and top threat countries.")

    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    traffic_count = 50
    threat_count = 15
    if len(sys.argv) > 1:
        try:
            traffic_count = int(sys.argv[1])
        except ValueError:
            pass
    if len(sys.argv) > 2:
        try:
            threat_count = int(sys.argv[2])
        except ValueError:
            pass

    print(f"Usage: python scripts/seed_geo_traffic.py [traffic_count] [threat_count]")
    print(f"Seeding with traffic_count={traffic_count}, threat_count={threat_count}\n")

    seed_geo_traffic(traffic_count=traffic_count, threat_count=threat_count)
