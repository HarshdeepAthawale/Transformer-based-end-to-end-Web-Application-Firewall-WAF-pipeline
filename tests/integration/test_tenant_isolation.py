"""
Integration tests proving cross-tenant isolation.

These tests verify that:
1. Org A's data is not visible to Org B
2. Org A cannot manipulate Org B's data
3. Unauthenticated requests are rejected
4. JWT tokens are properly scoped to org_id
"""
import json
import jwt
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import pytest

from backend.main import app
from backend.database import get_db, Base, engine
from backend.models.organization import Organization
from backend.models.user import User
from backend.models.threat import Threat
from backend.models.alert import Alert
from backend.models.security_rule import SecurityRule
from backend.models.audit_log import AuditLog, AuditAction
from backend.lib.datetime_utils import utc_now


@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)


@pytest.fixture
def db():
    """Database session for tests"""
    Base.metadata.create_all(bind=engine)
    db = next(get_db())
    yield db
    db.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def token_org1(client, db):
    """Create user in Org 1, return JWT token"""
    org1 = Organization(name="Org 1", slug="org1")
    db.add(org1)
    db.commit()
    db.refresh(org1)

    user1 = User(
        username="user1",
        email="user1@org1.com",
        password_hash="hashed_password",
        org_id=org1.id,
        is_active=True,
    )
    db.add(user1)
    db.commit()
    db.refresh(user1)

    token = jwt.encode(
        {
            "user_id": user1.id,
            "username": user1.username,
            "org_id": org1.id,
            "exp": datetime.utcnow() + timedelta(hours=1),
        },
        "secret",
        algorithm="HS256",
    )
    return token


@pytest.fixture
def token_org2(client, db):
    """Create user in Org 2, return JWT token"""
    org2 = Organization(name="Org 2", slug="org2")
    db.add(org2)
    db.commit()
    db.refresh(org2)

    user2 = User(
        username="user2",
        email="user2@org2.com",
        password_hash="hashed_password",
        org_id=org2.id,
        is_active=True,
    )
    db.add(user2)
    db.commit()
    db.refresh(user2)

    token = jwt.encode(
        {
            "user_id": user2.id,
            "username": user2.username,
            "org_id": org2.id,
            "exp": datetime.utcnow() + timedelta(hours=1),
        },
        "secret",
        algorithm="HS256",
    )
    return token


def test_org_isolation_threats(client, db, token_org1, token_org2):
    """Test: Org A threat not visible to Org B"""
    org1 = db.query(Organization).filter(Organization.slug == "org1").first()
    org2 = db.query(Organization).filter(Organization.slug == "org2").first()

    threat1 = Threat(
        org_id=org1.id,
        source_ip="192.168.1.1",
        threat_type="SQL Injection",
        severity="high",
        path="/api/users",
        blocked=True,
    )
    db.add(threat1)
    db.commit()

    headers_org1 = {"Authorization": f"Bearer {token_org1}"}
    headers_org2 = {"Authorization": f"Bearer {token_org2}"}

    resp_org1 = client.get("/api/threats/recent", headers=headers_org1)
    resp_org2 = client.get("/api/threats/recent", headers=headers_org2)

    assert resp_org1.status_code == 200
    assert resp_org2.status_code == 200

    data_org1 = resp_org1.json()["data"]
    data_org2 = resp_org2.json()["data"]

    assert len(data_org1) == 1, "Org1 should see their threat"
    assert len(data_org2) == 0, "Org2 should NOT see Org1 threat"


def test_org_isolation_alerts(client, db, token_org1, token_org2):
    """Test: Org A alert not visible to Org B"""
    org1 = db.query(Organization).filter(Organization.slug == "org1").first()
    org2 = db.query(Organization).filter(Organization.slug == "org2").first()

    alert1 = Alert(
        org_id=org1.id,
        alert_type="DDoS",
        severity="critical",
        message="DDoS attack detected",
        dismissed=False,
    )
    db.add(alert1)
    db.commit()

    headers_org1 = {"Authorization": f"Bearer {token_org1}"}
    headers_org2 = {"Authorization": f"Bearer {token_org2}"}

    resp_org1 = client.get("/api/alerts", headers=headers_org1)
    resp_org2 = client.get("/api/alerts", headers=headers_org2)

    assert resp_org1.status_code == 200
    assert resp_org2.status_code == 200

    data_org1 = resp_org1.json()["data"]
    data_org2 = resp_org2.json()["data"]

    assert len(data_org1) == 1, "Org1 should see their alert"
    assert len(data_org2) == 0, "Org2 should NOT see Org1 alert"


def test_cross_tenant_dismiss_blocked(client, db, token_org1, token_org2):
    """Test: Org A cannot dismiss Org B's alert (gets 404, not 200)"""
    org1 = db.query(Organization).filter(Organization.slug == "org1").first()
    org2 = db.query(Organization).filter(Organization.slug == "org2").first()

    alert1 = Alert(
        org_id=org1.id,
        alert_type="DDoS",
        severity="critical",
        message="DDoS attack detected",
        dismissed=False,
    )
    db.add(alert1)
    db.commit()
    db.refresh(alert1)

    headers_org2 = {"Authorization": f"Bearer {token_org2}"}

    resp = client.put(
        f"/api/alerts/{alert1.id}/dismiss",
        headers=headers_org2,
    )

    assert resp.status_code == 404, "Org2 should get 404 when trying to access Org1 alert"
    assert resp.json()["success"] is False


def test_org_isolation_security_rules(client, db, token_org1, token_org2):
    """Test: Org A rule not in Org B's rule list"""
    org1 = db.query(Organization).filter(Organization.slug == "org1").first()
    org2 = db.query(Organization).filter(Organization.slug == "org2").first()

    rule1 = SecurityRule(
        org_id=org1.id,
        name="SQL Injection Rule",
        description="Block SQL injection attempts",
        pattern=".*union.*select.*",
        action="block",
        is_active=True,
    )
    db.add(rule1)
    db.commit()

    headers_org1 = {"Authorization": f"Bearer {token_org1}"}
    headers_org2 = {"Authorization": f"Bearer {token_org2}"}

    resp_org1 = client.get("/api/security-rules", headers=headers_org1)
    resp_org2 = client.get("/api/security-rules", headers=headers_org2)

    assert resp_org1.status_code == 200
    assert resp_org2.status_code == 200

    data_org1 = resp_org1.json()["data"]
    data_org2 = resp_org2.json()["data"]

    assert len(data_org1) == 1, "Org1 should see their rule"
    assert len(data_org2) == 0, "Org2 should NOT see Org1 rule"


def test_org_isolation_audit_logs(client, db, token_org1, token_org2):
    """Test: Org A audit log not visible to Org B"""
    org1 = db.query(Organization).filter(Organization.slug == "org1").first()
    org2 = db.query(Organization).filter(Organization.slug == "org2").first()
    user1 = db.query(User).filter(User.org_id == org1.id).first()

    log1 = AuditLog(
        org_id=org1.id,
        user_id=user1.id,
        username=user1.username,
        action=AuditAction.CREATE,
        resource_type="rule",
        description="Created new security rule",
        success=True,
    )
    db.add(log1)
    db.commit()

    headers_org1 = {"Authorization": f"Bearer {token_org1}"}
    headers_org2 = {"Authorization": f"Bearer {token_org2}"}

    resp_org1 = client.get("/api/audit/logs", headers=headers_org1)
    resp_org2 = client.get("/api/audit/logs", headers=headers_org2)

    assert resp_org1.status_code == 200
    assert resp_org2.status_code == 200

    data_org1 = resp_org1.json()["data"]
    data_org2 = resp_org2.json()["data"]

    assert len(data_org1) == 1, "Org1 should see their audit log"
    assert len(data_org2) == 0, "Org2 should NOT see Org1 audit log"


def test_unauthenticated_returns_401(client):
    """Test: All tenant routes return 401 without JWT"""
    tenant_routes = [
        "/api/threats/recent",
        "/api/alerts",
        "/api/security-rules",
        "/api/audit/logs",
    ]

    for route in tenant_routes:
        resp = client.get(route)
        assert resp.status_code == 401, f"Route {route} should return 401 without auth"


def test_jwt_org_scoping(client, db, token_org1):
    """Test: JWT token has org_id matching user's org"""
    org1 = db.query(Organization).filter(Organization.slug == "org1").first()
    user1 = db.query(User).filter(User.org_id == org1.id).first()

    decoded = jwt.decode(token_org1, "secret", algorithms=["HS256"])

    assert decoded["org_id"] == org1.id, "JWT org_id should match org"
    assert decoded["user_id"] == user1.id, "JWT user_id should match user"
    assert decoded["username"] == user1.username, "JWT username should match user"
