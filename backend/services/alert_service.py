"""
Alert Service
"""
import json
from datetime import datetime
from typing import List

from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.lib.datetime_utils import utc_now
from backend.models.alerts import Alert, AlertType, AlertSeverity


class AlertService:
    """Service for managing alerts"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_active_alerts(self, org_id: int) -> List[Alert]:
        """Get active alerts (active and not dismissed) for organization."""
        return self.db.query(Alert)\
            .filter(Alert.org_id == org_id)\
            .filter(Alert.is_active)\
            .filter(~Alert.is_dismissed)\
            .order_by(desc(Alert.timestamp))\
            .all()
    
    def get_alert_history(self, org_id: int, start_time: datetime) -> List[Alert]:
        """Get alert history for organization"""
        return self.db.query(Alert)\
            .filter(Alert.org_id == org_id)\
            .filter(Alert.timestamp >= start_time)\
            .order_by(desc(Alert.timestamp))\
            .all()
    
    def create_alert(
        self,
        org_id: int,
        type: AlertType,
        severity: AlertSeverity,
        title: str,
        description: str,
        source: str = "waf",
        icon: str = "alert-circle",
        related_ip: str = None,
        related_endpoint: str = None,
        related_threat_id: int = None,
        actions: List[str] = None
    ) -> Alert:
        """Create a new alert for organization"""
        alert = Alert(
            org_id=org_id,
            type=type,
            severity=severity,
            title=title,
            description=description,
            source=source,
            icon=icon,
            related_ip=related_ip,
            related_endpoint=related_endpoint,
            related_threat_id=related_threat_id,
            actions=json.dumps(actions) if actions else None,
            is_active=True,
            is_acknowledged=False,
            is_dismissed=False
        )
        self.db.add(alert)
        self.db.commit()
        self.db.refresh(alert)
        return alert
    
    def dismiss_alert(self, org_id: int, alert_id: int) -> bool:
        """Dismiss an alert for organization"""
        alert = self.db.query(Alert).filter(
            Alert.id == alert_id,
            Alert.org_id == org_id
        ).first()
        if alert:
            alert.is_dismissed = True
            alert.is_active = False
            alert.dismissed_at = utc_now()
            self.db.commit()
            return True
        return False
    
    def acknowledge_alert(self, org_id: int, alert_id: int) -> bool:
        """Acknowledge an alert for organization"""
        alert = self.db.query(Alert).filter(
            Alert.id == alert_id,
            Alert.org_id == org_id
        ).first()
        if alert:
            alert.is_acknowledged = True
            alert.acknowledged_at = utc_now()
            self.db.commit()
            return True
        return False
