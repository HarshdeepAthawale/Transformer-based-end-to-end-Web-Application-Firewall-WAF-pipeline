"""
Alert Service
"""
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime
from typing import List
import json

from src.api.models.alerts import Alert, AlertType, AlertSeverity


class AlertService:
    """Service for managing alerts"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts"""
        return self.db.query(Alert)\
            .filter(Alert.is_active == True)\
            .filter(Alert.is_dismissed == False)\
            .order_by(desc(Alert.timestamp))\
            .all()
    
    def get_alert_history(self, start_time: datetime) -> List[Alert]:
        """Get alert history"""
        return self.db.query(Alert)\
            .filter(Alert.timestamp >= start_time)\
            .order_by(desc(Alert.timestamp))\
            .all()
    
    def create_alert(
        self,
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
        """Create a new alert"""
        alert = Alert(
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
    
    def dismiss_alert(self, alert_id: int) -> bool:
        """Dismiss an alert"""
        alert = self.db.query(Alert).filter(Alert.id == alert_id).first()
        if alert:
            alert.is_dismissed = True
            alert.is_active = False
            alert.dismissed_at = datetime.utcnow()
            self.db.commit()
            return True
        return False
    
    def acknowledge_alert(self, alert_id: int) -> bool:
        """Acknowledge an alert"""
        alert = self.db.query(Alert).filter(Alert.id == alert_id).first()
        if alert:
            alert.is_acknowledged = True
            alert.acknowledged_at = datetime.utcnow()
            self.db.commit()
            return True
        return False
