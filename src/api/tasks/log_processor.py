"""
Log Processor Background Worker
"""
import asyncio
import threading
from datetime import datetime
from loguru import logger
from sqlalchemy.orm import Session

from src.api.database import SessionLocal
from src.api.services.traffic_service import TrafficService
from src.api.services.threat_service import ThreatService, ThreatSeverity
from src.api.services.activity_service import ActivityService, ActivityType
from src.api.services.alert_service import AlertService, AlertType, AlertSeverity
from src.api.services.security_checker import SecurityChecker
from src.api.services.websocket_service import broadcast_update_sync
from src.ingestion.ingestion import LogIngestionSystem
from src.parsing.pipeline import ParsingPipeline
from src.integration.waf_service import WAFService
from src.api.config import config
import json


class LogProcessor:
    """Process logs and integrate with WAF service"""
    
    def __init__(self):
        self.running = False
        self.thread = None
        self.waf_service = None
        self.parser = ParsingPipeline()
        self._initialize_waf_service()
    
    def _initialize_waf_service(self):
        """Initialize WAF service"""
        try:
            from pathlib import Path
            model_path = Path("models/checkpoints/best_model.pt")
            vocab_path = Path("models/vocabularies/http_vocab.json")
            
            if model_path.exists() and vocab_path.exists():
                self.waf_service = WAFService(
                    model_path=str(model_path),
                    vocab_path=str(vocab_path),
                    threshold=0.5
                )
                logger.info("WAF service initialized for log processing")
            else:
                logger.warning("Model files not found, log processing will run without ML detection")
        except Exception as e:
            logger.error(f"Failed to initialize WAF service: {e}")
    
    def start(self):
        """Start log processor"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("Log processor started")
    
    def stop(self):
        """Stop log processor"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Log processor stopped")
    
    def _run(self):
        """Main processing loop"""
        if not config.LOG_INGESTION_ENABLED or not config.LOG_PATH:
            logger.info("Log ingestion disabled or no log path configured")
            return
        
        try:
            ingestion_system = LogIngestionSystem()
            
            # Process logs in streaming mode
            for log_line in ingestion_system.ingest_stream(config.LOG_PATH, follow=True):
                if not self.running:
                    break
                
                try:
                    self._process_log_line(log_line)
                except Exception as e:
                    logger.error(f"Error processing log line: {e}")
        
        except Exception as e:
            logger.error(f"Log processor error: {e}")
    
    def _process_log_line(self, log_line: str):
        """Process a single log line"""
        try:
            # Parse log line
            parsed = self.parser.process_log_line(log_line)
            
            if not parsed:
                return
            
            # Extract request details
            method = parsed.get("method", "GET")
            path = parsed.get("path", "/")
            query_params = parsed.get("query_params", {})
            headers = parsed.get("headers", {})
            body = parsed.get("body")
            ip = parsed.get("remote_addr", "unknown")
            status_code = parsed.get("status", 200)
            user_agent = headers.get("User-Agent", "")
            
            # Security checks (IP fencing, geo-fencing, bot detection, etc.)
            db = SessionLocal()
            country_code = None
            try:
                security_checker = SecurityChecker(db)
                security_result = security_checker.check_request(
                    ip=ip,
                    method=method,
                    path=path,
                    headers=headers,
                    query_params=query_params,
                    body=body,
                    user_agent=user_agent
                )
                
                # Extract geo data if available
                if 'geo_fencing' in security_result.get('checks', {}):
                    geo_check = security_result['checks']['geo_fencing']
                    country_code = geo_check.get('country_code')
                
                # If blocked by security checks, skip ML check
                if security_result.get('blocked'):
                    is_anomaly = True
                    anomaly_score = 1.0
                    threat_type = security_result.get('reason', 'Security rule violation')
                else:
                    # Check with WAF service if available
                    anomaly_score = None
                    is_anomaly = False
                    threat_type = None
                    
                    if self.waf_service:
                        try:
                            result = self.waf_service.check_request(
                                method=method,
                                path=path,
                                query_params=query_params,
                                headers=headers,
                                body=body
                            )
                            anomaly_score = result.get("anomaly_score", 0.0)
                            is_anomaly = result.get("is_anomaly", False)
                            
                            # Classify threat type based on patterns
                            if is_anomaly:
                                threat_type = self._classify_threat(path, query_params, body, headers)
                        except Exception as e:
                            logger.error(f"WAF check failed: {e}")
                
                # Update IP reputation stats
                ip_fencing = security_checker.ip_fencing
                ip_fencing.increment_ip_stats(
                    ip=ip,
                    was_blocked=is_anomaly,
                    is_anomaly=bool(anomaly_score and anomaly_score > 0.5),
                    is_threat=bool(is_anomaly and threat_type)
                )
                
                # Update geo data in IP reputation if available
                if country_code:
                    try:
                        ip_fencing.update_ip_reputation(
                            ip=ip,
                            country_code=country_code
                        )
                    except Exception as e:
                        logger.debug(f"Failed to update IP geo data: {e}")
                
            finally:
                db.close()
            
            # Store in database
            db = SessionLocal()
            try:
                traffic_service = TrafficService(db)
                traffic_log = traffic_service.create_traffic_log(
                    ip=ip,
                    method=method,
                    endpoint=path,
                    status_code=status_code,
                    response_size=parsed.get("response_size", 0),
                    user_agent=user_agent,
                    query_string=json.dumps(query_params) if query_params else None,
                    request_body=body[:1000] if body else None,
                    processing_time_ms=int(parsed.get("processing_time", 0) * 1000),
                    was_blocked=is_anomaly,
                    anomaly_score=anomaly_score,
                    threat_type=threat_type,
                    country_code=country_code  # Add geo data
                )
                
                # Create threat if anomaly detected
                if is_anomaly and anomaly_score:
                    threat_service = ThreatService(db)
                    severity = self._get_threat_severity(anomaly_score)
                    threat = threat_service.create_threat(
                        type=threat_type or "Unknown",
                        severity=severity,
                        source_ip=ip,
                        endpoint=path,
                        method=method,
                        blocked=is_anomaly,
                        anomaly_score=anomaly_score,
                        details=f"Anomaly detected with score {anomaly_score:.3f}",
                        payload=body[:500] if body else None,
                        user_agent=user_agent,
                        country_code=country_code  # Add geo data
                    )
                    
                    # Create alert for high severity threats
                    alert = None
                    if severity in [ThreatSeverity.HIGH, ThreatSeverity.CRITICAL]:
                        alert_service = AlertService(db)
                        alert = alert_service.create_alert(
                            type=AlertType.WARNING if severity == ThreatSeverity.HIGH else AlertType.CRITICAL,
                            severity=severity,
                            title=f"{threat_type or 'Threat'} detected",
                            description=f"Anomaly score: {anomaly_score:.3f} from {ip}",
                            source="waf",
                            related_ip=ip,
                            related_endpoint=path,
                            related_threat_id=threat.id,
                            actions=["block_ip", "investigate"]
                        )
                        
                        # Broadcast alert
                        broadcast_update_sync("alert", alert.to_dict())
                    
                    # Broadcast threat
                    broadcast_update_sync("threat", threat.to_dict())
                
                # Create activity
                activity_service = ActivityService(db)
                activity_type = ActivityType.BLOCKED if is_anomaly else ActivityType.ALLOWED
                activity = activity_service.create_activity(
                    type=activity_type,
                    title=f"{method} {path}",
                    details=f"Request from {ip} - {'Blocked' if is_anomaly else 'Allowed'}",
                    ip=ip,
                    endpoint=path,
                    method=method,
                    threat_type=threat_type,
                    anomaly_score=anomaly_score
                )
                
                # Broadcast updates
                broadcast_update_sync("activity", activity.to_dict())
                broadcast_update_sync("traffic", traffic_log.to_dict())
                
            finally:
                db.close()
        
        except Exception as e:
            logger.error(f"Error processing log line: {e}")
    
    def _classify_threat(self, path: str, query_params: dict, body: str, headers: dict) -> str:
        """Classify threat type based on patterns"""
        # Simple pattern-based classification
        combined_text = f"{path} {json.dumps(query_params)} {body or ''}".lower()
        
        sql_patterns = ["union", "select", "insert", "delete", "drop", "exec", "1=1", "' or '"]
        xss_patterns = ["<script", "javascript:", "onerror=", "onclick=", "alert("]
        path_traversal = ["../", "..\\", "/etc/passwd", "boot.ini"]
        
        if any(pattern in combined_text for pattern in sql_patterns):
            return "SQL Injection"
        elif any(pattern in combined_text for pattern in xss_patterns):
            return "XSS"
        elif any(pattern in combined_text for pattern in path_traversal):
            return "Path Traversal"
        else:
            return "Anomaly"
    
    def _get_threat_severity(self, anomaly_score: float) -> ThreatSeverity:
        """Get threat severity from anomaly score"""
        if anomaly_score >= 0.9:
            return ThreatSeverity.CRITICAL
        elif anomaly_score >= 0.7:
            return ThreatSeverity.HIGH
        elif anomaly_score >= 0.5:
            return ThreatSeverity.MEDIUM
        else:
            return ThreatSeverity.LOW
