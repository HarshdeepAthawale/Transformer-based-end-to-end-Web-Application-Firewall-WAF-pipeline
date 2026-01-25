"""
Geo-fencing Service
"""
from sqlalchemy.orm import Session
from typing import Dict, Optional, List
from datetime import datetime
from loguru import logger

from backend.models.geo_rules import GeoRule, GeoRuleType
try:
    from backend.services.geoip_lookup import GeoIPLookupService
except ImportError:
    GeoIPLookupService = None


class GeoFencingService:
    """Service for geo-fencing"""
    
    def __init__(self, db: Session):
        self.db = db
        self.geoip = GeoIPLookupService() if GeoIPLookupService else None
    
    def check_country(self, ip: str) -> Dict:
        """
        Check if IP's country is allowed
        Returns: {
            'allowed': bool,
            'reason': str,
            'country_code': str,
            'country_name': str
        }
        """
        # Get country from IP
        geo_data = None
        if self.geoip:
            geo_data = self.geoip.lookup(ip)
        
        if not geo_data:
            # If lookup fails, allow by default (fail open)
            return {
                'allowed': True,
                'reason': 'GeoIP lookup failed',
                'country_code': None,
                'country_name': None
            }
        
        country_code = geo_data.get('country_code')
        country_name = geo_data.get('country_name')
        
        # Get active geo rules
        rules = self.db.query(GeoRule)\
            .filter(GeoRule.is_active == True)\
            .order_by(GeoRule.priority.desc())\
            .all()
        
        if not rules:
            # No rules = allow all
            return {
                'allowed': True,
                'reason': None,
                'country_code': country_code,
                'country_name': country_name
            }
        
        # Check exception IPs first
        for rule in rules:
            if rule.exception_ips:
                import json
                exception_ips = json.loads(rule.exception_ips) if isinstance(rule.exception_ips, str) else rule.exception_ips
                if ip in exception_ips:
                    return {
                        'allowed': True,
                        'reason': 'IP in exception list',
                        'country_code': country_code,
                        'country_name': country_name
                    }
        
        # Check allow list rules
        allow_rules = [r for r in rules if r.rule_type == GeoRuleType.ALLOW]
        if allow_rules:
            # If allow list exists, only allow listed countries
            allowed_countries = {r.country_code for r in allow_rules}
            if country_code not in allowed_countries:
                return {
                    'allowed': False,
                    'reason': f'Country {country_name} not in allow list',
                    'country_code': country_code,
                    'country_name': country_name
                }
        
        # Check deny list rules
        deny_rules = [r for r in rules if r.rule_type == GeoRuleType.DENY]
        for rule in deny_rules:
            if rule.country_code == country_code:
                # Update statistics
                rule.blocked_requests += 1
                rule.last_applied = datetime.utcnow()
                self.db.commit()
                
                return {
                    'allowed': False,
                    'reason': f'Country {country_name} is in deny list',
                    'country_code': country_code,
                    'country_name': country_name
                }
        
        return {
            'allowed': True,
            'reason': None,
            'country_code': country_code,
            'country_name': country_name
        }
    
    def create_rule(
        self,
        rule_type: GeoRuleType,
        country_code: str,
        country_name: str,
        priority: int = 0,
        exception_ips: List[str] = None,
        reason: str = None,
        created_by: str = None
    ) -> GeoRule:
        """Create a geo rule"""
        import json
        
        rule = GeoRule(
            rule_type=rule_type,
            country_code=country_code,
            country_name=country_name,
            priority=priority,
            exception_ips=json.dumps(exception_ips) if exception_ips else None,
            reason=reason,
            created_by=created_by
        )
        
        self.db.add(rule)
        self.db.commit()
        self.db.refresh(rule)
        return rule
    
    def get_rules(self, active_only: bool = True) -> List[GeoRule]:
        """Get geo rules"""
        query = self.db.query(GeoRule)
        if active_only:
            query = query.filter(GeoRule.is_active == True)
        return query.order_by(GeoRule.priority.desc(), GeoRule.timestamp.desc()).all()
    
    def get_geographic_stats(self, start_time: datetime) -> List[Dict]:
        """Get geographic threat statistics"""
        from backend.models.traffic import TrafficLog
        from backend.models.threats import Threat
        from sqlalchemy import func
        
        # Get traffic stats by country
        traffic_stats = self.db.query(
            TrafficLog.country_code,
            func.count(TrafficLog.id).label('total_requests'),
            func.sum(TrafficLog.was_blocked).label('blocked_requests')
        )\
        .filter(TrafficLog.timestamp >= start_time)\
        .filter(TrafficLog.country_code.isnot(None))\
        .group_by(TrafficLog.country_code)\
        .all()
        
        # Get threat counts by country
        threat_stats = self.db.query(
            Threat.source_ip,
            func.count(Threat.id).label('threat_count')
        )\
        .filter(Threat.timestamp >= start_time)\
        .group_by(Threat.source_ip)\
        .all()
        
        # Map IPs to countries (simplified - in production, use GeoIP lookup)
        # For now, we'll aggregate by country_code from traffic logs
        country_threat_map = {}
        for source_ip, threat_count in threat_stats:
            # Try to get country from traffic logs
            traffic_log = self.db.query(TrafficLog)\
                .filter(TrafficLog.source_ip == source_ip)\
                .filter(TrafficLog.timestamp >= start_time)\
                .filter(TrafficLog.country_code.isnot(None))\
                .first()
            if traffic_log and traffic_log.country_code:
                country_code = traffic_log.country_code
                country_threat_map[country_code] = country_threat_map.get(country_code, 0) + threat_count
        
        # Build result list
        result = []
        country_names = {}  # Cache country names from rules
        
        for country_code, total_requests, blocked_requests in traffic_stats:
            # Get country name from rules or use code as fallback
            if country_code not in country_names:
                rule = self.db.query(GeoRule)\
                    .filter(GeoRule.country_code == country_code)\
                    .first()
                country_names[country_code] = rule.country_name if rule else country_code
            
            result.append({
                'country_code': country_code,
                'country_name': country_names[country_code],
                'total_requests': int(total_requests),
                'blocked_requests': int(blocked_requests or 0),
                'threat_count': int(country_threat_map.get(country_code, 0))
            })
        
        return result
