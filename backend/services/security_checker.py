"""
Unified Security Checker - Integrates all security features
"""
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from loguru import logger
from sqlalchemy.orm import Session

from backend.services.ip_fencing import IPFencingService
try:
    from backend.services.geo_fencing import GeoFencingService
    from backend.services.bot_detection import BotDetectionService
    from backend.services.threat_intel_service import ThreatIntelService
    from backend.services.rules_service import RulesService
    from backend.services.advanced_rate_limiting import AdvancedRateLimiter
    SERVICES_AVAILABLE = True
except ImportError as e:
    SERVICES_AVAILABLE = False
    logger.warning(f"Some security services not available: {e}")


class SecurityChecker:
    """Unified security checker that integrates all security features"""
    
    def __init__(self, db: Session):
        self.db = db
        self.ip_fencing = IPFencingService(db)
        
        if SERVICES_AVAILABLE:
            self.geo_fencing = GeoFencingService(db)
            self.bot_detection = BotDetectionService(db)
            self.threat_intel = ThreatIntelService(db)
            self.rules_engine = RulesService(db)
            self.rate_limiter = AdvancedRateLimiter()
        else:
            self.geo_fencing = None
            self.bot_detection = None
            self.threat_intel = None
            self.rules_engine = None
            self.rate_limiter = None
    
    def check_request(
        self,
        ip: str,
        method: str,
        path: str,
        headers: Dict,
        query_params: Dict = None,
        body: str = None,
        user_agent: str = None
    ) -> Dict:
        """
        Comprehensive security check for a request
        Returns: {
            'allowed': bool,
            'blocked': bool,
            'reason': str,
            'checks': {
                'ip_fencing': {...},
                'geo_fencing': {...},
                'bot_detection': {...},
                'threat_intel': {...},
                'rate_limiting': {...},
                'security_rules': {...}
            }
        }
        """
        result = {
            'allowed': True,
            'blocked': False,
            'reason': None,
            'checks': {}
        }
        
        # 1. Check IP Whitelist (bypasses all other checks)
        if self.ip_fencing.is_ip_whitelisted(ip):
            result['checks']['ip_fencing'] = {'status': 'whitelisted', 'bypass': True}
            return result
        
        # 2. Check IP Blacklist
        is_blocked, reason = self.ip_fencing.is_ip_blocked(ip)
        if is_blocked:
            result['blocked'] = True
            result['allowed'] = False
            result['reason'] = f"IP blocked: {reason}"
            result['checks']['ip_fencing'] = {'status': 'blocked', 'reason': reason}
            return result
        
        # 3. Check Geo-fencing
        if self.geo_fencing:
            geo_result = self.geo_fencing.check_country(ip)
            if not geo_result['allowed']:
                result['blocked'] = True
                result['allowed'] = False
                result['reason'] = geo_result['reason']
                result['checks']['geo_fencing'] = geo_result
                return result
        else:
            geo_result = {'allowed': True, 'reason': 'Geo-fencing not available'}
        
        # 4. Check Rate Limiting
        if self.rate_limiter:
            rate_limit_result = self.rate_limiter.check_rate_limit(ip, path)
            if not rate_limit_result['allowed']:
                result['blocked'] = True
                result['allowed'] = False
                result['reason'] = "Rate limit exceeded"
                result['checks']['rate_limiting'] = rate_limit_result
                return result
        else:
            rate_limit_result = {'allowed': True}
        
        # 5. Check Bot Detection
        if self.bot_detection:
            bot_result = self.bot_detection.detect_bot(user_agent or "", ip, headers)
            if bot_result['is_bot'] and not bot_result['is_whitelisted']:
                if bot_result['action'] == 'block':
                    result['blocked'] = True
                    result['allowed'] = False
                    result['reason'] = f"Bot detected: {bot_result['bot_name']}"
                    result['checks']['bot_detection'] = bot_result
                    return result
                elif bot_result['action'] == 'challenge':
                    result['checks']['bot_detection'] = bot_result
        else:
            bot_result = {'is_bot': False}
        
        # 6. Check Threat Intelligence
        if self.threat_intel:
            threat_intel_result = self.threat_intel.check_threat(ip, path, query_params, body)
            if threat_intel_result['is_threat']:
                result['blocked'] = True
                result['allowed'] = False
                result['reason'] = f"Threat detected: {threat_intel_result['threat_type']}"
                result['checks']['threat_intel'] = threat_intel_result
                return result
        else:
            threat_intel_result = {'is_threat': False}
        
        # 7. Check Security Rules
        if self.rules_engine:
            rules_result = self.rules_engine.check_rules(method, path, headers, query_params, body)
            if rules_result['matched']:
                if rules_result['action'] == 'block':
                    result['blocked'] = True
                    result['allowed'] = False
                    result['reason'] = f"Security rule matched: {rules_result['rule_name']}"
                    result['checks']['security_rules'] = rules_result
                    return result
                elif rules_result['action'] == 'alert':
                    result['checks']['security_rules'] = rules_result
        else:
            rules_result = {'matched': False}
        
        # All checks passed
        result['checks']['ip_fencing'] = {'status': 'allowed'}
        result['checks']['geo_fencing'] = geo_result
        result['checks']['rate_limiting'] = rate_limit_result
        if bot_result.get('is_bot'):
            result['checks']['bot_detection'] = bot_result
        if threat_intel_result.get('is_threat'):
            result['checks']['threat_intel'] = threat_intel_result
        
        return result
