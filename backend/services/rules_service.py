"""
Security Rules Service
"""
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
from datetime import datetime
import re
import json
from loguru import logger

from backend.models.security_rules import SecurityRule, RulePriority, RuleAction


class RulesService:
    """Service for security rules engine"""
    
    def __init__(self, db: Session):
        self.db = db
        self._rules_cache = None
        self._load_rules()
    
    def _load_rules(self):
        """Load active rules into cache"""
        self._rules_cache = self.db.query(SecurityRule)\
            .filter(SecurityRule.is_active == True)\
            .order_by(
                SecurityRule.priority.desc(),
                SecurityRule.timestamp.desc()
            )\
            .all()
        logger.info(f"Loaded {len(self._rules_cache)} active security rules")
    
    def check_rules(
        self,
        method: str,
        path: str,
        headers: Dict,
        query_params: Dict = None,
        body: str = None
    ) -> Dict:
        """
        Check request against security rules
        Returns: {
            'matched': bool,
            'rule_id': int,
            'rule_name': str,
            'action': str,
            'priority': str
        }
        """
        if not self._rules_cache:
            return {
                'matched': False,
                'rule_id': None,
                'rule_name': None,
                'action': None,
                'priority': None
            }
        
        # Check rules in priority order
        for rule in self._rules_cache:
            if self._rule_matches(rule, method, path, headers, query_params, body):
                # Update statistics
                rule.match_count += 1
                rule.last_matched = datetime.utcnow()
                self.db.commit()
                
                return {
                    'matched': True,
                    'rule_id': rule.id,
                    'rule_name': rule.name,
                    'action': rule.action.value if rule.action else 'block',
                    'priority': rule.priority.value if rule.priority else 'medium'
                }
        
        return {
            'matched': False,
            'rule_id': None,
            'rule_name': None,
            'action': None,
            'priority': None
        }
    
    def _rule_matches(
        self,
        rule: SecurityRule,
        method: str,
        path: str,
        headers: Dict,
        query_params: Dict,
        body: str
    ) -> bool:
        """Check if rule matches request"""
        # Parse match conditions
        conditions = {}
        if rule.match_conditions:
            try:
                conditions = json.loads(rule.match_conditions) if isinstance(rule.match_conditions, str) else rule.match_conditions
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in rule {rule.id} match_conditions")
                return False
        
        # Check applies_to scope
        text_to_check = ""
        if rule.applies_to == "all":
            text_to_check = f"{method} {path} {str(query_params)} {str(headers)} {body or ''}"
        elif rule.applies_to == "path":
            text_to_check = path
        elif rule.applies_to == "query":
            text_to_check = str(query_params)
        elif rule.applies_to == "headers":
            text_to_check = str(headers)
        elif rule.applies_to == "body":
            text_to_check = body or ""
        
        # Check pattern
        if rule.pattern:
            try:
                if re.search(rule.pattern, text_to_check, re.IGNORECASE):
                    return True
            except re.error:
                logger.warning(f"Invalid regex pattern in rule {rule.id}: {rule.pattern}")
                return False
        
        # Check additional conditions
        if conditions:
            # Method condition
            if 'method' in conditions:
                if method.upper() not in [m.upper() for m in conditions['method']]:
                    return False
            
            # Path condition
            if 'path' in conditions:
                path_pattern = conditions['path']
                if not re.search(path_pattern, path, re.IGNORECASE):
                    return False
        
        return False
    
    def create_rule(
        self,
        name: str,
        rule_type: str,
        pattern: str,
        applies_to: str = "all",
        action: RuleAction = RuleAction.BLOCK,
        priority: RulePriority = RulePriority.MEDIUM,
        description: str = None,
        owasp_category: str = None,
        created_by: str = None
    ) -> SecurityRule:
        """Create security rule"""
        rule = SecurityRule(
            name=name,
            description=description,
            rule_type=rule_type,
            pattern=pattern,
            applies_to=applies_to,
            action=action,
            priority=priority,
            owasp_category=owasp_category,
            created_by=created_by
        )
        
        self.db.add(rule)
        self.db.commit()
        self.db.refresh(rule)
        
        # Reload cache
        self._load_rules()
        
        return rule
    
    def get_rules(self, active_only: bool = True) -> List[SecurityRule]:
        """Get security rules"""
        query = self.db.query(SecurityRule)
        if active_only:
            query = query.filter(SecurityRule.is_active == True)
        return query.order_by(SecurityRule.priority.desc(), SecurityRule.timestamp.desc()).all()
    
    def get_owasp_rules(self) -> List[SecurityRule]:
        """Get OWASP Top 10 rules"""
        return self.db.query(SecurityRule)\
            .filter(SecurityRule.owasp_category.isnot(None))\
            .filter(SecurityRule.is_active == True)\
            .order_by(SecurityRule.owasp_category)\
            .all()
