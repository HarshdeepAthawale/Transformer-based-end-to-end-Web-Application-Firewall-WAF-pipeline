"""Phase wrappers: thin adapters around existing gateway modules."""

from gateway.pipeline.phases.ip_blacklist import IPBlacklistPhase
from gateway.pipeline.phases.edge_cache import EdgeCachePhase
from gateway.pipeline.phases.rate_limit import RateLimitPhase
from gateway.pipeline.phases.ddos_protection import DDoSProtectionPhase
from gateway.pipeline.phases.bot_detection import BotDetectionPhase
from gateway.pipeline.phases.upload_scan import UploadScanPhase
from gateway.pipeline.phases.firewall_ai import FirewallAIPhase
from gateway.pipeline.phases.credential_leak import CredentialLeakPhase
from gateway.pipeline.phases.managed_rules import ManagedRulesPhase
from gateway.pipeline.phases.waf_ml import WAFMLPhase

__all__ = [
    "IPBlacklistPhase",
    "EdgeCachePhase",
    "RateLimitPhase",
    "DDoSProtectionPhase",
    "BotDetectionPhase",
    "UploadScanPhase",
    "FirewallAIPhase",
    "CredentialLeakPhase",
    "ManagedRulesPhase",
    "WAFMLPhase",
]
