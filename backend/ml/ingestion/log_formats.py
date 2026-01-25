"""
Log Format Detection Module

Detects and validates various web server log formats (Apache, Nginx)
"""
from enum import Enum
from typing import Optional
import re
from loguru import logger


class LogFormat(Enum):
    """Supported log formats"""
    APACHE_COMMON = "apache_common"
    APACHE_COMBINED = "apache_combined"
    APACHE_DETAILED = "apache_detailed"
    NGINX_COMBINED = "nginx_combined"
    NGINX_DETAILED = "nginx_detailed"
    UNKNOWN = "unknown"


class LogFormatDetector:
    """Detect log format from sample lines"""
    
    # Apache Common Log Format: IP user auth [timestamp] "method path protocol" status size
    APACHE_COMMON_PATTERN = re.compile(
        r'^(\S+) (\S+) (\S+) \[([^\]]+)\] "(\S+) ([^"]+)" (\d+)'
    )
    
    # Apache Combined Log Format: Common + referer + user-agent
    APACHE_COMBINED_PATTERN = re.compile(
        r'^(\S+) (\S+) (\S+) \[([^\]]+)\] "(\S+) ([^"]+)" (\d+) (\d+|-) "([^"]*)" "([^"]*)"'
    )
    
    # Apache Detailed Format (with additional fields)
    APACHE_DETAILED_PATTERN = re.compile(
        r'^(\S+) (\S+) (\S+) \[([^\]]+)\] "(\S+) ([^"]+)" (\d+) (\d+|-) '
        r'"([^"]*)" "([^"]*)" "([^"]*)" "([^"]*)" "([^"]*)" "([^"]*)"'
    )
    
    # Nginx Combined Format: IP - user [timestamp] "method path protocol" status size "referer" "user-agent"
    NGINX_COMBINED_PATTERN = re.compile(
        r'^(\S+) - (\S+) \[([^\]]+)\] "(\S+) ([^"]+)" (\d+) (\d+|-) "([^"]*)" "([^"]*)"'
    )
    
    # Nginx Detailed Format (with additional fields)
    NGINX_DETAILED_PATTERN = re.compile(
        r'^(\S+) - (\S+) \[([^\]]+)\] "(\S+) ([^"]+)" (\d+) (\d+|-) '
        r'"([^"]*)" "([^"]*)" "([^"]*)" "([^"]*)" "([^"]*)"'
    )
    
    @staticmethod
    def detect_format(log_line: str) -> LogFormat:
        """Detect log format from a sample line"""
        if not log_line or not log_line.strip():
            return LogFormat.UNKNOWN
        
        line = log_line.strip()
        
        # Check Nginx patterns first (they have distinct "- -" pattern)
        # Nginx Detailed
        if LogFormatDetector.NGINX_DETAILED_PATTERN.match(line):
            return LogFormat.NGINX_DETAILED
        
        # Nginx Combined (check before Apache as Nginx uses "- -" pattern)
        if LogFormatDetector.NGINX_COMBINED_PATTERN.match(line):
            return LogFormat.NGINX_COMBINED
        
        # Check Apache Detailed (most specific first)
        if LogFormatDetector.APACHE_DETAILED_PATTERN.match(line):
            return LogFormat.APACHE_DETAILED
        
        # Check Apache Combined
        if LogFormatDetector.APACHE_COMBINED_PATTERN.match(line):
            return LogFormat.APACHE_COMBINED
        
        # Check Apache Common
        if LogFormatDetector.APACHE_COMMON_PATTERN.match(line):
            return LogFormat.APACHE_COMMON
        
        return LogFormat.UNKNOWN
    
    @staticmethod
    def validate_format(log_file: str, format_type: LogFormat) -> bool:
        """Validate if log file matches expected format"""
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                checked = 0
                for i, line in enumerate(f):
                    if i >= 10:  # Check first 10 lines
                        break
                    if not line.strip():
                        continue
                    detected = LogFormatDetector.detect_format(line)
                    checked += 1
                    if detected != format_type and detected != LogFormat.UNKNOWN:
                        logger.warning(f"Line {i+1} doesn't match expected format. Expected: {format_type}, Got: {detected}")
                        return False
                return checked > 0
        except Exception as e:
            logger.error(f"Error validating format: {e}")
            return False
    
    @staticmethod
    def detect_from_file(log_file: str, sample_lines: int = 10) -> LogFormat:
        """Detect format by analyzing multiple lines from file"""
        formats = []
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= sample_lines:
                        break
                    if line.strip():
                        detected = LogFormatDetector.detect_format(line)
                        if detected != LogFormat.UNKNOWN:
                            formats.append(detected)
            
            if not formats:
                return LogFormat.UNKNOWN
            
            # Return most common format
            from collections import Counter
            format_counts = Counter(formats)
            return format_counts.most_common(1)[0][0]
        except Exception as e:
            logger.error(f"Error detecting format from file: {e}")
            return LogFormat.UNKNOWN
