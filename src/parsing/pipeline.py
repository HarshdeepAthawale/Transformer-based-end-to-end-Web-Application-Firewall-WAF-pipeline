"""
Parsing Pipeline Module

Complete parsing and normalization pipeline
"""
from typing import Iterator, Optional
from .log_parser import LogParser, HTTPRequest
from .normalizer import RequestNormalizer
from .serializer import RequestSerializer
from loguru import logger


class ParsingPipeline:
    """Complete parsing and normalization pipeline"""
    
    def __init__(self, config_path: str = None):
        self.parser = LogParser()
        self.normalizer = RequestNormalizer(config_path)
        self.serializer = RequestSerializer()
    
    def process_log_line(self, log_line: str) -> Optional[str]:
        """Process single log line: parse -> normalize -> serialize"""
        # Parse
        request = self.parser.parse(log_line)
        if not request:
            return None
        
        # Normalize
        normalized = self.normalizer.normalize(request)
        
        # Serialize
        serialized = self.serializer.to_string(normalized, format="compact")
        
        return serialized
    
    def process_batch(self, log_lines: Iterator[str]) -> Iterator[str]:
        """Process batch of log lines"""
        for line in log_lines:
            result = self.process_log_line(line)
            if result:
                yield result
    
    def parse_only(self, log_line: str) -> Optional[HTTPRequest]:
        """Parse log line without normalization"""
        return self.parser.parse(log_line)
    
    def normalize_only(self, request: HTTPRequest) -> HTTPRequest:
        """Normalize request without parsing"""
        return self.normalizer.normalize(request)
