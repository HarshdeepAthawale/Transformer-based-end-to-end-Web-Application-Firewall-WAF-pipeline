"""
Parsing & Normalization Pipeline

Parse log lines → structured HTTPRequest → normalize dynamic values → serialize
for model input.
"""

from backend.parsing.log_parser import HTTPRequest, parse_log_line, parse_request_dict
from backend.parsing.normalizer import normalize_request
from backend.parsing.serializer import serialize_request
from backend.parsing.pipeline import ParsingPipeline

__all__ = [
    "HTTPRequest",
    "parse_log_line",
    "parse_request_dict",
    "normalize_request",
    "serialize_request",
    "ParsingPipeline",
]
