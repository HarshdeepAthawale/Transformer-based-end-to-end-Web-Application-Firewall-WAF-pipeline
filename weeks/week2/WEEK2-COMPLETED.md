# Week 2: Parsing & Normalization — Completed

> **Duration**: Week 2 of Phase 1
> **Goal**: Parse log lines → structured HTTPRequest → normalize dynamic values → serialize for model input
> **Status**: COMPLETE — 6/6 tasks done, 54/54 parsing tests passing (108 total across Week 1+2)

---

## Summary

Built the entire `backend/parsing/` module — 5 Python files providing log parsing into structured `HTTPRequest` objects, normalization of dynamic values (UUIDs, timestamps, IPs, JWTs, session IDs, numeric IDs) into stable placeholders, serialization into the exact format the WAF classifier expects, and a `ParsingPipeline` class that chains everything together with two entry points: `process(log_line)` for batch log processing and `process_request(req)` / `process_dict(data)` for live requests.

---

## Tasks Completed

| # | Task | File | Description |
|---|------|------|-------------|
| 1 | Create parsing module | `backend/parsing/__init__.py` | Package with public exports |
| 2 | Log parser | `backend/parsing/log_parser.py` | Parses Apache Common, Combined, Nginx Combined → `HTTPRequest` dataclass |
| 3 | Normalizer | `backend/parsing/normalizer.py` | Replaces UUIDs, IPs, timestamps, JWTs, session IDs, numeric IDs with `{PLACEHOLDER}` |
| 4 | Serializer | `backend/parsing/serializer.py` | Converts `HTTPRequest` to compact string matching WAFClassifier format |
| 5 | Pipeline | `backend/parsing/pipeline.py` | `ParsingPipeline` chaining parse → normalize → serialize |
| 6 | Unit tests | `tests/unit/test_parsing.py` | 54 tests covering all modules |

---

## Files Created

```
backend/parsing/
├── __init__.py       # Public exports: HTTPRequest, parse_log_line, parse_request_dict,
│                     #   normalize_request, serialize_request, ParsingPipeline
├── log_parser.py     # Apache/Nginx regex parsing → HTTPRequest dataclass
├── normalizer.py     # Dynamic value → placeholder replacement
├── serializer.py     # HTTPRequest → compact model-input string
└── pipeline.py       # ParsingPipeline: process(), process_request(), process_dict()

tests/unit/
└── test_parsing.py   # 54 unit tests
```

---

## API Reference

### HTTPRequest Dataclass

```python
@dataclass
class HTTPRequest:
    method: str = "GET"
    path: str = "/"
    query_params: dict = {}
    headers: dict = {}
    body: str | None = None
    remote_addr: str | None = None
    timestamp: str | None = None
    user_agent: str | None = None
    referer: str | None = None
    status_code: int | None = None
    response_size: int | None = None
    protocol: str = "HTTP/1.1"
```

### Log Parser

```python
parse_log_line(log_line: str, log_format: LogFormat | None = None) -> HTTPRequest | None
parse_request_dict(request_data: dict) -> HTTPRequest
```

### Normalizer — Placeholder Mappings

| Pattern | Placeholder | Example |
|---------|-------------|---------|
| UUID | `{UUID}` | `a1b2c3d4-e5f6-7890-abcd-ef1234567890` |
| ISO timestamp | `{TIMESTAMP}` | `2025-02-15T10:30:00Z` |
| Unix timestamp | `{TIMESTAMP}` | `1739615400` |
| IPv4/IPv6 | `{IP}` | `192.168.1.1` |
| JWT | `{JWT}` | `eyJhbGciOiJIUzI1NiJ9.eyJ...` |
| Session param keys | `{SESSION_ID}` | `PHPSESSID`, `JSESSIONID`, `sid`, etc. |
| Hex tokens (32+ chars) | `{TOKEN}` | `abc123def456...` |
| Numeric path IDs (2+ digits) | `{ID}` | `/api/users/12345` → `/api/users/{ID}` |

```python
normalize_request(request: HTTPRequest) -> HTTPRequest  # returns new object, no mutation
```

### Serializer

```python
serialize_request(
    request: HTTPRequest,
    include_headers: bool = True,
    include_body: bool = True,
    max_body_length: int = 2048,
) -> str
```

Output format (matches `WAFClassifier._build_request_text()`):
```
METHOD /path?param=val HTTP/1.1
Header-Name: value

body content
```

### ParsingPipeline

```python
pipeline = ParsingPipeline(
    log_format=None,          # auto-detect if None
    include_headers=True,
    include_body=True,
    max_body_length=2048,
)

# From log lines (batch processing)
result = pipeline.process(log_line)         # -> str | None

# From live requests
result = pipeline.process_request(request)  # -> str
result = pipeline.process_dict(data)        # -> str

pipeline.stats()  # {"processed": N, "failed": N, "log_format": "auto"}
```

---

## Test Coverage — 54 Tests

| Test Class | Count | Coverage |
|------------|-------|----------|
| TestParseLogLine | 12 | Apache Common, Combined, Nginx, Juice Shop, WebGoat, DVWA, UUID/ID in path, empty, garbage, explicit format, response size dash |
| TestParseRequestDict | 3 | Full dict, minimal dict, method case normalization |
| TestNormalizeRequest | 14 | UUID, numeric ID, IP, timestamp, JWT, session ID, ISO/Unix timestamps, preserves method/protocol, no mutation, multiple IDs, single-digit not replaced |
| TestSerializeRequest | 10 | GET, query params, headers, skip headers, body, truncation, no-headers/no-body flags, JSON compaction, protocol |
| TestParsingPipeline | 15 | Nginx/Apache/JuiceShop/DVWA lines, normalization applied, empty/garbage, process_request, process_dict, explicit format, stats, batch processing, output format matches classifier |

---

## How Week 2 Connects to the Pipeline

```
Week 1 (Ingestion)                    Week 2 (Parsing)
┌──────────────────┐                  ┌──────────────────────────┐
│ LogProcessor     │                  │ ParsingPipeline          │
│   batch_reader   │── log lines ──▶ │   parse_log_line()       │
│   stream_tailer  │                  │   normalize_request()    │
│   IngestionQueue │                  │   serialize_request()    │
└──────────────────┘                  └──────────┬───────────────┘
                                                 │
                                      normalized serialized text
                                                 │
                                                 ▼
                                      Week 3: Tokenizer + Model
                                      Week 4: WAFClassifier.classify()
```

---

## Dependencies Unlocked

| Week | What's Unlocked |
|------|----------------|
| **Week 3 — Training Data** | `ParsingPipeline.process()` converts raw logs into normalized model input for `benign_requests.json` |
| **Week 4 — Inference** | `ParsingPipeline.process_request()` / `process_dict()` feeds live requests into WAFClassifier |
