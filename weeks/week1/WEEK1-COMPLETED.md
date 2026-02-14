# Week 1: Log Ingestion Pipeline — Completed

> **Duration**: Week 1 of Phase 1
> **Goal**: Batch and streaming ingestion from Apache/Nginx access logs with format auto-detection, async queue, retry logic, config integration, and wiring into the existing background task system.
> **Status**: COMPLETE — 12/12 tasks done, 54/54 tests passing

---

## Summary

Built the entire `backend/ingestion/` module from scratch — 7 Python files providing log format detection, chunk-based batch reading, async file tailing, a producer/consumer queue, exponential backoff retry, and typed YAML config loading. Wired it all into the existing `LogProcessor` background task so it actually ingests logs instead of being an empty stub. Covered everything with 50 unit tests and 4 integration tests.

---

## Tasks Completed

| # | Task | File | Description |
|---|------|------|-------------|
| 1 | Create ingestion module | `backend/ingestion/__init__.py` | Package with public exports for all components |
| 2 | Log format detector | `backend/ingestion/format_detector.py` | Auto-detects Apache Common, Apache Combined, Nginx Combined via regex + majority voting |
| 3 | Batch reader | `backend/ingestion/batch_reader.py` | Chunk-based file reading with skip_lines, max_lines, .gz support |
| 4 | Stream tailer | `backend/ingestion/stream_tailer.py` | Async file tailing with polling, log rotation detection, follow mode |
| 5 | Async queue | `backend/ingestion/queue.py` | asyncio.Queue wrapper with producer/consumer pattern and graceful shutdown |
| 6 | Config verification | `config/config.yaml` | Verified existing `ingestion:` section matches plan spec |
| 7 | Unit tests (initial) | `tests/unit/test_ingestion.py` | 37 tests for core modules |
| 8 | Retry wrapper | `backend/ingestion/retry.py` | Exponential backoff for sync/async with decorator support |
| 9 | Config loader | `backend/ingestion/config.py` | Typed dataclasses loaded from YAML with defaults fallback |
| 10 | Wire into LogProcessor | `backend/tasks/log_processor.py` | Replaced empty stub with real ingestion logic |
| 11 | Integration test | `tests/integration/test_ingestion_e2e.py` | End-to-end: log file → LogProcessor → queue |
| 12 | Extended unit tests | `tests/unit/test_ingestion.py` | Added retry + config tests (total: 50 unit tests) |

---

## Files Created / Modified

### New Files (7)

```
backend/ingestion/
├── __init__.py           # Public exports for all ingestion components
├── format_detector.py    # Log format detection (regex + majority vote)
├── batch_reader.py       # Chunk-based file reading (.log, .gz)
├── stream_tailer.py      # Async file tailing with rotation detection
├── queue.py              # Async producer/consumer queue
├── retry.py              # Exponential backoff retry (sync + async + decorator)
└── config.py             # YAML config loader → typed dataclasses
```

### Modified Files (1)

```
backend/tasks/
└── log_processor.py      # Replaced empty _process_logs() stub with real ingestion
```

### Test Files (2)

```
tests/unit/
└── test_ingestion.py         # 50 unit tests
tests/integration/
└── test_ingestion_e2e.py     # 4 integration tests
```

---

## API Reference

### Format Detection

```python
class LogFormat(Enum):
    APACHE_COMMON = "apache_common"
    APACHE_COMBINED = "apache_combined"
    NGINX_COMBINED = "nginx_combined"
    UNKNOWN = "unknown"

detect_format(log_line: str) -> LogFormat
detect_from_file(log_path: str, sample_lines: int = 10) -> LogFormat
```

### Batch Reading

```python
read_chunks(
    log_path: str,
    chunk_size: int = 1000,
    max_lines: int | None = None,
    skip_lines: int = 0,
) -> Iterator[list[str]]
```

### Stream Tailing

```python
async tail_lines(
    log_path: str,
    follow: bool = True,
    poll_interval: float = 0.1,
    start_from_end: bool = True,
) -> AsyncIterator[str]
```

### Async Queue

```python
queue = IngestionQueue(maxsize=10000, name="ingestion")
await queue.put(item)
item = await queue.get()
await queue.consume(handler)   # consumer loop
await queue.shutdown()         # graceful stop
queue.stats()                  # {"name", "size", "produced", "consumed", "running"}
```

### Retry

```python
config = RetryConfig(max_retries=3, initial_delay=1.0, max_delay=60.0, exponential_base=2.0)
result = with_retry_sync(fn, config, *args, **kwargs)
result = await with_retry_async(fn, config, *args, **kwargs)

@retry(config)           # decorator for sync or async functions
def my_function(): ...
```

### Config Loading

```python
config = load_ingestion_config()              # from config/config.yaml
config = load_ingestion_config("/custom.yaml") # from custom path

config.batch.chunk_size       # 1000
config.streaming.poll_interval # 0.1
config.retry.max_retries      # 3
```

### LogProcessor (Wired)

```python
processor = LogProcessor(
    log_path="/var/log/nginx/access.log",
    interval_seconds=10,
    config=ingestion_config,    # optional, loads from YAML if None
    queue=ingestion_queue,      # optional, creates one if None
)
processor.start()               # background thread
processor.queue                 # access the queue for downstream consumers
processor.stats()               # {"running", "log_path", "detected_format", "lines_read", "queue": {...}}
processor.stop()
```

---

## Test Results

```
54 passed in 10.61s
```

### Unit Tests — 50 tests

| Test Class | Count | Coverage |
|------------|-------|----------|
| TestDetectFormat | 10 | Apache Common, Combined, Nginx, app-specific (Juice Shop, WebGoat, DVWA), edge cases |
| TestDetectFromFile | 6 | File detection, .gz, empty file, missing file, majority vote |
| TestReadChunks | 10 | Chunking, max_lines, skip_lines, .gz, empty lines, errors |
| TestTailLines | 5 | Read existing, follow new lines, start-from-end, missing file, empty lines |
| TestIngestionQueue | 6 | put/get, consumer loop, shutdown, stats, error handling, timeout |
| TestRetry | 7 | First-try success, retries on OSError, exhaustion, non-retryable, async, decorators |
| TestIngestionConfig | 6 | Project config, custom path, missing file, empty section, partial config, defaults |

### Integration Tests — 4 tests

| Test | Coverage |
|------|----------|
| test_full_pipeline | 10-line Nginx log → LogProcessor → format detected → all 10 lines queued |
| test_incremental_reading | 5 lines → process → append 5 more → process → total 10 queued |
| test_missing_log_file | Handles missing file gracefully (0 lines, no crash) |
| test_stats | Stats reflect correct state after processing cycle |

---

## Config Used (from `config/config.yaml`)

```yaml
ingestion:
  batch:
    chunk_size: 1000
    max_lines: null
    skip_lines: 0
  streaming:
    poll_interval: 0.1
    follow: true
    buffer_size: 10000
  retry:
    max_retries: 3
    initial_delay: 1.0
    max_delay: 60.0
    exponential_base: 2.0
```

---

## What This Unlocks for Later Weeks

| Week | Dependency |
|------|-----------|
| **Week 2 — Parsing** | `LogProcessor.queue` provides raw log lines for `ParsingPipeline` to consume |
| **Week 3 — Training Data** | `batch_reader` + `stream_tailer` feed `generate_training_data.py` to collect logs from apps |
| **Week 4 — Integration** | `stream_tailer` enables live log monitoring for the WAF service on port 8000 |
