# Week 4: Scale or Die - PostgreSQL + Redis Usage + ONNX + Celery

**Period:** April 7-11, 2026
**Goal:** Fix all 6 critical scalability bottlenecks. Target: 5,000+ req/sec, 1,000+ tenants.

## Week 4 Overview

| Day | Theme | Target Metric |
|-----|-------|--------------|
| Day 1 | PostgreSQL migration + connection pooling | Concurrent writes, pool=20 |
| Day 2 | Redis-backed usage counting | 0 DB writes/request |
| Day 3 | ONNX ML optimization | 3-5x inference speedup |
| Day 4 | Celery task queue | Proper retries, multi-worker |
| Day 5 | Load testing + benchmarks | Prove 5,000+ req/sec |

## Expected Outcome

| Metric | Before (Week 3) | After (Week 4) |
|--------|-----------------|----------------|
| Max throughput | ~100 req/sec | 5,000+ req/sec |
| DB writes/request | 2-3 | 0 (Redis batch) |
| ML inference latency | 50-200ms | 15-60ms (ONNX) |
| Max concurrent tenants | ~50 | 1,000+ |
| Background task resilience | Thread crash = lost | Celery retry + DLQ |

## Stack Changes
- **Database:** SQLite -> PostgreSQL (with connection pooling)
- **Usage tracking:** Per-request DB write -> Redis INCR + 30s batch flush
- **ML inference:** PyTorch -> ONNX Runtime (opt-in)
- **Background tasks:** Python threads -> Celery + Redis broker
