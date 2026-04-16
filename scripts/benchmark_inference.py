"""
Comprehensive inference benchmark: compares latency across optimization tiers.

Matrix:
  - PyTorch FP32 (baseline)
  - ONNX FP32
  - ONNX INT8 (if available)
  - Cache hit simulation

Usage:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --model-path models/waf-distilbert-multiclass --rounds 200
"""
import argparse
import statistics
import time
from pathlib import Path


def _latency_stats(times_ms: list) -> dict:
    """Compute P50, P95, P99, mean from a list of latencies in ms."""
    times_ms.sort()
    n = len(times_ms)
    return {
        "mean": round(statistics.mean(times_ms), 3),
        "p50": round(times_ms[n // 2], 3),
        "p95": round(times_ms[int(n * 0.95)], 3),
        "p99": round(times_ms[int(n * 0.99)], 3),
        "min": round(min(times_ms), 3),
        "max": round(max(times_ms), 3),
    }


def benchmark_tokenizer(model_path: str, texts: list, max_seq_len: int):
    """Compare slow vs fast tokenizer speed."""
    from transformers import AutoTokenizer

    print("\n--- Tokenizer Benchmark ---")

    # Fast tokenizer
    tok_fast = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    times = []
    for t in texts:
        t0 = time.perf_counter()
        tok_fast(t, truncation=True, max_length=max_seq_len, return_tensors="np")
        times.append((time.perf_counter() - t0) * 1000)
    fast_stats = _latency_stats(times)
    print(f"  Fast tokenizer (Rust): P50={fast_stats['p50']}ms, P99={fast_stats['p99']}ms")

    # Slow tokenizer
    try:
        tok_slow = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        times = []
        for t in texts:
            t0 = time.perf_counter()
            tok_slow(t, truncation=True, max_length=max_seq_len, return_tensors="np")
            times.append((time.perf_counter() - t0) * 1000)
        slow_stats = _latency_stats(times)
        print(f"  Slow tokenizer (Python): P50={slow_stats['p50']}ms, P99={slow_stats['p99']}ms")
        print(f"  Fast tokenizer speedup: {slow_stats['p50'] / fast_stats['p50']:.2f}x (P50)")
    except Exception as e:
        print(f"  Slow tokenizer not available: {e}")


def benchmark_onnx(model_path: str, texts: list, max_seq_len: int):
    """Compare ONNX FP32 vs INT8 inference."""
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        print("\n--- ONNX Benchmark: SKIPPED (onnxruntime not installed) ---")
        return

    from transformers import AutoTokenizer

    model_dir = Path(model_path)
    fp32_path = model_dir / "model.onnx"
    int8_path = model_dir / "model_int8.onnx"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    results = {}

    for label, onnx_path in [("ONNX FP32", fp32_path), ("ONNX INT8", int8_path)]:
        if not onnx_path.exists():
            print(f"\n--- {label} Benchmark: SKIPPED ({onnx_path} not found) ---")
            continue

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
        )

        # Warmup
        enc = tokenizer(texts[0], truncation=True, max_length=max_seq_len, return_tensors="np")
        feed = {
            "input_ids": enc["input_ids"].astype("int64"),
            "attention_mask": enc["attention_mask"].astype("int64"),
        }
        for _ in range(10):
            session.run(["logits"], feed)

        # Benchmark
        times = []
        for t in texts:
            enc = tokenizer(t, truncation=True, max_length=max_seq_len, return_tensors="np")
            feed = {
                "input_ids": enc["input_ids"].astype("int64"),
                "attention_mask": enc["attention_mask"].astype("int64"),
            }
            t0 = time.perf_counter()
            session.run(["logits"], feed)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)

        stats = _latency_stats(times)
        results[label] = stats
        print(f"\n--- {label} Benchmark ({len(texts)} requests) ---")
        print(f"  P50={stats['p50']}ms, P95={stats['p95']}ms, P99={stats['p99']}ms")
        print(f"  Mean={stats['mean']}ms, Min={stats['min']}ms, Max={stats['max']}ms")

    if "ONNX FP32" in results and "ONNX INT8" in results:
        speedup = results["ONNX FP32"]["p50"] / results["ONNX INT8"]["p50"]
        print(f"\n  INT8 speedup over FP32: {speedup:.2f}x (P50)")


def benchmark_pytorch(model_path: str, texts: list, max_seq_len: int):
    """Benchmark PyTorch FP32 inference."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        print("\n--- PyTorch Benchmark: SKIPPED (torch not installed) ---")
        return

    print(f"\n--- PyTorch FP32 Benchmark ({len(texts)} requests) ---")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Warmup
    with torch.no_grad():
        enc = tokenizer(texts[0], truncation=True, max_length=max_seq_len, return_tensors="pt")
        enc.pop("token_type_ids", None)
        for _ in range(5):
            model(**enc)

    times = []
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, truncation=True, max_length=max_seq_len, return_tensors="pt")
            enc.pop("token_type_ids", None)
            t0 = time.perf_counter()
            model(**enc)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)

    stats = _latency_stats(times)
    print(f"  P50={stats['p50']}ms, P95={stats['p95']}ms, P99={stats['p99']}ms")
    print(f"  Mean={stats['mean']}ms, Min={stats['min']}ms, Max={stats['max']}ms")


def benchmark_seq_length(model_path: str, text: str):
    """Compare inference latency at different sequence lengths."""
    try:
        import onnxruntime as ort
    except ImportError:
        return

    from transformers import AutoTokenizer

    model_dir = Path(model_path)
    onnx_path = model_dir / "model.onnx"
    if not onnx_path.exists():
        return

    print("\n--- Sequence Length Impact ---")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    for seq_len in [64, 128, 256, 512]:
        enc = tokenizer(text, truncation=True, max_length=seq_len, return_tensors="np")
        feed = {
            "input_ids": enc["input_ids"].astype("int64"),
            "attention_mask": enc["attention_mask"].astype("int64"),
        }
        # Warmup
        for _ in range(10):
            sess.run(["logits"], feed)
        # Measure
        t0 = time.perf_counter()
        for _ in range(100):
            sess.run(["logits"], feed)
        avg_ms = (time.perf_counter() - t0) * 1000 / 100
        actual_len = enc["input_ids"].shape[1]
        print(f"  max_length={seq_len:>3}, actual_tokens={actual_len:>3}, avg={avg_ms:.3f}ms")


def main():
    project_root = Path(__file__).parent.parent
    default_model = str(project_root / "models" / "waf-distilbert-multiclass")

    parser = argparse.ArgumentParser(description="Benchmark WAF inference pipeline")
    parser.add_argument("--model-path", default=default_model)
    parser.add_argument("--rounds", type=int, default=100, help="Number of inference rounds")
    parser.add_argument("--max-seq-len", type=int, default=256)
    args = parser.parse_args()

    # Test payloads: mix of benign and attack traffic
    test_texts = [
        "GET /index.html HTTP/1.1\nHost: example.com\nUser-Agent: Mozilla/5.0",
        "GET /api/v1/users?page=1&limit=10 HTTP/1.1\nAuthorization: Bearer token123",
        "POST /login HTTP/1.1\nContent-Type: application/json\n\n{\"user\":\"admin\",\"pass\":\"secret\"}",
        "GET /search?q=<script>alert(document.cookie)</script> HTTP/1.1",
        "POST /api HTTP/1.1\n\nuser=admin' OR 1=1-- -",
        "GET /../../etc/passwd HTTP/1.1",
        "GET /api?url=http://169.254.169.254/latest/meta-data/ HTTP/1.1",
        "POST /upload HTTP/1.1\nContent-Type: text/xml\n\n<?xml version=\"1.0\"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM \"file:///etc/shadow\">]><root>&xxe;</root>",
        "GET /admin;cat /etc/shadow HTTP/1.1",
        "GET /static/style.css HTTP/1.1\nAccept: text/css",
    ]

    # Extend to desired rounds
    texts = (test_texts * ((args.rounds // len(test_texts)) + 1))[:args.rounds]

    print(f"WAF Inference Benchmark")
    print(f"  Model: {args.model_path}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Max seq length: {args.max_seq_len}")
    print(f"  Unique payloads: {len(test_texts)}")

    benchmark_tokenizer(args.model_path, texts, args.max_seq_len)
    benchmark_pytorch(args.model_path, texts, args.max_seq_len)
    benchmark_onnx(args.model_path, texts, args.max_seq_len)
    benchmark_seq_length(args.model_path, texts[3])  # Use the XSS payload (longer)

    print("\n--- Done ---")


if __name__ == "__main__":
    main()
