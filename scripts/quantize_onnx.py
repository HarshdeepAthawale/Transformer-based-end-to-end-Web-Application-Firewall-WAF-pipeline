"""
Quantize the ONNX WAF model to INT8 for 2-3x faster CPU inference.

Usage:
    python scripts/quantize_onnx.py
    python scripts/quantize_onnx.py --input models/waf-distilbert-multiclass/model.onnx
"""
import argparse
import time
from pathlib import Path

import numpy as np


def quantize(input_path: str, output_path: str) -> None:
    inp = Path(input_path)
    out = Path(output_path)

    if not inp.exists():
        print(f"ERROR: ONNX model not found: {inp}")
        print("Run scripts/export_onnx.py first.")
        return

    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    # Preprocess: load model, clear faulty shape info, re-save for shape inference
    print("Preprocessing ONNX model for quantization ...")
    model = onnx.load(str(inp))

    # Strip value_info entries that cause shape inference conflicts
    # (common with dynamo-based torch.onnx.export)
    while model.graph.value_info:
        model.graph.value_info.pop()

    preprocessed = inp.parent / f"{inp.stem}_preprocessed{inp.suffix}"
    onnx.save(model, str(preprocessed))

    print(f"Quantizing {inp} -> {out} (INT8 dynamic) ...")
    quantize_dynamic(
        model_input=str(preprocessed),
        model_output=str(out),
        weight_type=QuantType.QInt8,
    )

    # Clean up preprocessed temp file
    preprocessed.unlink(missing_ok=True)

    # Report size reduction
    orig_size = inp.stat().st_size
    quant_size = out.stat().st_size
    # ONNX external data files
    orig_data = inp.parent / (inp.name + ".data")
    quant_data = out.parent / (out.name + ".data")
    if orig_data.exists():
        orig_size += orig_data.stat().st_size
    if quant_data.exists():
        quant_size += quant_data.stat().st_size

    ratio = orig_size / quant_size if quant_size > 0 else 0
    print(f"Original: {orig_size / 1024 / 1024:.1f}MB")
    print(f"Quantized: {quant_size / 1024 / 1024:.1f}MB")
    print(f"Compression: {ratio:.1f}x")

    # Validate: compare FP32 vs INT8 outputs
    print("\nValidating INT8 accuracy ...")
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(inp.parent), use_fast=True)

    test_inputs = [
        "GET /index.html HTTP/1.1\nHost: example.com",
        "GET /search?q=<script>alert(1)</script> HTTP/1.1",
        "POST /login HTTP/1.1\nContent-Type: application/x-www-form-urlencoded\n\nuser=admin' OR 1=1--",
        "GET /../../etc/passwd HTTP/1.1",
        "GET /api/v1/users HTTP/1.1\nAuthorization: Bearer eyJhbGciOiJIUzI1NiJ9",
    ]

    sess_fp32 = ort.InferenceSession(str(inp), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])

    max_diff = 0.0
    for text in test_inputs:
        enc = tokenizer(text, truncation=True, max_length=256, return_tensors="np")
        feed = {
            "input_ids": enc["input_ids"].astype("int64"),
            "attention_mask": enc["attention_mask"].astype("int64"),
        }
        fp32_out = sess_fp32.run(["logits"], feed)[0]
        int8_out = sess_int8.run(["logits"], feed)[0]
        diff = float(np.abs(fp32_out - int8_out).max())
        max_diff = max(max_diff, diff)

    print(f"Max logit diff (FP32 vs INT8): {max_diff:.6f}")
    if max_diff > 0.5:
        print("WARNING: large logit divergence -- check quantized model quality")
    else:
        print("Validation passed (acceptable divergence).")

    # Latency benchmark: FP32 vs INT8
    print("\nLatency benchmark (100 inferences) ...")
    sample = "GET /search?q=<script>alert(1)</script> HTTP/1.1\nHost: example.com"
    enc = tokenizer(sample, truncation=True, max_length=256, return_tensors="np")
    feed = {
        "input_ids": enc["input_ids"].astype("int64"),
        "attention_mask": enc["attention_mask"].astype("int64"),
    }

    # Warmup
    for _ in range(10):
        sess_fp32.run(["logits"], feed)
        sess_int8.run(["logits"], feed)

    t0 = time.perf_counter()
    for _ in range(100):
        sess_fp32.run(["logits"], feed)
    fp32_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(100):
        sess_int8.run(["logits"], feed)
    int8_ms = (time.perf_counter() - t0) * 1000

    speedup = fp32_ms / int8_ms if int8_ms > 0 else float("inf")
    print(f"  ONNX FP32: {fp32_ms:.1f}ms  ({fp32_ms / 100:.2f}ms per request)")
    print(f"  ONNX INT8: {int8_ms:.1f}ms  ({int8_ms / 100:.2f}ms per request)")
    print(f"  Speedup:   {speedup:.2f}x")

    print(f"\nINT8 model saved to: {out}")


def main():
    project_root = Path(__file__).parent.parent
    default_input = str(project_root / "models" / "waf-distilbert-multiclass" / "model.onnx")
    default_output = str(project_root / "models" / "waf-distilbert-multiclass" / "model_int8.onnx")

    parser = argparse.ArgumentParser(description="Quantize ONNX WAF model to INT8")
    parser.add_argument("--input", default=default_input, help="Input ONNX model path")
    parser.add_argument("--output", default=default_output, help="Output INT8 ONNX model path")
    args = parser.parse_args()

    quantize(args.input, args.output)


if __name__ == "__main__":
    main()
