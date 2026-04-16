"""
One-time script to export the PyTorch WAF classifier to ONNX format.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --model-path models/waf-distilbert --output models/waf-distilbert.onnx
"""
import argparse
import time
from pathlib import Path

import torch
import numpy as np


def export(model_path: str, output_path: str) -> None:
    model_dir = Path(model_path)
    onnx_out = Path(output_path)

    if not model_dir.exists():
        print(f"ERROR: model directory not found: {model_dir}")
        print("Run scripts/finetune_waf_model.py first.")
        return

    print(f"Loading PyTorch model from {model_dir} ...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    # Dummy input — batch=1, seq_len=16
    sample_text = "GET /login HTTP/1.1\nUser-Agent: test"
    enc = tokenizer(sample_text, return_tensors="pt", max_length=256, truncation=True)
    # DistilBERT does not use token_type_ids
    enc.pop("token_type_ids", None)
    dummy_input_ids = enc["input_ids"]
    dummy_attention_mask = enc["attention_mask"]

    print(f"Exporting to {onnx_out} ...")
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(onnx_out),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size"},
        },
        opset_version=18,
        do_constant_folding=True,
    )
    print("ONNX export complete.")

    # Validate with onnxruntime
    print("Validating ONNX model ...")
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(onnx_out),
            providers=["CPUExecutionProvider"],
        )

        feed = {
            "input_ids": dummy_input_ids.numpy().astype("int64"),
            "attention_mask": dummy_attention_mask.numpy().astype("int64"),
        }
        ort_out = session.run(["logits"], feed)[0]

        # Compare with PyTorch output
        with torch.no_grad():
            pt_out = model(**enc).logits.numpy()

        max_diff = float(np.abs(ort_out - pt_out).max())
        print(f"Max logit diff (PyTorch vs ONNX): {max_diff:.6f}")
        if max_diff > 1e-4:
            print("WARNING: logit diff is larger than expected — check export.")
        else:
            print("Validation passed.")

        # Latency benchmark — 100 single-request inferences
        texts = [sample_text] * 100

        # PyTorch latency
        t0 = time.perf_counter()
        with torch.no_grad():
            for t in texts:
                enc_t = tokenizer(t, return_tensors="pt", max_length=256, truncation=True)
                enc_t.pop("token_type_ids", None)
                model(**enc_t)
        pt_ms = (time.perf_counter() - t0) * 1000

        # ONNX latency
        t0 = time.perf_counter()
        for t in texts:
            enc_t = tokenizer(t, return_tensors="np", max_length=256, truncation=True)
            session.run(["logits"], {
                "input_ids": enc_t["input_ids"].astype("int64"),
                "attention_mask": enc_t["attention_mask"].astype("int64"),
            })
        onnx_ms = (time.perf_counter() - t0) * 1000

        speedup = pt_ms / onnx_ms if onnx_ms > 0 else float("inf")
        print(f"\nLatency benchmark (100 single-request inferences):")
        print(f"  PyTorch : {pt_ms:.1f}ms  ({pt_ms/100:.2f}ms per request)")
        print(f"  ONNX    : {onnx_ms:.1f}ms  ({onnx_ms/100:.2f}ms per request)")
        print(f"  Speedup : {speedup:.2f}x")

    except ImportError:
        print("onnxruntime not installed — skipping validation. pip install onnxruntime>=1.17.0")

    print(f"\nONNX model saved to: {onnx_out}")
    print("Set WAF_USE_ONNX=true to enable ONNX inference.")


def main():
    project_root = Path(__file__).parent.parent
    default_model = str(project_root / "models" / "waf-distilbert")
    default_output = str(project_root / "models" / "waf-distilbert.onnx")

    parser = argparse.ArgumentParser(description="Export WAF PyTorch model to ONNX")
    parser.add_argument("--model-path", default=default_model, help="Path to HuggingFace model directory")
    parser.add_argument("--output", default=default_output, help="Output ONNX file path")
    args = parser.parse_args()

    export(args.model_path, args.output)


if __name__ == "__main__":
    main()
