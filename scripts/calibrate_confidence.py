#!/usr/bin/env python3
"""
Calibrate confidence scores using isotonic regression on a held-out dataset.

Runs inference on the validation split of multiclass_dataset.json, then fits
isotonic regression calibrators per-class so that a reported confidence of X%
corresponds to approximately X% empirical accuracy.

Output: models/waf-distilbert-multiclass/calibration.json

Usage:
    python scripts/calibrate_confidence.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "waf-distilbert-multiclass"
DATASET_PATH = PROJECT_ROOT / "data" / "training" / "multiclass_dataset.json"

LABEL_NAMES = [
    "benign", "sqli", "xss", "rce",
    "path_traversal", "xxe", "ssrf", "other_attack",
]
NUM_LABELS = len(LABEL_NAMES)


def main():
    print("Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Loading dataset ...")
    with open(DATASET_PATH) as f:
        raw_data = json.load(f)

    # Use 10% as calibration set (same seed as training split)
    import random
    random.seed(42)
    random.shuffle(raw_data)
    cal_size = len(raw_data) // 10
    cal_data = raw_data[:cal_size]
    print(f"Calibration set: {len(cal_data)} samples")

    # Run inference on calibration set
    all_probs = []
    all_labels = []

    batch_size = 32
    texts = [r["text"] for r in cal_data]
    labels = [r["label_id"] for r in cal_data]

    print("Running inference ...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(device)
        enc.pop("token_type_ids", None)

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        all_probs.append(probs)
        all_labels.extend(labels[i:i + batch_size])

        if (i // batch_size) % 20 == 0:
            print(f"  {i + len(batch_texts)}/{len(texts)}")

    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    print(f"Inference complete: {all_probs.shape}")

    # Fit isotonic regression per class
    # For each class, we have:
    #   - predicted probability for that class (from softmax)
    #   - binary indicator: was this the true class?
    calibrators = {}

    print("\nFitting isotonic regression per class:")
    for class_idx, class_name in enumerate(LABEL_NAMES):
        class_probs = all_probs[:, class_idx]
        class_true = (all_labels == class_idx).astype(float)

        # Need at least some positive and negative examples
        n_pos = int(class_true.sum())
        n_neg = len(class_true) - n_pos
        if n_pos < 5 or n_neg < 5:
            print(f"  {class_name}: skipped (pos={n_pos}, neg={n_neg})")
            continue

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(class_probs, class_true)

        # Store as piecewise-linear function (X, Y breakpoints)
        # IsotonicRegression stores X_ and y_ (the fitted values)
        calibrators[class_name] = {
            "x": iso.X_thresholds_.tolist(),
            "y": iso.y_thresholds_.tolist(),
        }

        # Report calibration quality
        calibrated = iso.predict(class_probs)
        # Bin into 10 buckets and check calibration
        bins = np.linspace(0, 1, 11)
        bin_counts = []
        bin_acc = []
        for b_lo, b_hi in zip(bins[:-1], bins[1:]):
            mask = (calibrated >= b_lo) & (calibrated < b_hi)
            if mask.sum() > 0:
                bin_counts.append(int(mask.sum()))
                bin_acc.append(float(class_true[mask].mean()))

        print(f"  {class_name}: {n_pos} pos, {n_neg} neg, {len(iso.X_thresholds_)} breakpoints")

    # Save calibration data
    output = {
        "num_labels": NUM_LABELS,
        "label_names": LABEL_NAMES,
        "calibrators": calibrators,
        "calibration_set_size": len(cal_data),
    }

    output_path = MODEL_PATH / "calibration.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nCalibration saved to {output_path}")
    print(f"Classes calibrated: {list(calibrators.keys())}")


if __name__ == "__main__":
    main()
