#!/usr/bin/env python3
"""
WAF Model Evaluation Script

Loads the WAFClassifier directly (no running server needed), runs it against
all attack payloads and benign samples, and produces:
  - Per-category precision / recall / F1
  - Overall confusion matrix PNG
  - ROC curve PNG
  - Structured metrics JSON

Usage:
    python benchmarks/run_evaluation.py
    python benchmarks/run_evaluation.py --threshold 0.65
    python benchmarks/run_evaluation.py --output-dir benchmarks/results
"""
import sys
import json
import time
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ATTACK_TESTS_DIR = PROJECT_ROOT / "scripts" / "attack_tests"
BENIGN_SAMPLES_PATH = PROJECT_ROOT / "scripts" / "data" / "extra_benign_samples.json"
MODEL_PATH = PROJECT_ROOT / "models" / "waf-distilbert"

# Category name -> (script filename, list variable name)
ATTACK_SUITES: List[Tuple[str, str, str]] = [
    ("SQL Injection",           "01_sql_injection.py",     "SQL_INJECTION_ATTACKS"),
    ("XSS",                     "02_xss_attacks.py",       "XSS_ATTACKS"),
    ("Command Injection",       "03_command_injection.py",  "COMMAND_INJECTION_ATTACKS"),
    ("Path Traversal",          "04_path_traversal.py",    "PATH_TRAVERSAL_ATTACKS"),
    ("XXE",                     "05_xxe_attacks.py",       "XXE_ATTACKS"),
    ("SSRF",                    "06_ssrf_attacks.py",      "SSRF_ATTACKS"),
    ("Header Injection",        "07_header_injection.py",  "HEADER_INJECTION_ATTACKS"),
    ("LDAP/XPath Injection",    "08_ldap_xpath_injection.py", "INJECTION_ATTACKS"),
    ("DoS Patterns",            "09_dos_patterns.py",      "DOS_ATTACKS"),
    ("Mixed/Blended",           "10_mixed_blended.py",     "MIXED_ATTACKS"),
]

BENIGN_SUITE = ("Benign (FP Regression)", "11_fp_regression.py", "BENIGN_REQUESTS")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_payload_list(script_file: str, var_name: str) -> list:
    """Dynamically import a payload list from an attack test script."""
    path = ATTACK_TESTS_DIR / script_file
    if not path.exists():
        print(f"  [WARN] {script_file} not found, skipping")
        return []
    spec = importlib.util.spec_from_file_location("mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    payloads = getattr(mod, var_name, None)
    if payloads is None:
        # Fallback: find largest list of dicts in module
        for attr in dir(mod):
            val = getattr(mod, attr)
            if isinstance(val, list) and len(val) > 3 and isinstance(val[0], dict):
                payloads = val
                break
    return payloads or []


def payload_to_request_text(payload: dict) -> str:
    """Convert a payload dict to raw HTTP request text (matches classifier format)."""
    method = payload.get("method", "POST" if "body" in payload else "GET")
    path = payload.get("path", "/")
    query_params = payload.get("query", {})
    headers = payload.get("headers", {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Content-Type": "application/json",
    })
    body = payload.get("body")

    # Flatten query params that are dicts (e.g. MongoDB NoSQL payloads)
    flat_query = {}
    for k, v in query_params.items():
        flat_query[k] = json.dumps(v) if isinstance(v, dict) else str(v)

    # Build path with query string
    full_path = path
    if flat_query:
        query_str = "&".join(f"{k}={v}" for k, v in flat_query.items())
        if query_str:
            full_path = f"{path}?{query_str}"

    lines = [f"{method} {full_path} HTTP/1.1"]

    if headers:
        for key, value in headers.items():
            if key.lower() not in ("host", "content-length"):
                lines.append(f"{key}: {value}")

    if body:
        lines.append("")
        if isinstance(body, dict):
            lines.append(json.dumps(body))
        elif isinstance(body, str):
            lines.append(body)
        else:
            lines.append(str(body))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(threshold: float = 0.65, output_dir: str = "benchmarks/results"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    print(f"Loading WAF model from {MODEL_PATH} ...")
    from backend.ml.waf_classifier import WAFClassifier
    classifier = WAFClassifier(model_path=str(MODEL_PATH), threshold=threshold)
    if not classifier.is_loaded:
        print("ERROR: Model failed to load. Ensure models/waf-distilbert/ exists.")
        sys.exit(1)
    print(f"Model loaded on {classifier.device} | threshold={threshold}\n")

    # ---- Collect all samples ----
    # Structure: list of (text, true_label, category)
    samples: List[Tuple[str, int, str]] = []  # label: 1=malicious, 0=benign

    # Attack payloads (true label = 1)
    for category, script, var_name in ATTACK_SUITES:
        payloads = load_payload_list(script, var_name)
        print(f"  {category}: {len(payloads)} payloads")
        for p in payloads:
            text = payload_to_request_text(p)
            samples.append((text, 1, category))

    # Benign payloads from FP regression (true label = 0)
    benign_payloads = load_payload_list(BENIGN_SUITE[1], BENIGN_SUITE[2])
    print(f"  Benign (FP regression): {len(benign_payloads)} samples")
    for p in benign_payloads:
        text = payload_to_request_text(p)
        samples.append((text, 0, "Benign"))

    # Extra benign samples from JSON (raw HTTP text)
    if BENIGN_SAMPLES_PATH.exists():
        with open(BENIGN_SAMPLES_PATH) as f:
            extra_benign = json.load(f)
        print(f"  Benign (extra): {len(extra_benign)} samples")
        for text in extra_benign:
            samples.append((str(text), 0, "Benign"))

    total = len(samples)
    malicious_count = sum(1 for _, lbl, _ in samples if lbl == 1)
    benign_count = total - malicious_count
    print(f"\nTotal samples: {total} ({malicious_count} malicious, {benign_count} benign)\n")

    # ---- Run inference ----
    print("Running inference ...")
    texts = [s[0] for s in samples]
    true_labels = [s[1] for s in samples]
    categories = [s[2] for s in samples]

    t0 = time.perf_counter()
    results = classifier.classify_batch(texts, batch_size=32)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    malicious_scores = [r.get("malicious_score", 0.0) for r in results]
    pred_labels = [1 if r.get("is_malicious", False) else 0 for r in results]

    print(f"Inference complete: {elapsed_ms:.0f}ms for {total} samples "
          f"({elapsed_ms / total:.1f}ms/sample)\n")

    # ---- Compute metrics ----
    y_true = np.array(true_labels)
    y_pred = np.array(pred_labels)
    y_scores = np.array(malicious_scores)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)
    print(f"  Accuracy:   {accuracy:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1 Score:   {f1:.4f}")
    print(f"  FPR:        {fpr:.4f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    # ---- Per-category metrics ----
    category_metrics = {}
    all_categories = [c for c, _, _ in ATTACK_SUITES] + ["Benign"]

    print(f"\n{'Category':<25} {'Prec':>6} {'Recall':>6} {'F1':>6} {'Det':>5} {'Total':>5}")
    print("-" * 60)

    for cat in all_categories:
        mask = np.array([c == cat for c in categories])
        cat_true = y_true[mask]
        cat_pred = y_pred[mask]
        cat_total = int(mask.sum())

        if cat == "Benign":
            # For benign: "detected" means correctly allowed (pred=0)
            correct = int(np.sum(cat_pred == 0))
            false_positives = int(np.sum(cat_pred == 1))
            cat_prec = correct / cat_total if cat_total > 0 else 0.0
            fp_rate = false_positives / cat_total if cat_total > 0 else 0.0
            category_metrics[cat] = {
                "total": cat_total,
                "correctly_allowed": correct,
                "false_positives": false_positives,
                "fp_rate": round(fp_rate, 4),
                "accuracy": round(cat_prec, 4),
            }
            print(f"  {'Benign (FP rate)':<23} {'':<6} {'':<6} {'':<6} "
                  f"{false_positives:>5} / {cat_total}")
        else:
            # For attack categories: detected = pred=1 (blocked)
            cat_tp = int(np.sum((cat_true == 1) & (cat_pred == 1)))
            cat_fn = int(np.sum((cat_true == 1) & (cat_pred == 0)))
            det_rate = cat_tp / cat_total if cat_total > 0 else 0.0
            # Precision within this category is always 1.0 (no benign mixed in)
            cat_recall = det_rate
            cat_f1 = det_rate  # Same as recall when precision=1.0 for pure attack sets
            category_metrics[cat] = {
                "total": cat_total,
                "detected": cat_tp,
                "missed": cat_fn,
                "detection_rate": round(det_rate, 4),
                "recall": round(cat_recall, 4),
                "f1": round(cat_f1, 4),
            }
            print(f"  {cat:<23} {'--':>6} {det_rate:>6.2%} {cat_f1:>6.3f} "
                  f"{cat_tp:>5} / {cat_total}")

    print("-" * 60)

    # ---- Build metrics JSON ----
    metrics = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": "waf-distilbert",
        "threshold": threshold,
        "device": str(classifier.device),
        "total_samples": total,
        "malicious_samples": malicious_count,
        "benign_samples": benign_count,
        "inference_time_ms": round(elapsed_ms, 1),
        "avg_latency_ms": round(elapsed_ms / total, 2),
        "overall": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "fpr": round(fpr, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        },
        "per_category": category_metrics,
    }

    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # ---- Generate charts ----
    generate_confusion_matrix(tp, fp, tn, fn, output_path)
    generate_roc_curve(y_true, y_scores, output_path)
    generate_category_chart(category_metrics, output_path)

    print(f"\nAll outputs saved to {output_path}/")
    return metrics


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def generate_confusion_matrix(tp: int, fp: int, tn: int, fn: int, output_path: Path):
    """Generate and save confusion matrix as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.array([[tn, fp], [fn, tp]])
    labels = ["Benign", "Malicious"]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")

    # Annotate cells
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, str(matrix[i, j]),
                    ha="center", va="center", fontsize=20, fontweight="bold",
                    color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title("WAF Model - Confusion Matrix", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    path = output_path / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved to {path}")


def generate_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, output_path: Path):
    """Generate and save ROC curve as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Compute ROC points
    thresholds = np.linspace(0, 1, 200)
    fprs = []
    tprs = []

    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        tp_t = np.sum((y_true == 1) & (preds == 1))
        fp_t = np.sum((y_true == 0) & (preds == 1))
        fn_t = np.sum((y_true == 1) & (preds == 0))
        tn_t = np.sum((y_true == 0) & (preds == 0))

        tpr = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
        fpr = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)

    # AUC (trapezoidal rule, sorted by FPR)
    sorted_pairs = sorted(zip(fprs, tprs))
    fprs_sorted = [p[0] for p in sorted_pairs]
    tprs_sorted = [p[1] for p in sorted_pairs]
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = _trapz(tprs_sorted, fprs_sorted)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fprs_sorted, tprs_sorted, color="#2563eb", linewidth=2,
            label=f"WAF Model (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1,
            label="Random Classifier")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("WAF Model - ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = output_path / "roc_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ROC curve saved to {path} (AUC={auc:.4f})")


def generate_category_chart(category_metrics: dict, output_path: Path):
    """Generate per-category detection rate bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter to attack categories only (not Benign)
    attack_cats = {k: v for k, v in category_metrics.items() if k != "Benign"}
    if not attack_cats:
        return

    names = list(attack_cats.keys())
    rates = [v.get("detection_rate", 0.0) * 100 for v in attack_cats.values()]

    # Color by performance
    colors = []
    for r in rates:
        if r >= 80:
            colors.append("#22c55e")   # green
        elif r >= 50:
            colors.append("#eab308")   # yellow
        else:
            colors.append("#ef4444")   # red

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, rates, color=colors, edgecolor="white", height=0.6)

    # Add value labels
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}%", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Detection Rate (%)", fontsize=12)
    ax.set_title("WAF Model - Detection Rate by Attack Category",
                 fontsize=14, fontweight="bold")
    ax.set_xlim([0, 110])
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    path = output_path / "detection_by_category.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Category chart saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WAF Model Evaluation")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Classification threshold (default: 0.65)")
    parser.add_argument("--output-dir", default="benchmarks/results",
                        help="Output directory for results (default: benchmarks/results)")
    args = parser.parse_args()

    run_evaluation(threshold=args.threshold, output_dir=args.output_dir)
