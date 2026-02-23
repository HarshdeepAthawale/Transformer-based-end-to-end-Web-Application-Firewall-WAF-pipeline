#!/usr/bin/env python3
"""
Fine-tune DistilBERT on the WAF dataset from HuggingFace.
Uses HuggingFace Trainer API for simplicity.

Dataset: notesbymuneeb/ai-waf-dataset
Model: distilbert-base-uncased -> binary classifier (benign/malicious)

Augmentation (--augment flag):
  - Adds extra benign samples to reduce false positives
  - Adds evasive malicious payloads (from stress test + missed_malicious_samples.json)
    as hard negatives to close detection gaps

Usage:
    python scripts/finetune_waf_model.py
    python scripts/finetune_waf_model.py --epochs 5 --batch-size 16
    python scripts/finetune_waf_model.py --augment          # with augmentation
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
EXTRA_BENIGN_SAMPLES_PATH = DATA_DIR / "extra_benign_samples.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for WAF")
    parser.add_argument("--model-name", default="distilbert-base-uncased", help="Base model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output-dir", default="models/waf-distilbert", help="Output directory")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Augment training data with extra benign + evasive malicious samples",
    )
    return parser.parse_args()


def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, F1."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _req_to_text(req: Dict) -> str:
    """Convert a stress-test request dict into a raw HTTP-like string for training.

    Now also serializes headers (when present) to match the inference-time
    format produced by backend.parsing.serializer.serialize_request().
    This closes the training-inference gap that caused the model to never
    learn header-based attack patterns (CRLF, smuggling, header overwrite).
    """
    method = req.get("method", "GET")
    path = req.get("url", "/")
    params = req.get("params")
    headers = req.get("headers")
    body = req.get("json")

    full_path = path
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        if qs:
            full_path = f"{path}?{qs}"

    lines = [f"{method} {full_path} HTTP/1.1"]

    # Serialize headers (matching inference-time serializer skip list)
    if headers:
        skip = {"host", "content-length", "connection", "accept-encoding", "transfer-encoding"}
        for key, value in headers.items():
            if key.lower() not in skip:
                lines.append(f"{key}: {value}")

    if body is not None:
        lines.append("")
        lines.append(json.dumps(body) if isinstance(body, dict) else str(body))

    return "\n".join(lines)


def _load_extra_benign_samples() -> List[str]:
    """Load extra benign samples from data file (reduces false positives)."""
    if not EXTRA_BENIGN_SAMPLES_PATH.exists():
        return []
    with open(EXTRA_BENIGN_SAMPLES_PATH) as f:
        return json.load(f)


def _build_augmentation_samples() -> List[Dict[str, str]]:
    """
    Build augmentation rows (list of {text, label}) from:
      1. scripts/data/extra_benign_samples.json (benign)
      2. Stress-test MALICIOUS_POOL entries from scripts/data/malicious_pool.json (malicious)
      3. scripts/missed_malicious_samples.json if it exists (malicious)
    """
    rows: List[Dict[str, str]] = []

    # 1. Extra benign (from scripts/data/extra_benign_samples.json)
    extra_benign = _load_extra_benign_samples()
    for text in extra_benign:
        rows.append({"text": text, "label": "benign"})

    # 2. Full evasive malicious pool from stress test
    stress_test_path = SCRIPT_DIR / "stress_test_1000_evasive.py"
    if stress_test_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("stress_test", str(stress_test_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        malicious_pool = getattr(mod, "MALICIOUS_POOL", [])
        for req in malicious_pool:
            rows.append({"text": _req_to_text(req), "label": "malicious"})
        print(f"  Loaded {len(malicious_pool)} evasive malicious payloads from stress test pool")
    else:
        print("  WARNING: stress_test_1000_evasive.py not found, skipping malicious pool.")

    # 3. Missed malicious samples file (produced by stress test / harvest)
    missed_path = SCRIPT_DIR / "missed_malicious_samples.json"
    if missed_path.exists():
        with open(missed_path) as f:
            missed = json.load(f)
        # Repeat each missed sample 3x to give it more weight
        for req in missed:
            text = _req_to_text(req)
            for _ in range(3):
                rows.append({"text": text, "label": "malicious"})
        print(f"  Loaded {len(missed)} missed-malicious samples (x3 = {len(missed)*3} rows)")

    # 4. Categorized payloads from data/malicious/ (1x weight)
    data_malicious_dir = PROJECT_ROOT / "data" / "malicious"
    if data_malicious_dir.exists():
        cat_count = 0
        for json_file in sorted(data_malicious_dir.glob("*.json")):
            try:
                entries = json.loads(json_file.read_text())
                for req in entries:
                    rows.append({"text": _req_to_text(req), "label": "malicious"})
                cat_count += len(entries)
            except (json.JSONDecodeError, IOError):
                pass
        if cat_count:
            print(f"  Loaded {cat_count} categorized malicious samples from data/malicious/")

    print(f"  Augmentation: {sum(1 for r in rows if r['label']=='benign')} benign, "
          f"{sum(1 for r in rows if r['label']=='malicious')} malicious")
    return rows


def main():
    args = parse_args()

    output_dir = PROJECT_ROOT / args.output_dir

    print(f"Loading dataset: notesbymuneeb/ai-waf-dataset")
    dataset = load_dataset("notesbymuneeb/ai-waf-dataset")

    # Map labels to integers
    label_map = {"benign": 0, "malicious": 1}

    def map_labels(example):
        example["label"] = label_map[example["label"]]
        return example

    dataset = dataset.map(map_labels)

    # --- Augmentation ---
    if args.augment:
        print("\nBuilding augmentation samples ...")
        aug_rows = _build_augmentation_samples()
        if aug_rows:
            aug_ds = Dataset.from_list([
                {"text": r["text"], "label": label_map[r["label"]]} for r in aug_rows
            ])
            # Match features (cast label column)
            from datasets import ClassLabel, Features, Value
            aug_ds = aug_ds.cast_column("label", dataset["train"].features["label"]
                                        if "label" in dataset["train"].features
                                        and hasattr(dataset["train"].features["label"], "names")
                                        else Value("int64"))
            dataset["train"] = concatenate_datasets([dataset["train"], aug_ds])
            print(f"  Combined train size after augmentation: {len(dataset['train'])}")

    # Cast label to ClassLabel for stratification
    from datasets import ClassLabel
    dataset = dataset.cast_column("label", ClassLabel(names=["benign", "malicious"]))

    # Split into train/val (90/10)
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "benign", 1: "malicious"},
        label2id={"benign": 0, "malicious": 1},
    )

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,  # Dynamic padding via DataCollator
        )

    print("Tokenizing dataset...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
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
        report_to="none",  # Disable wandb/tensorboard
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Evaluate
    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Save
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("\nDone! Model saved to:", output_dir)
    print("\nTo use in WAF, the model will be loaded from:", output_dir)
    if args.augment:
        print("  (trained with augmented data: extra benign + evasive malicious samples)")


if __name__ == "__main__":
    main()
