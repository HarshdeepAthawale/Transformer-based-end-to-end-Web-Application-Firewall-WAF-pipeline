#!/usr/bin/env python3
"""
Fine-tune DistilBERT for 8-class WAF attack classification.

Dataset: data/training/multiclass_dataset.json (built by build_multiclass_dataset.py)
Model: distilbert-base-uncased -> 8-class classifier

Classes:
  0: benign           4: path_traversal
  1: sqli             5: xxe
  2: xss              6: ssrf
  3: rce              7: other_attack

Usage:
    python scripts/build_multiclass_dataset.py          # build dataset first
    python scripts/finetune_multiclass.py               # train
    python scripts/finetune_multiclass.py --epochs 5 --batch-size 16
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import numpy as np
from datasets import Dataset, ClassLabel
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
    classification_report,
    confusion_matrix,
)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_PATH = PROJECT_ROOT / "data" / "training" / "multiclass_dataset.json"

LABEL_NAMES = [
    "benign",
    "sqli",
    "xss",
    "rce",
    "path_traversal",
    "xxe",
    "ssrf",
    "other_attack",
]
NUM_LABELS = len(LABEL_NAMES)
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for 8-class WAF")
    parser.add_argument("--model-name", default="distilbert-base-uncased", help="Base model")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--output-dir",
        default="models/waf-distilbert-multiclass",
        help="Output directory",
    )
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    return parser.parse_args()


def compute_metrics(eval_pred):
    """Compute weighted accuracy, precision, recall, F1 for multi-class."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, predictions)

    # Per-class macro F1 (useful for monitoring minority classes)
    _, _, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,
    }


class WeightedTrainer(Trainer):
    """Trainer with class-weight support to handle imbalanced datasets."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir

    # Load dataset
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        print("Run: python scripts/build_multiclass_dataset.py")
        return

    print(f"Loading dataset from {DATASET_PATH} ...")
    with open(DATASET_PATH) as f:
        raw_data = json.load(f)

    print(f"  Total samples: {len(raw_data)}")

    # Build HuggingFace dataset
    texts = [row["text"] for row in raw_data]
    labels = [row["label_id"] for row in raw_data]

    ds = Dataset.from_dict({"text": texts, "label": labels})
    ds = ds.cast_column("label", ClassLabel(names=LABEL_NAMES))

    # Print class distribution
    label_counts = Counter(labels)
    print("\nClass distribution:")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {i}: {name:<20s} {label_counts.get(i, 0):>6d}")

    # Compute class weights (inverse frequency, capped)
    total = sum(label_counts.values())
    class_weights = []
    for i in range(NUM_LABELS):
        count = label_counts.get(i, 1)
        weight = total / (NUM_LABELS * count)
        # Cap weight to avoid extreme values for very rare classes
        weight = min(weight, 10.0)
        class_weights.append(weight)
    print("\nClass weights:")
    for i, (name, w) in enumerate(zip(LABEL_NAMES, class_weights)):
        print(f"  {name:<20s} {w:.2f}")

    # Split: train / val / test
    test_ratio = args.test_split
    val_ratio = args.val_split

    # First split off test set
    split1 = ds.train_test_split(test_size=test_ratio, seed=42, stratify_by_column="label")
    train_val = split1["train"]
    test_dataset = split1["test"]

    # Then split train_val into train and val
    val_relative = val_ratio / (1.0 - test_ratio)
    split2 = train_val.train_test_split(test_size=val_relative, seed=42, stratify_by_column="label")
    train_dataset = split2["train"]
    val_dataset = split2["test"]

    print(f"\nSplits: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # Load tokenizer and model
    print(f"\nLoading base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,
        )

    print("Tokenizing ...")
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    test_dataset_tok = test_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

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
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    # Trainer with class weights
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training ...")
    trainer.train()

    # Evaluate on validation set
    print("\n--- Validation Results ---")
    val_metrics = trainer.evaluate()
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Evaluate on held-out test set
    print("\n--- Test Set Results ---")
    test_metrics = trainer.evaluate(test_dataset_tok)
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Detailed per-class report on test set
    print("\n--- Per-Class Classification Report (Test Set) ---")
    test_preds = trainer.predict(test_dataset_tok)
    y_true = test_preds.label_ids
    y_pred = np.argmax(test_preds.predictions, axis=-1)

    print(classification_report(
        y_true, y_pred,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    ))

    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    # Header
    header = "           " + " ".join(f"{name[:6]:>6s}" for name in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{v:>6d}" for v in row)
        print(f"  {LABEL_NAMES[i]:<10s} {row_str}")

    # Save model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training metadata
    metadata = {
        "base_model": args.model_name,
        "num_labels": NUM_LABELS,
        "label_names": LABEL_NAMES,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "epochs": args.epochs,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "class_weights": class_weights,
        "class_distribution": {LABEL_NAMES[i]: label_counts.get(i, 0) for i in range(NUM_LABELS)},
        "final_val_metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in val_metrics.items()},
        "final_test_metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in test_metrics.items()},
        "per_class_report": classification_report(
            y_true, y_pred,
            target_names=LABEL_NAMES,
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
    }

    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Training metadata saved to {metadata_path}")
    print("\nDone!")
    print(f"Model directory: {output_dir}")
    print(f"To use: set WAF_MODEL_PATH={output_dir} or update waf_classifier.py")


if __name__ == "__main__":
    main()
