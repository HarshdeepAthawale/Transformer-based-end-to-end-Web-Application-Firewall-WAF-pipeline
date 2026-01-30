#!/usr/bin/env python3
"""
Fine-tune DistilBERT on the WAF dataset from HuggingFace.
Uses HuggingFace Trainer API for simplicity.

Dataset: notesbymuneeb/ai-waf-dataset
Model: distilbert-base-uncased -> binary classifier (benign/malicious)

Usage:
    python scripts/finetune_waf_model.py
    python scripts/finetune_waf_model.py --epochs 5 --batch-size 16
"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for WAF")
    parser.add_argument("--model-name", default="distilbert-base-uncased", help="Base model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output-dir", default="models/waf-distilbert", help="Output directory")
    return parser.parse_args()


def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, F1."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main():
    args = parse_args()

    # Project root
    project_root = Path(__file__).parent.parent
    output_dir = project_root / args.output_dir

    print(f"Loading dataset: notesbymuneeb/ai-waf-dataset")
    dataset = load_dataset("notesbymuneeb/ai-waf-dataset")

    # Map labels to integers
    label_map = {"benign": 0, "malicious": 1}

    def map_labels(example):
        example["label"] = label_map[example["label"]]
        return example

    dataset = dataset.map(map_labels)

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


if __name__ == "__main__":
    main()
