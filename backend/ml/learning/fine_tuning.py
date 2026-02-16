"""Incremental fine-tuning pipeline using HuggingFace Trainer."""

import json
from pathlib import Path
from typing import List, Optional

from loguru import logger


class IncrementalFineTuner:
    """
    Fine-tune WAF model on incremental benign data.
    Uses scripts/finetune_waf_model.py with a custom dataset of benign texts.
    """

    def __init__(
        self,
        base_model_path: str = "models/waf-distilbert",
        output_dir: str = "models/waf-distilbert-incremental",
        epochs: int = 2,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ):
        self.base_model_path = Path(base_model_path)
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def fine_tune(
        self,
        train_texts: List[str],
        output_path: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Fine-tune on new benign data.
        Creates a temporary dataset and runs HuggingFace training.
        Returns path to saved model or None on failure.
        """
        if len(train_texts) < 50:
            logger.warning(f"Insufficient data: {len(train_texts)} samples (min 50)")
            return None

        out = Path(output_path) if output_path else self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        # Create dataset in HF format (benign=0)
        data_path = out / "incremental_train.jsonl"
        with open(data_path, "w") as f:
            for text in train_texts:
                row = {"text": text, "label": 0}
                f.write(json.dumps(row) + "\n")

        logger.info(f"Fine-tuning on {len(train_texts)} samples...")
        try:
            from datasets import load_dataset
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                TrainingArguments,
                Trainer,
                DataCollatorWithPadding,
            )
        except ImportError as e:
            logger.error(f"Missing deps for fine-tuning: {e}")
            return None

        try:
            ds = load_dataset("json", data_files=str(data_path), split="train")
            tokenizer = AutoTokenizer.from_pretrained(str(self.base_model_path))
            model = AutoModelForSequenceClassification.from_pretrained(
                str(self.base_model_path),
                num_labels=2,
            )

            def tokenize(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=512,
                    padding=False,
                )

            ds = ds.map(tokenize, batched=True, remove_columns=["text"])
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            training_args = TrainingArguments(
                output_dir=str(out / "checkpoints"),
                num_train_epochs=self.epochs,
                per_device_train_batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                save_strategy="epoch",
                logging_steps=20,
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=ds,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
            trainer.train()
            trainer.save_model(str(out))
            tokenizer.save_pretrained(str(out))
            logger.info(f"Fine-tuning complete, model saved to {out}")
            return out
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return None
