"""
Dataset Generator Module

Generate training, validation, and test datasets from normalized requests.
"""
import json
from typing import List, Tuple
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split


class DatasetGenerator:
    """Generate datasets for training"""
    
    @staticmethod
    def load_data(data_path: str) -> List[str]:
        """Load normalized request strings from file"""
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'requests' in data:
            return data['requests']
        else:
            raise ValueError(f"Unexpected data format in {data_path}")
    
    @staticmethod
    def split_data(
        texts: List[str],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Split data into train/val/test sets
        
        Args:
            texts: List of normalized request strings
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_state: Random seed
        
        Returns:
            Tuple of (train_texts, val_texts, test_texts)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        logger.info(f"Splitting {len(texts)} samples into train/val/test sets...")
        
        # First split: train vs (val + test)
        train_texts, temp_texts = train_test_split(
            texts,
            test_size=(val_ratio + test_ratio),
            random_state=random_state
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_texts, test_texts = train_test_split(
            temp_texts,
            test_size=(1 - val_size),
            random_state=random_state
        )
        
        logger.info(f"Train: {len(train_texts)} samples")
        logger.info(f"Validation: {len(val_texts)} samples")
        logger.info(f"Test: {len(test_texts)} samples")
        
        return train_texts, val_texts, test_texts
    
    @staticmethod
    def save_datasets(
        train_texts: List[str],
        val_texts: List[str],
        test_texts: List[str],
        output_dir: str
    ):
        """Save datasets to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save training set
        train_path = output_path / "train.json"
        with open(train_path, 'w') as f:
            json.dump(train_texts, f, indent=2)
        logger.info(f"Training set saved to {train_path}")
        
        # Save validation set
        val_path = output_path / "val.json"
        with open(val_path, 'w') as f:
            json.dump(val_texts, f, indent=2)
        logger.info(f"Validation set saved to {val_path}")
        
        # Save test set
        test_path = output_path / "test.json"
        with open(test_path, 'w') as f:
            json.dump(test_texts, f, indent=2)
        logger.info(f"Test set saved to {test_path}")
    
    @staticmethod
    def generate_from_logs(
        log_path: str,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        max_samples: int = None
    ):
        """
        Generate datasets directly from log file
        
        Args:
            log_path: Path to log file
            output_dir: Output directory for datasets
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            max_samples: Maximum number of samples to process
        """
        from backend.ml.ingestion.ingestion import LogIngestionSystem
        from backend.ml.parsing.pipeline import ParsingPipeline
        
        logger.info(f"Generating datasets from {log_path}...")
        
        # Initialize pipeline
        ingestion = LogIngestionSystem()
        parser = ParsingPipeline()
        
        # Process logs
        texts = []
        for log_line in ingestion.ingest_batch(log_path, max_lines=max_samples):
            normalized = parser.process_log_line(log_line)
            if normalized:
                texts.append(normalized)
        
        logger.info(f"Processed {len(texts)} normalized requests")
        
        # Split data
        train_texts, val_texts, test_texts = DatasetGenerator.split_data(
            texts,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        # Save datasets
        DatasetGenerator.save_datasets(
            train_texts,
            val_texts,
            test_texts,
            output_dir
        )
