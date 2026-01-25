"""
Data Loader Module for Training

PyTorch DataLoader for HTTP request sequences.
"""
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from .tokenizer import HTTPTokenizer
from .sequence_prep import SequencePreparator


class HTTPRequestDataset(Dataset):
    """Dataset for HTTP requests"""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: HTTPTokenizer,
        max_length: int = 512,
        labels: Optional[List[float]] = None
    ):
        """
        Initialize dataset
        
        Args:
            texts: List of normalized request strings
            tokenizer: HTTPTokenizer instance
            max_length: Maximum sequence length
            labels: Optional labels (for supervised learning, not used in anomaly detection)
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels if labels is not None else [0.0] * len(texts)  # All benign (0.0)
        
        self.preparator = SequencePreparator(tokenizer)
        
        if len(self.texts) != len(self.labels):
            raise ValueError(f"Texts ({len(self.texts)}) and labels ({len(self.labels)}) must have same length")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Prepare sequence
        token_ids, attention_mask = self.preparator.prepare_sequence(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True
        )
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float)
        }


def create_dataloader(
    texts: List[str],
    tokenizer: HTTPTokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
    labels: Optional[List[float]] = None
) -> DataLoader:
    """
    Create a DataLoader for HTTP requests
    
    Args:
        texts: List of normalized request strings
        tokenizer: HTTPTokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        labels: Optional labels
    
    Returns:
        DataLoader instance
    """
    dataset = HTTPRequestDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
        labels=labels
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
