"""
Data Loader Module

PyTorch Dataset and DataLoader for HTTP requests
"""
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import torch
from .tokenizer import HTTPTokenizer
from .sequence_prep import SequencePreparator


class HTTPRequestDataset(Dataset):
    """Dataset for HTTP requests"""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: HTTPTokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preparator = SequencePreparator(tokenizer)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        token_ids, attention_mask = self.preparator.prepare_sequence(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True
        )
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'text': text
        }


def create_dataloader(
    texts: List[str],
    tokenizer: HTTPTokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 0  # Set to 0 to avoid multiprocessing issues
) -> DataLoader:
    """Create DataLoader for training"""
    dataset = HTTPRequestDataset(texts, tokenizer, max_length)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
