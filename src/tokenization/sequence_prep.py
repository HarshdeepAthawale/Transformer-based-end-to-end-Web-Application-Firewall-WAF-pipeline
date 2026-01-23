"""
Sequence Preparation Module

Prepares token sequences for model input with padding and truncation
"""
from typing import List, Tuple, Dict, Optional
import torch
import numpy as np
from .tokenizer import HTTPTokenizer


class SequencePreparator:
    """Prepare sequences for model input"""
    
    def __init__(self, tokenizer: HTTPTokenizer):
        self.tokenizer = tokenizer
    
    def prepare_sequence(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True
    ) -> Tuple[List[int], List[int]]:
        """Prepare single sequence with attention mask"""
        max_len = max_length or self.tokenizer.max_length
        
        # Encode text
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate if needed
        if truncation and len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        
        # Pad if needed
        if padding:
            pad_length = max_len - len(token_ids)
            pad_id = self.tokenizer.word_to_id.get(self.tokenizer.PAD_TOKEN, 0)
            token_ids.extend([pad_id] * pad_length)
            attention_mask.extend([0] * pad_length)
        
        return token_ids, attention_mask
    
    def prepare_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Prepare batch of sequences"""
        max_len = max_length or self.tokenizer.max_length
        
        batch_token_ids = []
        batch_attention_masks = []
        
        for text in texts:
            token_ids, attention_mask = self.prepare_sequence(
                text,
                max_length=max_len,
                padding=True,
                truncation=True
            )
            batch_token_ids.append(token_ids)
            batch_attention_masks.append(attention_mask)
        
        # Convert to tensors
        if return_tensors == "pt":
            return {
                'input_ids': torch.tensor(batch_token_ids, dtype=torch.long),
                'attention_mask': torch.tensor(batch_attention_masks, dtype=torch.long)
            }
        else:
            return {
                'input_ids': np.array(batch_token_ids),
                'attention_mask': np.array(batch_attention_masks)
            }
