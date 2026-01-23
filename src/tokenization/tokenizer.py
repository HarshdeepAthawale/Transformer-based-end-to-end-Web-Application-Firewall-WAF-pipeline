"""
HTTP Tokenizer Module

Tokenizes HTTP requests for Transformer models
"""
from typing import List, Dict, Optional
from collections import Counter
import json
from pathlib import Path
from loguru import logger
import re


class HTTPTokenizer:
    """Tokenizer for HTTP requests optimized for anomaly detection"""
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    CLS_TOKEN = "<CLS>"
    SEP_TOKEN = "<SEP>"
    MASK_TOKEN = "<MASK>"
    
    # HTTP-specific tokens
    METHOD_TOKENS = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
    STATUS_TOKENS = ["200", "404", "500", "403", "401"]
    
    def __init__(
        self,
        vocab_size: int = 10000,
        min_frequency: int = 2,
        max_length: int = 512
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.max_length = max_length
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_built = False
    
    def _initialize_special_tokens(self):
        """Initialize special tokens"""
        special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.CLS_TOKEN,
            self.SEP_TOKEN,
            self.MASK_TOKEN
        ]
        
        # Add HTTP method tokens
        special_tokens.extend(self.METHOD_TOKENS)
        
        # Add normalization placeholders
        special_tokens.extend([
            "<UUID>", "<TIMESTAMP>", "<SESSION_ID>", "<NUMERIC_ID>",
            "<EMAIL>", "<IP_ADDRESS>", "<CARD_NUMBER>", "<BASE64>",
            "<JWT_TOKEN>", "<API_KEY>", "<ID>", "<SENSITIVE>", "<DOMAIN>",
            "<VERSION>"
        ])
        
        return special_tokens
    
    def build_vocab(self, texts: List[str], save_path: Optional[str] = None):
        """Build vocabulary from training texts"""
        logger.info("Building vocabulary...")
        
        # Initialize with special tokens
        special_tokens = self._initialize_special_tokens()
        token_counter = Counter()
        
        # Tokenize all texts and count tokens
        for text in texts:
            tokens = self._tokenize_text(text)
            token_counter.update(tokens)
        
        # Build vocabulary
        self.word_to_id = {}
        self.id_to_word = {}
        
        # Add special tokens first
        token_id = 0
        for token in special_tokens:
            self.word_to_id[token] = token_id
            self.id_to_word[token_id] = token
            token_id += 1
        
        # Add most frequent tokens
        sorted_tokens = sorted(
            token_counter.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for token, count in sorted_tokens:
            if token not in self.word_to_id and count >= self.min_frequency:
                if len(self.word_to_id) >= self.vocab_size:
                    break
                self.word_to_id[token] = token_id
                self.id_to_word[token_id] = token
                token_id += 1
        
        self.vocab_built = True
        logger.info(f"Vocabulary built: {len(self.word_to_id)} tokens")
        
        # Save vocabulary
        if save_path:
            self.save_vocab(save_path)
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into subwords and characters"""
        tokens = []
        
        # Split by common delimiters
        parts = re.split(r'([\s|:;=/?&<>(){}[\]"\',])', text)
        
        for part in parts:
            if not part.strip():
                continue
            
            # If part is a delimiter, add as token
            if re.match(r'^[\s|:;=/?&<>(){}[\]"\',]$', part):
                tokens.append(part)
            else:
                # Try to split into subwords
                subwords = self._split_into_subwords(part)
                tokens.extend(subwords)
        
        return tokens
    
    def _split_into_subwords(self, word: str) -> List[str]:
        """Split word into subwords using common patterns"""
        if len(word) <= 3:
            return [word]
        
        # Split camelCase
        if re.match(r'^[a-z]+[A-Z]', word):
            parts = re.findall(r'[a-z]+|[A-Z][a-z]*', word)
            return [p.lower() for p in parts]
        
        # Split by underscores/hyphens
        if '_' in word or '-' in word:
            return re.split(r'[_-]', word)
        
        # Split long words into chunks
        if len(word) > 10:
            chunks = []
            for i in range(0, len(word), 5):
                chunks.append(word[i:i+5])
            return chunks
        
        return [word]
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        tokens = self._tokenize_text(text)
        token_ids = []
        
        # Add CLS token
        if add_special_tokens:
            token_ids.append(self.word_to_id.get(self.CLS_TOKEN, 0))
        
        # Convert tokens to IDs
        for token in tokens:
            token_id = self.word_to_id.get(token, self.word_to_id.get(self.UNK_TOKEN, 0))
            token_ids.append(token_id)
        
        # Add SEP token
        if add_special_tokens:
            token_ids.append(self.word_to_id.get(self.SEP_TOKEN, 0))
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        tokens = []
        for token_id in token_ids:
            token = self.id_to_word.get(token_id, self.UNK_TOKEN)
            if skip_special_tokens and token in [self.PAD_TOKEN, self.CLS_TOKEN, self.SEP_TOKEN]:
                continue
            tokens.append(token)
        return " ".join(tokens)
    
    def save_vocab(self, path: str):
        """Save vocabulary to file"""
        vocab_data = {
            'word_to_id': self.word_to_id,
            'id_to_word': {int(k): v for k, v in self.id_to_word.items()},
            'vocab_size': len(self.word_to_id),
            'max_length': self.max_length
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        logger.info(f"Vocabulary saved to {path}")
    
    def load_vocab(self, path: str):
        """Load vocabulary from file"""
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        
        self.word_to_id = vocab_data['word_to_id']
        self.id_to_word = {int(k): v for k, v in vocab_data['id_to_word'].items()}
        self.max_length = vocab_data.get('max_length', 512)
        self.vocab_built = True
        
        logger.info(f"Vocabulary loaded from {path}: {len(self.word_to_id)} tokens")
