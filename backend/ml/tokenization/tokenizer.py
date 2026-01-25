"""
HTTP-Aware Tokenizer for Anomaly Detection

Tokenizes HTTP requests with special handling for HTTP components.
Uses subword tokenization with character-level fallback for unknown tokens.
"""
from typing import List, Dict, Optional, Set
from collections import Counter
import json
import re
from pathlib import Path
from loguru import logger


class HTTPTokenizer:
    """Tokenizer optimized for HTTP request anomaly detection"""
    
    # Special tokens
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    
    # HTTP-specific tokens
    HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
    HTTP_HEADERS = [
        "content-type", "user-agent", "accept", "authorization", "cookie",
        "referer", "host", "connection", "cache-control", "accept-language"
    ]
    
    def __init__(
        self,
        vocab_size: int = 10000,
        min_frequency: int = 2,
        max_length: int = 512
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.max_length = max_length
        
        # Vocabulary
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_built = False
        
        # Initialize special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Initialize special tokens in vocabulary"""
        special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.CLS_TOKEN,
            self.SEP_TOKEN
        ]
        
        for i, token in enumerate(special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
    
    def build_vocab(self, texts: List[str], save_path: Optional[str] = None):
        """Build vocabulary from training texts"""
        logger.info(f"Building vocabulary from {len(texts)} texts...")
        
        # Tokenize all texts and count tokens
        all_tokens: List[str] = []
        
        for text in texts:
            tokens = self._tokenize_text(text)
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        logger.info(f"Found {len(token_counts)} unique tokens")
        
        # Filter by minimum frequency
        filtered_tokens = {
            token: count for token, count in token_counts.items()
            if count >= self.min_frequency
        }
        logger.info(f"After filtering (min_freq={self.min_frequency}): {len(filtered_tokens)} tokens")
        
        # Sort by frequency (descending)
        sorted_tokens = sorted(
            filtered_tokens.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Add tokens to vocabulary (starting after special tokens)
        start_id = len(self.word_to_id)
        for token, count in sorted_tokens:
            if len(self.word_to_id) >= self.vocab_size:
                break
            token_id = start_id + len(self.word_to_id) - start_id
            self.word_to_id[token] = token_id
            self.id_to_word[token_id] = token
        
        self.vocab_built = True
        logger.info(f"Vocabulary built: {len(self.word_to_id)} tokens")
        
        # Save if path provided
        if save_path:
            self.save_vocab(save_path)
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize a single text into tokens"""
        tokens = []
        
        # Split by common delimiters while preserving structure
        # Split on spaces, but keep HTTP components together
        parts = re.split(r'[\s|]+', text)
        
        for part in parts:
            if not part:
                continue
            
            # Check if it's an HTTP method
            if part.upper() in self.HTTP_METHODS:
                tokens.append(part.upper())
                continue
            
            # Check if it's a header (format: "header:value")
            if ':' in part:
                header_parts = part.split(':', 1)
                if len(header_parts) == 2:
                    header_name = header_parts[0].lower()
                    header_value = header_parts[1]
                    
                    # Add header name as token
                    if header_name in self.HTTP_HEADERS:
                        tokens.append(f"HEADER:{header_name}")
                    else:
                        tokens.append("HEADER:OTHER")
                    
                    # Tokenize header value
                    tokens.extend(self._tokenize_value(header_value))
                    continue
            
            # Check for query parameters (format: "key=value")
            if '=' in part:
                param_parts = part.split('=', 1)
                if len(param_parts) == 2:
                    param_key = param_parts[0]
                    param_value = param_parts[1]
                    
                    tokens.append(f"PARAM:{param_key}")
                    tokens.extend(self._tokenize_value(param_value))
                    continue
            
            # Check for path segments
            if part.startswith('/'):
                # Tokenize path segments
                path_parts = part.split('/')
                for path_part in path_parts:
                    if path_part:
                        tokens.extend(self._tokenize_value(path_part))
                continue
            
            # Check for special markers
            if part.startswith('BODY:'):
                tokens.append("BODY_START")
                body_content = part[5:]
                tokens.extend(self._tokenize_value(body_content))
                continue
            
            # Default: tokenize as value
            tokens.extend(self._tokenize_value(part))
        
        return tokens
    
    def _tokenize_value(self, value: str) -> List[str]:
        """Tokenize a value (parameter value, header value, body, etc.)"""
        if not value:
            return []
        
        tokens = []
        
        # Try to split on common separators first
        # Split on non-alphanumeric characters but keep them
        parts = re.split(r'([^a-zA-Z0-9]+)', value)
        
        for part in parts:
            if not part:
                continue
            
            # If it's alphanumeric, try subword tokenization
            if part.isalnum():
                # For short tokens, add as-is
                if len(part) <= 4:
                    tokens.append(part.lower())
                else:
                    # For longer tokens, split into subwords
                    # Simple approach: split into chunks of 3-4 characters
                    chunk_size = 4
                    for i in range(0, len(part), chunk_size):
                        chunk = part[i:i+chunk_size].lower()
                        tokens.append(chunk)
            else:
                # Non-alphanumeric: add as single token
                tokens.append(part)
        
        return tokens
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None
    ) -> List[int]:
        """Encode text to token IDs"""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        # Tokenize
        tokens = self._tokenize_text(text)
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.CLS_TOKEN] + tokens + [self.SEP_TOKEN]
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.word_to_id:
                token_ids.append(self.word_to_id[token])
            else:
                # Unknown token
                token_ids.append(self.word_to_id[self.UNK_TOKEN])
        
        # Truncate if needed
        max_len = max_length or self.max_length
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                token = self.id_to_word[token_id]
                if skip_special_tokens and token in [self.PAD_TOKEN, self.CLS_TOKEN, self.SEP_TOKEN]:
                    continue
                tokens.append(token)
            else:
                tokens.append(self.UNK_TOKEN)
        
        return " ".join(tokens)
    
    def save_vocab(self, path: str):
        """Save vocabulary to file"""
        vocab_data = {
            'word_to_id': self.word_to_id,
            'id_to_word': {str(k): v for k, v in self.id_to_word.items()},
            'vocab_size': len(self.word_to_id),
            'max_length': self.max_length,
            'min_frequency': self.min_frequency
        }
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        logger.info(f"Vocabulary saved to {path}")
    
    def load_vocab(self, path: str):
        """Load vocabulary from file"""
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        
        self.word_to_id = vocab_data['word_to_id']
        # Convert string keys back to int for id_to_word
        self.id_to_word = {int(k): v for k, v in vocab_data['id_to_word'].items()}
        self.max_length = vocab_data.get('max_length', 512)
        self.min_frequency = vocab_data.get('min_frequency', 2)
        self.vocab_built = True
        
        logger.info(f"Vocabulary loaded from {path}: {len(self.word_to_id)} tokens")
