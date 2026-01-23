"""
Data Augmentation Module

Augment HTTP requests for better model generalization
"""
import random
from typing import List, Optional
from urllib.parse import urlparse, parse_qs, urlencode, unquote
from loguru import logger


class RequestAugmenter:
    """Augment HTTP requests for training"""
    
    def __init__(self, augmentation_prob: float = 0.5):
        """
        Initialize request augmenter
        
        Args:
            augmentation_prob: Probability of applying augmentation
        """
        self.augmentation_prob = augmentation_prob
    
    def augment(self, request_text: str) -> str:
        """
        Augment a single request
        
        Args:
            request_text: Original request text
        
        Returns:
            Augmented request text
        """
        if random.random() > self.augmentation_prob:
            return request_text
        
        # Choose random augmentation
        augmentations = [
            self._shuffle_query_params,
            self._vary_case,
            self._add_trailing_slash,
            self._remove_trailing_slash,
            self._normalize_path,
            self._reorder_headers,
        ]
        
        augmented = request_text
        for aug_func in random.sample(augmentations, k=random.randint(1, 2)):
            try:
                augmented = aug_func(augmented)
            except Exception as e:
                logger.debug(f"Augmentation failed: {e}")
                continue
        
        return augmented
    
    def augment_batch(self, requests: List[str]) -> List[str]:
        """
        Augment a batch of requests
        
        Args:
            requests: List of request texts
        
        Returns:
            List of augmented requests
        """
        return [self.augment(req) for req in requests]
    
    def _shuffle_query_params(self, request: str) -> str:
        """Shuffle query parameters"""
        try:
            # Parse request line
            parts = request.split(' ', 2)
            if len(parts) < 2:
                return request
            
            method = parts[0]
            url_part = parts[1] if len(parts) > 1 else '/'
            
            # Parse URL
            if '?' in url_part:
                path, query = url_part.split('?', 1)
                params = parse_qs(query, keep_blank_values=True)
                
                # Shuffle parameter order
                param_items = list(params.items())
                random.shuffle(param_items)
                
                # Rebuild query string
                new_query = urlencode(param_items, doseq=True)
                new_url = f"{path}?{new_query}"
            else:
                new_url = url_part
            
            # Reconstruct request
            if len(parts) > 2:
                return f"{method} {new_url} {parts[2]}"
            else:
                return f"{method} {new_url}"
        except Exception:
            return request
    
    def _vary_case(self, request: str) -> str:
        """Vary case of HTTP method"""
        parts = request.split(' ', 2)
        if len(parts) > 0:
            method = parts[0].upper()
            # Sometimes use lowercase
            if random.random() < 0.3:
                method = method.lower()
            parts[0] = method
        return ' '.join(parts)
    
    def _add_trailing_slash(self, request: str) -> str:
        """Add trailing slash to path"""
        parts = request.split(' ', 2)
        if len(parts) > 1:
            path = parts[1].split('?')[0]
            if not path.endswith('/') and '.' not in path.split('/')[-1]:
                parts[1] = parts[1].replace(path, path + '/', 1)
        return ' '.join(parts)
    
    def _remove_trailing_slash(self, request: str) -> str:
        """Remove trailing slash from path"""
        parts = request.split(' ', 2)
        if len(parts) > 1:
            path = parts[1].split('?')[0]
            if path.endswith('/') and len(path) > 1:
                parts[1] = parts[1].replace(path, path[:-1], 1)
        return ' '.join(parts)
    
    def _normalize_path(self, request: str) -> str:
        """Normalize path (remove //, resolve . and ..)"""
        parts = request.split(' ', 2)
        if len(parts) > 1:
            url_part = parts[1]
            if '?' in url_part:
                path, query = url_part.split('?', 1)
            else:
                path, query = url_part, ''
            
            # Basic normalization
            path = path.replace('//', '/')
            if query:
                parts[1] = f"{path}?{query}"
            else:
                parts[1] = path
        return ' '.join(parts)
    
    def _reorder_headers(self, request: str) -> str:
        """Reorder headers (if present in request)"""
        # For now, just return as-is
        # Headers are typically not in the request line
        return request


def augment_dataset(requests: List[str], num_augmentations: int = 1) -> List[str]:
    """
    Augment entire dataset
    
    Args:
        requests: List of request texts
        num_augmentations: Number of augmented versions per request
    
    Returns:
        List of original + augmented requests
    """
    augmenter = RequestAugmenter()
    augmented = list(requests)  # Start with originals
    
    for _ in range(num_augmentations):
        for request in requests:
            augmented.append(augmenter.augment(request))
    
    return augmented
