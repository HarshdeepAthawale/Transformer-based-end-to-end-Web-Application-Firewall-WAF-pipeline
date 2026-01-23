"""
Synthetic Dataset Generator Module

Generate synthetic benign requests for training
"""
import random
from typing import List, Optional
from loguru import logger
from pathlib import Path


class SyntheticDatasetGenerator:
    """Generate synthetic benign requests for training"""
    
    def __init__(self):
        self.methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
        self.paths = []
        self.query_params = []
        self.headers = []
        self.loaded = False
    
    def load_from_applications(self, log_paths: List[str], max_lines: int = 10000):
        """
        Load patterns from application logs
        
        Args:
            log_paths: List of log file paths
            max_lines: Maximum lines to process per file
        """
        try:
            from src.ingestion.ingestion import LogIngestionSystem
            from src.parsing.pipeline import ParsingPipeline
        except ImportError as e:
            logger.warning(f"Could not import ingestion/parsing modules: {e}")
            logger.info("Using default patterns instead")
            self._load_default_patterns()
            return
        
        pipeline = ParsingPipeline()
        ingestion = LogIngestionSystem()
        
        all_paths = set()
        all_query_params = set()
        
        for log_path in log_paths:
            if not Path(log_path).exists():
                logger.warning(f"Log file not found: {log_path}, skipping...")
                continue
            
            logger.info(f"Loading patterns from {log_path}")
            line_count = 0
            for log_line in ingestion.ingest_batch(log_path, max_lines=max_lines):
                normalized = pipeline.process_log_line(log_line)
                if normalized:
                    self._extract_patterns(normalized, all_paths, all_query_params)
                line_count += 1
                if line_count >= max_lines:
                    break
        
        # Store extracted patterns
        self.paths = list(all_paths) if all_paths else self._get_default_paths()
        self.query_params = list(all_query_params) if all_query_params else self._get_default_query_params()
        self.loaded = True
        
        logger.info(f"Loaded {len(self.paths)} unique paths and {len(self.query_params)} query patterns")
    
    def _extract_patterns(self, request_text: str, paths: set, query_params: set):
        """Extract request patterns from normalized text"""
        # Extract paths (simplified - look for patterns like "GET /path")
        parts = request_text.split()
        for i, part in enumerate(parts):
            if part in self.methods and i + 1 < len(parts):
                path_part = parts[i + 1]
                # Remove query parameters
                if '?' in path_part:
                    path_part = path_part.split('?')[0]
                if path_part.startswith('/'):
                    paths.add(path_part)
            
            # Extract query parameters
            if '=' in part:
                if '?' in part:
                    query_part = part.split('?')[-1]
                else:
                    query_part = part
                if '=' in query_part:
                    params = query_part.split('&')
                    for param in params:
                        if '=' in param:
                            key = param.split('=')[0]
                            query_params.add(key)
    
    def _load_default_patterns(self):
        """Load default patterns if log files are not available"""
        self.paths = self._get_default_paths()
        self.query_params = self._get_default_query_params()
        self.loaded = True
        logger.info("Using default patterns")
    
    def _get_default_paths(self) -> List[str]:
        """Get default path patterns"""
        return [
            "/api/users",
            "/api/products",
            "/api/orders",
            "/api/data",
            "/dashboard",
            "/login",
            "/logout",
            "/profile",
            "/home",
            "/index",
            "/about",
            "/contact",
            "/search",
            "/api/search",
            "/api/health",
            "/api/status"
        ]
    
    def _get_default_query_params(self) -> List[str]:
        """Get default query parameter patterns"""
        return [
            "page", "limit", "offset", "sort", "filter", "search",
            "id", "userId", "productId", "orderId", "category",
            "status", "type", "format", "q", "query"
        ]
    
    def generate_request(self) -> str:
        """Generate synthetic benign request"""
        method = random.choice(self.methods)
        path = self._generate_path()
        query = self._generate_query()
        
        request = f"{method} {path}"
        if query:
            request += f"?{query}"
        
        return request
    
    def _generate_path(self) -> str:
        """Generate synthetic path"""
        if self.paths:
            base_path = random.choice(self.paths)
            # Sometimes add sub-paths
            if random.random() < 0.3:
                sub_paths = ["/detail", "/list", "/view", "/edit", "/create"]
                base_path += random.choice(sub_paths)
            return base_path
        else:
            return random.choice(self._get_default_paths())
    
    def _generate_query(self) -> str:
        """Generate synthetic query string"""
        if not self.query_params:
            params = self._get_default_query_params()
        else:
            params = self.query_params
        
        num_params = random.randint(0, 3)
        if num_params == 0:
            return ""
        
        selected_params = random.sample(params, min(num_params, len(params)))
        query_parts = []
        
        for param in selected_params:
            if param in ["page", "limit", "offset"]:
                value = str(random.randint(1, 100))
            elif param in ["sort"]:
                value = random.choice(["asc", "desc", "name", "date"])
            elif param in ["filter", "status"]:
                value = random.choice(["active", "inactive", "pending", "completed"])
            elif param in ["search", "q", "query"]:
                value = random.choice(["test", "sample", "demo", "example"])
            else:
                value = str(random.randint(1, 1000))
            
            query_parts.append(f"{param}={value}")
        
        return "&".join(query_parts)
    
    def generate_dataset(self, num_samples: int) -> List[str]:
        """
        Generate dataset of synthetic requests
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            List of synthetic request strings
        """
        if not self.loaded:
            self._load_default_patterns()
        
        logger.info(f"Generating {num_samples} synthetic requests...")
        requests = []
        for i in range(num_samples):
            requests.append(self.generate_request())
            if (i + 1) % 1000 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} requests")
        
        logger.info(f"Generated {len(requests)} synthetic requests")
        return requests
