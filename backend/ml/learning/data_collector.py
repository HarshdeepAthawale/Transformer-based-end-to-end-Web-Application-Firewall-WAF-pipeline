"""
Incremental Data Collector

Collect new benign traffic for incremental learning.
"""
import json
from typing import List, Optional
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

from backend.ml.ingestion.ingestion import LogIngestionSystem
from backend.ml.parsing.pipeline import ParsingPipeline


class IncrementalDataCollector:
    """Collect new data for incremental learning"""
    
    def __init__(
        self,
        log_path: str,
        last_collection_time: Optional[datetime] = None,
        min_samples: int = 1000
    ):
        """
        Initialize data collector
        
        Args:
            log_path: Path to log file
            last_collection_time: Last collection timestamp
            min_samples: Minimum samples required for update
        """
        self.log_path = log_path
        self.last_collection_time = last_collection_time
        self.min_samples = min_samples
        self.ingestion = LogIngestionSystem()
        self.pipeline = ParsingPipeline()
    
    def collect_new_data(
        self,
        output_path: str,
        max_samples: Optional[int] = None,
        since_timestamp: Optional[datetime] = None
    ) -> List[str]:
        """
        Collect new data since last collection
        
        Args:
            output_path: Path to save collected data
            max_samples: Maximum samples to collect
            since_timestamp: Collect data since this timestamp
        
        Returns:
            List of normalized request strings
        """
        logger.info("Collecting new data for incremental learning...")
        
        texts = []
        collected_count = 0
        
        # Determine cutoff time
        cutoff_time = since_timestamp or self.last_collection_time
        
        # Collect from logs
        for log_line in self.ingestion.ingest_batch(self.log_path):
            # Check timestamp if available (simplified - would need proper log parsing)
            # For now, collect all new data
            
            # Process and normalize
            normalized = self.pipeline.process_log_line(log_line)
            if normalized:
                texts.append(normalized)
                collected_count += 1
                
                if max_samples and collected_count >= max_samples:
                    break
        
        # Check minimum samples
        if len(texts) < self.min_samples:
            logger.warning(
                f"Only collected {len(texts)} samples, minimum required: {self.min_samples}"
            )
            return []
        
        # Save collected data
        if texts:
            self._save_data(texts, output_path)
            logger.info(f"Collected {len(texts)} new samples")
        
        return texts
    
    def _save_data(self, texts: List[str], output_path: str):
        """Save collected data"""
        path_obj = Path(output_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(texts, f, indent=2)
        
        logger.info(f"Saved data to {output_path}")
    
    def load_collected_data(self, data_path: str) -> List[str]:
        """Load previously collected data"""
        with open(data_path, 'r') as f:
            return json.load(f)
