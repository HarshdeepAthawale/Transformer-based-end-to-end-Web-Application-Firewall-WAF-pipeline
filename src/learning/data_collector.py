"""
Incremental Data Collector Module

Collects new data from logs for incremental learning
"""
from typing import List, Optional, Iterator
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import json
import os

from src.ingestion.ingestion import LogIngestionSystem
from src.parsing.pipeline import ParsingPipeline


class IncrementalDataCollector:
    """Collect new data for incremental learning"""
    
    def __init__(
        self,
        log_path: str,
        last_collection_time: Optional[datetime] = None,
        min_samples: int = 1000
    ):
        """
        Initialize incremental data collector
        
        Args:
            log_path: Path to log file or directory
            last_collection_time: Last time data was collected (for filtering)
            min_samples: Minimum samples required for collection
        """
        self.log_path = log_path
        self.last_collection_time = last_collection_time
        self.min_samples = min_samples
        self.ingestion = LogIngestionSystem()
        self.pipeline = ParsingPipeline()
        
        logger.info(f"IncrementalDataCollector initialized: log_path={log_path}, min_samples={min_samples}")
    
    def collect_new_data(
        self,
        output_path: str,
        max_samples: Optional[int] = None,
        since_time: Optional[datetime] = None
    ) -> List[str]:
        """
        Collect new data since last collection
        
        Args:
            output_path: Path to save collected data
            max_samples: Maximum samples to collect (None = no limit)
            since_time: Collect data since this time (overrides last_collection_time)
        
        Returns:
            List of normalized request texts
        """
        logger.info("Collecting new data for incremental learning...")
        
        texts = []
        collected_count = 0
        skipped_count = 0
        
        # Use since_time if provided, otherwise use last_collection_time
        collection_since = since_time or self.last_collection_time
        
        try:
            # Collect from logs
            if os.path.isfile(self.log_path):
                # Single log file
                log_files = [self.log_path]
            elif os.path.isdir(self.log_path):
                # Directory of log files
                log_files = [
                    str(f) for f in Path(self.log_path).glob("*.log")
                ] + [
                    str(f) for f in Path(self.log_path).glob("*.txt")
                ]
            else:
                logger.warning(f"Log path does not exist: {self.log_path}")
                return []
            
            for log_file in log_files:
                logger.info(f"Processing log file: {log_file}")
                
                for log_line in self.ingestion.ingest_batch(log_file):
                    # Check timestamp if available
                    if collection_since:
                        # Try to extract timestamp from log line
                        # This is a simplified version - in production, parse actual log format
                        try:
                            # Common log formats: [timestamp] or timestamp at start
                            # For now, we'll collect all if timestamp parsing fails
                            pass  # TODO: Implement timestamp parsing based on log format
                        except:
                            pass
                    
                    # Process and normalize
                    try:
                        normalized = self.pipeline.process_log_line(log_line)
                        if normalized:
                            texts.append(normalized)
                            collected_count += 1
                            
                            if max_samples and collected_count >= max_samples:
                                logger.info(f"Reached max_samples limit: {max_samples}")
                                break
                        else:
                            skipped_count += 1
                    except Exception as e:
                        logger.debug(f"Error processing log line: {e}")
                        skipped_count += 1
                
                if max_samples and collected_count >= max_samples:
                    break
        
        except Exception as e:
            logger.error(f"Error collecting data: {e}")
            import traceback
            traceback.print_exc()
        
        # Save collected data
        if texts:
            self._save_data(texts, output_path)
            logger.info(f"Collected {len(texts)} new samples (skipped {skipped_count} invalid)")
        else:
            logger.warning(f"No valid data collected (skipped {skipped_count} invalid)")
        
        # Update last collection time
        self.last_collection_time = datetime.now()
        
        return texts
    
    def _save_data(self, texts: List[str], output_path: str):
        """
        Save collected data to file
        
        Args:
            texts: List of normalized request texts
            output_path: Path to save data
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(texts, f, indent=2)
        
        logger.info(f"Saved {len(texts)} samples to {output_path}")
        
        # Also save metadata
        metadata_path = output_path.with_suffix('.meta.json')
        metadata = {
            'count': len(texts),
            'collected_at': datetime.now().isoformat(),
            'last_collection_time': self.last_collection_time.isoformat() if self.last_collection_time else None
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_collected_data(self, data_path: str) -> List[str]:
        """
        Load previously collected data
        
        Args:
            data_path: Path to saved data file
        
        Returns:
            List of normalized request texts
        """
        data_path = Path(data_path)
        if not data_path.exists():
            logger.warning(f"Data file not found: {data_path}")
            return []
        
        with open(data_path, 'r') as f:
            texts = json.load(f)
        
        logger.info(f"Loaded {len(texts)} samples from {data_path}")
        return texts
    
    def get_collection_stats(self, data_path: str) -> dict:
        """
        Get statistics about collected data
        
        Args:
            data_path: Path to saved data file
        
        Returns:
            Dictionary with statistics
        """
        metadata_path = Path(data_path).with_suffix('.meta.json')
        
        if not metadata_path.exists():
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
