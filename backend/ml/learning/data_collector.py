"""Incremental data collector for continuous learning from benign traffic logs."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from loguru import logger

from backend.ingestion.batch_reader import read_chunks
from backend.parsing.pipeline import ParsingPipeline


class IncrementalDataCollector:
    """Collect new benign request data from logs for incremental learning."""

    def __init__(
        self,
        log_path: str,
        last_collection_time: Optional[datetime] = None,
        min_samples: int = 100,
    ):
        self.log_path = Path(log_path)
        self.last_collection_time = last_collection_time
        self.min_samples = min_samples
        self._pipeline = ParsingPipeline(
            include_headers=True,
            include_body=True,
        )

    def collect_new_data(
        self,
        output_path: str,
        max_samples: Optional[int] = None,
    ) -> List[str]:
        """Collect and normalize log lines, save to JSON. Returns list of serialized request texts."""
        logger.info("Collecting new data for incremental learning...")

        if not self.log_path.exists():
            logger.warning(f"Log path not found: {self.log_path}")
            return []

        texts: List[str] = []
        for chunk in read_chunks(
            str(self.log_path),
            chunk_size=500,
            max_lines=max_samples or 50000,
        ):
            for line in chunk:
                line = line.strip()
                if not line:
                    continue
                try:
                    normalized = self._pipeline.process(line)
                    if normalized:
                        texts.append(normalized)
                        if max_samples and len(texts) >= max_samples:
                            break
                except Exception as e:
                    logger.debug(f"Skipping line: {e}")
            if max_samples and len(texts) >= max_samples:
                break

        if texts:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(texts, f, indent=2)
            logger.info(f"Collected {len(texts)} samples to {output_path}")

        return texts

    def load_collected_data(self, data_path: str) -> List[str]:
        """Load previously collected data from JSON."""
        with open(data_path) as f:
            return json.load(f)
