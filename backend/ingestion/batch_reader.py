"""
Batch Log Reader

Chunk-based file reading for log ingestion.
Supports plain text and gzip-compressed files.
"""

import gzip
import logging
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


def read_chunks(
    log_path: str,
    chunk_size: int = 1000,
    max_lines: Optional[int] = None,
    skip_lines: int = 0,
    encoding: str = "utf-8",
) -> Iterator[list[str]]:
    """Read a log file in chunks, yielding lists of lines.

    Args:
        log_path: Path to the log file (.log, .txt, .gz).
        chunk_size: Number of lines per chunk.
        max_lines: Maximum total lines to read (None = unlimited).
        skip_lines: Number of lines to skip from the beginning.
        encoding: File encoding.

    Yields:
        Lists of stripped, non-empty log lines (up to chunk_size each).

    Raises:
        FileNotFoundError: If the log file does not exist.
        ValueError: If chunk_size < 1.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    opener = gzip.open if path.suffix == ".gz" else open
    total_read = 0
    skipped = 0

    logger.info(
        "Reading %s (chunk_size=%d, max_lines=%s, skip_lines=%d)",
        log_path, chunk_size, max_lines, skip_lines,
    )

    try:
        with opener(path, "rt", encoding=encoding) as f:
            chunk: list[str] = []

            for raw_line in f:
                # Skip initial lines
                if skipped < skip_lines:
                    skipped += 1
                    continue

                # Check max_lines limit
                if max_lines is not None and total_read >= max_lines:
                    break

                stripped = raw_line.strip()
                if not stripped:
                    continue

                chunk.append(stripped)
                total_read += 1

                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []

            # Yield remaining lines
            if chunk:
                yield chunk

    except Exception as e:
        logger.error("Error reading %s: %s", log_path, e)
        raise

    logger.info("Finished reading %s: %d lines total", log_path, total_read)
