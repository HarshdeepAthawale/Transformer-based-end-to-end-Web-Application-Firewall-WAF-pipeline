"""
Stream Tailer

Async file tailing for live log ingestion.
Uses polling-based approach to watch for new lines appended to a log file.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import AsyncIterator

logger = logging.getLogger(__name__)


async def tail_lines(
    log_path: str,
    follow: bool = True,
    poll_interval: float = 0.1,
    encoding: str = "utf-8",
    start_from_end: bool = True,
) -> AsyncIterator[str]:
    """Asynchronously tail a log file, yielding new lines as they appear.

    Args:
        log_path: Path to the log file to tail.
        follow: If True, keep watching for new lines indefinitely.
                If False, read to end of file and stop.
        poll_interval: Seconds between polls when no new data is available.
        encoding: File encoding.
        start_from_end: If True, start reading from the current end of file.
                        If False, read from the beginning first.

    Yields:
        Stripped, non-empty log lines.

    Raises:
        FileNotFoundError: If the log file does not exist at start.
    """
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    logger.info(
        "Tailing %s (follow=%s, poll_interval=%.2fs, start_from_end=%s)",
        log_path, follow, poll_interval, start_from_end,
    )

    with open(path, "r", encoding=encoding) as f:
        if start_from_end:
            f.seek(0, os.SEEK_END)

        partial_line = ""

        while True:
            line = f.readline()

            if line:
                # Accumulate partial lines (no newline yet)
                if not line.endswith("\n"):
                    partial_line += line
                    await asyncio.sleep(poll_interval)
                    continue

                full_line = (partial_line + line).strip()
                partial_line = ""

                if full_line:
                    yield full_line
            else:
                # No new data
                if not follow:
                    # Flush any remaining partial line
                    if partial_line.strip():
                        yield partial_line.strip()
                    break

                # Check if file was rotated (inode changed)
                try:
                    current_stat = os.stat(log_path)
                    fd_stat = os.fstat(f.fileno())
                    if current_stat.st_ino != fd_stat.st_ino:
                        logger.info("Log file rotated: %s", log_path)
                        # Flush partial
                        if partial_line.strip():
                            yield partial_line.strip()
                            partial_line = ""
                        break  # Caller should re-open
                except OSError:
                    pass

                await asyncio.sleep(poll_interval)
