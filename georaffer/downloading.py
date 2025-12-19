"""Download orchestration for tile fetching.

This module handles downloading raw tiles from NRW, RLP, and BB servers,
with support for parallel downloads across different endpoints.
"""

import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from georaffer.runtime import InterruptManager


@dataclass
class DownloadStats:
    """Statistics for a download batch."""

    downloaded: int = 0
    skipped: int = 0
    failed: int = 0


class ByteProgress:
    """Thread-safe byte counter for progress tracking."""

    def __init__(self, pbar: Any | None = None):
        self.total_bytes = 0
        self.pbar = pbar
        self.start_time = time.perf_counter()
        import threading

        self._lock = threading.Lock()

    def update(self, chunk_size: int) -> None:
        with self._lock:
            self.total_bytes += chunk_size
            if self.pbar is not None:
                elapsed = time.perf_counter() - self.start_time
                mb = self.total_bytes / 1_000_000
                rate = mb / elapsed if elapsed > 0 else 0
                if mb >= 1000:
                    self.pbar.set_description_str(f"↓ {mb / 1000:.1f}GB @ {rate:.0f}MB/s")
                else:
                    self.pbar.set_description_str(f"↓ {mb:.0f}MB @ {rate:.0f}MB/s")


def download_files(
    downloads: list[tuple[str, str]],
    downloader: Any,
    force: bool = False,
    desc: str = "Downloading",
    disable_progress: bool = False,
    progress_bar: Any | None = None,
    log_progress: bool = True,
    byte_progress: ByteProgress | None = None,
) -> DownloadStats:
    """Download a list of files sequentially.

    Empirical testing shows no benefit from parallel downloads due to
    server-side rate limiting. Sequential downloads with connection reuse
    provide optimal throughput.

    Args:
        downloads: List of (url, path) tuples
        downloader: Downloader instance with download_file() method
        force: Re-download even if file exists
        desc: Progress bar description
        disable_progress: Disable tqdm progress bar (use for parallel execution)
        progress_bar: Optional shared tqdm instance to increment per file
        log_progress: Whether to log progress to stdout
        byte_progress: Optional ByteProgress for live byte updates

    Returns:
        DownloadStats with results

    Raises:
        Exception: If download fails (propagates from downloader)
    """
    stats = DownloadStats()

    if not downloads:
        return stats

    interrupt = InterruptManager.get()
    on_progress = byte_progress.update if byte_progress else None

    if disable_progress:
        # Simple progress when running in parallel to avoid overlapping bars
        last_log_time = datetime.now()

        for i, (url, path) in enumerate(downloads, 1):
            if interrupt.is_set():
                break

            filename = Path(url).name

            if Path(path).exists() and not force:
                stats.skipped += 1
            else:
                try:
                    downloader.download_file(url, path, on_progress=on_progress)
                    stats.downloaded += 1
                except Exception as e:
                    stats.failed += 1
                    print(f"Failed to download {filename}: {e}", file=sys.stderr)
                    raise

            # Print progress every 30 seconds
            now = datetime.now()
            if log_progress and (now - last_log_time).total_seconds() >= 30:
                print(
                    f"{desc}: Progress {i}/{len(downloads)} "
                    f"({stats.downloaded} downloaded, {stats.skipped} skipped)"
                )
                last_log_time = now

            if progress_bar is not None:
                progress_bar.update(1)
    else:
        # Use tqdm for progress bar with automatic TTY detection
        with tqdm(downloads, desc=desc, unit="file", ncols=100) as pbar:
            for url, path in pbar:
                if interrupt.is_set():
                    break

                filename = Path(url).name[:40]
                pbar.set_postfix_str(filename, refresh=False)

                if Path(path).exists() and not force:
                    stats.skipped += 1
                    continue

                try:
                    downloader.download_file(url, path, on_progress=on_progress)
                    stats.downloaded += 1
                except Exception as e:
                    stats.failed += 1
                    print(f"Failed to download {filename}: {e}", file=sys.stderr)
                    raise

                if progress_bar is not None:
                    progress_bar.update(1)

    if log_progress:
        print(
            f"{desc} complete: {stats.downloaded} downloaded, "
            f"{stats.skipped} skipped, {stats.failed} failed"
        )

    return stats


@dataclass
class DownloadTask:
    """Specification for a download stream."""

    name: str
    downloads: list[tuple[str, str]]
    downloader: Any


@dataclass
class DownloadResult:
    """Result from a completed download stream."""

    name: str
    count: int
    stats: DownloadStats
    duration: float


def download_parallel_streams(
    tasks: list[DownloadTask],
    force: bool = False,
    max_streams: int = 4,
    on_interrupt: Callable[[], None] | None = None,
) -> tuple[list[DownloadResult], DownloadStats]:
    """Download from multiple sources in parallel.

    Each source (NRW JP2, NRW LAZ, RLP JP2, RLP LAZ) is rate-limited
    independently, so we can download from all simultaneously while
    keeping each stream sequential internally.

    Args:
        tasks: List of DownloadTask specifications
        force: Re-download existing files
        max_streams: Maximum concurrent download streams
        on_interrupt: Optional callback when interrupted

    Returns:
        Tuple of (results_list, total_stats) where:
        - results_list: List of DownloadResult objects, one per file attempted
        - total_stats: DownloadStats aggregating downloaded/skipped/failed counts
    """
    if not tasks:
        return [], DownloadStats()

    interrupt = InterruptManager.get()
    total_files = sum(len(t.downloads) for t in tasks)
    results: list[DownloadResult] = []
    total_stats = DownloadStats()

    with (
        ThreadPoolExecutor(max_workers=max_streams) as executor,
        tqdm(
            total=total_files,
            desc="↓ 0MB @ 0MB/s",
            unit="file",
            ncols=90,
            bar_format="Downloading: [{bar:25}] {n}/{total} | ⏱ {elapsed} | {desc}",
            mininterval=0.1,
        ) as pbar,
    ):
        try:
            futures: dict[Any, str] = {}
            meta: dict[Any, dict] = {}

            # Shared byte counter for live progress across all streams
            byte_progress = ByteProgress(pbar)

            for task in tasks:
                start = time.perf_counter()
                future = executor.submit(
                    download_files,
                    task.downloads,
                    task.downloader,
                    force,
                    desc=task.name,
                    disable_progress=True,
                    progress_bar=pbar,
                    log_progress=False,
                    byte_progress=byte_progress,
                )
                futures[future] = task.name
                meta[future] = {"count": len(task.downloads), "start": start}

            # Collect results with timeout for responsive Ctrl+C
            pending = set(futures.keys())
            while pending:
                if interrupt.is_set():
                    break
                try:
                    for future in as_completed(pending, timeout=0.5):
                        pending.discard(future)
                        name = futures[future]
                        stats = future.result()
                        duration = time.perf_counter() - meta[future]["start"]
                        count = meta[future]["count"]

                        results.append(
                            DownloadResult(
                                name=name,
                                count=count,
                                stats=stats,
                                duration=duration,
                            )
                        )

                        total_stats.downloaded += stats.downloaded
                        total_stats.skipped += stats.skipped
                        total_stats.failed += stats.failed

                except TimeoutError:
                    continue

        except KeyboardInterrupt:
            interrupt.signal()
            pbar.leave = False  # Don't leave final bar render
            pbar.close()
            print()  # Clean newline after cleared bar
            executor.shutdown(wait=False, cancel_futures=True)
            if on_interrupt:
                on_interrupt()
            raise

    return results, total_stats
