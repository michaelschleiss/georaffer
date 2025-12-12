"""Conversion orchestration for tile processing.

This module handles parallel conversion of raw JP2 and LAZ files to GeoTIFF
format using ProcessPoolExecutor for CPU-bound operations.
"""

import multiprocessing
import os
import signal
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field

from tqdm import tqdm

from pathlib import Path

from georaffer.config import get_tile_size_km
from georaffer.converters import convert_jp2, convert_laz
from georaffer.converters.utils import parse_tile_coords
from georaffer.grids import compute_split_factor
from georaffer.metadata import create_provenance_csv
from georaffer.provenance import compute_split_coordinates, extract_year_from_filename
from georaffer.runtime import InterruptManager, shutdown_executor
from georaffer.workers import (
    convert_jp2_worker,
    convert_laz_worker,
    detect_region,
    generate_output_name,
    init_worker,
)


@dataclass
class ConversionStats:
    """Statistics for conversion operations."""

    converted: int = 0
    failed: int = 0
    jp2_sources: int = 0
    jp2_converted: int = 0
    jp2_failed: int = 0
    jp2_skipped: int = 0  # Skipped because output exists
    laz_sources: int = 0
    laz_converted: int = 0
    laz_failed: int = 0
    laz_skipped: int = 0  # Skipped because output exists
    jp2_duration: float = 0.0
    laz_duration: float = 0.0
    interrupted: bool = False
    jp2_split_performed: bool = False
    laz_split_performed: bool = False
    tiles_metadata: list[dict] = field(default_factory=list)


def _outputs_exist(
    filename: str,
    processed_dir: str,
    data_type: str,
    resolutions: list[int],
    grid_size_km: float,
) -> bool:
    """Check if all output files exist for this source file at all resolutions.

    Args:
        filename: Source filename (JP2/TIF/LAZ)
        processed_dir: Output directory
        data_type: 'image' or 'dsm'
        resolutions: List of target resolutions
        grid_size_km: User's grid size for determining splits

    Returns:
        True if ALL expected output files exist, False if any are missing.
    """
    region = detect_region(filename)
    year = extract_year_from_filename(filename, require=False) or "latest"

    # Get base coordinates from filename
    coords = parse_tile_coords(filename)
    if not coords:
        # Can't predict output without coords - don't skip
        return False

    base_x, base_y = coords
    tile_km = get_tile_size_km(region)

    # Get all output coordinates (handles splits)
    output_coords = compute_split_coordinates(base_x, base_y, tile_km, grid_size_km)

    # Check each resolution and each split coordinate
    split_factor = compute_split_factor(tile_km, grid_size_km)
    for res in resolutions:
        res_dir = Path(processed_dir) / data_type / ("native" if res is None else str(res))
        for grid_x, grid_y in output_coords:
            # Generate expected output filename using same logic as workers
            output_name = generate_output_name(
                filename, region, year, data_type
            )
            # For splits, replace coordinates in the generated name
            if split_factor > 1:
                from georaffer.config import METERS_PER_KM, UTM_ZONE_STR
                easting = grid_x * METERS_PER_KM
                northing = grid_y * METERS_PER_KM
                output_name = f"{region.value.lower()}_{UTM_ZONE_STR}_{easting}_{northing}_{year}.tif"
            output_path = res_dir / output_name
            if not output_path.exists():
                return False
    return True


def convert_tiles(
    raw_dir: str,
    processed_dir: str,
    resolutions: list[int],
    max_workers: int = 4,
    process_images: bool = True,
    process_pointclouds: bool = True,
    grid_size_km: float = 1.0,
    profiling: bool = False,
    reprocess: bool = False,
) -> ConversionStats:
    """Convert all raw tiles to GeoTIFF using parallel workers.

    Args:
        raw_dir: Directory with raw tiles (contains image/ and dsm/ subdirs)
        processed_dir: Output directory
        resolutions: Target resolutions in pixels
        max_workers: Number of parallel conversion workers
        process_images: Convert JP2 imagery
        process_pointclouds: Convert LAZ point clouds
        grid_size_km: User's grid size for splitting
        profiling: Enable profiling output
        reprocess: If False (default), skip files where outputs already exist.
            If True, reconvert all files regardless of existing outputs.

    Returns:
        ConversionStats with results

    Note:
        Each worker processes one source file with ALL its resolutions,
        ensuring the file is read once and kept in memory for all conversions.
        Thread count per worker is controlled by THREADS_PER_WORKER in config.py.
    """
    from georaffer.config import THREADS_PER_WORKER

    stats = ConversionStats()

    # Collect files to convert
    jp2_files: list[str] = []
    laz_files: list[str] = []

    jp2_dir = os.path.join(raw_dir, "image")
    if process_images and os.path.exists(jp2_dir):
        jp2_files = sorted(f for f in os.listdir(jp2_dir) if f.endswith((".jp2", ".tif")))

    laz_dir = os.path.join(raw_dir, "dsm")
    if process_pointclouds and os.path.exists(laz_dir):
        laz_files = sorted(f for f in os.listdir(laz_dir) if f.endswith(".laz"))

    total_files = len(jp2_files) + len(laz_files)
    if total_files == 0:
        return stats

    jp2_start = time.perf_counter()
    laz_start: float | None = None

    # Compute threads per worker from config
    cpu_count = os.cpu_count() or 4
    if THREADS_PER_WORKER == "auto":
        threads_per_worker = max(2, cpu_count // max_workers)
    elif THREADS_PER_WORKER is None:
        threads_per_worker = None  # Don't set thread limits
    else:
        threads_per_worker = THREADS_PER_WORKER

    # Determine executor type
    use_process_pool = os.getenv("GEORAFFER_DISABLE_PROCESS_POOL") != "1"

    # Check if converters are mocked (unit tests)
    try:
        from unittest.mock import Mock

        converters_mocked = isinstance(convert_jp2, Mock) or isinstance(convert_laz, Mock)
    except Exception:
        converters_mocked = False

    if not use_process_pool or converters_mocked:
        from concurrent.futures import ThreadPoolExecutor

        executor_class = ThreadPoolExecutor
    else:
        executor_class = ProcessPoolExecutor

    # Create executor
    executor = None
    if executor_class is ProcessPoolExecutor:
        if threads_per_worker is not None:
            # Limit threads for lazrs/numba to avoid oversubscription
            threads_for_lazrs = max(1, min(threads_per_worker, cpu_count))
            executor = executor_class(
                max_workers=max_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=init_worker,
                initargs=(threads_for_lazrs,),
            )
        else:
            # None = don't set thread limits, use library defaults
            executor = executor_class(
                max_workers=max_workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
    else:
        executor = executor_class(max_workers=max_workers)

    interrupted = False
    stop_event = threading.Event()
    interrupt_manager = InterruptManager.get()

    # Signal handler for fast shutdown
    def _signal_handler(signum, frame):
        nonlocal interrupted
        if not stop_event.is_set():
            interrupted = True
            interrupt_manager.signal()
            print("\nConversion interrupted by signal")
            shutdown_executor(executor, stop_event)

    old_int = signal.getsignal(signal.SIGINT)
    old_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        def _update_files_per_second(pbar: tqdm) -> None:
            rate = pbar.format_dict.get("rate")
            if rate is None:
                return
            pbar.set_description_str(f"{rate:0.1f} files/s", refresh=False)

        with tqdm(
            total=total_files,
            desc="0.0 files/s",
            unit="file",
            ncols=90,
            bar_format="Converting: [{bar:25}] {n}/{total} | ⏱ {elapsed} | {desc}",
            mininterval=0.1,
        ) as pbar:
            # Convert JP2 files first (fail fast on missing GDAL driver)
            if jp2_files:
                jp2_args = [
                    (
                        f,
                        jp2_dir,
                        processed_dir,
                        resolutions,
                        threads_per_worker,
                        grid_size_km,
                        profiling,
                    )
                    for f in jp2_files
                ]

                jp2_futures = []
                for args in jp2_args:
                    if stop_event.is_set():
                        break
                    f = args[0]
                    if not reprocess and _outputs_exist(f, processed_dir, "image", resolutions, grid_size_km):
                        stats.jp2_skipped += 1
                        pbar.update(1)
                        _update_files_per_second(pbar)
                        continue
                    jp2_futures.append(executor.submit(convert_jp2_worker, args))

                pending = set(jp2_futures)
                while pending and not stop_event.is_set():
                    try:
                        for future in as_completed(pending, timeout=0.1):
                            pending.discard(future)
                            stats.jp2_sources += 1  # Count source regardless of success
                            try:
                                _, metadata, _filename, out_count = future.result()
                                stats.converted += out_count
                                stats.jp2_converted += out_count
                                if out_count > len(resolutions):
                                    stats.jp2_split_performed = True
                                stats.tiles_metadata.extend(metadata)
                                pbar.update(1)
                                _update_files_per_second(pbar)
                            except Exception as e:
                                stats.jp2_failed += 1
                                stats.failed += 1
                                print(f"\nConversion failed: {e}", file=sys.stderr)
                                pbar.update(1)
                                _update_files_per_second(pbar)
                    except TimeoutError:
                        continue

            # Convert LAZ files
            if laz_files:
                laz_start = time.perf_counter()

                laz_args = [
                    (
                        f,
                        laz_dir,
                        processed_dir,
                        resolutions,
                        threads_per_worker,
                        grid_size_km,
                        profiling,
                    )
                    for f in laz_files
                ]

                laz_futures = []
                for args in laz_args:
                    if stop_event.is_set():
                        break
                    f = args[0]
                    if not reprocess and _outputs_exist(f, processed_dir, "dsm", resolutions, grid_size_km):
                        stats.laz_skipped += 1
                        pbar.update(1)
                        _update_files_per_second(pbar)
                        continue
                    laz_futures.append(executor.submit(convert_laz_worker, args))

                pending = set(laz_futures)
                while pending and not stop_event.is_set():
                    try:
                        for future in as_completed(pending, timeout=0.1):
                            pending.discard(future)
                            stats.laz_sources += 1  # Count source regardless of success
                            try:
                                _, metadata, _filename, out_count = future.result()
                                stats.converted += out_count
                                stats.laz_converted += out_count
                                if out_count > len(resolutions):
                                    stats.laz_split_performed = True
                                stats.tiles_metadata.extend(metadata)
                                pbar.update(1)
                                _update_files_per_second(pbar)
                            except Exception as e:
                                stats.laz_failed += 1
                                stats.failed += 1
                                print(f"\nConversion failed: {e}", file=sys.stderr)
                                pbar.update(1)
                                _update_files_per_second(pbar)
                    except TimeoutError:
                        continue

    except KeyboardInterrupt:
        interrupted = True
        shutdown_executor(executor, stop_event)

    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        if not interrupted:
            executor.shutdown(wait=True)
        else:
            shutdown_executor(executor, stop_event)

    # Write provenance CSV
    if stats.tiles_metadata and not interrupted:
        csv_path = os.path.join(processed_dir, "provenance.csv")
        create_provenance_csv(stats.tiles_metadata, csv_path)
        print(f"Wrote provenance to {csv_path}")

    # NOTE: We intentionally do NOT clean up lock files here. Deleting them while
    # another georaffer process is still running would break cross-process mutual
    # exclusion (the other process holds fcntl locks on the inodes, but a new process
    # would create fresh files with new inodes, bypassing the locks). Lock files are
    # 0-byte and harmless to leave in place.

    # Calculate durations
    if jp2_files:
        stats.jp2_duration = (laz_start or time.perf_counter()) - jp2_start
    if laz_files and laz_start:
        stats.laz_duration = time.perf_counter() - laz_start

    stats.interrupted = interrupted
    return stats
