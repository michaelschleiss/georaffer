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
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from georaffer.config import Region, get_tile_size_km
from georaffer.converters import convert_jp2, convert_laz
from georaffer.converters.utils import parse_tile_coords
from georaffer.grids import compute_split_factor
from georaffer.runtime import InterruptManager, shutdown_executor
from georaffer.workers import (
    convert_dsm_worker,
    convert_jp2_worker,
    detect_region,
    generate_output_name,
    init_worker,
    resolve_source_year,
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


def _outputs_exist(
    filename: str,
    processed_dir: str,
    data_type: str,
    resolutions: list[int | None],
    grid_size_km: float,
    *,
    source_dir: str | None = None,
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
    input_path = Path(source_dir) / filename if source_dir is not None else Path(filename)
    # Match worker year logic as closely as possible to ensure we check
    # the exact output filenames that conversion would produce.
    year = resolve_source_year(filename, input_path, data_type=data_type, region=region)

    # Get base coordinates from filename
    coords = parse_tile_coords(filename)
    if not coords:
        # Can't predict output without coords - don't skip
        return False

    base_x, base_y = coords
    tile_km = get_tile_size_km(region)

    try:
        split_factor = compute_split_factor(tile_km, grid_size_km)
    except Exception:
        # If splitting isn't possible (non-integer ratio), we can't reliably predict
        # output paths here, so don't skip.
        return False

    from georaffer.config import METERS_PER_KM, utm_zone_str_for_region
    from georaffer.converters.utils import generate_split_output_path

    ratio = round(tile_km / grid_size_km) if split_factor > 1 else 1
    grid_size_m = round(grid_size_km * METERS_PER_KM)
    utm_zone = utm_zone_str_for_region(region)

    for res in resolutions:
        # Keep directory conventions aligned with workers.py
        if data_type == "dsm":
            res_dir = Path(processed_dir) / "dsm" / str(res)
        else:
            res_dir = Path(processed_dir) / "image" / ("native" if res is None else str(res))

        base_output_name = generate_output_name(filename, region, year, data_type)
        base_output_path = res_dir / base_output_name

        if split_factor == 1:
            if not base_output_path.exists():
                return False
            continue

        # Split outputs use UTM-based filenames derived from the base output path.
        for r_idx in range(ratio):
            for c_idx in range(ratio):
                new_x = base_x + c_idx
                new_y = base_y + (ratio - 1 - r_idx)
                easting = int(base_x * METERS_PER_KM + c_idx * grid_size_m)
                northing = int(base_y * METERS_PER_KM + (ratio - 1 - r_idx) * grid_size_m)
                output_path = generate_split_output_path(
                    str(base_output_path),
                    new_x,
                    new_y,
                    easting=easting,
                    northing=northing,
                    utm_zone=utm_zone,
                )
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
    image_files: list[str] | None = None,
    dsm_files: list[str] | None = None,
) -> ConversionStats:
    """Convert raw tiles to GeoTIFF using parallel workers.

    Args:
        raw_dir: Directory with raw tiles (contains image/ and dsm/ subdirs)
        processed_dir: Output directory
        resolutions: Target resolutions in pixels
        max_workers: Number of parallel conversion workers
        process_images: Convert JP2 imagery
        process_pointclouds: Convert DSM sources (LAZ/TIF)
        grid_size_km: User's grid size for splitting
        profiling: Enable profiling output
        reprocess: If False (default), skip files where outputs already exist.
            If True, reconvert all files regardless of existing outputs.
        image_files: Optional list of image file paths to convert. If None,
            scans raw_dir/image/ for all supported files.
        dsm_files: Optional list of DSM file paths to convert. If None,
            scans raw_dir/dsm/ for all supported files.

    Returns:
        ConversionStats with results

    Note:
        Each worker processes one source file with ALL its resolutions,
        ensuring the file is read once and kept in memory for all conversions.
        Thread count per worker is controlled by THREADS_PER_WORKER in config.py.
    """
    from georaffer.config import THREADS_PER_WORKER

    stats = ConversionStats()

    # Collect files to convert (use provided lists or scan directories)
    jp2_files: list[str] = []
    laz_files: list[str] = []

    jp2_dir = os.path.join(raw_dir, "image")
    if process_images:
        if image_files is not None:
            # Use provided file paths (extract basenames for files that exist)
            jp2_files = sorted(
                os.path.basename(f) for f in image_files
                if os.path.exists(f)
            )
        elif os.path.exists(jp2_dir):
            jp2_files = sorted(
                f for f in os.listdir(jp2_dir)
                if f.endswith((".jp2", ".tif", ".zip"))
            )

    laz_dir = os.path.join(raw_dir, "dsm")
    if process_pointclouds:
        if dsm_files is not None:
            # Use provided file paths (extract basenames for files that exist)
            laz_files = sorted(
                os.path.basename(f) for f in dsm_files
                if os.path.exists(f)
            )
        elif os.path.exists(laz_dir):
            laz_files = sorted(
                f for f in os.listdir(laz_dir)
                if f.lower().endswith((".laz", ".tif", ".zip"))
            )

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
    def _signal_handler(*_):
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
            bar_format="Converting: [{bar:23}] {n}/{total} | ⏱ {elapsed} | {desc}",
            mininterval=0.1,
        ) as pbar:
            # Convert JP2 files first (fail fast on missing GDAL driver)
            if jp2_files:
                jp2_args = []
                for f in jp2_files:
                    jp2_args.append(
                        (
                            f,
                            jp2_dir,
                            processed_dir,
                            resolutions,
                            threads_per_worker,
                            grid_size_km,
                            profiling,
                            None,  # unused
                        )
                    )

                jp2_futures = []
                for args in jp2_args:
                    if stop_event.is_set():
                        break
                    f = args[0]
                    if not reprocess and _outputs_exist(
                        f,
                        processed_dir,
                        "image",
                        resolutions,
                        grid_size_km,
                        source_dir=jp2_dir,
                    ):
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
                                _, _filename, out_count = future.result()
                                stats.converted += out_count
                                stats.jp2_converted += out_count
                                if out_count > len(resolutions):
                                    stats.jp2_split_performed = True
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

                laz_args = []
                for f in laz_files:
                    laz_args.append(
                        (
                            f,
                            laz_dir,
                            processed_dir,
                            resolutions,
                            threads_per_worker,
                            grid_size_km,
                            profiling,
                            None,  # unused
                        )
                    )

                laz_futures = []
                for args in laz_args:
                    if stop_event.is_set():
                        break
                    f = args[0]
                    if not reprocess and _outputs_exist(
                        f,
                        processed_dir,
                        "dsm",
                        resolutions,
                        grid_size_km,
                        source_dir=laz_dir,
                    ):
                        stats.laz_skipped += 1
                        pbar.update(1)
                        _update_files_per_second(pbar)
                        continue
                    laz_futures.append(executor.submit(convert_dsm_worker, args))

                pending = set(laz_futures)
                while pending and not stop_event.is_set():
                    try:
                        for future in as_completed(pending, timeout=0.1):
                            pending.discard(future)
                            stats.laz_sources += 1  # Count source regardless of success
                            try:
                                _, _filename, out_count = future.result()
                                stats.converted += out_count
                                stats.laz_converted += out_count
                                if out_count > len(resolutions):
                                    stats.laz_split_performed = True
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
