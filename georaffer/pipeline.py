"""Main processing pipeline for downloading and converting tiles.

This module provides the high-level orchestration for the georaffer pipeline,
coordinating tile discovery, downloading, and conversion phases.
"""

import multiprocessing
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from georaffer.config import Region, get_tile_size_km
from georaffer.conversion import convert_tiles
from georaffer.downloaders import NRWDownloader, RLPDownloader
from georaffer.downloading import DownloadTask, download_parallel_streams
from georaffer.grids import compute_split_factor, generate_user_grid_tiles
from georaffer.reporting import (
    print_catalog_summary,
    print_config,
    print_pipeline_banner,
    print_step_header,
    print_table,
    render_table,
)
from georaffer.runtime import InterruptManager
from georaffer.tiles import RegionCatalog, TileSet, calculate_required_tiles

# Set multiprocessing to use 'spawn' mode for GDAL/rasterio safety
# Why 'spawn' instead of 'fork'?
#   - GDAL and rasterio use internal state (file handles, caches, threads)
#   - fork() copies parent process memory, including these non-fork-safe resources
#   - Child processes can inherit corrupted GDAL state, causing crashes or hangs
#   - spawn starts fresh processes without parent state (slower startup, safer execution)
#
# Platform notes:
#   - Windows/macOS: spawn is already the default
#   - Linux: fork is default, so we explicitly force spawn here
if sys.platform not in ("win32", "darwin"):
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set (e.g., if another module called this first)


@dataclass
class ProcessingStats:
    """Statistics for the processing run."""

    downloaded: int = 0
    skipped: int = 0
    failed_download: int = 0
    converted: int = 0
    failed_convert: int = 0
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


def _estimate_outputs(
    tile_set: TileSet, resolutions: list[int], grid_size_km: float
) -> tuple[int, int]:
    """Estimate number of outputs for JP2 and LAZ given grid size and splits.

    Args:
        tile_set: TileSet with tiles grouped by region
        resolutions: List of target resolutions
        grid_size_km: User grid size for split calculation

    Returns:
        Tuple of (jp2_output_count, laz_output_count) where each count includes:
        - Split factor (e.g., 2km RLP tile -> four 1km outputs)
        - Resolution multiplier (e.g., 3 resolutions = 3x outputs)
    """
    res_factor = len(resolutions)

    # Region native tile sizes in km
    region_tile_km = {
        "nrw": get_tile_size_km(Region.NRW),
        "rlp": get_tile_size_km(Region.RLP),
    }

    def _count(tiles_by_region: dict, tile_type: str) -> int:
        total = 0
        for region, tiles in tiles_by_region.items():
            tile_km = region_tile_km.get(region, 1.0)
            split = compute_split_factor(tile_km, grid_size_km)
            total += len(tiles) * split * res_factor
        return total

    jp2_est = _count(tile_set.jp2, "jp2")
    laz_est = _count(tile_set.laz, "laz")
    return jp2_est, laz_est


def process_tiles(
    coords: list[tuple[float, float]],
    output_dir: str,
    resolutions: list[int] | None = None,
    grid_size_km: float = 1.0,
    margin_km: float = 1.0,
    imagery_from: tuple[int, int | None] | None = None,
    force_download: bool = False,
    max_workers: int = 4,
    profiling: bool = False,
    process_images: bool = True,
    process_pointclouds: bool = True,
    reprocess: bool = False,
) -> ProcessingStats:
    """Main entry point: download and process tiles for given coordinates.

    Args:
        coords: List of (utm_x, utm_y) coordinates
        output_dir: Base output directory
        resolutions: Target resolutions in pixels (default: [1000])
        grid_size_km: User's working grid resolution in km
        margin_km: Buffer distance from center tile border in km
        imagery_from: Download historic NRW imagery: (from_year, to_year) or (from_year, None).
            None = latest only. (2015, None) = all years from 2015. (2015, 2018) = years 2015-2018.
        force_download: Re-download existing files
        max_workers: Number of parallel workers for conversion
        profiling: Enable profiling output for conversion timing
        process_images: Download and convert orthophoto imagery (JP2)
        process_pointclouds: Download and convert point clouds to DSM (LAZ → GeoTIFF)
        reprocess: If True, re-download and re-convert existing files.
            By default (False), skip files where outputs already exist.

    Returns:
        ProcessingStats with processing results
    """
    if resolutions is None:
        resolutions = [1000]

    run_start = time.perf_counter()

    # Print banner and configuration
    print_pipeline_banner()
    print_config(
        num_coords=len(coords),
        grid_size_km=grid_size_km,
        margin_km=margin_km,
        resolutions=resolutions,
        output_dir=output_dir,
        imagery_from=imagery_from,
    )

    # Initialize downloaders
    # imagery_from is (from_year, to_year) or None; both NRW and RLP take year range
    nrw_downloader = NRWDownloader(output_dir, imagery_from=imagery_from)
    rlp_downloader = RLPDownloader(output_dir, imagery_from=imagery_from)

    # Create output directories
    for subdir in ["raw/image", "raw/dsm", "processed/image", "processed/dsm"]:
        (Path(output_dir) / subdir).mkdir(parents=True, exist_ok=True)

    # STEP 1: Calculate user grid first (needed for RLP WMS queries)
    print_step_header(1, "Calculating User Grid")
    print(
        f"Generating {grid_size_km:.2f}km grid with {margin_km:.2f}km margin around flight path..."
    )
    grid_start = time.perf_counter()
    user_tiles = generate_user_grid_tiles(coords, grid_size_km, margin_km)
    grid_duration = time.perf_counter() - grid_start
    print(f"  Generated {len(user_tiles)} user tiles in {grid_duration:.1f}s")

    # Convert user tiles to RLP native grid coords for WMS queries
    # User tiles are in user grid coords; we need RLP 2km grid coords
    rlp_native_coords: set[tuple[int, int]] = set()
    for x, y in user_tiles:
        # Convert user grid to UTM (center of tile)
        utm_x = (x + 0.5) * grid_size_km * 1000
        utm_y = (y + 0.5) * grid_size_km * 1000
        # Convert to RLP native grid
        rlp_coords, _ = rlp_downloader.utm_to_grid_coords(utm_x, utm_y)
        rlp_native_coords.add(rlp_coords)

    # STEP 2: Load tile catalogs
    print_step_header(2, "Loading Available Tiles from Remote Servers")
    print("Querying tile catalogs from NRW and RLP servers...")
    phase_start = time.perf_counter()
    nrw_jp2, nrw_laz = nrw_downloader.get_available_tiles()
    rlp_jp2, rlp_laz = rlp_downloader.get_available_tiles(requested_coords=rlp_native_coords)
    catalogs_duration = time.perf_counter() - phase_start

    # Use total counts (includes historical) for JP2, unique locations for LAZ
    print_catalog_summary(
        nrw_downloader.total_jp2_count or len(nrw_jp2),
        len(nrw_laz),
        rlp_downloader.total_jp2_count or len(rlp_jp2),
        len(rlp_laz),
        catalogs_duration,
    )

    # Build region catalogs in priority order (first match wins)
    regions = [
        RegionCatalog("nrw", nrw_downloader, nrw_jp2, nrw_laz),
        RegionCatalog("rlp", rlp_downloader, rlp_jp2, rlp_laz),
    ]
    tile_set, downloads_by_source = calculate_required_tiles(user_tiles, grid_size_km, regions)
    calc_duration = time.perf_counter() - phase_start

    # Print coverage analysis
    covered_jp2 = len(user_tiles) - len(tile_set.missing_jp2)
    covered_laz = len(user_tiles) - len(tile_set.missing_laz)
    coverage_jp2_pct = (covered_jp2 / len(user_tiles) * 100) if user_tiles else 0
    coverage_laz_pct = (covered_laz / len(user_tiles) * 100) if user_tiles else 0

    print()
    print_table(
        "Tile Coverage Analysis",
        [
            "User Tiles",
            "JP2 Available",
            "JP2 Missing",
            "LAZ Available",
            "LAZ Missing",
            "Compute Time",
        ],
        [
            (
                f"{len(user_tiles)}",
                f"{covered_jp2} ({coverage_jp2_pct:.0f}%)",
                f"{len(tile_set.missing_jp2)} ({100 - coverage_jp2_pct:.0f}%)",
                f"{covered_laz} ({coverage_laz_pct:.0f}%)",
                f"{len(tile_set.missing_laz)} ({100 - coverage_laz_pct:.0f}%)",
                f"{calc_duration:.1f}s",
            )
        ],
    )

    # Coverage breakdown by region - use actual download counts (includes multi-year)
    nrw_jp2_count = len(downloads_by_source.get("nrw_jp2", []))
    nrw_laz_count = len(downloads_by_source.get("nrw_laz", []))
    rlp_jp2_count = len(downloads_by_source.get("rlp_jp2", []))
    rlp_laz_count = len(downloads_by_source.get("rlp_laz", []))

    # Calculate split factors per region (only for regions with tiles)
    nrw_tile_km = get_tile_size_km(Region.NRW)
    rlp_tile_km = get_tile_size_km(Region.RLP)
    nrw_has_tiles = nrw_jp2_count > 0 or nrw_laz_count > 0
    rlp_has_tiles = rlp_jp2_count > 0 or rlp_laz_count > 0
    nrw_split = compute_split_factor(nrw_tile_km, grid_size_km) if nrw_has_tiles else 1
    rlp_split = compute_split_factor(rlp_tile_km, grid_size_km) if rlp_has_tiles else 1
    nrw_split_side = int(nrw_split**0.5)  # e.g., 4 -> 2×2
    rlp_split_side = int(rlp_split**0.5)

    # Calculate outputs per region
    nrw_jp2_out = nrw_jp2_count * nrw_split
    nrw_laz_out = nrw_laz_count * nrw_split
    rlp_jp2_out = rlp_jp2_count * rlp_split
    rlp_laz_out = rlp_laz_count * rlp_split
    total_jp2_out = nrw_jp2_out + rlp_jp2_out
    total_laz_out = nrw_laz_out + rlp_laz_out

    print()
    print(f"Conversion Plan (target: {grid_size_km}km grid)")
    table = render_table(
        ["Region", "Native", "Target", "Split", "Imagery", "Elevation"],
        [
            (
                "NRW",
                f"{nrw_tile_km:.0f}km",
                f"{grid_size_km}km",
                f"{nrw_split_side}×{nrw_split_side}",
                f"{nrw_jp2_count} → {nrw_jp2_out}",
                f"{nrw_laz_count} → {nrw_laz_out}",
            ),
            (
                "RLP",
                f"{rlp_tile_km:.0f}km",
                f"{grid_size_km}km",
                f"{rlp_split_side}×{rlp_split_side}",
                f"{rlp_jp2_count} → {rlp_jp2_out}",
                f"{rlp_laz_count} → {rlp_laz_out}",
            ),
            ("", "", "", "Total", str(total_jp2_out), str(total_laz_out)),
        ],
    )
    print(table)

    # Missing tile warnings
    if tile_set.missing_jp2 or tile_set.missing_laz:
        print()
        if tile_set.missing_jp2:
            sample = list(tile_set.missing_jp2)[:5]
            print(f"Warning: {len(tile_set.missing_jp2)} imagery tiles not available")
            print(f"  Sample missing JP2 tiles (user grid coords): {sample}")
        if tile_set.missing_laz:
            sample = list(tile_set.missing_laz)[:5]
            print(f"Warning: {len(tile_set.missing_laz)} point cloud tiles not available")
            print(f"  Sample missing LAZ tiles (user grid coords): {sample}")
        print("  Note: Tiles may be outside coverage area or not yet published")
    print()

    # STEP 3: Download raw tiles
    total_stats = ProcessingStats()
    download_tasks = []

    # Filter downloads based on --only flag
    if process_images:
        if downloads_by_source["nrw_jp2"]:
            download_tasks.append(
                DownloadTask("NRW Imagery", downloads_by_source["nrw_jp2"], nrw_downloader)
            )
        if downloads_by_source["rlp_jp2"]:
            download_tasks.append(
                DownloadTask("RLP Imagery", downloads_by_source["rlp_jp2"], rlp_downloader)
            )
    if process_pointclouds:
        if downloads_by_source["nrw_laz"]:
            download_tasks.append(
                DownloadTask("NRW Point Clouds", downloads_by_source["nrw_laz"], nrw_downloader)
            )
        if downloads_by_source["rlp_laz"]:
            download_tasks.append(
                DownloadTask("RLP Point Clouds", downloads_by_source["rlp_laz"], rlp_downloader)
            )

    print_step_header(3, "Downloading Raw Tiles")

    if download_tasks:
        total_files = sum(len(t.downloads) for t in download_tasks)
        print(f"Found {total_files} files to download from {len(download_tasks)} sources")
        print("Downloading in parallel (4 concurrent streams)...")
        print()

        # Clear interrupt state
        InterruptManager.get().clear()

        def on_interrupt():
            print()
            print("=" * 80)
            print("INTERRUPTED BY USER (Ctrl+C)")
            print("=" * 80)
            print()
            print("Partial results may be available in the output directory.")
            print()

        try:
            results, dl_stats = download_parallel_streams(
                download_tasks,
                force=force_download,
                on_interrupt=on_interrupt,
            )

            total_stats.downloaded = dl_stats.downloaded
            total_stats.skipped = dl_stats.skipped
            total_stats.failed_download = dl_stats.failed

            # Print download summary
            print()
            rows = []
            for r in results:
                rate = r.stats.downloaded / r.duration if r.duration > 0 else 0.0
                rows.append(
                    (
                        r.name,
                        f"{r.count}",
                        f"{r.stats.downloaded}",
                        f"{r.stats.skipped}",
                        f"{r.stats.failed}",
                        f"{r.duration:.1f}s",
                        f"{rate:.1f} files/s",
                    )
                )
            print_table(
                "Download Summary",
                ["Source", "Total", "Downloaded", "Skipped", "Failed", "Duration", "Speed"],
                rows,
            )
            print()

        except KeyboardInterrupt:
            raise
    else:
        print("No files to download (all tiles already cached)")
        print()

    # STEP 4: Convert tiles
    print_step_header(4, "Converting to GeoTIFF")
    data_types = []
    if process_images:
        data_types.append("imagery (JP2)")
    if process_pointclouds:
        data_types.append("point clouds (LAZ)")
    print(f"Converting {' and '.join(data_types)} to GeoTIFF format...")
    print(f"Target resolutions: {', '.join(f'{r}px' for r in resolutions)}")
    print()

    convert_stats = convert_tiles(
        str(Path(output_dir) / "raw"),
        str(Path(output_dir) / "processed"),
        resolutions,
        max_workers,
        process_images=process_images,
        process_pointclouds=process_pointclouds,
        grid_size_km=grid_size_km,
        profiling=profiling,
        reprocess=reprocess,
    )

    total_stats.converted = convert_stats.converted
    total_stats.failed_convert = convert_stats.failed
    total_stats.jp2_sources = convert_stats.jp2_sources
    total_stats.jp2_converted = convert_stats.jp2_converted
    total_stats.jp2_failed = convert_stats.jp2_failed
    total_stats.jp2_skipped = convert_stats.jp2_skipped
    total_stats.laz_sources = convert_stats.laz_sources
    total_stats.laz_converted = convert_stats.laz_converted
    total_stats.laz_failed = convert_stats.laz_failed
    total_stats.laz_skipped = convert_stats.laz_skipped
    total_stats.jp2_duration = convert_stats.jp2_duration
    total_stats.laz_duration = convert_stats.laz_duration
    total_stats.jp2_split_performed = convert_stats.jp2_split_performed
    total_stats.laz_split_performed = convert_stats.laz_split_performed
    total_stats.interrupted = convert_stats.interrupted

    # Print conversion summary
    jp2_rate = (
        (convert_stats.jp2_converted / convert_stats.jp2_duration)
        if convert_stats.jp2_duration
        else 0.0
    )
    laz_rate = (
        (convert_stats.laz_converted / convert_stats.laz_duration)
        if convert_stats.laz_duration
        else 0.0
    )
    res_str = ", ".join(str(r) for r in resolutions) if resolutions else "native"

    rows = [
        (
            "Imagery (JP2->GeoTIFF)",
            f"{convert_stats.jp2_sources}",
            f"{convert_stats.jp2_converted}",
            res_str,
            f"{convert_stats.jp2_failed}",
            f"{convert_stats.jp2_duration:.1f}s" if convert_stats.jp2_duration else "-",
            f"{jp2_rate:.1f} out/s" if convert_stats.jp2_duration else "-",
        ),
        (
            "Point Clouds (LAZ->DSM)",
            f"{convert_stats.laz_sources}",
            f"{convert_stats.laz_converted}",
            res_str,
            f"{convert_stats.laz_failed}",
            f"{convert_stats.laz_duration:.1f}s" if convert_stats.laz_duration else "-",
            f"{laz_rate:.1f} out/s" if convert_stats.laz_duration else "-",
        ),
    ]
    print()
    print_table(
        "Conversion Summary",
        ["Type", "In", "Out", "Resolutions", "Failed", "Duration", "Speed"],
        rows,
    )
    print()

    # Final summary
    total_duration = time.perf_counter() - run_start
    print("=" * 80)
    print("PROCESSING COMPLETE" if not convert_stats.interrupted else "PROCESSING INTERRUPTED")
    print("=" * 80)
    print()
    print(f"Total Duration: {total_duration:.1f}s ({total_duration / 60:.1f} minutes)")
    print()
    print("Results:")
    print(f"  Files downloaded: {total_stats.downloaded} new, {total_stats.skipped} cached")
    print(f"  JP2 outputs: {convert_stats.jp2_converted} from {convert_stats.jp2_sources} sources")
    print(f"  LAZ outputs: {convert_stats.laz_converted} from {convert_stats.laz_sources} sources")
    print(f"  Total outputs: {total_stats.converted} successful")
    if total_stats.failed_download > 0:
        print(f"  Warning: Download failures: {total_stats.failed_download}")
    if total_stats.failed_convert > 0:
        print(f"  Warning: Conversion failures: {total_stats.failed_convert}")
    if convert_stats.interrupted:
        print("  Warning: Run was interrupted; outputs may be partial.")
    print()
    print(f"Output Directory: {output_dir}")
    print(f"  Raw tiles: {output_dir}/raw/")
    print(f"  Processed GeoTIFFs: {output_dir}/processed/")
    print()
    print("=" * 80)
    print()

    if convert_stats.interrupted:
        sys.exit(130)

    return total_stats
