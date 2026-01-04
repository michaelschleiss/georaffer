"""Main processing pipeline for downloading and converting tiles.

This module provides the high-level orchestration for the georaffer pipeline,
coordinating tile discovery, downloading, and conversion phases.
"""

import multiprocessing
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from georaffer.config import UTM_ZONE_BY_REGION, Region, get_tile_size_km
from georaffer.conversion import convert_tiles
from georaffer.downloaders import BBDownloader, BWDownloader, BYDownloader, NRWDownloader, RLPDownloader
from georaffer.downloading import DownloadTask, download_parallel_streams
from georaffer.grids import compute_split_factor, generate_tiles_by_zone
from georaffer.reporting import (
    print_catalog_summary,
    print_config,
    print_pipeline_banner,
    print_step_header,
    print_table,
    render_table,
)
from georaffer.runtime import InterruptManager
from georaffer.tiles import TileSet, build_filtered_download_list

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
    source_zone: int = 32,
    regions: list[Region] | None = None,
    refresh_catalog: bool = False,
) -> ProcessingStats:
    """Main entry point: download and process tiles for given coordinates.

    Args:
        coords: List of (utm_x, utm_y) coordinates in source_zone UTM
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
        process_pointclouds: Download and convert DSM sources (LAZ/TIF → GeoTIFF)
        reprocess: If True, re-download and re-convert existing files.
            By default (False), skip files where outputs already exist.
        source_zone: UTM zone of input coordinates (default: 32 for NRW/RLP)
        regions: Optional list of regions to include (default: all)
        refresh_catalog: Force refresh of tile catalog cache

    Returns:
        ProcessingStats with processing results
    """
    if resolutions is None:
        resolutions = [1000]

    run_start = time.perf_counter()

    selected_regions = regions or [Region.NRW, Region.RLP, Region.BB, Region.BW, Region.BY]
    selected_zones = {UTM_ZONE_BY_REGION[region] for region in selected_regions}
    if source_zone not in selected_zones:
        region_names = ", ".join(region.value for region in selected_regions)
        zone_list = ", ".join(str(zone) for zone in sorted(selected_zones))
        raise ValueError(
            f"Input coordinates are in UTM zone {source_zone}; selected regions "
            f"({region_names}) use zones {zone_list}. Include the matching region."
        )

    # Print banner and configuration
    print_pipeline_banner()
    print_config(
        num_coords=len(coords),
        grid_size_km=grid_size_km,
        margin_km=margin_km,
        resolutions=resolutions,
        output_dir=output_dir,
        imagery_from=imagery_from,
        regions=selected_regions,
    )

    # Initialize downloaders
    # imagery_from is (from_year, to_year) or None.
    # NRW supports historic imagery via catalog feeds; RLP supports historic imagery via WMS.
    nrw_downloader = (
        NRWDownloader(output_dir, imagery_from=imagery_from)
        if Region.NRW in selected_regions
        else None
    )
    rlp_downloader = (
        RLPDownloader(output_dir, imagery_from=imagery_from)
        if Region.RLP in selected_regions
        else None
    )
    bb_downloader = BBDownloader(output_dir) if Region.BB in selected_regions else None
    bw_downloader = (
        BWDownloader(output_dir, imagery_from=imagery_from)
        if Region.BW in selected_regions
        else None
    )
    by_downloader = (
        BYDownloader(output_dir, imagery_from=imagery_from)
        if Region.BY in selected_regions
        else None
    )

    # Create output directories
    for subdir in ["raw/image", "raw/dsm", "processed/image", "processed/dsm"]:
        (Path(output_dir) / subdir).mkdir(parents=True, exist_ok=True)

    # STEP 1: Calculate user grid first (needed for RLP WMS queries)
    print_step_header(1, "Calculating User Grid")
    print(
        f"Generating {grid_size_km:.2f}km grid with {margin_km:.2f}km margin around flight path..."
    )
    grid_start = time.perf_counter()
    tiles_by_zone = generate_tiles_by_zone(coords, source_zone, grid_size_km, margin_km)
    tiles_by_zone = {zone: tiles_by_zone.get(zone, set()) for zone in sorted(selected_zones)}
    # User tiles = unique grid cells from original coords (source zone only)
    total_user_tiles = len(tiles_by_zone.get(source_zone, set()))
    grid_duration = time.perf_counter() - grid_start
    print(f"  Generated {total_user_tiles} user tiles in {grid_duration:.1f}s")

    # Convert user tiles to RLP native grid coords for WMS queries
    # User tiles are in user grid coords; we need RLP 2km grid coords
    rlp_native_coords: set[tuple[int, int]] = set()
    if rlp_downloader is not None:
        rlp_zone = UTM_ZONE_BY_REGION[Region.RLP]
        rlp_user_tiles = tiles_by_zone.get(rlp_zone, set())
        for x, y in rlp_user_tiles:
            # Convert user grid to UTM (center of tile)
            utm_x = (x + 0.5) * grid_size_km * 1000
            utm_y = (y + 0.5) * grid_size_km * 1000
            # Convert to RLP native grid
            rlp_coords, _ = rlp_downloader.utm_to_grid_coords(utm_x, utm_y)
            rlp_native_coords.add(rlp_coords)

    # STEP 2: Load tile catalogs
    print_step_header(2, "Loading Available Tiles from Remote Servers")
    region_labels = ", ".join(region.value for region in selected_regions)
    print(f"Querying tile catalogs from {region_labels} servers...")
    phase_start = time.perf_counter()
    catalog_rows: list[tuple[str, int, int]] = []
    downloaders: list = []

    for name, dl in [("NRW", nrw_downloader), ("RLP", rlp_downloader), ("BW", bw_downloader), ("BY", by_downloader)]:
        if dl is None:
            continue
        cat = dl.build_catalog(refresh=refresh_catalog)
        if process_images and process_pointclouds:
            jp2, laz = set(cat.image_tiles.keys()), set(cat.dsm_tiles.keys())
            if jp2 != laz:
                raise ValueError(f"{name} JP2/LAZ mismatch: {len(jp2)} vs {len(laz)}")
        catalog_rows.append((name, dl.total_image_count, len(cat.dsm_tiles)))
        downloaders.append(dl)
    if bb_downloader is not None:
        bb_catalog = bb_downloader.build_catalog(refresh=refresh_catalog)
        if process_images and process_pointclouds:
            jp2_keys = set(bb_catalog.image_tiles.keys())
            laz_keys = set(bb_catalog.dsm_tiles.keys())
            # Strict check: bDOM must be subset of DOP (no orphan bDOM tiles)
            orphan_bdom = laz_keys - jp2_keys
            if orphan_bdom:
                sample = ", ".join(map(str, sorted(orphan_bdom)[:5]))
                raise ValueError(
                    f"BB has {len(orphan_bdom)} bDOM tiles without DOP (sample): {sample}"
                )
            # Count check: difference must match expected missing bDOM
            missing_count = len(jp2_keys) - len(laz_keys)
            if missing_count != bb_downloader.EXPECTED_MISSING_BDOM:
                raise ValueError(
                    f"BB bDOM missing count changed! "
                    f"Expected {bb_downloader.EXPECTED_MISSING_BDOM}, got {missing_count}. "
                    f"Update BBDownloader.EXPECTED_MISSING_BDOM if this is expected."
                )
        catalog_rows.append(("BB", len(bb_catalog.image_tiles), len(bb_catalog.dsm_tiles)))
        downloaders.append(bb_downloader)
    catalogs_duration = time.perf_counter() - phase_start

    print_catalog_summary(catalog_rows, catalogs_duration)

    zone_by_region = {
        region.value.lower(): UTM_ZONE_BY_REGION[region] for region in selected_regions
    }
    original_coords = np.array(coords) if coords else None
    tile_set, downloads_by_source = build_filtered_download_list(
        tiles_by_zone,
        grid_size_km,
        downloaders,
        zone_by_region,
        original_coords=original_coords,
        source_zone=source_zone,
    )
    calc_duration = time.perf_counter() - phase_start

    # Print coverage analysis
    covered_jp2 = total_user_tiles - len(tile_set.missing_jp2)
    covered_laz = total_user_tiles - len(tile_set.missing_laz)
    coverage_jp2_pct = (covered_jp2 / total_user_tiles * 100) if total_user_tiles else 0
    coverage_laz_pct = (covered_laz / total_user_tiles * 100) if total_user_tiles else 0

    print()
    print_table(
        "Tile Coverage Analysis",
        [
            "User Tiles",
            "JP2 Available",
            "JP2 Missing",
            "DSM Available",
            "DSM Missing",
            "Compute Time",
        ],
        [
            (
                f"{total_user_tiles}",
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
    bb_jp2_count = len(downloads_by_source.get("bb_jp2", []))
    bb_laz_count = len(downloads_by_source.get("bb_laz", []))
    bw_jp2_count = len(downloads_by_source.get("bw_jp2", []))
    bw_laz_count = len(downloads_by_source.get("bw_laz", []))
    by_jp2_count = len(downloads_by_source.get("by_jp2", []))
    by_laz_count = len(downloads_by_source.get("by_laz", []))

    # Calculate split factors per region (only for regions with tiles)
    nrw_tile_km = get_tile_size_km(Region.NRW)
    rlp_tile_km = get_tile_size_km(Region.RLP)
    bb_tile_km = get_tile_size_km(Region.BB)
    bw_tile_km = get_tile_size_km(Region.BW)
    by_tile_km = get_tile_size_km(Region.BY)
    nrw_has_tiles = nrw_jp2_count > 0 or nrw_laz_count > 0
    rlp_has_tiles = rlp_jp2_count > 0 or rlp_laz_count > 0
    bb_has_tiles = bb_jp2_count > 0 or bb_laz_count > 0
    bw_has_tiles = bw_jp2_count > 0 or bw_laz_count > 0
    by_has_tiles = by_jp2_count > 0 or by_laz_count > 0
    nrw_split = compute_split_factor(nrw_tile_km, grid_size_km) if nrw_has_tiles else 1
    rlp_split = compute_split_factor(rlp_tile_km, grid_size_km) if rlp_has_tiles else 1
    bb_split = compute_split_factor(bb_tile_km, grid_size_km) if bb_has_tiles else 1
    bw_split = compute_split_factor(bw_tile_km, grid_size_km) if bw_has_tiles else 1
    by_split = compute_split_factor(by_tile_km, grid_size_km) if by_has_tiles else 1
    nrw_split_side = int(nrw_split**0.5)  # e.g., 4 -> 2×2
    rlp_split_side = int(rlp_split**0.5)
    bb_split_side = int(bb_split**0.5)
    bw_split_side = int(bw_split**0.5)
    by_split_side = int(by_split**0.5)

    # Calculate outputs per region
    nrw_jp2_out = nrw_jp2_count * nrw_split
    nrw_laz_out = nrw_laz_count * nrw_split
    rlp_jp2_out = rlp_jp2_count * rlp_split
    rlp_laz_out = rlp_laz_count * rlp_split
    bb_jp2_out = bb_jp2_count * bb_split
    bb_laz_out = bb_laz_count * bb_split
    bw_jp2_out = bw_jp2_count * bw_split
    bw_laz_out = bw_laz_count * bw_split
    by_jp2_out = by_jp2_count * by_split
    by_laz_out = by_laz_count * by_split
    total_jp2_out = nrw_jp2_out + rlp_jp2_out + bb_jp2_out + bw_jp2_out + by_jp2_out
    total_laz_out = nrw_laz_out + rlp_laz_out + bb_laz_out + bw_laz_out + by_laz_out

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
            (
                "BB",
                f"{bb_tile_km:.0f}km",
                f"{grid_size_km}km",
                f"{bb_split_side}×{bb_split_side}",
                f"{bb_jp2_count} → {bb_jp2_out}",
                f"{bb_laz_count} → {bb_laz_out}",
            ),
            (
                "BW",
                f"{bw_tile_km:.0f}km",
                f"{grid_size_km}km",
                f"{bw_split_side}×{bw_split_side}",
                f"{bw_jp2_count} → {bw_jp2_out}",
                f"{bw_laz_count} → {bw_laz_out}",
            ),
            (
                "BY",
                f"{by_tile_km:.0f}km",
                f"{grid_size_km}km",
                f"{by_split_side}×{by_split_side}",
                f"{by_jp2_count} → {by_jp2_out}",
                f"{by_laz_count} → {by_laz_out}",
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
            print(f"  Sample missing JP2 tiles (zone, grid_x, grid_y): {sample}")
        if tile_set.missing_laz:
            sample = list(tile_set.missing_laz)[:5]
            print(f"Warning: {len(tile_set.missing_laz)} DSM tiles not available")
            print(f"  Sample missing DSM tiles (zone, grid_x, grid_y): {sample}")
        print("  Note: Tiles may be outside coverage area or not yet published")
    print()

    # STEP 3: Download raw tiles
    total_stats = ProcessingStats()
    download_tasks = []

    # Filter downloads based on --type flag
    if process_images:
        nrw_jp2_downloads = downloads_by_source.get("nrw_jp2", [])
        if nrw_jp2_downloads:
            download_tasks.append(DownloadTask("NRW Imagery", nrw_jp2_downloads, nrw_downloader))
        rlp_jp2_downloads = downloads_by_source.get("rlp_jp2", [])
        if rlp_jp2_downloads:
            download_tasks.append(DownloadTask("RLP Imagery", rlp_jp2_downloads, rlp_downloader))
        bb_jp2_downloads = downloads_by_source.get("bb_jp2", [])
        if bb_jp2_downloads:
            download_tasks.append(DownloadTask("BB Imagery", bb_jp2_downloads, bb_downloader))
        bw_jp2_downloads = downloads_by_source.get("bw_jp2", [])
        if bw_jp2_downloads:
            download_tasks.append(DownloadTask("BW Imagery", bw_jp2_downloads, bw_downloader))
        by_jp2_downloads = downloads_by_source.get("by_jp2", [])
        if by_jp2_downloads:
            download_tasks.append(DownloadTask("BY Imagery", by_jp2_downloads, by_downloader))
    if process_pointclouds:
        nrw_laz_downloads = downloads_by_source.get("nrw_laz", [])
        if nrw_laz_downloads:
            download_tasks.append(DownloadTask("NRW DSM", nrw_laz_downloads, nrw_downloader))
        rlp_laz_downloads = downloads_by_source.get("rlp_laz", [])
        if rlp_laz_downloads:
            download_tasks.append(DownloadTask("RLP DSM", rlp_laz_downloads, rlp_downloader))
        bb_laz_downloads = downloads_by_source.get("bb_laz", [])
        if bb_laz_downloads:
            download_tasks.append(DownloadTask("BB DSM", bb_laz_downloads, bb_downloader))
        bw_laz_downloads = downloads_by_source.get("bw_laz", [])
        if bw_laz_downloads:
            download_tasks.append(DownloadTask("BW DSM", bw_laz_downloads, bw_downloader))
        by_laz_downloads = downloads_by_source.get("by_laz", [])
        if by_laz_downloads:
            download_tasks.append(DownloadTask("BY DSM", by_laz_downloads, by_downloader))

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
        data_types.append("DSM (LAZ/TIF)")
    print(f"Converting {' and '.join(data_types)} to GeoTIFF format...")
    print(f"Target resolutions: {', '.join(f'{r}px' for r in resolutions)}")
    print()

    # Extract file paths from download list (respects year filter)
    image_files = [
        path for key, items in downloads_by_source.items()
        if "_jp2" in key for _, path in items
    ]
    dsm_files = [
        path for key, items in downloads_by_source.items()
        if "_laz" in key for _, path in items
    ]

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
        image_files=image_files,
        dsm_files=dsm_files,
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
            "DSM (LAZ/TIF)",
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
    print(f"  DSM outputs: {convert_stats.laz_converted} from {convert_stats.laz_sources} sources")
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
