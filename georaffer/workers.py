"""Conversion worker functions for ProcessPoolExecutor.

These functions must be top-level (not methods) because ProcessPoolExecutor
uses pickle, which cannot serialize methods. Each worker processes one source
file with all its target resolutions.
"""

import os
from pathlib import Path


def init_worker(threads_per_worker: int) -> None:
    """Initialize worker process environment.

    Sets thread limits for parallel libraries to avoid oversubscription
    when multiple workers run concurrently.

    Args:
        threads_per_worker: Maximum threads for parallel libraries (lazrs, numba)
    """
    threads_str = str(threads_per_worker)
    # lazrs uses Rayon (Rust parallel runtime)
    os.environ["RAYON_NUM_THREADS"] = threads_str
    # numba parallel loops
    os.environ["NUMBA_NUM_THREADS"] = threads_str
    # OpenMP (used by some GDAL operations)
    os.environ["OMP_NUM_THREADS"] = threads_str
    # Suppress "omp_set_nested deprecated" warning from GDAL/rasterio internals.
    # This is an upstream issue - GDAL still uses the deprecated OpenMP API.
    # See: https://github.com/OSGeo/gdal - waiting for fix to use omp_set_max_active_levels
    os.environ["KMP_WARNINGS"] = "0"


from georaffer.config import METERS_PER_KM, UTM_ZONE_STR, Region, get_tile_size_km
from georaffer.converters import convert_jp2, convert_laz, get_laz_year
from georaffer.converters.utils import parse_tile_coords
from georaffer.grids import compute_split_factor
from georaffer.metadata import get_wms_metadata_for_region
from georaffer.provenance import (
    build_metadata_rows,
    extract_year_from_filename,
    get_tile_center_utm,
)


def detect_region(filename: str) -> Region:
    """Detect source region from filename.

    Args:
        filename: Source filename

    Returns:
        Region enum (NRW or RLP)
    """
    # RLP files contain "_rp" or "_rp_" in the name
    if "_rp" in filename.lower():
        return Region.RLP
    return Region.NRW


def generate_output_name(
    filename: str,
    region: Region,
    year: str,
    tile_type: str,
) -> str:
    """Generate standardized output filename.

    Output format: {region}_{zone}_{easting}_{northing}_{year}.tif

    Args:
        filename: Source filename (used to extract coordinates)
        region: Source region
        year: Year string
        tile_type: 'image' or 'dsm' (for logging only)

    Returns:
        Standardized output filename
    """
    # Extract grid coordinates from filename
    match_result = parse_tile_coords(filename)
    if match_result:
        grid_x, grid_y = match_result
    else:
        grid_x, grid_y = 0, 0

    year_str = year if year else "latest"

    easting = grid_x * METERS_PER_KM
    northing = grid_y * METERS_PER_KM

    return f"{region.value.lower()}_{UTM_ZONE_STR}_{easting}_{northing}_{year_str}.tif"


def convert_jp2_worker(args: tuple) -> tuple[bool, list[dict], str, int]:
    """Worker function to convert a single JP2 file with all its resolutions.

    Args:
        args: Tuple of (filename, jp2_dir, processed_dir, resolutions,
              num_threads, grid_size_km, profiling)

    Returns:
        Tuple of (success, metadata_rows, filename, outputs_count) where:
        - success: True if conversion succeeded, False otherwise
        - metadata_rows: List of dict with provenance metadata for each output tile
        - filename: Original JP2 filename (for logging/tracking)
        - outputs_count: Number of GeoTIFF files created (accounts for splits and resolutions)

    Raises:
        RuntimeError: If conversion fails
    """
    filename, jp2_dir, processed_dir, resolutions, num_threads, grid_size_km, profiling = args

    input_path = Path(jp2_dir) / filename
    region = detect_region(filename)
    year = extract_year_from_filename(filename, require=True)

    # Setup output paths for each resolution
    output_paths: dict[int | None, str] = {}
    for res in resolutions:
        output_name = generate_output_name(filename, region, year, "image")
        res_dir = Path(processed_dir) / "image" / ("native" if res is None else str(res))
        res_dir.mkdir(parents=True, exist_ok=True)
        output_paths[res] = str(res_dir / output_name)

    try:
        convert_jp2(
            input_path,
            output_paths,
            region,
            year,
            resolutions,
            num_threads=num_threads,
            grid_size_km=grid_size_km,
            profiling=profiling,
        )

        # Get acquisition date from WMS for provenance
        acquisition_date = None
        metadata_source = None
        coords = parse_tile_coords(filename)
        if coords:
            base_x, base_y = coords
            tile_km = get_tile_size_km(region)
            center_x, center_y = get_tile_center_utm(base_x, base_y, tile_km)
            try:
                wms_meta = get_wms_metadata_for_region(
                    center_x, center_y, region, int(year) if year.isdigit() else None
                )
                if wms_meta:
                    acquisition_date = wms_meta.get("acquisition_date")
                    metadata_source = wms_meta.get("metadata_source")
            except Exception:
                pass  # WMS failures are not fatal

        # Build metadata rows using representative output path
        rep_path = next(iter(output_paths.values()))
        metadata = build_metadata_rows(
            filename=filename,
            output_path=rep_path,
            region=region,
            year=year,
            file_type="orthophoto",
            grid_size_km=grid_size_km,
            acquisition_date=acquisition_date,
            metadata_source=metadata_source,
        )

        # Calculate output count: resolutions × split tiles
        tile_km = get_tile_size_km(region)
        split_factor = compute_split_factor(tile_km, grid_size_km)
        outputs_count = len(resolutions) * split_factor

        return (True, metadata, filename, outputs_count)

    except Exception as e:
        raise RuntimeError(
            f"JP2 conversion failed for {filename} "
            f"(region={region}, year={year}, resolutions={resolutions})"
        ) from e


def convert_laz_worker(args: tuple) -> tuple[bool, list[dict], str, int]:
    """Worker function to convert a single LAZ file with all its resolutions.

    Args:
        args: Tuple of (filename, laz_dir, processed_dir, resolutions,
              num_threads, grid_size_km, profiling)

    Returns:
        Tuple of (success, metadata_rows, filename, outputs_count) where:
        - success: True if conversion succeeded, False otherwise
        - metadata_rows: List of dict with provenance metadata for each output tile
        - filename: Original LAZ filename (for logging/tracking)
        - outputs_count: Number of DSM GeoTIFF files created (accounts for splits and resolutions)

    Raises:
        RuntimeError: If conversion fails
    """
    filename, laz_dir, processed_dir, resolutions, num_threads, grid_size_km, profiling = args

    input_path = Path(laz_dir) / filename
    region = detect_region(filename)
    year = extract_year_from_filename(filename, require=False)

    # Fallback to LAZ header if year not in filename
    if year == "latest":
        header_year = get_laz_year(str(input_path))
        if header_year:
            year = header_year

    if not year or year == "latest":
        raise RuntimeError(f"Year not found in filename or LAS header: {filename}")

    # Setup output paths for each resolution
    output_paths: dict[int | None, str] = {}
    for res in resolutions:
        output_name = generate_output_name(filename, region, year, "dsm")
        res_dir = Path(processed_dir) / "dsm" / str(res)
        res_dir.mkdir(parents=True, exist_ok=True)
        output_paths[res] = str(res_dir / output_name)

    try:
        convert_laz(
            input_path,
            output_paths,
            region,
            target_sizes=resolutions,
            num_threads=num_threads,
            grid_size_km=grid_size_km,
            profiling=profiling,
        )

        # Get acquisition date from WMS for provenance
        acquisition_date = None
        metadata_source = None
        coords = parse_tile_coords(filename)
        if coords:
            base_x, base_y = coords
            tile_km = get_tile_size_km(region)
            center_x, center_y = get_tile_center_utm(base_x, base_y, tile_km)
            try:
                wms_meta = get_wms_metadata_for_region(
                    center_x, center_y, region, int(year) if year.isdigit() else None
                )
                if wms_meta:
                    acquisition_date = wms_meta.get("acquisition_date")
                    metadata_source = wms_meta.get("metadata_source")
            except Exception:
                pass  # WMS failures are not fatal

        # Build metadata rows using representative output path
        rep_path = next(iter(output_paths.values()))
        metadata = build_metadata_rows(
            filename=filename,
            output_path=rep_path,
            region=region,
            year=year,
            file_type="dsm",
            grid_size_km=grid_size_km,
            acquisition_date=acquisition_date,
            metadata_source=metadata_source,
        )

        # Calculate output count: resolutions × split tiles
        tile_km = get_tile_size_km(region)
        split_factor = compute_split_factor(tile_km, grid_size_km)
        outputs_count = len(resolutions) * split_factor

        return (True, metadata, filename, outputs_count)

    except Exception as e:
        raise RuntimeError(
            f"LAZ conversion failed for {filename} "
            f"(region={region}, year={year}, resolutions={resolutions}): {e}"
        ) from e
