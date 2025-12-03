"""LAZ to GeoTIFF DSM converter with optional RLP tile splitting.

CRITICAL ASSUMPTION: Pre-gridded DSM data only
================================================
This converter is designed EXCLUSIVELY for pre-gridded Digital Surface Model (DSM)
point clouds from official NRW/RLP datasets. It expects:

✓ Regular grid layout with exactly 1 point per cell
✓ Points in row-major order (west-to-east, north-to-south)
✓ Known grid spacing (NRW: 0.5m, RLP: 0.2m)
✓ Square tiles (NRW: 1km, RLP: 2km)

✗ Will NOT work with:
  - Unstructured point clouds (not gridded)
  - Multiple returns per cell
  - Sparse or irregular point distributions
  - Classified point clouds needing filtering

For raw point clouds, use PDAL or CloudCompare to create a regular DSM grid first:
  Example: pdal translate input.laz output_dsm.laz writers.gdal
           --writers.gdal.resolution=0.5 --writers.gdal.output_type=mean
"""

import os
import time
from pathlib import Path

import laspy
import numpy as np
from laspy import LazBackend
from numba import njit, prange
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin

from georaffer.config import (
    DSM_DTYPE,
    DSM_NODATA,
    METERS_PER_KM,
    REPROJECT_THREADS,
    Region,
    get_tile_size_km,
    laz_spacing_for_region,
)
from georaffer.converters.utils import (
    compute_split_bounds,
    generate_split_output_path,
    parse_tile_coords,
    resample_raster,
    write_geotiff,
)
from georaffer.grids import compute_split_factor


def get_laz_year(input_path: str) -> str | None:
    """Extract year from LAZ file header.

    Args:
        input_path: Path to LAZ file

    Returns:
        Year string or None if not available in header

    Raises:
        OSError: If file cannot be read
        laspy.errors.LaspyException: If file is corrupted
    """
    with laspy.open(input_path) as f:
        if f.header.creation_date:
            return str(f.header.creation_date.year)
    return None


@njit(parallel=True, cache=True)
def _fill_raster_numba(
    raster,
    x_int,
    y_int,
    z_int,
    x_scale,
    y_scale,
    z_scale,
    x_offset,
    y_offset,
    z_offset,
    min_x,
    max_y,
    inv_resolution,
    height,
    width,
):
    """Fill raster with point data using numba JIT (parallel).

    Operates on unscaled LAS integer coordinates to avoid Python-level copies.
    The raster array is provided by the caller to keep allocation outside the kernel.

    Performance strategy:
    - Uses numba's JIT compilation (@njit) for ~100x speedup over pure Python
    - Parallel execution via prange() distributes points across CPU cores
    - Works directly on integer LAS coordinates (x_int, y_int, z_int) to avoid
      memory copies from laspy - scaling happens inside the hot loop
    - Caller pre-allocates raster to keep allocation outside compiled code
    """
    n = len(x_int)
    for i in prange(n):
        # Scale from LAS integer storage to real-world coordinates
        x = x_int[i] * x_scale + x_offset
        y = y_int[i] * y_scale + y_offset
        z = z_int[i] * z_scale + z_offset

        # Convert UTM coordinates to pixel indices
        # Uses ties-to-even rounding (Python round()) for consistent pixel assignment
        # when points land exactly on cell boundaries
        col = round((x - min_x) * inv_resolution)
        row = round((max_y - y) * inv_resolution)
        if 0 <= row < height and 0 <= col < width:
            raster[row, col] = z

    return raster


def convert_laz(
    input_path: str,
    output_paths: str | dict[int, str],
    region: Region,
    resolution: float | None = None,
    target_sizes: list[int] | None = None,
    num_threads: int | None = None,
    stream_chunks: int | None = 4_000_000,
    grid_size_km: float = 1.0,
    profiling: bool = False,
) -> bool:
    """Convert LAZ to GeoTIFF DSM(s) with optional resampling and RLP splitting.

    Unified function handling both single and batch conversion.

    Args:
        input_path: Path to input LAZ file
        output_paths: Single output path (str) or dict mapping target_size -> path
        region: Region code ('NRW', 'RLP')
        resolution: Grid spacing in meters for initial rasterization (default: region-specific)
        target_sizes: List of target output sizes in pixels. None for native resolution.
        num_threads: Number of threads for reprojection (default: REPROJECT_THREADS)
        stream_chunks: Chunk size for LAZ streaming (default: 4M points)
        grid_size_km: User grid size to determine split factor
        profiling: Enable timing output for performance analysis

    Returns:
        True if successful

    Raises:
        RuntimeError: If lazrs-parallel backend unavailable, point cloud format invalid,
            or GeoTIFF write fails
        ValueError: If point count validation fails or grid structure is irregular

    Example:
        >>> # NRW: LAZ to 2000px DSM
        >>> convert_laz("raw/nrw.laz", "dsm.tif", Region.NRW, target_sizes=[2000])
        True

        >>> # RLP: 2km LAZ split to four 1km tiles at native resolution
        >>> convert_laz("raw/rlp.laz", {None: "dsm.tif"}, Region.RLP,
        ...             target_sizes=[None], grid_size_km=1.0)
        True
    """
    # Default resolution per region when not provided
    if resolution is None:
        resolution = laz_spacing_for_region(region)

    # Normalize to dict format
    if isinstance(output_paths, str):
        if target_sizes and len(target_sizes) == 1:
            output_paths = {target_sizes[0]: output_paths}
        else:
            output_paths = {None: output_paths}
            target_sizes = [None]
    elif target_sizes is None:
        target_sizes = list(output_paths.keys())

    threads = num_threads or REPROJECT_THREADS
    tile_km = get_tile_size_km(region)
    ratio_int = round(tile_km / grid_size_km) if grid_size_km > 0 else 1
    split_factor = compute_split_factor(tile_km, grid_size_km)

    # Validation: Check for required lazrs-parallel backend
    # This backend provides significant speedup over default lazrs for large files
    # Failure mode: RuntimeError with installation instructions
    try:
        available = LazBackend.detect_available()
        if LazBackend.LazrsParallel not in available:
            raise RuntimeError(
                "lazrs-parallel backend not available. Install lazrs-python>=0.6 from conda-forge."
            )
        t_decode_start = time.perf_counter()
        with laspy.open(
            input_path,
            laz_backend=LazBackend.LazrsParallel,
        ) as reader:
            header = reader.header

            # Extract year from header
            year = None
            if header.creation_date:
                try:
                    year = str(header.creation_date.year)
                except AttributeError:
                    pass  # Could not extract year

            # Grid extents from header mins/maxs
            header_min_x, header_min_y, _ = header.mins
            header_max_x, header_max_y, _ = header.maxs

            x_scale, y_scale, z_scale = header.scales
            x_offset, y_offset, z_offset = header.offsets

            # Calculate expected grid dimensions from header bounding box
            # Note: LAZ header max values are sometimes inclusive, sometimes exclusive
            # This causes num_points vs num_cells mismatches even for valid regular grids
            width = round((header_max_x - header_min_x) / resolution)
            height = round((header_max_y - header_min_y) / resolution)
            inv_resolution = 1.0 / resolution

            # CRITICAL: Detect sub-pixel offset between header bounds and actual points
            # RLP data has 0.1m offset (points at 0.1, 0.3, 0.5... not 0.0, 0.2, 0.4...)
            # NRW data has 0.0m offset (points align with header bounds)
            # Without correction, banker's rounding causes checkerboard artifacts
            #
            # Read first point to detect sub-pixel offset, then apply to header bounds.
            # Wrap in iter() so this works with any iterable (including lists in tests).
            first_chunk_iter = iter(reader.chunk_iterator(1))
            first_chunk = next(first_chunk_iter)
            first_x = first_chunk.X[0] * x_scale + x_offset
            first_y = first_chunk.Y[0] * y_scale + y_offset

            # Calculate sub-pixel offset (how far first point is from header bounds, mod resolution)
            x_subpixel = (first_x - header_min_x) % resolution
            y_subpixel = (first_y - header_min_y) % resolution

            # Apply sub-pixel offset to header bounds
            # min_x: shift right by offset (points start at min + offset)
            # max_y: shift down by offset (points end at max - offset)
            min_x = header_min_x + x_subpixel
            max_y = header_max_y - y_subpixel

            # Re-seek to beginning for full read
            reader.seek(0)

            # Validation: Check for regular grid structure
            # Expected: exactly one point per grid cell (perfectly gridded DSM)
            num_points = header.point_count
            num_cells = height * width
            if num_points != num_cells:
                # Mismatch detected - could be header issue or irregular data
                # Try recovery: if it's a perfect square, assume square grid
                side = round(num_points**0.5)
                if side * side == num_points:
                    # Recovery successful: square grid with header bounding box issue
                    # The header extent is slightly off, but the actual point spacing
                    # matches the region's nominal resolution. Keep resolution unchanged
                    # to avoid rounding errors from tiny extent/spacing mismatches.
                    width = height = side
                    num_cells = num_points
                else:
                    # Failure mode: ValueError for truly irregular point clouds
                    # Common causes:
                    #   - Point clouds with multiple returns per cell
                    #   - Sparse/filtered point clouds with gaps
                    #   - Unstructured scanning data (not gridded)
                    # Solution: Use PDAL/CloudCompare to create regular grid first
                    raise ValueError(
                        f"Non-regular point cloud: {num_points:,} points but {num_cells:,} cells expected. "
                        f"This converter is for regularly-spaced 2.5D DSM grids only."
                    )

            chunk_size = stream_chunks or 4_000_000
            raster = np.full((height, width), DSM_NODATA, dtype=np.float32)
            for pts in reader.chunk_iterator(chunk_size):
                raster = _fill_raster_numba(
                    raster,
                    pts.X,
                    pts.Y,
                    pts.Z,
                    x_scale,
                    y_scale,
                    z_scale,
                    x_offset,
                    y_offset,
                    z_offset,
                    min_x,
                    max_y,
                    inv_resolution,
                    height,
                    width,
                )
        t_decode = time.perf_counter() - t_decode_start
        # Transform origin is top-left corner of pixel (0,0), but min_x/max_y are pixel centers
        # Offset by half a pixel so pixel centers align with point positions
        half_res = resolution / 2
        transform = from_origin(min_x - half_res, max_y + half_res, resolution, resolution)
        crs = "EPSG:25832"  # UTM Zone 32N

        filename = os.path.basename(input_path)

        coords = parse_tile_coords(filename)
        if split_factor > 1:
            if not coords:
                raise RuntimeError(
                    f"Splitting failed: could not read grid coordinates from '{filename}'. "
                    "This tile name must include grid coords (e.g., …_350_5600_…)."
                )
            rows, cols = raster.shape
            if rows < ratio_int or cols < ratio_int:
                raise RuntimeError(
                    f"Splitting failed: the data read is only {rows}x{cols} pixels, "
                    f"but a {ratio_int}x{ratio_int} split needs at least {ratio_int} pixels on each side. "
                    "Try a higher read resolution or a coarser grid."
                )
            if rows % ratio_int != 0 or cols % ratio_int != 0:
                raise RuntimeError(
                    f"Splitting failed: the read window {rows}x{cols} does not divide evenly into a {ratio_int}x{ratio_int} grid. "
                    "Read at a resolution where both dimensions are multiples of the split ratio, or choose a compatible grid size."
                )
            return _convert_split_laz(
                raster,
                transform,
                crs,
                coords,
                output_paths,
                target_sizes,
                resolution,
                filename,
                region,
                year,
                ratio_int,
                grid_size_km=grid_size_km,
                num_threads=threads,
                profiling=profiling,
                t_decode=t_decode,
            )

        # Build provenance metadata
        metadata = {
            "source_file": filename,
            "source_region": region,
            "file_type": "dsm",
        }
        if year:
            metadata["acquisition_date"] = year

        # Standard conversion
        timings = []
        for target_size in target_sizes:
            path_str = output_paths.get(target_size)
            if not path_str:
                continue
            output_path = Path(path_str)

            if target_size:
                t_res_start = time.perf_counter()
                out_data, out_transform = resample_raster(
                    raster,
                    transform,
                    crs,
                    target_size,
                    threads,
                    dtype=np.float32,
                    resampling=Resampling.bilinear,
                    nodata=DSM_NODATA,
                )
                t_res = time.perf_counter() - t_res_start
            else:
                out_data, out_transform = raster, transform
                t_res = 0.0

            t_write_start = time.perf_counter()
            year_int = int(year) if year and year.isdigit() else None
            write_geotiff(
                output_path,
                out_data,
                out_transform,
                crs,
                dtype=DSM_DTYPE,
                count=1,
                nodata=DSM_NODATA,
                area_or_point="Point",
                metadata=metadata,
                year_int=year_int,
            )
            t_write = time.perf_counter() - t_write_start
            # Add provenance tags after write for DSM outputs
            from georaffer.metadata import add_provenance_to_geotiff

            add_provenance_to_geotiff(output_path, metadata, year=year_int)
            timings.append((target_size, t_res, t_write))

        if profiling:
            total_resample = sum(t[1] for t in timings)
            total_write = sum(t[2] for t in timings)
            total = t_decode + total_resample + total_write
            timing_str = "; ".join(
                f"res={ts or 'native'} resample={t_res:.3f}s write={t_w:.3f}s"
                for ts, t_res, t_w in timings
            )
            print(
                f"[laz] {os.path.basename(input_path)} region={region} split_factor={split_factor} "
                f"decode={t_decode:.3f}s resample_total={total_resample:.3f}s write_total={total_write:.3f}s total={total:.3f}s | {timing_str}"
            )

        return True

    except Exception as e:
        # Capture original exception with context for debugging parallel runs
        raise RuntimeError(f"Failed to convert {input_path}: {e.__class__.__name__}: {e}") from e


def _convert_split_laz(
    raster: np.ndarray,
    transform: Affine,
    crs: str,
    coords: tuple[int, int],
    output_paths: dict[int, str],
    target_sizes: list[int],
    resolution: float,
    source_file: str,
    region: str,
    year: str | None,
    ratio: int,
    grid_size_km: float = 1.0,
    num_threads: int = REPROJECT_THREADS,
    profiling: bool = False,
    t_decode: float = 0.0,
) -> bool:
    """Split a DSM raster evenly into ratio x ratio subtiles and write each."""
    base_x, base_y = coords
    rows, cols = raster.shape
    tile_size_m = METERS_PER_KM  # grid indices are kilometer-based
    grid_size_m = round(grid_size_km * METERS_PER_KM)

    total_resample = 0.0
    total_write = 0.0

    # Split the DSM raster into ratio × ratio sub-tiles
    # Same coordinate logic as JP2 splitting (see jp2.py for detailed explanation)
    for r_idx in range(ratio):
        row_start, row_end = compute_split_bounds(r_idx, rows, ratio)
        for c_idx in range(ratio):
            col_start, col_end = compute_split_bounds(c_idx, cols, ratio)
            tile_raster = raster[row_start:row_end, col_start:col_end]

            # Calculate new grid coordinates (km indices)
            # X increases east (cols), Y increases north
            # Row index inverted: row 0 = north (max Y), increasing rows go south
            new_x = base_x + c_idx
            new_y = base_y + (ratio - 1 - r_idx)

            # Adjust geotransform for sub-tile's pixel offset
            split_transform = transform * Affine.translation(col_start, row_start)

            metadata = {
                "source_file": source_file,
                "source_region": region,
                "file_type": "dsm",
                "split_from": f"{base_x}_{base_y}",
                "grid_x": new_x,
                "grid_y": new_y,
            }
            if year:
                metadata["acquisition_date"] = year

            for target_size in target_sizes:
                base_path = output_paths.get(target_size)
                if not base_path:
                    continue

                # Embedding precise UTM coords ensures unique filenames for split outputs
                # (grid coords alone could be ambiguous for different split ratios)
                easting = int(base_x * tile_size_m + c_idx * grid_size_m)
                northing = int(base_y * tile_size_m + (ratio - 1 - r_idx) * grid_size_m)

                output_path = generate_split_output_path(
                    base_path, new_x, new_y, easting=easting, northing=northing
                )

                t_res_start = time.perf_counter()
                if target_size:
                    out_data, out_transform = resample_raster(
                        tile_raster,
                        split_transform,
                        crs,
                        target_size,
                        num_threads,
                        dtype=np.float32,
                        resampling=Resampling.bilinear,
                        nodata=DSM_NODATA,
                    )
                else:
                    out_data, out_transform = tile_raster, split_transform
                total_resample += time.perf_counter() - t_res_start

                t_write_start = time.perf_counter()
                year_int = int(year) if year and year.isdigit() else None
                write_geotiff(
                    output_path,
                    out_data,
                    out_transform,
                    crs,
                    dtype=DSM_DTYPE,
                    count=1,
                    nodata=DSM_NODATA,
                    area_or_point="Point",
                    metadata=metadata,
                    year_int=year_int,
                )
                total_write += time.perf_counter() - t_write_start

    if profiling:
        split_factor = ratio * ratio
        total = t_decode + total_resample + total_write
        print(
            f"[laz] {source_file} region={region} split={ratio}x{ratio}={split_factor} "
            f"decode={t_decode:.3f}s resample={total_resample:.3f}s write={total_write:.3f}s total={total:.3f}s"
        )

    return True
