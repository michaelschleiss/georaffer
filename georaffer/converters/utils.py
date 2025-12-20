"""Shared utilities for converters."""

import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
from rasterio.transform import Affine

from georaffer.config import (
    BB_BDOM_PATTERN,
    BB_DOP_PATTERN,
    NRW_JP2_PATTERN,
    NRW_LAZ_PATTERN,
    RLP_JP2_PATTERN,
    RLP_LAZ_PATTERN,
)

# Pattern for processed output files: {region}_{zone}_{easting}_{northing}_{year}.tif
# Easting is 6 digits, northing is 7 digits.
OUTPUT_FILE_PATTERN = re.compile(
    r"(?:nrw|rlp|bb)_(?:32|33)_(\d{6})_(\d{7})_\d{4}(?:_\d+)?\.tif$"
)


@contextmanager
def atomic_rasterio_write(output_path: Path, *args, **kwargs):
    """Context manager for atomic GeoTIFF writes using temp file + rename.

    Writes to a temporary file first, then atomically renames to the final path
    on success. If writing fails or an exception occurs, the temporary file is
    cleaned up and the original file (if it exists) remains unchanged.

    Args:
        output_path: Final output file path
        *args: Positional arguments passed to rasterio.open()
        **kwargs: Keyword arguments passed to rasterio.open()

    Yields:
        rasterio.DatasetWriter for writing raster data

    Raises:
        Any exceptions from rasterio.open() or file operations are propagated
        after cleanup of temporary files

    Example:
        with atomic_rasterio_write(output_path, 'w', driver='GTiff', ...) as dst:
            dst.write(data, 1)
    """
    temp_path = output_path.parent / (output_path.name + ".tmp")
    try:
        with rasterio.open(temp_path, *args, **kwargs) as dst:
            yield dst
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def parse_rlp_tile_coords(filename: str) -> tuple[int, int] | None:
    """Extract grid coordinates from RLP filename.

    Args:
        filename: RLP tile filename (JP2 or LAZ)

    Returns:
        (grid_x, grid_y) tuple or None if pattern doesn't match
    """
    for pattern in [RLP_JP2_PATTERN, RLP_LAZ_PATTERN]:
        match = pattern.search(filename)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def parse_tile_coords(filename: str) -> tuple[int, int] | None:
    """Extract grid coordinates from NRW or RLP style filenames.

    Handles:
    - NRW JP2: dop10rgbi_32_350_5600_1_nw_2021.jp2 → (350, 5600) [raw input tile coords]
    - NRW LAZ: bdom50_32350_5600_1_nw_2025.laz → (350, 5600) [raw input tile coords]
    - RLP JP2: dop20rgb_32_362_5604_2_rp_2023.jp2 → (362, 5604) [raw input tile coords]
    - RLP LAZ: bdom20rgbi_32_364_5582_2_rp.laz → (364, 5582) [raw input tile coords]
    - BB bDOM: bdom_33250-5888.zip → (250, 5888) [raw input tile coords]
    - BB DOP: dop_33250-5888.zip → (250, 5888) [raw input tile coords]
    - Output files: nrw_32_350500_5600000_2021.tif → (350500, 5600000) [UTM coordinates]

    Pattern matching strategy:
      1. Try NRW patterns first (strict validation catches format errors early)
      2. Fall back to RLP patterns (different naming conventions)
      3. Finally check processed output files (UTM coordinates in meters)
      4. Return None if no match (signals invalid/unexpected filename)

    Returns None for:
      - Wrong region prefix (e.g., "hessen_..." instead of nrw/rlp)
      - Malformed coordinates (non-numeric, wrong format)
      - Missing required components (zone, year, coordinates)
      - Completely unrecognized filename patterns

    This defensive approach ensures only valid, parseable files enter the pipeline.
    """
    # Try NRW patterns first (they handle zone prefix correctly)
    for pattern in [NRW_JP2_PATTERN, NRW_LAZ_PATTERN]:
        match = pattern.search(filename)
        if match:
            return int(match.group(1)), int(match.group(2))

    # Try RLP patterns
    rlp_result = parse_rlp_tile_coords(filename)
    if rlp_result:
        return rlp_result

    # Try BB patterns (zone prefix + km coords)
    for pattern in (BB_BDOM_PATTERN, BB_DOP_PATTERN):
        match = pattern.match(filename)
        if match:
            east_code = match.group(1)
            grid_x = int(east_code[2:])
            grid_y = int(match.group(2))
            return grid_x, grid_y

    # Processed output files: {region}_{zone}_{easting}_{northing}_{year}.tif
    # Uses UTM coordinates in meters (e.g., 350500, 5600000)
    match = OUTPUT_FILE_PATTERN.search(filename)
    if match:
        return int(match.group(1)), int(match.group(2))

    # No match - caller should handle None appropriately
    return None


def get_rlp_quadrant_splits(half: int, original_x: int, original_y: int):
    """Get slice definitions for splitting RLP 2km tile into 4x1km quadrants.

    Args:
        half: Half tile size in pixels
        original_x, original_y: Original 2km grid coordinates

    Returns:
        List of (row_slice, col_slice, new_grid_x, new_grid_y) tuples where:
        - row_slice: Slice object for extracting rows from source array
        - col_slice: Slice object for extracting columns from source array
        - new_grid_x, new_grid_y: Grid coordinates for this quadrant (km indices)
        Order: [SW, SE, NW, NE] quadrants
    """
    return [
        (slice(half, None), slice(None, half), original_x, original_y),  # SW
        (slice(half, None), slice(half, None), original_x + 1, original_y),  # SE
        (slice(None, half), slice(None, half), original_x, original_y + 1),  # NW
        (slice(None, half), slice(half, None), original_x + 1, original_y + 1),  # NE
    ]


def generate_split_output_path(
    base_output_path: str,
    grid_x: int,
    grid_y: int,
    *,
    easting: int | None = None,
    northing: int | None = None,
    utm_zone: str = "32",
) -> Path:
    """Generate output path for a split tile.

    Args:
        base_output_path: Original output path (e.g., /path/rlp_32_362000_5604000_2023.tif)
        grid_x, grid_y: Grid coordinates for the split tile (km indices)
        easting, northing: UTM coordinates (meters) for sub-km splits.

    Returns:
        Modified output path with new UTM coordinates to avoid collisions.
    """
    path = Path(base_output_path)
    parts = path.stem.split("_")

    if easting is None or northing is None:
        raise ValueError("Split output naming requires easting/northing UTM coordinates.")
    if len(parts) < 5:
        raise ValueError(
            "Output path must be {region}_{zone}_{easting}_{northing}_{year}.tif to build split outputs."
        )
    region = parts[0]
    year = parts[4]
    new_name = f"{region}_{utm_zone}_{int(easting)}_{int(northing)}_{year}{path.suffix}"
    return path.parent / new_name


def compute_split_bounds(idx: int, total: int, ratio: int) -> tuple[int, int]:
    """Calculate start/end indices for splitting a tile.

    Args:
        idx: Current index (row or column) in the split grid
        total: Total size in pixels (rows or columns)
        ratio: Split ratio (e.g., 2 for 2×2 split)

    Returns:
        Tuple of (start_pixel, end_pixel) for this segment where:
        - start_pixel: Inclusive start index in the array
        - end_pixel: Exclusive end index in the array (Python slice convention)
    """
    start = round(idx * total / ratio)
    end = round((idx + 1) * total / ratio)
    return start, end


def resample_raster(
    data: np.ndarray,
    src_transform: Affine,
    src_crs,
    target_size: int,
    num_threads: int,
    dtype: np.dtype,
    resampling: Resampling,
    nodata: float | None = None,
) -> tuple[np.ndarray, Affine]:
    """Resample raster data to target size using rasterio's warp.reproject.

    Handles both 2D (single band) and 3D (multi-band) arrays. Output is always
    square (target_size × target_size). Maintains aspect ratio by scaling both
    dimensions proportionally.

    Args:
        data: Input array, shape (h, w) for single band or (bands, h, w) for multi-band
        src_transform: Source affine transform
        src_crs: Source coordinate reference system
        target_size: Target size in pixels (output will be target_size × target_size)
        num_threads: Number of threads for reprojection
        dtype: Output data type (e.g., np.uint8, np.float32)
        resampling: Resampling method (e.g., Resampling.lanczos, Resampling.bilinear)
        nodata: Optional nodata value for src and dst

    Returns:
        Tuple of (resampled_data, output_transform) where:
        - resampled_data: Array with shape (target_size, target_size) or (bands, target_size, target_size)
        - output_transform: Adjusted Affine transform for the resampled raster

    Raises:
        ValueError: If data.ndim is not 2 or 3
        rasterio.errors.CRSError: If src_crs is invalid
    """
    # Handle both 2D and 3D arrays
    if data.ndim == 2:
        output_shape = (target_size, target_size)
        scale_x = data.shape[1] / target_size
        scale_y = data.shape[0] / target_size
    else:
        output_shape = (data.shape[0], target_size, target_size)
        scale_x = data.shape[2] / target_size
        scale_y = data.shape[1] / target_size

    output_transform = src_transform * Affine.scale(scale_x, scale_y)
    if nodata is None:
        output_data = np.zeros(output_shape, dtype=dtype)
    else:
        output_data = np.full(output_shape, nodata, dtype=dtype)

    reproject_kwargs = {
        "src_transform": src_transform,
        "src_crs": src_crs,
        "dst_transform": output_transform,
        "dst_crs": src_crs,
        "resampling": resampling,
        "num_threads": num_threads,
        "init_dest_nodata": False,
    }
    if nodata is not None:
        reproject_kwargs["src_nodata"] = nodata
        reproject_kwargs["dst_nodata"] = nodata

    if data.ndim == 2:
        rasterio.warp.reproject(
            source=data,
            destination=output_data,
            **reproject_kwargs,
        )
    else:
        for band_index in range(data.shape[0]):
            rasterio.warp.reproject(
                source=data[band_index],
                destination=output_data[band_index],
                **reproject_kwargs,
            )
    return output_data, output_transform


def uniquify_output_path(path: Path) -> Path:
    """Return a unique path, reserving it atomically to avoid races across processes.

    Uses atomic file creation (O_EXCL) to prevent TOCTOU race conditions when
    multiple processes try to write to the same path simultaneously. If the path
    is available, it's reserved by creating an empty file. If taken, generates a
    unique temporary filename in the same directory.

    Args:
        path: Desired output file path

    Returns:
        The original path if available, otherwise a unique temporary path in the
        same directory with pattern "{prefix}__{random}{suffix}"
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic create - fails if file exists (cross-platform)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return path
    except FileExistsError:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f"{path.stem}__", suffix=path.suffix, dir=path.parent
        )
        os.close(fd)
        return Path(tmp_path)


def write_geotiff(
    output_path: Path,
    data: np.ndarray,
    transform: Affine,
    crs,
    dtype: str = "uint8",
    count: int = 3,
    nodata: float | None = None,
    area_or_point: str = "Area",
    metadata: dict | None = None,
    year_int: int | None = None,
) -> None:
    """Write data to GeoTIFF with optional provenance metadata.

    Supports two data types with different conventions:

    Orthophoto (imagery):
        - dtype='uint8': RGB color values 0-255
        - count=3: Red, Green, Blue bands
        - nodata=None: No invalid pixels in imagery
        - area_or_point='Area': Pixels represent averaged color over cell area

    DSM (Digital Surface Model):
        - dtype='float32': Elevation in meters (0.06mm precision at 500m)
        - count=1: Single elevation band
        - nodata=-9999.0: Standard sentinel for invalid/missing elevation
        - area_or_point='Point': Pixels represent point samples at cell center
          (critical for accurate slope/aspect calculations)

    Args:
        output_path: Output file path
        data: Array data (bands, height, width) for multi-band or (height, width) for single
        transform: Affine transform
        crs: Coordinate reference system
        dtype: Data type (default: 'uint8')
        count: Number of bands (default: 3)
        nodata: NoData value (optional)
        area_or_point: 'Area' for imagery, 'Point' for DSM
        metadata: Provenance metadata dict (optional)
        year_int: Year for WMS lookup (optional)

    Example:
        >>> # RGB orthophoto (data shape: (3, 1000, 1000))
        >>> write_geotiff(Path('out.tif'), rgb_data, transform, 'EPSG:25832',
        ...               dtype='uint8', count=3, metadata={'source_region': 'NRW'})

        >>> # DSM with nodata (data shape: (2000, 2000))
        >>> write_geotiff(Path('dsm.tif'), elevation_data, transform, 'EPSG:25832',
        ...               dtype='float32', count=1, nodata=-9999.0, area_or_point='Point')
    """
    # Import here to avoid circular dependency
    from datetime import datetime

    from georaffer.config import Region
    from georaffer.metadata import get_wms_metadata_for_region

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle shape for single vs multi-band
    if data.ndim == 2:
        height, width = data.shape
        actual_count = 1
    else:
        actual_count, height, width = data.shape

    # Prepare all metadata tags BEFORE atomic write (WMS lookup done here)
    tags = {"AREA_OR_POINT": area_or_point}
    if metadata:
        center_x = transform.c + (width / 2) * transform.a
        center_y = transform.f + (height / 2) * transform.e

        # Query WMS for precise acquisition date if region provided
        source_region = metadata.get("source_region")
        if source_region:
            # Handle both Region enum and string values
            if isinstance(source_region, Region):
                region_enum = source_region
            elif source_region == "NRW":
                region_enum = Region.NRW
            elif source_region == "RLP":
                region_enum = Region.RLP
            else:
                region_enum = None

            if region_enum:
                wms_metadata = get_wms_metadata_for_region(
                    center_x, center_y, region_enum, year_int
                )
                if wms_metadata:
                    metadata.update(wms_metadata)

        # Build tags dict
        if metadata.get("acquisition_date"):
            tags["ACQUISITION_DATE"] = str(metadata["acquisition_date"])
        if metadata.get("source_region"):
            tags["SOURCE_REGION"] = metadata["source_region"]
        if metadata.get("source_file"):
            tags["SOURCE_FILE"] = metadata["source_file"]
        if metadata.get("file_type"):
            tags["SOURCE_TYPE"] = metadata["file_type"]
        if metadata.get("metadata_source"):
            tags["METADATA_SOURCE"] = metadata["metadata_source"]

        tags["PROCESSING_DATE"] = datetime.now().strftime("%Y-%m-%d")

    # Use actual band count from data, not default parameter
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": actual_count,
        "height": height,
        "width": width,
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if nodata is not None:
        profile["nodata"] = nodata

    # Atomic write with all tags included
    with atomic_rasterio_write(output_path, "w", **profile) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            dst.write(data)
        dst.update_tags(**tags)
