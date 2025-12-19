"""Provenance metadata collection utilities for processed tiles.

This module provides functions to build metadata rows for provenance tracking
during tile conversion. It extracts year from filenames and assembles metadata
dictionaries that get written to provenance.csv.
"""

import re
from datetime import datetime
from pathlib import Path

from georaffer.config import METERS_PER_KM, Region, get_tile_size_km
from georaffer.converters.utils import generate_split_output_path, parse_tile_coords
from georaffer.grids import compute_split_factor


def extract_year_from_filename(filename: str, require: bool = False) -> str:
    """Extract year from filename.

    Looks for a 4-digit year pattern immediately before the file extension,
    e.g., 'tile_2021.jp2' -> '2021'.

    Args:
        filename: Source filename
        require: If True, raise RuntimeError if no 4-digit year is found

    Returns:
        Year string or 'latest' if not found and require=False

    Raises:
        RuntimeError: If require=True and no year found
    """
    match = re.search(r"_(\d{4})\.", filename)
    if match:
        return match.group(1)
    if require:
        raise RuntimeError(f"Year not found in filename: {filename}")
    return "latest"


def compute_split_coordinates(
    base_x: int,
    base_y: int,
    tile_km: float,
    grid_size_km: float,
) -> list[tuple[int, int]]:
    """Compute output tile coordinates from a source tile after splitting.

    When the source tile is larger than the user's grid, it gets split into
    multiple output tiles. This function computes the grid coordinates for
    each output tile.

    Args:
        base_x: Source tile X coordinate (km index)
        base_y: Source tile Y coordinate (km index)
        tile_km: Source tile size in kilometers
        grid_size_km: User's target grid size in kilometers

    Returns:
        List of (grid_x, grid_y) tuples for each output tile.
        If no splitting occurs, returns [(base_x, base_y)].

    Raises:
        ValueError: If base coordinates are None (unparseable filename)
    """
    if base_x is None or base_y is None:
        raise ValueError(
            "Cannot compute coordinates: base_x and base_y are required. "
            "Source filename must contain parseable grid coordinates."
        )

    split_factor = compute_split_factor(tile_km, grid_size_km)

    if split_factor == 1:
        return [(base_x, base_y)]

    ratio_int = round(tile_km / grid_size_km)
    coords = []
    for r in range(ratio_int):
        for c in range(ratio_int):
            # Row-major order, top-to-bottom then left-to-right
            coords.append((base_x + c, base_y + (ratio_int - 1 - r)))
    return coords


def build_metadata_rows(
    filename: str,
    output_path: str,
    region: Region,
    year: str,
    file_type: str,
    grid_size_km: float,
    acquisition_date: str | None = None,
    metadata_source: str | None = None,
) -> list[dict]:
    """Build provenance metadata rows for a converted tile.

    Handles both single-tile output and split output cases. For split tiles,
    creates one metadata row per output tile with correct grid coordinates.

    Args:
        filename: Source filename (used to parse coordinates)
        output_path: Representative output path (used to derive split paths)
        region: Source region (NRW or RLP)
        year: Year string from filename or header
        file_type: Type of output ('orthophoto' or 'dsm')
        grid_size_km: User's target grid size in kilometers
        acquisition_date: Optional acquisition date from WMS
        metadata_source: Optional metadata source identifier

    Returns:
        List of metadata dictionaries, one per output tile. Each dict contains:
        - processed_file: Path to output GeoTIFF
        - source_file: Original source filename
        - source_region: Region name (NRW or RLP)
        - grid_x, grid_y: Grid coordinates (km indices)
        - year: Acquisition year
        - file_type: 'orthophoto' or 'dsm'
        - acquisition_date: Optional precise date from WMS
        - metadata_source: Optional source identifier
        - conversion_date: ISO timestamp when GeoTIFF was created
    """
    tile_km = get_tile_size_km(region)
    tile_size_m = METERS_PER_KM  # Grid indices are kilometer-based
    grid_size_m = round(grid_size_km * METERS_PER_KM)
    split_factor = compute_split_factor(tile_km, grid_size_km)

    # Parse coordinates from source filename - fail loudly if unparseable
    coords = parse_tile_coords(filename)
    if coords is None:
        raise ValueError(
            f"Cannot parse grid coordinates from filename: {filename}. "
            "Source files must have parseable NRW/RLP/BB naming patterns."
        )
    base_x, base_y = coords

    # Compute output tile coordinates
    split_coords = compute_split_coordinates(base_x, base_y, tile_km, grid_size_km)

    metadata_rows = []
    for gx, gy in split_coords:
        # Determine output path for this tile
        if split_factor > 1:
            dx = gx - base_x
            dy = gy - base_y
            easting = int(base_x * tile_size_m + dx * grid_size_m)
            northing = int(base_y * tile_size_m + dy * grid_size_m)
            out_path = generate_split_output_path(
                output_path, gx, gy, easting=easting, northing=northing
            )
        else:
            out_path = output_path

        metadata_rows.append({
            "processed_file": Path(out_path).name,
            "source_file": filename,
            "source_region": region.value,
            "year": year,
            "file_type": file_type,
            "grid_x": gx,
            "grid_y": gy,
            "acquisition_date": acquisition_date,
            "metadata_source": metadata_source,
            "conversion_date": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        })

    return metadata_rows


def get_tile_center_utm(
    base_x: int,
    base_y: int,
    tile_km: float,
) -> tuple[float, float]:
    """Get UTM center coordinates for a tile.

    Args:
        base_x: Tile X coordinate (km index)
        base_y: Tile Y coordinate (km index)
        tile_km: Tile size in kilometers

    Returns:
        (utm_x, utm_y) center coordinates in meters
    """
    center_x = (base_x + tile_km / 2.0) * METERS_PER_KM
    center_y = (base_y + tile_km / 2.0) * METERS_PER_KM
    return center_x, center_y
