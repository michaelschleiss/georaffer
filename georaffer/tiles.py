"""Tile entities and catalog resolution logic."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

import numpy as np

from georaffer.config import METERS_PER_KM
from georaffer.grids import reproject_utm_coords_vectorized, user_tile_to_utm_center

if TYPE_CHECKING:
    from georaffer.downloaders.base import RegionDownloader


def _filename_from_url(url: str) -> str:
    """Extract or generate a proper filename from a URL.

    For regular URLs (ATOM feed), extracts basename.
    For WMS GetMap URLs, generates a filename from parameters.

    Args:
        url: Download URL (either file URL or WMS GetMap URL)

    Returns:
        Filename like "dop20rgb_32_380_5540_2_rp_2020.tif"
    """
    # Check if this is a WMS GetMap URL
    if "GetMap" in url and "BBOX" in url:
        # Parse URL query parameters
        parsed = urlparse(url)
        params = parse_qs(parsed.query)

        # Extract BBOX to get grid coordinates
        bbox = params.get("BBOX", [""])[0].split(",")
        if len(bbox) >= 2:
            minx = int(float(bbox[0]) / 1000)  # Convert meters to km
            miny = int(float(bbox[1]) / 1000)

        # Extract year from layer name (e.g., "rp_dop20_rgb_2020")
        layer = params.get("LAYERS", [""])[0]
        year_match = re.search(r"_(\d{4})$", layer)
        year = int(year_match.group(1)) if year_match else 0

        # Determine extension from format
        fmt = params.get("FORMAT", [""])[0]
        ext = ".tif" if "tiff" in fmt.lower() else ".png"

        # Generate filename matching RLP JP2 naming convention
        return f"dop20rgb_32_{minx}_{miny}_2_rp_{year}{ext}"

    # Standard URL - extract basename
    return os.path.basename(url)


@dataclass
class TileSet:
    """Tracks tiles from different sources.

    Coordinates are stored as (grid_x, grid_y) tuples representing
    kilometer indices in the respective grid system.
    Missing sets store (zone, grid_x, grid_y) to disambiguate zones.
    """

    # Tiles per region: {region_name: {coords}}
    jp2: dict[str, set[tuple[int, int]]] = field(default_factory=dict)
    laz: dict[str, set[tuple[int, int]]] = field(default_factory=dict)
    missing_jp2: set[tuple[int, int, int]] = field(default_factory=set)
    missing_laz: set[tuple[int, int, int]] = field(default_factory=set)

    def jp2_count(self, region: str) -> int:
        """JP2 tile count for a specific region."""
        return len(self.jp2.get(region, set()))

    def laz_count(self, region: str) -> int:
        """LAZ tile count for a specific region."""
        return len(self.laz.get(region, set()))


def check_missing_coords(
    coords: np.ndarray,
    source_zone: int,
    grid_size_km: float,
    downloaders: list[RegionDownloader],
    zone_by_region: dict[str, int],
) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int]]]:
    """Check which original coords have no coverage from any region.

    Uses vectorized operations for speed. For each original coordinate,
    checks if ANY region (across all zones) has data for that location.

    Args:
        coords: Nx2 array of (easting, northing) in source_zone
        source_zone: UTM zone of input coordinates
        grid_size_km: User's grid resolution in km
        downloaders: List of region downloaders
        zone_by_region: Dict mapping region name -> UTM zone

    Returns:
        (missing_jp2, missing_laz) - sets of (zone, x, y) tiles with no coverage
    """
    n = len(coords)
    if n == 0:
        return set(), set()

    jp2_covered = np.zeros(n, dtype=bool)
    laz_covered = np.zeros(n, dtype=bool)
    grid_size_m = grid_size_km * METERS_PER_KM

    for downloader in downloaders:
        zone = zone_by_region[downloader.region_name.lower()]
        catalog = downloader.build_catalog()

        # Get tile coordinate sets from catalog
        jp2_tiles = catalog.image_tiles.keys()
        laz_tiles = catalog.dsm_tiles.keys()

        # Vectorized reproject to region's zone
        xs, ys = reproject_utm_coords_vectorized(coords, source_zone, zone)

        jp2_tuples = []
        laz_tuples = []
        for utm_x, utm_y in zip(xs, ys):
            jp2_coords, laz_coords = downloader.utm_to_grid_coords(
                float(utm_x), float(utm_y)
            )
            jp2_tuples.append((int(jp2_coords[0]), int(jp2_coords[1])))
            laz_tuples.append((int(laz_coords[0]), int(laz_coords[1])))

        # Check each coord's tile against catalogs (set lookup is O(1))
        jp2_covered |= np.array([t in jp2_tiles for t in jp2_tuples])
        laz_covered |= np.array([t in laz_tiles for t in laz_tuples])

    # Convert uncovered coords to source_zone tiles for reporting
    source_tile_xs = (coords[:, 0] // grid_size_m).astype(int)
    source_tile_ys = (coords[:, 1] // grid_size_m).astype(int)

    missing_jp2: set[tuple[int, int, int]] = {
        (source_zone, int(source_tile_xs[i]), int(source_tile_ys[i]))
        for i in range(n)
        if not jp2_covered[i]
    }
    missing_laz: set[tuple[int, int, int]] = {
        (source_zone, int(source_tile_xs[i]), int(source_tile_ys[i]))
        for i in range(n)
        if not laz_covered[i]
    }

    return missing_jp2, missing_laz


def build_filtered_download_list(
    tiles_by_zone: dict[int, set[tuple[int, int]]],
    grid_size_km: float,
    downloaders: list[RegionDownloader],
    zone_by_region: dict[str, int],
    original_coords: np.ndarray | None = None,
    source_zone: int = 32,
) -> tuple[TileSet, dict[str, list[tuple[str, str]]]]:
    """Map user tiles to native grids, filter by year, build download list.

    Args:
        tiles_by_zone: Dict mapping UTM zone -> set of (grid_x, grid_y) user tiles.
        grid_size_km: User's grid resolution in km
        downloaders: List of region downloaders
        zone_by_region: Dict mapping region name -> UTM zone number
        original_coords: Optional Nx2 array of original coords for missing tile detection
        source_zone: UTM zone of original_coords (default 32)

    Returns:
        Tuple of (tile_set, downloads_by_source)
    """
    tile_set = TileSet()
    downloads: dict[str, list[tuple[str, str]]] = {}

    # Track which native tiles we've already processed (for deduplication)
    seen_jp2: dict[str, set[tuple[int, int]]] = {}
    seen_laz: dict[str, set[tuple[int, int]]] = {}

    for downloader in downloaders:
        name = downloader.region_name.lower()
        zone = zone_by_region[name]
        zone_tiles = tiles_by_zone.get(zone, set())
        catalog = downloader.build_catalog()

        # Get year filter from downloader
        from_year, to_year = None, None
        if downloader.imagery_from:
            from_year, to_year = downloader.imagery_from

        downloads[f"{name}_jp2"] = []
        downloads[f"{name}_laz"] = []
        tile_set.jp2[name] = set()
        tile_set.laz[name] = set()
        seen_jp2[name] = set()
        seen_laz[name] = set()

        for user_tile in zone_tiles:
            # Convert user grid coords to UTM center point
            utm_x, utm_y = user_tile_to_utm_center(user_tile[0], user_tile[1], grid_size_km)

            # Map to native tile
            grid_coords, _ = downloader.utm_to_grid_coords(utm_x, utm_y)
            native_tile = (int(grid_coords[0]), int(grid_coords[1]))

            # JP2: check catalog and add matching years
            if native_tile not in seen_jp2[name]:
                years_data = catalog.image_tiles.get(native_tile, {})
                if years_data:
                    # Filter years, then pick latest only if no range specified
                    valid_years = {
                        y: tile for y, tile in years_data.items()
                        if downloader._year_in_range(y, from_year, to_year)
                    }
                    if valid_years:
                        # No filter = latest only, filter = all matching years
                        years_to_download = (
                            valid_years if from_year is not None
                            else {max(valid_years): valid_years[max(valid_years)]}
                        )
                        for year, tile_info in years_to_download.items():
                            url = tile_info["url"]
                            if hasattr(downloader, "image_filename_from_url"):
                                filename = downloader.image_filename_from_url(url)
                            else:
                                filename = _filename_from_url(url)
                            path = os.path.join(downloader.raw_dir, "image", filename)
                            downloads[f"{name}_jp2"].append((url, path))
                            tile_set.jp2[name].add(native_tile)
                seen_jp2[name].add(native_tile)

            # LAZ/DSM: check catalog (use latest year available)
            if native_tile not in seen_laz[name]:
                years_data = catalog.dsm_tiles.get(native_tile, {})
                if years_data:
                    # Use latest year available
                    latest_year = max(years_data)
                    tile_info = years_data[latest_year]
                    url = tile_info["url"]
                    if hasattr(downloader, "dsm_filename_from_url"):
                        filename = downloader.dsm_filename_from_url(url)
                    else:
                        filename = os.path.basename(url)
                    path = os.path.join(downloader.raw_dir, "dsm", filename)
                    downloads[f"{name}_laz"].append((url, path))
                    tile_set.laz[name].add(native_tile)
                seen_laz[name].add(native_tile)

    # ========== Phase 3: Calculate missing tiles ==========
    coords_from_input = original_coords is not None
    if original_coords is None:
        coords = np.empty((0, 2), dtype=float)
    else:
        coords = np.asarray(original_coords)
        if coords.size == 0:
            coords = np.empty((0, 2), dtype=float)
            coords_from_input = False
        elif coords.ndim == 1:
            if coords.size != 2:
                raise ValueError("original_coords must be an Nx2 array.")
            coords = coords.reshape(1, 2)
    source_tiles = tiles_by_zone.get(source_zone, set())
    if coords.size == 0:
        if source_tiles:
            coords = np.array(
                [user_tile_to_utm_center(x, y, grid_size_km) for x, y in source_tiles]
            )
    elif coords_from_input and source_tiles:
        grid_size_m = grid_size_km * METERS_PER_KM
        coord_tiles = set(
            zip(
                (coords[:, 0] // grid_size_m).astype(int),
                (coords[:, 1] // grid_size_m).astype(int),
            )
        )
        extra_tiles = source_tiles - coord_tiles
        if extra_tiles:
            extra_coords = np.array(
                [user_tile_to_utm_center(x, y, grid_size_km) for x, y in extra_tiles]
            )
            coords = np.vstack((coords, extra_coords))

    tile_set.missing_jp2, tile_set.missing_laz = check_missing_coords(
        coords, source_zone, grid_size_km, downloaders, zone_by_region
    )

    return tile_set, downloads
