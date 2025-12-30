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
    filtered_urls: dict[str, tuple[dict, dict]],
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
        filtered_urls: Dict mapping region name -> (jp2_urls, laz_urls) dicts

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
        name = downloader.region_name.lower()
        zone = zone_by_region[name]
        jp2_catalog, laz_catalog = filtered_urls[name]

        jp2_tiles = jp2_catalog.keys()
        laz_tiles = laz_catalog.keys()

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


def calculate_required_tiles(
    tiles_by_zone: dict[int, set[tuple[int, int]]],
    grid_size_km: float,
    downloaders: list[RegionDownloader],
    zone_by_region: dict[str, int],
    original_coords: np.ndarray | None = None,
    source_zone: int = 32,
) -> tuple[TileSet, dict[str, list[tuple[str, str]]]]:
    """Map user-grid tiles to native grids, check availability, build download lists.

    Args:
        tiles_by_zone: Dict mapping UTM zone -> set of (grid_x, grid_y) user tiles.
            Each zone's tiles are in that zone's coordinate system.
        grid_size_km: User's grid resolution in km
        downloaders: List of region downloaders in priority order (first match wins)
        zone_by_region: Dict mapping region name -> UTM zone number
        original_coords: Optional Nx2 array of original (easting, northing) coords
            in source_zone. If provided, enables precise cross-zone missing tile detection.
        source_zone: UTM zone of original_coords (default 32)

    Returns:
        Tuple of (tile_set, downloads_by_source) where:
        - tile_set: TileSet object with:
          - jp2/laz: Dict[region_name, Set[(grid_x, grid_y)]] of available tiles
          - missing_jp2/missing_laz: Set[(zone, grid_x, grid_y)] of unavailable tiles
        - downloads_by_source: Dict[source_key, List[(url, output_path)]] where:
          - source_key: '{region}_{type}' (e.g., 'nrw_jp2', 'rlp_laz')
          - url: Download URL for the tile
          - output_path: Local path to save the downloaded file

    Algorithm overview:
        User defines an arbitrary grid (e.g., 0.5km). Data providers publish in fixed
        native grids (NRW: 1km, RLP: 2km). This function maps user tiles to provider
        tiles by converting to UTM, checking which native tile each user tile falls
        within, and verifying that tile exists in the provider's catalog.

        Each region only receives tiles from its native UTM zone, ensuring correct
        coordinate mapping (Zone 32 for NRW/RLP, Zone 33 for BB).

    Example:
        >>> tiles_by_zone = {32: {(350, 5600), (351, 5600)}}
        >>> zone_by_region = {'nrw': 32}
        >>> tile_set, downloads = calculate_required_tiles(
        ...     tiles_by_zone, 1.0, [nrw_downloader], zone_by_region
        ... )
    """
    # Get filtered URLs from each downloader
    filtered_urls: dict[str, tuple[dict, dict]] = {}
    for downloader in downloaders:
        name = downloader.region_name.lower()
        filtered_urls[name] = downloader.get_filtered_tile_urls()

    # ========== Phase 1: Map user tiles to native tiles per region ==========
    # For each region, get tiles from its native zone and check catalog availability.
    # Each region only sees tiles in its own coordinate system.
    jp2_natives: dict[str, set[tuple[int, int]]] = {
        d.region_name.lower(): set() for d in downloaders
    }
    laz_natives: dict[str, set[tuple[int, int]]] = {
        d.region_name.lower(): set() for d in downloaders
    }

    for downloader in downloaders:
        name = downloader.region_name.lower()
        zone = zone_by_region[name]
        zone_tiles = tiles_by_zone.get(zone, set())
        jp2_urls, laz_urls = filtered_urls[name]

        for user_tile in zone_tiles:
            # Convert user grid coords to UTM center point (in meters)
            # These coords are in the region's native zone coordinate system
            utm_x, utm_y = user_tile_to_utm_center(user_tile[0], user_tile[1], grid_size_km)

            # Ask region downloader: which native tile contains this UTM point?
            grid_coords, _ = downloader.utm_to_grid_coords(utm_x, utm_y)
            native_tile = (int(grid_coords[0]), int(grid_coords[1]))

            # Check JP2 catalog
            if native_tile in jp2_urls:
                jp2_natives[name].add(native_tile)

            # Check LAZ catalog
            if native_tile in laz_urls:
                laz_natives[name].add(native_tile)

    # ========== Phase 2: Build download lists and tile set ==========
    # For each unique native tile, look up its URL and create download spec.
    # Special handling for multi-year NRW downloads (--historic-since flag).
    tile_set = TileSet()
    downloads: dict[str, list[tuple[str, str]]] = {}

    for downloader in downloaders:
        name = downloader.region_name.lower()
        downloads[f"{name}_jp2"] = []
        downloads[f"{name}_laz"] = []
        tile_set.jp2[name] = set()
        tile_set.laz[name] = set()
        jp2_urls, laz_urls = filtered_urls[name]

        # JP2 downloads - support multi-year mode (e.g., NRW catalogs, RLP WMS)
        for native_tile in jp2_natives[name]:
            url = jp2_urls[native_tile]
            tile_set.jp2[name].add(native_tile)

            # Multi-year handling: If downloader loaded multiple years of data for the
            # same tile coords, download all of them (e.g., NRW 2020, 2021, 2022 for tile 350,5600)
            all_urls = downloader.get_all_urls_for_coord(native_tile) or None

            if all_urls:
                # Multi-year: add ALL URLs for this coord (different years = different files)
                for u in all_urls:
                    if hasattr(downloader, "image_filename_from_url"):
                        filename = downloader.image_filename_from_url(u)
                    else:
                        filename = _filename_from_url(u)
                    p = os.path.join(downloader.raw_dir, "image", filename)
                    downloads[f"{name}_jp2"].append((u, p))
            else:
                # Standard single-year mode or fallback
                if hasattr(downloader, "image_filename_from_url"):
                    filename = downloader.image_filename_from_url(url)
                else:
                    filename = _filename_from_url(url)
                path = os.path.join(downloader.raw_dir, "image", filename)
                downloads[f"{name}_jp2"].append((url, path))

        # LAZ downloads (no multi-year support currently)
        for native_tile in laz_natives[name]:
            url = laz_urls[native_tile]
            tile_set.laz[name].add(native_tile)
            if hasattr(downloader, "dsm_filename_from_url"):
                filename = downloader.dsm_filename_from_url(url)
            else:
                filename = os.path.basename(url)
            path = os.path.join(downloader.raw_dir, "dsm", filename)
            downloads[f"{name}_laz"].append((url, path))

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
        coords, source_zone, grid_size_km, downloaders, zone_by_region, filtered_urls
    )

    return tile_set, downloads
