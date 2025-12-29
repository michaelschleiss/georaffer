"""Tile entities and catalog resolution logic."""

import os
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from georaffer.config import METERS_PER_KM
from georaffer.grids import reproject_utm_coords_vectorized, user_tile_to_utm_center


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
class RegionCatalog:
    """Catalog for a single region's tiles."""

    name: str
    downloader: Any
    jp2_catalog: dict[tuple[int, int], str]
    laz_catalog: dict[tuple[int, int], str]


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


def _build_catalog_index(
    catalog: dict[tuple[int, int], str],
) -> dict[tuple[int, int], tuple[tuple, str]]:
    """Build normalized index for catalog (handles 2 or 3-tuple keys).

    Args:
        catalog: Input catalog mapping tile coords -> URL

    Returns:
        Dict mapping normalized (x, y) coords -> (original_key, url) tuple
        - Key: (grid_x, grid_y) normalized to 2-tuple
        - Value: (original_tile_key, download_url) where original_tile_key
          may be 2-tuple (x, y) or 3-tuple (x, y, year) depending on source
    """
    index: dict[tuple[int, int], tuple[tuple, str]] = {}
    for tile_key, url in catalog.items():
        key_coord = tile_key[:2] if len(tile_key) == 3 else tile_key
        index[key_coord] = (tile_key, url)
    return index


def check_missing_coords(
    coords: np.ndarray,
    source_zone: int,
    grid_size_km: float,
    regions: list["RegionCatalog"],
    zone_by_region: dict[str, int],
) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int]]]:
    """Check which original coords have no coverage from any region.

    Uses vectorized operations for speed. For each original coordinate,
    checks if ANY region (across all zones) has data for that location.

    Args:
        coords: Nx2 array of (easting, northing) in source_zone
        source_zone: UTM zone of input coordinates
        grid_size_km: User's grid resolution in km
        regions: List of RegionCatalog objects
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

    for region in regions:
        zone = zone_by_region[region.name]

        # Build catalog tile sets (normalized to 2-tuple keys)
        jp2_tiles = {k[:2] if len(k) == 3 else k for k in region.jp2_catalog.keys()}
        laz_tiles = {k[:2] if len(k) == 3 else k for k in region.laz_catalog.keys()}

        # Vectorized reproject to region's zone
        xs, ys = reproject_utm_coords_vectorized(coords, source_zone, zone)

        jp2_tuples = []
        laz_tuples = []
        if hasattr(region.downloader, "utm_to_grid_coords"):
            for utm_x, utm_y in zip(xs, ys):
                jp2_coords, laz_coords = region.downloader.utm_to_grid_coords(
                    float(utm_x), float(utm_y)
                )
                jp2_tuples.append((int(jp2_coords[0]), int(jp2_coords[1])))
                laz_tuples.append((int(laz_coords[0]), int(laz_coords[1])))
        else:
            # Vectorized snap to native grid
            # Get native grid size from downloader (1000m for NRW/BB, 2000m for RLP)
            native_size_m = getattr(region.downloader, "GRID_SIZE", None)
            if not isinstance(native_size_m, (int, float)):
                # Fallback: try to infer from region name
                native_size_m = 2000 if "rlp" in region.name.lower() else 1000

            tile_xs = (xs // native_size_m).astype(int)
            tile_ys = (ys // native_size_m).astype(int)

            # RLP special case: tiles are 2km but indexed by km coordinate
            if native_size_m == 2000:
                tile_xs *= 2
                tile_ys *= 2

            jp2_tuples = list(zip(tile_xs, tile_ys))
            laz_tuples = list(zip(tile_xs, tile_ys))

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
    regions: list[RegionCatalog],
    zone_by_region: dict[str, int],
    original_coords: np.ndarray | None = None,
    source_zone: int = 32,
) -> tuple[TileSet, dict[str, list[tuple[str, str]]]]:
    """Map user-grid tiles to native grids, check availability, build download lists.

    Args:
        tiles_by_zone: Dict mapping UTM zone -> set of (grid_x, grid_y) user tiles.
            Each zone's tiles are in that zone's coordinate system.
        grid_size_km: User's grid resolution in km
        regions: List of RegionCatalog in priority order (first match wins within zone)
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
        >>> # Map user tiles to native grid, identify missing tiles
        >>> tiles_by_zone = {32: {(350, 5600), (351, 5600)}}
        >>> catalog = RegionCatalog('nrw', downloader,
        ...     jp2_catalog={(350, 5600): 'url1', (351, 5600): 'url2'},
        ...     laz_catalog={(350, 5600): 'url3'})
        >>> zone_by_region = {'nrw': 32}
        >>> tile_set, downloads = calculate_required_tiles(tiles_by_zone, 1.0, [catalog], zone_by_region)
        >>> tile_set.jp2['nrw']  # Both JP2 tiles available
        {(350, 5600), (351, 5600)}
        >>> tile_set.missing_laz  # (351, 5600) LAZ not in catalog
        {(32, 351, 5600)}
    """
    # Build normalized indexes for all catalogs (handles 2 or 3-tuple keys)
    jp2_indexes = {r.name: _build_catalog_index(r.jp2_catalog) for r in regions}
    laz_indexes = {r.name: _build_catalog_index(r.laz_catalog) for r in regions}

    # ========== Phase 1: Map user tiles to native tiles per region ==========
    # For each region, get tiles from its native zone and check catalog availability.
    # Each region only sees tiles in its own coordinate system.
    jp2_natives: dict[str, set[tuple[int, int]]] = {r.name: set() for r in regions}
    laz_natives: dict[str, set[tuple[int, int]]] = {r.name: set() for r in regions}

    # Track which user tiles (per zone) were matched, for missing tile reporting
    jp2_matched: set[tuple[int, int, int]] = set()  # (zone, x, y)
    laz_matched: set[tuple[int, int, int]] = set()

    for region in regions:
        zone = zone_by_region[region.name]
        zone_tiles = tiles_by_zone.get(zone, set())

        for user_tile in zone_tiles:
            # Convert user grid coords to UTM center point (in meters)
            # These coords are in the region's native zone coordinate system
            utm_x, utm_y = user_tile_to_utm_center(user_tile[0], user_tile[1], grid_size_km)

            # Ask region downloader: which native tile contains this UTM point?
            coords, _ = region.downloader.utm_to_grid_coords(utm_x, utm_y)
            native = (int(coords[0]), int(coords[1]))

            # Check JP2 catalog
            if native in jp2_indexes[region.name]:
                jp2_natives[region.name].add(native)
                jp2_matched.add((zone, user_tile[0], user_tile[1]))

            # Check LAZ catalog
            if native in laz_indexes[region.name]:
                laz_natives[region.name].add(native)
                laz_matched.add((zone, user_tile[0], user_tile[1]))

    # ========== Phase 3: Build download lists and tile set ==========
    # For each unique native tile, look up its URL and create download spec.
    # Special handling for multi-year NRW downloads (--historic-since flag).
    tile_set = TileSet()
    downloads: dict[str, list[tuple[str, str]]] = {}

    for region in regions:
        name = region.name
        downloads[f"{name}_jp2"] = []
        downloads[f"{name}_laz"] = []
        tile_set.jp2[name] = set()
        tile_set.laz[name] = set()

        # JP2 downloads - support multi-year mode (e.g., NRW catalogs, RLP WMS)
        for native in jp2_natives[name]:
            tile_key, url = jp2_indexes[name][native]
            tile_set.jp2[name].add(tile_key)

            # Multi-year handling: If downloader loaded multiple years of data for the
            # same tile coords, download all of them (e.g., NRW 2020, 2021, 2022 for tile 350,5600)
            all_urls = None
            if hasattr(region.downloader, "get_all_urls_for_coord"):
                result = region.downloader.get_all_urls_for_coord(native)
                # Only use if it's actually a non-empty list (handles MagicMock in tests)
                if isinstance(result, list) and result:
                    all_urls = result

            if all_urls:
                # Multi-year: add ALL URLs for this coord (different years = different files)
                for u in all_urls:
                    if hasattr(region.downloader, "image_filename_from_url"):
                        filename = region.downloader.image_filename_from_url(u)
                    else:
                        filename = _filename_from_url(u)
                    p = os.path.join(region.downloader.raw_dir, "image", filename)
                    downloads[f"{name}_jp2"].append((u, p))
            else:
                # Standard single-year mode or fallback
                if hasattr(region.downloader, "image_filename_from_url"):
                    filename = region.downloader.image_filename_from_url(url)
                else:
                    filename = _filename_from_url(url)
                path = os.path.join(region.downloader.raw_dir, "image", filename)
                downloads[f"{name}_jp2"].append((url, path))

        # LAZ downloads (no multi-year support currently)
        for native in laz_natives[name]:
            tile_key, url = laz_indexes[name][native]
            tile_set.laz[name].add(tile_key)
            if hasattr(region.downloader, "dsm_filename_from_url"):
                filename = region.downloader.dsm_filename_from_url(url)
            else:
                filename = os.path.basename(url)
            path = os.path.join(region.downloader.raw_dir, "dsm", filename)
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
        coords, source_zone, grid_size_km, regions, zone_by_region
    )

    return tile_set, downloads
