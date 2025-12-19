"""Tile entities and catalog resolution logic."""

import os
import re
from dataclasses import dataclass, field
from typing import Any, NamedTuple
from urllib.parse import parse_qs, urlparse

from georaffer.grids import user_tile_to_utm_center


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


class TileCoord(NamedTuple):
    """Immutable tile coordinate."""

    x: int
    y: int


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
    """

    # Tiles per region: {region_name: {coords}}
    jp2: dict[str, set[tuple[int, int]]] = field(default_factory=dict)
    laz: dict[str, set[tuple[int, int]]] = field(default_factory=dict)
    missing_jp2: set[tuple[int, int]] = field(default_factory=set)
    missing_laz: set[tuple[int, int]] = field(default_factory=set)

    @property
    def total_jp2(self) -> int:
        """Total JP2 tiles to download."""
        return sum(len(tiles) for tiles in self.jp2.values())

    @property
    def total_laz(self) -> int:
        """Total LAZ tiles to download."""
        return sum(len(tiles) for tiles in self.laz.values())

    def jp2_count(self, region: str) -> int:
        """JP2 tile count for a specific region."""
        return len(self.jp2.get(region, set()))

    def laz_count(self, region: str) -> int:
        """LAZ tile count for a specific region."""
        return len(self.laz.get(region, set()))


@dataclass
class DownloadSpec:
    """Specification for a single tile download."""

    url: str
    path: str
    source: str  # '{region}_jp2' or '{region}_laz'


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


def calculate_required_tiles(
    user_tiles: set[tuple[int, int]],
    grid_size_km: float,
    regions: list[RegionCatalog],
) -> tuple[TileSet, dict[str, list[tuple[str, str]]]]:
    """Map user-grid tiles to native grids, check availability, build download lists.

    Args:
        user_tiles: Set of (grid_x, grid_y) in user grid coordinates
        grid_size_km: User's grid resolution in km
        regions: List of RegionCatalog in priority order (first match wins)

    Returns:
        Tuple of (tile_set, downloads_by_source) where:
        - tile_set: TileSet object with:
          - jp2/laz: Dict[region_name, Set[(grid_x, grid_y)]] of available tiles
          - missing_jp2/missing_laz: Set[(grid_x, grid_y)] of unavailable tiles
        - downloads_by_source: Dict[source_key, List[(url, output_path)]] where:
          - source_key: '{region}_{type}' (e.g., 'nrw_jp2', 'rlp_laz')
          - url: Download URL for the tile
          - output_path: Local path to save the downloaded file

    Algorithm overview:
        User defines an arbitrary grid (e.g., 0.5km). Data providers publish in fixed
        native grids (NRW: 1km, RLP: 2km). This function maps user tiles to provider
        tiles by converting to UTM, checking which native tile each user tile falls
        within, and verifying that tile exists in the provider's catalog.

    Example:
        >>> # Map user tiles to native grid, identify missing tiles
        >>> user_tiles = {(350, 5600), (351, 5600)}
        >>> catalog = RegionCatalog('nrw', downloader,
        ...     jp2_catalog={(350, 5600): 'url1', (351, 5600): 'url2'},
        ...     laz_catalog={(350, 5600): 'url3'})
        >>> tile_set, downloads = calculate_required_tiles(user_tiles, 1.0, [catalog])
        >>> tile_set.jp2['nrw']  # Both JP2 tiles available
        {(350, 5600), (351, 5600)}
        >>> tile_set.missing_laz  # (351, 5600) LAZ not in catalog
        {(351, 5600)}
    """
    # Build normalized indexes for all catalogs (handles 2 or 3-tuple keys)
    jp2_indexes = {r.name: _build_catalog_index(r.jp2_catalog) for r in regions}
    laz_indexes = {r.name: _build_catalog_index(r.laz_catalog) for r in regions}

    # ========== Phase 1: Map user tiles to native tiles ==========
    # For each user tile, find which provider tile (if any) covers it.
    # Priority order: first region in the list that has the tile wins.
    # This allows NRW to take precedence over RLP for overlapping coverage areas.
    jp2_mapping: dict[tuple[int, int], tuple[str, tuple] | None] = {}
    laz_mapping: dict[tuple[int, int], tuple[str, tuple] | None] = {}

    for user_tile in user_tiles:
        # Convert user grid coords to UTM center point (in meters)
        utm_x, utm_y = user_tile_to_utm_center(user_tile[0], user_tile[1], grid_size_km)

        # Try each region in priority order for JP2
        jp2_mapping[user_tile] = None
        for region in regions:
            # Ask region downloader: which native tile contains this UTM point?
            coords, _ = region.downloader.utm_to_grid_coords(utm_x, utm_y)
            native = (int(coords[0]), int(coords[1]))
            # Check if that native tile is actually available in the catalog
            if native in jp2_indexes[region.name]:
                jp2_mapping[user_tile] = (region.name, native)
                break  # First match wins - stop searching other regions

        # Try each region in priority order for LAZ (independent from JP2)
        laz_mapping[user_tile] = None
        for region in regions:
            coords, _ = region.downloader.utm_to_grid_coords(utm_x, utm_y)
            native = (int(coords[0]), int(coords[1]))
            if native in laz_indexes[region.name]:
                laz_mapping[user_tile] = (region.name, native)
                break

    # ========== Phase 2: Collect unique native tiles per region ==========
    # Multiple user tiles may map to the same native tile (e.g., four 0.5km user tiles
    # all fall within one 1km NRW tile). Deduplicate to avoid downloading the same tile
    # multiple times.
    jp2_natives: dict[str, set[tuple[int, int]]] = {r.name: set() for r in regions}
    laz_natives: dict[str, set[tuple[int, int]]] = {r.name: set() for r in regions}

    for mapping in jp2_mapping.values():
        if mapping:
            region_name, native = mapping
            jp2_natives[region_name].add(native)

    for mapping in laz_mapping.values():
        if mapping:
            region_name, native = mapping
            laz_natives[region_name].add(native)

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

        # JP2 downloads - support multi-year mode (NRW only currently)
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
                    p = os.path.join(region.downloader.raw_dir, "image", _filename_from_url(u))
                    downloads[f"{name}_jp2"].append((u, p))
            else:
                # Standard single-year mode or fallback
                path = os.path.join(region.downloader.raw_dir, "image", _filename_from_url(url))
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

    # ========== Phase 4: Calculate missing tiles ==========
    # Track which user tiles couldn't be satisfied by any region (for reporting)
    tile_set.missing_jp2 = {ut for ut, m in jp2_mapping.items() if m is None}
    tile_set.missing_laz = {ut for ut, m in laz_mapping.items() if m is None}

    return tile_set, downloads
