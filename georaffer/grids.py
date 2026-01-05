"""Grid calculations, UTM transforms, and tile coordinate utilities."""

import numpy as np
import utm

from georaffer.config import METERS_PER_KM, UTM_ZONE, UTM_ZONE_BY_REGION, Region, get_tile_size_km


def latlon_to_utm(
    lat: float, lon: float, *, force_zone_number: int | None = UTM_ZONE
) -> tuple[float, float]:
    """Convert lat/lon (WGS84) to UTM coordinates.

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        force_zone_number: Optional UTM zone number override. None = auto-detect.

    Returns:
        (easting, northing) tuple in meters
    """
    easting, northing, _, _ = utm.from_latlon(lat, lon, force_zone_number=force_zone_number)
    return easting, northing


def latlon_array_to_utm(
    lats: np.ndarray, lons: np.ndarray, *, force_zone_number: int | None = UTM_ZONE
) -> tuple[np.ndarray, np.ndarray]:
    """Convert arrays of lat/lon (WGS84) to UTM coordinates.

    Vectorized version for efficient bulk conversion.

    Args:
        lats: Array of latitudes in decimal degrees
        lons: Array of longitudes in decimal degrees

    Returns:
        (eastings, northings) arrays in meters
    """
    eastings, northings, _, _ = utm.from_latlon(lats, lons, force_zone_number=force_zone_number)
    return eastings, northings


def generate_user_grid_tiles(
    utm_coords: list[tuple[float, float]], grid_size_km: float, margin_km: float
) -> set[tuple[int, int]]:
    """Generate user-grid tile coordinates within margin of given UTM points.

    Args:
        utm_coords: List of (easting, northing) coordinates in UTM meters
        grid_size_km: User's working grid resolution in kilometers
        margin_km: Buffer distance from tile border in kilometers

    Returns:
        Set of (grid_x, grid_y) tuples in user grid coordinates (km indices)

    Example:
        >>> # Single point at 350500m E, 5600000m N with 1km grid and no margin
        >>> coords = [(350500, 5600000)]
        >>> tiles = generate_user_grid_tiles(coords, grid_size_km=1.0, margin_km=0)
        >>> tiles
        {(350, 5600)}

        >>> # Same point with 1km margin - adds surrounding tiles
        >>> tiles = generate_user_grid_tiles(coords, grid_size_km=1.0, margin_km=1.0)
        >>> len(tiles)  # 3x3 grid around center
        9
    """
    user_tiles: set[tuple[int, int]] = set()
    grid_size_m = grid_size_km * METERS_PER_KM
    margin_m = margin_km * METERS_PER_KM

    for utm_x, utm_y in utm_coords:
        # Snap to user grid
        center_x = int(utm_x // grid_size_m)
        center_y = int(utm_y // grid_size_m)

        # Calculate how many tiles we need in each direction from border
        # Margin is from tile border, so center tile + margin tiles
        margin_tiles = int(np.ceil(margin_m / grid_size_m)) if margin_km > 0 else 0

        # Generate grid
        for dx in range(-margin_tiles, margin_tiles + 1):
            for dy in range(-margin_tiles, margin_tiles + 1):
                user_tiles.add((center_x + dx, center_y + dy))

    return user_tiles


def dedupe_by_output_tile(
    utm_coords: list[tuple[float, float]], tile_size_km: float
) -> list[tuple[float, float]]:
    """Deduplicate UTM coordinates at the output tile level.

    Args:
        utm_coords: List of (easting, northing) coordinates in UTM meters
        tile_size_km: Output tile size in kilometers

    Returns:
        List of deduplicated coordinates that map to unique output tiles
    """
    if not utm_coords:
        return []
    coords_array = np.array(utm_coords)
    grid_size_m = tile_size_km * METERS_PER_KM
    tile_x = (coords_array[:, 0] // grid_size_m).astype(int)
    tile_y = (coords_array[:, 1] // grid_size_m).astype(int)
    tiles = np.column_stack([tile_x, tile_y])
    _, unique_indices = np.unique(tiles, axis=0, return_index=True)
    return coords_array[unique_indices].tolist()


def compute_split_factor(tile_km: float, grid_size_km: float) -> int:
    """Calculate how many output tiles to create from one source tile.

    Splitting is allowed only when the source tile side length is an integer
    multiple (>=2) of the requested grid size.

    Why integer ratios only?
      - Ensures each output tile has equal dimensions (no partial/remainder tiles)
      - Simplifies pixel math: each sub-tile gets exactly (width/ratio) × (height/ratio) pixels
      - Avoids edge cases with overlapping or gap pixels between sub-tiles
      - Example: 2km RLP tile split to 1km grid = 2×2 = 4 tiles (clean)
      - Counter-example: 2km tile to 0.7km grid = 2.86x (messy, rejected)

    Args:
        tile_km: Source tile size in kilometers (e.g., 2.0 for RLP)
        grid_size_km: User's requested grid size in kilometers

    Returns:
        Number of output tiles (split_factor^2 for 2D grid)

    Raises:
        RuntimeError: If grid_size_km is non-positive or ratio isn't a clean integer >=2

    Example:
        >>> # RLP 2km tile split to 1km grid = 2x2 = 4 outputs
        >>> compute_split_factor(tile_km=2.0, grid_size_km=1.0)
        4

        >>> # NRW 1km tile with 1km grid = no split needed
        >>> compute_split_factor(tile_km=1.0, grid_size_km=1.0)
        1

        >>> # Invalid: 2km tile to 0.7km grid (non-integer ratio)
        >>> compute_split_factor(tile_km=2.0, grid_size_km=0.7)
        RuntimeError: Splitting isn't possible...
    """
    if grid_size_km <= 0:
        raise RuntimeError("Grid size must be positive for splitting.")
    ratio = tile_km / grid_size_km
    if ratio < 2:
        # No splitting needed: user grid is same size or larger than native tile
        return 1
    ratio_rounded = round(ratio)
    if abs(ratio - ratio_rounded) < 1e-6 and ratio_rounded >= 2:
        return ratio_rounded * ratio_rounded  # 2D split: ratio² output tiles
    raise RuntimeError(
        f"Splitting isn't possible: the tile is {tile_km:.3g} km and your grid is {grid_size_km:.3g} km "
        f"({ratio:.2f}x). Splitting needs a whole-number ratio (2x, 3x, ...). "
        "Choose a grid size that divides the tile size evenly."
    )


def user_tile_to_utm_center(grid_x: int, grid_y: int, grid_size_km: float) -> tuple[float, float]:
    """Convert user grid tile coordinates to UTM center point.

    Args:
        grid_x: Grid X coordinate (km index)
        grid_y: Grid Y coordinate (km index)
        grid_size_km: Grid size in kilometers

    Returns:
        (easting, northing) of tile center in meters
    """
    grid_size_m = grid_size_km * METERS_PER_KM
    utm_x = grid_x * grid_size_m + grid_size_m / 2
    utm_y = grid_y * grid_size_m + grid_size_m / 2
    return utm_x, utm_y


def tile_to_utm_center(tile_x: int, tile_y: int, tile_size_m: float) -> tuple[float, float]:
    """Convert km-indexed tile coordinates to UTM center point.

    Tile indices are km-based (e.g., 350, 5600 means origin at 350km, 5600km).
    The tile_size_m determines the center offset within the tile.

    Args:
        tile_x: Tile X coordinate (km index)
        tile_y: Tile Y coordinate (km index)
        tile_size_m: Tile size in meters (e.g., 1000 for 1km, 2000 for 2km)

    Returns:
        (easting, northing) of tile center in meters
    """
    utm_x = tile_x * METERS_PER_KM + tile_size_m / 2
    utm_y = tile_y * METERS_PER_KM + tile_size_m / 2
    return utm_x, utm_y


def reproject_utm_coords(
    coords: list[tuple[float, float]],
    source_zone: int,
    target_zone: int,
) -> list[tuple[float, float]]:
    """Reproject UTM coordinates from one zone to another.

    Converts through lat/lon as intermediate step.

    Args:
        coords: List of (easting, northing) in source zone
        source_zone: Source UTM zone number (e.g., 32)
        target_zone: Target UTM zone number (e.g., 33)

    Returns:
        List of (easting, northing) in target zone
    """
    if source_zone == target_zone:
        return coords

    result = []
    for utm_x, utm_y in coords:
        # Convert to lat/lon (assumes northern hemisphere)
        lat, lon = utm.to_latlon(utm_x, utm_y, source_zone, northern=True)
        # Convert to target zone
        easting, northing, _, _ = utm.from_latlon(lat, lon, force_zone_number=target_zone)
        result.append((easting, northing))
    return result


def reproject_utm_coords_vectorized(
    coords: np.ndarray,
    source_zone: int,
    target_zone: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized UTM reprojection for numpy arrays.

    Much faster than reproject_utm_coords for large coordinate sets.
    Uses utm library's native numpy support.

    Args:
        coords: Nx2 array of (easting, northing) in source zone
        source_zone: Source UTM zone number (e.g., 32)
        target_zone: Target UTM zone number (e.g., 33)

    Returns:
        (eastings, northings) arrays in target zone
    """
    if source_zone == target_zone:
        return coords[:, 0].copy(), coords[:, 1].copy()

    # Batch convert to lat/lon (assumes northern hemisphere)
    lats, lons = utm.to_latlon(coords[:, 0], coords[:, 1], source_zone, northern=True)
    # Batch convert to target zone
    eastings, northings, _, _ = utm.from_latlon(lats, lons, force_zone_number=target_zone)
    return eastings, northings


def generate_tiles_by_zone(
    coords: list[tuple[float, float]],
    source_zone: int,
    grid_size_km: float,
    margin_km: float,
) -> dict[int, set[tuple[int, int]]]:
    """Generate user-grid tiles for each UTM zone.

    Takes coordinates in a source zone and generates tile sets for all
    supported zones by reprojecting coordinates appropriately.

    Args:
        coords: List of (easting, northing) in source_zone UTM
        source_zone: UTM zone of input coordinates
        grid_size_km: User's working grid resolution in km
        margin_km: Buffer distance from tile border in km

    Returns:
        Dict mapping zone number -> set of (grid_x, grid_y) tiles
        Example: {32: {(350, 5600), (351, 5600)}, 33: {(400, 5800)}}
    """
    # Get unique zones from all regions
    unique_zones = set(UTM_ZONE_BY_REGION.values())

    tiles_by_zone: dict[int, set[tuple[int, int]]] = {}

    for target_zone in unique_zones:
        # Reproject coords to target zone
        if target_zone == source_zone:
            zone_coords = coords
        else:
            zone_coords = reproject_utm_coords(coords, source_zone, target_zone)

        # Generate tiles for this zone
        tiles = generate_user_grid_tiles(zone_coords, grid_size_km, margin_km)
        tiles_by_zone[target_zone] = tiles

    return tiles_by_zone


def get_regions_for_zone(zone: int) -> list[Region]:
    """Get all regions that use a specific UTM zone.

    Args:
        zone: UTM zone number (e.g., 32 or 33)

    Returns:
        List of Region enums that use this zone
    """
    return [region for region, z in UTM_ZONE_BY_REGION.items() if z == zone]


def expand_to_1km(coords: tuple[int, int], region: Region) -> list[tuple[int, int]]:
    """Expand native tile coords to 1km grid cells.

    For regions with tiles larger than 1km (e.g., RLP with 2km tiles),
    returns all 1km cells covered by the tile. For 1km regions, returns
    the input coords as a single-element list.

    Args:
        coords: Tile anchor coordinates (lower-left corner, km indices)
        region: Region enum to determine native tile size

    Returns:
        List of (x, y) tuples for each 1km cell covered

    Example:
        >>> expand_to_1km((362, 5604), Region.NRW)  # 1km tile
        [(362, 5604)]
        >>> expand_to_1km((362, 5604), Region.RLP)  # 2km tile -> 4 cells
        [(362, 5604), (362, 5605), (363, 5604), (363, 5605)]
    """
    n = int(get_tile_size_km(region))
    return [(coords[0] + dx, coords[1] + dy) for dx in range(n) for dy in range(n)]


def coords_to_utm(coords: tuple[int, int]) -> tuple[int, int]:
    """Convert 1km grid coords to UTM meters (lower-left corner).

    Args:
        coords: Grid coordinates (km indices)

    Returns:
        (easting, northing) in meters
    """
    return (coords[0] * METERS_PER_KM, coords[1] * METERS_PER_KM)
