"""Load tile coordinates from bounding box."""

from georaffer.config import METERS_PER_KM, OUTPUT_TILE_SIZE_KM


def load_from_bbox(
    min_x: float, min_y: float, max_x: float, max_y: float, tile_size_m: int | None = None
) -> list[tuple[float, float]]:
    """Generate tile center UTM coordinates covering a bounding box.

    Args:
        min_x, min_y: Lower-left corner (UTM meters)
        max_x, max_y: Upper-right corner (UTM meters)
        tile_size_m: Tile size in meters (default: from OUTPUT_TILE_SIZE_KM)

    Returns:
        List of (utm_x, utm_y) tile center coordinates in meters
    """
    if tile_size_m is None:
        tile_size_m = int(OUTPUT_TILE_SIZE_KM * METERS_PER_KM)
    if tile_size_m <= 0:
        raise ValueError("tile_size_m must be positive")

    # Use set of snapped origins to deduplicate tiles
    tile_origins: set[tuple[float, float]] = set()

    # Determine tile indices covering the bbox and snap to origins
    start_x = int(min_x // tile_size_m)
    end_x = int(max_x // tile_size_m)
    start_y = int(min_y // tile_size_m)
    end_y = int(max_y // tile_size_m)

    for tile_x in range(start_x, end_x + 1):
        origin_x = tile_x * tile_size_m
        for tile_y in range(start_y, end_y + 1):
            origin_y = tile_y * tile_size_m
            tile_origins.add((origin_x, origin_y))

    # Convert origins to centers and return sorted
    coords = [
        (origin_x + tile_size_m / 2, origin_y + tile_size_m / 2)
        for origin_x, origin_y in tile_origins
    ]
    return sorted(coords)
