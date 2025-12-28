"""Load GeoTIFF bounds for coordinate extraction."""

from pathlib import Path

import rasterio
from rasterio.crs import CRS


def load_from_geotiff(path: str) -> tuple[tuple[float, float, float, float], CRS]:
    """Load bounds and CRS from a GeoTIFF.

    Args:
        path: Path to a GeoTIFF file

    Returns:
        ((min_x, min_y, max_x, max_y), crs) tuple in the dataset CRS
    """
    if not Path(path).exists():
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"GeoTIFF has no CRS: {path}")
        bounds = src.bounds
        return (bounds.left, bounds.bottom, bounds.right, bounds.top), src.crs
