"""Converter for TH DOM ASCII XYZ files to GeoTIFF.

TH (Thüringen) distributes Digital Surface Models (DOM) and Digital Terrain
Models (DGM) as ASCII XYZ files with regular grid spacing.

Format: X Y Z (space-separated, one point per line, pixel center coordinates)
Grid: DOM1 = 1000×1000 points @ 1m (1km² tile), DOM2 = 500×500 points @ 2m
CRS: EPSG:25832 (UTM Zone 32N)

Note: XYZ coordinates are pixel centers. The GeoTIFF transform is adjusted to
reference pixel corners as per GDAL convention.
"""

import numpy as np
import rasterio
from pathlib import Path
from rasterio.transform import Affine
from rasterio.crs import CRS


def convert_dom_xyz_to_geotiff(
    xyz_path: Path,
    output_path: Path,
    expected_resolution: float = 2.0,
    quiet: bool = False,
) -> Path:
    """
    Convert TH DOM/DGM ASCII XYZ file to GeoTIFF.

    Args:
        xyz_path: Path to input .xyz file
        output_path: Path for output .tif file
        expected_resolution: Expected grid spacing in meters (1.0 for DOM1, 2.0 for DOM2)
        quiet: Suppress progress output

    Returns:
        Path to created GeoTIFF

    Raises:
        ValueError: If XYZ data doesn't match expected grid structure

    Note:
        XYZ coordinates represent pixel centers. The output GeoTIFF transform is
        adjusted by -0.5 pixels to reference the top-left corner per GDAL convention.
    """
    if not quiet:
        print(f"Converting {xyz_path.name} to GeoTIFF...")

    # Read ASCII XYZ
    try:
        data = np.loadtxt(xyz_path, dtype=float)
    except Exception as e:
        raise ValueError(f"Failed to read XYZ file {xyz_path}: {e}")

    if data.shape[1] != 3:
        raise ValueError(f"Expected 3 columns (X Y Z), got {data.shape[1]}")

    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    # Get unique coordinates
    x_unique = np.unique(x)
    y_unique = np.unique(y)

    if not quiet:
        print(f"  Grid: {len(x_unique)} × {len(y_unique)} points ({len(data):,} total)")

    # Verify grid spacing
    if len(x_unique) > 1:
        actual_resolution = x_unique[1] - x_unique[0]
        if not np.isclose(actual_resolution, expected_resolution, atol=0.01):
            raise ValueError(
                f"Expected {expected_resolution}m resolution, "
                f"got {actual_resolution:.3f}m"
            )
    else:
        raise ValueError("Insufficient unique X coordinates for grid determination")

    # Verify complete grid
    expected_points = len(x_unique) * len(y_unique)
    if len(data) != expected_points:
        raise ValueError(
            f"Incomplete grid: expected {expected_points} points, got {len(data)}"
        )

    # Reshape to 2D grid
    nx, ny = len(x_unique), len(y_unique)

    # XYZ files typically scan bottom-to-top (Y increasing)
    # GeoTIFF requires top-to-bottom, so flip if needed
    if y[1] > y[0]:  # Y increases as we read (bottom-to-top)
        grid = z.reshape((ny, nx))
        grid = np.flipud(grid)
        y_origin = y_unique[-1]  # Top edge for GeoTIFF
    else:  # Y decreases (already top-to-bottom)
        grid = z.reshape((ny, nx))
        y_origin = y_unique[0]

    x_origin = x_unique[0]

    # XYZ coordinates are pixel centers. Adjust to top-left corner for GeoTIFF transform.
    # For 1m resolution with first point at 621000.50, corner is at 621000.00
    half_pixel = actual_resolution / 2.0
    x_corner = x_origin - half_pixel
    y_corner = y_origin + half_pixel  # Y decreases in GeoTIFF

    if not quiet:
        print(f"  Resolution: {actual_resolution:.1f}m")
        print(f"  Bounds: ({x_origin:.0f}, {y_unique[0]:.0f}) to ({x_unique[-1]:.0f}, {y_unique[-1]:.0f})")
        print(f"  Elevation: {z.min():.1f}m to {z.max():.1f}m")

    # Create affine transform (references top-left corner, not pixel center)
    transform = Affine.translation(x_corner, y_corner) * Affine.scale(
        actual_resolution, -actual_resolution
    )

    # Write GeoTIFF
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=ny,
        width=nx,
        count=1,
        dtype=grid.dtype,
        crs=CRS.from_epsg(25832),
        transform=transform,
        compress="lzw",
        tiled=True,
        predictor=3,
        nodata=-9999,
    ) as dst:
        dst.write(grid, 1)

    if not quiet:
        print(f"  → {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

    return output_path
