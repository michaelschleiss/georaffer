"""Centralized configuration constants for georaffer."""

import os
import re
from enum import Enum

# =============================================================================
# Regions
# =============================================================================


class Region(str, Enum):
    """Supported German federal states.

    Each region has different tile sizes and resolutions:

                    NRW                 RLP
    Tile size       1km × 1km           2km × 2km
    Orthophoto      0.1 m/px            0.2 m/px
    DSM spacing     0.5 m               0.2 m

    All data uses EPSG:25832 (UTM Zone 32N).
    """

    NRW = "NRW"
    RLP = "RLP"


# =============================================================================
# Coordinate System
# =============================================================================

UTM_ZONE = 32
UTM_ZONE_STR = "32"
METERS_PER_KM = 1000

# Native tile sizes from data providers (meters)
NRW_GRID_SIZE = 1000  # 1km
RLP_GRID_SIZE = 2000  # 2km

# Output tile size (km). Must evenly divide native sizes for splitting to work.
OUTPUT_TILE_SIZE_KM = 1.0


# =============================================================================
# Network
# =============================================================================

# Timeouts (seconds)
DEFAULT_TIMEOUT = 120
FEED_TIMEOUT = 30
WMS_TIMEOUT = 10

# Streaming
CHUNK_SIZE = 65536  # 64KB
MIN_FILE_SIZE = 1024  # 1KB minimum for valid file
HTTP_POOL_MAXSIZE = 20

# Retry with exponential backoff
MAX_RETRIES = 15
RETRY_BACKOFF_BASE = 2  # 2^attempt seconds
RETRY_MAX_WAIT = 300  # 5 minutes max


# =============================================================================
# WMS Metadata
# =============================================================================

WMS_QUERY_WORKERS = 5
WMS_NRW_BUFFER_M = 500
WMS_RLP_BUFFER_M = 1000


# =============================================================================
# Processing
# =============================================================================

# Parallelism
DEFAULT_WORKERS = 4
REPROJECT_THREADS = min(os.cpu_count() or 4, 10)
THREADS_PER_WORKER = "auto"
# - "auto": max(2, cpu_count // workers)
# - None: use library defaults
# - int: fixed thread count

# Raster output
DSM_DTYPE = "float32"  # 0.06mm precision, 50% smaller than float64
DSM_NODATA = -9999.0
DEFAULT_RESOLUTION = 1.0  # meters
GDAL_CACHEMAX_MB = 512

# LAZ point spacing by region (meters)
LAZ_SPACING_BY_REGION = {
    Region.NRW: 0.5,
    Region.RLP: 0.2,
}

LAZ_SAMPLE_SIZE = 10000  # Points to sample for verification


# =============================================================================
# Filename Patterns
# =============================================================================

# Capture groups: (grid_x, grid_y, year)
NRW_JP2_PATTERN = re.compile(r"dop10rgbi_32_(\d{3})_(\d{4})_\d_nw_(\d{4})\.jp2$")
NRW_LAZ_PATTERN = re.compile(
    r"bdom50_32_?(\d{3})_(\d{4})_\d_nw_(\d{4})\.laz$"
)  # _? : underscore sometimes missing
RLP_JP2_PATTERN = re.compile(r"dop20rgb_32_(\d{3})_(\d{4})_2_rp_(\d{4})\.jp2$")
RLP_LAZ_PATTERN = re.compile(r"bdom20rgbi_32_(\d{3})_(\d{4})_2_rp\.laz$")


# =============================================================================
# Helper Functions
# =============================================================================


def get_tile_size_km(region: Region) -> float:
    """Native tile size in kilometers."""
    return (RLP_GRID_SIZE if region == Region.RLP else NRW_GRID_SIZE) / METERS_PER_KM


def get_tile_size_m(region: Region) -> int:
    """Native tile size in meters."""
    return RLP_GRID_SIZE if region == Region.RLP else NRW_GRID_SIZE


def laz_spacing_for_region(region: Region) -> float:
    """LAZ point spacing in meters."""
    if region not in LAZ_SPACING_BY_REGION:
        raise ValueError(f"No LAZ spacing configured for region {region}")
    return LAZ_SPACING_BY_REGION[region]
