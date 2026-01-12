"""Centralized configuration constants for georaffer."""

import os
import re
from enum import Enum
from pathlib import Path

# =============================================================================
# Regions
# =============================================================================


class Region(str, Enum):
    """Supported German federal states.

    Each region has different tile sizes and resolutions:

                    NRW                 RLP                 BB                  BW                  BY                 TH
    Tile size       1km x 1km           2km x 2km           1km x 1km           2km x 2km           1km x 1km          1km x 1km
    Orthophoto      0.1 m/px            0.2 m/px            (not available)     0.2 m/px            0.2 m/px           0.2 m/px
    DSM spacing     0.5 m               0.2 m               0.2 m (raster)      1.0 m (raster)      0.2 m (raster)     0.2 m (raster)

    NRW/RLP/BW/BY/TH use EPSG:25832 (UTM Zone 32N). BB uses EPSG:25833 (UTM Zone 33N).
    """

    NRW = "NRW"
    RLP = "RLP"
    BB = "BB"
    BW = "BW"
    BY = "BY"
    TH = "TH"


# =============================================================================
# Coordinate System
# =============================================================================

UTM_ZONE = 32
METERS_PER_KM = 1000

# *_GRID_SIZE (meters): Download tile size. Used for WMS/download URLs.
# FILE_TILE_SIZE_KM (km): File size after extraction. Used for conversion/splitting.
# These differ for BW: downloads are 2km ZIPs containing 1km sub-tiles.
NRW_GRID_SIZE = 1000  # 1km
RLP_GRID_SIZE = 2000  # 2km
BB_GRID_SIZE = 1000  # 1km
BW_GRID_SIZE = 2000  # 2km ZIPs (contain 1km sub-tiles)
BY_GRID_SIZE = 1000  # 1km
TH_GRID_SIZE = 1000  # 1km

FILE_TILE_SIZE_KM: dict[Region, int] = {
    Region.NRW: 1,
    Region.RLP: 2,
    Region.BB: 1,
    Region.BW: 1,  # Extracted from 2km ZIPs
    Region.BY: 1,
    Region.TH: 1,
}

# Output tile size (km). Must evenly divide native sizes for splitting to work.
OUTPUT_TILE_SIZE_KM = 1.0


# =============================================================================
# Network
# =============================================================================

# Timeouts (seconds)
DEFAULT_TIMEOUT = 120
FEED_TIMEOUT = 30
WMS_TIMEOUT = 10
WMS_COVERAGE_RETRIES = 4
WMS_RETRY_MAX_WAIT = 10  # Keep GetFeatureInfo retries bounded

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

WMS_QUERY_WORKERS = 32
WMS_NRW_BUFFER_M = 500
WMS_RLP_BUFFER_M = 1000


# =============================================================================
# Catalog Cache
# =============================================================================

CATALOG_CACHE_DIR = Path("~/.cache/georaffer").expanduser()
CATALOG_TTL_DAYS = 90


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
DEFAULT_PIXEL_SIZE = 0.5  # meters per pixel

# LAZ point spacing by region (meters)
# Note: BB, BW, BY use raster DSM (not LAZ), but spacing is still needed for grid calculations
LAZ_SPACING_BY_REGION = {
    Region.NRW: 0.5,
    Region.RLP: 0.2,
    Region.BB: 0.2,
    Region.BW: 1.0,  # DOM1 is 1m raster
    Region.BY: 0.2,  # DOM20 is 0.2m raster
    Region.TH: 0.2,
}

LAZ_SAMPLE_SIZE = 10000  # Points to sample for verification


# =============================================================================
# Filename Patterns
# =============================================================================

# Capture groups: (grid_x, grid_y, year)
# Historic NRW feeds vary slightly:
# - 2010: dop10rgb_32_XXX_YYYY_1_nw_2010.jp2 (no IR band, "rgb" not "rgbi")
# - 2015: dop10rgbi_32288_5736_1_nw_2015.jp2 (missing underscore after "32")
NRW_JP2_PATTERN = re.compile(r"dop10rgb(?:i)?_32_?(\d{3})_(\d{4})_\d_nw_(\d{4})\.jp2$")
NRW_LAZ_PATTERN = re.compile(
    r"bdom50_32_?(\d{3})_(\d{4})_\d_nw_(\d{4})\.laz$"
)  # _? : underscore sometimes missing
# Matches both .jp2 (ATOM feed) and .tif (WMS downloads)
RLP_JP2_PATTERN = re.compile(r"dop20rgb_32_(\d{3})_(\d{4})_2_rp_(\d{4})\.(jp2|tif)$")
RLP_LAZ_PATTERN = re.compile(r"bdom20rgbi_32_(\d{3})_(\d{4})_2_rp\.laz$")
BB_BDOM_PATTERN = re.compile(r"bdom_(\d{5})-(\d{4})\.zip$", re.IGNORECASE)
BB_DOP_PATTERN = re.compile(r"dop_(\d{5})-(\d{4})\.zip$", re.IGNORECASE)
# BW patterns:
# - Download ZIPs: dop20rgb_32_{E}_{N}_2_bw.zip / dom1_32_{E}_{N}_2_bw.zip
# - Subtiles: dop20rgb_32_{E}_{N}_1_bw_YYYY.tif / dom1_32_{E}_{N}_1_bw_YYYY.tif
BW_DOP_PATTERN = re.compile(
    r"dop20rgb_32_(\d{3})_(\d{4})_[12]_bw(?:_(\d{4}))?\.(zip|tif|tiff|png|jpg|jpeg)$"
)
BW_DOM_PATTERN = re.compile(
    r"dom1_32_(\d{3})_(\d{4})_[12]_bw(?:_(\d{4}))?\.(zip|tif|tiff|png|jpg|jpeg)$"
)
# BY patterns: 32{E}_{N}.tif for DOP20, optional _YYYY for historic WMS, 32{E}_{N}_20_DOM.tif for DOM20
BY_DOP_PATTERN = re.compile(r"32(\d{3})_(\d{4})(?:_(\d{4}))?\.tif$")
BY_DOM_PATTERN = re.compile(r"32(\d{3})_(\d{4})_20_DOM\.tif$")
TH_DOP_PATTERN = re.compile(
    r"dop20rgb(?:i)?_32_?(\d{3})_(\d{4})_[12]_th_(\d{4})\.zip$",
    re.IGNORECASE,
)
TH_LAZ_PATTERN = re.compile(
    r"las_(?:32_)?(\d{3})_(\d{4})_1_th_(\d{4}-\d{4})\.zip$",
    re.IGNORECASE,
)


# =============================================================================
# Helper Functions
# =============================================================================


UTM_ZONE_BY_REGION = {
    Region.NRW: 32,
    Region.RLP: 32,
    Region.BB: 33,
    Region.BW: 32,
    Region.BY: 32,
    Region.TH: 32,
}


def utm_zone_for_region(region: Region) -> int:
    """UTM zone number for a region."""
    if region not in UTM_ZONE_BY_REGION:
        raise ValueError(f"No UTM zone configured for region {region}")
    return UTM_ZONE_BY_REGION[region]


def utm_zone_str_for_region(region: Region) -> str:
    """UTM zone number as string for a region."""
    return str(utm_zone_for_region(region))


def laz_spacing_for_region(region: Region) -> float:
    """LAZ point spacing in meters."""
    if region not in LAZ_SPACING_BY_REGION:
        raise ValueError(f"No LAZ spacing configured for region {region}")
    return LAZ_SPACING_BY_REGION[region]
