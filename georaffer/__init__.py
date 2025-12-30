"""georaffer - Download and process geospatial tiles from German state geodata providers."""

import os as _os

# Suppress "omp_set_nested deprecated" warning from OpenMP runtime.
# Must be set before any library that uses OpenMP (numpy, numba, rasterio) is imported.
_os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")

from georaffer.converters import convert_dsm_raster, convert_jp2, convert_laz
from georaffer.downloaders import BBDownloader, NRWDownloader, RLPDownloader
from georaffer.grids import latlon_to_utm
from georaffer.pipeline import process_tiles

__version__ = "0.2.0"
__all__ = [
    "BBDownloader",
    "NRWDownloader",
    "RLPDownloader",
    "convert_dsm_raster",
    "convert_jp2",
    "convert_laz",
    "latlon_to_utm",
    "process_tiles",
]
