"""Tile metadata extraction and provenance tracking."""

import csv
import os
import re
import time
from datetime import datetime
from pathlib import Path

import rasterio
import requests

from georaffer.config import (
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    RETRY_MAX_WAIT,
    WMS_NRW_BUFFER_M,
    WMS_RLP_BUFFER_M,
    WMS_TIMEOUT,
    Region,
)

# Module-level session for connection pooling
_wms_session = requests.Session()


def get_wms_metadata(
    utm_x: float,
    utm_y: float,
    region: str = "NRW",
    year: int | None = None,
    session: requests.Session | None = None,
) -> dict | None:
    """Query WMS GetFeatureInfo for tile acquisition date.

    Args:
        utm_x, utm_y: UTM coordinates
        region: Region (currently only 'NRW' supported)
        year: Optional year for historical vs current layer
        session: Optional requests.Session for dependency injection (testability)

    Returns:
        Dictionary with acquisition_date and metadata_source, or None

    Raises:
        RuntimeError: If all retries fail
    """
    if os.getenv("GEORAFFER_DISABLE_WMS") == "1":
        return None

    session = session or _wms_session
    if region != "NRW":
        return None

    buffer = WMS_NRW_BUFFER_M

    if year and year < 2024:
        wms_url = "https://www.wms.nrw.de/geobasis/wms_nw_hist_dop"
        layers = "nw_hist_dop_info"
    else:
        wms_url = "https://www.wms.nrw.de/geobasis/wms_nw_dop"
        layers = "nw_dop_utm_info"

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetFeatureInfo",
        "LAYERS": layers,
        "QUERY_LAYERS": layers,
        "CRS": "EPSG:25832",
        "BBOX": f"{utm_x - buffer},{utm_y - buffer},{utm_x + buffer},{utm_y + buffer}",
        "WIDTH": "100",
        "HEIGHT": "100",
        "I": "50",
        "J": "50",
        "INFO_FORMAT": "text/plain",
        "FEATURE_COUNT": "10",
    }

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(wms_url, params=params, timeout=WMS_TIMEOUT)
            response.raise_for_status()

            dates = re.findall(r"Bildflugdatum = '([^']+)'", response.text)
            if not dates:
                return None

            found_dates = {}

            for date_str in dates:
                if "." in date_str:
                    dt = datetime.strptime(date_str, "%d.%m.%Y")
                    acquisition_date = dt.strftime("%Y-%m-%d")
                else:
                    acquisition_date = date_str
                found_year = datetime.strptime(acquisition_date, "%Y-%m-%d").date().year
                found_dates[found_year] = {
                    "acquisition_date": acquisition_date, "metadata_source": "WMS GetFeatureInfo"
                }
            
            if year is None or year not in found_dates:
                return found_dates[max(found_dates.keys())] # return most recent entry

            return found_dates[year]

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait_time = min(RETRY_BACKOFF_BASE**attempt, RETRY_MAX_WAIT)
                # Sleep in 0.1s increments to allow interruption
                for _ in range(int(wait_time * 10)):
                    time.sleep(0.1)

    # Exhausted retries
    raise RuntimeError(f"Failed to get WMS metadata after {MAX_RETRIES} attempts: {last_error}")


def get_wms_metadata_rlp(
    utm_x: float, utm_y: float, session: requests.Session | None = None
) -> dict | None:
    """Query RLP WMS GetFeatureInfo for acquisition date.

    Uses rp_dop20 WMS (20cm) with the info layer rp_dop20_info.
    Returns acquisition_date (erstellung) when available.

    Args:
        utm_x: UTM easting coordinate in meters
        utm_y: UTM northing coordinate in meters
        session: Optional requests.Session for dependency injection (testability)

    Returns:
        Dictionary with acquisition_date and metadata_source, or None if not available

    Raises:
        RuntimeError: If all retries fail after MAX_RETRIES attempts
    """
    if os.getenv("GEORAFFER_DISABLE_WMS") == "1":
        return None

    session = session or _wms_session

    wms_url = "https://geo4.service24.rlp.de/wms/rp_dop20.fcgi"
    layer = "rp_dop20"
    info_layer = "rp_dop20_info"

    buffer = WMS_RLP_BUFFER_M
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetFeatureInfo",
        "LAYERS": layer,
        "QUERY_LAYERS": info_layer,
        "CRS": "EPSG:25832",
        "BBOX": f"{utm_x - buffer},{utm_y - buffer},{utm_x + buffer},{utm_y + buffer}",
        "WIDTH": "512",
        "HEIGHT": "512",
        "I": "256",
        "J": "256",
        "INFO_FORMAT": "text/plain",
        "FEATURE_COUNT": "5",
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(wms_url, params=params, timeout=WMS_TIMEOUT)
            resp.raise_for_status()
            text = resp.text
            # look for erstellung or Bildflugdatum
            import re

            match = re.search(r"erstellung\s*=\s*'([^']+)'", text, re.IGNORECASE)
            if not match:
                match = re.search(r"Bildflugdatum\s*=\s*'([^']+)'", text, re.IGNORECASE)
            if not match:
                return None
            date_str = match.group(1)
            # normalize DD.MM.YYYY -> YYYY-MM-DD
            if "." in date_str:
                try:
                    dt = datetime.strptime(date_str, "%d.%m.%Y")
                    date_str = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
            return {
                "acquisition_date": date_str,
                "metadata_source": "RLP WMS GetFeatureInfo",
            }
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait_time = min(RETRY_BACKOFF_BASE**attempt, RETRY_MAX_WAIT)
                for _ in range(int(wait_time * 10)):
                    time.sleep(0.1)
    raise RuntimeError(f"Failed to get RLP WMS metadata after {MAX_RETRIES} attempts: {last_error}")


def get_wms_metadata_for_region(
    utm_x: float,
    utm_y: float,
    region: Region,
    year: int | None = None,
    session: requests.Session | None = None,
) -> dict | None:
    """Unified WMS metadata lookup dispatching to region-specific functions.

    Args:
        utm_x: UTM easting coordinate in meters
        utm_y: UTM northing coordinate in meters
        region: Region enum (NRW or RLP)
        year: Optional year for historic layer selection (NRW only)
        session: Optional requests.Session for dependency injection

    Returns:
        Dictionary with acquisition_date and metadata_source, or None
    """
    if region == Region.NRW:
        return get_wms_metadata(utm_x, utm_y, "NRW", year, session)
    elif region == Region.RLP:
        return get_wms_metadata_rlp(utm_x, utm_y, session)


def add_provenance_to_geotiff(
    geotiff_path: Path, metadata: dict, utm_center: tuple | None = None, year: int | None = None
) -> bool:
    """Add provenance metadata tags to GeoTIFF file.

    Args:
        geotiff_path: Path to GeoTIFF file
        metadata: Dictionary with metadata to add
        utm_center: Optional (x, y) UTM coordinates for WMS lookup
        year: Optional year for historic WMS layer selection

    Returns:
        True if successful

    Raises:
        RuntimeError: If metadata tags cannot be written to GeoTIFF
    """
    # Query WMS for precise acquisition date if coordinates provided
    if utm_center and metadata.get("source_region") == "NRW":
        wms_metadata = get_wms_metadata(utm_center[0], utm_center[1], "NRW", year)
        if wms_metadata:
            metadata.update(wms_metadata)

    try:
        tags = {}

        if metadata.get("acquisition_date"):
            tags["ACQUISITION_DATE"] = str(metadata["acquisition_date"])
        if metadata.get("source_region"):
            tags["SOURCE_REGION"] = metadata["source_region"]
        if metadata.get("source_file"):
            tags["SOURCE_FILE"] = metadata["source_file"]
        if metadata.get("file_type"):
            tags["SOURCE_TYPE"] = metadata["file_type"]
        if metadata.get("metadata_source"):
            tags["METADATA_SOURCE"] = metadata["metadata_source"]

        tags["PROCESSING_DATE"] = datetime.now().strftime("%Y-%m-%d")

        with rasterio.open(geotiff_path, "r+") as dst:
            dst.update_tags(**tags)

        return True

    except Exception as e:
        raise RuntimeError(f"Failed to add provenance to {geotiff_path}: {e}") from e


def create_provenance_csv(tiles_metadata: list[dict], output_csv: str) -> bool:
    """Create CSV catalog of tile metadata.

    Args:
        tiles_metadata: List of metadata dictionaries
        output_csv: Output CSV file path

    Returns:
        True if successful

    Raises:
        IOError: If CSV file cannot be created or written
    """
    try:
        fieldnames = [
            "processed_file",
            "source_file",
            "source_region",
            "grid_x",
            "grid_y",
            "year",
            "acquisition_date",
            "file_type",
            "metadata_source",
            "point_count",
            "split_from",
            "processing_date",
        ]

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(tiles_metadata)

        return True

    except Exception as e:
        raise OSError(f"Failed to create CSV: {e}") from e
