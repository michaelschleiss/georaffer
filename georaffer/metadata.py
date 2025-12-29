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


def _normalize_wms_date(date_str: str) -> str | None:
    if not date_str:
        return None
    stripped = date_str.strip()
    if not stripped:
        return None

    match = re.search(r"\d{2}\.\d{2}\.\d{4}", stripped)
    if match:
        try:
            dt = datetime.strptime(match.group(0), "%d.%m.%Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", stripped)
    if not match:
        return None
    year, month, day = match.groups()
    normalized = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    try:
        datetime.strptime(normalized, "%Y-%m-%d")
    except ValueError:
        return None
    return normalized


def _fetch_nrw_wms_dates(
    utm_x: float,
    utm_y: float,
    historic: bool,
    session: requests.Session,
) -> dict[int, dict] | None:
    """Fetch all acquisition dates from NRW WMS in one request.

    Args:
        utm_x, utm_y: UTM coordinates
        historic: If True, use historic WMS endpoint; otherwise current
        session: requests.Session for HTTP requests

    Returns:
        Dict mapping year -> {acquisition_date, metadata_source}, or None
    """
    buffer = WMS_NRW_BUFFER_M

    if historic:
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
                normalized = _normalize_wms_date(date_str)
                if not normalized:
                    continue
                found_year = datetime.strptime(normalized, "%Y-%m-%d").date().year
                found_dates[found_year] = {
                    "acquisition_date": normalized,
                    "metadata_source": "WMS GetFeatureInfo",
                }

            return found_dates if found_dates else None

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait_time = min(RETRY_BACKOFF_BASE**attempt, RETRY_MAX_WAIT)
                for _ in range(int(wait_time * 10)):
                    time.sleep(0.1)

    raise RuntimeError(f"Failed to get WMS metadata after {MAX_RETRIES} attempts: {last_error}")


def get_wms_metadata_all_years(
    utm_x: float,
    utm_y: float,
    session: requests.Session | None = None,
) -> dict[int, dict]:
    """Query NRW WMS for all available acquisition dates at a location.

    Makes two requests (historic + current) and merges results.

    Args:
        utm_x, utm_y: UTM coordinates
        session: Optional requests.Session for dependency injection

    Returns:
        Dict mapping year -> {acquisition_date, metadata_source}
        Empty dict if no dates found or WMS disabled.

    Raises:
        RuntimeError: If all retries fail
    """
    if os.getenv("GEORAFFER_DISABLE_WMS") == "1":
        return {}

    session = session or _wms_session
    result: dict[int, dict] = {}

    # Fetch historic dates
    historic = _fetch_nrw_wms_dates(utm_x, utm_y, historic=True, session=session)
    if historic:
        result.update(historic)

    # Fetch current dates (may overlap with historic for recent years)
    current = _fetch_nrw_wms_dates(utm_x, utm_y, historic=False, session=session)
    if current:
        result.update(current)

    return result


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

    historic = year is not None and year < 2024
    found_dates = _fetch_nrw_wms_dates(utm_x, utm_y, historic=historic, session=session)

    if not found_dates:
        return None

    if year is None or year not in found_dates:
        return found_dates[max(found_dates.keys())]

    return found_dates[year]


def get_wms_metadata_rlp(
    utm_x: float,
    utm_y: float,
    session: requests.Session | None = None,
    year: int | None = None,
) -> dict | None:
    """Query RLP WMS GetFeatureInfo for acquisition date.

    Uses rp_dop20 WMS (20cm) for current imagery, or rp_hkdop20 for historical.
    Returns acquisition_date (erstellung or bildflugdatum) when available.

    Args:
        utm_x: UTM easting coordinate in meters
        utm_y: UTM northing coordinate in meters
        session: Optional requests.Session for dependency injection (testability)
        year: Optional year for historical imagery (uses historical WMS endpoint)

    Returns:
        Dictionary with acquisition_date and metadata_source, or None if not available

    Raises:
        RuntimeError: If all retries fail after MAX_RETRIES attempts
    """
    if os.getenv("GEORAFFER_DISABLE_WMS") == "1":
        return None

    session = session or _wms_session

    # Use historical WMS for past years, current WMS otherwise
    if year is not None and year < datetime.now().year:
        wms_url = "https://geo4.service24.rlp.de/wms/rp_hkdop20.fcgi"
        layer = f"rp_dop20_rgb_{year}"
        info_layer = f"rp_dop20_info_{year}"
    else:
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
        "STYLES": "",  # Required by historical WMS
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
            match = re.search(r"erstellung\s*=\s*'([^']+)'", text, re.IGNORECASE)
            if not match:
                match = re.search(r"Bildflugdatum\s*=\s*'([^']+)'", text, re.IGNORECASE)
            if not match:
                return None
            date_str = match.group(1)
            normalized = _normalize_wms_date(date_str)
            if not normalized:
                return None
            return {
                "acquisition_date": normalized,
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
        region: Region enum (NRW, RLP, or BB)
        year: Optional year for historic layer selection
        session: Optional requests.Session for dependency injection

    Returns:
        Dictionary with acquisition_date and metadata_source, or None
    """
    if region == Region.NRW:
        return get_wms_metadata(utm_x, utm_y, "NRW", year, session)
    if region == Region.RLP:
        return get_wms_metadata_rlp(utm_x, utm_y, session, year)
    return None


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
    """Create or update CSV catalog of tile metadata.

    Merges new metadata with any existing CSV file, preserving existing rows
    for outputs that weren't processed in this run. Uses processed_file plus
    file_type as the unique key for merging.

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
            "conversion_date",
        ]

        # Merge with existing CSV if it exists
        existing_rows: dict[str, dict] = {}

        def _provenance_key(row: dict) -> str:
            processed = row.get("processed_file", "")
            file_type = row.get("file_type", "")
            if not processed:
                return ""
            return f"{processed}::{file_type}"

        if os.path.exists(output_csv):
            with open(output_csv, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    key = _provenance_key(row)
                    if key:
                        existing_rows[key] = row

        # Update with new metadata (overwrites existing entries for same processed_file)
        for row in tiles_metadata:
            key = _provenance_key(row)
            if key:
                existing_rows[key] = row

        # Write merged result
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing_rows.values())

        return True

    except Exception as e:
        raise OSError(f"Failed to create CSV: {e}") from e
