"""WMS imagery source for downloading tiles via OGC Web Map Service."""

import os
import re
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from threading import Lock

import requests

from georaffer.config import (
    CHUNK_SIZE,
    DEFAULT_TIMEOUT,
    MAX_RETRIES,
    METERS_PER_KM,
    MIN_FILE_SIZE,
    NRW_GRID_SIZE,
    RLP_GRID_SIZE,
    RETRY_BACKOFF_BASE,
    RETRY_MAX_WAIT,
    WMS_COVERAGE_RETRIES,
    WMS_NRW_BUFFER_M,
    WMS_RETRY_MAX_WAIT,
    WMS_TIMEOUT,
)


class WMSImagerySource:
    """Reusable WMS imagery download logic.

    Handles GetMap requests for tile downloads and GetFeatureInfo for coverage queries.
    Designed for RLP historical imagery but can be reused for other WMS sources.
    """

    def __init__(
        self,
        base_url: str,
        rgb_layer_pattern: str,
        info_layer_pattern: str,
        tile_size_m: int = RLP_GRID_SIZE,
        resolution_m: float = 0.2,
        image_format: str = "image/tiff-lzw",
        coverage_format: str | None = None,
        crs: str = "EPSG:25832",
        coverage_mode: str = "featureinfo",
        preview_size: int = 64,
        session: requests.Session | None = None,
    ):
        """Initialize WMS imagery source.

        Args:
            base_url: WMS service endpoint URL
            rgb_layer_pattern: Layer name pattern with {year} placeholder (e.g., "rp_dop20_rgb_{year}")
            info_layer_pattern: Info layer pattern with {year} placeholder (e.g., "rp_dop20_info_{year}")
            tile_size_m: Tile size in meters (default 2000 for RLP 2km tiles)
            resolution_m: Native resolution in meters/pixel (default 0.2 for RLP)
            image_format: WMS output format (default "image/tiff-lzw" for lossless)
            coverage_format: Optional GetMap format for coverage checks (defaults to image_format)
            crs: Coordinate reference system (default "EPSG:25832")
            coverage_mode: "featureinfo" or "getmap" coverage detection
            preview_size: Size in pixels for GetMap coverage checks
            session: Optional requests session for connection pooling
        """
        self.base_url = base_url
        self.rgb_layer_pattern = rgb_layer_pattern
        self.info_layer_pattern = info_layer_pattern
        self.tile_size_m = tile_size_m
        self.resolution_m = resolution_m
        self.image_format = image_format
        self.coverage_format = coverage_format or image_format
        self.crs = crs
        self.coverage_mode = coverage_mode
        self.preview_size = preview_size
        self._session = session or requests.Session()
        self._coverage_cache: dict[tuple[int, int, int], dict | None] = {}
        self._coverage_cache_lock = Lock()

        if self.coverage_mode not in ("featureinfo", "getmap"):
            raise ValueError(f"Unknown WMS coverage_mode '{coverage_mode}'.")

    def _rgb_layer(self, year: int) -> str:
        """Get RGB layer name for a year."""
        return self.rgb_layer_pattern.format(year=year)

    def _info_layer(self, year: int) -> str:
        """Get info layer name for a year."""
        return self.info_layer_pattern.format(year=year)

    def _grid_to_bbox(self, grid_x: int, grid_y: int) -> tuple[float, float, float, float]:
        """Convert grid coordinates to BBOX (minx, miny, maxx, maxy).

        Args:
            grid_x: Grid X coordinate in km
            grid_y: Grid Y coordinate in km

        Returns:
            Tuple of (minx, miny, maxx, maxy) in meters
        """
        minx = grid_x * METERS_PER_KM
        miny = grid_y * METERS_PER_KM
        maxx = minx + self.tile_size_m
        maxy = miny + self.tile_size_m
        return (minx, miny, maxx, maxy)

    def _tile_pixels(self) -> int:
        """Calculate tile size in pixels at native resolution."""
        return int(self.tile_size_m / self.resolution_m)

    def check_coverage(
        self,
        year: int,
        grid_x: int,
        grid_y: int,
    ) -> dict | None:
        """Query GetFeatureInfo to check if tile has coverage for given year.

        Args:
            year: Year to check
            grid_x: Grid X coordinate in km
            grid_y: Grid Y coordinate in km

        Returns:
            None if no coverage, else dict with:
            - acquisition_date: Actual image date (e.g., "2020-08-07")
            - tile_name: Tile identifier (e.g., "dop_32_380_5540")

        Raises:
            RuntimeError: If WMS request fails after MAX_RETRIES attempts
        """
        if self.coverage_mode == "getmap":
            return self._check_coverage_getmap(year, grid_x, grid_y)

        cache_key = (year, grid_x, grid_y)
        with self._coverage_cache_lock:
            if cache_key in self._coverage_cache:
                return self._coverage_cache[cache_key]

        bbox = self._grid_to_bbox(grid_x, grid_y)
        layer = self._info_layer(year)

        params = {
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetFeatureInfo",
            "LAYERS": layer,
            "QUERY_LAYERS": layer,
            "STYLES": "",
            "SRS": self.crs,
            "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "WIDTH": "100",
            "HEIGHT": "100",
            "X": "50",
            "Y": "50",
            "INFO_FORMAT": "text/plain",
        }

        last_error = None
        for attempt in range(WMS_COVERAGE_RETRIES):
            try:
                response = self._session.get(
                    self.base_url,
                    params=params,
                    timeout=WMS_TIMEOUT,
                )
                response.raise_for_status()
                text = response.text

                # Check for "no results" response
                if "Search returned no results" in text or "no results" in text.lower():
                    with self._coverage_cache_lock:
                        self._coverage_cache[cache_key] = None
                    return None

                # Extract kachelname (tile name) - required
                tile_match = re.search(r"kachelname\s*=\s*'([^']+)'", text)
                if not tile_match:
                    with self._coverage_cache_lock:
                        self._coverage_cache[cache_key] = None
                    return None

                # Validate kachelname matches expected tile coordinates
                # WMS format varies by year:
                #   - Older years: "dop_32_{grid_x}_{grid_y}" (e.g., "dop_32_380_5540")
                #   - Newer years: "dop_32{grid_x}_{grid_y}" (e.g., "dop_32380_5540")
                tile_name = tile_match.group(1)
                expected_with_underscore = f"dop_32_{grid_x}_{grid_y}"
                expected_without_underscore = f"dop_32{grid_x}_{grid_y}"
                if tile_name not in (expected_with_underscore, expected_without_underscore):
                    with self._coverage_cache_lock:
                        self._coverage_cache[cache_key] = None
                    return None

                # Extract acquisition date - try bildflugdatum first, fall back to erstellung
                acquisition_date = None
                date_match = re.search(r"bildflugdatum\s*=\s*'([^']*)'", text)
                if date_match and date_match.group(1):  # Non-empty value
                    acquisition_date = date_match.group(1)
                else:
                    # Fall back to erstellung (creation date) when bildflugdatum is empty
                    erstellung_match = re.search(r"erstellung\s*=\s*'([^']+)'", text)
                    if erstellung_match:
                        acquisition_date = erstellung_match.group(1)

                if not acquisition_date:
                    with self._coverage_cache_lock:
                        self._coverage_cache[cache_key] = None
                    return None

                result = {
                    "acquisition_date": acquisition_date,
                    "tile_name": tile_name,
                }
                with self._coverage_cache_lock:
                    self._coverage_cache[cache_key] = result
                return result

            except requests.RequestException as e:
                last_error = e
                if attempt < WMS_COVERAGE_RETRIES - 1:
                    wait_time = min(RETRY_BACKOFF_BASE**attempt, WMS_RETRY_MAX_WAIT)
                    time.sleep(wait_time)

        raise RuntimeError(
            f"WMS coverage check failed after {WMS_COVERAGE_RETRIES} attempts: {last_error}"
        )

    def _check_coverage_getmap(self, year: int, grid_x: int, grid_y: int) -> dict | None:
        """Check coverage using GetMap + transparent pixel detection."""
        cache_key = (year, grid_x, grid_y)
        with self._coverage_cache_lock:
            if cache_key in self._coverage_cache:
                return self._coverage_cache[cache_key]

        bbox = self._grid_to_bbox(grid_x, grid_y)
        layer = self._rgb_layer(year)

        params = {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetMap",
            "LAYERS": layer,
            "STYLES": "",
            "CRS": self.crs,
            "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "WIDTH": str(self.preview_size),
            "HEIGHT": str(self.preview_size),
            "FORMAT": self.coverage_format,
            "TRANSPARENT": "TRUE",
            "TILED": "TRUE",
        }

        from PIL import Image
        import io

        last_error = None
        for attempt in range(WMS_COVERAGE_RETRIES):
            try:
                response = self._session.get(
                    self.base_url,
                    params=params,
                    timeout=WMS_TIMEOUT,
                )
                response.raise_for_status()
                if not response.headers.get("Content-Type", "").startswith("image/"):
                    with self._coverage_cache_lock:
                        self._coverage_cache[cache_key] = None
                    return None
                img = Image.open(io.BytesIO(response.content)).convert("RGBA")
                pixels = img.getdata()
                has_coverage = any(p[3] != 0 for p in pixels)
                result = {"acquisition_date": None, "tile_name": f"dop_{grid_x}_{grid_y}"}
                with self._coverage_cache_lock:
                    self._coverage_cache[cache_key] = result if has_coverage else None
                return result if has_coverage else None
            except requests.RequestException as e:
                last_error = e
                if attempt < WMS_COVERAGE_RETRIES - 1:
                    wait_time = min(RETRY_BACKOFF_BASE**attempt, WMS_RETRY_MAX_WAIT)
                    time.sleep(wait_time)

        raise RuntimeError(
            f"WMS GetMap coverage check failed after {WMS_COVERAGE_RETRIES} attempts: {last_error}"
        )

    def check_coverage_multi(
        self,
        years: list[int],
        grid_x: int,
        grid_y: int,
    ) -> dict[int, dict]:
        """Query GetFeatureInfo for multiple years in one request.

        The RLP WMS returns a plain-text report that contains multiple "Layer '...'"
        sections. For each requested rp_dop20_info_{year} layer, it may include a
        metadata block (e.g., "Metadaten_27") containing kachelname/date fields.

        Returns:
            Dict[year] -> {"acquisition_date": str, "tile_name": str}
        """
        if not years:
            return {}

        # Fast-path: if everything is cached, return from cache.
        cached: dict[int, dict] = {}
        missing: list[int] = []
        with self._coverage_cache_lock:
            for year in years:
                key = (year, grid_x, grid_y)
                if key in self._coverage_cache:
                    value = self._coverage_cache[key]
                    if value is not None:
                        cached[year] = value
                else:
                    missing.append(year)
        if not missing:
            return cached

        bbox = self._grid_to_bbox(grid_x, grid_y)
        layers = ",".join(self._info_layer(y) for y in missing)

        params = {
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetFeatureInfo",
            "LAYERS": layers,
            "QUERY_LAYERS": layers,
            "STYLES": "",
            "SRS": self.crs,
            "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "WIDTH": "100",
            "HEIGHT": "100",
            "X": "50",
            "Y": "50",
            "INFO_FORMAT": "text/plain",
        }

        last_error = None
        for attempt in range(WMS_COVERAGE_RETRIES):
            try:
                response = self._session.get(
                    self.base_url,
                    params=params,
                    timeout=WMS_TIMEOUT,
                )
                response.raise_for_status()
                text = response.text

                # Quick "no results" shortcut
                if "Search returned no results" in text or "no results" in text.lower():
                    with self._coverage_cache_lock:
                        for y in missing:
                            self._coverage_cache[(y, grid_x, grid_y)] = None
                    return dict(cached)

                expected_with_underscore = f"dop_32_{grid_x}_{grid_y}"
                expected_without_underscore = f"dop_32{grid_x}_{grid_y}"

                current_year: int | None = None
                per_year: dict[int, dict[str, str | None]] = {}

                for line in text.splitlines():
                    layer_match = re.match(r"Layer 'rp_dop20_info_(\d{4})'\s*$", line.strip())
                    if layer_match:
                        y = int(layer_match.group(1))
                        current_year = y if y in missing else None
                        if current_year is not None and current_year not in per_year:
                            per_year[current_year] = {
                                "tile_name": None,
                                "bildflugdatum": None,
                                "erstellung": None,
                            }
                        continue

                    if current_year is None:
                        continue

                    tile_match = re.search(r"kachelname\s*=\s*'([^']+)'", line)
                    if tile_match:
                        per_year[current_year]["tile_name"] = tile_match.group(1)
                        continue

                    date_match = re.search(r"bildflugdatum\s*=\s*'([^']*)'", line)
                    if date_match:
                        per_year[current_year]["bildflugdatum"] = date_match.group(1)
                        continue

                    erstellung_match = re.search(r"erstellung\s*=\s*'([^']+)'", line)
                    if erstellung_match:
                        per_year[current_year]["erstellung"] = erstellung_match.group(1)

                result = dict(cached)
                with self._coverage_cache_lock:
                    for y in missing:
                        info = per_year.get(y)
                        if not info:
                            self._coverage_cache[(y, grid_x, grid_y)] = None
                            continue

                        tile_name = info.get("tile_name")
                        if tile_name not in (expected_with_underscore, expected_without_underscore):
                            self._coverage_cache[(y, grid_x, grid_y)] = None
                            continue

                        acquisition_date = info.get("bildflugdatum") or None
                        if not acquisition_date:
                            acquisition_date = info.get("erstellung") or None
                        if not acquisition_date:
                            self._coverage_cache[(y, grid_x, grid_y)] = None
                            continue

                        payload = {"acquisition_date": acquisition_date, "tile_name": tile_name}
                        self._coverage_cache[(y, grid_x, grid_y)] = payload
                        result[y] = payload

                return result

            except requests.RequestException as e:
                last_error = e
                if attempt < WMS_COVERAGE_RETRIES - 1:
                    wait_time = min(RETRY_BACKOFF_BASE**attempt, WMS_RETRY_MAX_WAIT)
                    time.sleep(wait_time)

        raise RuntimeError(
            f"WMS coverage check failed after {WMS_COVERAGE_RETRIES} attempts: {last_error}"
        )

    def get_tile_url(
        self,
        year: int,
        grid_x: int,
        grid_y: int,
    ) -> str:
        """Build GetMap URL for a tile.

        Args:
            year: Year for imagery
            grid_x: Grid X coordinate in km
            grid_y: Grid Y coordinate in km

        Returns:
            Complete GetMap URL for the tile
        """
        bbox = self._grid_to_bbox(grid_x, grid_y)
        pixels = self._tile_pixels()
        layer = self._rgb_layer(year)

        params = {
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetMap",
            "LAYERS": layer,
            "STYLES": "",
            "SRS": self.crs,
            "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "WIDTH": str(pixels),
            "HEIGHT": str(pixels),
            "FORMAT": self.image_format,
        }

        # Build URL with params
        param_str = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.base_url}?{param_str}"

    def download_tile(
        self,
        url: str,
        output_path: str,
        on_progress: Callable | None = None,
    ) -> bool:
        """Download tile from WMS with retry logic.

        Args:
            url: GetMap URL
            output_path: Path to save the tile
            on_progress: Optional callback(bytes_downloaded) for progress updates

        Returns:
            True if download succeeded, False otherwise
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output.with_suffix(".tmp")

        for attempt in range(MAX_RETRIES):
            try:
                response = self._session.get(
                    url,
                    stream=True,
                    timeout=DEFAULT_TIMEOUT,
                )
                response.raise_for_status()

                # Check content type - WMS may return XML error
                content_type = response.headers.get("Content-Type", "")
                if "xml" in content_type.lower():
                    # WMS error response - retry or fail
                    if attempt < MAX_RETRIES - 1:
                        wait = min(RETRY_BACKOFF_BASE**attempt, RETRY_MAX_WAIT)
                        time.sleep(wait)
                        continue
                    return False

                # Stream to temp file
                total_bytes = 0
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            total_bytes += len(chunk)
                            if on_progress:
                                on_progress(len(chunk))

                # Validate file size
                if total_bytes < MIN_FILE_SIZE:
                    tmp_path.unlink(missing_ok=True)
                    return False

                # Move to final location
                tmp_path.rename(output)
                return True

            except requests.RequestException:
                if attempt < MAX_RETRIES - 1:
                    wait = min(RETRY_BACKOFF_BASE**attempt, RETRY_MAX_WAIT)
                    time.sleep(wait)
                else:
                    tmp_path.unlink(missing_ok=True)
                    return False

        return False

    def output_filename(self, grid_x: int, grid_y: int, year: int) -> str:
        """Generate output filename for a WMS tile.

        Args:
            grid_x: Grid X coordinate in km
            grid_y: Grid Y coordinate in km
            year: Acquisition year

        Returns:
            Filename like "dop20rgb_32_380_5540_2_rp_2020.tif"
        """
        # Match RLP JP2 naming convention but with .tif extension
        return f"dop20rgb_32_{grid_x}_{grid_y}_2_rp_{year}.tif"


# =============================================================================
# NRW WMS Date Functions
# =============================================================================


def _normalize_wms_date(date_str: str) -> str | None:
    """Normalize WMS date string to ISO format (YYYY-MM-DD)."""
    if not date_str:
        return None
    stripped = date_str.strip()
    if not stripped:
        return None

    # Try DD.MM.YYYY format
    match = re.search(r"\d{2}\.\d{2}\.\d{4}", stripped)
    if match:
        try:
            dt = datetime.strptime(match.group(0), "%d.%m.%Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    # Try YYYY-MM-DD format
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


def fetch_nrw_dates(
    grid_x: int,
    grid_y: int,
    session: requests.Session,
) -> dict[int, str]:
    """Fetch all acquisition dates for an NRW tile via WMS.

    Makes two requests (historic + current) and merges results.

    Args:
        grid_x, grid_y: Grid coordinates (1km grid)
        session: HTTP session

    Returns:
        Dict mapping year -> acquisition_date (ISO string)
    """
    if os.getenv("GEORAFFER_DISABLE_WMS") == "1":
        return {}

    # Convert grid to UTM (center of tile)
    utm_x = grid_x * NRW_GRID_SIZE + NRW_GRID_SIZE / 2
    utm_y = grid_y * NRW_GRID_SIZE + NRW_GRID_SIZE / 2
    buffer = WMS_NRW_BUFFER_M

    result: dict[int, str] = {}

    for historic, wms_url, layers in [
        (True, "https://www.wms.nrw.de/geobasis/wms_nw_hist_dop", "nw_hist_dop_info"),
        (False, "https://www.wms.nrw.de/geobasis/wms_nw_dop", "nw_dop_utm_info"),
    ]:
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

        for attempt in range(MAX_RETRIES):
            try:
                response = session.get(wms_url, params=params, timeout=WMS_TIMEOUT)
                response.raise_for_status()

                for date_str in re.findall(r"Bildflugdatum = '([^']+)'", response.text):
                    normalized = _normalize_wms_date(date_str)
                    if normalized:
                        year = datetime.strptime(normalized, "%Y-%m-%d").year
                        result[year] = normalized
                break

            except Exception:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(min(RETRY_BACKOFF_BASE**attempt, RETRY_MAX_WAIT))

    return result
