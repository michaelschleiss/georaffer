"""WMS imagery source for downloading tiles via OGC Web Map Service."""

import re
import time
from collections.abc import Callable
from pathlib import Path

import requests

from georaffer.config import (
    CHUNK_SIZE,
    DEFAULT_TIMEOUT,
    MAX_RETRIES,
    METERS_PER_KM,
    MIN_FILE_SIZE,
    RETRY_BACKOFF_BASE,
    RETRY_MAX_WAIT,
    RLP_GRID_SIZE,
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
        crs: str = "EPSG:25832",
        session: requests.Session | None = None,
        verify_ssl: bool = True,
    ):
        """Initialize WMS imagery source.

        Args:
            base_url: WMS service endpoint URL
            rgb_layer_pattern: Layer name pattern with {year} placeholder (e.g., "rp_dop20_rgb_{year}")
            info_layer_pattern: Info layer pattern with {year} placeholder (e.g., "rp_dop20_info_{year}")
            tile_size_m: Tile size in meters (default 2000 for RLP 2km tiles)
            resolution_m: Native resolution in meters/pixel (default 0.2 for RLP)
            image_format: WMS output format (default "image/tiff-lzw" for lossless)
            crs: Coordinate reference system (default "EPSG:25832")
            session: Optional requests session for connection pooling
            verify_ssl: Whether to verify SSL certificates for WMS requests
        """
        self.base_url = base_url
        self.rgb_layer_pattern = rgb_layer_pattern
        self.info_layer_pattern = info_layer_pattern
        self.tile_size_m = tile_size_m
        self.resolution_m = resolution_m
        self.image_format = image_format
        self.crs = crs
        self._session = session or requests.Session()
        self.verify_ssl = verify_ssl

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
        for attempt in range(MAX_RETRIES):
            try:
                response = self._session.get(
                    self.base_url,
                    params=params,
                    timeout=DEFAULT_TIMEOUT,
                    verify=self.verify_ssl,
                )
                response.raise_for_status()
                text = response.text

                # Check for "no results" response
                if "Search returned no results" in text or "no results" in text.lower():
                    return None

                # Extract kachelname (tile name) - required
                tile_match = re.search(r"kachelname\s*=\s*'([^']+)'", text)
                if not tile_match:
                    return None

                # Validate kachelname matches expected tile coordinates
                # WMS format varies by year:
                #   - Older years: "dop_32_{grid_x}_{grid_y}" (e.g., "dop_32_380_5540")
                #   - Newer years: "dop_32{grid_x}_{grid_y}" (e.g., "dop_32380_5540")
                tile_name = tile_match.group(1)
                expected_with_underscore = f"dop_32_{grid_x}_{grid_y}"
                expected_without_underscore = f"dop_32{grid_x}_{grid_y}"
                if tile_name not in (expected_with_underscore, expected_without_underscore):
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
                    return None

                return {
                    "acquisition_date": acquisition_date,
                    "tile_name": tile_name,
                }

            except requests.RequestException as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    wait_time = min(RETRY_BACKOFF_BASE**attempt, RETRY_MAX_WAIT)
                    time.sleep(wait_time)

        raise RuntimeError(
            f"WMS coverage check failed after {MAX_RETRIES} attempts: {last_error}"
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
                    verify=self.verify_ssl,
                )
                response.raise_for_status()

                # Check content type - WMS may return XML error
                content_type = response.headers.get("Content-Type", "")
                if "xml" in content_type.lower():
                    # WMS error response
                    error_text = response.text[:500]
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
