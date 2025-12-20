"""BB (Brandenburg) tile downloader for DOP RGBI and bDOM raster tiles.

Catalog source:
https://data.geobasis-bb.de/geobasis/daten/bdom/tif/
https://data.geobasis-bb.de/geobasis/daten/dop/rgbi_tif/
"""

import re
import time
from pathlib import Path
from typing import ClassVar

import requests

from georaffer.config import (
    BB_BDOM_PATTERN,
    BB_DOP_PATTERN,
    BB_GRID_SIZE,
    FEED_TIMEOUT,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    RETRY_MAX_WAIT,
    Region,
)
from georaffer.downloaders.base import RegionDownloader


class BrandenburgDownloader(RegionDownloader):
    """BB (Brandenburg) downloader for DOP RGBI + bDOM raster tiles (GeoTIFF in ZIP)."""

    BDOM_BASE_URL: ClassVar[str] = "https://data.geobasis-bb.de/geobasis/daten/bdom/tif/"
    DOP_BASE_URL: ClassVar[str] = "https://data.geobasis-bb.de/geobasis/daten/dop/rgbi_tif/"
    UTM_ZONE: ClassVar[int] = 33

    def __init__(self, output_dir: str, session: requests.Session | None = None):
        super().__init__(Region.BB, output_dir, imagery_from=None, session=session)
        self._jp2_feed_url = self.DOP_BASE_URL
        self._laz_feed_url = self.BDOM_BASE_URL

    @property
    def jp2_feed_url(self) -> str:
        return self._jp2_feed_url

    @property
    def laz_feed_url(self) -> str:
        return self._laz_feed_url

    @property
    def verify_ssl(self) -> bool:
        return True

    def utm_to_grid_coords(
        self, utm_x: float, utm_y: float
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """BB uses a 1km grid with km-based coordinates."""
        grid_x = int(utm_x // BB_GRID_SIZE)
        grid_y = int(utm_y // BB_GRID_SIZE)
        return (grid_x, grid_y), (grid_x, grid_y)

    def dsm_filename_from_url(self, url: str) -> str:
        """Return raw filename for a bDOM URL (keeps ZIPs intact)."""
        name = Path(url).name
        if not name.lower().endswith(".zip"):
            raise ValueError(f"BB DSM downloads must be ZIP archives (got {name}).")
        return name

    def image_filename_from_url(self, url: str) -> str:
        """Return raw filename for a DOP URL (keeps ZIPs intact)."""
        name = Path(url).name
        if not name.lower().endswith(".zip"):
            raise ValueError(f"BB imagery downloads must be ZIP archives (got {name}).")
        return name

    def get_available_tiles(self) -> tuple[dict, dict]:
        """Parse the HTML listing and return available DOP and bDOM tiles."""
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    delay = min(RETRY_BACKOFF_BASE ** (attempt - 1), RETRY_MAX_WAIT)
                    time.sleep(delay)
                dop_resp = self._session.get(
                    self.DOP_BASE_URL, timeout=FEED_TIMEOUT, verify=self.verify_ssl
                )
                dop_resp.raise_for_status()
                jp2_tiles = self._parse_dop_listing(dop_resp.text)

                bdom_resp = self._session.get(
                    self.BDOM_BASE_URL, timeout=FEED_TIMEOUT, verify=self.verify_ssl
                )
                bdom_resp.raise_for_status()
                laz_tiles = self._parse_bdom_listing(bdom_resp.text)
                return jp2_tiles, laz_tiles
            except Exception as e:
                last_error = e
        raise RuntimeError(
            f"Failed to fetch BB catalog after {MAX_RETRIES} retries: {last_error}"
        )

    def _parse_jp2_feed(
        self, session: requests.Session, root
    ) -> dict[tuple[int, int], str]:
        return {}

    def _parse_laz_feed(
        self, session: requests.Session, root
    ) -> dict[tuple[int, int], str]:
        return {}

    def _parse_bdom_listing(self, html: str) -> dict[tuple[int, int], str]:
        tiles: dict[tuple[int, int], str] = {}
        for match in re.finditer(r'href="(bdom_\d{5}-\d{4}\.zip)"', html):
            filename = match.group(1)
            coords = self._parse_filename(filename, BB_BDOM_PATTERN)
            if not coords:
                continue
            tiles[coords] = f"{self.BDOM_BASE_URL}{filename}"
        return tiles

    def _parse_dop_listing(self, html: str) -> dict[tuple[int, int], str]:
        tiles: dict[tuple[int, int], str] = {}
        for match in re.finditer(r'href="(dop_\d{5}-\d{4}\.zip)"', html):
            filename = match.group(1)
            coords = self._parse_filename(filename, BB_DOP_PATTERN)
            if not coords:
                continue
            tiles[coords] = f"{self.DOP_BASE_URL}{filename}"
        return tiles

    def _parse_filename(self, filename: str, pattern: re.Pattern) -> tuple[int, int] | None:
        match = pattern.match(filename)
        if not match:
            return None
        east_code = match.group(1)
        zone = int(east_code[:2])
        if zone != self.UTM_ZONE:
            return None
        grid_x = int(east_code[2:])
        grid_y = int(match.group(2))
        return grid_x, grid_y
