"""BB (Brandenburg) tile downloader for bDOM raster DSM tiles.

Catalog source:
https://data.geobasis-bb.de/geobasis/daten/bdom/tif/
"""

import re
import time
import zipfile
from pathlib import Path
from typing import ClassVar

import requests

from georaffer.config import (
    BB_BDOM_PATTERN,
    BB_GRID_SIZE,
    FEED_TIMEOUT,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    RETRY_MAX_WAIT,
    Region,
)
from georaffer.downloaders.base import RegionDownloader


class BrandenburgDownloader(RegionDownloader):
    """BB (Brandenburg) downloader for bDOM raster tiles (GeoTIFF in ZIP)."""

    BASE_URL: ClassVar[str] = "https://data.geobasis-bb.de/geobasis/daten/bdom/tif/"
    UTM_ZONE: ClassVar[int] = 33

    def __init__(self, output_dir: str, session: requests.Session | None = None):
        super().__init__(Region.BB, output_dir, imagery_from=None, session=session)
        self._jp2_feed_url = self.BASE_URL
        self._laz_feed_url = self.BASE_URL

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
        """Return extracted GeoTIFF filename for a bDOM ZIP URL."""
        name = Path(url).name
        if name.lower().endswith(".zip"):
            return name[:-4] + ".tif"
        return name

    def get_available_tiles(self) -> tuple[dict, dict]:
        """Parse the HTML listing and return available bDOM tiles."""
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    delay = min(RETRY_BACKOFF_BASE ** (attempt - 1), RETRY_MAX_WAIT)
                    time.sleep(delay)
                response = self._session.get(
                    self.BASE_URL, timeout=FEED_TIMEOUT, verify=self.verify_ssl
                )
                response.raise_for_status()
                laz_tiles = self._parse_bdom_listing(response.text)
                return {}, laz_tiles
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

    def download_file(self, url: str, output_path: str, on_progress=None) -> bool:
        """Download a ZIP and extract the GeoTIFF (and meta XML) to output_path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        zip_path = output_path.with_name(output_path.name + ".zip")
        super().download_file(url, str(zip_path), on_progress=on_progress)

        try:
            with zipfile.ZipFile(zip_path) as zf:
                tif_name = self._find_member(zf, ".tif")
                if not tif_name:
                    raise RuntimeError(f"No GeoTIFF found in {zip_path.name}")
                self._write_member(zf, tif_name, output_path)

                meta_name = self._find_member(zf, "_meta.xml")
                if meta_name:
                    meta_path = output_path.with_name(meta_name)
                    self._write_member(zf, meta_name, meta_path)
        finally:
            zip_path.unlink(missing_ok=True)
        return True

    def _parse_bdom_listing(self, html: str) -> dict[tuple[int, int], str]:
        tiles: dict[tuple[int, int], str] = {}
        for match in re.finditer(r'href="(bdom_\d{5}-\d{4}\.zip)"', html):
            filename = match.group(1)
            coords = self._parse_filename(filename)
            if not coords:
                continue
            tiles[coords] = f"{self.BASE_URL}{filename}"
        return tiles

    def _parse_filename(self, filename: str) -> tuple[int, int] | None:
        match = BB_BDOM_PATTERN.match(filename)
        if not match:
            return None
        east_code = match.group(1)
        zone = int(east_code[:2])
        if zone != self.UTM_ZONE:
            return None
        grid_x = int(east_code[2:])
        grid_y = int(match.group(2))
        return grid_x, grid_y

    @staticmethod
    def _find_member(zf: zipfile.ZipFile, suffix: str) -> str | None:
        for name in zf.namelist():
            if name.lower().endswith(suffix):
                return name
        return None

    @staticmethod
    def _write_member(zf: zipfile.ZipFile, member: str, output_path: Path) -> None:
        temp_path = output_path.with_name(output_path.name + ".tmp")
        with zf.open(member) as src, open(temp_path, "wb") as dst:
            dst.write(src.read())
        temp_path.replace(output_path)
