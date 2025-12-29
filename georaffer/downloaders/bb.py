"""BB (Brandenburg) tile downloader for DOP RGBI and bDOM raster tiles.

Catalog source: OGC API Features at ogc-api.geobasis-bb.de
Download source: https://data.geobasis-bb.de/geobasis/daten/
"""

import re
from datetime import date
from pathlib import Path
from typing import ClassVar

import requests

from georaffer.config import (
    BB_GRID_SIZE,
    CATALOG_CACHE_DIR,
    FEED_TIMEOUT,
    Region,
)
from georaffer.downloaders.base import Catalog, RegionDownloader

OGC_API_BASE = "https://ogc-api.geobasis-bb.de/datasets/aktualitaeten"


class BBDownloader(RegionDownloader):
    """BB (Brandenburg) downloader for DOP RGBI + bDOM raster tiles (GeoTIFF in ZIP)."""

    BDOM_BASE_URL: ClassVar[str] = "https://data.geobasis-bb.de/geobasis/daten/bdom/tif/"
    DOP_BASE_URL: ClassVar[str] = "https://data.geobasis-bb.de/geobasis/daten/dop/rgbi_tif/"
    UTM_ZONE: ClassVar[int] = 33

    def __init__(
        self,
        output_dir: str,
        session: requests.Session | None = None,
        quiet: bool = False,
    ):
        super().__init__(Region.BB, output_dir, imagery_from=None, session=session, quiet=quiet)
        self._cache_path = CATALOG_CACHE_DIR / "bb_catalog.json"
        self._jp2_feed_url = self.DOP_BASE_URL
        self._laz_feed_url = self.BDOM_BASE_URL

    @property
    def jp2_feed_url(self) -> str:
        return self._jp2_feed_url

    @property
    def laz_feed_url(self) -> str:
        return self._laz_feed_url

    def utm_to_grid_coords(
        self, utm_x: float, utm_y: float
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """BB uses a 1km grid with km-based coordinates."""
        grid_x = int(utm_x // BB_GRID_SIZE)
        grid_y = int(utm_y // BB_GRID_SIZE)
        return (grid_x, grid_y), (grid_x, grid_y)

    def _filename_from_url(self, url: str) -> str:
        """Return filename from URL, validating it's a ZIP archive."""
        name = Path(url).name
        if not name.lower().endswith(".zip"):
            raise ValueError(f"BB downloads must be ZIP archives (got {name}).")
        return name

    def dsm_filename_from_url(self, url: str) -> str:
        return self._filename_from_url(url)

    def image_filename_from_url(self, url: str) -> str:
        return self._filename_from_url(url)

    def get_available_tiles(self) -> tuple[dict, dict]:
        """Return available DOP and bDOM tiles."""
        # DOP from catalog (OGC API) - fetch_catalog handles retries
        catalog = self.fetch_catalog()
        jp2_tiles = {
            coords: years[max(years.keys())]
            for coords, years in catalog.tiles.items()
        }

        # bDOM from HTML listing (not in OGC API)
        resp = self._session.get(self.BDOM_BASE_URL, timeout=FEED_TIMEOUT)
        resp.raise_for_status()
        laz_tiles = self._parse_bdom_listing(resp.text)

        return jp2_tiles, laz_tiles

    def _parse_bdom_listing(self, html: str) -> dict[tuple[int, int], str]:
        """Parse HTML directory listing for bDOM tiles."""
        tiles: dict[tuple[int, int], str] = {}
        for match in re.finditer(r'href="(bdom_(\d{5})-(\d{4})\.zip)"', html):
            filename, east_code, north_code = match.groups()
            coords = self._parse_coords(east_code, north_code)
            if coords:
                tiles[coords] = f"{self.BDOM_BASE_URL}{filename}"
        return tiles

    def _parse_coords(self, east_code: str, north_code: str) -> tuple[int, int] | None:
        """Parse 5-digit easting and 4-digit northing into grid coords."""
        try:
            if len(east_code) != 5:
                return None
            zone = int(east_code[:2])
            if zone != self.UTM_ZONE:
                return None
            return int(east_code[2:]), int(north_code)
        except ValueError:
            return None

    def _load_catalog(self) -> Catalog:
        """Load BB catalog from OGC API Features.

        Uses bulk download (~2s for 32k tiles) with pagination fallback.
        The 'creationdate' field is the Bildflugdatum (verified against ZIP metadata).
        """
        tiles: dict[tuple[int, int], dict[int, str]] = {}

        if not self.quiet:
            print("  Loading DOP tiles from OGC API...")

        try:
            for sheetnr, creation_date in self._fetch_ogc_tiles():
                east, north = sheetnr.split("-")
                coords = self._parse_coords(east, north)
                if coords:
                    year = creation_date.year
                    tiles.setdefault(coords, {})[year] = f"{self.DOP_BASE_URL}dop_{sheetnr}.zip"

            if not self.quiet:
                print(f"    {len(tiles)} tiles")
        except Exception as e:
            if not self.quiet:
                print(f"  Warning: Failed to load from OGC API: {e}")

        return Catalog(tiles=tiles)

    def _fetch_ogc_tiles(self) -> list[tuple[str, date]]:
        """Fetch all DOP tiles from OGC API."""
        url = f"{OGC_API_BASE}/collections/dop_single/items"

        # Try bulk download first (faster)
        try:
            resp = self._session.get(url, params={"f": "json", "bulk": "true"}, timeout=120)
            resp.raise_for_status()
            return self._parse_ogc_features(resp.json().get("features", []))
        except Exception:
            pass

        # Fallback to pagination
        results = []
        offset = 0
        while True:
            resp = self._session.get(
                url, params={"f": "json", "limit": 1000, "offset": offset}, timeout=FEED_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            features = data.get("features", [])
            if not features:
                break
            results.extend(self._parse_ogc_features(features))
            offset += data.get("numberReturned", len(features))
            if offset >= data.get("numberMatched", 0):
                break
        return results

    def _parse_ogc_features(self, features: list[dict]) -> list[tuple[str, date]]:
        """Parse OGC API features into (sheetnr, creation_date) tuples."""
        results = []
        for f in features:
            props = f.get("properties", {})
            sheetnr = props.get("sheetnr", "")
            date_str = props.get("creationdate", "")
            if sheetnr and date_str:
                try:
                    results.append((sheetnr, date.fromisoformat(date_str)))
                except ValueError:
                    continue
        return results
