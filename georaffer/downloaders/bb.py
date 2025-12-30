"""BB (Brandenburg) tile downloader for DOP RGBI and bDOM raster tiles.

Catalog source: OGC API Features at ogc-api.geobasis-bb.de
Download source: https://data.geobasis-bb.de/geobasis/daten/
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
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

# OGC API pagination settings
OGC_PAGE_SIZE = 1000  # Server max per request
OGC_PARALLEL_WORKERS = 8

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

        Uses parallel pagination for DOP and bDOM collections.
        The 'creationdate' field is the Bildflugdatum (verified against ZIP metadata).
        Note: Only current imagery is available for download; historic is WMS-only.
        """
        # DOP tiles (current only - historic requires WMS)
        if not self.quiet:
            print("  Loading DOP tiles from OGC API...")
        image_tiles: dict[tuple[int, int], dict[int, dict]] = {}
        for sheetnr, creation_date in self._fetch_ogc_collection("dop_single"):
            east, north = sheetnr.split("-")
            coords = self._parse_coords(east, north)
            if coords:
                year = creation_date.year
                image_tiles.setdefault(coords, {})[year] = {
                    "url": f"{self.DOP_BASE_URL}dop_{sheetnr}.zip",
                    "acquisition_date": creation_date.isoformat(),
                }
        if not self.quiet:
            print(f"    {len(image_tiles)} tiles")

        # bDOM tiles (current only)
        if not self.quiet:
            print("  Loading bDOM tiles from OGC API...")
        dsm_tiles: dict[tuple[int, int], dict] = {}
        for sheetnr, creation_date in self._fetch_ogc_collection("bdom_single"):
            east, north = sheetnr.split("-")
            coords = self._parse_coords(east, north)
            if coords:
                dsm_tiles[coords] = {
                    "url": f"{self.BDOM_BASE_URL}bdom_{sheetnr}.zip",
                    "acquisition_date": creation_date.isoformat(),
                }
        if not self.quiet:
            print(f"    {len(dsm_tiles)} tiles")

        return Catalog(image_tiles=image_tiles, dsm_tiles=dsm_tiles)

    def _fetch_ogc_collection(self, collection: str) -> list[tuple[str, date]]:
        """Fetch all tiles from an OGC API collection using parallel pagination.

        Makes a first request to get total count, then fetches all pages
        concurrently. Much faster than sequential pagination or bulk download.
        """
        url = f"{OGC_API_BASE}/collections/{collection}/items"

        # Get total count from first page
        resp = self._session.get(
            url, params={"f": "json", "limit": 1}, timeout=FEED_TIMEOUT
        )
        resp.raise_for_status()
        total = resp.json().get("numberMatched", 0)
        if total == 0:
            return []

        # Fetch all pages in parallel
        offsets = range(0, total, OGC_PAGE_SIZE)

        def fetch_page(offset: int) -> list[tuple[str, date]]:
            page_resp = self._session.get(
                url,
                params={"f": "json", "limit": OGC_PAGE_SIZE, "offset": offset},
                timeout=FEED_TIMEOUT,
            )
            page_resp.raise_for_status()
            return self._parse_ogc_features(page_resp.json().get("features", []))

        results: list[tuple[str, date]] = []
        with ThreadPoolExecutor(max_workers=OGC_PARALLEL_WORKERS) as executor:
            futures = {executor.submit(fetch_page, o): o for o in offsets}
            for fut in as_completed(futures):
                results.extend(fut.result())

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
