"""BY (Bayern/Bavaria) tile downloader for DOP20 and DOM20 raster tiles.

Catalog source: Metalink files at geodaten.bayern.de/odd/a/
Download source: https://download1.bayernwolke.de/a/

Grid system:
- 1km x 1km tiles (like NRW)
- CRS: EPSG:25832 (UTM Zone 32N)
- Direct GeoTIFF downloads (no ZIP, unlike BW/BB)

Metalink structure:
- 7 district metalinks (091-097) for Regierungsbezirke
- Each contains ~10k tiles with SHA-256 verification hashes
- URLs: https://geodaten.bayern.de/odd/a/{product}/meta/metalink/09{1-7}.meta4
"""

import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import ClassVar

import requests

from georaffer.config import (
    BY_GRID_SIZE,
    CATALOG_CACHE_DIR,
    FEED_TIMEOUT,
    Region,
)
from georaffer.downloaders.base import Catalog, RegionDownloader

# Metalink pagination settings
METALINK_PARALLEL_WORKERS = 7  # One per district

# Bayern districts (Regierungsbezirke)
BY_DISTRICTS = ["091", "092", "093", "094", "095", "096", "097"]

# Metalink namespace
METALINK_NS = {"ml": "urn:ietf:params:xml:ns:metalink"}


class BYDownloader(RegionDownloader):
    """BY (Bayern) downloader for DOP20 + DOM20 raster tiles (direct GeoTIFF)."""

    # Metalink catalog URLs
    DOP_METALINK_BASE: ClassVar[str] = "https://geodaten.bayern.de/odd/a/dop20/meta/metalink/"
    DOM_METALINK_BASE: ClassVar[str] = "https://geodaten.bayern.de/odd/a/dom20/meta/DOM/metalink/"

    # Download base URLs
    DOP_BASE_URL: ClassVar[str] = "https://download1.bayernwolke.de/a/dop20/data/"
    DOM_BASE_URL: ClassVar[str] = "https://download1.bayernwolke.de/a/dom20/DOM/"

    UTM_ZONE: ClassVar[int] = 32

    def __init__(
        self,
        output_dir: str,
        imagery_from: tuple[int, int | None] | None = None,
        session: requests.Session | None = None,
        quiet: bool = False,
    ):
        super().__init__(
            Region.BY, output_dir, imagery_from=imagery_from, session=session, quiet=quiet
        )
        self._cache_path = CATALOG_CACHE_DIR / "by_catalog.json"

        # Parse imagery_from for historic support (if we add it later)
        if imagery_from is not None:
            from_year, to_year = imagery_from
            self._from_year = from_year
            self._to_year = to_year
        else:
            self._from_year = None
            self._to_year = None

    def utm_to_grid_coords(
        self, utm_x: float, utm_y: float
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """BY uses a 1km grid with km-based coordinates."""
        grid_x = int(utm_x // BY_GRID_SIZE)
        grid_y = int(utm_y // BY_GRID_SIZE)
        return (grid_x, grid_y), (grid_x, grid_y)

    def _filename_from_url(self, url: str) -> str:
        """Return filename from URL, validating it's a TIF file."""
        name = Path(url).name
        if not name.lower().endswith(".tif"):
            raise ValueError(f"BY downloads must be TIF files (got {name}).")
        return name

    def dsm_filename_from_url(self, url: str) -> str:
        return self._filename_from_url(url)

    def image_filename_from_url(self, url: str) -> str:
        return self._filename_from_url(url)

    def _load_catalog(self) -> Catalog:
        """Load BY catalog from district metalink files.

        Fetches all 7 district metalinks in parallel for both DOP20 and DOM20.
        """
        # DOP tiles
        if not self.quiet:
            print("  Loading DOP20 tiles from metalink catalogs...")
        image_tiles: dict[tuple[int, int], dict[int, dict]] = {}

        for coords, url in self._fetch_all_metalinks(self.DOP_METALINK_BASE, "dop"):
            if coords:
                # Bayern metalinks don't include year in filename, use current
                # Could query WMS by_dop20_info for actual dates if needed
                from datetime import date

                year = date.today().year
                image_tiles.setdefault(coords, {})[year] = {
                    "url": url,
                    "acquisition_date": None,
                }

        if not self.quiet:
            print(f"    {len(image_tiles)} tiles")

        # DOM tiles
        if not self.quiet:
            print("  Loading DOM20 tiles from metalink catalogs...")
        dsm_tiles: dict[tuple[int, int], dict[int, dict]] = {}

        for coords, url in self._fetch_all_metalinks(self.DOM_METALINK_BASE, "dom"):
            if coords:
                from datetime import date

                year = date.today().year
                dsm_tiles.setdefault(coords, {})[year] = {
                    "url": url,
                    "acquisition_date": None,
                }

        if not self.quiet:
            print(f"    {len(dsm_tiles)} tiles")

        return Catalog(image_tiles=image_tiles, dsm_tiles=dsm_tiles)

    def _fetch_all_metalinks(
        self, base_url: str, product: str
    ) -> list[tuple[tuple[int, int] | None, str]]:
        """Fetch all district metalinks in parallel.

        Returns list of (coords, download_url) tuples.
        """

        def fetch_district(district: str) -> list[tuple[tuple[int, int] | None, str]]:
            url = f"{base_url}{district}.meta4"
            try:
                resp = self._session.get(url, timeout=FEED_TIMEOUT)
                resp.raise_for_status()
                return self._parse_metalink(resp.content, product)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    # District metalink may not exist for DOM
                    return []
                raise

        results: list[tuple[tuple[int, int] | None, str]] = []
        with ThreadPoolExecutor(max_workers=METALINK_PARALLEL_WORKERS) as executor:
            futures = {executor.submit(fetch_district, d): d for d in BY_DISTRICTS}
            for fut in as_completed(futures):
                results.extend(fut.result())

        return results

    def _parse_metalink(
        self, content: bytes, product: str
    ) -> list[tuple[tuple[int, int] | None, str]]:
        """Parse metalink XML to extract tile coordinates and URLs.

        Metalink format:
        <metalink xmlns="urn:ietf:params:xml:ns:metalink">
          <file name="32679_5392.tif">
            <url>https://download1.bayernwolke.de/a/dop20/data/32679_5392.tif</url>
            <hash type="sha-256">...</hash>
          </file>
          ...
        </metalink>
        """
        results: list[tuple[tuple[int, int] | None, str]] = []

        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            return results

        for file_elem in root.findall("ml:file", METALINK_NS):
            name = file_elem.get("name", "")
            url_elem = file_elem.find("ml:url", METALINK_NS)

            if url_elem is None or url_elem.text is None:
                continue

            url = url_elem.text.strip()
            coords = self._parse_coords_from_filename(name, product)
            results.append((coords, url))

        return results

    def _parse_coords_from_filename(
        self, filename: str, product: str
    ) -> tuple[int, int] | None:
        """Parse grid coordinates from BY filename.

        DOP format: 32{E}_{N}.tif (e.g., 32679_5392.tif)
        DOM format: 32{E}_{N}_20_DOM.tif (e.g., 32686_5369_20_DOM.tif)
        """
        import re

        if product == "dop":
            # DOP: 32679_5392.tif
            match = re.match(r"32(\d{3})_(\d{4})\.tif$", filename)
        else:
            # DOM: 32686_5369_20_DOM.tif
            match = re.match(r"32(\d{3})_(\d{4})_20_DOM\.tif$", filename)

        if not match:
            return None

        try:
            return int(match.group(1)), int(match.group(2))
        except ValueError:
            return None
