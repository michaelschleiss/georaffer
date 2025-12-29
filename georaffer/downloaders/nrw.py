"""NRW (North Rhine-Westphalia) tile downloader."""

import xml.etree.ElementTree as ET
from datetime import datetime

import requests

from georaffer.config import (
    CATALOG_CACHE_DIR,
    FEED_TIMEOUT,
    NRW_GRID_SIZE,
    NRW_JP2_PATTERN,
    NRW_LAZ_PATTERN,
    Region,
)
from georaffer.downloaders.base import Catalog, RegionDownloader
from georaffer.runtime import parallel_map


class NRWDownloader(RegionDownloader):
    """NRW (North Rhine-Westphalia) downloader."""

    # Current feed URLs (used for deduplication in historic mode)
    CURRENT_JP2_BASE_URL = "https://www.opengeodata.nrw.de/produkte/geobasis/lusat/akt/dop/dop_jp2_f10/"
    CURRENT_JP2_FEED_URL = CURRENT_JP2_BASE_URL + "index.xml"

    # Historic years range (2014+ only - earlier years use different tiling)
    HISTORIC_JP2_BASE = "https://www.opengeodata.nrw.de/produkte/geobasis/lusat/hist/hist_dop/hist_dop_jp2_f10/hist_dop_{year}/epsg_25832/"

    @property
    def HISTORIC_YEARS(self) -> range:
        """Historic years from 2014 to current year."""
        return range(2014, datetime.now().year + 1)

    def __init__(
        self,
        output_dir: str,
        imagery_from: tuple[int, int | None] | None = None,
        session: requests.Session | None = None,
        quiet: bool = False,
    ):
        super().__init__(Region.NRW, output_dir, imagery_from=imagery_from, session=session, quiet=quiet)
        self._cache_path = CATALOG_CACHE_DIR / "nrw_catalog.json"

        if imagery_from is None:
            self._jp2_feed_url = self.CURRENT_JP2_FEED_URL
            self._jp2_base_url = self.CURRENT_JP2_BASE_URL
            self._from_year = None
            self._to_year = None
        else:
            from_year, to_year = imagery_from
            if from_year < 2010:
                raise ValueError(
                    f"Year {from_year} not supported. Use year >= 2010 (UTM coordinates)."
                )
            # Store year range for multi-year loading
            self._from_year = from_year
            self._to_year = to_year  # None means "to present"
            # Primary feed URL (used for compatibility, actual loading handles all years)
            self._jp2_base_url = self.HISTORIC_JP2_BASE.format(year=from_year)
            self._jp2_feed_url = self._jp2_base_url + "index.xml"

        self._laz_feed_url = (
            "https://www.opengeodata.nrw.de/produkte/geobasis/hm/bdom50_las/bdom50_las/index.xml"
        )
        self._laz_base_url = (
            "https://www.opengeodata.nrw.de/produkte/geobasis/hm/bdom50_las/bdom50_las/"
        )

    def utm_to_grid_coords(
        self, utm_x: float, utm_y: float
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """NRW uses 1km grid for both JP2 and LAZ."""
        grid_x = int(utm_x // NRW_GRID_SIZE)
        grid_y = int(utm_y // NRW_GRID_SIZE)
        return (grid_x, grid_y), (grid_x, grid_y)

    @property
    def jp2_feed_url(self) -> str:
        return self._jp2_feed_url

    @property
    def jp2_base_url(self) -> str:
        return self._jp2_base_url

    @property
    def laz_feed_url(self) -> str:
        return self._laz_feed_url

    def _parse_jp2_feed_with_year(
        self, session: requests.Session, feed_url: str, base_url: str
    ) -> dict[tuple[int, int], tuple[str, int]]:
        """Parse NRW JP2 feed and return tiles with year info.

        Args:
            session: HTTP session for requests
            feed_url: URL to XML feed (must point directly to file listing)
            base_url: Base URL for tile downloads

        Returns:
            Dict mapping (grid_x, grid_y) -> (download_url, year)
        """
        response = session.get(feed_url, timeout=FEED_TIMEOUT)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        jp2_tiles = {}
        for file_elem in root.findall(".//file"):
            filename = file_elem.get("name")
            if filename and filename.endswith(".jp2"):
                match = NRW_JP2_PATTERN.match(filename)
                if not match:
                    raise ValueError(
                        f"NRW JP2 '{filename}' doesn't match pattern. "
                        f"Expected: dop10rgbi_32_XXX_YYYY_N_nw_YEAR.jp2"
                    )
                grid_x = int(match.group(1))
                grid_y = int(match.group(2))
                tile_year = int(match.group(3))
                jp2_tiles[(grid_x, grid_y)] = (base_url + filename, tile_year)

        return jp2_tiles

    def get_available_tiles(self) -> tuple[dict, dict]:
        """Get available JP2 and LAZ tiles.

        Uses cached catalog. Filters by year range if --imagery-from is set.

        Returns:
            Tuple of (jp2_tiles, laz_tiles) dicts mapping coords to URLs.
        """
        laz_tiles = self._fetch_and_parse_feed(self.laz_feed_url, "laz")
        catalog = self.fetch_catalog()

        jp2_tiles = {}
        self._all_jp2_by_coord: dict[tuple[int, int], list[str]] = {}

        for coords, years in catalog.tiles.items():
            # Filter by year range
            valid = {y: url for y, url in years.items()
                     if self._year_in_range(y, self._from_year, self._to_year)}
            if valid:
                jp2_tiles[coords] = valid[max(valid)]  # Latest year for display
                self._all_jp2_by_coord[coords] = list(valid.values())

        return jp2_tiles, laz_tiles

    def _parse_laz_feed(
        self, session: requests.Session, root: ET.Element
    ) -> dict[tuple[int, int], str]:
        """Parse NRW LAZ feed."""
        laz_tiles = {}

        for file_elem in root.findall(".//file"):
            filename = file_elem.get("name")
            if filename and filename.endswith(".laz"):
                match = NRW_LAZ_PATTERN.match(filename)
                if not match:
                    raise ValueError(
                        f"NRW LAZ '{filename}' doesn't match pattern. "
                        f"Expected: bdom50_32XXX_YYYY_N_nw_YEAR.laz"
                    )
                grid_x = int(match.group(1))
                grid_y = int(match.group(2))
                laz_tiles[(grid_x, grid_y)] = self._laz_base_url + filename

        return laz_tiles

    def _load_catalog(self) -> Catalog:
        """Load NRW catalog from ATOM feeds (current + historic).

        Fetches current feed and all historic year feeds in parallel,
        combining them into a single catalog.
        """
        tiles: dict[tuple[int, int], dict[int, str]] = {}

        # 1. Current tiles
        if not self.quiet:
            print("  Loading current tiles from ATOM feed...")
        try:
            current_tiles = self._parse_jp2_feed_with_year(
                self._session, self.CURRENT_JP2_FEED_URL, self.CURRENT_JP2_BASE_URL
            )
            for coords, (url, year) in current_tiles.items():
                tiles.setdefault(coords, {})[year] = url
            if not self.quiet:
                print(f"  Current: {len(current_tiles)} tiles")
        except Exception as e:
            print(f"  Warning: Failed to load current feed: {e}")

        # 2. Historic tiles (parallel fetch)
        historic_years = list(self.HISTORIC_YEARS)
        if not self.quiet:
            print(f"  Loading {len(historic_years)} historic feeds...")

        def fetch_year(hist_year: int) -> dict | None:
            base_url = self.HISTORIC_JP2_BASE.format(year=hist_year)
            feed_url = base_url + "index.xml"
            try:
                return self._parse_jp2_feed_with_year(self._session, feed_url, base_url)
            except requests.HTTPError as e:
                if e.response.status_code == 404:
                    return None
                raise

        successful_years = []
        failed_years: list[int] = []

        for hist_year, historic_tiles in parallel_map(fetch_year, historic_years, max_workers=8):
            if historic_tiles is None:
                failed_years.append(hist_year)
                continue

            added = 0
            for coords, (url, year) in historic_tiles.items():
                if year not in tiles.get(coords, {}):
                    tiles.setdefault(coords, {})[year] = url
                    added += 1
            successful_years.append(f"{hist_year}:+{added}")

        if successful_years and not self.quiet:
            successful_years.sort(key=lambda s: int(s.split(":")[0]))
            print(f"  Historic: {', '.join(successful_years)}")
        if failed_years and not self.quiet:
            print(f"  Skipped: {sorted(failed_years)}")

        return Catalog(tiles=tiles)
