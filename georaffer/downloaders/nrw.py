"""NRW (North Rhine-Westphalia) tile downloader."""

import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock

import requests

from georaffer.config import (
    CATALOG_CACHE_DIR,
    FEED_TIMEOUT,
    NRW_GRID_SIZE,
    NRW_JP2_PATTERN,
    NRW_LAZ_PATTERN,
    WMS_QUERY_WORKERS,
    WMS_TIMEOUT,
    WMS_COVERAGE_RETRIES,
    WMS_RETRY_MAX_WAIT,
    RETRY_BACKOFF_BASE,
    Region,
)
from georaffer.downloaders.base import Catalog, RegionDownloader
from georaffer.downloaders.feeds import fetch_xml_feed
from georaffer.runtime import parallel_map


class NRWDownloader(RegionDownloader):
    """NRW (North Rhine-Westphalia) downloader."""

    # Current feed URLs (used for deduplication in historic mode)
    CURRENT_JP2_BASE_URL = "https://www.opengeodata.nrw.de/produkte/geobasis/lusat/akt/dop/dop_jp2_f10/"
    CURRENT_JP2_FEED_URL = CURRENT_JP2_BASE_URL + "index.xml"

    # Historic years range (2014+ only - earlier years use different tiling)
    HISTORIC_JP2_BASE = "https://www.opengeodata.nrw.de/produkte/geobasis/lusat/hist/hist_dop/hist_dop_jp2_f10/hist_dop_{year}/epsg_25832/"

    # WMS endpoints for acquisition date lookup
    WMS_CURRENT_URL = "https://www.wms.nrw.de/geobasis/wms_nw_dop"
    WMS_CURRENT_INFO_LAYER = "nw_dop_utm_info"
    WMS_HISTORIC_URL = "https://www.wms.nrw.de/geobasis/wms_nw_hist_dop"
    WMS_HISTORIC_INFO_LAYER = "nw_hist_dop_info"

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
                    f"Year {from_year} not supported. NRW imagery before 2010 uses "
                    f"Gauß-Krüger CRS instead of UTM32."
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

    def _query_wms_dates_for_tile(
        self, grid_x: int, grid_y: int
    ) -> dict[int, str]:
        """Query WMS GetFeatureInfo for all acquisition dates at a tile.

        Queries both current and historic WMS endpoints to get dates for all years.
        Includes retry logic for transient failures.

        Args:
            grid_x: Grid X coordinate (NRW uses meters, e.g., 288 for 288000m)
            grid_y: Grid Y coordinate

        Returns:
            Dict mapping year -> acquisition date string (YYYY-MM-DD)

        Raises:
            RuntimeError: If WMS queries fail after all retries
        """
        results: dict[int, str] = {}
        errors: list[str] = []

        # NRW grid coords are in km, convert to meters for bbox
        min_x = grid_x * NRW_GRID_SIZE
        min_y = grid_y * NRW_GRID_SIZE
        max_x = min_x + NRW_GRID_SIZE
        max_y = min_y + NRW_GRID_SIZE

        base_params = {
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetFeatureInfo",
            "INFO_FORMAT": "text/plain",
            "X": "50",
            "Y": "50",
            "WIDTH": "100",
            "HEIGHT": "100",
            "BBOX": f"{min_x},{min_y},{max_x},{max_y}",
            "SRS": "EPSG:25832",
        }

        # 1. Query current WMS (single date for current/recent tiles)
        params = {**base_params, "LAYERS": self.WMS_CURRENT_INFO_LAYER,
                  "QUERY_LAYERS": self.WMS_CURRENT_INFO_LAYER}
        last_error = None
        for attempt in range(WMS_COVERAGE_RETRIES):
            try:
                response = self._session.get(self.WMS_CURRENT_URL, params=params, timeout=WMS_TIMEOUT)
                response.raise_for_status()
                # Current format: "Bildflugdatum = '06.04.2024'" (DD.MM.YYYY)
                match = re.search(r"Bildflugdatum\s*=\s*'([^']+)'", response.text)
                if match:
                    date_str = match.group(1)
                    # Normalize DD.MM.YYYY to YYYY-MM-DD
                    if "." in date_str:
                        parts = date_str.split(".")
                        if len(parts) == 3:
                            date_str = f"{parts[2]}-{parts[1]}-{parts[0]}"
                    # Extract year from date
                    year = int(date_str[:4])
                    results[year] = date_str
                last_error = None
                break
            except requests.RequestException as e:
                last_error = e
                if attempt < WMS_COVERAGE_RETRIES - 1:
                    time.sleep(min(RETRY_BACKOFF_BASE ** attempt, WMS_RETRY_MAX_WAIT))
        if last_error:
            errors.append(f"current WMS: {last_error}")

        # 2. Query historic WMS (multiple features with different years)
        params = {**base_params, "LAYERS": self.WMS_HISTORIC_INFO_LAYER,
                  "QUERY_LAYERS": self.WMS_HISTORIC_INFO_LAYER, "FEATURE_COUNT": "50"}
        last_error = None
        for attempt in range(WMS_COVERAGE_RETRIES):
            try:
                response = self._session.get(self.WMS_HISTORIC_URL, params=params, timeout=WMS_TIMEOUT)
                response.raise_for_status()
                # Historic format has multiple Feature blocks with year in Download path
                # Parse: "Bildflugdatum = '2022-06-15'" and "Download der Originalkachel = 'hist_dop_2022/..."
                for feature_match in re.finditer(
                    r"Feature \d+:.*?Bildflugdatum\s*=\s*'([^']+)'.*?Download der Originalkachel\s*=\s*'[^']*hist_dop_(\d{4})",
                    response.text,
                    re.DOTALL,
                ):
                    date_str = feature_match.group(1)
                    year = int(feature_match.group(2))
                    if year not in results:  # Don't overwrite current with historic
                        results[year] = date_str
                last_error = None
                break
            except requests.RequestException as e:
                last_error = e
                if attempt < WMS_COVERAGE_RETRIES - 1:
                    time.sleep(min(RETRY_BACKOFF_BASE ** attempt, WMS_RETRY_MAX_WAIT))
        if last_error:
            errors.append(f"historic WMS: {last_error}")

        # Raise if both queries failed
        if len(errors) == 2:
            raise RuntimeError(f"WMS date queries failed for tile ({grid_x}, {grid_y}): {'; '.join(errors)}")

        return results

    def _fetch_acquisition_dates_parallel(
        self, tiles: dict[tuple[int, int], dict[int, dict]]
    ) -> None:
        """Fetch acquisition dates for all tiles via WMS in parallel.

        Uses _query_wms_dates_for_tile to get all years per tile in one request,
        then applies the results to matching entries in the tiles dict.

        Raises:
            RuntimeError: If any tiles fail to fetch dates after retries
        """
        # Find unique tile coords that need date lookups
        coords_needing_dates = set()
        for coords, years in tiles.items():
            for year, data in years.items():
                if data.get("acquisition_date") is None:
                    coords_needing_dates.add(coords)
                    break

        if not coords_needing_dates:
            return

        if not self.quiet:
            print(f"  Fetching dates for {len(coords_needing_dates)} tiles via WMS ({WMS_QUERY_WORKERS} workers)...")

        fetched = 0
        dates_applied = 0
        failed_tiles: list[tuple[tuple[int, int], str]] = []
        started = time.perf_counter()
        lock = Lock()

        def fetch_dates_for_tile(coords: tuple[int, int]) -> tuple[tuple[int, int], dict[int, str], str | None]:
            try:
                year_dates = self._query_wms_dates_for_tile(coords[0], coords[1])
                return coords, year_dates, None
            except RuntimeError as e:
                return coords, {}, str(e)

        with ThreadPoolExecutor(max_workers=WMS_QUERY_WORKERS) as executor:
            futures = {executor.submit(fetch_dates_for_tile, c): c for c in coords_needing_dates}

            for fut in as_completed(futures):
                coords, year_dates, error = fut.result()
                with lock:
                    fetched += 1
                    if error:
                        failed_tiles.append((coords, error))
                    else:
                        # Apply dates to all matching (coords, year) entries
                        if coords in tiles:
                            for year, date_str in year_dates.items():
                                if year in tiles[coords] and tiles[coords][year].get("acquisition_date") is None:
                                    tiles[coords][year]["acquisition_date"] = date_str
                                    dates_applied += 1

                    if not self.quiet and (fetched % 500 == 0 or fetched == len(coords_needing_dates)):
                        elapsed = time.perf_counter() - started
                        rate = fetched / elapsed if elapsed > 0 else 0
                        print(
                            f"\r  {fetched}/{len(coords_needing_dates)} tiles queried, {dates_applied} dates applied ({rate:.1f} req/s)",
                            end="",
                            flush=True,
                        )

        if not self.quiet:
            print()

        if failed_tiles:
            failed_coords = [str(coords) for coords, _ in failed_tiles]
            raise RuntimeError(
                f"WMS date fetch failed for {len(failed_tiles)} tiles: {', '.join(failed_coords[:10])}"
                + (f" and {len(failed_tiles) - 10} more" if len(failed_tiles) > 10 else "")
            )

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

    def _load_catalog(self) -> Catalog:
        """Load NRW catalog from ATOM feeds (current + historic).

        Fetches current feed and all historic year feeds in parallel,
        combining them into a single catalog.
        """
        tiles: dict[tuple[int, int], dict[int, dict]] = {}

        # 1. Current tiles
        if not self.quiet:
            print("  Loading current tiles from ATOM feed...")
        try:
            current_tiles = self._parse_jp2_feed_with_year(
                self._session, self.CURRENT_JP2_FEED_URL, self.CURRENT_JP2_BASE_URL
            )
            for coords, (url, year) in current_tiles.items():
                tiles.setdefault(coords, {})[year] = {"url": url, "acquisition_date": None}
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
                    tiles.setdefault(coords, {})[year] = {"url": url, "acquisition_date": None}
                    added += 1
            successful_years.append(f"{hist_year}:+{added}")

        if successful_years and not self.quiet:
            successful_years.sort(key=lambda s: int(s.split(":")[0]))
            print(f"  Historic: {', '.join(successful_years)}")
        if failed_years and not self.quiet:
            print(f"  Skipped: {sorted(failed_years)}")

        # 3. Fetch acquisition dates via WMS
        self._fetch_acquisition_dates_parallel(tiles)

        # 4. LAZ tiles
        root = fetch_xml_feed(self._session, self._laz_feed_url)
        laz_tiles = self._parse_laz_tiles(root)
        if not self.quiet:
            print(f"  LAZ: {len(laz_tiles)} tiles")

        return Catalog(image_tiles=tiles, dsm_tiles=laz_tiles)

    def _parse_laz_tiles(self, root: ET.Element) -> dict[tuple[int, int], dict[int, dict]]:
        """Parse LAZ tiles from XML feed."""
        laz_tiles: dict[tuple[int, int], dict[int, dict]] = {}
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
                year = int(match.group(3))
                coords = (grid_x, grid_y)
                laz_tiles.setdefault(coords, {})[year] = {
                    "url": self._laz_base_url + filename,
                    "acquisition_date": None,
                }
        return laz_tiles
