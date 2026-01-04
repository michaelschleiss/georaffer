"""BY (Bayern/Bavaria) tile downloader for DOP20 and DOM20 raster tiles.

Catalog source: Metalink files at geodaten.bayern.de/odd/a/
Download source: https://download1.bayernwolke.de/a/
Historic DOP WMS: https://geoservices.bayern.de/od/wms/histdop/v1/histdop

Grid system:
- 1km x 1km tiles (like NRW)
- CRS: EPSG:25832 (UTM Zone 32N)
- Direct GeoTIFF downloads (no ZIP, unlike BW/BB)

Metalink structure:
- 7 district metalinks (091-097) for Regierungsbezirke
- Each contains ~10k tiles with SHA-256 verification hashes
- URLs: https://geodaten.bayern.de/odd/a/{product}/meta/metalink/09{1-7}.meta4
"""

import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import ClassVar

import requests

from georaffer.config import (
    BY_GRID_SIZE,
    CATALOG_CACHE_DIR,
    FEED_TIMEOUT,
    RETRY_BACKOFF_BASE,
    Region,
    WMS_COVERAGE_RETRIES,
    WMS_QUERY_WORKERS,
    WMS_RETRY_MAX_WAIT,
    WMS_TIMEOUT,
)
from georaffer.downloaders.base import Catalog, RegionDownloader
from georaffer.downloaders.wms import WMSImagerySource, _normalize_wms_date

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

    # Historic DOP via WMS
    WMS_BASE_URL: ClassVar[str] = "https://geoservices.bayern.de/od/wms/histdop/v1/histdop"
    WMS_RGB_LAYER_PATTERN: ClassVar[str] = "by_dop_{year}_h"
    WMS_INFO_LAYER_PATTERN: ClassVar[str] = "by_dop_{year}_h_info"

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
        self._wms_years_cache: list[int] | None = None
        self._wms: WMSImagerySource | None = None

        # Parse imagery_from for download filtering
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
        lowered = url.lower()
        if "service=wms" in lowered and "request=getmap" in lowered:
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            bbox = params.get("BBOX", params.get("bbox", [""]))[0].split(",")
            if len(bbox) < 4:
                raise ValueError(f"BY WMS URL missing BBOX: {url}")
            minx_m = float(bbox[0])
            miny_m = float(bbox[1])
            grid_x = int(round(minx_m / BY_GRID_SIZE))
            grid_y = int(round(miny_m / BY_GRID_SIZE))
            layer = params.get("LAYERS", params.get("layers", [""]))[0]
            year_match = re.search(r"(\d{4})", layer)
            if not year_match:
                raise ValueError(f"BY WMS URL missing year layer: {url}")
            year = year_match.group(1)
            return f"32{grid_x:03d}_{grid_y:04d}_{year}.tif"

        name = Path(url).name
        if not name.lower().endswith(".tif"):
            raise ValueError(f"BY downloads must be TIF files (got {name}).")
        return name

    @property
    def wms(self) -> WMSImagerySource:
        """Lazy-init WMS imagery source for historic DOP."""
        if self._wms is None:
            self._wms = WMSImagerySource(
                base_url=self.WMS_BASE_URL,
                rgb_layer_pattern=self.WMS_RGB_LAYER_PATTERN,
                info_layer_pattern=self.WMS_INFO_LAYER_PATTERN,
                tile_size_m=BY_GRID_SIZE,
                resolution_m=0.2,
                image_format="image/tiff",
                crs="EPSG:25832",
                session=self._session,
            )
        return self._wms

    def _historic_years(self) -> list[int]:
        """Fetch available historic years from WMS capabilities."""
        if self._wms_years_cache is not None:
            return self._wms_years_cache

        params = {"SERVICE": "WMS", "REQUEST": "GetCapabilities"}
        resp = self._session.get(self.WMS_BASE_URL, params=params, timeout=FEED_TIMEOUT)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        years: list[int] = []
        # BY WMS capabilities are not namespaced; handle both variants.
        for xpath, ns in (
            (".//Layer/Layer", None),
            (".//wms:Layer/wms:Layer", {"wms": "http://www.opengis.net/wms"}),
        ):
            for layer in root.findall(xpath, ns or {}):
                name_el = layer.find("Name") if ns is None else layer.find("wms:Name", ns)
                if name_el is None or not name_el.text:
                    continue
                name = name_el.text.strip()
                match = re.fullmatch(r"by_dop_(\d{4})_h", name)
                if match:
                    years.append(int(match.group(1)))

        years = sorted(set(years))
        self._wms_years_cache = years
        return years

    def _wms_bbox(self, grid_x: int, grid_y: int) -> tuple[int, int, int, int]:
        """Return WMS BBOX for a 1km BY grid tile."""
        minx = grid_x * BY_GRID_SIZE
        miny = grid_y * BY_GRID_SIZE
        maxx = minx + BY_GRID_SIZE
        maxy = miny + BY_GRID_SIZE
        return (minx, miny, maxx, maxy)

    def _parse_wms_featureinfo(
        self, text: str, years: set[int]
    ) -> dict[int, dict[str, str | None]]:
        """Parse BY WMS GetFeatureInfo response for multiple layers."""
        if "Search returned no results" in text or "no results" in text.lower():
            return {}

        result: dict[int, dict[str, str | None]] = {}
        current_year: int | None = None

        for line in text.splitlines():
            line = line.strip()
            layer_match = re.match(r"Layer 'by_dop_(\d{4})_h_info'\s*$", line)
            if layer_match:
                year = int(layer_match.group(1))
                current_year = year if year in years else None
                if current_year is not None and current_year not in result:
                    result[current_year] = {"acquisition_date": None}
                continue

            if current_year is None:
                continue

            ua_match = re.search(r"ua\s*=\s*'([^']*)'", line)
            if ua_match:
                raw_date = ua_match.group(1).strip()
                if raw_date:
                    normalized = _normalize_wms_date(raw_date) or raw_date
                    result[current_year]["acquisition_date"] = normalized

        return result

    def _wms_check_coverage_multi(
        self,
        years: list[int],
        grid_x: int,
        grid_y: int,
    ) -> dict[int, dict[str, str | None]]:
        """Query BY WMS for multiple years in one GetFeatureInfo call."""
        if not years:
            return {}

        bbox = self._wms_bbox(grid_x, grid_y)
        layers = ",".join(self.WMS_INFO_LAYER_PATTERN.format(year=y) for y in years)
        params = {
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetFeatureInfo",
            "LAYERS": layers,
            "QUERY_LAYERS": layers,
            "STYLES": "",
            "SRS": "EPSG:25832",
            "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "WIDTH": "100",
            "HEIGHT": "100",
            "X": "50",
            "Y": "50",
            "INFO_FORMAT": "text/plain",
        }

        last_error: Exception | None = None
        for attempt in range(WMS_COVERAGE_RETRIES):
            try:
                response = self._session.get(
                    self.WMS_BASE_URL,
                    params=params,
                    timeout=WMS_TIMEOUT,
                )
                response.raise_for_status()
                return self._parse_wms_featureinfo(response.text, set(years))
            except requests.RequestException as e:
                last_error = e
                if attempt < WMS_COVERAGE_RETRIES - 1:
                    wait_time = min(RETRY_BACKOFF_BASE**attempt, WMS_RETRY_MAX_WAIT)
                    time.sleep(wait_time)

        raise RuntimeError(
            f"BY WMS coverage check failed after {WMS_COVERAGE_RETRIES} attempts: {last_error}"
        )

    def dsm_filename_from_url(self, url: str) -> str:
        return self._filename_from_url(url)

    def image_filename_from_url(self, url: str) -> str:
        return self._filename_from_url(url)

    def _load_catalog(self) -> Catalog:
        """Load BY catalog from district metalink files.

        Fetches all 7 district metalinks in parallel for both DOP20 and DOM20,
        then queries historic DOP coverage via WMS.
        """
        # DOP tiles
        if not self.quiet:
            print("  Loading DOP20 tiles from metalink catalogs...")
        image_tiles: dict[tuple[int, int], dict[int, dict]] = {}
        current_urls: dict[tuple[int, int], str] = {}

        for coords, url in self._fetch_all_metalinks(self.DOP_METALINK_BASE, "dop"):
            if coords:
                current_urls[coords] = url
                image_tiles.setdefault(coords, {})

        if not self.quiet:
            print(f"    {len(image_tiles)} tiles")

        # Historic DOP tiles via WMS (independent of imagery_from)
        current_assigned: set[tuple[int, int]] = set()
        if os.getenv("GEORAFFER_DISABLE_WMS") != "1" and image_tiles:
            historic_years = [y for y in self._historic_years() if y >= 2010]
            if historic_years and not self.quiet:
                print(
                    f"  Querying BY historic WMS for {len(image_tiles)} tiles × {len(historic_years)} years "
                    f"({WMS_QUERY_WORKERS} workers)..."
                )

            checked = 0
            failed = 0
            historic_found = 0
            started = time.perf_counter()

            tiles_list = list(image_tiles.keys())
            with ThreadPoolExecutor(max_workers=WMS_QUERY_WORKERS) as executor:
                futures = {}
                for grid_x, grid_y in tiles_list:
                    missing_years = [
                        y for y in historic_years if y not in image_tiles.get((grid_x, grid_y), {})
                    ]
                    if not missing_years:
                        continue
                    futures[
                        executor.submit(
                            self._wms_check_coverage_multi, missing_years, grid_x, grid_y
                        )
                    ] = (grid_x, grid_y)

                total = len(futures)
                for fut in as_completed(futures):
                    checked += 1
                    grid_x, grid_y = futures[fut]
                    try:
                        coverage = fut.result()
                    except RuntimeError:
                        failed += 1
                        coverage = {}

                    for year, meta in coverage.items():
                        if year in image_tiles.get((grid_x, grid_y), {}):
                            continue
                        url = self.wms.get_tile_url(year, grid_x, grid_y)
                        acq_date = meta.get("acquisition_date") if meta else None
                        image_tiles.setdefault((grid_x, grid_y), {})[year] = {
                            "url": url,
                            "acquisition_date": acq_date,
                        }
                        historic_found += 1
                    if coverage:
                        current_url = current_urls.get((grid_x, grid_y))
                        if current_url:
                            latest_year = max(coverage)
                            meta = coverage[latest_year]
                            image_tiles.setdefault((grid_x, grid_y), {})[latest_year] = {
                                "url": current_url,
                                "acquisition_date": meta.get("acquisition_date") if meta else None,
                            }
                            current_assigned.add((grid_x, grid_y))

                    if not self.quiet and total and (checked % 100 == 0 or checked == total):
                        elapsed = time.perf_counter() - started
                        rate = checked / elapsed if elapsed > 0 else 0
                        print(
                            f"\r  {checked}/{total} checked, "
                            f"{historic_found} historic found ({rate:.1f} req/s)",
                            end="",
                            flush=True,
                        )

            if not self.quiet and total:
                print()
            if failed and not self.quiet:
                print(f"  Warning: {failed} WMS queries failed")

        if current_urls:
            from datetime import date

            fallback_year = date.today().year
            for coords, url in current_urls.items():
                if coords in current_assigned:
                    continue
                if fallback_year in image_tiles.get(coords, {}):
                    continue
                image_tiles.setdefault(coords, {})[fallback_year] = {
                    "url": url,
                    "acquisition_date": None,
                }

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
