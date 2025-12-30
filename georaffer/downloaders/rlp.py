"""RLP (Rhineland-Palatinate) tile downloader.

RLP Open Data Portal: https://geobasis-rlp.de/data/
├── dop20rgb/     - Current orthophotos (JP2, 0.2m resolution)
├── dop20rgbi/    - Current orthophotos with infrared (JP2, 0.2m resolution)
├── bdom20rgbi/   - Point clouds / DSM (LAZ, 0.2m spacing)
└── ...           - Other datasets (DGM, DTK, etc.)

Each dataset has:
  - current/jp2/ or current/las/  - Raw data files
  - current/meta4/                - Metalink feeds with SHA-256 hashes
  - current/*/atomfeed-links/     - ATOM-style link lists (used here)
  - current/metadata/             - ISO 19139 XML metadata per tile

Historical imagery (1994-2024) is only available via WMS:
  https://geo4.service24.rlp.de/wms/rp_hkdop20.fcgi

Note: The old geoportal.rlp.de INSPIRE ATOM feeds were deprecated April 2025.
"""

import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import ClassVar
from urllib.parse import parse_qs, urlparse

import requests
import truststore

from georaffer.config import (
    CATALOG_CACHE_DIR,
    METERS_PER_KM,
    RLP_GRID_SIZE,
    RLP_JP2_PATTERN,
    RLP_LAZ_PATTERN,
    WMS_QUERY_WORKERS,
    Region,
)
from georaffer.downloaders.base import Catalog, RegionDownloader
from georaffer.downloaders.feeds import fetch_xml_feed
from georaffer.downloaders.wms import WMSImagerySource

truststore.inject_into_ssl()


class RLPDownloader(RegionDownloader):
    """RLP (Rhineland-Palatinate) downloader.

    Supports both current imagery (via ATOM feed) and historical imagery (via WMS).
    Historical imagery available from 1994-2024 via WMS service.
    """

    # WMS service for historical imagery
    WMS_BASE_URL = "https://geo4.service24.rlp.de/wms/rp_hkdop20.fcgi"
    WMS_RGB_LAYER_PATTERN = "rp_dop20_rgb_{year}"
    WMS_INFO_LAYER_PATTERN = "rp_dop20_info_{year}"

    # Available historic years via WMS
    HISTORIC_YEARS: ClassVar[list[int]] = list(range(1994, 2025))  # 1994-2024

    def __init__(
        self,
        output_dir: str,
        imagery_from: tuple[int, int | None] | None = None,
        session: requests.Session | None = None,
        quiet: bool = False,
    ):
        super().__init__(Region.RLP, output_dir, imagery_from=imagery_from, session=session, quiet=quiet)
        self._cache_path = CATALOG_CACHE_DIR / "rlp_catalog.json"

        # GeoBasis-RLP ATOM feeds (replaced deprecated geoportal feeds April 2025)
        self._jp2_feed_url = (
            "https://geobasis-rlp.de/data/dop20rgb/current/jp2/atomfeed-links/atomfeed-links.xml"
        )
        self._laz_feed_url = (
            "https://geobasis-rlp.de/data/bdom20rgbi/current/las/atomfeed-links/atomfeed-links.xml"
        )

        # Parse imagery_from (like NRW)
        if imagery_from is None:
            self._from_year = None
            self._to_year = None
        else:
            from_year, to_year = imagery_from
            if from_year < 1994:
                raise ValueError(f"Year {from_year} not supported. Use year >= 1994.")
            if from_year > 2024:
                raise ValueError(f"Year {from_year} not supported. Use year <= 2024.")
            self._from_year = from_year
            self._to_year = to_year

        # Lazy-init WMS source
        self._wms: WMSImagerySource | None = None

    @property
    def wms(self) -> WMSImagerySource:
        """Lazy-init WMS imagery source."""
        if self._wms is None:
            self._wms = WMSImagerySource(
                base_url=self.WMS_BASE_URL,
                rgb_layer_pattern=self.WMS_RGB_LAYER_PATTERN,
                info_layer_pattern=self.WMS_INFO_LAYER_PATTERN,
                tile_size_m=RLP_GRID_SIZE,
                resolution_m=0.2,
                session=self._session,
            )
        return self._wms

    def utm_to_grid_coords(
        self, utm_x: float, utm_y: float
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """RLP uses 2km grid with km-based coordinates.

        Filenames use km coordinates (e.g., 362, 5604).
        Formula: tile_index * km_per_tile
        Example: 362500m // 2000 = 181 → 181 * 2 = 362km
        """
        km_per_tile = RLP_GRID_SIZE // METERS_PER_KM  # 2
        grid_x = int(utm_x // RLP_GRID_SIZE) * km_per_tile
        grid_y = int(utm_y // RLP_GRID_SIZE) * km_per_tile
        return (grid_x, grid_y), (grid_x, grid_y)

    @property
    def jp2_feed_url(self) -> str:
        return self._jp2_feed_url

    @property
    def laz_feed_url(self) -> str:
        return self._laz_feed_url

    @staticmethod
    def _is_wms_getmap(url: str) -> bool:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        service = params.get("SERVICE") or params.get("service")
        request = params.get("REQUEST") or params.get("request")
        if service and request:
            return service[0].lower() == "wms" and request[0].lower() == "getmap"
        lowered = url.lower()
        return "service=wms" in lowered and "request=getmap" in lowered

    def download_file(self, url: str, output_path: str, on_progress=None) -> bool:
        """Download a file, routing WMS GetMap URLs through WMS validation."""
        if self._is_wms_getmap(url):
            if self.wms.download_tile(url, output_path, on_progress=on_progress):
                return True
            raise RuntimeError(f"WMS download failed for {url}")
        return super().download_file(url, output_path, on_progress=on_progress)

    def _extract_year_from_url(self, url: str) -> int | None:
        """Extract year from JP2 URL filename."""
        filename = url.split("/")[-1]
        match = RLP_JP2_PATTERN.match(filename)
        if match:
            return int(match.group(3))
        return None

    def _parse_jp2_tiles(self, root: ET.Element) -> dict[tuple[int, int], str]:
        """Parse JP2 tiles from ATOM feed XML."""
        jp2_tiles = {}
        for link_elem in root.findall('.//link[@type="image/jp2"]'):
            url = link_elem.get("href")
            if url and url.endswith(".jp2"):
                filename = url.split("/")[-1]
                match = RLP_JP2_PATTERN.match(filename)
                if not match:
                    raise ValueError(
                        f"RLP JP2 '{filename}' doesn't match pattern. "
                        f"Expected: dop20rgb_32_XXX_YYYY_2_rp_YEAR.jp2"
                    )
                grid_x = int(match.group(1))
                grid_y = int(match.group(2))
                jp2_tiles[(grid_x, grid_y)] = url
        return jp2_tiles

    def _parse_laz_tiles(self, root: ET.Element) -> dict[tuple[int, int], dict]:
        """Parse LAZ tiles from ATOM feed XML."""
        laz_tiles = {}
        for link_elem in root.findall(".//link"):
            url = link_elem.get("href")
            if url and url.endswith(".laz"):
                filename = url.split("/")[-1]
                match = RLP_LAZ_PATTERN.match(filename)
                if not match:
                    raise ValueError(
                        f"RLP LAZ '{filename}' doesn't match pattern. "
                        f"Expected: bdom20rgbi_32_XXX_YYYY_2_rp.laz"
                    )
                grid_x = int(match.group(1))
                grid_y = int(match.group(2))
                laz_tiles[(grid_x, grid_y)] = {"url": url, "acquisition_date": None}
        return laz_tiles

    def _load_catalog(self) -> Catalog:
        """Load RLP catalog from ATOM feed (current) + WMS (historic).

        Uses ATOM feed to discover valid tile coordinates, then queries WMS
        for historic coverage at each coordinate.
        """
        tiles: dict[tuple[int, int], dict[int, dict]] = {}

        # 1. Current tiles from ATOM feed
        if not self.quiet:
            print("  Loading current tiles from ATOM feed...")
        root = fetch_xml_feed(self._session, self._jp2_feed_url, wrap_content=True)
        current_tiles = self._parse_jp2_tiles(root)
        for coords, url in current_tiles.items():
            year = self._extract_year_from_url(url)
            if year:
                tiles.setdefault(coords, {})[year] = {"url": url, "acquisition_date": None}
        if not self.quiet:
            print(f"  Current: {len(current_tiles)} tiles")

        # 2. Historic tiles from WMS (query each ATOM coordinate)
        all_coords = set(current_tiles.keys())
        historic_years = self.HISTORIC_YEARS

        if not self.quiet:
            print(
                f"  Querying WMS for {len(all_coords)} tiles × {len(historic_years)} years "
                f"({WMS_QUERY_WORKERS} workers)..."
            )

        checked = 0
        failed = 0
        historic_found = 0
        started = time.perf_counter()

        with ThreadPoolExecutor(max_workers=WMS_QUERY_WORKERS) as executor:
            futures = {
                executor.submit(
                    self.wms.check_coverage_multi, historic_years, grid_x, grid_y
                ): (grid_x, grid_y)
                for grid_x, grid_y in all_coords
            }

            for fut in as_completed(futures):
                checked += 1
                grid_x, grid_y = futures[fut]
                try:
                    coverage = fut.result()
                except RuntimeError:
                    # WMS retries exhausted for this tile - count and continue
                    failed += 1
                    coverage = {}

                for year, meta in coverage.items():
                    coords = (grid_x, grid_y)
                    if year not in tiles.get(coords, {}):
                        url = self.wms.get_tile_url(year, grid_x, grid_y)
                        acq_date = meta.get("acquisition_date") if meta else None
                        tiles.setdefault(coords, {})[year] = {"url": url, "acquisition_date": acq_date}
                        historic_found += 1

                if not self.quiet and (checked % 100 == 0 or checked == len(all_coords)):
                    elapsed = time.perf_counter() - started
                    rate = checked / elapsed if elapsed > 0 else 0
                    print(
                        f"\r  {checked}/{len(all_coords)} checked, "
                        f"{historic_found} historic found ({rate:.1f} req/s)",
                        end="",
                        flush=True,
                    )

        if not self.quiet:
            print()
        if failed and not self.quiet:
            print(f"  Warning: {failed} WMS queries failed")

        # 3. LAZ tiles
        root = fetch_xml_feed(self._session, self._laz_feed_url, wrap_content=True)
        laz_tiles = self._parse_laz_tiles(root)
        if not self.quiet:
            print(f"  LAZ: {len(laz_tiles)} tiles")

        return Catalog(image_tiles=tiles, dsm_tiles=laz_tiles)
