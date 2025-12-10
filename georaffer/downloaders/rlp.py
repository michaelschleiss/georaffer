"""RLP (Rhineland-Palatinate) tile downloader."""

import sys
import xml.etree.ElementTree as ET
from typing import ClassVar

import requests
import urllib3

from georaffer.config import METERS_PER_KM, RLP_GRID_SIZE, RLP_JP2_PATTERN, RLP_LAZ_PATTERN, Region
from georaffer.downloaders.base import RegionDownloader
from georaffer.downloaders.wms import WMSImagerySource

# Suppress SSL warnings (RLP has certificate issues)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
    ):
        super().__init__(Region.RLP, output_dir, imagery_from=imagery_from, session=session)

        # ATOM feeds (current imagery)
        self._jp2_feed_url = "https://www.geoportal.rlp.de/mapbender/php/mod_inspireDownloadFeed.php?id=2b009ae4-aa3e-ff21-870b-49846d9561b2&type=DATASET&generateFrom=remotelist"
        self._laz_feed_url = "https://www.geoportal.rlp.de/mapbender/php/mod_inspireDownloadFeed.php?id=3d2dda7d-b4b5-47d2-b074-dd45edd36738&type=DATASET&generateFrom=remotelist"

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

    @property
    def verify_ssl(self) -> bool:
        return False  # RLP has SSL certificate issues

    def get_available_tiles(
        self,
        requested_coords: set[tuple[int, int]] | None = None,
    ) -> tuple[dict, dict]:
        """Get available JP2 and LAZ tiles.

        Args:
            requested_coords: For WMS historical mode only - tiles to query.
                            Ignored for ATOM mode (returns full catalog).

        Returns:
            Tuple of (jp2_tiles, laz_tiles) dicts mapping coords to URLs.
        """
        # LAZ always from ATOM (no historical LAZ via WMS)
        laz_tiles = self._fetch_and_parse_feed(self.laz_feed_url, "laz")

        # Current mode: use ATOM feed
        if self._from_year is None:
            jp2_tiles = self._fetch_and_parse_feed(self.jp2_feed_url, "jp2")
            return jp2_tiles, laz_tiles

        # Historical mode: use WMS
        if not requested_coords:
            print("  WMS mode requires requested_coords, returning empty catalog")
            return {}, laz_tiles

        return self._get_wms_tiles(requested_coords), laz_tiles

    def _get_wms_tiles(
        self,
        requested_coords: set[tuple[int, int]],
    ) -> dict[tuple[int, int], str]:
        """Query WMS for historical tiles (NRW-style multi-year).

        Args:
            requested_coords: Set of (grid_x, grid_y) coordinates to check

        Returns:
            Dict mapping coords to URLs (last year wins for display)
        """
        from_year = self._from_year
        to_year = self._to_year

        # from_year is guaranteed non-None here (called only when _from_year is set)
        assert from_year is not None

        # Determine year range
        if to_year is None:
            historic_years = [y for y in self.HISTORIC_YEARS if y >= from_year]
        else:
            historic_years = [y for y in self.HISTORIC_YEARS if from_year <= y <= to_year]

        # Collect tiles: (coords, year) -> url
        all_tiles: dict[tuple[tuple[int, int], int], str] = {}

        # Load current ATOM feed first (takes precedence, like NRW)
        current_tiles = self._fetch_and_parse_feed(self._jp2_feed_url, "jp2")
        for coords, url in current_tiles.items():
            year = self._extract_year_from_url(url)
            if year:
                all_tiles[(coords, year)] = url
        print(f"  Current feed: {len(current_tiles)} tiles")

        # Query WMS for each year (only requested tiles)
        total_wms = 0
        for hist_year in historic_years:
            added = 0
            for grid_x, grid_y in requested_coords:
                coverage = self.wms.check_coverage(hist_year, grid_x, grid_y)
                if coverage:
                    key = ((grid_x, grid_y), hist_year)
                    if key not in all_tiles:
                        url = self.wms.get_tile_url(hist_year, grid_x, grid_y)
                        all_tiles[key] = url
                        added += 1
            total_wms += added
            if added > 0:
                print(f"  hist_{hist_year}: +{added} tiles")

        # Flatten for interface (same as NRW)
        jp2_tiles = {}
        self._all_jp2_by_coord: dict[tuple[int, int], list[str]] = {}

        for (coords, _year), url in all_tiles.items():
            jp2_tiles[coords] = url  # Last year wins for display
            if coords not in self._all_jp2_by_coord:
                self._all_jp2_by_coord[coords] = []
            self._all_jp2_by_coord[coords].append(url)

        # Summary
        year_str = f"{from_year}-{to_year}" if to_year else f"{from_year}-present"
        print(
            f"  Total: {len(all_tiles)} tiles across {len(jp2_tiles)} locations ({year_str})"
        )

        return jp2_tiles

    def _extract_year_from_url(self, url: str) -> int | None:
        """Extract year from JP2 URL filename."""
        filename = url.split("/")[-1]
        match = RLP_JP2_PATTERN.match(filename)
        if match:
            return int(match.group(3))
        return None

    def get_all_urls_for_coord(self, coords: tuple[int, int]) -> list[str]:
        """Get all URLs (all years) for a coordinate. For multi-year mode."""
        if hasattr(self, "_all_jp2_by_coord"):
            return self._all_jp2_by_coord.get(coords, [])
        return []

    def _parse_jp2_feed(
        self, session: requests.Session, root: ET.Element
    ) -> dict[tuple[int, int], str]:
        """Parse RLP JP2 feed using INSPIRE/Atom namespace."""
        jp2_tiles = {}
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for link_elem in root.findall('.//atom:link[@type="image/jp2"]', ns):
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

    def _parse_laz_feed(
        self, session: requests.Session, root: ET.Element
    ) -> dict[tuple[int, int], str]:
        """Parse RLP LAZ feed using INSPIRE/Atom namespace."""
        laz_tiles = {}
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for link_elem in root.findall(".//atom:link", ns):
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
                laz_tiles[(grid_x, grid_y)] = url

        return laz_tiles
