"""NRW (North Rhine-Westphalia) tile downloader."""

import sys
import xml.etree.ElementTree as ET
from typing import ClassVar

import requests

from georaffer.config import FEED_TIMEOUT, NRW_GRID_SIZE, NRW_JP2_PATTERN, NRW_LAZ_PATTERN, Region
from georaffer.downloaders.base import Catalog, RegionDownloader
from georaffer.runtime import parallel_map


class NRWDownloader(RegionDownloader):
    """NRW (North Rhine-Westphalia) downloader."""

    # Current feed URL (always the same, used for deduplication in historic mode)
    CURRENT_JP2_FEED_URL = (
        "https://www.opengeodata.nrw.de/produkte/geobasis/lusat/akt/dop/dop_jp2_f10/index.xml"
    )
    CURRENT_JP2_BASE_URL = (
        "https://www.opengeodata.nrw.de/produkte/geobasis/lusat/akt/dop/dop_jp2_f10/"
    )

    # Available historic years (UTM era)
    # NOTE: 2014 is the first year where NRW historic orthophotos are consistently "dop10"
    # and aligned with the georaffer tiling assumptions; 2010–2013 include "dop20" variants
    # that cover different areas and are skipped in this pipeline.
    # The upper bound is discovered dynamically from the provider index when needed.
    HISTORIC_YEARS: ClassVar[list[int]] = list(range(2014, 2024))  # fallback: 2014-2023
    HISTORIC_INDEX_URL = (
        "https://www.opengeodata.nrw.de/produkte/geobasis/lusat/hist/"
        "hist_dop/hist_dop_jp2_f10/index.xml"
    )

    def __init__(
        self,
        output_dir: str,
        imagery_from: tuple[int, int | None] | None = None,
        session: requests.Session | None = None,
    ):
        super().__init__(Region.NRW, output_dir, imagery_from=imagery_from, session=session)

        if imagery_from is None:
            self._jp2_feed_url = self.CURRENT_JP2_FEED_URL
            self.jp2_base_url = self.CURRENT_JP2_BASE_URL
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
            self._jp2_feed_url = f"https://www.opengeodata.nrw.de/produkte/geobasis/lusat/hist/hist_dop/hist_dop_jp2_f10/hist_dop_{from_year}/index.xml"
            self.jp2_base_url = f"https://www.opengeodata.nrw.de/produkte/geobasis/lusat/hist/hist_dop/hist_dop_jp2_f10/hist_dop_{from_year}/"

        self._laz_feed_url = (
            "https://www.opengeodata.nrw.de/produkte/geobasis/hm/bdom50_las/bdom50_las/index.xml"
        )
        self.laz_base_url = (
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
    def laz_feed_url(self) -> str:
        return self._laz_feed_url

    def _parse_jp2_feed_with_year(
        self, session: requests.Session, feed_url: str, base_url: str
    ) -> dict[tuple[int, int], tuple[str, int]]:
        """Parse NRW JP2 feed and return tiles with year info.

        Args:
            session: HTTP session for requests
            feed_url: URL to XML feed
            base_url: Base URL for tile downloads

        Returns:
            Dict mapping (grid_x, grid_y) -> (download_url, year) where:
            - Key: (grid_x, grid_y) tile coordinates in km
            - Value: (url, year) tuple with download URL and acquisition year
        """
        response = session.get(feed_url, timeout=FEED_TIMEOUT)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        # Check for EPSG subdirectory (historic feeds have this)
        folders = root.findall(".//folder")
        if folders and any(folder.get("name") == "epsg_25832" for folder in folders):
            epsg_feed_url = feed_url.replace("/index.xml", "/epsg_25832/index.xml")
            response = session.get(epsg_feed_url, timeout=FEED_TIMEOUT)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            base_url = base_url + "epsg_25832/"

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

    def _available_historic_years(self) -> list[int]:
        try:
            response = self._session.get(self.HISTORIC_INDEX_URL, timeout=FEED_TIMEOUT)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            years = sorted(
                {
                    int((folder.get("name") or "").rsplit("_", 1)[-1])
                    for folder in root.findall(".//folder")
                    if (folder.get("name") or "").startswith("hist_dop_")
                    and (folder.get("name") or "").rsplit("_", 1)[-1].isdigit()
                    and int((folder.get("name") or "").rsplit("_", 1)[-1]) >= 2014
                }
            )
            return years or self.HISTORIC_YEARS
        except Exception:
            return self.HISTORIC_YEARS

    def _parse_jp2_feed(
        self, session: requests.Session, root: ET.Element
    ) -> dict[tuple[int, int], str]:
        """Parse NRW JP2 feed with historical EPSG subdirectory support."""
        jp2_tiles = {}

        folders = root.findall(".//folder")
        if folders and any(folder.get("name") == "epsg_25832" for folder in folders):
            epsg_feed_url = self._jp2_feed_url.replace("/index.xml", "/epsg_25832/index.xml")
            response = session.get(epsg_feed_url, timeout=FEED_TIMEOUT)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            base_url = self.jp2_base_url + "epsg_25832/"
        else:
            base_url = self.jp2_base_url

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
                jp2_tiles[(grid_x, grid_y)] = base_url + filename

        return jp2_tiles

    def get_available_tiles(self) -> tuple[dict, dict]:
        """Get available JP2 and LAZ tiles, loading multiple years if --imagery-from is set.

        When imagery_from is set, loads years in the specified range (from_year to to_year,
        or from_year to present if to_year is None). Returns all available tile versions.
        Tiles with the same coords+year are deduplicated (current feed takes precedence).

        Returns:
            Tuple of (jp2_tiles, laz_tiles) dicts mapping coords to URLs.
            In multi-year mode, returns ALL tile versions (unique filenames).
        """
        # LAZ tiles use base class method (no historic LAZ feeds exist)
        laz_tiles = self._fetch_and_parse_feed(self.laz_feed_url, "laz")

        # If not in historic mode, use standard parsing
        if self._from_year is None:
            jp2_tiles = self._fetch_and_parse_feed(self.jp2_feed_url, "jp2")
            # Set _all_jp2_by_coord for base class total_jp2_count property
            self._all_jp2_by_coord = {coords: [url] for coords, url in jp2_tiles.items()}
            return jp2_tiles, laz_tiles

        # Multi-year historic mode: load feeds for the requested year range
        # This allows downloading orthophotos from multiple years for the same location
        from_year = self._from_year
        to_year = self._to_year

        # Collect tiles from all years: (coords, year) -> url
        # Key is (coords, year) because same coords can appear in multiple years
        all_tiles: dict[tuple[tuple[int, int], int], str] = {}

        # Load current feed first (takes precedence for duplicates)
        # Current feed may have updated versions of historic tiles - use these when available
        try:
            current_tiles = self._parse_jp2_feed_with_year(
                self._session, self.CURRENT_JP2_FEED_URL, self.CURRENT_JP2_BASE_URL
            )
            kept_current = 0
            skipped_current = 0
            for coords, (url, year) in current_tiles.items():
                if self._year_in_range(year, from_year, to_year):
                    all_tiles[(coords, year)] = url
                    kept_current += 1
                else:
                    skipped_current += 1
            if skipped_current:
                print(
                    f"  Current feed: {len(current_tiles)} tiles "
                    f"({kept_current} in range, {skipped_current} skipped)"
                )
            else:
                print(f"  Current feed: {len(current_tiles)} tiles")
        except Exception as e:
            print(f"Failed to fetch current JP2 feed: {e}", file=sys.stderr)
            raise

        # Determine year range: from_year to to_year (or latest historic if to_year is None)
        historic_years = [
            y
            for y in self._available_historic_years()
            if self._year_in_range(y, from_year, to_year)
        ]

        total_historic = 0
        duplicates_skipped = 0
        successful_years = []
        failed_years = []

        def fetch_year(hist_year: int) -> dict | None:
            """Fetch a single year's feed. Returns tiles_dict or None on error."""
            feed_url = f"https://www.opengeodata.nrw.de/produkte/geobasis/lusat/hist/hist_dop/hist_dop_jp2_f10/hist_dop_{hist_year}/index.xml"
            base_url = f"https://www.opengeodata.nrw.de/produkte/geobasis/lusat/hist/hist_dop/hist_dop_jp2_f10/hist_dop_{hist_year}/"
            try:
                return self._parse_jp2_feed_with_year(self._session, feed_url, base_url)
            except Exception:
                return None

        # Fetch all years in parallel
        for hist_year, historic_tiles in parallel_map(fetch_year, historic_years, max_workers=8):
            if historic_tiles is None:
                failed_years.append(str(hist_year))
                continue
            added = 0
            for coords, (url, year) in historic_tiles.items():
                key = (coords, year)
                if key in all_tiles:
                    duplicates_skipped += 1
                else:
                    all_tiles[key] = url
                    added += 1
            total_historic += added
            successful_years.append(f"{hist_year}:+{added}")

        if successful_years:
            # Sort by year for consistent output
            successful_years.sort(key=lambda s: int(s.split(":")[0]))
            print(f"  Historic: {', '.join(successful_years)}")
        if failed_years:
            print(f"  Skipped (format mismatch): {', '.join(failed_years)}")

        # Flatten all_tiles to match the expected return interface
        # Challenge: base interface expects dict[coords -> url] (single URL per location)
        # but we have multiple years per location that we want to download
        #
        # Solution: Return one URL per coords in jp2_tiles (for catalog/display purposes),
        # but store ALL URLs in instance variables so tiles.py can retrieve them via
        # get_all_urls_for_coord() when building download lists
        jp2_tiles = {}
        for (coords, _year), url in all_tiles.items():
            jp2_tiles[coords] = url  # Last year wins for display, but doesn't matter

        # Store complete mapping for multi-year downloads
        # tiles.py checks for this attribute to enable multi-year mode
        self._all_jp2_by_coord: dict[tuple[int, int], list[str]] = {}
        for (coords, _year), url in all_tiles.items():
            if coords not in self._all_jp2_by_coord:
                self._all_jp2_by_coord[coords] = []
            self._all_jp2_by_coord[coords].append(url)

        total_tiles = len(all_tiles)
        unique_coords = len(jp2_tiles)
        year_range_str = f"{from_year}-{to_year}" if to_year else f"{from_year}-present"
        print(
            f"  Total: {total_tiles} tiles across {unique_coords} locations ({year_range_str}), {duplicates_skipped} duplicates skipped"
        )

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
                laz_tiles[(grid_x, grid_y)] = self.laz_base_url + filename

        return laz_tiles

    def _load_catalog(self) -> Catalog:
        """Load NRW catalog from ATOM feeds. TODO: implement."""
        return Catalog()
