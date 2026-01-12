"""TH (Thüringen) DOP downloader using LAS feed as query mask."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import ClassVar
from urllib.parse import parse_qs, urlparse

import requests

from georaffer.config import (
    CATALOG_CACHE_DIR,
    FEED_TIMEOUT,
    METERS_PER_KM,
    Region,
    TH_GRID_SIZE,
)
from georaffer.downloaders.base import Catalog, RegionDownloader
from georaffer.downloaders.feeds import fetch_xml_feed
from georaffer.runtime import InterruptManager


class THDownloader(RegionDownloader):
    """TH (Thüringen) DOP downloader (Orthophotos) using LAS tiles as mask."""

    # LAS Atom dataset feed (INSPIRE)
    LAS_DATASET_URL: ClassVar[str] = (
        "https://geoportal.geoportal-th.de/dienste/atom_th_hoehendaten_las"
        "?type=dataset&id=c8363eb8-7f2a-49b5-bb59-a1571f40a21f"
    )

    # DOP downloader endpoints (GaiaLight)
    DOP_OVERVIEW_URL: ClassVar[str] = (
        "https://geoportal.geoportal-th.de/gaialight-th/_apps/dladownload/_ajax/overview.php"
    )
    DOP_DOWNLOAD_URL: ClassVar[str] = (
        "https://geoportal.geoportal-th.de/gaialight-th/_apps/dladownload/download.php"
    )

    # LAS feed link pattern (captures year-range, x, y)
    LAS_LINK_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"/las_(\d{4}-\d{4})/las_(?:32_)?(\d{3})_(\d{4})_1_th_\1\.zip",
        re.IGNORECASE,
    )

    # DOP tile id pattern (e.g. 32666_5658 or 666_5658)
    BILDNR_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^(?:32)?(\d{3})_(\d{4})$")

    # Minimum acquisition year to include
    MIN_YEAR: ClassVar[int] = 2020

    # Concurrency for DOP overview queries
    QUERY_WORKERS: ClassVar[int] = 16

    def __init__(
        self,
        output_dir: str,
        imagery_from: tuple[int, int | None] | None = None,
        session: requests.Session | None = None,
        quiet: bool = False,
    ) -> None:
        super().__init__(Region.TH, output_dir, imagery_from=imagery_from, session=session, quiet=quiet)
        self._cache_path = CATALOG_CACHE_DIR / "th_catalog.json"
        self._min_year = self.MIN_YEAR
        self._max_year: int | None = None
        if imagery_from is not None:
            from_year, to_year = imagery_from
            if from_year is not None:
                self._min_year = max(self.MIN_YEAR, from_year)
            self._max_year = to_year

    def utm_to_grid_coords(
        self, utm_x: float, utm_y: float
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """TH uses a 1km grid for DOP tiles (EPSG:25832)."""
        grid_x = int(utm_x // TH_GRID_SIZE)
        grid_y = int(utm_y // TH_GRID_SIZE)
        return (grid_x, grid_y), (grid_x, grid_y)

    def image_filename_from_url(self, url: str) -> str:
        """Generate a stable filename from GaiaLight download URLs."""
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        log = params.get("log", [""])[0]
        year = "0000"
        bildnr = ""
        if log:
            bildflugnr, _, bildnr = log.partition("-")
            if len(bildflugnr) >= 4 and bildflugnr[:4].isdigit():
                year = bildflugnr[:4]
        match = self.BILDNR_PATTERN.match(bildnr)
        if match:
            grid_x = int(match.group(1))
            grid_y = int(match.group(2))
            return f"dop20rgb_32_{grid_x:03d}_{grid_y:04d}_1_th_{year}.zip"
        # Fallback: keep something deterministic even if parsing fails
        if log:
            safe = re.sub(r"[^A-Za-z0-9_\\-]", "_", log)
            return f"dop20rgb_th_{safe}_{year}.zip"
        return "dop20rgb_th_unknown.zip"

    # =========================== Catalog loading ===========================

    def _load_catalog(self) -> Catalog:
        """Build DOP and DSM catalog from LAS feed and DOP overview queries."""
        las_tiles = self._fetch_las_tiles()
        image_tiles: dict[tuple[int, int], dict[int, dict]] = {}
        dsm_tiles: dict[tuple[int, int], dict[int, dict]] = {}

        if not las_tiles:
            return Catalog(image_tiles=image_tiles, dsm_tiles=dsm_tiles)

        # Populate DSM tiles from LAS feed
        for coords, year, url in las_tiles.values():
            dsm_tiles.setdefault(coords, {})[year] = {
                "url": url,
                "acquisition_date": None,
            }

        if not self.quiet:
            print(f"  TH: Querying DOP overview for {len(las_tiles)} LAS tiles...")

        start = time.perf_counter()
        completed = 0

        with ThreadPoolExecutor(max_workers=self.QUERY_WORKERS) as executor:
            futures = {
                executor.submit(self._fetch_dop_for_tile, x, y): (x, y)
                for x, y in las_tiles
            }
            for fut in as_completed(futures):
                if InterruptManager.get().is_set():
                    break
                results = fut.result()
                for coords, year, url, acq_date in results:
                    image_tiles.setdefault(coords, {})[year] = {
                        "url": url,
                        "acquisition_date": acq_date,
                    }
                completed += 1
                if not self.quiet and completed % 1000 == 0:
                    elapsed = time.perf_counter() - start
                    print(f"    {completed}/{len(las_tiles)} tiles checked ({elapsed:.1f}s)")

        return Catalog(image_tiles=image_tiles, dsm_tiles=dsm_tiles)

    def _fetch_las_tiles(self) -> dict[tuple[int, int], tuple[tuple[int, int], int, str]]:
        """Parse LAS Atom dataset feed and return tile info keyed by coords.

        Returns:
            Dict mapping (grid_x, grid_y) to (coords, year, url) tuples.
            Year is extracted from the year-range in the URL (uses end year).
        """
        tiles: dict[tuple[int, int], tuple[tuple[int, int], int, str]] = {}

        root = fetch_xml_feed(self._session, self.LAS_DATASET_URL, timeout=FEED_TIMEOUT)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for link in root.findall(".//atom:link", ns):
            if InterruptManager.get().is_set():
                break
            if link.attrib.get("rel") != "section":
                continue
            href = link.attrib.get("href", "")
            match = self.LAS_LINK_PATTERN.search(href)
            if not match:
                continue
            year_range = match.group(1)  # e.g., "2020-2022"
            year = int(year_range.split("-")[1])  # Use end year
            coords = (int(match.group(2)), int(match.group(3)))
            tiles[coords] = (coords, year, href)

        return tiles

    def _fetch_dop_for_tile(
        self, grid_x: int, grid_y: int
    ) -> list[tuple[tuple[int, int], int, str, str | None]]:
        """Query DOP overview for a 1km tile and return matching download items."""
        if InterruptManager.get().is_set():
            return []

        minx = grid_x * METERS_PER_KM
        miny = grid_y * METERS_PER_KM
        maxx = minx + METERS_PER_KM
        maxy = miny + METERS_PER_KM
        params = {
            "crs": "EPSG:25832",
            "bbox[]": [minx, miny, maxx, maxy],
            "type[]": "op",
        }

        try:
            resp = self._session.get(self.DOP_OVERVIEW_URL, params=params, timeout=FEED_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            return []

        if not payload.get("success"):
            return []

        features = payload.get("result", {}).get("features", [])
        results: list[tuple[tuple[int, int], int, str, str | None]] = []
        for feature in features:
            item = self._parse_dop_feature(feature)
            if item:
                results.append(item)
        return results

    def _parse_dop_feature(
        self, feature: dict
    ) -> tuple[tuple[int, int], int, str, str | None] | None:
        props = feature.get("properties", {})
        if props.get("type") != "op":
            return None

        date_str = props.get("datum") or ""
        year = int(date_str[:4]) if date_str[:4].isdigit() else None
        if year is None:
            return None
        if year < self._min_year:
            return None
        if self._max_year is not None and year > self._max_year:
            return None
        coords = self._coords_from_feature(feature)
        if coords is None:
            return None

        gid = props.get("gid")
        bildflugnr = props.get("bildflugnr")
        bildnr = props.get("bildnr")
        if not (gid and bildflugnr and bildnr):
            return None

        url = f"{self.DOP_DOWNLOAD_URL}?type=op&id={gid}&log={bildflugnr}-{bildnr}"
        acq_date = date_str[:10] if date_str else None
        return coords, year, url, acq_date

    def _coords_from_feature(self, feature: dict) -> tuple[int, int] | None:
        props = feature.get("properties", {})
        bildnr = props.get("bildnr", "")
        match = self.BILDNR_PATTERN.match(str(bildnr))
        if match:
            return int(match.group(1)), int(match.group(2))

        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [])
        if not coords:
            return None
        ring = coords[0] if coords else []
        if not ring:
            return None
        minx = min(pt[0] for pt in ring)
        miny = min(pt[1] for pt in ring)
        return int(minx // METERS_PER_KM), int(miny // METERS_PER_KM)
