"""TH (Thüringen) DOP downloader using LAS feed as query mask - OPTIMIZED VERSION."""

from __future__ import annotations

import re
import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path
from typing import ClassVar
from urllib.parse import parse_qs, urlparse

import requests

from georaffer.config import (
    CATALOG_CACHE_DIR,
    FEED_TIMEOUT,
    MAX_RETRIES,
    METERS_PER_KM,
    MIN_FILE_SIZE,
    Region,
    RETRY_BACKOFF_BASE,
    TH_GRID_SIZE,
)
from georaffer.downloaders.base import Catalog, RegionDownloader
from georaffer.downloaders.feeds import fetch_xml_feed
from georaffer.runtime import InterruptManager


class THDownloader(RegionDownloader):
    """TH (Thüringen) DOP + DOM downloader (Orthophotos + Digital Surface Model)."""

    # LAS Atom dataset feed (INSPIRE) - for spatial coverage mask only
    LAS_DATASET_URL: ClassVar[str] = (
        "https://geoportal.geoportal-th.de/dienste/atom_th_hoehendaten_las"
        "?type=dataset&id=c8363eb8-7f2a-49b5-bb59-a1571f40a21f"
    )

    # DOM Atom dataset feed (INSPIRE) - Digital Surface Model (DSM)
    DOM_DATASET_URL: ClassVar[str] = (
        "https://geoportal.geoportal-th.de/dienste/atom_th_hoehendaten_dom"
        "?type=dataset&id=3b5d8d9c-775d-4617-8dfe-71480d6472a6"
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

    # DOM feed link pattern (captures year-range, x, y)
    # Example: /dom_2020-2025/dom1_32_663_5657_1_th_2020-2025.zip
    DOM_LINK_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"/dom_(\d{4}-\d{4})/dom1_(?:32_)?(\d{3})_(\d{4})_1_th_\1\.zip",
        re.IGNORECASE,
    )

    # DOP tile id pattern (e.g. 32666_5658 or 666_5658)
    BILDNR_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^(?:32)?(\d{3})_(\d{4})$")

    # Minimum acquisition year to include
    MIN_YEAR: ClassVar[int] = 2020

    # Concurrency for DOP overview queries (server rate-limits to ~2-4 req/s)
    # Reduced to 8 to avoid rate limiting after DOM feed fetch
    QUERY_WORKERS: ClassVar[int] = 8

    # Thread-local storage for sessions (requests.Session is not thread-safe)
    _thread_local: ClassVar[threading.local] = threading.local()

    # API returns max 200 objects per request. Each tile has ~12 years of imagery.
    # Strategy: 3.9×3.9km bbox centered on grid points spaced 4km apart.
    # 3.9km stays safely under limit, 4km spacing provides near-complete coverage.
    BBOX_SIZE_KM: ClassVar[float] = 3.9  # Bbox size (centered on grid point)
    GRID_SPACING_KM: ClassVar[int] = 4  # Grid point spacing

    # Known DOP coverage gaps (40 tiles as of 2026-01-13)
    # These 1km LAS tiles fall at boundaries where 2km DOP coverage ends.
    # Validated: API returns 0 features for these areas (genuine data gaps, not extraction bugs).
    # Pattern: Clustered along Y-boundaries (e.g., Y=5652) and scattered isolated gaps.
    KNOWN_DOP_GAPS: ClassVar[set[tuple[int, int]]] = {
        (561, 5612), (561, 5613), (563, 5616), (563, 5617), (568, 5632),
        (572, 5652), (573, 5652), (574, 5652), (577, 5665), (577, 5666),
        (578, 5652), (579, 5659), (579, 5660), (587, 5595), (589, 5706),
        (592, 5593), (593, 5593), (595, 5589), (632, 5567), (648, 5698),
        (651, 5573), (651, 5576), (664, 5696), (674, 5671), (675, 5585),
        (675, 5586), (675, 5587), (682, 5666), (708, 5600), (708, 5607),
        (714, 5603), (715, 5603), (716, 5652), (717, 5652), (727, 5660),
        (729, 5656), (731, 5666), (732, 5630), (732, 5631), (746, 5661),
    }

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

    def dsm_filename_from_url(self, url: str) -> str:
        """Generate a stable filename for DSM downloads.

        DOM: Keep .zip extension (contains .xyz file to be converted)
        LAS: Change .zip to .laz (for extracted LAZ file)
        """
        parsed = urlparse(url)
        filename = Path(parsed.path).name
        if filename.lower().startswith("dom"):
            # DOM files: keep as ZIP (will be processed to extract XYZ)
            return filename
        elif filename.lower().startswith("las"):
            # LAS files: extract LAZ from ZIP
            if filename.lower().endswith(".zip"):
                return f"{filename[:-4]}.laz"
        return filename

    def _extract_laz_zip(self, zip_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            laz_name = next((n for n in zf.namelist() if n.lower().endswith(".laz")), None)
            if not laz_name:
                raise RuntimeError(f"No LAZ found in {zip_path.name}")
            tmp_path = output_path.with_suffix(".tmp")
            with zf.open(laz_name) as src, tmp_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            tmp_path.replace(output_path)

            meta_name = next((n for n in zf.namelist() if n.lower().endswith(".meta")), None)
            if meta_name:
                meta_path = output_path.with_suffix(".meta")
                tmp_meta = meta_path.with_suffix(".tmp")
                with zf.open(meta_name) as src, tmp_meta.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                tmp_meta.replace(meta_path)

    def download_file(self, url: str, output_path: str, on_progress=None) -> bool:
        output = Path(output_path)
        if output.suffix.lower() == ".laz":
            zip_path = output.with_suffix(".zip")
            if zip_path.exists() and zip_path.stat().st_size >= MIN_FILE_SIZE:
                try:
                    self._extract_laz_zip(zip_path, output)
                    with suppress(OSError):
                        zip_path.unlink()
                    return True
                except Exception:
                    with suppress(OSError):
                        zip_path.unlink()
            super().download_file(url, str(zip_path), on_progress=on_progress)
            self._extract_laz_zip(zip_path, output)
            with suppress(OSError):
                zip_path.unlink()
            return True
        return super().download_file(url, output_path, on_progress=on_progress)

    # =========================== Catalog loading ===========================

    def _load_catalog(self) -> Catalog:
        """Build DOP and DSM catalog from DOM/LAS feeds and DOP overview queries."""
        # Use LAS feed for spatial coverage mask (all 17,127 tiles)
        las_tiles = self._fetch_las_tiles()
        # Use DOM feed for actual DSM tiles (1m resolution DOM1)
        dom_tiles = self._fetch_dom_tiles()

        image_tiles: dict[tuple[int, int], dict[int, dict]] = {}
        dsm_tiles: dict[tuple[int, int], dict[int, dict]] = {}

        if not las_tiles:
            return Catalog(image_tiles=image_tiles, dsm_tiles=dsm_tiles)

        # Populate DSM tiles from DOM feed (1m resolution)
        for coords, year, url in dom_tiles.values():
            dsm_tiles.setdefault(coords, {})[year] = self._tile_info(
                url,
                acquisition_date=None,
                source_kind="direct",
                source_age="current",
            )

        # Build chunked bbox queries to reduce API calls (~200x faster)
        # API limits to 200 objects per request, so we query 3.9km chunks
        las_coords = set(las_tiles.keys())
        chunks = self._compute_bbox_chunks(las_coords)

        if not self.quiet:
            # Server rate: ~2 req/s, querying 1 bbox per ~3x3 LAS tiles
            est_seconds = len(chunks) // 2
            if est_seconds < 120:
                print(f"  TH: Querying DOP for {len(las_tiles)} tiles (~{est_seconds}s)...", flush=True)
            else:
                est_minutes = est_seconds // 60
                print(f"  TH: Querying DOP for {len(las_tiles)} tiles (~{est_minutes} min, cached after first build)...", flush=True)

        start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=self.QUERY_WORKERS) as executor:
            futures = {executor.submit(self._fetch_dop_chunk, bbox): bbox for bbox in chunks}
            for i, fut in enumerate(as_completed(futures), 1):
                if InterruptManager.get().is_set():
                    break
                results = fut.result()
                for coords, year, url, acq_date in results:
                    # Only include tiles that have LAS coverage
                    if coords in las_coords:
                        image_tiles.setdefault(coords, {})[year] = self._tile_info(
                            url,
                            acquisition_date=acq_date,
                            source_kind="direct",
                            source_age="current",
                        )
                if not self.quiet and i % 50 == 0:
                    elapsed = time.perf_counter() - start
                    rate = i / elapsed if elapsed > 0 else 1
                    eta = (len(chunks) - i) / rate if rate > 0 else 0
                    print(f"    {i}/{len(chunks)} ({elapsed:.0f}s, ~{eta:.0f}s remaining)", flush=True)

        # Verify DOP coverage matches expected pattern (40 known gaps)
        missing_dop = las_coords - set(image_tiles.keys())

        # Check for unexpected changes in coverage
        unexpected_missing = missing_dop - self.KNOWN_DOP_GAPS
        newly_covered = self.KNOWN_DOP_GAPS - missing_dop

        if unexpected_missing:
            # NEW gaps found - possible regression or data source change
            sorted_unexpected = sorted(unexpected_missing)
            print(f"  ❌ ERROR: {len(unexpected_missing)} UNEXPECTED missing tiles (possible regression):", flush=True)
            for coord in sorted_unexpected[:20]:  # Show first 20
                print(f"      {coord}", flush=True)
            if len(sorted_unexpected) > 20:
                print(f"      ... and {len(sorted_unexpected) - 20} more", flush=True)
            print(f"  Expected coverage: 99.77% (17,087/17,127 with 40 known gaps)", flush=True)
            print(f"  Actual coverage: {len(image_tiles)}/{len(las_coords)} tiles", flush=True)
            raise RuntimeError(
                f"TH downloader coverage regression: {len(unexpected_missing)} unexpected missing tiles. "
                f"Expected 40 known gaps, found {len(missing_dop)} total gaps."
            )

        if newly_covered:
            # Known gaps are now covered - DOP data was added!
            if not self.quiet:
                print(f"  ✓ Good news: {len(newly_covered)} previously missing tiles now have DOP coverage!", flush=True)
                for coord in sorted(newly_covered)[:10]:
                    print(f"      {coord}", flush=True)
                if len(newly_covered) > 10:
                    print(f"      ... and {len(newly_covered) - 10} more", flush=True)
                print(f"  Consider updating KNOWN_DOP_GAPS constant (remove these {len(newly_covered)} tiles)", flush=True)

        if missing_dop == self.KNOWN_DOP_GAPS:
            # Perfect match - expected coverage
            if not self.quiet:
                print(f"  ✓ Coverage: 99.77% (17,087/17,127 tiles, 40 known gaps at DOP boundaries)", flush=True)

        return Catalog(image_tiles=image_tiles, dsm_tiles=dsm_tiles)

    def _fetch_las_tiles(self) -> dict[tuple[int, int], tuple[tuple[int, int], int, str]]:
        """Parse LAS Atom dataset feed and return tile info keyed by coords.

        Returns:
            Dict mapping (grid_x, grid_y) to (coords, year, url) tuples.
            Year is extracted from the year-range in the URL (uses end year).
        """
        tiles: dict[tuple[int, int], tuple[tuple[int, int], int, str]] = {}

        if not self.quiet:
            print("  TH: Fetching LAS feed...", flush=True)
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

    def _fetch_dom_tiles(self) -> dict[tuple[int, int], tuple[tuple[int, int], int, str]]:
        """Parse DOM Atom dataset feed and return tile info keyed by coords.

        Returns:
            Dict mapping (grid_x, grid_y) to (coords, year, url) tuples.
            Year is extracted from the year-range in the URL (uses end year).
            Uses both 2014-2019 and 2020-2025 vintages for full coverage:
            - 2014-2019: Full coverage (17,127 tiles), 1m resolution DOM1
            - 2020-2025: Reduced coverage (14,869 tiles), 1m resolution DOM1
            Where both exist, 2020-2025 takes priority (newer data).
        """
        tiles: dict[tuple[int, int], tuple[tuple[int, int], int, str]] = {}

        if not self.quiet:
            print("  TH: Fetching DOM feed...", flush=True)
        root = fetch_xml_feed(self._session, self.DOM_DATASET_URL, timeout=FEED_TIMEOUT)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        # Collect tiles from both vintages
        tiles_2014_2019 = []
        tiles_2020_2025 = []

        for link in root.findall(".//atom:link", ns):
            if InterruptManager.get().is_set():
                break
            if link.attrib.get("rel") != "section":
                continue
            href = link.attrib.get("href", "")
            match = self.DOM_LINK_PATTERN.search(href)
            if not match:
                continue
            year_range = match.group(1)  # e.g., "2020-2025"
            year = int(year_range.split("-")[1])  # Use end year
            coords = (int(match.group(2)), int(match.group(3)))

            if year_range == "2014-2019":
                tiles_2014_2019.append((coords, year, href))
            elif year_range == "2020-2025":
                tiles_2020_2025.append((coords, year, href))

        # First, populate with 2014-2019 vintage (full coverage)
        for coords, year, href in tiles_2014_2019:
            tiles[coords] = (coords, year, href)

        # Then override with 2020-2025 where available (newer data)
        for coords, year, href in tiles_2020_2025:
            tiles[coords] = (coords, year, href)

        return tiles

    def _compute_bbox_chunks(
        self, coords: set[tuple[int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """Create grid of centered bbox queries spaced GRID_SPACING_KM apart.

        Generates regular grid covering all LAS tiles, with BBOX_SIZE_KM boxes
        centered on grid points spaced GRID_SPACING_KM apart. Skips grid cells with no LAS tiles.

        Example: Grid cell starting at 564km with 4km spacing covers 564-568km.
        Center is at 566km, and 3.9km bbox extends 1.95km in each direction
        (from 564.05km to 567.95km, leaving small 50m gaps at edges).

        Returns list of (minx, miny, maxx, maxy) in meters.
        """
        if not coords:
            return []

        # Find bounds of LAS coverage
        xs = [x for x, y in coords]
        ys = [y for x, y in coords]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        spacing_km = self.GRID_SPACING_KM
        bbox_size_m = int(self.BBOX_SIZE_KM * METERS_PER_KM)
        half_box = bbox_size_m // 2

        # Create regular grid of query points spaced GRID_SPACING_KM apart
        # Start from first grid point that covers min coords
        start_x = (min_x // spacing_km) * spacing_km
        start_y = (min_y // spacing_km) * spacing_km

        chunks = []
        y_grid = start_y
        while y_grid <= max_y:
            x_grid = start_x
            while x_grid <= max_x:
                # Check if this grid cell (spacing_km × spacing_km) has any LAS tiles
                has_las = any(
                    x_grid <= x < x_grid + spacing_km and
                    y_grid <= y < y_grid + spacing_km
                    for x, y in coords
                )

                if not has_las:
                    x_grid += spacing_km
                    continue

                # Center of the grid cell (e.g., cell starting at 564km with 4km spacing → center at 566km)
                center_x = x_grid * METERS_PER_KM + (spacing_km * METERS_PER_KM // 2)
                center_y = y_grid * METERS_PER_KM + (spacing_km * METERS_PER_KM // 2)

                # Create bbox (BBOX_SIZE_KM) centered on this point
                minx = center_x - half_box
                miny = center_y - half_box
                maxx = center_x + half_box
                maxy = center_y + half_box
                chunks.append((minx, miny, maxx, maxy))

                x_grid += spacing_km
            y_grid += spacing_km

        return chunks

    def _get_thread_session(self) -> requests.Session:
        """Get thread-local session for parallel requests."""
        if not hasattr(self._thread_local, "session"):
            self._thread_local.session = requests.Session()
        return self._thread_local.session

    def _fetch_dop_chunk(
        self, bbox: tuple[int, int, int, int]
    ) -> list[tuple[tuple[int, int], int, str, str | None]]:
        """Query DOP overview for a bbox chunk and return all matching items.

        Multi-km tiles (e.g., 2km×2km) are expanded into multiple entries,
        one for each 1km grid cell they cover.
        """
        if InterruptManager.get().is_set():
            return []

        minx, miny, maxx, maxy = bbox
        params = {
            "crs": "EPSG:25832",
            "bbox[]": [minx, miny, maxx, maxy],
            "type[]": "op",
        }

        # Retry with exponential backoff to handle rate limiting
        session = self._get_thread_session()
        for attempt in range(MAX_RETRIES):
            if InterruptManager.get().is_set():
                return []

            try:
                resp = session.get(self.DOP_OVERVIEW_URL, params=params, timeout=FEED_TIMEOUT)
                resp.raise_for_status()
                payload = resp.json()

                if not payload.get("success"):
                    return []

                features = payload.get("result", {}).get("features", [])
                results: list[tuple[tuple[int, int], int, str, str | None]] = []
                for feature in features:
                    items = self._parse_dop_feature_with_expansion(feature)
                    if items:
                        results.extend(items)
                return results
            except (requests.Timeout, requests.ConnectionError) as e:
                # Network errors - retry with backoff
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    time.sleep(min(wait, 10))  # Cap at 10s for bbox queries
                    continue
                # Last attempt failed - log and return empty
                if not self.quiet:
                    print(f"  ⚠ DOP query failed for bbox {bbox}: {e}", flush=True)
                return []
            except Exception as e:
                # Other errors (JSON parse, HTTP error) - don't retry
                if not self.quiet:
                    print(f"  ⚠ DOP query error for bbox {bbox}: {e}", flush=True)
                return []

        return []

    def _parse_dop_feature_with_expansion(
        self, feature: dict
    ) -> list[tuple[tuple[int, int], int, str, str | None]]:
        """Parse DOP feature and expand multi-km tiles into all covered 1km grid cells.

        Returns a list of (coords, year, url, acq_date) tuples, one for each 1km
        grid cell covered by this tile. A 2km×2km tile returns 4 entries.
        """
        props = feature.get("properties", {})
        if props.get("type") != "op":
            return []

        date_str = props.get("datum") or ""
        year = int(date_str[:4]) if date_str[:4].isdigit() else None
        if year is None:
            return []

        # NOTE: For TH, we include ALL years in catalog to match LAS coverage.
        # LAS tiles are not year-filtered, so DOP shouldn't be either to ensure 1:1 coverage.
        # Year filtering is applied at download/conversion time, not catalog build time.

        gid = props.get("gid")
        bildflugnr = props.get("bildflugnr")
        bildnr = props.get("bildnr")
        if not (gid and bildflugnr and bildnr):
            return []

        url = f"{self.DOP_DOWNLOAD_URL}?type=op&id={gid}&log={bildflugnr}-{bildnr}"
        acq_date = date_str[:10] if date_str else None

        # Get tile geometry to determine all 1km grid cells it covers
        geom = feature.get("geometry", {})
        coords_list = geom.get("coordinates", [])
        if not coords_list or not coords_list[0]:
            return []

        ring = coords_list[0]
        minx_m = min(pt[0] for pt in ring)
        miny_m = min(pt[1] for pt in ring)
        maxx_m = max(pt[0] for pt in ring)
        maxy_m = max(pt[1] for pt in ring)

        # Convert to km grid coordinates
        min_x_km = int(minx_m // METERS_PER_KM)
        min_y_km = int(miny_m // METERS_PER_KM)
        max_x_km = int(maxx_m // METERS_PER_KM)
        max_y_km = int(maxy_m // METERS_PER_KM)

        # Generate entries for all 1km grid cells covered by this tile
        results = []
        for y_km in range(min_y_km, max_y_km):
            for x_km in range(min_x_km, max_x_km):
                results.append(((x_km, y_km), year, url, acq_date))

        return results
