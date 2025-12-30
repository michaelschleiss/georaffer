"""BW (Baden-Württemberg) tile downloader for DOP20 RGB and DOM1 raster tiles.

Catalog source: WFS at owsproxy.lgl-bw.de
Download source: https://opengeodata.lgl-bw.de/data/

Grid system:
- 2km x 2km download tiles (like RLP)
- Easting: odd km values only (389, 391, ..., 609)
- Northing: even km values only (5266, 5268, ..., 5514)
- CRS: EPSG:25832 (UTM Zone 32N)

Each ZIP contains 4 x 1km GeoTIFF sub-tiles.

Historic imagery available via WMS with decade-based layers:
- WMS_LGL-BW_HIST_DOP_1960-1969, ..., WMS_LGL-BW_HIST_DOP_2020-2029
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import ClassVar

import requests

from georaffer.config import (
    BW_GRID_SIZE,
    CATALOG_CACHE_DIR,
    FEED_TIMEOUT,
    METERS_PER_KM,
    Region,
)
from georaffer.downloaders.base import Catalog, RegionDownloader

# WFS pagination settings
WFS_PAGE_SIZE = 1000
WFS_PARALLEL_WORKERS = 8


class BWDownloader(RegionDownloader):
    """BW (Baden-Württemberg) downloader for DOP20 RGB + DOM1 raster tiles (GeoTIFF in ZIP)."""

    # WFS endpoints for tile catalogs
    DOP_WFS_URL: ClassVar[str] = (
        "https://owsproxy.lgl-bw.de/owsproxy/wfs/"
        "WFS_LGL-BW_ATKIS_DOP_20_Bildflugkacheln_Aktualitaet"
    )
    DOM_WFS_URL: ClassVar[str] = (
        "https://owsproxy.lgl-bw.de/owsproxy/wfs/WFS_LGL-BW_ATKIS_DGM_Aktualitaet"
    )
    DOP_FEATURE: ClassVar[str] = "verm:v_dop_20_bildflugkacheln"
    DOM_FEATURE: ClassVar[str] = "verm:v_dgm_kacheln"

    # Download base URLs
    DOP_BASE_URL: ClassVar[str] = "https://opengeodata.lgl-bw.de/data/dop20/"
    DOM_BASE_URL: ClassVar[str] = "https://opengeodata.lgl-bw.de/data/dom1/"

    # Historic WMS (decade-based layers)
    WMS_BASE_URL: ClassVar[str] = "https://owsproxy.lgl-bw.de/owsproxy/wms/"
    HISTORIC_DECADES: ClassVar[list[str]] = [
        "1960-1969",
        "1970-1979",
        "1980-1989",
        "1990-1999",
        "2000-2009",
        "2010-2019",
        "2020-2029",
    ]

    UTM_ZONE: ClassVar[int] = 32

    def __init__(
        self,
        output_dir: str,
        imagery_from: tuple[int, int | None] | None = None,
        session: requests.Session | None = None,
        quiet: bool = False,
    ):
        super().__init__(
            Region.BW, output_dir, imagery_from=imagery_from, session=session, quiet=quiet
        )
        self._cache_path = CATALOG_CACHE_DIR / "bw_catalog.json"

        # Parse imagery_from for historic support
        if imagery_from is None:
            self._from_year = None
            self._to_year = None
        else:
            from_year, to_year = imagery_from
            # BW historic goes back to 1960s, but quality varies
            if from_year < 1960:
                raise ValueError(
                    f"Year {from_year} not supported. BW historic imagery starts from 1960."
                )
            self._from_year = from_year
            self._to_year = to_year

    def utm_to_grid_coords(
        self, utm_x: float, utm_y: float
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """BW uses 2km grid with odd easting, even northing alignment.

        The download grid uses:
        - Easting: odd km values (389, 391, 393, ...)
        - Northing: even km values (5266, 5268, 5270, ...)

        Each 2km tile covers E to E+2km and N to N+2km.
        """
        # Get km position
        km_x = int(utm_x // METERS_PER_KM)
        km_y = int(utm_y // METERS_PER_KM)

        # Align to BW 2km grid: odd easting, even northing
        # Round down to nearest odd for E
        if km_x % 2 == 0:
            km_x -= 1
        # Round down to nearest even for N
        if km_y % 2 == 1:
            km_y -= 1

        return (km_x, km_y), (km_x, km_y)

    def _filename_from_url(self, url: str) -> str:
        """Return filename from URL, validating it's a ZIP archive."""
        name = Path(url).name
        if not name.lower().endswith(".zip"):
            raise ValueError(f"BW downloads must be ZIP archives (got {name}).")
        return name

    def dsm_filename_from_url(self, url: str) -> str:
        return self._filename_from_url(url)

    def image_filename_from_url(self, url: str) -> str:
        return self._filename_from_url(url)

    def _load_catalog(self) -> Catalog:
        """Load BW catalog from WFS endpoints.

        Uses parallel pagination for DOP and DOM tile catalogs.
        Maps 1km WFS tiles to 2km download grid.
        """
        # DOP tiles
        if not self.quiet:
            print("  Loading DOP tiles from WFS...")
        image_tiles: dict[tuple[int, int], dict[int, dict]] = {}

        for grid_x, grid_y, flight_date in self._fetch_wfs_tiles(
            self.DOP_WFS_URL, self.DOP_FEATURE, "dop_kachel", "befliegungsdatum"
        ):
            if flight_date is None:
                continue  # WFS feature missing befliegungsdatum - skip
            # Map 1km WFS coords to 2km download coords
            download_coords = self._wfs_to_download_coords(grid_x, grid_y)
            if download_coords:
                year = flight_date.year
                url = self._build_dop_url(download_coords[0], download_coords[1])
                image_tiles.setdefault(download_coords, {})[year] = {
                    "url": url,
                    "acquisition_date": flight_date.isoformat(),
                }

        if not self.quiet:
            print(f"    {len(image_tiles)} download tiles (from 1km WFS catalog)")

        # DOM tiles (uses same WFS as DGM - same laser flights)
        # Note: Most BW DOM tiles lack dates in WFS (fortfuehrungsdatum missing).
        # We use current year as placeholder when date unavailable.
        if not self.quiet:
            print("  Loading DOM tiles from WFS...")
        dsm_tiles: dict[tuple[int, int], dict[int, dict]] = {}
        current_year = date.today().year

        for grid_x, grid_y, flight_date in self._fetch_wfs_tiles(
            self.DOM_WFS_URL, self.DOM_FEATURE, "dgm_kachel", "fortfuehrungsdatum"
        ):
            # Map 1km WFS coords to 2km download coords
            download_coords = self._wfs_to_download_coords(grid_x, grid_y)
            if download_coords:
                year = flight_date.year if flight_date else current_year
                if year not in dsm_tiles.get(download_coords, {}):
                    url = self._build_dom_url(download_coords[0], download_coords[1])
                    dsm_tiles.setdefault(download_coords, {})[year] = {
                        "url": url,
                        "acquisition_date": flight_date.isoformat() if flight_date else None,
                    }

        if not self.quiet:
            print(f"    {len(dsm_tiles)} download tiles")

        return Catalog(image_tiles=image_tiles, dsm_tiles=dsm_tiles)

    def _wfs_to_download_coords(
        self, wfs_x: int, wfs_y: int
    ) -> tuple[int, int] | None:
        """Map 1km WFS tile coords to 2km download grid coords.

        WFS returns 1km tiles. Downloads are 2km tiles aligned to odd E, even N.
        """
        # Round down to nearest odd for E, nearest even for N
        download_x = wfs_x if wfs_x % 2 == 1 else wfs_x - 1
        download_y = wfs_y if wfs_y % 2 == 0 else wfs_y - 1
        return (download_x, download_y)

    def _build_dop_url(self, grid_x: int, grid_y: int) -> str:
        """Build DOP download URL from grid coordinates."""
        return f"{self.DOP_BASE_URL}dop20rgb_32_{grid_x}_{grid_y}_2_bw.zip"

    def _build_dom_url(self, grid_x: int, grid_y: int) -> str:
        """Build DOM download URL from grid coordinates."""
        return f"{self.DOM_BASE_URL}dom1_32_{grid_x}_{grid_y}_2_bw.zip"

    def _fetch_wfs_tiles(
        self, wfs_url: str, feature_type: str, id_field: str, date_field: str
    ) -> list[tuple[int, int, date | None]]:
        """Fetch all tiles from a WFS endpoint using parallel pagination.

        Returns list of (grid_x_km, grid_y_km, flight_date) tuples.
        """
        # Get total count
        params = {
            "SERVICE": "WFS",
            "VERSION": "2.0.0",
            "REQUEST": "GetFeature",
            "TYPENAMES": feature_type,
            "RESULTTYPE": "hits",
        }
        resp = self._session.get(wfs_url, params=params, timeout=FEED_TIMEOUT)
        resp.raise_for_status()

        # Parse numberMatched from XML response
        import re

        match = re.search(r'numberMatched="(\d+)"', resp.text)
        if not match:
            return []
        total = int(match.group(1))

        if total == 0:
            return []

        # Fetch all pages in parallel
        offsets = range(0, total, WFS_PAGE_SIZE)

        def fetch_page(offset: int) -> list[tuple[int, int, date | None]]:
            page_params = {
                "SERVICE": "WFS",
                "VERSION": "2.0.0",
                "REQUEST": "GetFeature",
                "TYPENAMES": feature_type,
                "COUNT": WFS_PAGE_SIZE,
                "STARTINDEX": offset,
                "OUTPUTFORMAT": "application/json",
            }
            page_resp = self._session.get(wfs_url, params=page_params, timeout=FEED_TIMEOUT)
            page_resp.raise_for_status()
            return self._parse_wfs_features(
                page_resp.json().get("features", []), id_field, date_field
            )

        results: list[tuple[int, int, date | None]] = []
        with ThreadPoolExecutor(max_workers=WFS_PARALLEL_WORKERS) as executor:
            futures = {executor.submit(fetch_page, o): o for o in offsets}
            for fut in as_completed(futures):
                results.extend(fut.result())

        return results

    def _parse_wfs_features(
        self, features: list[dict], id_field: str, date_field: str
    ) -> list[tuple[int, int, date | None]]:
        """Parse WFS GeoJSON features into (grid_x, grid_y, date) tuples.

        The tile ID format is like "325955426" where:
        - 32 = UTM zone
        - 595 = easting in km
        - 5426 = northing in km
        """
        results = []
        for f in features:
            props = f.get("properties", {})
            tile_id = str(props.get(id_field, ""))
            date_str = props.get(date_field)

            if not tile_id or len(tile_id) < 9:
                continue

            try:
                # Parse tile ID: zone(2) + easting(3) + northing(4)
                zone = int(tile_id[:2])
                if zone != self.UTM_ZONE:
                    continue
                grid_x = int(tile_id[2:5])
                grid_y = int(tile_id[5:9])

                # Parse date if available
                flight_date = None
                if date_str:
                    try:
                        flight_date = date.fromisoformat(date_str[:10])
                    except (ValueError, TypeError):
                        pass

                results.append((grid_x, grid_y, flight_date))
            except (ValueError, IndexError):
                continue

        return results
