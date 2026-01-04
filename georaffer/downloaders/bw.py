"""BW (Baden-Württemberg) tile downloader for DOP20 RGB and DOM1 raster tiles.

Catalog source: WFS at owsproxy.lgl-bw.de
Download source: https://opengeodata.lgl-bw.de/data/

Grid system:
- Catalog tiles: 1km x 1km (WFS)
- Download tiles: 2km x 2km ZIPs (odd easting, even northing)
- CRS: EPSG:25832 (UTM Zone 32N)

Each ZIP contains multiple 1km GeoTIFF sub-tiles.

Historic imagery: Only 2010+ is supported because:
- DOP20 (20cm resolution) was first available statewide in 2010 (was 25cm before)
- Pre-2005 imagery is grayscale only
- Pre-2018 data was in Gauss-Krüger CRS (now reprojected to UTM32 with artifacts)

WMS decade services exist for 1960-2009 but are lower resolution, often B&W, and reprojected.
"""

import time
import xml.etree.ElementTree as ET
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
    WMS_QUERY_WORKERS,
)
from georaffer.downloaders.base import Catalog, RegionDownloader
from georaffer.downloaders.wms import WMSImagerySource

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

    # Historic WMS decade services with per-year layers
    WMS_HIST_BASE_PREFIX: ClassVar[str] = (
        "https://owsproxy.lgl-bw.de/owsproxy/ows/WMS_LGL-BW_HIST_DOP_"
    )
    # Only 2010+ supported (see module docstring for rationale)
    HISTORIC_DECADES: ClassVar[list[str]] = [
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
        self._wms_years_cache: dict[str, list[tuple[str, list[int]]]] = {}

        # Parse imagery_from for historic support (only 2010+ supported, see docstring)
        if imagery_from is None:
            self._from_year = None
            self._to_year = None
        else:
            from_year, to_year = imagery_from
            if from_year < 2010:
                raise ValueError(
                    f"Year {from_year} not supported. BW historic imagery requires 2010+ "
                    f"(DOP20 was introduced in 2010; earlier data has lower resolution and "
                    f"was in Gauss-Krüger CRS)."
                )
            self._from_year = from_year
            self._to_year = to_year

    def utm_to_grid_coords(
        self, utm_x: float, utm_y: float
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """BW uses a 2km download grid aligned to odd E / even N."""
        grid_x = int(utm_x // METERS_PER_KM)
        grid_y = int(utm_y // METERS_PER_KM)
        download_coords = self._wfs_to_download_coords(grid_x, grid_y)
        return download_coords, download_coords

    def _filename_from_url(self, url: str) -> str:
        """Return filename from URL, validating it's a ZIP archive or WMS GetMap."""
        lowered = url.lower()
        if "service=wms" in lowered and "request=getmap" in lowered:
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            bbox = params.get("BBOX", params.get("bbox", [""]))[0].split(",")
            if len(bbox) < 4:
                raise ValueError(f"BW WMS URL missing BBOX: {url}")
            minx_m = float(bbox[0])
            miny_m = float(bbox[1])
            maxx_m = float(bbox[2])
            maxy_m = float(bbox[3])
            minx = int(minx_m / METERS_PER_KM)
            miny = int(miny_m / METERS_PER_KM)
            tile_km_x = (maxx_m - minx_m) / METERS_PER_KM
            tile_km_y = (maxy_m - miny_m) / METERS_PER_KM
            tile_km = int(round(max(tile_km_x, tile_km_y)))
            if tile_km <= 0:
                tile_km = 1
            layer = params.get("LAYERS", params.get("layers", [""]))[0]
            year = layer.strip() if layer else "0000"
            fmt = params.get("FORMAT", params.get("format", [""]))[0].lower()
            ext = ".png" if "png" in fmt else ".jpg"
            return f"dop20rgb_32_{minx}_{miny}_{tile_km}_bw_{year}{ext}"

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
        Stores 1km tiles with URLs pointing at 2km ZIP downloads.
        """
        # DOP tiles
        if not self.quiet:
            print("  Loading DOP tiles from WFS...")
        image_tiles: dict[tuple[int, int], dict[int, dict]] = {}

        dop_dates: dict[tuple[int, int], date] = {}
        for grid_x, grid_y, flight_date in self._fetch_wfs_tiles(
            self.DOP_WFS_URL,
            self.DOP_FEATURE,
            "dop_kachel",
            "befliegungsdatum",
        ):
            if flight_date is None:
                raise ValueError(
                    "BW DOP WFS tile "
                    f"{grid_x},{grid_y} missing befliegungsdatum; "
                    "cannot proceed without flight dates."
                )
            dop_dates[(grid_x, grid_y)] = flight_date
            download_coords = self._wfs_to_download_coords(grid_x, grid_y)
            if download_coords:
                year = flight_date.year
                url = self._build_dop_url(download_coords[0], download_coords[1])
                image_tiles.setdefault((grid_x, grid_y), {})[year] = {
                    "url": url,
                    "acquisition_date": flight_date.isoformat(),
                }

        if not self.quiet:
            print(f"    {len(image_tiles)} download tiles (from 1km WFS catalog)")

        # Historic tiles via WMS GetMap.
        # Catalog always uses 2010+ regardless of user's imagery_from because:
        # 1. Catalog is cached and shared across runs with different parameters
        # 2. DOP20 (20cm) was introduced in 2010; earlier decades have lower resolution
        # 3. Pre-2018 data was in Gauss-Krüger CRS (reprojection artifacts)
        from_year = 2010
        to_year = date.today().year
        if from_year <= to_year:
            historic_layers = self._historic_layers(from_year, to_year)
            bw_workers = WMS_QUERY_WORKERS * 2
            if historic_layers and not self.quiet:
                print(
                    f"  Querying BW historic WMS for {len(image_tiles)} tiles × {len(historic_layers)} layers "
                    f"({bw_workers} workers)..."
                )
            checked = 0
            found = 0
            started = time.perf_counter()

            tiles_list = list(image_tiles.keys())
            with ThreadPoolExecutor(max_workers=bw_workers) as executor:
                futures = {}
                wms_sources: dict[str, WMSImagerySource] = {}

                def wms_for(base_url: str) -> WMSImagerySource:
                    if base_url not in wms_sources:
                        wms_sources[base_url] = WMSImagerySource(
                            base_url=base_url,
                            rgb_layer_pattern="{year}",
                            info_layer_pattern="{year}",
                            tile_size_m=BW_GRID_SIZE,
                            resolution_m=0.2,
                            image_format="image/jpeg",
                            coverage_format="image/png",
                            crs="EPSG:25832",
                            coverage_mode="getmap",
                            preview_size=16,
                            session=self._session,
                        )
                    return wms_sources[base_url]

                for base_url, layer_name, years in historic_layers:
                    for grid_x, grid_y in tiles_list:
                        if any(y in image_tiles.get((grid_x, grid_y), {}) for y in years):
                            continue
                        wms = wms_for(base_url)
                        futures[
                            executor.submit(wms.check_coverage, layer_name, grid_x, grid_y)
                        ] = (base_url, layer_name, years, grid_x, grid_y)

                total = len(futures)
                for fut in as_completed(futures):
                    checked += 1
                    base_url, layer_name, years, grid_x, grid_y = futures[fut]
                    try:
                        has_coverage = fut.result()
                    except Exception:
                        has_coverage = False
                    if has_coverage:
                        url = wms_sources[base_url].get_tile_url(
                            layer_name, grid_x, grid_y
                        )
                        for year in years:
                            if year in image_tiles.get((grid_x, grid_y), {}):
                                continue
                            image_tiles.setdefault((grid_x, grid_y), {})[year] = {
                                "url": url,
                                "acquisition_date": None,
                            }
                        found += 1

                    if not self.quiet and (checked % 500 == 0 or checked == total):
                        elapsed = time.perf_counter() - started
                        rate = checked / elapsed if elapsed > 0 else 0
                        print(
                            f"\r  {checked}/{total} checked, "
                            f"{found} historic found ({rate:.1f} req/s)",
                            end="",
                            flush=True,
                        )

                if not self.quiet and total:
                    print()

        # DOM tiles (uses DGM WFS - same laser flights provide both DGM and DOM)
        # If fortfuehrungsdatum is missing, fall back to corresponding DOP date.
        if not self.quiet:
            print("  Loading DOM tiles from WFS...")
        dsm_tiles: dict[tuple[int, int], dict[int, dict]] = {}

        skipped_dom_missing_date = 0
        for grid_x, grid_y, flight_date in self._fetch_wfs_tiles(
            self.DOM_WFS_URL,
            self.DOM_FEATURE,
            "dgm_kachel",
            "fortfuehrungsdatum",
        ):
            if flight_date is None:
                # Fall back to DOP date for this tile
                dop_date = dop_dates.get((grid_x, grid_y))
                if dop_date is None:
                    skipped_dom_missing_date += 1
                    continue
                flight_date = dop_date
            download_coords = self._wfs_to_download_coords(grid_x, grid_y)
            if download_coords:
                year = flight_date.year
                url = self._build_dom_url(download_coords[0], download_coords[1])
                dsm_tiles.setdefault((grid_x, grid_y), {})[year] = {
                    "url": url,
                    "acquisition_date": flight_date.isoformat(),
                }

        if not self.quiet:
            print(f"    {len(dsm_tiles)} download tiles")
            if skipped_dom_missing_date:
                print(
                    "    (skipped DOM tiles missing fortfuehrungsdatum and DOP fallback: "
                    f"{skipped_dom_missing_date})"
                )

        missing_dom_coords = set(image_tiles) - set(dsm_tiles)
        if missing_dom_coords:
            for coords in missing_dom_coords:
                image_tiles.pop(coords, None)
            if not self.quiet:
                print(
                    "    (dropped DOP tiles without DOM counterpart: "
                    f"{len(missing_dom_coords)})"
                )

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

    def _historic_layers(
        self, from_year: int, to_year: int
    ) -> list[tuple[str, str, list[int]]]:
        """Return (base_url, layer_name, years) from BW historic WMS services."""
        import re

        results: list[tuple[str, str, list[int]]] = []
        base_urls = [
            f"{self.WMS_HIST_BASE_PREFIX}{decade}" for decade in self.HISTORIC_DECADES
        ]
        for base_url in base_urls:
            if base_url in self._wms_years_cache:
                all_layers = self._wms_years_cache[base_url]
            else:
                params = {"SERVICE": "WMS", "REQUEST": "GetCapabilities"}
                resp = self._session.get(base_url, params=params, timeout=FEED_TIMEOUT)
                resp.raise_for_status()

                ns = {"wms": "http://www.opengis.net/wms"}
                root = ET.fromstring(resp.text)
                all_layers: list[tuple[str, list[int]]] = []
                for layer in root.findall(".//wms:Layer/wms:Layer", ns):
                    name_el = layer.find("wms:Name", ns)
                    if name_el is None or not name_el.text:
                        continue
                    name = name_el.text.strip()
                    if not re.fullmatch(r"\d{4}", name):
                        continue
                    year = int(name)
                    all_layers.append((name, [year]))

                self._wms_years_cache[base_url] = all_layers

            for name, years in all_layers:
                filtered_years = [y for y in years if from_year <= y <= to_year]
                if filtered_years:
                    results.append((base_url, name, filtered_years))

        return results

    def _fetch_wfs_tiles(
        self,
        wfs_url: str,
        feature_type: str,
        id_field: str,
        date_field: str,
    ) -> list[tuple[int, int, date | None]]:
        """Fetch all tiles from a WFS endpoint using parallel pagination.

        Returns list of (grid_x_km, grid_y_km, flight_date) tuples.
        Date may be None if the WFS doesn't provide it for a tile.
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
            raise ValueError(
                "BW WFS hits response missing numeric numberMatched; "
                f"cannot paginate {wfs_url}"
            )
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
                page_resp.json().get("features", []),
                id_field,
                date_field,
            )

        results: list[tuple[int, int, date | None]] = []
        with ThreadPoolExecutor(max_workers=WFS_PARALLEL_WORKERS) as executor:
            futures = {executor.submit(fetch_page, o): o for o in offsets}
            for fut in as_completed(futures):
                results.extend(fut.result())

        return results

    def _parse_wfs_features(
        self,
        features: list[dict],
        id_field: str,
        date_field: str,
    ) -> list[tuple[int, int, date | None]]:
        """Parse WFS GeoJSON features into (grid_x, grid_y, date) tuples.

        The tile ID format is like "325955426" where:
        - 32 = UTM zone
        - 595 = easting in km
        - 5426 = northing in km

        Returns None for date if the field is missing or empty.
        """
        results = []
        for f in features:
            props = f.get("properties", {})
            tile_id = str(props.get(id_field, ""))
            date_str = props.get(date_field)

            if not tile_id:
                raise ValueError(f"BW WFS feature missing {id_field}.")
            tile_id = tile_id.strip()
            if not tile_id.isdigit() or len(tile_id) != 9:
                raise ValueError(
                    f"BW WFS tile id '{tile_id}' for {id_field} is invalid; "
                    "expected 9 digits."
                )

            # Parse tile ID: zone(2) + easting(3) + northing(4)
            zone = int(tile_id[:2])
            if zone != self.UTM_ZONE:
                raise ValueError(
                    f"BW WFS tile id '{tile_id}' uses UTM zone {zone}; "
                    f"expected {self.UTM_ZONE}."
                )
            grid_x = int(tile_id[2:5])
            grid_y = int(tile_id[5:9])

            if not date_str:
                results.append((grid_x, grid_y, None))
                continue

            try:
                flight_date = date.fromisoformat(str(date_str)[:10])
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"BW WFS tile id '{tile_id}' has invalid {date_field}: {date_str}."
                ) from exc

            results.append((grid_x, grid_y, flight_date))

        return results
