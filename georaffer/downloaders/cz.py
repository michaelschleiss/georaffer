"""CZ (Czech Republic) downloader using official CUZK INSPIRE/open data feeds."""

from __future__ import annotations

import re
import shutil
import zipfile
from contextlib import suppress
from pathlib import Path
from typing import ClassVar

from georaffer.config import CATALOG_CACHE_DIR, CZ_GRID_SIZE, METERS_PER_KM, MIN_FILE_SIZE, Region
from georaffer.downloaders.base import Catalog, RegionDownloader
from georaffer.downloaders.feeds import fetch_xml_feed


class CZDownloader(RegionDownloader):
    """CZ orthophoto + DSM downloader.

    Uses:
    - OI (INSPIRE orthophoto) for imagery
    - DMPOK TIFF where available for modern DSM
    - DMP1G LAZ as nationwide fallback once the irregular LAZ backend is available

    CUZK ATOM tile filenames do not expose stable per-tile acquisition years in the
    top-level feeds, so we use dataset end-year labels for output naming:
    - OI: 2024 (dataset extent 2023-2024)
    - DMPOK: 2025 (dataset extent 2024-2025)

    DMP1G remains an irregular/TIN-derived fallback source. The worker layer is
    responsible for routing it through the irregular LAZ backend instead of the
    regular-grid LAZ converter.
    """

    OI_FEED_URL: ClassVar[str] = "https://atom.cuzk.gov.cz/OI/OI.xml"
    DMPOK_FEED_URL: ClassVar[str] = "https://atom.cuzk.gov.cz/DMPOK-ETRS89-TIFF/DMPOK-ETRS89-TIFF.xml"
    DMP1G_FEED_URL: ClassVar[str] = "https://atom.cuzk.cz/DMP1G-ETRS89/DMP1G-ETRS89.xml"
    OI_DOWNLOAD_BASE: ClassVar[str] = "https://openzu.cuzk.gov.cz/opendata/OI"
    DMPOK_DOWNLOAD_BASE: ClassVar[str] = "https://openzu.cuzk.gov.cz/opendata/DMPOK-TIFF/epsg-3045"
    DMP1G_DOWNLOAD_BASE: ClassVar[str] = "https://openzu.cuzk.cz/opendata/DMP1G/epsg-3045"

    OI_YEAR: ClassVar[int] = 2024
    DMPOK_YEAR: ClassVar[int] = 2025
    DMP1G_YEAR: ClassVar[int] = 2013

    TILE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"(\d{3})_(\d{4})")

    _catalog_granularity_km: int = 2

    def __init__(
        self,
        output_dir: str,
        imagery_from: tuple[int, int | None] | None = None,
        session=None,
        quiet: bool = False,
    ) -> None:
        super().__init__(Region.CZ, output_dir, imagery_from=imagery_from, session=session, quiet=quiet)
        # Versioned cache name to avoid reusing earlier experimental Czech catalogs.
        self._cache_path = CATALOG_CACHE_DIR / "cz_catalog_v3.json"

    def utm_to_grid_coords(
        self, utm_x: float, utm_y: float
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """CZ INSPIRE products use 2km tiles with km-based origin indices."""
        km_per_tile = CZ_GRID_SIZE // METERS_PER_KM  # 2
        grid_x = int(utm_x // CZ_GRID_SIZE) * km_per_tile
        grid_y = int(utm_y // CZ_GRID_SIZE) * km_per_tile
        return (grid_x, grid_y), (grid_x, grid_y)

    def image_filename_from_url(self, url: str) -> str:
        tile = self._tile_name_from_url(url)
        return f"oi_{tile}_{self.OI_YEAR}.jp2"

    def dsm_filename_from_url(self, url: str) -> str:
        tile = self._tile_name_from_url(url)
        url_lower = url.lower()
        if "dmpok" in url_lower:
            return f"dmpok_{tile}_{self.DMPOK_YEAR}.tif"
        if "dmp1g" in url_lower:
            return f"dmp1g_{tile}_{self.DMP1G_YEAR}.laz"
        raise ValueError(f"Unrecognized CZ DSM URL: {url}")

    def download_file(self, url: str, output_path: str, on_progress=None) -> bool:
        """Download CUZK ZIPs and extract the single relevant payload member."""
        output = Path(output_path)
        if output.suffix.lower() not in {".jp2", ".tif", ".laz"}:
            return super().download_file(url, output_path, on_progress=on_progress)

        zip_path = output.with_suffix(".zip")
        if zip_path.exists() and zip_path.stat().st_size >= MIN_FILE_SIZE:
            try:
                self._extract_payload(zip_path, output)
                with suppress(OSError):
                    zip_path.unlink()
                return True
            except Exception:
                with suppress(OSError):
                    zip_path.unlink()

        super().download_file(url, str(zip_path), on_progress=on_progress)
        self._extract_payload(zip_path, output)
        with suppress(OSError):
            zip_path.unlink()
        return True

    def _extract_payload(self, zip_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        with zipfile.ZipFile(zip_path) as zf:
            if suffix == ".jp2":
                member = self._find_zip_member(zf, ".jp2")
            elif suffix == ".tif":
                member = self._find_zip_member(zf, ".tif", ".tiff")
            elif suffix == ".laz":
                member = self._find_zip_member(zf, ".laz")
            else:
                raise RuntimeError(f"Unsupported CZ payload suffix: {suffix}")

            if not member:
                raise RuntimeError(f"No {suffix} payload found in {zip_path.name}")

            tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            with zf.open(member) as src, tmp_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            tmp_path.replace(output_path)

    @staticmethod
    def _find_zip_member(zf: zipfile.ZipFile, *suffixes: str) -> str | None:
        suffixes = tuple(s.lower() for s in suffixes)
        for name in zf.namelist():
            lowered = name.lower()
            if any(lowered.endswith(s) for s in suffixes):
                return name
        return None

    @classmethod
    def _tile_name_from_url(cls, url: str) -> str:
        name = Path(url).stem
        match = cls.TILE_PATTERN.search(name)
        if not match:
            raise ValueError(f"Cannot parse CZ tile name from URL: {url}")
        return f"{int(match.group(1)):03d}_{int(match.group(2)):04d}"

    def _load_catalog(self) -> Catalog:
        image_tiles: dict[tuple[int, int], dict[int, dict]] = {}
        dsm_tiles: dict[tuple[int, int], dict[int, dict]] = {}

        oi_tiles = self._fetch_tiles(self.OI_FEED_URL, self.OI_DOWNLOAD_BASE)
        dmp1g_tiles = self._fetch_tiles(self.DMP1G_FEED_URL, self.DMP1G_DOWNLOAD_BASE)
        dmpok_tiles = self._fetch_tiles(self.DMPOK_FEED_URL, self.DMPOK_DOWNLOAD_BASE)

        for coords, url in oi_tiles.items():
            image_tiles.setdefault(coords, {})[self.OI_YEAR] = self._tile_info(
                url,
                acquisition_date=None,
                source_kind="direct",
                source_age="current",
            )

        for coords, url in dmp1g_tiles.items():
            dsm_tiles.setdefault(coords, {})[self.DMP1G_YEAR] = self._tile_info(
                url,
                acquisition_date=None,
                source_kind="direct",
                source_age="historic",
            )
        for coords, url in dmpok_tiles.items():
            dsm_tiles.setdefault(coords, {})[self.DMPOK_YEAR] = self._tile_info(
                url,
                acquisition_date=None,
                source_kind="direct",
                source_age="current",
            )

        return Catalog(image_tiles=image_tiles, dsm_tiles=dsm_tiles)

    def _fetch_tiles(self, feed_url: str, download_base: str) -> dict[tuple[int, int], str]:
        """Parse a CUZK ATOM feed into {coords: direct_zip_url}."""
        root = fetch_xml_feed(self._session, feed_url)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        tiles: dict[tuple[int, int], str] = {}

        for entry in root.findall("atom:entry", ns):
            title = entry.findtext("atom:title", default="", namespaces=ns)
            match = self.TILE_PATTERN.search(title)
            if not match:
                continue

            grid_x = int(match.group(1))
            grid_y = int(match.group(2))
            tile_name = f"{grid_x:03d}_{grid_y:04d}.zip"
            tiles[(grid_x, grid_y)] = f"{download_base}/{tile_name}"

        return tiles
