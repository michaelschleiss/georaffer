"""Base class for region-specific tile downloaders."""

import json
import os
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import laspy
import requests
from PIL import Image
from requests.adapters import HTTPAdapter

from georaffer.config import (
    CATALOG_CACHE_DIR,
    CATALOG_TTL_DAYS,
    CHUNK_SIZE,
    DEFAULT_TIMEOUT,
    HTTP_POOL_MAXSIZE,
    LAZ_SAMPLE_SIZE,
    MAX_RETRIES,
    MIN_FILE_SIZE,
    RETRY_BACKOFF_BASE,
    RETRY_MAX_WAIT,
)
from georaffer.runtime import InterruptManager

# Suppress PIL decompression bomb warnings for large aerial orthophotos.
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None


@dataclass
class Catalog:
    """Complete tile catalog for a region (all years).

    Stores mapping of (grid_x, grid_y) -> {year: tile_info} for both image and DSM tiles.

    tile_info is a dict with keys:
        - "url": str (download URL)
        - "acquisition_date": str | None (ISO format, e.g. "2023-05-27")
    """

    image_tiles: dict[tuple[int, int], dict[int, dict]] = field(default_factory=dict)
    dsm_tiles: dict[tuple[int, int], dict[int, dict]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def is_stale(self, ttl_days: int = CATALOG_TTL_DAYS) -> bool:
        """Check if catalog has exceeded TTL."""
        return datetime.now() - self.created_at > timedelta(days=ttl_days)

    def to_dict(self) -> dict:
        """Serialize catalog to JSON-compatible dict."""
        return {
            "created_at": self.created_at.isoformat(),
            "image_tiles": {
                f"{x},{y}": {str(year): tile for year, tile in years.items()}
                for (x, y), years in self.image_tiles.items()
            },
            "dsm_tiles": {
                f"{x},{y}": {str(year): tile for year, tile in years.items()}
                for (x, y), years in self.dsm_tiles.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Catalog":
        """Deserialize catalog from JSON-compatible dict."""
        image_tiles: dict[tuple[int, int], dict[int, dict]] = {}
        for coord_str, years in data["image_tiles"].items():
            x, y = map(int, coord_str.split(","))
            image_tiles[(x, y)] = {int(year): tile for year, tile in years.items()}

        dsm_tiles: dict[tuple[int, int], dict[int, dict]] = {}
        for coord_str, years in data["dsm_tiles"].items():
            x, y = map(int, coord_str.split(","))
            dsm_tiles[(x, y)] = {int(year): tile for year, tile in years.items()}

        return cls(
            image_tiles=image_tiles,
            dsm_tiles=dsm_tiles,
            created_at=datetime.fromisoformat(data["created_at"]),
        )


class RegionDownloader(ABC):
    """Base class for region-specific downloaders.

    Implements Template Method pattern: subclasses override hook methods
    for region-specific parsing while sharing common download logic.
    """

    def __init__(
        self,
        region_name: str,
        output_dir: str,
        imagery_from: tuple[int, int | None] | None = None,
        session: requests.Session | None = None,
        quiet: bool = False,
    ):
        """Initialize downloader.

        Args:
            region_name: Short name for the region (e.g., "NRW", "RLP")
            output_dir: Base directory for output files
            imagery_from: Optional (from_year, to_year) for historic imagery.
                None = latest only. (2015, None) = all years from 2015. (2015, 2018) = years 2015-2018.
            session: Optional requests.Session for dependency injection; inject a mock for testing
            quiet: Suppress progress output
        """
        self.region_name = region_name
        self.output_dir = output_dir
        self.imagery_from = imagery_from
        self.quiet = quiet
        self._session = session or requests.Session()

        adapter = HTTPAdapter(
            pool_connections=HTTP_POOL_MAXSIZE, pool_maxsize=HTTP_POOL_MAXSIZE, max_retries=0
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        # Directory structure
        self.raw_dir = Path(output_dir) / "raw"
        self.processed_dir = Path(output_dir) / "processed"

        # Catalog cache (subclass sets _cache_path in its __init__)
        self._catalog: Catalog | None = None
        self._cache_path: Path | None = None

    @property
    def session(self) -> requests.Session:
        """Get HTTP session (allows injection for testing)."""
        return self._session

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        """Calculate exponential backoff delay for retry attempt.

        Args:
            attempt: The current attempt number (0-indexed, delay only for attempt > 0)

        Returns:
            Delay in seconds, capped at RETRY_MAX_WAIT
        """
        if attempt <= 0:
            return 0.0
        return min(RETRY_BACKOFF_BASE ** (attempt - 1), RETRY_MAX_WAIT)

    @staticmethod
    def _year_in_range(year: int, from_year: int | None, to_year: int | None) -> bool:
        """Check if a year falls within the specified range.

        Args:
            year: The year to check
            from_year: Start of range (inclusive), or None for no lower bound
            to_year: End of range (inclusive), or None for no upper bound

        Returns:
            True if year is in range
        """
        if from_year is None:
            return True
        if to_year is None:
            return year >= from_year
        return from_year <= year <= to_year

    @abstractmethod
    def utm_to_grid_coords(
        self, utm_x: float, utm_y: float
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Convert UTM to grid coordinates for JP2 and LAZ files.

        Args:
            utm_x: UTM easting in meters
            utm_y: UTM northing in meters

        Returns:
            Tuple of ((jp2_grid_x, jp2_grid_y), (laz_grid_x, laz_grid_y)) where:
            - jp2_grid coords: Grid indices for orthophoto tiles (km-based)
            - laz_grid coords: Grid indices for DSM tiles (km-based)
            - For most regions these are identical (same grid system)
        """
        pass

    # =========================================================================
    # Catalog caching
    # =========================================================================

    def build_catalog(self, refresh: bool = False) -> Catalog:
        """Fetch and cache complete tile catalog for this region.

        Checks instance cache first, then disk cache. If both miss (or refresh=True),
        loads from sources via _load_catalog() and persists to disk.

        Args:
            refresh: Force reload from sources, ignoring cache

        Returns:
            Catalog with all available tiles for this region
        """
        start = time.perf_counter()

        # Instance cache hit
        if self._catalog is not None and not refresh:
            self._print_catalog_summary("memory")
            return self._catalog

        # Disk cache hit
        if not refresh:
            self._catalog = self._read_cache()
            if self._catalog is not None:
                self._print_catalog_summary("cache")
                return self._catalog

        # Load from sources and persist
        if not self.quiet:
            print(f"{self.region_name}: Building catalog...")
        self._catalog = self._load_catalog()
        self._write_cache()
        elapsed = time.perf_counter() - start
        self._print_catalog_summary("built", elapsed)
        return self._catalog

    def _print_catalog_summary(self, source: str, elapsed: float | None = None) -> None:
        """Print catalog summary line."""
        if self._catalog is None or self.quiet:
            return
        tiles = len(self._catalog.image_tiles)
        images = sum(len(years) for years in self._catalog.image_tiles.values())
        if source == "memory":
            print(f"{self.region_name}: Catalog ready ({tiles:,} tiles, {images:,} images)")
        elif source == "cache":
            print(f"{self.region_name}: Loaded from cache ({tiles:,} tiles, {images:,} images)")
        else:
            time_str = f", {elapsed:.1f}s" if elapsed else ""
            print(f"  Catalog built ({tiles:,} tiles, {images:,} images{time_str})")

    @abstractmethod
    def _load_catalog(self) -> Catalog:
        """Load catalog from sources. Subclass must implement."""
        pass

    def _read_cache(self) -> Catalog | None:
        """Read catalog from disk cache if fresh."""
        if self._cache_path is None:
            return None

        try:
            if not self._cache_path.exists():
                return None

            with open(self._cache_path) as f:
                data = json.load(f)

            catalog = Catalog.from_dict(data)

            if catalog.is_stale():
                return None

            return catalog

        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def _write_cache(self) -> None:
        """Write catalog to disk cache."""
        if self._cache_path is None or self._catalog is None:
            return

        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w") as f:
                json.dump(self._catalog.to_dict(), f, indent=2)
        except OSError:
            pass  # Cache write failure is not fatal

    def download_file(
        self, url: str, output_path: str, on_progress: Callable | None = None
    ) -> bool:
        """Download a file with retry logic and integrity checking.

        Streams to disk to avoid large in-memory buffers, validates size and integrity,
        then atomically renames the temp file. Retries up to MAX_RETRIES times with
        exponential backoff on transient failures.

        Args:
            url: URL to download
            output_path: Local path to save file
            on_progress: Optional callback(bytes_downloaded: int) called after each chunk

        Returns:
            True if download succeeded

        Raises:
            RuntimeError: If download fails after MAX_RETRIES attempts or on unexpected errors
            KeyboardInterrupt: If interrupted via InterruptManager
        """
        last_error = None
        temp_path = output_path + ".tmp"

        for attempt in range(MAX_RETRIES):
            try:
                delay = self._backoff_delay(attempt)
                if delay > 0:
                    time.sleep(delay)

                with self._session.get(url, timeout=DEFAULT_TIMEOUT, stream=True) as response:
                    response.raise_for_status()

                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

                    expected_size = int(response.headers.get("content-length", 0))
                    downloaded = 0

                    with open(temp_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                            if InterruptManager.get().is_set():
                                raise KeyboardInterrupt()
                            if not chunk:
                                continue
                            f.write(chunk)
                            downloaded += len(chunk)
                            if on_progress:
                                on_progress(len(chunk))

                # Validate size
                if downloaded < MIN_FILE_SIZE:
                    self._cleanup_temp(output_path)
                    continue

                if expected_size > 0 and downloaded != expected_size:
                    self._cleanup_temp(output_path)
                    continue

                # Verify integrity from disk to avoid large memory use
                with open(temp_path, "rb") as f:
                    if not self._verify_file_integrity(f, output_path):
                        self._cleanup_temp(output_path)
                        continue

                os.replace(temp_path, output_path)
                return True

            except requests.RequestException as e:
                last_error = e
                self._cleanup_temp(output_path)
            except Exception as e:
                self._cleanup_temp(output_path)
                raise RuntimeError(f"Download failed for {url}: {e}") from e

        raise RuntimeError(
            f"Download failed after {MAX_RETRIES} retries for {url}: {last_error}"
        ) from last_error

    def _cleanup_temp(self, output_path: str) -> None:
        """Remove temporary file if it exists."""
        with suppress(FileNotFoundError, PermissionError):
            Path(output_path + ".tmp").unlink()

    def _verify_file_integrity(self, buffer: BytesIO, filename: str) -> bool:
        """Verify file integrity from memory buffer."""
        ext = Path(filename).suffix.lower()

        try:
            if ext == ".laz":
                buffer.seek(0)
                with laspy.open(buffer) as laz:
                    if laz.header.point_count == 0:
                        return False
                    chunk_iter = laz.chunk_iterator(LAZ_SAMPLE_SIZE)
                    first_chunk = next(chunk_iter)
                    if len(first_chunk) == 0:
                        return False
                return True

            elif ext == ".jp2":
                buffer.seek(0)
                img = Image.open(buffer)
                try:
                    return not (img.size[0] == 0 or img.size[1] == 0)
                finally:
                    img.close()

            return True

        except Exception:
            return False

    @property
    def total_image_count(self) -> int:
        """Total image tiles including all historical years (filtered by imagery_from)."""
        catalog = self.build_catalog()

        from_year, to_year = None, None
        if self.imagery_from:
            from_year, to_year = self.imagery_from

        return sum(
            1 for years in catalog.image_tiles.values()
            for year in years
            if self._year_in_range(year, from_year, to_year)
        )

    # Catalog coordinate granularity in km. Override in subclasses with larger native tiles.
    _catalog_granularity_km: int = 1

    def get_tiles(
        self, coords: tuple[int, int], tile_type: str = "image"
    ) -> list[dict]:
        """Get available tiles at 1km grid coordinates.

        On-demand query for tiles at a specific location. Returns all years
        available at that coordinate. Handles coordinate mapping for regions
        with larger native tile sizes (e.g., RLP 2km tiles).

        Args:
            coords: (grid_x, grid_y) in 1km grid coordinates
            tile_type: "image" for orthophotos, "dsm" for elevation data

        Returns:
            List of tile info dicts, each with:
                - "url": str (download URL)
                - "acquisition_date": str | None (ISO format)
                - "year": int
        """
        catalog = self.build_catalog()
        tiles_dict = catalog.image_tiles if tile_type == "image" else catalog.dsm_tiles

        # Map 1km query coords to catalog coords (e.g., 351 -> 350 for 2km granularity)
        g = self._catalog_granularity_km
        catalog_coords = ((coords[0] // g) * g, (coords[1] // g) * g)

        years = tiles_dict.get(catalog_coords, {})
        return [
            {"url": info["url"], "acquisition_date": info.get("acquisition_date"), "year": year}
            for year, info in years.items()
        ]
