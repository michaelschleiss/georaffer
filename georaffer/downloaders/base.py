"""Base class for region-specific tile downloaders."""

import os
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import suppress
from io import BytesIO
from pathlib import Path

import laspy
import requests
from PIL import Image
from requests.adapters import HTTPAdapter

from georaffer.config import (
    CHUNK_SIZE,
    DEFAULT_TIMEOUT,
    FEED_TIMEOUT,
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
    ):
        """Initialize downloader.

        Args:
            region_name: Short name for the region (e.g., "NRW", "RLP")
            output_dir: Base directory for output files
            imagery_from: Optional (from_year, to_year) for historic imagery.
                None = latest only. (2015, None) = all years from 2015. (2015, 2018) = years 2015-2018.
            session: Optional requests.Session for dependency injection (testability)
        """
        self.region_name = region_name
        self.output_dir = output_dir
        self.imagery_from = imagery_from
        self._session = session or requests.Session()

        # Increase pool size so concurrent downloads reuse connections efficiently.
        adapter = HTTPAdapter(
            pool_connections=HTTP_POOL_MAXSIZE, pool_maxsize=HTTP_POOL_MAXSIZE, max_retries=0
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        # Directory structure
        self.raw_dir = Path(output_dir) / "raw"
        self.processed_dir = Path(output_dir) / "processed"

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

    def _parse_jp2_feed(
        self, session: requests.Session, root: ET.Element
    ) -> dict[tuple[int, int], str]:
        """Parse JP2 feed XML and return dict mapping (grid_x, grid_y) -> URL.

        Default implementation returns empty dict. Override for XML feed parsing.
        Subclasses using non-XML sources (e.g., HTML scraping) can skip this.
        """
        return {}

    def _parse_laz_feed(
        self, session: requests.Session, root: ET.Element
    ) -> dict[tuple[int, int], str]:
        """Parse LAZ feed XML and return dict mapping (grid_x, grid_y) -> URL.

        Default implementation returns empty dict. Override for XML feed parsing.
        Subclasses using non-XML sources (e.g., HTML scraping) can skip this.
        """
        return {}

    @property
    @abstractmethod
    def jp2_feed_url(self) -> str:
        """URL to JP2 tile feed."""
        pass

    @property
    @abstractmethod
    def laz_feed_url(self) -> str:
        """URL to LAZ tile feed."""
        pass

    def get_available_tiles(self) -> tuple[dict, dict]:
        """Get available JP2 and LAZ tiles from feeds.

        Fetches and parses the region's feed URLs to build catalogs of available
        tiles. Tile coordinates are extracted from filenames using region-specific
        patterns.

        Returns:
            Tuple of (jp2_tiles, laz_tiles) where each dict maps:
            - Key: (grid_x, grid_y) tuple of tile coordinates in km
            - Value: Download URL for that tile

        Raises:
            Exception: If feed parsing fails for either JP2 or LAZ feeds
        """
        jp2_tiles = {}
        laz_tiles = {}

        try:
            jp2_tiles = self._fetch_and_parse_feed(self.jp2_feed_url, "jp2")
        except Exception as e:
            print(f"Failed to fetch {self.region_name} JP2 tiles: {e}", file=sys.stderr)
            raise

        try:
            laz_tiles = self._fetch_and_parse_feed(self.laz_feed_url, "laz")
        except Exception as e:
            print(f"Failed to fetch {self.region_name} LAZ tiles: {e}", file=sys.stderr)
            raise

        return jp2_tiles, laz_tiles

    def _fetch_and_parse_feed(self, feed_url: str, tile_type: str) -> dict[tuple[int, int], str]:
        """Fetch XML feed and parse using region-specific parser."""
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                delay = self._backoff_delay(attempt)
                if delay > 0:
                    time.sleep(delay)

                response = self._session.get(feed_url, timeout=FEED_TIMEOUT)
                response.raise_for_status()
                root = ET.fromstring(response.content)

                if tile_type == "jp2":
                    return self._parse_jp2_feed(self._session, root)
                else:
                    return self._parse_laz_feed(self._session, root)

            except Exception as e:
                last_error = e

        raise RuntimeError(
            f"Failed to fetch feed {feed_url} after {MAX_RETRIES} retries: {last_error}"
        )

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

                with self._session.get(
                    url, timeout=DEFAULT_TIMEOUT, stream=True
                ) as response:
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

    def get_all_urls_for_coord(self, coords: tuple[int, int]) -> list[str]:
        """Get all URLs (all years) for a coordinate. For multi-year mode."""
        if hasattr(self, "_all_jp2_by_coord"):
            return self._all_jp2_by_coord.get(coords, [])
        return []

    @property
    def total_jp2_count(self) -> int:
        """Total JP2 files including all historical years."""
        if hasattr(self, "_all_jp2_by_coord"):
            return sum(len(urls) for urls in self._all_jp2_by_coord.values())
        return 0
