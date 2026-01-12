"""High-level TileStore API for querying and retrieving geodata tiles.

This module provides a simplified interface for pygeon and other clients
to query available tiles and retrieve them with automatic download and
conversion handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING, Sequence

from georaffer.config import Region, utm_zone_for_region
from georaffer.conversion import convert_file
from georaffer.tiles import _filename_from_url

if TYPE_CHECKING:
    from georaffer.downloaders.base import RegionDownloader


@dataclass(frozen=True)
class Tile:
    """Immutable, hashable tile identifier.

    Represents a single tile from any supported region, suitable for use
    as dict keys and in sets. Replaces pygeon's ReferenceTile and DSM.

    Attributes:
        region: The federal state (NRW, RLP, BB, BW, BY)
        zone: UTM zone number (32 or 33)
        x: Grid X coordinate in kilometers
        y: Grid Y coordinate in kilometers
        tile_type: "image" for orthophotos, "dsm" for elevation data
        url: Download URL for this tile
        year: Year of the data
        recording_date: Acquisition date (for imagery only, None for DSM)
    """

    region: Region
    zone: int
    x: int
    y: int
    tile_type: str
    url: str
    year: int
    recording_date: date | None = None

    def __hash__(self) -> int:
        """Hash using all identifying fields."""
        return hash(
            (self.region, self.zone, self.x, self.y, self.tile_type, self.url, self.year)
        )

    @property
    def coords(self) -> tuple[int, int]:
        """Grid coordinates as (x, y) tuple."""
        return (self.x, self.y)

    @property
    def easting(self) -> int:
        """UTM easting in meters."""
        return self.x * 1000

    @property
    def northing(self) -> int:
        """UTM northing in meters."""
        return self.y * 1000



class TileStore:
    """High-level interface for querying and retrieving geodata tiles.

    Wraps multiple region downloaders, handles caching, downloading,
    and conversion transparently. Users just ask "give me this tile".

    Example:
        store = TileStore(path="/data/tiles", regions=["NRW", "RLP"])
        tiles = store.query(coords=(350, 5600), tile_type="image")
        path = store.get(tiles[0], resolution=2000)  # Downloads if needed

    Directory structure:
        {path}/
        ├── raw/
        │   ├── image/     (downloaded JP2/TIF files)
        │   └── dsm/       (downloaded LAZ/TIF files)
        └── processed/
            ├── image/{resolution}/   (converted GeoTIFF orthophotos)
            └── dsm/{resolution}/     (converted GeoTIFF DSMs)
    """

    def __init__(
        self,
        path: str | Path,
        regions: Sequence[str] | None = None,
        imagery_from: tuple[int, int | None] | None = None,
        quiet: bool = True,
        delete_raw: bool = False,
    ) -> None:
        """Initialize TileStore.

        Args:
            path: Base directory for tile storage (raw and processed subdirs)
            regions: List of region codes to support (e.g., ["NRW", "RLP"]).
                    If None, defaults to ["NRW", "RLP"].
            imagery_from: Optional (from_year, to_year) filter for historic imagery.
                         None = latest only. (2010, None) = all from 2010.
            quiet: Suppress progress output during catalog building.
            delete_raw: Delete raw source files after successful conversion.
        """
        self.path = Path(path)
        self.regions = list(regions) if regions else ["NRW", "RLP"]
        self.imagery_from = imagery_from
        self.quiet = quiet
        self.delete_raw = delete_raw

        # Lazy-initialized downloaders
        self._downloaders: dict[Region, RegionDownloader] | None = None

        # Ensure directory structure exists
        for subdir in ("raw/image", "raw/dsm", "processed/image", "processed/dsm"):
            (self.path / subdir).mkdir(parents=True, exist_ok=True)

    @property
    def downloaders(self) -> dict[Region, RegionDownloader]:
        """Lazily initialize and return region downloaders."""
        if self._downloaders is None:
            # Import here to avoid circular imports
            from georaffer.downloaders import (
                BBDownloader,
                BWDownloader,
                BYDownloader,
                NRWDownloader,
                RLPDownloader,
                THDownloader,
            )

            downloader_classes: dict[str, type[RegionDownloader]] = {
                "NRW": NRWDownloader,
                "RLP": RLPDownloader,
                "BB": BBDownloader,
                "BW": BWDownloader,
                "BY": BYDownloader,
                "TH": THDownloader,
            }

            self._downloaders = {}
            for region_name in self.regions:
                region_name_upper = region_name.upper()
                if region_name_upper not in downloader_classes:
                    raise ValueError(f"Unknown region: {region_name}")
                cls = downloader_classes[region_name_upper]
                region = Region(region_name_upper)
                self._downloaders[region] = cls(
                    str(self.path),
                    imagery_from=self.imagery_from,
                    quiet=self.quiet,
                )
        return self._downloaders

    def query(
        self,
        coords: tuple[int, int],
        tile_type: str = "image",
        *,
        only_recent: bool = False,
    ) -> list[Tile]:
        """Query available tiles at 1km grid coordinates.

        Args:
            coords: (grid_x, grid_y) in 1km coordinates
            tile_type: "image" for orthophotos, "dsm" for elevation data
            only_recent: If True, filter out historic tiles (e.g., WMS) and
                prefer current sources only. Falls back to URL-based inference
                if catalog metadata does not include source_age/source_kind.

        Returns:
            List of Tile objects available at this location (may include
            multiple years if imagery_from is specified).
        """
        tiles: list[Tile] = []

        def _is_wms_url(url: str) -> bool:
            lowered = url.lower()
            return "service=wms" in lowered and "request=getmap" in lowered

        for region, downloader in self.downloaders.items():
            zone = utm_zone_for_region(region)
            for info in downloader.get_tiles(coords, tile_type):
                if only_recent:
                    source_kind = info.get("source_kind")
                    if not source_kind:
                        source_kind = "wms" if _is_wms_url(info["url"]) else "direct"

                    source_age = info.get("source_age")
                    if not source_age:
                        source_age = "historic" if source_kind == "wms" else "current"

                    if source_age != "current":
                        continue
                    if source_kind == "wms":
                        continue

                year = info.get("year")
                acq_date = None
                if acq_str := info.get("acquisition_date"):
                    acq_date = date.fromisoformat(acq_str[:10])
                    if year is None:
                        year = acq_date.year

                if year is None:
                    continue  # Skip tiles without year info

                tiles.append(
                    Tile(
                        region=region,
                        zone=zone,
                        x=coords[0],
                        y=coords[1],
                        tile_type=tile_type,
                        url=info["url"],
                        year=year,
                        recording_date=acq_date,
                    )
                )

        return tiles

    def get(self, tile: Tile, resolution: int) -> Path:
        """Ensure tile is ready and return path to processed file.

        Checks processed cache first, then raw cache, downloads if needed,
        converts if needed.

        Args:
            tile: Tile to retrieve
            resolution: Target resolution in pixels (e.g., 2000 for 2000x2000)

        Returns:
            Path to the processed GeoTIFF file
        """
        # Check processed cache
        proc_path = self._processed_path(tile, resolution)
        if proc_path.exists():
            return proc_path

        # Check raw cache, download if needed
        raw_path = self._raw_path(tile)
        if not raw_path.exists():
            self._download_single(tile, raw_path)

        # Convert
        self._convert_single(raw_path, resolution)

        return proc_path

    def get_many(
        self,
        tiles: Sequence[Tile],
        resolution: int,
        max_pending: int = 4,
    ) -> dict[Tile, Path]:
        """Batch retrieve multiple tiles with pipelined download/convert.

        Downloads and conversions run in parallel with backpressure:
        downloads pause when `max_pending` tiles are waiting for conversion.
        This overlaps network I/O with CPU-bound conversion work.

        Args:
            tiles: Tiles to retrieve
            resolution: Target resolution in pixels
            max_pending: Max tiles downloaded but not yet converted (backpressure).
                        Higher = more parallelism, lower = less disk usage.

        Returns:
            Dict mapping each Tile to its processed file Path
        """
        result: dict[Tile, Path] = {}
        to_process: list[tuple[Tile, Path, bool]] = []  # (tile, raw_path, needs_download)

        # Phase 1: Check caches and categorize work
        for tile in tiles:
            proc_path = self._processed_path(tile, resolution)
            if proc_path.exists():
                result[tile] = proc_path
                continue

            raw_path = self._raw_path(tile)
            needs_download = not raw_path.exists()
            to_process.append((tile, raw_path, needs_download))

        if not to_process:
            return result

        # Phase 2: Pipelined download + convert with backpressure
        # Queue holds tiles ready for conversion; bounded to limit pending work
        ready_queue: Queue[tuple[Tile, Path] | None] = Queue(maxsize=max_pending)
        errors: list[Exception] = []

        def producer() -> None:
            """Download tiles and enqueue for conversion."""
            try:
                for tile, raw_path, needs_download in to_process:
                    if needs_download:
                        self._download_single(tile, raw_path)
                    ready_queue.put((tile, raw_path))  # Blocks if queue full
            except Exception as e:
                errors.append(e)
            finally:
                ready_queue.put(None)  # Sentinel signals completion

        def consumer() -> None:
            """Convert tiles from queue."""
            try:
                while True:
                    item = ready_queue.get()  # Blocks until available
                    if item is None:
                        break
                    tile, raw_path = item
                    self._convert_single(raw_path, resolution)
            except Exception as e:
                errors.append(e)

        producer_thread = Thread(target=producer, name="TileStore-producer")
        consumer_thread = Thread(target=consumer, name="TileStore-consumer")

        producer_thread.start()
        consumer_thread.start()

        producer_thread.join()
        consumer_thread.join()

        if errors:
            raise errors[0]

        # Build result dict for all requested tiles
        for tile in tiles:
            if tile not in result:
                result[tile] = self._processed_path(tile, resolution)

        return result

    # =========================================================================
    # Private path helpers
    # =========================================================================

    def _processed_path(self, tile: Tile, resolution: int) -> Path:
        """Construct path to processed tile."""
        name = self._tile_filename(tile)
        subdir = "image" if tile.tile_type == "image" else "dsm"
        return self.path / "processed" / subdir / str(resolution) / name

    def _raw_path(self, tile: Tile) -> Path:
        """Construct path to raw tile."""
        filename = _filename_from_url(tile.url)
        subdir = "image" if tile.tile_type == "image" else "dsm"
        return self.path / "raw" / subdir / filename

    def _tile_filename(self, tile: Tile) -> str:
        """Generate standardized output filename for a tile.

        Format: {region}_{zone}_{easting}_{northing}_{year}.tif
        """
        return (
            f"{tile.region.value.lower()}_{tile.zone}_"
            f"{tile.easting}_{tile.northing}_{tile.year}.tif"
        )

    # =========================================================================
    # Private download helpers
    # =========================================================================

    def _download_single(self, tile: Tile, raw_path: Path) -> None:
        """Download a single tile."""
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        downloader = self.downloaders[tile.region]
        downloader.download_file(tile.url, str(raw_path))

    # =========================================================================
    # Private conversion helpers
    # =========================================================================

    def _convert_single(self, raw_path: Path, resolution: int) -> None:
        """Convert a single raw tile to processed format."""
        convert_file(str(raw_path), str(self.path / "processed"), resolution, self.delete_raw)
