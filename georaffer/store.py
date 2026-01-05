"""High-level TileStore API for querying and retrieving geodata tiles.

This module provides a simplified interface for pygeon and other clients
to query available tiles and retrieve them with automatic download and
conversion handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from georaffer.config import Region, utm_zone_for_region
from georaffer.conversion import convert_tiles
from georaffer.downloading import DownloadTask, download_parallel_streams
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
    ) -> None:
        """Initialize TileStore.

        Args:
            path: Base directory for tile storage (raw and processed subdirs)
            regions: List of region codes to support (e.g., ["NRW", "RLP"]).
                    If None, defaults to ["NRW", "RLP"].
            imagery_from: Optional (from_year, to_year) filter for historic imagery.
                         None = latest only. (2010, None) = all from 2010.
            quiet: Suppress progress output during catalog building.
        """
        self.path = Path(path)
        self.regions = list(regions) if regions else ["NRW", "RLP"]
        self.imagery_from = imagery_from
        self.quiet = quiet

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
            )

            downloader_classes: dict[str, type[RegionDownloader]] = {
                "NRW": NRWDownloader,
                "RLP": RLPDownloader,
                "BB": BBDownloader,
                "BW": BWDownloader,
                "BY": BYDownloader,
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
    ) -> list[Tile]:
        """Query available tiles at 1km grid coordinates.

        Args:
            coords: (grid_x, grid_y) in 1km coordinates
            tile_type: "image" for orthophotos, "dsm" for elevation data

        Returns:
            List of Tile objects available at this location (may include
            multiple years if imagery_from is specified).
        """
        tiles: list[Tile] = []

        for region, downloader in self.downloaders.items():
            zone = utm_zone_for_region(region)
            for info in downloader.get_tiles(coords, tile_type):
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
        self._convert_single(tile, raw_path, resolution)

        return proc_path

    def get_many(
        self,
        tiles: Sequence[Tile],
        resolution: int,
    ) -> dict[Tile, Path]:
        """Batch retrieve multiple tiles with parallel download/convert.

        More efficient than calling get() repeatedly as it batches
        downloads by region and converts in parallel.

        Args:
            tiles: Tiles to retrieve
            resolution: Target resolution in pixels

        Returns:
            Dict mapping each Tile to its processed file Path
        """
        result: dict[Tile, Path] = {}
        to_download: dict[Region, list[tuple[Tile, Path]]] = {}
        to_convert: list[tuple[Tile, Path]] = []

        # Phase 1: Check caches and categorize work
        for tile in tiles:
            proc_path = self._processed_path(tile, resolution)
            if proc_path.exists():
                result[tile] = proc_path
                continue

            raw_path = self._raw_path(tile)
            if raw_path.exists():
                to_convert.append((tile, raw_path))
            else:
                if tile.region not in to_download:
                    to_download[tile.region] = []
                to_download[tile.region].append((tile, raw_path))

        # Phase 2: Parallel download by region
        if to_download:
            self._download_batch(to_download)
            # Add downloaded tiles to convert queue
            for region_tiles in to_download.values():
                to_convert.extend(region_tiles)

        # Phase 3: Batch convert
        if to_convert:
            self._convert_batch(to_convert, resolution)

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

    def _download_batch(
        self,
        by_region: dict[Region, list[tuple[Tile, Path]]],
    ) -> None:
        """Batch download tiles by region using parallel streams."""
        tasks: list[DownloadTask] = []

        for region, tile_paths in by_region.items():
            downloader = self.downloaders[region]

            # Group by tile type
            image_downloads: list[tuple[str, str]] = []
            dsm_downloads: list[tuple[str, str]] = []

            for tile, raw_path in tile_paths:
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                download_entry = (tile.url, str(raw_path))
                if tile.tile_type == "image":
                    image_downloads.append(download_entry)
                else:
                    dsm_downloads.append(download_entry)

            if image_downloads:
                tasks.append(
                    DownloadTask(
                        name=f"{region.value} Imagery",
                        downloads=image_downloads,
                        downloader=downloader,
                    )
                )
            if dsm_downloads:
                tasks.append(
                    DownloadTask(
                        name=f"{region.value} DSM",
                        downloads=dsm_downloads,
                        downloader=downloader,
                    )
                )

        if tasks:
            _, stats = download_parallel_streams(tasks, force=False)
            if stats.failed > 0:
                raise RuntimeError(f"Download failed: {stats.failed} files")

    # =========================================================================
    # Private conversion helpers
    # =========================================================================

    def _convert_single(
        self,
        tile: Tile,
        raw_path: Path,
        resolution: int,
    ) -> None:
        """Convert a single raw tile to processed format."""
        convert_tiles(
            raw_dir=str(self.path / "raw"),
            processed_dir=str(self.path / "processed"),
            resolutions=[resolution],
            max_workers=1,
            process_images=(tile.tile_type == "image"),
            process_pointclouds=(tile.tile_type == "dsm"),
            image_files=[str(raw_path)] if tile.tile_type == "image" else None,
            dsm_files=[str(raw_path)] if tile.tile_type == "dsm" else None,
        )

    def _convert_batch(
        self,
        tiles_and_paths: list[tuple[Tile, Path]],
        resolution: int,
    ) -> None:
        """Batch convert multiple raw tiles."""
        image_files = [str(p) for t, p in tiles_and_paths if t.tile_type == "image"]
        dsm_files = [str(p) for t, p in tiles_and_paths if t.tile_type == "dsm"]

        if not image_files and not dsm_files:
            return

        convert_tiles(
            raw_dir=str(self.path / "raw"),
            processed_dir=str(self.path / "processed"),
            resolutions=[resolution],
            max_workers=4,
            process_images=bool(image_files),
            process_pointclouds=bool(dsm_files),
            image_files=image_files if image_files else None,
            dsm_files=dsm_files if dsm_files else None,
        )
