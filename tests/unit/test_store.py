"""Tests for the TileStore API."""

from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from georaffer.config import Region
from georaffer.store import Tile, TileStore


class TestTile:
    """Tests for Tile dataclass."""

    def test_tile_is_hashable(self):
        """Tiles can be used as dict keys and in sets."""
        tile = Tile(
            region=Region.NRW,
            zone=32,
            x=350,
            y=5600,
            tile_type="image",
            url="http://example.com/tile.jp2",
            year=2021,
            recording_date=date(2021, 5, 15),
        )
        assert hash(tile) is not None
        d = {tile: "value"}
        assert d[tile] == "value"

    def test_tile_in_set(self):
        """Tiles can be stored in sets."""
        tile1 = Tile(
            region=Region.NRW,
            zone=32,
            x=350,
            y=5600,
            tile_type="image",
            url="http://example.com/tile.jp2",
            year=2021,
        )
        tile2 = Tile(
            region=Region.NRW,
            zone=32,
            x=350,
            y=5600,
            tile_type="image",
            url="http://example.com/tile.jp2",
            year=2021,
        )
        s = {tile1}
        assert tile2 in s

    def test_tile_is_frozen(self):
        """Tiles are immutable."""
        tile = Tile(
            region=Region.NRW,
            zone=32,
            x=350,
            y=5600,
            tile_type="image",
            url="http://example.com/tile.jp2",
            year=2021,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            tile.x = 999

    def test_tile_coords_property(self):
        """coords property returns (x, y) tuple."""
        tile = Tile(
            region=Region.NRW,
            zone=32,
            x=350,
            y=5600,
            tile_type="image",
            url="http://example.com/tile.jp2",
            year=2021,
        )
        assert tile.coords == (350, 5600)

    def test_tile_easting_northing(self):
        """easting/northing properties return meters."""
        tile = Tile(
            region=Region.NRW,
            zone=32,
            x=350,
            y=5600,
            tile_type="image",
            url="http://example.com/tile.jp2",
            year=2021,
        )
        assert tile.easting == 350000
        assert tile.northing == 5600000

    def test_tile_without_date(self):
        """DSM tiles can have recording_date=None."""
        tile = Tile(
            region=Region.NRW,
            zone=32,
            x=350,
            y=5600,
            tile_type="dsm",
            url="http://example.com/tile.laz",
            year=2021,
            recording_date=None,
        )
        assert tile.recording_date is None
        assert tile.year == 2021


class TestTileStoreInit:
    """Tests for TileStore initialization."""

    def test_init_creates_directories(self, tmp_path):
        """TileStore creates required directory structure."""
        store = TileStore(path=tmp_path, regions=["NRW"])
        assert (tmp_path / "raw/image").exists()
        assert (tmp_path / "raw/dsm").exists()
        assert (tmp_path / "processed/image").exists()
        assert (tmp_path / "processed/dsm").exists()

    def test_init_default_regions(self, tmp_path):
        """Default regions are NRW and RLP."""
        store = TileStore(path=tmp_path)
        assert store.regions == ["NRW", "RLP"]

    def test_init_custom_regions(self, tmp_path):
        """Custom regions are preserved."""
        store = TileStore(path=tmp_path, regions=["BB", "BY"])
        assert store.regions == ["BB", "BY"]

    def test_downloaders_not_created_on_init(self, tmp_path):
        """Downloaders are lazily initialized."""
        store = TileStore(path=tmp_path, regions=["NRW"])
        assert store._downloaders is None


class TestTileStorePaths:
    """Tests for path construction logic."""

    def test_processed_path_image(self, tmp_path):
        """Processed image path follows expected format."""
        store = TileStore(path=tmp_path, regions=["NRW"])
        tile = Tile(
            region=Region.NRW,
            zone=32,
            x=350,
            y=5600,
            tile_type="image",
            url="http://example.com/tile.jp2",
            year=2021,
        )
        path = store._processed_path(tile, resolution=2000)
        expected = tmp_path / "processed/image/2000/nrw_32_350000_5600000_2021.tif"
        assert path == expected

    def test_processed_path_dsm(self, tmp_path):
        """Processed DSM path follows expected format."""
        store = TileStore(path=tmp_path, regions=["NRW"])
        tile = Tile(
            region=Region.NRW,
            zone=32,
            x=350,
            y=5600,
            tile_type="dsm",
            url="http://example.com/tile.laz",
            year=2021,
        )
        path = store._processed_path(tile, resolution=2000)
        expected = tmp_path / "processed/dsm/2000/nrw_32_350000_5600000_2021.tif"
        assert path == expected

    def test_raw_path_extracts_filename(self, tmp_path):
        """Raw paths use filename from URL."""
        store = TileStore(path=tmp_path, regions=["NRW"])
        tile = Tile(
            region=Region.NRW,
            zone=32,
            x=350,
            y=5600,
            tile_type="image",
            url="http://example.com/dop10rgbi_32_350_5600_1_nw_2021.jp2",
            year=2021,
        )
        path = store._raw_path(tile)
        expected = tmp_path / "raw/image/dop10rgbi_32_350_5600_1_nw_2021.jp2"
        assert path == expected

    def test_tile_filename_format(self, tmp_path):
        """Tile filename follows standard format."""
        store = TileStore(path=tmp_path, regions=["NRW"])
        tile = Tile(
            region=Region.NRW,
            zone=32,
            x=350,
            y=5600,
            tile_type="image",
            url="http://example.com/tile.jp2",
            year=2021,
        )
        filename = store._tile_filename(tile)
        assert filename == "nrw_32_350000_5600000_2021.tif"

    def test_tile_filename_rlp(self, tmp_path):
        """Tile filename works for RLP region."""
        store = TileStore(path=tmp_path, regions=["RLP"])
        tile = Tile(
            region=Region.RLP,
            zone=32,
            x=380,
            y=5540,
            tile_type="image",
            url="http://example.com/tile.jp2",
            year=2020,
        )
        filename = store._tile_filename(tile)
        assert filename == "rlp_32_380000_5540000_2020.tif"


class TestTileStoreGet:
    """Tests for get() behavior."""

    def test_get_returns_cached_if_exists(self, tmp_path):
        """get() returns immediately if processed file exists."""
        store = TileStore(path=tmp_path, regions=["NRW"])
        tile = Tile(
            region=Region.NRW,
            zone=32,
            x=350,
            y=5600,
            tile_type="image",
            url="http://example.com/tile.jp2",
            year=2021,
        )

        # Create processed file
        proc_path = store._processed_path(tile, 2000)
        proc_path.parent.mkdir(parents=True, exist_ok=True)
        proc_path.touch()

        result = store.get(tile, resolution=2000)
        assert result == proc_path

    @patch("georaffer.store.convert_tiles")
    def test_get_converts_if_raw_exists(self, mock_convert, tmp_path):
        """get() converts if raw exists but processed doesn't."""
        store = TileStore(path=tmp_path, regions=["NRW"])
        tile = Tile(
            region=Region.NRW,
            zone=32,
            x=350,
            y=5600,
            tile_type="image",
            url="http://example.com/dop10rgbi_32_350_5600_1_nw_2021.jp2",
            year=2021,
        )

        # Create raw file
        raw_path = store._raw_path(tile)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.touch()

        store.get(tile, resolution=2000)

        # Verify convert_tiles was called
        mock_convert.assert_called_once()
        call_kwargs = mock_convert.call_args.kwargs
        assert call_kwargs["resolutions"] == [2000]
        assert call_kwargs["process_images"] is True


class TestTileStoreGetMany:
    """Tests for get_many() behavior."""

    def test_get_many_returns_cached(self, tmp_path):
        """get_many() returns cached paths for processed tiles."""
        store = TileStore(path=tmp_path, regions=["NRW"])
        tile1 = Tile(
            region=Region.NRW,
            zone=32,
            x=350,
            y=5600,
            tile_type="image",
            url="http://example.com/tile1.jp2",
            year=2021,
        )
        tile2 = Tile(
            region=Region.NRW,
            zone=32,
            x=351,
            y=5600,
            tile_type="image",
            url="http://example.com/tile2.jp2",
            year=2021,
        )

        # Create both processed files
        for tile in [tile1, tile2]:
            proc_path = store._processed_path(tile, 2000)
            proc_path.parent.mkdir(parents=True, exist_ok=True)
            proc_path.touch()

        result = store.get_many([tile1, tile2], resolution=2000)

        assert tile1 in result
        assert tile2 in result
        assert result[tile1] == store._processed_path(tile1, 2000)
        assert result[tile2] == store._processed_path(tile2, 2000)


class TestTileStoreQuery:
    """Tests for query() behavior."""

    def test_query_returns_tiles(self, tmp_path):
        """query() returns Tile objects from downloader results."""
        mock_downloader = Mock()
        mock_downloader.get_tiles.return_value = [
            {
                "url": "http://example.com/tile.jp2",
                "acquisition_date": "2021-05-15",
                "year": 2021,
            }
        ]

        store = TileStore(path=tmp_path, regions=["NRW"])
        store._downloaders = {Region.NRW: mock_downloader}

        tiles = store.query(coords=(350, 5600), tile_type="image")

        assert len(tiles) == 1
        assert tiles[0].region == Region.NRW
        assert tiles[0].x == 350
        assert tiles[0].y == 5600
        assert tiles[0].year == 2021
        assert tiles[0].recording_date == date(2021, 5, 15)
        assert tiles[0].tile_type == "image"

    def test_query_dsm_tiles(self, tmp_path):
        """query() handles DSM tiles without acquisition_date."""
        mock_downloader = Mock()
        mock_downloader.get_tiles.return_value = [
            {"url": "http://example.com/tile.laz", "year": 2020}
        ]

        store = TileStore(path=tmp_path, regions=["NRW"])
        store._downloaders = {Region.NRW: mock_downloader}

        tiles = store.query(coords=(350, 5600), tile_type="dsm")

        assert len(tiles) == 1
        assert tiles[0].tile_type == "dsm"
        assert tiles[0].year == 2020
        assert tiles[0].recording_date is None

    def test_query_multiple_regions(self, tmp_path):
        """query() aggregates tiles from multiple regions."""
        mock_nrw = Mock()
        mock_nrw.get_tiles.return_value = [
            {"url": "http://nrw.de/tile.jp2", "acquisition_date": "2021-05-15", "year": 2021}
        ]
        mock_rlp = Mock()
        mock_rlp.get_tiles.return_value = [
            {"url": "http://rlp.de/tile.jp2", "acquisition_date": "2020-06-20", "year": 2020}
        ]

        store = TileStore(path=tmp_path, regions=["NRW", "RLP"])
        store._downloaders = {Region.NRW: mock_nrw, Region.RLP: mock_rlp}

        tiles = store.query(coords=(350, 5600), tile_type="image")

        assert len(tiles) == 2
        regions = {t.region for t in tiles}
        assert regions == {Region.NRW, Region.RLP}

    def test_query_skips_tiles_without_year(self, tmp_path):
        """query() skips tiles without year info."""
        mock_downloader = Mock()
        mock_downloader.get_tiles.return_value = [
            {"url": "http://example.com/tile.jp2"},  # No year or date
            {"url": "http://example.com/tile2.jp2", "year": 2021},
        ]

        store = TileStore(path=tmp_path, regions=["NRW"])
        store._downloaders = {Region.NRW: mock_downloader}

        tiles = store.query(coords=(350, 5600), tile_type="image")

        assert len(tiles) == 1
        assert tiles[0].year == 2021
