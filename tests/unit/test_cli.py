"""Tests for CLI module."""

from argparse import Namespace
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from georaffer.cli import load_coordinates, normalize_regions, validate_args
from georaffer.config import Region


class TestLoadCoordinatesCSV:
    """Tests for load_coordinates with CSV source."""

    def test_csv_utm_coords(self, tmp_path):
        """Test loading UTM coordinates from CSV."""
        csv_file = tmp_path / "coords.csv"
        csv_file.write_text("x,y\n350000,5600000\n351000,5601000\n")

        args = Namespace(
            command="csv",
            file=str(csv_file),
            cols="x,y",
            utm_zone=32,
        )

        coords, _ = load_coordinates(args)

        assert len(coords) == 2
        assert coords[0] == (350000.0, 5600000.0)

    def test_csv_latlon_conversion(self, tmp_path):
        """Test lat/lon to UTM conversion from CSV (auto-detected by value range)."""
        csv_file = tmp_path / "coords.csv"
        csv_file.write_text("longitude,latitude\n6.9603,50.9375\n")

        args = Namespace(
            command="csv",
            file=str(csv_file),
            cols="longitude,latitude",
        )

        with patch("utm.from_latlon") as mock_utm:
            mock_utm.return_value = (350000, 5640000, 32, "N")
            coords, _ = load_coordinates(args)

        assert len(coords) == 1
        mock_utm.assert_called_once()

    def test_csv_latlon_rejects_utm_zone(self, tmp_path):
        """Lat/lon CSV inputs should not accept --utm-zone."""
        csv_file = tmp_path / "coords.csv"
        csv_file.write_text("longitude,latitude\n6.9603,50.9375\n")

        args = Namespace(
            command="csv",
            file=str(csv_file),
            cols="longitude,latitude",
            utm_zone=32,
        )

        with pytest.raises(ValueError, match="lat/lon"):
            load_coordinates(args)

    def test_csv_missing_file_raises(self, tmp_path):
        """Test error when file doesn't exist."""
        args = Namespace(
            command="csv",
            file=str(tmp_path / "nonexistent.csv"),
            cols="x,y",
        )

        with pytest.raises(FileNotFoundError):
            load_coordinates(args)


class TestNormalizeRegions:
    def test_deduplicates_and_preserves_order(self):
        regions = normalize_regions(["nrw", "rlp", "nrw"])
        assert regions == [Region.NRW, Region.RLP]

    def test_case_insensitive(self):
        regions = normalize_regions(["BB"])
        assert regions == [Region.BB]


class TestLoadCoordinatesBbox:
    """Tests for load_coordinates with bbox source."""

    def test_bbox_single_tile(self):
        """Test bbox converts tiles to UTM center points."""
        args = Namespace(
            command="bbox",
            bbox="350000,5600000,350500,5600500",  # Single tile bbox
            utm_zone=32,
        )

        coords, _ = load_coordinates(args)

        # Should convert tile coords to UTM center points
        assert len(coords) == 1
        # Default grid (1km): 350*1000 + 500 = 350500
        assert coords[0] == (350500.0, 5600500.0)

    def test_bbox_multiple_tiles(self):
        """Test bbox spanning multiple tiles."""
        args = Namespace(
            command="bbox",
            bbox="350000,5600000,351500,5601500",  # 2x2 tile bbox
            utm_zone=32,
        )

        coords, _ = load_coordinates(args)

        # Should generate 4 tiles (2x2)
        assert len(coords) == 4

    def test_bbox_invalid_format_raises(self):
        """Test error when bbox has wrong format."""
        args = Namespace(
            command="bbox",
            bbox="invalid",
        )

        with pytest.raises(ValueError):
            load_coordinates(args)

    def test_bbox_latlon_auto_detected(self):
        """Test lat/lon bbox is auto-detected and converted to UTM."""
        args = Namespace(
            command="bbox",
            bbox="6.9,50.9,7.0,51.0",  # Small values = lat/lon
        )

        coords, _ = load_coordinates(args)

        # Should detect as lat/lon and convert to UTM
        # Results should be in UTM range (not small lat/lon values)
        assert len(coords) >= 1
        x, y = coords[0]
        assert x > 100000  # UTM easting is large
        assert y > 5000000  # UTM northing is large

    def test_bbox_latlon_rejects_utm_zone(self):
        """Lat/lon bbox inputs should not accept --utm-zone."""
        args = Namespace(
            command="bbox",
            bbox="6.9,50.9,7.0,51.0",
            utm_zone=32,
        )

        with pytest.raises(ValueError, match="lat/lon"):
            load_coordinates(args)


class TestLoadCoordinatesTif:
    """Tests for load_coordinates with tif source."""

    def test_tif_utm_bounds(self, tmp_path):
        """Test loading coordinates from a UTM GeoTIFF."""
        tif_path = tmp_path / "input.tif"
        transform = from_origin(500250, 5800250, 10, 10)
        data = np.zeros((1, 10, 10), dtype=np.uint8)

        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=data.shape[1],
            width=data.shape[2],
            count=1,
            dtype=data.dtype,
            crs="EPSG:25833",
            transform=transform,
        ) as dst:
            dst.write(data)

        args = Namespace(
            command="tif",
            tif=str(tif_path),
        )

        coords, source_zone = load_coordinates(args)

        assert source_zone == 33
        assert coords == [(500500.0, 5800500.0)]

    def test_tif_latlon_auto_detected(self, tmp_path):
        """Test lat/lon GeoTIFF is auto-detected and converted to UTM."""
        tif_path = tmp_path / "input_latlon.tif"
        transform = from_origin(13.4, 52.6, 0.001, 0.001)
        data = np.zeros((1, 10, 10), dtype=np.uint8)

        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=data.shape[1],
            width=data.shape[2],
            count=1,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

        args = Namespace(
            command="tif",
            tif=str(tif_path),
        )

        with patch("utm.from_latlon") as mock_utm:
            mock_utm.side_effect = [
                (350000, 5600000, 32, "N"),
                (350500, 5600500, 32, "N"),
            ]
            coords, source_zone = load_coordinates(args)

        assert source_zone == 32
        assert coords == [(350500.0, 5600500.0)]


class TestValidateArgs:
    def test_from_before_1994_errors(self):
        args = Namespace(
            command="bbox",
            bbox="350000,5600000,350500,5600500",
            output="out",
            margin=0,
            pixel_size=[0.5],
            workers=None,
            reprocess=False,
            profiling=False,
            from_year=1993,
            to_year=None,
            data_types=None,
            utm_zone=32,
            region=["rlp"],
        )
        errors = validate_args(args)
        assert any("before 1994" in e for e in errors)


class TestLoadCoordinatesTiles:
    """Tests for load_coordinates with tiles source."""

    def test_tiles_single(self):
        """Test single tile coordinate."""
        args = Namespace(
            command="tiles",
            tiles=["350,5600"],
            utm_zone=32,
        )

        coords, _ = load_coordinates(args)

        assert len(coords) == 1
        # Default grid: center at 350500, 5600500
        assert coords[0] == (350500.0, 5600500.0)

    def test_tiles_multiple(self):
        """Test multiple tile coordinates."""
        args = Namespace(
            command="tiles",
            tiles=["350,5600", "351,5601"],
            utm_zone=32,
        )

        coords, _ = load_coordinates(args)

        assert len(coords) == 2
        assert coords[0] == (350500.0, 5600500.0)
        assert coords[1] == (351500.0, 5601500.0)

    def test_tiles_invalid_format_raises(self):
        """Test error when tile has wrong format."""
        args = Namespace(
            command="tiles",
            tiles=["invalid"],
            utm_zone=32,
        )

        with pytest.raises(ValueError):
            load_coordinates(args)


class TestLoadCoordinatesPygeon:
    """Tests for load_coordinates with pygeon source."""

    def test_pygeon_missing_path_raises(self):
        """Test error when dataset path doesn't exist."""
        args = Namespace(command="pygeon", dataset_path="/nonexistent/path")

        # Should fail when trying to load from nonexistent path
        with pytest.raises(Exception):  # Could be FileNotFoundError or similar
            load_coordinates(args)

    def test_pygeon_auto_detects_zone(self):
        """Test auto-detecting UTM zone for pygeon lat/lon coordinates."""
        args = Namespace(command="pygeon", dataset_path="/fake/path")
        raw_coords = [(52.5, 13.4, 100.0), (52.6, 13.5, 110.0)]

        with patch("georaffer.cli.load_from_pygeon") as mock_load:
            with patch("georaffer.cli.latlon_array_to_utm") as mock_utm:
                mock_load.return_value = raw_coords
                mock_utm.return_value = ([1.0, 2.0], [3.0, 4.0])

                coords, source_zone = load_coordinates(args)

        assert source_zone == 33
        assert coords == [(1.0, 3.0), (2.0, 4.0)]
        assert mock_utm.call_args.kwargs["force_zone_number"] == 33
