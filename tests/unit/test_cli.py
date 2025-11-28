"""Tests for CLI module."""

from argparse import Namespace
from unittest.mock import patch

import pytest

from georaffer.cli import load_coordinates


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
        )

        coords = load_coordinates(args)

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

        with patch("georaffer.grids.latlon_to_utm") as mock_utm:
            mock_utm.return_value = (350000, 5640000)
            coords = load_coordinates(args)

        assert len(coords) == 1
        mock_utm.assert_called_once()

    def test_csv_missing_file_raises(self, tmp_path):
        """Test error when file doesn't exist."""
        args = Namespace(
            command="csv",
            file=str(tmp_path / "nonexistent.csv"),
            cols="x,y",
        )

        with pytest.raises(FileNotFoundError):
            load_coordinates(args)


class TestLoadCoordinatesBbox:
    """Tests for load_coordinates with bbox source."""

    def test_bbox_single_tile(self):
        """Test bbox converts tiles to UTM center points."""
        args = Namespace(
            command="bbox",
            bbox="350000,5600000,350500,5600500",  # Single tile bbox
        )

        coords = load_coordinates(args)

        # Should convert tile coords to UTM center points
        assert len(coords) == 1
        # Default grid (1km): 350*1000 + 500 = 350500
        assert coords[0] == (350500.0, 5600500.0)

    def test_bbox_multiple_tiles(self):
        """Test bbox spanning multiple tiles."""
        args = Namespace(
            command="bbox",
            bbox="350000,5600000,351500,5601500",  # 2x2 tile bbox
        )

        coords = load_coordinates(args)

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

        coords = load_coordinates(args)

        # Should detect as lat/lon and convert to UTM
        # Results should be in UTM range (not small lat/lon values)
        assert len(coords) >= 1
        x, y = coords[0]
        assert x > 100000  # UTM easting is large
        assert y > 5000000  # UTM northing is large


class TestLoadCoordinatesTiles:
    """Tests for load_coordinates with tiles source."""

    def test_tiles_single(self):
        """Test single tile coordinate."""
        args = Namespace(
            command="tiles",
            tiles=["350,5600"],
        )

        coords = load_coordinates(args)

        assert len(coords) == 1
        # Default grid: center at 350500, 5600500
        assert coords[0] == (350500.0, 5600500.0)

    def test_tiles_multiple(self):
        """Test multiple tile coordinates."""
        args = Namespace(
            command="tiles",
            tiles=["350,5600", "351,5601"],
        )

        coords = load_coordinates(args)

        assert len(coords) == 2
        assert coords[0] == (350500.0, 5600500.0)
        assert coords[1] == (351500.0, 5601500.0)

    def test_tiles_invalid_format_raises(self):
        """Test error when tile has wrong format."""
        args = Namespace(
            command="tiles",
            tiles=["invalid"],
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
