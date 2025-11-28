"""Tests for CSV input adapter."""

import pytest

from georaffer.inputs.csv import load_from_csv


class TestLoadFromCSV:
    """Tests for load_from_csv function."""

    def test_load_valid_csv(self, tmp_path):
        """Test loading valid CSV with default columns."""
        csv_file = tmp_path / "coords.csv"
        csv_file.write_text("x,y\n350000.5,5600000.5\n351000.0,5601000.0\n")

        coords = load_from_csv(str(csv_file))

        assert len(coords) == 2
        assert coords[0] == (350000.5, 5600000.5)
        assert coords[1] == (351000.0, 5601000.0)

    def test_load_custom_columns(self, tmp_path):
        """Test loading CSV with custom column names."""
        csv_file = tmp_path / "coords.csv"
        csv_file.write_text("easting,northing\n350000,5600000\n")

        coords = load_from_csv(str(csv_file), x_col="easting", y_col="northing")

        assert len(coords) == 1
        assert coords[0] == (350000.0, 5600000.0)

    def test_invalid_row_raises_error(self, tmp_path):
        """Test that invalid rows raise ValueError."""
        csv_file = tmp_path / "coords.csv"
        csv_file.write_text("x,y\n350000,5600000\ninvalid,data\n351000,5601000\n")

        with pytest.raises(ValueError, match="Row 3: Cannot parse coordinate"):
            load_from_csv(str(csv_file))

    def test_empty_csv_raises_error(self, tmp_path):
        """Test loading empty CSV raises ValueError."""
        csv_file = tmp_path / "coords.csv"
        csv_file.write_text("x,y\n")

        with pytest.raises(ValueError, match="No coordinates found"):
            load_from_csv(str(csv_file))

    def test_missing_column_raises_error(self, tmp_path):
        """Test missing column raises KeyError."""
        csv_file = tmp_path / "coords.csv"
        csv_file.write_text("x,z\n350000,100\n")

        with pytest.raises(KeyError, match="Column 'y' not found"):
            load_from_csv(str(csv_file))

    def test_file_not_found(self, tmp_path):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_from_csv(str(tmp_path / "nonexistent.csv"))
