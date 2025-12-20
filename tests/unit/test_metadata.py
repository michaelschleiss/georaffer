"""Tests for metadata module."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from georaffer.metadata import add_provenance_to_geotiff, create_provenance_csv, get_wms_metadata


@pytest.fixture(autouse=True)
def enable_wms_for_tests():
    """Ensure WMS lookups are enabled for metadata tests (may be disabled by other test modules)."""
    original = os.environ.get("GEORAFFER_DISABLE_WMS")
    os.environ.pop("GEORAFFER_DISABLE_WMS", None)
    yield
    if original is not None:
        os.environ["GEORAFFER_DISABLE_WMS"] = original


class TestGetWmsMetadata:
    """Tests for get_wms_metadata function."""

    def test_returns_none_for_non_nrw(self):
        """Test that non-NRW regions return None."""
        result = get_wms_metadata(362000, 5604000, region="RLP")
        assert result is None

    def test_parses_date_from_response(self):
        """Test parsing acquisition date from WMS response."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.text = "Bildflugdatum = '15.06.2021'"
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response

        result = get_wms_metadata(350000, 5600000, region="NRW", session=mock_session)

        assert result is not None
        assert result["acquisition_date"] == "2021-06-15"
        assert result["metadata_source"] == "WMS GetFeatureInfo"

    def test_returns_none_when_no_date_found(self):
        """Test returns None when no date in response."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.text = "No data found for this location"
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response

        result = get_wms_metadata(350000, 5600000, region="NRW", session=mock_session)

        assert result is None

    def test_uses_historical_wms_for_old_years(self):
        """Test that historical WMS URL is used for years < 2024."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.text = "Bildflugdatum = '15.06.2015'"
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response

        get_wms_metadata(350000, 5600000, region="NRW", year=2015, session=mock_session)

        call_args = mock_session.get.call_args
        assert "hist_dop" in call_args[0][0]

    def test_uses_current_wms_for_current_year(self):
        """Test that current WMS URL is used for year >= 2024 or None."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.text = "Bildflugdatum = '15.06.2024'"
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response

        get_wms_metadata(350000, 5600000, region="NRW", year=None, session=mock_session)

        call_args = mock_session.get.call_args
        assert "wms_nw_dop" in call_args[0][0]
        assert "hist" not in call_args[0][0]

    def test_retries_on_error(self):
        """Test retry logic on request failure."""
        import requests

        mock_session = Mock()

        # First call fails, second succeeds
        mock_response = Mock()
        mock_response.text = "Bildflugdatum = '15.06.2021'"
        mock_response.raise_for_status = Mock()

        mock_session.get.side_effect = [requests.RequestException("Network error"), mock_response]

        with patch("georaffer.metadata.time.sleep"):
            with patch("georaffer.metadata.MAX_RETRIES", 3):
                result = get_wms_metadata(350000, 5600000, region="NRW", session=mock_session)

        assert result is not None
        assert mock_session.get.call_count == 2

    def test_raises_after_max_retries(self):
        """Test RuntimeError after exhausting retries."""
        import requests

        mock_session = Mock()
        mock_session.get.side_effect = requests.RequestException("Network error")

        with patch("georaffer.metadata.time.sleep"):
            with patch("georaffer.metadata.MAX_RETRIES", 2):
                with pytest.raises(RuntimeError, match="Failed to get WMS metadata"):
                    get_wms_metadata(350000, 5600000, region="NRW", session=mock_session)

    def test_handles_iso_date_format(self):
        """Test handling of ISO date format (no conversion needed)."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.text = "Bildflugdatum = '2021-06-15'"
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response

        result = get_wms_metadata(350000, 5600000, region="NRW", session=mock_session)

        assert result["acquisition_date"] == "2021-06-15"

    def test_returns_none_when_all_dates_invalid(self):
        """Test invalid dates are ignored and return None when no valid dates remain."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.text = "Bildflugdatum = '1983-00-00'\nBildflugdatum = 'not-a-date'"
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response

        result = get_wms_metadata(350000, 5600000, region="NRW", session=mock_session)

        assert result is None


class TestAddProvenanceToGeotiff:
    """Tests for add_provenance_to_geotiff function."""

    def test_adds_tags_to_file(self, tmp_path):
        """Test adding metadata tags to GeoTIFF."""
        metadata = {
            "acquisition_date": "2021-06-15",
            "source_region": "NRW",
            "source_file": "dop10rgbi_32_350_5600_1_nw_2021.jp2",
            "metadata_source": "WMS GetFeatureInfo",
        }

        with patch("georaffer.metadata.rasterio.open") as mock_open:
            mock_dst = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_dst

            result = add_provenance_to_geotiff(str(tmp_path / "test.tif"), metadata)

        assert result is True
        mock_dst.update_tags.assert_called_once()
        call_kwargs = mock_dst.update_tags.call_args[1]
        assert call_kwargs["ACQUISITION_DATE"] == "2021-06-15"
        assert call_kwargs["SOURCE_REGION"] == "NRW"
        assert "PROCESSING_DATE" in call_kwargs

    def test_handles_partial_metadata(self, tmp_path):
        """Test with partial metadata (only some fields)."""
        metadata = {"source_region": "RLP"}

        with patch("georaffer.metadata.rasterio.open") as mock_open:
            mock_dst = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_dst

            result = add_provenance_to_geotiff(str(tmp_path / "test.tif"), metadata)

        assert result is True
        call_kwargs = mock_dst.update_tags.call_args[1]
        assert "ACQUISITION_DATE" not in call_kwargs
        assert call_kwargs["SOURCE_REGION"] == "RLP"

    def test_raises_on_error(self, tmp_path):
        """Test raises RuntimeError on error (fail-fast)."""
        with patch("georaffer.metadata.rasterio.open") as mock_open:
            mock_open.side_effect = Exception("File error")

            with pytest.raises(RuntimeError, match="Failed to add provenance"):
                add_provenance_to_geotiff(str(tmp_path / "test.tif"), {})


class TestCreateProvenanceCsv:
    """Tests for create_provenance_csv function."""

    def test_creates_csv_file(self, tmp_path):
        """Test CSV file creation."""
        metadata_list = [
            {
                "processed_file": "nrw_32_350000_5600000_2021.tif",
                "source_file": "dop10rgbi_32_350_5600_1_nw_2021.jp2",
                "source_region": "NRW",
                "grid_x": 350,
                "grid_y": 5600,
                "year": 2021,
                "acquisition_date": "2021-06-15",
                "file_type": "jp2",
            },
            {
                "processed_file": "nrw_32_351000_5600000_2021.tif",
                "source_region": "NRW",
                "grid_x": 351,
                "grid_y": 5600,
            },
        ]

        output_csv = str(tmp_path / "provenance.csv")
        result = create_provenance_csv(metadata_list, output_csv)

        assert result is True
        assert (tmp_path / "provenance.csv").exists()

        # Verify content
        with open(output_csv) as f:
            content = f.read()
            assert "processed_file" in content  # header
            assert "nrw_32_350000_5600000_2021.tif" in content
            assert "nrw_32_351000_5600000_2021.tif" in content

    def test_handles_empty_list(self, tmp_path):
        """Test with empty metadata list."""
        output_csv = str(tmp_path / "empty.csv")
        result = create_provenance_csv([], output_csv)

        assert result is True
        assert (tmp_path / "empty.csv").exists()

    def test_raises_on_error(self, tmp_path):
        """Test raises IOError on write error (fail-fast)."""
        from unittest.mock import patch

        # Mock open to raise a permission error
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(IOError, match="Failed to create CSV"):
                create_provenance_csv([{"test": "data"}], str(tmp_path / "file.csv"))

    def test_ignores_extra_fields(self, tmp_path):
        """Test that extra fields in metadata are ignored."""
        metadata_list = [
            {
                "processed_file": "test.tif",
                "extra_field": "should be ignored",
                "another_extra": 123,
            }
        ]

        output_csv = str(tmp_path / "provenance.csv")
        result = create_provenance_csv(metadata_list, output_csv)

        assert result is True

        with open(output_csv) as f:
            content = f.read()
            assert "extra_field" not in content
            assert "another_extra" not in content
