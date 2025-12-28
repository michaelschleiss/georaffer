"""Tests for WMS imagery source."""

from unittest.mock import Mock, patch

import pytest

from georaffer.downloaders.wms import WMSImagerySource


class TestWMSImagerySourceInit:
    """Tests for WMS imagery source initialization."""

    def test_init_sets_base_url(self):
        """Test initialization sets base URL."""
        wms = WMSImagerySource(
            base_url="https://example.com/wms",
            rgb_layer_pattern="layer_{year}",
            info_layer_pattern="info_{year}",
        )
        assert wms.base_url == "https://example.com/wms"

    def test_init_sets_layer_patterns(self):
        """Test initialization sets layer patterns."""
        wms = WMSImagerySource(
            base_url="https://example.com/wms",
            rgb_layer_pattern="dop_{year}",
            info_layer_pattern="info_{year}",
        )
        assert wms.rgb_layer_pattern == "dop_{year}"
        assert wms.info_layer_pattern == "info_{year}"

    def test_init_sets_defaults(self):
        """Test initialization sets default values."""
        wms = WMSImagerySource(
            base_url="https://example.com/wms",
            rgb_layer_pattern="layer_{year}",
            info_layer_pattern="info_{year}",
        )
        assert wms.tile_size_m == 2000
        assert wms.resolution_m == 0.2
        assert wms.image_format == "image/tiff-lzw"
        assert wms.crs == "EPSG:25832"

    def test_init_accepts_custom_values(self):
        """Test initialization accepts custom values."""
        wms = WMSImagerySource(
            base_url="https://example.com/wms",
            rgb_layer_pattern="layer_{year}",
            info_layer_pattern="info_{year}",
            tile_size_m=1000,
            resolution_m=0.1,
            image_format="image/png",
            crs="EPSG:4326",
        )
        assert wms.tile_size_m == 1000
        assert wms.resolution_m == 0.1
        assert wms.image_format == "image/png"
        assert wms.crs == "EPSG:4326"


class TestWMSLayers:
    """Tests for layer name generation."""

    @pytest.fixture
    def wms(self):
        return WMSImagerySource(
            base_url="https://example.com/wms",
            rgb_layer_pattern="rp_dop20_rgb_{year}",
            info_layer_pattern="rp_dop20_info_{year}",
        )

    def test_rgb_layer_formats_year(self, wms):
        """Test RGB layer name includes year."""
        assert wms._rgb_layer(2020) == "rp_dop20_rgb_2020"
        assert wms._rgb_layer(1994) == "rp_dop20_rgb_1994"

    def test_info_layer_formats_year(self, wms):
        """Test info layer name includes year."""
        assert wms._info_layer(2020) == "rp_dop20_info_2020"
        assert wms._info_layer(1994) == "rp_dop20_info_1994"


class TestWMSGridToBbox:
    """Tests for grid to bbox conversion."""

    @pytest.fixture
    def wms(self):
        return WMSImagerySource(
            base_url="https://example.com/wms",
            rgb_layer_pattern="layer_{year}",
            info_layer_pattern="info_{year}",
            tile_size_m=2000,
        )

    def test_grid_to_bbox_basic(self, wms):
        """Test basic grid to bbox conversion."""
        # Grid coord 362km, 5604km -> 362000m, 5604000m
        bbox = wms._grid_to_bbox(362, 5604)

        assert bbox == (362000, 5604000, 364000, 5606000)

    def test_grid_to_bbox_zero(self, wms):
        """Test grid to bbox at origin."""
        bbox = wms._grid_to_bbox(0, 0)

        assert bbox == (0, 0, 2000, 2000)

    def test_grid_to_bbox_respects_tile_size(self):
        """Test grid to bbox uses correct tile size."""
        wms = WMSImagerySource(
            base_url="https://example.com/wms",
            rgb_layer_pattern="layer_{year}",
            info_layer_pattern="info_{year}",
            tile_size_m=1000,  # 1km tiles
        )
        bbox = wms._grid_to_bbox(350, 5600)

        assert bbox == (350000, 5600000, 351000, 5601000)


class TestWMSTilePixels:
    """Tests for tile pixel size calculation."""

    def test_tile_pixels_default(self):
        """Test default 2km at 0.2m = 10000 pixels."""
        wms = WMSImagerySource(
            base_url="https://example.com/wms",
            rgb_layer_pattern="layer_{year}",
            info_layer_pattern="info_{year}",
            tile_size_m=2000,
            resolution_m=0.2,
        )
        assert wms._tile_pixels() == 10000

    def test_tile_pixels_1km_at_0_1m(self):
        """Test 1km at 0.1m = 10000 pixels."""
        wms = WMSImagerySource(
            base_url="https://example.com/wms",
            rgb_layer_pattern="layer_{year}",
            info_layer_pattern="info_{year}",
            tile_size_m=1000,
            resolution_m=0.1,
        )
        assert wms._tile_pixels() == 10000


class TestWMSGetTileUrl:
    """Tests for GetMap URL generation."""

    @pytest.fixture
    def wms(self):
        return WMSImagerySource(
            base_url="https://geo4.service24.rlp.de/wms/rp_hkdop20.fcgi",
            rgb_layer_pattern="rp_dop20_rgb_{year}",
            info_layer_pattern="rp_dop20_info_{year}",
            tile_size_m=2000,
            resolution_m=0.2,
        )

    def test_get_tile_url_contains_required_params(self, wms):
        """Test URL contains all required WMS parameters."""
        url = wms.get_tile_url(2020, 362, 5604)

        assert "SERVICE=WMS" in url
        assert "VERSION=1.1.1" in url
        assert "REQUEST=GetMap" in url
        assert "LAYERS=rp_dop20_rgb_2020" in url
        assert "STYLES=" in url
        assert "SRS=EPSG:25832" in url
        assert "FORMAT=image/tiff-lzw" in url
        assert "WIDTH=10000" in url
        assert "HEIGHT=10000" in url

    def test_get_tile_url_contains_bbox(self, wms):
        """Test URL contains correct BBOX."""
        url = wms.get_tile_url(2020, 362, 5604)

        # BBOX for grid 362,5604 with 2km tiles
        assert "BBOX=362000,5604000,364000,5606000" in url


class TestWMSCheckCoverage:
    """Tests for coverage checking via GetFeatureInfo."""

    @pytest.fixture
    def wms(self):
        return WMSImagerySource(
            base_url="https://example.com/wms",
            rgb_layer_pattern="layer_{year}",
            info_layer_pattern="info_{year}",
        )

    def test_check_coverage_returns_none_on_no_results(self, wms):
        """Test coverage check returns None when no results."""
        mock_response = Mock()
        mock_response.text = "Search returned no results"
        mock_response.raise_for_status = Mock()

        with patch.object(wms._session, "get", return_value=mock_response):
            result = wms.check_coverage(2020, 362, 5604)

        assert result is None

    def test_check_coverage_parses_bildflugdatum_new_format(self, wms):
        """Test coverage check parses newer kachelname format (no underscore after 32)."""
        mock_response = Mock()
        mock_response.text = """
        Some header info
        bildflugdatum = '2020-08-07'
        kachelname = 'dop_32362_5604'
        """
        mock_response.raise_for_status = Mock()

        with patch.object(wms._session, "get", return_value=mock_response):
            result = wms.check_coverage(2020, 362, 5604)

        assert result is not None
        assert result["acquisition_date"] == "2020-08-07"
        assert result["tile_name"] == "dop_32362_5604"

    def test_check_coverage_parses_bildflugdatum_old_format(self, wms):
        """Test coverage check parses older kachelname format (with underscore after 32)."""
        mock_response = Mock()
        mock_response.text = """
        Some header info
        bildflugdatum = '2020-08-07'
        kachelname = 'dop_32_362_5604'
        """
        mock_response.raise_for_status = Mock()

        with patch.object(wms._session, "get", return_value=mock_response):
            result = wms.check_coverage(2020, 362, 5604)

        assert result is not None
        assert result["acquisition_date"] == "2020-08-07"
        assert result["tile_name"] == "dop_32_362_5604"

    def test_check_coverage_raises_on_network_error(self, wms):
        """Test coverage check raises RuntimeError after retries on network error."""
        import requests

        with patch.object(wms._session, "get", side_effect=requests.RequestException("timeout")):
            with patch("georaffer.downloaders.wms.WMS_COVERAGE_RETRIES", 2):
                with patch("georaffer.downloaders.wms.time.sleep"):
                    with pytest.raises(RuntimeError, match="WMS coverage check failed"):
                        wms.check_coverage(2020, 362, 5604)

    def test_check_coverage_returns_none_when_missing_bildflugdatum(self, wms):
        """Test coverage check returns None when both bildflugdatum and erstellung are missing."""
        mock_response = Mock()
        mock_response.text = """
        Some header info
        kachelname = 'dop_32362_5604'
        other_field = 'value'
        """
        mock_response.raise_for_status = Mock()

        with patch.object(wms._session, "get", return_value=mock_response):
            result = wms.check_coverage(2020, 362, 5604)

        assert result is None

    def test_check_coverage_returns_none_when_missing_kachelname(self, wms):
        """Test coverage check returns None when kachelname is missing."""
        mock_response = Mock()
        mock_response.text = """
        Some header info
        bildflugdatum = '2020-08-07'
        other_field = 'value'
        """
        mock_response.raise_for_status = Mock()

        with patch.object(wms._session, "get", return_value=mock_response):
            result = wms.check_coverage(2020, 362, 5604)

        assert result is None

    def test_check_coverage_returns_none_when_kachelname_wrong_coords(self, wms):
        """Test coverage check returns None when kachelname doesn't match query coords."""
        mock_response = Mock()
        mock_response.text = """
        Some header info
        bildflugdatum = '2020-08-07'
        kachelname = 'dop_32999_9999'
        """
        mock_response.raise_for_status = Mock()

        # Query for 362,5604 but response says tile is 999,9999
        with patch.object(wms._session, "get", return_value=mock_response):
            result = wms.check_coverage(2020, 362, 5604)

        assert result is None

    def test_check_coverage_uses_erstellung_when_bildflugdatum_empty(self, wms):
        """Test coverage check falls back to erstellung when bildflugdatum is empty."""
        mock_response = Mock()
        mock_response.text = """
        Some header info
        bildflugdatum = ''
        kachelname = 'dop_32362_5604'
        erstellung = '2022-06-14'
        """
        mock_response.raise_for_status = Mock()

        with patch.object(wms._session, "get", return_value=mock_response):
            result = wms.check_coverage(2022, 362, 5604)

        assert result is not None
        assert result["acquisition_date"] == "2022-06-14"
        assert result["tile_name"] == "dop_32362_5604"

    def test_check_coverage_multi_parses_multiple_years(self, wms):
        """Multi-year query should return coverage for multiple years from one response."""
        mock_response = Mock()
        mock_response.text = """
        GetFeatureInfo results:

        Layer 'rp_dop20_info_2020'
          Feature 1:
        Layer 'Metadaten_27'
          Feature 1:
            bildflugdatum = '2020-08-07'
            kachelname = 'dop_32_380_5540'

        Layer 'rp_dop20_info_2022'
          Feature 2:
        Layer 'Metadaten_29'
          Feature 2:
            bildflugdatum = ''
            kachelname = 'dop_32380_5540'
            erstellung = '2022-06-14'
        """
        mock_response.raise_for_status = Mock()

        with patch.object(wms._session, "get", return_value=mock_response):
            result = wms.check_coverage_multi([2020, 2021, 2022], 380, 5540)

        assert set(result) == {2020, 2022}
        assert result[2020]["acquisition_date"] == "2020-08-07"
        assert result[2020]["tile_name"] == "dop_32_380_5540"
        assert result[2022]["acquisition_date"] == "2022-06-14"
        assert result[2022]["tile_name"] == "dop_32380_5540"

    def test_check_coverage_multi_respects_coords(self, wms):
        """Multi-year query should ignore results if kachelname doesn't match coords."""
        mock_response = Mock()
        mock_response.text = """
        GetFeatureInfo results:
        Layer 'rp_dop20_info_2020'
        Layer 'Metadaten_27'
          Feature 1:
            bildflugdatum = '2020-08-07'
            kachelname = 'dop_32_999_9999'
        """
        mock_response.raise_for_status = Mock()

        with patch.object(wms._session, "get", return_value=mock_response):
            result = wms.check_coverage_multi([2020], 380, 5540)

        assert result == {}


class TestWMSOutputFilename:
    """Tests for output filename generation."""

    @pytest.fixture
    def wms(self):
        return WMSImagerySource(
            base_url="https://example.com/wms",
            rgb_layer_pattern="layer_{year}",
            info_layer_pattern="info_{year}",
        )

    def test_output_filename_format(self, wms):
        """Test output filename matches expected format."""
        filename = wms.output_filename(362, 5604, 2020)

        assert filename == "dop20rgb_32_362_5604_2_rp_2020.tif"

    def test_output_filename_different_year(self, wms):
        """Test output filename changes with year."""
        filename = wms.output_filename(380, 5540, 1994)

        assert filename == "dop20rgb_32_380_5540_2_rp_1994.tif"
