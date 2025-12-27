"""Tests for NRW downloader."""

import xml.etree.ElementTree as ET
from unittest.mock import Mock

import pytest

from georaffer.downloaders.nrw import NRW_JP2_PATTERN, NRW_LAZ_PATTERN, NRWDownloader


class TestNRWDownloaderInit:
    """Tests for NRW downloader initialization."""

    def test_init_current_year(self, tmp_path):
        """Test initialization without year (current data)."""
        downloader = NRWDownloader(str(tmp_path))

        assert downloader.region_name == "NRW"
        assert "akt/dop" in downloader.jp2_feed_url
        assert "bdom50_las" in downloader.laz_feed_url

    def test_init_historical_year(self, tmp_path):
        """Test initialization with historical year range."""
        downloader = NRWDownloader(str(tmp_path), imagery_from=(2015, None))

        assert "hist_dop_2015" in downloader.jp2_feed_url
        assert "hist_dop_2015" in downloader.jp2_base_url

    def test_init_historical_year_range(self, tmp_path):
        """Test initialization with specific year range."""
        downloader = NRWDownloader(str(tmp_path), imagery_from=(2015, 2018))

        assert downloader._from_year == 2015
        assert downloader._to_year == 2018

    def test_init_rejects_old_year(self, tmp_path):
        """Test that years before 2010 are rejected."""
        with pytest.raises(ValueError, match="Year 2009 not supported"):
            NRWDownloader(str(tmp_path), imagery_from=(2009, None))

    def test_init_accepts_2010(self, tmp_path):
        """Test that year 2010 is accepted."""
        downloader = NRWDownloader(str(tmp_path), imagery_from=(2010, None))
        assert "hist_dop_2010" in downloader.jp2_feed_url


class TestNRWHistoricFiltering:
    """Tests for historical feed filtering logic."""

    def test_filters_current_feed_by_year_range(self, tmp_path, monkeypatch):
        """Current feed tiles outside the requested range should be skipped."""
        downloader = NRWDownloader(str(tmp_path), imagery_from=(2018, 2018))

        monkeypatch.setattr(NRWDownloader, "HISTORIC_YEARS", [])
        monkeypatch.setattr(downloader, "_available_historic_years", lambda: [])
        monkeypatch.setattr(downloader, "_fetch_and_parse_feed", lambda *args, **kwargs: {})

        def fake_parse(_session, _feed_url, _base_url):
            return {
                (350, 5600): ("https://example.com/dop10rgbi_32_350_5600_1_nw_2018.jp2", 2018),
                (351, 5600): ("https://example.com/dop10rgbi_32_351_5600_1_nw_2021.jp2", 2021),
            }

        monkeypatch.setattr(downloader, "_parse_jp2_feed_with_year", fake_parse)

        jp2_tiles, laz_tiles = downloader.get_available_tiles()

        assert jp2_tiles == {
            (350, 5600): "https://example.com/dop10rgbi_32_350_5600_1_nw_2018.jp2"
        }
        assert laz_tiles == {}
        assert downloader.total_jp2_count == 1


class TestNRWHistoricYearsDiscovery:
    def test_discovers_upper_years_from_provider_index(self, tmp_path, monkeypatch):
        """Historic year list should be discovered dynamically from the provider index."""
        downloader = NRWDownloader(str(tmp_path), imagery_from=(2023, None))

        # Fake provider index.xml includes 2024
        index_xml = b"""<?xml version='1.0' encoding='UTF-8'?>
<opengeodata>
  <folders>
    <folder name='hist_dop_2010'/>
    <folder name='hist_dop_2023'/>
    <folder name='hist_dop_2024'/>
  </folders>
</opengeodata>
"""

        # Mock GET for provider index only; any other GET should not be hit in this unit test.
        class Resp:
            def __init__(self, content: bytes):
                self.content = content

            def raise_for_status(self):
                return None

        def fake_get(url, *args, **kwargs):
            if url == downloader.HISTORIC_INDEX_URL:
                return Resp(index_xml)
            raise AssertionError(f"Unexpected GET in unit test: {url}")

        downloader._session.get = fake_get  # type: ignore[method-assign]

        years = downloader._available_historic_years()
        assert years == [2010, 2023, 2024]


class TestNRWUtmToGridCoords:
    """Tests for UTM to grid coordinate conversion."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return NRWDownloader(str(tmp_path))

    def test_simple_conversion(self, downloader):
        """Test basic UTM to grid conversion."""
        jp2_coords, laz_coords = downloader.utm_to_grid_coords(350500, 5600500)

        assert jp2_coords == (350, 5600)
        assert laz_coords == (350, 5600)

    def test_edge_of_tile(self, downloader):
        """Test coordinates at tile edge."""
        jp2_coords, laz_coords = downloader.utm_to_grid_coords(350000, 5600000)

        assert jp2_coords == (350, 5600)

    def test_jp2_laz_same_grid(self, downloader):
        """Test that NRW uses same grid for both types."""
        jp2_coords, laz_coords = downloader.utm_to_grid_coords(351234, 5601567)

        assert jp2_coords == laz_coords


class TestNRWFilenamePatterns:
    """Tests for NRW filename regex patterns."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("dop10rgbi_32_350_5600_1_nw_2021.jp2", (350, 5600, 2021)),
            ("dop10rgbi_32_400_5700_1_nw_2020.jp2", (400, 5700, 2020)),
            ("dop10rgbi_32_999_9999_9_nw_2015.jp2", (999, 9999, 2015)),
        ],
    )
    def test_jp2_pattern_valid(self, filename, expected):
        """Test JP2 pattern matches valid filenames."""
        match = NRW_JP2_PATTERN.match(filename)
        assert match is not None
        assert int(match.group(1)) == expected[0]
        assert int(match.group(2)) == expected[1]
        assert int(match.group(3)) == expected[2]

    @pytest.mark.parametrize(
        "filename",
        [
            "dop10rgbi_32_350_5600_1_nw.jp2",  # Missing year
            "dop20rgb_32_350_5600_1_nw_2021.jp2",  # Wrong prefix
            "dop10rgbi_32_350_5600_1_rp_2021.jp2",  # Wrong region
            "random.jp2",
        ],
    )
    def test_jp2_pattern_invalid(self, filename):
        """Test JP2 pattern rejects invalid filenames."""
        match = NRW_JP2_PATTERN.match(filename)
        assert match is None

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("bdom50_32350_5600_1_nw_2025.laz", (350, 5600, 2025)),
            ("bdom50_32400_5700_1_nw_2024.laz", (400, 5700, 2024)),
            ("bdom50_32_350_5600_1_nw_2025.laz", (350, 5600, 2025)),  # With underscore after 32
        ],
    )
    def test_laz_pattern_valid(self, filename, expected):
        """Test LAZ pattern matches valid filenames."""
        match = NRW_LAZ_PATTERN.match(filename)
        assert match is not None
        assert int(match.group(1)) == expected[0]
        assert int(match.group(2)) == expected[1]
        assert int(match.group(3)) == expected[2]


class TestNRWParseJP2Feed:
    """Tests for JP2 feed parsing."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return NRWDownloader(str(tmp_path))

    def test_parse_simple_feed(self, downloader):
        """Test parsing simple JP2 feed."""
        xml = """<index>
            <file name="dop10rgbi_32_350_5600_1_nw_2021.jp2"/>
            <file name="dop10rgbi_32_351_5600_1_nw_2021.jp2"/>
        </index>"""
        root = ET.fromstring(xml)
        mock_session = Mock()

        tiles = downloader._parse_jp2_feed(mock_session, root)

        assert len(tiles) == 2
        assert (350, 5600) in tiles
        assert (351, 5600) in tiles

    def test_parse_feed_with_epsg_subfolder(self, downloader):
        """Test parsing feed that has EPSG subfolder."""
        xml = """<index>
            <folder name="epsg_25832"/>
        </index>"""
        root = ET.fromstring(xml)

        # Mock session to return subfolder content
        mock_session = Mock()
        mock_response = Mock()
        mock_response.content = b"""<index>
            <file name="dop10rgbi_32_350_5600_1_nw_2015.jp2"/>
        </index>"""
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response

        tiles = downloader._parse_jp2_feed(mock_session, root)

        assert len(tiles) == 1
        assert "epsg_25832" in tiles[(350, 5600)]

    def test_parse_feed_raises_on_invalid_filename(self, downloader):
        """Test that invalid filenames raise ValueError."""
        xml = """<index>
            <file name="invalid_format.jp2"/>
        </index>"""
        root = ET.fromstring(xml)
        mock_session = Mock()

        with pytest.raises(ValueError, match="doesn't match pattern"):
            downloader._parse_jp2_feed(mock_session, root)


class TestNRWParseLAZFeed:
    """Tests for LAZ feed parsing."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return NRWDownloader(str(tmp_path))

    def test_parse_simple_feed(self, downloader):
        """Test parsing simple LAZ feed."""
        xml = """<index>
            <file name="bdom50_32350_5600_1_nw_2025.laz"/>
            <file name="bdom50_32351_5600_1_nw_2025.laz"/>
        </index>"""
        root = ET.fromstring(xml)
        mock_session = Mock()

        tiles = downloader._parse_laz_feed(mock_session, root)

        assert len(tiles) == 2
        assert (350, 5600) in tiles
        assert (351, 5600) in tiles

    def test_parse_feed_raises_on_invalid_filename(self, downloader):
        """Test that invalid filenames raise ValueError."""
        xml = """<index>
            <file name="wrong_format.laz"/>
        </index>"""
        root = ET.fromstring(xml)
        mock_session = Mock()

        with pytest.raises(ValueError, match="doesn't match pattern"):
            downloader._parse_laz_feed(mock_session, root)

    def test_parse_feed_skips_non_laz(self, downloader):
        """Test that non-LAZ files are skipped."""
        xml = """<index>
            <file name="bdom50_32350_5600_1_nw_2025.laz"/>
            <file name="readme.txt"/>
            <file name="metadata.xml"/>
        </index>"""
        root = ET.fromstring(xml)
        mock_session = Mock()

        tiles = downloader._parse_laz_feed(mock_session, root)

        assert len(tiles) == 1
