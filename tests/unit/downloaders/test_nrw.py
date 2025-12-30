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
        from georaffer.downloaders.base import Catalog

        downloader = NRWDownloader(str(tmp_path), imagery_from=(2018, 2018))

        # Mock catalog with tiles from different years (new tile_info format)
        fake_catalog = Catalog(
            image_tiles={
                (350, 5600): {2018: {"url": "https://example.com/tile_2018.jp2", "acquisition_date": None}},
                (351, 5600): {2021: {"url": "https://example.com/tile_2021.jp2", "acquisition_date": None}},
            },
            dsm_tiles={},
        )
        monkeypatch.setattr(downloader, "build_catalog", lambda: fake_catalog)

        jp2_tiles, laz_tiles = downloader.get_filtered_tile_urls()

        # Only 2018 tile should be included (2021 outside range)
        assert jp2_tiles == {(350, 5600): "https://example.com/tile_2018.jp2"}
        assert laz_tiles == {}
        assert downloader.total_image_count == 1


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
        xml = b"""<index>
            <file name="dop10rgbi_32_350_5600_1_nw_2021.jp2"/>
            <file name="dop10rgbi_32_351_5600_1_nw_2021.jp2"/>
        </index>"""
        mock_response = Mock()
        mock_response.content = xml
        mock_response.raise_for_status = Mock()
        mock_session = Mock()
        mock_session.get.return_value = mock_response

        tiles = downloader._parse_jp2_feed_with_year(
            mock_session, "https://example.com/index.xml", "https://example.com/"
        )

        assert len(tiles) == 2
        assert (350, 5600) in tiles
        assert (351, 5600) in tiles
        assert tiles[(350, 5600)][1] == 2021  # Year

    def test_parse_feed_raises_on_invalid_filename(self, downloader):
        """Test that invalid filenames raise ValueError."""
        xml = b"""<index>
            <file name="invalid_format.jp2"/>
        </index>"""
        mock_response = Mock()
        mock_response.content = xml
        mock_response.raise_for_status = Mock()
        mock_session = Mock()
        mock_session.get.return_value = mock_response

        with pytest.raises(ValueError, match="doesn't match pattern"):
            downloader._parse_jp2_feed_with_year(
                mock_session, "https://example.com/index.xml", "https://example.com/"
            )


class TestNRWParseLAZTiles:
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

        tiles = downloader._parse_laz_tiles(root)

        assert len(tiles) == 2
        assert (350, 5600) in tiles
        assert (351, 5600) in tiles
        # Verify tile_info format
        assert tiles[(350, 5600)]["url"].endswith(".laz")
        assert "acquisition_date" in tiles[(350, 5600)]

    def test_parse_feed_raises_on_invalid_filename(self, downloader):
        """Test that invalid filenames raise ValueError."""
        xml = """<index>
            <file name="wrong_format.laz"/>
        </index>"""
        root = ET.fromstring(xml)

        with pytest.raises(ValueError, match="doesn't match pattern"):
            downloader._parse_laz_tiles(root)

    def test_parse_feed_skips_non_laz(self, downloader):
        """Test that non-LAZ files are skipped."""
        xml = """<index>
            <file name="bdom50_32350_5600_1_nw_2025.laz"/>
            <file name="readme.txt"/>
            <file name="metadata.xml"/>
        </index>"""
        root = ET.fromstring(xml)

        tiles = downloader._parse_laz_tiles(root)

        assert len(tiles) == 1


class TestNRWLoadCatalog:
    """Tests for _load_catalog parallel execution."""

    def test_parallel_execution_processes_all_years(self, tmp_path, monkeypatch):
        """Verify parallel_map processes all historic years without dropping any."""
        from georaffer.downloaders import feeds
        downloader = NRWDownloader(str(tmp_path), quiet=True)

        # Track which years were processed
        processed_years = []

        def mock_parse(session, feed_url, base_url):
            # Extract year from URL
            import re
            match = re.search(r"hist_dop_(\d{4})", feed_url)
            if match:
                year = int(match.group(1))
                processed_years.append(year)
                return {(350, 5600): (f"http://example.com/{year}.jp2", year)}
            # Current feed
            return {(350, 5600): ("http://example.com/current.jp2", 2025)}

        # Mock the LAZ feed fetch to return empty XML
        def mock_fetch_xml(session, url):
            return ET.fromstring("<index></index>")

        monkeypatch.setattr(downloader, "_parse_jp2_feed_with_year", mock_parse)
        monkeypatch.setattr(feeds, "fetch_xml_feed", mock_fetch_xml)

        catalog = downloader._load_catalog()

        # Verify all historic years were processed
        expected_years = list(downloader.HISTORIC_YEARS)
        assert sorted(processed_years) == sorted(expected_years), (
            f"Missing years: {set(expected_years) - set(processed_years)}"
        )

    def test_404_errors_counted_not_raised(self, tmp_path, monkeypatch):
        """Verify 404 errors are counted but don't abort catalog build."""
        import requests
        from georaffer.downloaders import feeds
        downloader = NRWDownloader(str(tmp_path), quiet=True)

        call_count = {"total": 0, "errors": 0}

        def mock_parse(session, feed_url, base_url):
            call_count["total"] += 1
            if "2015" in feed_url or "2016" in feed_url:
                call_count["errors"] += 1
                resp = Mock()
                resp.status_code = 404
                raise requests.HTTPError(response=resp)
            return {(350, 5600): ("http://example.com/tile.jp2", 2020)}

        def mock_fetch_xml(session, url):
            return ET.fromstring("<index></index>")

        monkeypatch.setattr(downloader, "_parse_jp2_feed_with_year", mock_parse)
        monkeypatch.setattr(feeds, "fetch_xml_feed", mock_fetch_xml)

        # Should not raise despite 404 errors
        catalog = downloader._load_catalog()

        # Should have processed all years (including failed ones)
        assert call_count["total"] > 0
        assert call_count["errors"] == 2

    def test_non_404_errors_propagate(self, tmp_path, monkeypatch):
        """Verify non-404 HTTP errors are not silently swallowed."""
        import requests
        from georaffer.downloaders import feeds
        downloader = NRWDownloader(str(tmp_path), quiet=True)

        def mock_parse(session, feed_url, base_url):
            if "2015" in feed_url:
                resp = Mock()
                resp.status_code = 500
                raise requests.HTTPError(response=resp)
            return {(350, 5600): ("http://example.com/tile.jp2", 2020)}

        def mock_fetch_xml(session, url):
            return ET.fromstring("<index></index>")

        monkeypatch.setattr(downloader, "_parse_jp2_feed_with_year", mock_parse)
        monkeypatch.setattr(feeds, "fetch_xml_feed", mock_fetch_xml)

        with pytest.raises(requests.HTTPError):
            downloader._load_catalog()
