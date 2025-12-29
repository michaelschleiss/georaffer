"""Tests for BB (Brandenburg) downloader."""

from datetime import date
from unittest.mock import Mock, patch

import pytest

from georaffer.config import BB_BDOM_PATTERN, BB_DOP_PATTERN
from georaffer.downloaders.bb import BBDownloader


class TestBBDownloaderInit:
    """Tests for Brandenburg downloader initialization."""

    def test_init(self, tmp_path):
        """Test basic initialization."""
        downloader = BBDownloader(str(tmp_path))

        assert downloader.region_name == "BB"
        assert "data.geobasis-bb.de" in downloader.jp2_feed_url
        assert "dop/rgbi_tif" in downloader.jp2_feed_url
        assert "bdom/tif" in downloader.laz_feed_url

    def test_utm_zone(self, tmp_path):
        """Test that BB uses UTM Zone 33."""
        downloader = BBDownloader(str(tmp_path))
        assert downloader.UTM_ZONE == 33


class TestBBUtmToGridCoords:
    """Tests for UTM to grid coordinate conversion."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BBDownloader(str(tmp_path))

    def test_simple_conversion(self, downloader):
        """Test basic UTM to grid conversion."""
        jp2_coords, laz_coords = downloader.utm_to_grid_coords(350500, 5800500)

        assert jp2_coords == (350, 5800)
        assert laz_coords == (350, 5800)

    def test_edge_of_tile(self, downloader):
        """Test coordinates at tile edge."""
        jp2_coords, laz_coords = downloader.utm_to_grid_coords(350000, 5800000)

        assert jp2_coords == (350, 5800)

    def test_jp2_laz_same_grid(self, downloader):
        """Test that BB uses same grid for both types."""
        jp2_coords, laz_coords = downloader.utm_to_grid_coords(351234, 5801567)

        assert jp2_coords == laz_coords


class TestBBFilenamePatterns:
    """Tests for BB bDOM and DOP filename regex patterns."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("bdom_33250-5886.zip", (250, 5886)),
            ("bdom_33350-5700.zip", (350, 5700)),
            ("bdom_33400-5900.zip", (400, 5900)),
        ],
    )
    def test_bdom_pattern_valid(self, filename, expected):
        """Test bDOM pattern matches valid filenames."""
        match = BB_BDOM_PATTERN.match(filename)
        assert match is not None
        east_code = match.group(1)
        grid_x = int(east_code[2:])
        grid_y = int(match.group(2))
        assert grid_x == expected[0]
        assert grid_y == expected[1]

    @pytest.mark.parametrize(
        "filename",
        [
            "bdom_3325-5886.zip",
            "bdom_332500-5886.zip",
            "random.zip",
            "dop_33250-5886.zip",
        ],
    )
    def test_bdom_pattern_invalid(self, filename):
        """Test bDOM pattern rejects invalid filenames."""
        match = BB_BDOM_PATTERN.match(filename)
        assert match is None

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("dop_33250-5886.zip", (250, 5886)),
            ("dop_33350-5700.zip", (350, 5700)),
            ("dop_33400-5900.zip", (400, 5900)),
        ],
    )
    def test_dop_pattern_valid(self, filename, expected):
        """Test DOP pattern matches valid filenames."""
        match = BB_DOP_PATTERN.match(filename)
        assert match is not None
        east_code = match.group(1)
        grid_x = int(east_code[2:])
        grid_y = int(match.group(2))
        assert grid_x == expected[0]
        assert grid_y == expected[1]

    @pytest.mark.parametrize(
        "filename",
        [
            "dop_3325-5886.zip",
            "dop_332500-5886.zip",
            "random.zip",
            "bdom_33250-5886.zip",
        ],
    )
    def test_dop_pattern_invalid(self, filename):
        """Test DOP pattern rejects invalid filenames."""
        match = BB_DOP_PATTERN.match(filename)
        assert match is None


class TestBBParseSheetnr:
    """Tests for sheetnr parsing (OGC API format)."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BBDownloader(str(tmp_path))

    @pytest.mark.parametrize(
        "sheetnr,expected",
        [
            ("33250-5886", (250, 5886)),
            ("33350-5700", (350, 5700)),
            ("33400-5900", (400, 5900)),
        ],
    )
    def test_parse_valid_zone33(self, downloader, sheetnr, expected):
        """Test parsing sheetnr with correct UTM zone 33."""
        result = downloader._parse_sheetnr(sheetnr)
        assert result == expected

    def test_parse_rejects_zone32(self, downloader):
        """Test that zone 32 sheetnr is rejected."""
        result = downloader._parse_sheetnr("32250-5886")
        assert result is None

    @pytest.mark.parametrize(
        "sheetnr",
        [
            "invalid",
            "3325-5886",  # Too short
            "332500-5886",  # Too long
            "",
        ],
    )
    def test_parse_invalid_format(self, downloader, sheetnr):
        """Test that invalid formats return None."""
        result = downloader._parse_sheetnr(sheetnr)
        assert result is None


class TestBBParseBdomListing:
    """Tests for bDOM HTML listing parsing."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BBDownloader(str(tmp_path))

    def test_parse_simple_listing(self, downloader):
        """Test parsing simple HTML listing."""
        html = """
        <html>
        <body>
        <a href="bdom_33250-5886.zip">bdom_33250-5886.zip</a>
        <a href="bdom_33251-5886.zip">bdom_33251-5886.zip</a>
        <a href="bdom_33250-5887.zip">bdom_33250-5887.zip</a>
        </body>
        </html>
        """
        tiles = downloader._parse_bdom_listing(html)

        assert len(tiles) == 3
        assert (250, 5886) in tiles
        assert (251, 5886) in tiles
        assert (250, 5887) in tiles

    def test_parse_listing_skips_zone32(self, downloader):
        """Test that zone 32 tiles are skipped."""
        html = """
        <html>
        <a href="bdom_33250-5886.zip">bdom_33250-5886.zip</a>
        <a href="bdom_32250-5886.zip">bdom_32250-5886.zip</a>
        </html>
        """
        tiles = downloader._parse_bdom_listing(html)

        assert len(tiles) == 1
        assert (250, 5886) in tiles

    def test_parse_listing_ignores_non_bdom(self, downloader):
        """Test that non-bdom files are ignored."""
        html = """
        <html>
        <a href="bdom_33250-5886.zip">bdom_33250-5886.zip</a>
        <a href="readme.txt">readme.txt</a>
        <a href="index.html">index.html</a>
        </html>
        """
        tiles = downloader._parse_bdom_listing(html)

        assert len(tiles) == 1

    def test_parse_listing_generates_correct_urls(self, downloader):
        """Test that parsed tiles have correct download URLs."""
        html = '<a href="bdom_33250-5886.zip">bdom_33250-5886.zip</a>'
        tiles = downloader._parse_bdom_listing(html)

        url = tiles[(250, 5886)]
        assert url == "https://data.geobasis-bb.de/geobasis/daten/bdom/tif/bdom_33250-5886.zip"


class TestBBParseOGCFeatures:
    """Tests for OGC API feature parsing."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BBDownloader(str(tmp_path))

    def test_parse_valid_features(self, downloader):
        """Test parsing valid OGC features."""
        features = [
            {"properties": {"sheetnr": "33250-5886", "creationdate": "2025-05-13"}},
            {"properties": {"sheetnr": "33251-5886", "creationdate": "2024-04-30"}},
        ]
        result = downloader._parse_ogc_features(features)

        assert len(result) == 2
        assert result[0] == ("33250-5886", date(2025, 5, 13))
        assert result[1] == ("33251-5886", date(2024, 4, 30))

    def test_parse_skips_invalid_dates(self, downloader):
        """Test that features with invalid dates are skipped."""
        features = [
            {"properties": {"sheetnr": "33250-5886", "creationdate": "invalid"}},
            {"properties": {"sheetnr": "33251-5886", "creationdate": "2024-04-30"}},
        ]
        result = downloader._parse_ogc_features(features)

        assert len(result) == 1
        assert result[0][0] == "33251-5886"

    def test_parse_skips_missing_fields(self, downloader):
        """Test that features with missing fields are skipped."""
        features = [
            {"properties": {"sheetnr": "33250-5886"}},  # Missing creationdate
            {"properties": {"creationdate": "2024-04-30"}},  # Missing sheetnr
            {"properties": {"sheetnr": "33251-5886", "creationdate": "2024-04-30"}},
        ]
        result = downloader._parse_ogc_features(features)

        assert len(result) == 1


class TestBBDsmFilename:
    """Tests for DSM filename extraction."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BBDownloader(str(tmp_path))

    def test_dsm_filename_from_zip_url(self, downloader):
        """Test keeping zip name from zip URL."""
        url = "https://data.geobasis-bb.de/geobasis/daten/bdom/tif/bdom_33250-5886.zip"
        result = downloader.dsm_filename_from_url(url)
        assert result == "bdom_33250-5886.zip"


class TestBBImageFilename:
    """Tests for DOP filename extraction."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BBDownloader(str(tmp_path))

    def test_image_filename_from_zip_url(self, downloader):
        """Test keeping zip name from zip URL."""
        url = "https://data.geobasis-bb.de/geobasis/daten/dop/rgbi_tif/dop_33250-5886.zip"
        result = downloader.image_filename_from_url(url)
        assert result == "dop_33250-5886.zip"
