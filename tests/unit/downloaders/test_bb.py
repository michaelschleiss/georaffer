"""Tests for BB (Brandenburg) downloader."""

from datetime import date

import pytest

from georaffer.config import BB_BDOM_PATTERN, BB_DOP_PATTERN
from georaffer.downloaders.bb import BBDownloader


class TestBBDownloaderInit:
    """Tests for Brandenburg downloader initialization."""

    def test_init(self, tmp_path):
        downloader = BBDownloader(str(tmp_path))
        assert downloader.region_name == "BB"
        assert "data.geobasis-bb.de" in downloader.jp2_feed_url
        assert "dop/rgbi_tif" in downloader.jp2_feed_url
        assert "bdom/tif" in downloader.laz_feed_url

    def test_utm_zone(self, tmp_path):
        downloader = BBDownloader(str(tmp_path))
        assert downloader.UTM_ZONE == 33


class TestBBUtmToGridCoords:
    """Tests for UTM to grid coordinate conversion."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BBDownloader(str(tmp_path))

    def test_simple_conversion(self, downloader):
        jp2_coords, laz_coords = downloader.utm_to_grid_coords(350500, 5800500)
        assert jp2_coords == (350, 5800)
        assert laz_coords == (350, 5800)

    def test_edge_of_tile(self, downloader):
        jp2_coords, _ = downloader.utm_to_grid_coords(350000, 5800000)
        assert jp2_coords == (350, 5800)

    def test_jp2_laz_same_grid(self, downloader):
        jp2_coords, laz_coords = downloader.utm_to_grid_coords(351234, 5801567)
        assert jp2_coords == laz_coords


class TestBBFilenamePatterns:
    """Tests for BB bDOM and DOP filename regex patterns."""

    @pytest.mark.parametrize("filename,expected", [
        ("bdom_33250-5886.zip", (250, 5886)),
        ("bdom_33350-5700.zip", (350, 5700)),
        ("bdom_33400-5900.zip", (400, 5900)),
    ])
    def test_bdom_pattern_valid(self, filename, expected):
        match = BB_BDOM_PATTERN.match(filename)
        assert match is not None
        grid_x = int(match.group(1)[2:])
        grid_y = int(match.group(2))
        assert (grid_x, grid_y) == expected

    @pytest.mark.parametrize("filename", [
        "bdom_3325-5886.zip",
        "bdom_332500-5886.zip",
        "random.zip",
        "dop_33250-5886.zip",
    ])
    def test_bdom_pattern_invalid(self, filename):
        assert BB_BDOM_PATTERN.match(filename) is None

    @pytest.mark.parametrize("filename,expected", [
        ("dop_33250-5886.zip", (250, 5886)),
        ("dop_33350-5700.zip", (350, 5700)),
        ("dop_33400-5900.zip", (400, 5900)),
    ])
    def test_dop_pattern_valid(self, filename, expected):
        match = BB_DOP_PATTERN.match(filename)
        assert match is not None
        grid_x = int(match.group(1)[2:])
        grid_y = int(match.group(2))
        assert (grid_x, grid_y) == expected

    @pytest.mark.parametrize("filename", [
        "dop_3325-5886.zip",
        "dop_332500-5886.zip",
        "random.zip",
        "bdom_33250-5886.zip",
    ])
    def test_dop_pattern_invalid(self, filename):
        assert BB_DOP_PATTERN.match(filename) is None


class TestBBParseCoords:
    """Tests for coordinate parsing."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BBDownloader(str(tmp_path))

    @pytest.mark.parametrize("east,north,expected", [
        ("33250", "5886", (250, 5886)),
        ("33350", "5700", (350, 5700)),
        ("33400", "5900", (400, 5900)),
    ])
    def test_parse_valid_zone33(self, downloader, east, north, expected):
        assert downloader._parse_coords(east, north) == expected

    def test_parse_rejects_zone32(self, downloader):
        assert downloader._parse_coords("32250", "5886") is None

    @pytest.mark.parametrize("east,north", [
        ("3325", "5886"),  # Too short
        ("332500", "5886"),  # Too long
        ("abcde", "5886"),  # Invalid
    ])
    def test_parse_invalid_format(self, downloader, east, north):
        assert downloader._parse_coords(east, north) is None


class TestBBParseBdomListing:
    """Tests for bDOM HTML listing parsing."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BBDownloader(str(tmp_path))

    def test_parse_simple_listing(self, downloader):
        html = """
        <a href="bdom_33250-5886.zip">bdom_33250-5886.zip</a>
        <a href="bdom_33251-5886.zip">bdom_33251-5886.zip</a>
        <a href="bdom_33250-5887.zip">bdom_33250-5887.zip</a>
        """
        tiles = downloader._parse_bdom_listing(html)
        assert len(tiles) == 3
        assert (250, 5886) in tiles
        assert (251, 5886) in tiles
        assert (250, 5887) in tiles

    def test_parse_listing_skips_zone32(self, downloader):
        html = """
        <a href="bdom_33250-5886.zip">bdom_33250-5886.zip</a>
        <a href="bdom_32250-5886.zip">bdom_32250-5886.zip</a>
        """
        tiles = downloader._parse_bdom_listing(html)
        assert len(tiles) == 1
        assert (250, 5886) in tiles

    def test_parse_listing_generates_correct_urls(self, downloader):
        html = '<a href="bdom_33250-5886.zip">bdom_33250-5886.zip</a>'
        tiles = downloader._parse_bdom_listing(html)
        assert tiles[(250, 5886)] == "https://data.geobasis-bb.de/geobasis/daten/bdom/tif/bdom_33250-5886.zip"


class TestBBParseOGCFeatures:
    """Tests for OGC API feature parsing."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BBDownloader(str(tmp_path))

    def test_parse_valid_features(self, downloader):
        features = [
            {"properties": {"sheetnr": "33250-5886", "creationdate": "2025-05-13"}},
            {"properties": {"sheetnr": "33251-5886", "creationdate": "2024-04-30"}},
        ]
        result = downloader._parse_ogc_features(features)
        assert len(result) == 2
        assert result[0] == ("33250-5886", date(2025, 5, 13))
        assert result[1] == ("33251-5886", date(2024, 4, 30))

    def test_parse_skips_invalid_dates(self, downloader):
        features = [
            {"properties": {"sheetnr": "33250-5886", "creationdate": "invalid"}},
            {"properties": {"sheetnr": "33251-5886", "creationdate": "2024-04-30"}},
        ]
        result = downloader._parse_ogc_features(features)
        assert len(result) == 1

    def test_parse_skips_missing_fields(self, downloader):
        features = [
            {"properties": {"sheetnr": "33250-5886"}},
            {"properties": {"creationdate": "2024-04-30"}},
            {"properties": {"sheetnr": "33251-5886", "creationdate": "2024-04-30"}},
        ]
        result = downloader._parse_ogc_features(features)
        assert len(result) == 1


class TestBBFilenames:
    """Tests for filename extraction."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BBDownloader(str(tmp_path))

    def test_dsm_filename(self, downloader):
        url = "https://data.geobasis-bb.de/geobasis/daten/bdom/tif/bdom_33250-5886.zip"
        assert downloader.dsm_filename_from_url(url) == "bdom_33250-5886.zip"

    def test_image_filename(self, downloader):
        url = "https://data.geobasis-bb.de/geobasis/daten/dop/rgbi_tif/dop_33250-5886.zip"
        assert downloader.image_filename_from_url(url) == "dop_33250-5886.zip"

    def test_rejects_non_zip(self, downloader):
        with pytest.raises(ValueError, match="ZIP archives"):
            downloader._filename_from_url("https://example.com/file.tif")
