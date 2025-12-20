"""Tests for BB (Brandenburg) downloader."""

import pytest

from georaffer.config import BB_BDOM_PATTERN, BB_DOP_PATTERN
from georaffer.downloaders.bb import BrandenburgDownloader


class TestBrandenburgDownloaderInit:
    """Tests for Brandenburg downloader initialization."""

    def test_init(self, tmp_path):
        """Test basic initialization."""
        downloader = BrandenburgDownloader(str(tmp_path))

        assert downloader.region_name == "BB"
        assert "data.geobasis-bb.de" in downloader.jp2_feed_url
        assert "dop/rgbi_tif" in downloader.jp2_feed_url
        assert "bdom/tif" in downloader.laz_feed_url

    def test_utm_zone(self, tmp_path):
        """Test that BB uses UTM Zone 33."""
        downloader = BrandenburgDownloader(str(tmp_path))
        assert downloader.UTM_ZONE == 33


class TestBrandenburgUtmToGridCoords:
    """Tests for UTM to grid coordinate conversion."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BrandenburgDownloader(str(tmp_path))

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


class TestBrandenburgFilenamePatterns:
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
        # Extract grid_x from 5-digit code: first 2 = zone, last 3 = easting
        east_code = match.group(1)
        grid_x = int(east_code[2:])  # Skip zone prefix
        grid_y = int(match.group(2))
        assert grid_x == expected[0]
        assert grid_y == expected[1]

    @pytest.mark.parametrize(
        "filename",
        [
            "bdom_3325-5886.zip",  # Wrong easting format (4 digits)
            "bdom_332500-5886.zip",  # Wrong easting format (6 digits)
            "bdom_33250-5886.tif",  # Legacy raster naming (unsupported)
            "random.zip",
            "dop_33250-5886.zip",  # Wrong prefix
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
            "dop_33250-5886.tif",  # Legacy raster naming (unsupported)
            "random.zip",
            "bdom_33250-5886.zip",
        ],
    )
    def test_dop_pattern_invalid(self, filename):
        """Test DOP pattern rejects invalid filenames."""
        match = BB_DOP_PATTERN.match(filename)
        assert match is None


class TestBrandenburgParseFilename:
    """Tests for filename parsing."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BrandenburgDownloader(str(tmp_path))

    def test_parse_valid_zone33(self, downloader):
        """Test parsing filename with correct UTM zone 33."""
        result = downloader._parse_filename("bdom_33250-5886.zip", BB_BDOM_PATTERN)
        assert result == (250, 5886)

    def test_parse_rejects_zone32(self, downloader):
        """Test that zone 32 files are rejected (wrong zone for BB)."""
        result = downloader._parse_filename("bdom_32250-5886.zip", BB_BDOM_PATTERN)
        assert result is None

    def test_parse_invalid_format(self, downloader):
        """Test that invalid format returns None."""
        result = downloader._parse_filename("invalid.zip", BB_BDOM_PATTERN)
        assert result is None

    def test_parse_dop_valid(self, downloader):
        """Test parsing DOP filename with correct UTM zone 33."""
        result = downloader._parse_filename("dop_33250-5886.zip", BB_DOP_PATTERN)
        assert result == (250, 5886)


class TestBrandenburgParseBdomListing:
    """Tests for HTML listing parsing."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BrandenburgDownloader(str(tmp_path))

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


class TestBrandenburgParseDopListing:
    """Tests for DOP HTML listing parsing."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BrandenburgDownloader(str(tmp_path))

    def test_parse_simple_listing(self, downloader):
        """Test parsing simple DOP HTML listing."""
        html = """
        <html>
        <body>
        <a href="dop_33250-5886.zip">dop_33250-5886.zip</a>
        <a href="dop_33251-5886.zip">dop_33251-5886.zip</a>
        </body>
        </html>
        """
        tiles = downloader._parse_dop_listing(html)

        assert len(tiles) == 2
        assert (250, 5886) in tiles
        assert (251, 5886) in tiles

    def test_parse_listing_skips_zone32(self, downloader):
        """Test that zone 32 tiles are skipped for DOP."""
        html = """
        <html>
        <a href="dop_33250-5886.zip">dop_33250-5886.zip</a>
        <a href="dop_32250-5886.zip">dop_32250-5886.zip</a>
        </html>
        """
        tiles = downloader._parse_dop_listing(html)

        assert len(tiles) == 1
        assert (250, 5886) in tiles

    def test_parse_listing_generates_correct_urls(self, downloader):
        """Test that parsed DOP tiles have correct download URLs."""
        html = '<a href="dop_33250-5886.zip">dop_33250-5886.zip</a>'
        tiles = downloader._parse_dop_listing(html)

        url = tiles[(250, 5886)]
        assert url == "https://data.geobasis-bb.de/geobasis/daten/dop/rgbi_tif/dop_33250-5886.zip"


class TestBrandenburgDsmFilename:
    """Tests for DSM filename extraction."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BrandenburgDownloader(str(tmp_path))

    def test_dsm_filename_from_zip_url(self, downloader):
        """Test keeping zip name from zip URL."""
        url = "https://data.geobasis-bb.de/geobasis/daten/bdom/tif/bdom_33250-5886.zip"
        result = downloader.dsm_filename_from_url(url)
        assert result == "bdom_33250-5886.zip"

    def test_dsm_filename_from_tif_url(self, downloader):
        """Test that tif URL is rejected."""
        url = "https://example.com/bdom_33250-5886.tif"
        with pytest.raises(ValueError):
            downloader.dsm_filename_from_url(url)


class TestBrandenburgImageFilename:
    """Tests for DOP filename extraction."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BrandenburgDownloader(str(tmp_path))

    def test_image_filename_from_zip_url(self, downloader):
        """Test keeping zip name from zip URL."""
        url = "https://data.geobasis-bb.de/geobasis/daten/dop/rgbi_tif/dop_33250-5886.zip"
        result = downloader.image_filename_from_url(url)
        assert result == "dop_33250-5886.zip"

    def test_image_filename_from_tif_url(self, downloader):
        """Test that tif URL is rejected."""
        url = "https://example.com/dop_33250-5886.tif"
        with pytest.raises(ValueError):
            downloader.image_filename_from_url(url)
