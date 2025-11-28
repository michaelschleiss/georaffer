"""Tests for RLP downloader."""

import xml.etree.ElementTree as ET
from unittest.mock import Mock

import pytest

from georaffer.downloaders.rlp import RLP_JP2_PATTERN, RLP_LAZ_PATTERN, RLPDownloader


class TestRLPDownloaderInit:
    """Tests for RLP downloader initialization."""

    def test_init_sets_region(self, tmp_path):
        """Test initialization sets correct region."""
        downloader = RLPDownloader(str(tmp_path))
        assert downloader.region_name == "RLP"

    def test_init_sets_feed_urls(self, tmp_path):
        """Test initialization sets feed URLs."""
        downloader = RLPDownloader(str(tmp_path))
        assert "geoportal.rlp.de" in downloader.jp2_feed_url
        assert "geoportal.rlp.de" in downloader.laz_feed_url

    def test_verify_ssl_is_false(self, tmp_path):
        """Test SSL verification is disabled for RLP."""
        downloader = RLPDownloader(str(tmp_path))
        assert downloader.verify_ssl is False


class TestRLPUtmToGridCoords:
    """Tests for UTM to grid coordinate conversion."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return RLPDownloader(str(tmp_path))

    def test_simple_conversion(self, downloader):
        """Test basic UTM to grid conversion."""
        # 362000m -> 362km, which should map to tile 362 (even)
        jp2_coords, laz_coords = downloader.utm_to_grid_coords(362500, 5604500)

        # 362500 // 2000 = 181, * 2 = 362
        assert jp2_coords == (362, 5604)
        assert laz_coords == (362, 5604)

    def test_rounds_to_even(self, downloader):
        """Test coordinates round to even (2km grid)."""
        # 363500m -> would be 363km, but rounds to 362
        jp2_coords, _ = downloader.utm_to_grid_coords(363500, 5605500)

        # 363500 // 2000 = 181, * 2 = 362
        assert jp2_coords[0] == 362
        assert jp2_coords[1] == 5604

    def test_jp2_laz_same_grid(self, downloader):
        """Test that RLP uses same grid for both types."""
        jp2_coords, laz_coords = downloader.utm_to_grid_coords(365000, 5608000)
        assert jp2_coords == laz_coords


class TestRLPFilenamePatterns:
    """Tests for RLP filename regex patterns."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("dop20rgb_32_362_5604_2_rp_2023.jp2", (362, 5604, 2023)),
            ("dop20rgb_32_370_5590_2_rp_2022.jp2", (370, 5590, 2022)),
            ("dop20rgb_32_400_5700_2_rp_2021.jp2", (400, 5700, 2021)),
        ],
    )
    def test_jp2_pattern_valid(self, filename, expected):
        """Test JP2 pattern matches valid filenames."""
        match = RLP_JP2_PATTERN.match(filename)
        assert match is not None
        assert int(match.group(1)) == expected[0]
        assert int(match.group(2)) == expected[1]
        assert int(match.group(3)) == expected[2]

    @pytest.mark.parametrize(
        "filename",
        [
            "dop20rgb_32_362_5604_2_rp.jp2",  # Missing year
            "dop10rgbi_32_362_5604_2_rp_2023.jp2",  # Wrong prefix (NRW style)
            "dop20rgb_32_362_5604_1_rp_2023.jp2",  # Wrong tile number (1 instead of 2)
            "dop20rgb_32_362_5604_2_nw_2023.jp2",  # Wrong region (nw instead of rp)
            "random.jp2",
        ],
    )
    def test_jp2_pattern_invalid(self, filename):
        """Test JP2 pattern rejects invalid filenames."""
        match = RLP_JP2_PATTERN.match(filename)
        assert match is None

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("bdom20rgbi_32_364_5582_2_rp.laz", (364, 5582)),
            ("bdom20rgbi_32_370_5590_2_rp.laz", (370, 5590)),
        ],
    )
    def test_laz_pattern_valid(self, filename, expected):
        """Test LAZ pattern matches valid filenames."""
        match = RLP_LAZ_PATTERN.match(filename)
        assert match is not None
        assert int(match.group(1)) == expected[0]
        assert int(match.group(2)) == expected[1]

    @pytest.mark.parametrize(
        "filename",
        [
            "bdom20rgbi_32_364_5582_2_rp_2023.laz",  # Has year (RLP LAZ shouldn't)
            "bdom50_32364_5582_2_rp.laz",  # NRW style prefix
            "bdom20rgbi_32_364_5582_2_nw.laz",  # Wrong region
        ],
    )
    def test_laz_pattern_invalid(self, filename):
        """Test LAZ pattern rejects invalid filenames."""
        match = RLP_LAZ_PATTERN.match(filename)
        assert match is None


class TestRLPParseJP2Feed:
    """Tests for JP2 feed parsing."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return RLPDownloader(str(tmp_path))

    def test_parse_atom_feed(self, downloader):
        """Test parsing INSPIRE Atom feed with JP2 links."""
        xml = """<feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <link type="image/jp2" href="https://example.com/dop20rgb_32_362_5604_2_rp_2023.jp2"/>
            </entry>
            <entry>
                <link type="image/jp2" href="https://example.com/dop20rgb_32_364_5606_2_rp_2023.jp2"/>
            </entry>
        </feed>"""
        root = ET.fromstring(xml)
        mock_session = Mock()

        tiles = downloader._parse_jp2_feed(mock_session, root)

        assert len(tiles) == 2
        assert (362, 5604) in tiles
        assert (364, 5606) in tiles

    def test_parse_feed_raises_on_invalid_filename(self, downloader):
        """Test that invalid filenames raise ValueError."""
        xml = """<feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <link type="image/jp2" href="https://example.com/invalid.jp2"/>
            </entry>
        </feed>"""
        root = ET.fromstring(xml)
        mock_session = Mock()

        with pytest.raises(ValueError, match="doesn't match pattern"):
            downloader._parse_jp2_feed(mock_session, root)

    def test_parse_feed_skips_non_jp2_links(self, downloader):
        """Test that non-JP2 links are skipped."""
        xml = """<feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <link type="image/jp2" href="https://example.com/dop20rgb_32_362_5604_2_rp_2023.jp2"/>
                <link type="text/html" href="https://example.com/info.html"/>
                <link type="application/xml" href="https://example.com/metadata.xml"/>
            </entry>
        </feed>"""
        root = ET.fromstring(xml)
        mock_session = Mock()

        tiles = downloader._parse_jp2_feed(mock_session, root)

        assert len(tiles) == 1


class TestRLPParseLAZFeed:
    """Tests for LAZ feed parsing."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return RLPDownloader(str(tmp_path))

    def test_parse_atom_feed(self, downloader):
        """Test parsing INSPIRE Atom feed with LAZ links."""
        xml = """<feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <link href="https://example.com/bdom20rgbi_32_364_5582_2_rp.laz"/>
            </entry>
            <entry>
                <link href="https://example.com/bdom20rgbi_32_366_5584_2_rp.laz"/>
            </entry>
        </feed>"""
        root = ET.fromstring(xml)
        mock_session = Mock()

        tiles = downloader._parse_laz_feed(mock_session, root)

        assert len(tiles) == 2
        assert (364, 5582) in tiles
        assert (366, 5584) in tiles

    def test_parse_feed_raises_on_invalid_filename(self, downloader):
        """Test that invalid filenames raise ValueError."""
        xml = """<feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <link href="https://example.com/invalid.laz"/>
            </entry>
        </feed>"""
        root = ET.fromstring(xml)
        mock_session = Mock()

        with pytest.raises(ValueError, match="doesn't match pattern"):
            downloader._parse_laz_feed(mock_session, root)

    def test_parse_feed_skips_non_laz_links(self, downloader):
        """Test that non-LAZ links are skipped."""
        xml = """<feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <link href="https://example.com/bdom20rgbi_32_364_5582_2_rp.laz"/>
                <link href="https://example.com/metadata.xml"/>
                <link href="https://example.com/info.html"/>
            </entry>
        </feed>"""
        root = ET.fromstring(xml)
        mock_session = Mock()

        tiles = downloader._parse_laz_feed(mock_session, root)

        assert len(tiles) == 1
