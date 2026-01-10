"""Tests for RLP downloader."""

import xml.etree.ElementTree as ET
from unittest.mock import Mock, patch

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
        assert "geobasis-rlp.de" in downloader.jp2_feed_url
        assert "geobasis-rlp.de" in downloader.laz_feed_url


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


class TestRLPDownloaderDownloadFile:
    """Tests for RLP download_file handling."""

    def test_wms_getmap_uses_wms_downloader(self, tmp_path, monkeypatch):
        """WMS GetMap URLs should use WMS download handling."""
        from georaffer.downloaders.base import RegionDownloader

        downloader = RLPDownloader(str(tmp_path), imagery_from=(2020, 2021))

        called = {"count": 0}

        def fake_base(self, url, output_path, on_progress=None):
            called["count"] += 1
            return True

        monkeypatch.setattr(RegionDownloader, "download_file", fake_base)

        wms = Mock()
        wms.download_tile.return_value = True
        downloader._wms = wms

        url = "https://example.com/wms?SERVICE=WMS&REQUEST=GetMap&LAYERS=x&FORMAT=image/tiff"
        out_path = str(tmp_path / "tile.tif")

        assert downloader.download_file(url, out_path) is True
        assert called["count"] == 0
        wms.download_tile.assert_called_once_with(url, out_path, on_progress=None)

    def test_non_wms_uses_base_downloader(self, tmp_path, monkeypatch):
        """Non-WMS URLs should use the base downloader."""
        from georaffer.downloaders.base import RegionDownloader

        downloader = RLPDownloader(str(tmp_path))

        called = {"count": 0}

        def fake_base(self, url, output_path, on_progress=None):
            called["count"] += 1
            return True

        monkeypatch.setattr(RegionDownloader, "download_file", fake_base)

        url = "https://example.com/dop20rgb_32_362_5604_2_rp_2023.jp2"
        out_path = str(tmp_path / "tile.jp2")

        assert downloader.download_file(url, out_path) is True
        assert called["count"] == 1

    def test_wms_getmap_failure_raises(self, tmp_path):
        """WMS download failures should surface as RuntimeError."""
        downloader = RLPDownloader(str(tmp_path), imagery_from=(2020, 2021))

        wms = Mock()
        wms.download_tile.return_value = False
        downloader._wms = wms

        url = "https://example.com/wms?SERVICE=WMS&REQUEST=GetMap&LAYERS=x&FORMAT=image/tiff"
        out_path = str(tmp_path / "tile.tif")

        with pytest.raises(RuntimeError, match="WMS download failed"):
            downloader.download_file(url, out_path)


class TestRLPFilenamePatterns:
    """Tests for RLP filename regex patterns."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("dop20rgb_32_362_5604_2_rp_2023.jp2", (362, 5604, 2023)),
            ("dop20rgb_32_370_5590_2_rp_2022.jp2", (370, 5590, 2022)),
            ("dop20rgb_32_400_5700_2_rp_2021.jp2", (400, 5700, 2021)),
            ("dop20rgb_32_380_5540_2_rp_2020.tif", (380, 5540, 2020)),  # WMS download
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


class TestRLPParseJP2Tiles:
    """Tests for JP2 feed parsing (geobasis-rlp.de atomfeed-links format)."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return RLPDownloader(str(tmp_path))

    def test_parse_atomfeed_links(self, downloader):
        """Test parsing geobasis-rlp.de atomfeed-links.xml format."""
        xml = """<root>
            <link type="image/jp2" href="https://example.com/dop20rgb_32_362_5604_2_rp_2023.jp2"/>
            <link type="image/jp2" href="https://example.com/dop20rgb_32_364_5606_2_rp_2023.jp2"/>
        </root>"""
        root = ET.fromstring(xml)

        tiles = downloader._parse_jp2_tiles(root)

        assert len(tiles) == 2
        assert (362, 5604) in tiles
        assert (364, 5606) in tiles

    def test_parse_feed_raises_on_invalid_filename(self, downloader):
        """Test that invalid filenames raise ValueError."""
        xml = """<root>
            <link type="image/jp2" href="https://example.com/invalid.jp2"/>
        </root>"""
        root = ET.fromstring(xml)

        with pytest.raises(ValueError, match="doesn't match pattern"):
            downloader._parse_jp2_tiles(root)

    def test_parse_feed_skips_non_jp2_links(self, downloader):
        """Test that non-JP2 links are skipped."""
        xml = """<root>
            <link type="image/jp2" href="https://example.com/dop20rgb_32_362_5604_2_rp_2023.jp2"/>
            <link type="text/html" href="https://example.com/info.html"/>
            <link type="application/xml" href="https://example.com/metadata.xml"/>
        </root>"""
        root = ET.fromstring(xml)

        tiles = downloader._parse_jp2_tiles(root)

        assert len(tiles) == 1


class TestRLPParseLAZTiles:
    """Tests for LAZ feed parsing (geobasis-rlp.de atomfeed-links format)."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return RLPDownloader(str(tmp_path))

    def test_parse_atomfeed_links(self, downloader):
        """Test parsing geobasis-rlp.de atomfeed-links.xml format."""
        xml = """<root>
            <link href="https://example.com/bdom20rgbi_32_364_5582_2_rp.laz"/>
            <link href="https://example.com/bdom20rgbi_32_366_5584_2_rp.laz"/>
        </root>"""
        root = ET.fromstring(xml)

        tiles = downloader._parse_laz_tiles(root)

        assert len(tiles) == 2
        assert (364, 5582) in tiles
        assert (366, 5584) in tiles
        # Returns coords -> url mapping (year assigned later from image tile)
        assert tiles[(364, 5582)].endswith(".laz")
        assert tiles[(366, 5584)].endswith(".laz")

    def test_parse_feed_raises_on_invalid_filename(self, downloader):
        """Test that invalid filenames raise ValueError."""
        xml = """<root>
            <link href="https://example.com/invalid.laz"/>
        </root>"""
        root = ET.fromstring(xml)

        with pytest.raises(ValueError, match="doesn't match pattern"):
            downloader._parse_laz_tiles(root)

    def test_parse_feed_skips_non_laz_links(self, downloader):
        """Test that non-LAZ links are skipped."""
        xml = """<root>
            <link href="https://example.com/bdom20rgbi_32_364_5582_2_rp.laz"/>
            <link href="https://example.com/metadata.xml"/>
            <link href="https://example.com/info.html"/>
        </root>"""
        root = ET.fromstring(xml)

        tiles = downloader._parse_laz_tiles(root)

        assert len(tiles) == 1


class TestRLPHistoricalImagery:
    """Tests for historical imagery support via WMS."""

    def test_init_without_imagery_from(self, tmp_path):
        """Test initialization without imagery_from sets None."""
        downloader = RLPDownloader(str(tmp_path))
        assert downloader._from_year is None
        assert downloader._to_year is None

    def test_init_with_single_year(self, tmp_path):
        """Test initialization with single year (from only)."""
        downloader = RLPDownloader(str(tmp_path), imagery_from=(2020, None))
        assert downloader._from_year == 2020
        assert downloader._to_year is None

    def test_init_with_year_range(self, tmp_path):
        """Test initialization with year range."""
        downloader = RLPDownloader(str(tmp_path), imagery_from=(2015, 2020))
        assert downloader._from_year == 2015
        assert downloader._to_year == 2020

    def test_init_rejects_year_before_2010(self, tmp_path):
        """Test initialization rejects years before 2010 (WMS metadata compromised)."""
        with pytest.raises(ValueError, match="Year 2009 not supported"):
            RLPDownloader(str(tmp_path), imagery_from=(2009, None))

    def test_init_accepts_year_2010_and_later(self, tmp_path):
        """Test initialization accepts years from 2010 onwards."""
        # Should not raise
        downloader = RLPDownloader(str(tmp_path), imagery_from=(2010, None))
        assert downloader._from_year == 2010

    def test_historic_years_range(self, tmp_path):
        """Test HISTORIC_YEARS includes 2010-2024 (pre-2010 excluded due to metadata issues)."""
        downloader = RLPDownloader(str(tmp_path))
        assert 2010 in downloader.HISTORIC_YEARS
        assert 2024 in downloader.HISTORIC_YEARS
        assert 2009 not in downloader.HISTORIC_YEARS
        assert 2025 not in downloader.HISTORIC_YEARS

    def test_wms_lazy_initialized(self, tmp_path):
        """Test WMS source is lazily initialized."""
        downloader = RLPDownloader(str(tmp_path), imagery_from=(2020, None))
        assert downloader._wms is None

        # Access wms property triggers initialization
        wms = downloader.wms
        assert wms is not None
        assert downloader._wms is wms

    def test_extract_year_from_url(self, tmp_path):
        """Test _extract_year_from_url parses JP2 URLs."""
        downloader = RLPDownloader(str(tmp_path))
        url = "https://example.com/dop20rgb_32_362_5604_2_rp_2023.jp2"
        year = downloader._extract_year_from_url(url)
        assert year == 2023

    def test_extract_year_from_url_returns_none_for_invalid(self, tmp_path):
        """Test _extract_year_from_url returns None for invalid URLs."""
        downloader = RLPDownloader(str(tmp_path))
        url = "https://example.com/invalid.jp2"
        year = downloader._extract_year_from_url(url)
        assert year is None


class TestRLPLoadCatalog:
    """Tests for _load_catalog parallel execution."""

    def test_parallel_execution_processes_all_coords(self, tmp_path, monkeypatch):
        """Verify ThreadPoolExecutor processes all coordinates without dropping any."""
        from georaffer.downloaders import feeds
        downloader = RLPDownloader(str(tmp_path), quiet=True)

        # Mock current tiles from ATOM feed
        current_tiles = {
            (362, 5604): "http://example.com/dop20rgb_32_362_5604_2_rp_2023.jp2",
            (364, 5606): "http://example.com/dop20rgb_32_364_5606_2_rp_2023.jp2",
            (366, 5608): "http://example.com/dop20rgb_32_366_5608_2_rp_2023.jp2",
        }
        monkeypatch.setattr(downloader, "_parse_jp2_tiles", lambda root: current_tiles)
        monkeypatch.setattr(downloader, "_parse_laz_tiles", lambda root: {})

        # Mock the XML feed fetch
        def mock_fetch_xml(session, url, wrap_content=False):
            return ET.fromstring("<root></root>")

        monkeypatch.setattr(feeds, "fetch_xml_feed", mock_fetch_xml)

        # Track which coords were queried
        queried_coords = []

        def mock_check_coverage(years, grid_x, grid_y):
            queried_coords.append((grid_x, grid_y))
            return {2020: {"acquisition_date": "2020-06-01"}}

        monkeypatch.setattr(downloader.wms, "check_coverage_multi", mock_check_coverage)
        monkeypatch.setattr(downloader.wms, "get_tile_url", lambda y, x, g: f"http://wms/{y}/{x}/{g}")

        catalog = downloader._load_catalog()

        # Verify all coordinates were queried
        assert sorted(queried_coords) == sorted(current_tiles.keys()), (
            f"Missing coords: {set(current_tiles.keys()) - set(queried_coords)}"
        )

    def test_runtime_errors_counted_not_raised(self, tmp_path, monkeypatch):
        """Verify RuntimeError from WMS retries is counted but doesn't abort build."""
        from georaffer.downloaders import feeds
        downloader = RLPDownloader(str(tmp_path), quiet=True)

        current_tiles = {
            (362, 5604): "http://example.com/dop20rgb_32_362_5604_2_rp_2023.jp2",
            (364, 5606): "http://example.com/dop20rgb_32_364_5606_2_rp_2023.jp2",
        }
        monkeypatch.setattr(downloader, "_parse_jp2_tiles", lambda root: current_tiles)
        monkeypatch.setattr(downloader, "_parse_laz_tiles", lambda root: {})

        def mock_fetch_xml(session, url, wrap_content=False):
            return ET.fromstring("<root></root>")

        monkeypatch.setattr(feeds, "fetch_xml_feed", mock_fetch_xml)

        call_count = {"total": 0, "errors": 0}

        def mock_check_coverage(years, grid_x, grid_y):
            call_count["total"] += 1
            if grid_x == 362:
                call_count["errors"] += 1
                raise RuntimeError("WMS failed after retries")
            return {2020: {"acquisition_date": "2020-06-01"}}

        monkeypatch.setattr(downloader.wms, "check_coverage_multi", mock_check_coverage)
        monkeypatch.setattr(downloader.wms, "get_tile_url", lambda y, x, g: f"http://wms/{y}/{x}/{g}")

        # Should not raise despite RuntimeError
        catalog = downloader._load_catalog()

        assert call_count["total"] == 2
        assert call_count["errors"] == 1

    def test_unexpected_errors_propagate(self, tmp_path, monkeypatch):
        """Verify unexpected exceptions are not silently swallowed."""
        from georaffer.downloaders import feeds
        downloader = RLPDownloader(str(tmp_path), quiet=True)

        current_tiles = {(362, 5604): "http://example.com/dop20rgb_32_362_5604_2_rp_2023.jp2"}
        monkeypatch.setattr(downloader, "_parse_jp2_tiles", lambda root: current_tiles)

        def mock_fetch_xml(session, url, wrap_content=False):
            return ET.fromstring("<root></root>")

        monkeypatch.setattr(feeds, "fetch_xml_feed", mock_fetch_xml)

        def mock_check_coverage(years, grid_x, grid_y):
            raise ValueError("Unexpected parse error")

        monkeypatch.setattr(downloader.wms, "check_coverage_multi", mock_check_coverage)

        with pytest.raises(ValueError, match="Unexpected parse error"):
            downloader._load_catalog()
