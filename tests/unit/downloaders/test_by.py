"""Tests for BY (Bayern) downloader."""

from unittest.mock import Mock

import pytest

from georaffer.config import BY_DOP_PATTERN, BY_DOM_PATTERN
from georaffer.downloaders.by import BYDownloader


class TestBYDownloaderInit:
    """Tests for BY downloader initialization."""

    def test_init_creates_downloader(self, tmp_path):
        """Test basic initialization."""
        downloader = BYDownloader(str(tmp_path))

        assert downloader.region_name == "BY"

    def test_init_with_imagery_from(self, tmp_path):
        """Test initialization with year range."""
        downloader = BYDownloader(str(tmp_path), imagery_from=(2020, 2022))

        assert downloader._from_year == 2020
        assert downloader._to_year == 2022

    def test_init_without_imagery_from(self, tmp_path):
        """Test initialization without year range."""
        downloader = BYDownloader(str(tmp_path))

        assert downloader._from_year is None
        assert downloader._to_year is None


class TestBYUtmToGridCoords:
    """Tests for UTM to grid coordinate conversion."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BYDownloader(str(tmp_path))

    def test_simple_conversion(self, downloader):
        """Test basic UTM to grid conversion."""
        dop_coords, dom_coords = downloader.utm_to_grid_coords(679500, 5392500)

        assert dop_coords == (679, 5392)
        assert dom_coords == (679, 5392)

    def test_edge_of_tile(self, downloader):
        """Test coordinates at tile edge."""
        dop_coords, dom_coords = downloader.utm_to_grid_coords(679000, 5392000)

        assert dop_coords == (679, 5392)

    def test_dop_dom_same_grid(self, downloader):
        """Test that BY uses same grid for both DOP and DOM."""
        dop_coords, dom_coords = downloader.utm_to_grid_coords(680234, 5393567)

        assert dop_coords == dom_coords


class TestBYFilenamePatterns:
    """Tests for BY filename regex patterns."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("32679_5392.tif", (679, 5392)),
            ("32400_5700.tif", (400, 5700)),
            ("32550_5500.tif", (550, 5500)),
            ("32679_5392_2018.tif", (679, 5392)),
        ],
    )
    def test_dop_pattern_valid(self, filename, expected):
        """Test DOP pattern matches valid filenames."""
        match = BY_DOP_PATTERN.match(filename)
        assert match is not None
        assert int(match.group(1)) == expected[0]
        assert int(match.group(2)) == expected[1]

    @pytest.mark.parametrize(
        "filename",
        [
            "32679_5392_20_DOM.tif",  # DOM pattern
            "dop_32679_5392.tif",  # Wrong prefix
            "32679_5392.jp2",  # Wrong extension
            "random.tif",
        ],
    )
    def test_dop_pattern_invalid(self, filename):
        """Test DOP pattern rejects invalid filenames."""
        match = BY_DOP_PATTERN.match(filename)
        assert match is None

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("32686_5369_20_DOM.tif", (686, 5369)),
            ("32400_5700_20_DOM.tif", (400, 5700)),
            ("32550_5500_20_DOM.tif", (550, 5500)),
        ],
    )
    def test_dom_pattern_valid(self, filename, expected):
        """Test DOM pattern matches valid filenames."""
        match = BY_DOM_PATTERN.match(filename)
        assert match is not None
        assert int(match.group(1)) == expected[0]
        assert int(match.group(2)) == expected[1]

    @pytest.mark.parametrize(
        "filename",
        [
            "32679_5392.tif",  # DOP pattern
            "dom_32679_5392.tif",  # Wrong prefix
            "32679_5392_20_DOM.jp2",  # Wrong extension
            "random.tif",
        ],
    )
    def test_dom_pattern_invalid(self, filename):
        """Test DOM pattern rejects invalid filenames."""
        match = BY_DOM_PATTERN.match(filename)
        assert match is None


class TestBYParseMetalink:
    """Tests for metalink parsing."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BYDownloader(str(tmp_path))

    def test_parse_dop_metalink(self, downloader):
        """Test parsing DOP metalink XML."""
        xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <metalink xmlns="urn:ietf:params:xml:ns:metalink">
          <file name="32679_5392.tif">
            <url>https://download1.bayernwolke.de/a/dop20/data/32679_5392.tif</url>
            <hash type="sha-256">abc123</hash>
          </file>
          <file name="32680_5392.tif">
            <url>https://download1.bayernwolke.de/a/dop20/data/32680_5392.tif</url>
            <hash type="sha-256">def456</hash>
          </file>
        </metalink>"""

        results = downloader._parse_metalink(xml, "dop")

        assert len(results) == 2
        coords_urls = {r[0]: r[1] for r in results}
        assert (679, 5392) in coords_urls
        assert (680, 5392) in coords_urls
        assert "32679_5392.tif" in coords_urls[(679, 5392)]

    def test_parse_dom_metalink(self, downloader):
        """Test parsing DOM metalink XML."""
        xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <metalink xmlns="urn:ietf:params:xml:ns:metalink">
          <file name="32686_5369_20_DOM.tif">
            <url>https://download1.bayernwolke.de/a/dom20/DOM/32686_5369_20_DOM.tif</url>
            <hash type="sha-256">abc123</hash>
          </file>
        </metalink>"""

        results = downloader._parse_metalink(xml, "dom")

        assert len(results) == 1
        assert results[0][0] == (686, 5369)

    def test_parse_empty_metalink(self, downloader):
        """Test parsing empty metalink."""
        xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <metalink xmlns="urn:ietf:params:xml:ns:metalink">
        </metalink>"""

        results = downloader._parse_metalink(xml, "dop")

        assert len(results) == 0

    def test_parse_invalid_xml(self, downloader):
        """Test handling invalid XML."""
        xml = b"not valid xml"

        results = downloader._parse_metalink(xml, "dop")

        assert len(results) == 0


class TestBYFilenames:
    """Tests for filename generation."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BYDownloader(str(tmp_path))

    def test_image_filename_from_url(self, downloader):
        """Test image filename extraction from URL."""
        url = "https://download1.bayernwolke.de/a/dop20/data/32679_5392.tif"
        filename = downloader.image_filename_from_url(url)
        assert filename == "32679_5392.tif"


class TestBYCatalogDates:
    """Tests for BY catalog acquisition dates."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BYDownloader(str(tmp_path))

    def test_current_tile_uses_latest_wms_year(self, tmp_path, monkeypatch):
        """Metalink tiles should inherit the latest WMS year/date."""
        downloader = BYDownloader(str(tmp_path))

        def fake_fetch_all_metalinks(base_url, product):
            if product == "dop":
                return [((679, 5392), "https://download1.bayernwolke.de/a/dop20/data/32679_5392.tif")]
            return []

        def fake_wms_check(years, grid_x, grid_y):
            assert grid_x == 679
            assert grid_y == 5392
            return {
                2018: {"acquisition_date": "2018-05-01"},
                2020: {"acquisition_date": "2020-06-10"},
            }

        class FakeWMS:
            def get_tile_url(self, year, grid_x, grid_y):
                return f"wms://{year}/{grid_x}/{grid_y}"

        monkeypatch.delenv("GEORAFFER_DISABLE_WMS", raising=False)
        monkeypatch.setattr(downloader, "_fetch_all_metalinks", fake_fetch_all_metalinks)
        monkeypatch.setattr(downloader, "_historic_years", lambda: [2018, 2020])
        monkeypatch.setattr(downloader, "_wms_check_coverage_multi", fake_wms_check)
        downloader._wms = FakeWMS()

        catalog = downloader._load_catalog()
        tiles = catalog.image_tiles[(679, 5392)]

        assert 2020 in tiles
        assert tiles[2020]["url"].endswith("32679_5392.tif")
        assert tiles[2020]["acquisition_date"] == "2020-06-10"
        assert tiles[2018]["url"] == "wms://2018/679/5392"

    def test_image_filename_from_wms_url(self, downloader):
        """Test image filename generation from WMS GetMap URL."""
        url = (
            "https://geoservices.bayern.de/od/wms/histdop/v1/histdop?"
            "SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
            "&LAYERS=by_dop_2018_h&STYLES=&SRS=EPSG:25832"
            "&BBOX=679000,5392000,680000,5393000"
            "&WIDTH=5000&HEIGHT=5000&FORMAT=image/tiff"
        )
        filename = downloader.image_filename_from_url(url)
        assert filename == "32679_5392_2018.tif"

    def test_dsm_filename_from_url(self, downloader):
        """Test DSM filename extraction from URL."""
        url = "https://download1.bayernwolke.de/a/dom20/DOM/32686_5369_20_DOM.tif"
        filename = downloader.dsm_filename_from_url(url)
        assert filename == "32686_5369_20_DOM.tif"

    def test_rejects_non_tif_url(self, downloader):
        """Test that non-TIF URLs are rejected."""
        url = "https://example.com/file.zip"
        with pytest.raises(ValueError, match="TIF"):
            downloader.image_filename_from_url(url)


class TestBYHistoricYears:
    """Tests for BY historic year discovery."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BYDownloader(str(tmp_path))

    def test_historic_years_parses_non_namespaced_capabilities(self, downloader):
        """BY WMS capabilities should parse without XML namespaces."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <WMT_MS_Capabilities>
          <Capability>
            <Layer>
              <Layer><Name>by_dop_2018_h</Name></Layer>
              <Layer><Name>by_dop_2018_h_info</Name></Layer>
              <Layer><Name>by_dop_2020_h</Name></Layer>
            </Layer>
          </Capability>
        </WMT_MS_Capabilities>"""

        mock_response = Mock()
        mock_response.text = xml
        mock_response.raise_for_status = Mock()

        downloader._session.get = Mock(return_value=mock_response)

        years = downloader._historic_years()
        assert years == [2018, 2020]
