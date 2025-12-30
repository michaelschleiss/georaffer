"""Tests for BW (Baden-Württemberg) downloader."""

from datetime import date

import pytest

from georaffer.config import BW_DOM_PATTERN, BW_DOP_PATTERN
from georaffer.downloaders.bw import BWDownloader


class TestBWDownloaderInit:
    """Tests for Baden-Württemberg downloader initialization."""

    def test_init(self, tmp_path):
        downloader = BWDownloader(str(tmp_path))
        assert downloader.region_name == "BW"
        assert "opengeodata.lgl-bw.de" in BWDownloader.DOP_BASE_URL
        assert "opengeodata.lgl-bw.de" in BWDownloader.DOM_BASE_URL

    def test_utm_zone(self, tmp_path):
        downloader = BWDownloader(str(tmp_path))
        assert downloader.UTM_ZONE == 32

    def test_init_with_imagery_from(self, tmp_path):
        downloader = BWDownloader(str(tmp_path), imagery_from=(2015, 2020))
        assert downloader._from_year == 2015
        assert downloader._to_year == 2020

    def test_init_with_early_year_raises(self, tmp_path):
        with pytest.raises(ValueError, match="1960"):
            BWDownloader(str(tmp_path), imagery_from=(1950, None))


class TestBWUtmToGridCoords:
    """Tests for UTM to grid coordinate conversion.

    BW uses 2km grid with odd easting and even northing alignment.
    """

    @pytest.fixture
    def downloader(self, tmp_path):
        return BWDownloader(str(tmp_path))

    def test_simple_conversion_odd_even(self, downloader):
        # Coords in tile with odd E, even N
        jp2_coords, laz_coords = downloader.utm_to_grid_coords(489500, 5420500)
        assert jp2_coords == (489, 5420)  # 489 is odd, 5420 is even
        assert laz_coords == (489, 5420)

    def test_even_easting_rounds_to_odd(self, downloader):
        # Even easting should round down to nearest odd
        jp2_coords, _ = downloader.utm_to_grid_coords(490000, 5420000)
        # 490 is even, should become 489 (odd)
        assert jp2_coords[0] == 489

    def test_odd_northing_rounds_to_even(self, downloader):
        # Odd northing should round down to nearest even
        jp2_coords, _ = downloader.utm_to_grid_coords(489000, 5421000)
        # 5421 is odd, should become 5420 (even)
        assert jp2_coords[1] == 5420

    def test_both_adjustments(self, downloader):
        # Both adjustments needed
        jp2_coords, _ = downloader.utm_to_grid_coords(490000, 5421000)
        assert jp2_coords == (489, 5420)

    def test_jp2_laz_same_grid(self, downloader):
        jp2_coords, laz_coords = downloader.utm_to_grid_coords(490234, 5421567)
        assert jp2_coords == laz_coords


class TestBWFilenamePatterns:
    """Tests for BW DOP and DOM filename regex patterns."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("dop20rgb_32_489_5420_2_bw.zip", (489, 5420)),
            ("dop20rgb_32_391_5268_2_bw.zip", (391, 5268)),
            ("dop20rgb_32_609_5514_2_bw.zip", (609, 5514)),
        ],
    )
    def test_dop_pattern_valid(self, filename, expected):
        match = BW_DOP_PATTERN.match(filename)
        assert match is not None
        grid_x = int(match.group(1))
        grid_y = int(match.group(2))
        assert (grid_x, grid_y) == expected

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("dop20rgb_32_489_5420_2_bw_2015.zip", (489, 5420, "2015")),
            ("dop20rgb_32_391_5268_2_bw_2020.zip", (391, 5268, "2020")),
        ],
    )
    def test_dop_pattern_with_year(self, filename, expected):
        match = BW_DOP_PATTERN.match(filename)
        assert match is not None
        grid_x = int(match.group(1))
        grid_y = int(match.group(2))
        year = match.group(3)
        assert (grid_x, grid_y, year) == expected

    @pytest.mark.parametrize(
        "filename",
        [
            "dop20rgb_32_489_5420_2_nw.zip",  # Wrong region
            "dop20rgb_33_489_5420_2_bw.zip",  # Wrong zone
            "random.zip",
            "dom1_32_489_5420_2_bw.zip",  # DOM pattern
        ],
    )
    def test_dop_pattern_invalid(self, filename):
        assert BW_DOP_PATTERN.match(filename) is None

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("dom1_32_489_5420_2_bw.zip", (489, 5420)),
            ("dom1_32_391_5268_2_bw.zip", (391, 5268)),
            ("dom1_32_609_5514_2_bw.zip", (609, 5514)),
        ],
    )
    def test_dom_pattern_valid(self, filename, expected):
        match = BW_DOM_PATTERN.match(filename)
        assert match is not None
        grid_x = int(match.group(1))
        grid_y = int(match.group(2))
        assert (grid_x, grid_y) == expected

    @pytest.mark.parametrize(
        "filename",
        [
            "dom1_32_489_5420_2_nw.zip",  # Wrong region
            "dom1_33_489_5420_2_bw.zip",  # Wrong zone
            "random.zip",
            "dop20rgb_32_489_5420_2_bw.zip",  # DOP pattern
        ],
    )
    def test_dom_pattern_invalid(self, filename):
        assert BW_DOM_PATTERN.match(filename) is None


class TestBWWfsToDownloadCoords:
    """Tests for WFS 1km to download 2km grid coordinate mapping."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BWDownloader(str(tmp_path))

    @pytest.mark.parametrize(
        "wfs_x,wfs_y,expected",
        [
            (489, 5420, (489, 5420)),  # Already aligned (odd E, even N)
            (490, 5420, (489, 5420)),  # Even E -> round down to odd
            (489, 5421, (489, 5420)),  # Odd N -> round down to even
            (490, 5421, (489, 5420)),  # Both need adjustment
            (391, 5268, (391, 5268)),  # Already aligned
            (392, 5269, (391, 5268)),  # Both need adjustment
        ],
    )
    def test_wfs_to_download_coords(self, downloader, wfs_x, wfs_y, expected):
        result = downloader._wfs_to_download_coords(wfs_x, wfs_y)
        assert result == expected


class TestBWUrlBuilding:
    """Tests for download URL construction."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BWDownloader(str(tmp_path))

    def test_build_dop_url(self, downloader):
        url = downloader._build_dop_url(489, 5420)
        assert url == "https://opengeodata.lgl-bw.de/data/dop20/dop20rgb_32_489_5420_2_bw.zip"

    def test_build_dom_url(self, downloader):
        url = downloader._build_dom_url(489, 5420)
        assert url == "https://opengeodata.lgl-bw.de/data/dom1/dom1_32_489_5420_2_bw.zip"


class TestBWFilenames:
    """Tests for filename extraction."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BWDownloader(str(tmp_path))

    def test_dsm_filename(self, downloader):
        url = "https://opengeodata.lgl-bw.de/data/dom1/dom1_32_489_5420_2_bw.zip"
        assert downloader.dsm_filename_from_url(url) == "dom1_32_489_5420_2_bw.zip"

    def test_image_filename(self, downloader):
        url = "https://opengeodata.lgl-bw.de/data/dop20/dop20rgb_32_489_5420_2_bw.zip"
        assert downloader.image_filename_from_url(url) == "dop20rgb_32_489_5420_2_bw.zip"

    def test_rejects_non_zip(self, downloader):
        with pytest.raises(ValueError, match="ZIP archives"):
            downloader._filename_from_url("https://example.com/file.tif")


class TestBWParseWfsFeatures:
    """Tests for WFS feature parsing."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return BWDownloader(str(tmp_path))

    def test_parse_dop_features(self, downloader):
        features = [
            {
                "properties": {
                    "dop_kachel": "324895420",
                    "befliegungsdatum": "2023-06-15",
                }
            },
            {
                "properties": {
                    "dop_kachel": "324905421",
                    "befliegungsdatum": "2022-07-20",
                }
            },
        ]
        result = downloader._parse_wfs_features(features, "dop_kachel", "befliegungsdatum")
        assert len(result) == 2
        assert result[0] == (489, 5420, date(2023, 6, 15))
        assert result[1] == (490, 5421, date(2022, 7, 20))

    def test_parse_skips_wrong_zone(self, downloader):
        features = [
            {
                "properties": {
                    "dop_kachel": "334895420",  # Zone 33, not 32
                    "befliegungsdatum": "2023-06-15",
                }
            },
        ]
        result = downloader._parse_wfs_features(features, "dop_kachel", "befliegungsdatum")
        assert len(result) == 0

    def test_parse_handles_missing_date(self, downloader):
        features = [
            {
                "properties": {
                    "dop_kachel": "324895420",
                    # Missing befliegungsdatum
                }
            },
        ]
        result = downloader._parse_wfs_features(features, "dop_kachel", "befliegungsdatum")
        assert len(result) == 1
        assert result[0] == (489, 5420, None)

    def test_parse_handles_short_tile_id(self, downloader):
        features = [
            {
                "properties": {
                    "dop_kachel": "12345",  # Too short
                    "befliegungsdatum": "2023-06-15",
                }
            },
        ]
        result = downloader._parse_wfs_features(features, "dop_kachel", "befliegungsdatum")
        assert len(result) == 0
