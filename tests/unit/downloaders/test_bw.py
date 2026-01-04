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
        with pytest.raises(ValueError, match="2010"):
            BWDownloader(str(tmp_path), imagery_from=(2005, None))


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
        assert jp2_coords == (489, 5420)
        assert laz_coords == (489, 5420)

    def test_even_easting_rounds_to_odd(self, downloader):
        # Even easting should round down to nearest odd
        jp2_coords, _ = downloader.utm_to_grid_coords(490000, 5420000)
        assert jp2_coords[0] == 489

    def test_odd_northing_rounds_to_even(self, downloader):
        # Odd northing should round down to nearest even
        jp2_coords, _ = downloader.utm_to_grid_coords(489000, 5421000)
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
            ("dop20rgb_32_489_5420_1_bw_2024.tif", (489, 5420)),
            ("dop20rgb_32_391_5268_1_bw_2019.tif", (391, 5268)),
            ("dop20rgb_32_489_5420_1_bw_2024.jpg", (489, 5420)),
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
            ("dop20rgb_32_489_5420_1_bw_2024.tif", (489, 5420, "2024")),
            ("dop20rgb_32_489_5420_1_bw_2024.jpg", (489, 5420, "2024")),
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
            ("dom1_32_489_5420_1_bw_2019.tif", (489, 5420)),
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
        with pytest.raises(ValueError, match="UTM zone"):
            downloader._parse_wfs_features(features, "dop_kachel", "befliegungsdatum")

    def test_parse_returns_none_for_missing_date(self, downloader):
        """Missing date fields return None (caller handles fallback logic)."""
        features = [
            {
                "properties": {
                    "dop_kachel": "324895420",
                    # Missing befliegungsdatum
                }
            },
        ]
        result = downloader._parse_wfs_features(features, "dop_kachel", "befliegungsdatum")
        assert result == [(489, 5420, None)]

    def test_parse_handles_short_tile_id(self, downloader):
        features = [
            {
                "properties": {
                    "dop_kachel": "12345",  # Too short
                    "befliegungsdatum": "2023-06-15",
                }
            },
        ]
        with pytest.raises(ValueError, match="tile id"):
            downloader._parse_wfs_features(features, "dop_kachel", "befliegungsdatum")


class TestBWCatalog:
    """Tests for BW catalog loading behavior."""

    def test_missing_befliegungsdatum_raises(self, tmp_path, monkeypatch):
        downloader = BWDownloader(str(tmp_path))

        def fake_fetch(wfs_url, feature_type, id_field, date_field):
            if date_field == "befliegungsdatum":
                return [(489, 5420, None)]
            return []

        monkeypatch.setattr(downloader, "_fetch_wfs_tiles", fake_fetch)

        with pytest.raises(ValueError, match="befliegungsdatum"):
            downloader._load_catalog()

    def test_missing_fortfuehrungsdatum_uses_dop_date(self, tmp_path, monkeypatch):
        """When DOM date is missing, falls back to corresponding DOP date."""
        downloader = BWDownloader(str(tmp_path))

        def fake_fetch(wfs_url, feature_type, id_field, date_field):
            if date_field == "befliegungsdatum":
                return [(489, 5420, date(2022, 5, 1))]
            if date_field == "fortfuehrungsdatum":
                return [(489, 5420, None)]
            return []

        monkeypatch.setattr(downloader, "_fetch_wfs_tiles", fake_fetch)

        catalog = downloader._load_catalog()
        dom_years = catalog.dsm_tiles.get((489, 5420), {})
        assert 2022 in dom_years
        assert dom_years[2022]["acquisition_date"] == "2022-05-01"

    def test_missing_fortfuehrungsdatum_without_dop_skips(self, tmp_path, monkeypatch):
        """DOM tile without corresponding DOP is skipped (no date to fall back to)."""
        downloader = BWDownloader(str(tmp_path))

        def fake_fetch(wfs_url, feature_type, id_field, date_field):
            if date_field == "befliegungsdatum":
                return []
            if date_field == "fortfuehrungsdatum":
                return [(489, 5420, None)]
            return []

        monkeypatch.setattr(downloader, "_fetch_wfs_tiles", fake_fetch)

        catalog = downloader._load_catalog()
        assert (489, 5420) not in catalog.dsm_tiles

    def test_dop_without_dom_is_dropped(self, tmp_path, monkeypatch):
        """DOP tile without corresponding DOM is not included in catalog."""
        downloader = BWDownloader(str(tmp_path))

        def fake_fetch(wfs_url, feature_type, id_field, date_field):
            if date_field == "befliegungsdatum":
                return [(489, 5420, date(2022, 5, 1))]
            if date_field == "fortfuehrungsdatum":
                return []
            return []

        monkeypatch.setattr(downloader, "_fetch_wfs_tiles", fake_fetch)

        catalog = downloader._load_catalog()
        assert (489, 5420) not in catalog.image_tiles



class TestBWWfsPagination:
    """Tests for BW WFS pagination preflight behavior."""

    def test_missing_numbermatched_raises(self, tmp_path, monkeypatch):
        downloader = BWDownloader(str(tmp_path))

        class DummyResponse:
            text = "<wfs:FeatureCollection></wfs:FeatureCollection>"

            def raise_for_status(self):
                return None

        def fake_get(*_args, **_kwargs):
            return DummyResponse()

        monkeypatch.setattr(downloader._session, "get", fake_get)

        with pytest.raises(ValueError, match="numberMatched"):
            downloader._fetch_wfs_tiles("https://example.com", "feat", "id", "date")
