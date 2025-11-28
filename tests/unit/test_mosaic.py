"""Tests for mosaic_tiles.py helper functions."""

import os
import sys

# Allow importing from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from mosaic_tiles import _parse_coords, _season_from_date


class TestSeasonFromDate:
    """Tests for _season_from_date()."""

    def test_winter_december(self):
        assert _season_from_date("2023-12-15") == "Winter"

    def test_winter_january(self):
        assert _season_from_date("2023-01-15") == "Winter"

    def test_winter_february(self):
        assert _season_from_date("2023-02-28") == "Winter"

    def test_spring_march(self):
        assert _season_from_date("2023-03-21") == "Spring"

    def test_spring_april(self):
        assert _season_from_date("2023-04-15") == "Spring"

    def test_spring_may(self):
        assert _season_from_date("2023-05-01") == "Spring"

    def test_summer_june(self):
        assert _season_from_date("2023-06-21") == "Summer"

    def test_summer_july(self):
        assert _season_from_date("2023-07-15") == "Summer"

    def test_summer_august(self):
        assert _season_from_date("2023-08-31") == "Summer"

    def test_autumn_september(self):
        assert _season_from_date("2023-09-22") == "Autumn"

    def test_autumn_october(self):
        assert _season_from_date("2023-10-15") == "Autumn"

    def test_autumn_november(self):
        assert _season_from_date("2023-11-30") == "Autumn"

    def test_invalid_format_returns_unknown(self):
        assert _season_from_date("not-a-date") == "unknown"

    def test_empty_string_returns_unknown(self):
        assert _season_from_date("") == "unknown"

    def test_year_only_returns_unknown(self):
        assert _season_from_date("2023") == "unknown"

    def test_none_like_returns_unknown(self):
        # Function receives string, but empty/malformed should be safe
        assert _season_from_date("None") == "unknown"


class TestParseCoords:
    """Tests for _parse_coords()."""

    def test_explicit_grid_coords(self):
        row = {"grid_x": "350", "grid_y": "5600"}
        assert _parse_coords(row) == (350, 5600)

    def test_explicit_grid_coords_as_int(self):
        row = {"grid_x": 350, "grid_y": 5600}
        assert _parse_coords(row) == (350, 5600)

    def test_fallback_to_processed_file(self):
        # processed_file with grid coords pattern (3-5 digit x, 4 digit y)
        row = {"grid_x": None, "grid_y": None, "processed_file": "nrw_32_350_5600_2021.tif"}
        gx, gy = _parse_coords(row)
        assert (gx, gy) == (350, 5600)

    def test_fallback_to_source_file(self):
        row = {"grid_x": "", "grid_y": "", "source_file": "dop10rgbi_32_350_5600_1_nw_2021.jp2"}
        gx, gy = _parse_coords(row)
        assert (gx, gy) == (350, 5600)

    def test_none_grid_coords_string(self):
        row = {
            "grid_x": "None",
            "grid_y": "None",
            "source_file": "dop10rgbi_32_362_5604_1_nw_2021.jp2",
        }
        gx, gy = _parse_coords(row)
        assert (gx, gy) == (362, 5604)

    def test_empty_row_returns_none(self):
        row = {}
        assert _parse_coords(row) == (None, None)

    def test_no_parseable_coords_returns_none(self):
        row = {"grid_x": None, "processed_file": "random.tif", "source_file": "unknown.jp2"}
        assert _parse_coords(row) == (None, None)

    def test_partial_grid_coords_falls_back(self):
        # Only grid_x set, should fallback to filename
        row = {
            "grid_x": "350",
            "grid_y": None,
            "source_file": "dop10rgbi_32_360_5610_1_nw_2021.jp2",
        }
        gx, gy = _parse_coords(row)
        assert (gx, gy) == (360, 5610)
