"""Tests for the `preview` module and CLI subcommand."""

from __future__ import annotations

import json
import re
from argparse import Namespace

import pytest

from georaffer.cli import run_preview, validate_preview_args
from georaffer.preview import (
    PreviewBBox,
    UtmGrid,
    build_preview_data,
    latlon_bbox_to_utm,
    parse_kml,
    render_preview_html,
    utm_bbox_to_wgs84,
    utm_grid_for_bbox,
)

_MINIMAL_KML = """<?xml version="1.0" encoding="utf-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <Placemark><Polygon><outerBoundaryIs><LinearRing>
    <coordinates>
      10.0,50.0 10.1,50.0 10.1,50.1 10.0,50.1 10.0,50.0
    </coordinates>
  </LinearRing></outerBoundaryIs></Polygon></Placemark>
  <Placemark><Polygon><outerBoundaryIs><LinearRing>
    <coordinates>10.05,50.05 10.2,50.05 10.2,50.2 10.05,50.2 10.05,50.05</coordinates>
  </LinearRing></outerBoundaryIs></Polygon></Placemark>
</Document>
</kml>
"""


class TestParseKml:
    def test_extracts_all_rings_and_bbox(self, tmp_path):
        kml = tmp_path / "two_polys.kml"
        kml.write_text(_MINIMAL_KML)

        rings, bbox = parse_kml(kml)

        assert len(rings) == 2
        # First ring: 5 points closing back to start.
        assert rings[0][0] == (10.0, 50.0)
        assert rings[0][-1] == (10.0, 50.0)
        # Overall bbox spans the union of both polygons.
        assert bbox == PreviewBBox(min_lon=10.0, min_lat=50.0, max_lon=10.2, max_lat=50.2)

    def test_handles_missing_namespace(self, tmp_path):
        """Some producers emit <coordinates> without the KML namespace."""
        kml = tmp_path / "no_ns.kml"
        kml.write_text(
            """<?xml version="1.0"?>
<kml><Placemark><Polygon><outerBoundaryIs><LinearRing>
<coordinates>1.0,2.0 1.5,2.5 1.0,2.0</coordinates>
</LinearRing></outerBoundaryIs></Polygon></Placemark></kml>"""
        )

        rings, bbox = parse_kml(kml)

        assert len(rings) == 1
        assert bbox == PreviewBBox(1.0, 2.0, 1.5, 2.5)

    def test_rejects_kml_with_no_geometry(self, tmp_path):
        kml = tmp_path / "empty.kml"
        kml.write_text("""<?xml version="1.0"?><kml><Document/></kml>""")
        with pytest.raises(ValueError, match="No coordinates"):
            parse_kml(kml)


class TestUtmGridForBbox:
    def test_snaps_to_tile_boundaries(self):
        """Grid extent snaps floor/ceil to 1km UTM tile origins."""
        grid = utm_grid_for_bbox(
            xmin=621172.60,
            ymin=5629775.47,
            xmax=632345.00,
            ymax=5637505.35,
            zone=32,
            margin=0,
            tile_size_m=1000,
        )
        assert grid.e_min == 621000
        assert grid.n_min == 5629000
        assert grid.e_max == 633000
        assert grid.n_max == 5638000
        assert grid.cols == 12
        assert grid.rows == 9
        assert grid.tile_count == 108

    def test_margin_adds_ring(self):
        grid = utm_grid_for_bbox(
            xmin=621172.60,
            ymin=5629775.47,
            xmax=632345.00,
            ymax=5637505.35,
            zone=32,
            margin=1,
            tile_size_m=1000,
        )
        # One ring extends 1 km outward on every side.
        assert grid.e_min == 620000
        assert grid.n_min == 5628000
        assert grid.e_max == 634000
        assert grid.n_max == 5639000
        assert grid.tile_count == 14 * 11  # 154

    def test_rejects_negative_margin(self):
        with pytest.raises(ValueError, match="margin"):
            utm_grid_for_bbox(0, 0, 1000, 1000, zone=32, margin=-1, tile_size_m=1000)

    def test_rejects_non_positive_tile_size(self):
        with pytest.raises(ValueError, match="tile_size_m"):
            utm_grid_for_bbox(0, 0, 1000, 1000, zone=32, margin=0, tile_size_m=0)


class TestProjectionHelpers:
    def test_latlon_bbox_to_utm_roundtrip(self):
        wgs = PreviewBBox(10.7, 50.8, 10.9, 50.9)
        xmin, ymin, xmax, ymax, zone = latlon_bbox_to_utm(wgs)
        assert zone == 32
        restored = utm_bbox_to_wgs84(xmin, ymin, xmax, ymax, zone)
        # Roundtrip within ~1e-5 degrees (UTM rotation affects extremes only marginally).
        assert abs(restored.min_lon - wgs.min_lon) < 1e-3
        assert abs(restored.max_lon - wgs.max_lon) < 1e-3

    def test_zone_mismatch_raises(self):
        # Spans UTM zone 32/33 boundary (meridian 12 E).
        with pytest.raises(ValueError, match="UTM zones"):
            latlon_bbox_to_utm(PreviewBBox(11.9, 50.0, 12.5, 50.1))


class TestBuildPreviewData:
    def test_without_margin_has_no_margin_grid(self):
        data = build_preview_data(
            utm_zone=32,
            utm_xmin=621172.60,
            utm_ymin=5629775.47,
            utm_xmax=632345.00,
            utm_ymax=5637505.35,
            margin=0,
            tile_size_m=1000,
        )
        assert data["marginGrid"] is None
        assert data["tileGrid"]["cols"] == 12
        assert data["tileGrid"]["rows"] == 9

    def test_with_margin_has_both_grids(self):
        data = build_preview_data(
            utm_zone=32,
            utm_xmin=621172.60,
            utm_ymin=5629775.47,
            utm_xmax=632345.00,
            utm_ymax=5637505.35,
            margin=1,
            tile_size_m=1000,
        )
        assert data["marginGrid"] is not None
        assert data["marginGrid"]["cols"] == 14
        assert data["marginGrid"]["rows"] == 11

    def test_kml_polygons_are_lat_lon_pairs(self):
        rings = [[(10.0, 50.0), (10.1, 50.0), (10.1, 50.1), (10.0, 50.0)]]
        data = build_preview_data(
            utm_zone=32,
            utm_xmin=621000,
            utm_ymin=5629000,
            utm_xmax=633000,
            utm_ymax=5638000,
            margin=0,
            tile_size_m=1000,
            kml_polygons=rings,
        )
        assert data["polygons"] is not None
        # (lon, lat) tuples become [lat, lon] lists for Leaflet.
        assert data["polygons"][0][0] == [50.0, 10.0]
        assert data["polygons"][0][1] == [50.0, 10.1]


class TestRenderPreviewHtml:
    def test_html_contains_data_json_and_leaflet(self):
        data = build_preview_data(
            utm_zone=32,
            utm_xmin=621172.60,
            utm_ymin=5629775.47,
            utm_xmax=632345.00,
            utm_ymax=5637505.35,
            margin=0,
            tile_size_m=1000,
            title="test-title",
        )
        html_text = render_preview_html(data)

        assert "<title>test-title</title>" in html_text
        assert "leaflet.js" in html_text
        assert "World_Imagery/MapServer/tile" in html_text

        match = re.search(r"const DATA = (\{.*?\});", html_text)
        assert match is not None, "DATA payload missing"
        payload = json.loads(match.group(1))
        assert payload["tileGrid"]["cols"] == 12

    def test_title_is_html_escaped(self):
        data = {
            "title": "<script>alert(1)</script>",
            "polygons": None,
            "inputBbox": None,
            "tileGrid": None,
            "marginGrid": None,
            "cornerLabels": [],
        }
        html_text = render_preview_html(data)
        assert "<script>alert(1)</script>" not in html_text
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_text


class TestValidatePreviewArgs:
    def _args(self, **kw) -> Namespace:
        defaults = {"bbox": None, "kml": None, "margin": 0}
        defaults.update(kw)
        return Namespace(**defaults)

    def test_rejects_both(self):
        errors = validate_preview_args(self._args(bbox="10,50,11,51", kml="a.kml"))
        assert any("exactly one" in e for e in errors)

    def test_rejects_neither(self):
        errors = validate_preview_args(self._args())
        assert any("exactly one" in e for e in errors)

    def test_accepts_bbox_only(self):
        errors = validate_preview_args(self._args(bbox="10,50,11,51"))
        assert errors == []

    def test_accepts_kml_only(self):
        errors = validate_preview_args(self._args(kml="a.kml"))
        assert errors == []

    def test_rejects_bad_bbox_format(self):
        errors = validate_preview_args(self._args(bbox="10,50,11"))
        assert any("4 comma-separated" in e for e in errors)

    def test_rejects_reversed_bbox(self):
        errors = validate_preview_args(self._args(bbox="11,50,10,51"))
        assert any("WEST" in e for e in errors)

    def test_rejects_negative_margin(self):
        errors = validate_preview_args(self._args(bbox="10,50,11,51", margin=-1))
        assert any("--margin" in e for e in errors)


class TestRunPreview:
    def _args(self, **kw) -> Namespace:
        defaults = {
            "bbox": None,
            "kml": None,
            "margin": 0,
            "utm_zone": None,
            "region": None,
            "output": None,
        }
        defaults.update(kw)
        return Namespace(**defaults)

    def test_writes_html_from_latlon_bbox(self, tmp_path):
        out = tmp_path / "preview.html"
        rc = run_preview(self._args(bbox="10.72,50.80,10.88,50.88", output=str(out)))
        assert rc == 0
        text = out.read_text()
        assert "tileGrid" in text
        assert "DATA =" in text

    def test_writes_html_from_utm_bbox_with_region(self, tmp_path):
        out = tmp_path / "preview.html"
        rc = run_preview(
            self._args(
                bbox="621172,5629775,632345,5637505",
                region="th",
                output=str(out),
            )
        )
        assert rc == 0
        assert out.exists()

    def test_utm_bbox_without_zone_or_region_raises(self):
        with pytest.raises(ValueError, match="UTM BBOX"):
            run_preview(self._args(bbox="621172,5629775,632345,5637505"))

    def test_latlon_bbox_with_utm_zone_raises(self):
        with pytest.raises(ValueError, match="lat/lon"):
            run_preview(self._args(bbox="10.72,50.80,10.88,50.88", utm_zone=32))

    def test_kml_input_with_utm_zone_raises(self, tmp_path):
        kml = tmp_path / "a.kml"
        kml.write_text(_MINIMAL_KML)
        with pytest.raises(ValueError, match="WGS84"):
            run_preview(self._args(kml=str(kml), utm_zone=32))

    def test_kml_input_writes_polygons(self, tmp_path):
        kml = tmp_path / "a.kml"
        kml.write_text(_MINIMAL_KML)
        out = tmp_path / "preview.html"
        rc = run_preview(self._args(kml=str(kml), margin=1, output=str(out)))
        assert rc == 0
        text = out.read_text()
        match = re.search(r"const DATA = (\{.*?\});", text)
        assert match is not None
        payload = json.loads(match.group(1))
        assert payload["polygons"] is not None
        assert len(payload["polygons"]) == 2
        assert payload["marginGrid"] is not None


class TestUtmGridRing:
    def test_ring_has_four_lat_lon_corners(self):
        grid = UtmGrid(
            utm_zone=32,
            e_min=621000,
            n_min=5629000,
            e_max=633000,
            n_max=5638000,
            tile_size_m=1000,
        )
        ring = grid.wgs84_ring()
        assert len(ring) == 4
        # Corners are ordered SW, NW, NE, SE — the NW latitude must exceed SW.
        assert ring[1][0] > ring[0][0]
