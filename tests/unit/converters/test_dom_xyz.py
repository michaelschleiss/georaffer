"""Tests for TH DOM XYZ conversion."""

import rasterio

from georaffer.converters.dom_xyz import convert_dom_xyz_to_geotiff


class TestConvertDomXyzToGeotiff:
    def test_row_major_bottom_to_top_is_normalized(self, tmp_path):
        xyz_path = tmp_path / "bottom_to_top.xyz"
        tif_path = tmp_path / "bottom_to_top.tif"
        xyz_path.write_text(
            "0.5 0.5 10\n"
            "1.5 0.5 11\n"
            "0.5 1.5 20\n"
            "1.5 1.5 21\n",
            encoding="utf-8",
        )

        convert_dom_xyz_to_geotiff(xyz_path, tif_path, expected_resolution=1.0, quiet=True)

        with rasterio.open(tif_path) as dataset:
            assert dataset.read(1).tolist() == [[20.0, 21.0], [10.0, 11.0]]

    def test_row_major_top_to_bottom_is_preserved(self, tmp_path):
        xyz_path = tmp_path / "top_to_bottom.xyz"
        tif_path = tmp_path / "top_to_bottom.tif"
        xyz_path.write_text(
            "0.5 1.5 20\n"
            "1.5 1.5 21\n"
            "0.5 0.5 10\n"
            "1.5 0.5 11\n",
            encoding="utf-8",
        )

        convert_dom_xyz_to_geotiff(xyz_path, tif_path, expected_resolution=1.0, quiet=True)

        with rasterio.open(tif_path) as dataset:
            assert dataset.read(1).tolist() == [[20.0, 21.0], [10.0, 11.0]]
