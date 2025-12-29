"""Tests for JP2 converter."""

import os
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from rasterio.enums import Resampling
from rasterio.transform import Affine

from georaffer.converters.jp2 import _convert_split_jp2, convert_jp2
from georaffer.converters.utils import resample_raster, write_geotiff


class TestConvertJP2:
    """Tests for main convert_jp2 function."""

    @pytest.fixture(autouse=True)
    def enable_wms(self):
        """Ensure WMS lookups are enabled for converter tests."""
        original = os.environ.get("GEORAFFER_DISABLE_WMS")
        os.environ.pop("GEORAFFER_DISABLE_WMS", None)
        yield
        if original is not None:
            os.environ["GEORAFFER_DISABLE_WMS"] = original

    @pytest.fixture
    def mock_rasterio(self):
        """Mock rasterio.open context manager."""
        with patch("georaffer.converters.jp2.rasterio.open") as mock_open:
            mock_src = MagicMock()
            mock_src.read.return_value = np.zeros((3, 1000, 1000), dtype=np.uint8)
            mock_src.transform = Affine(0.2, 0, 350000, 0, -0.2, 5601000)
            mock_src.crs = "EPSG:25832"
            mock_src.__enter__ = Mock(return_value=mock_src)
            mock_src.__exit__ = Mock(return_value=False)
            mock_open.return_value = mock_src
            yield mock_open, mock_src

    @pytest.fixture
    def mock_atomic_write(self):
        """Mock atomic_rasterio_write."""
        with patch("georaffer.converters.utils.atomic_rasterio_write") as mock:
            mock_dst = MagicMock()
            mock.return_value.__enter__ = Mock(return_value=mock_dst)
            mock.return_value.__exit__ = Mock(return_value=False)
            yield mock, mock_dst

    def test_convert_single_output_success(
        self, mock_rasterio, mock_atomic_write, tmp_path
    ):
        """Test basic JP2 to GeoTIFF conversion."""
        mock_open, mock_src = mock_rasterio
        mock_write, mock_dst = mock_atomic_write

        output_path = str(tmp_path / "output" / "test.tif")
        result = convert_jp2("/fake/input.jp2", output_path, region="NRW", year="2021")

        assert result is True
        mock_open.assert_called_once()
        args, kwargs = mock_open.call_args
        assert args[0] == "/fake/input.jp2"
        assert kwargs.get("USE_TILE_AS_BLOCK") == "YES"
        mock_write.assert_called_once()

    def test_convert_dict_output_multiple_resolutions(
        self, mock_rasterio, mock_atomic_write, tmp_path
    ):
        """Test conversion to multiple resolutions."""
        mock_open, mock_src = mock_rasterio
        mock_write, mock_dst = mock_atomic_write

        output_paths = {
            1000: str(tmp_path / "out_1000.tif"),
            500: str(tmp_path / "out_500.tif"),
        }

        with patch("georaffer.converters.jp2.resample_raster") as mock_resample:
            mock_resample.return_value = (
                np.zeros((3, 500, 500), dtype=np.uint8),
                Affine(0.4, 0, 350000, 0, -0.4, 5601000),
            )

            result = convert_jp2(
                "/fake/input.jp2", output_paths, region="NRW", year="2021", resolutions=[1000, 500]
            )

        assert result is True
        assert mock_write.call_count == 2
        assert mock_resample.call_count == 2

    def test_convert_failure_raises_error(self, tmp_path):
        """Test that conversion failure raises RuntimeError."""
        with patch("georaffer.converters.jp2.rasterio.open") as mock_open:
            mock_open.side_effect = Exception("File not found")

            with pytest.raises(RuntimeError, match="Failed to convert"):
                convert_jp2(
                    "/nonexistent/input.jp2",
                    str(tmp_path / "output.tif"),
                    region="NRW",
                    year="2021",
                )

    def test_convert_rlp_triggers_split(self, mock_rasterio, mock_atomic_write, mock_wms, tmp_path):
        """Test RLP tiles trigger splitting when enabled."""
        mock_open, mock_src = mock_rasterio
        # RLP 2km tile at 0.2m resolution = 10000x10000 pixels
        mock_src.read.return_value = np.zeros((3, 10000, 10000), dtype=np.uint8)

        with patch("georaffer.converters.jp2._convert_split_jp2") as mock_split:
            mock_split.return_value = True
            with patch("georaffer.converters.jp2.parse_tile_coords") as mock_parse:
                mock_parse.return_value = (362, 5604)

                result = convert_jp2(
                    "/fake/dop20rgb_32_362_5604_2_rp_2023.jp2",
                    str(tmp_path / "output.tif"),
                    region="RLP",
                    year="2023",
                )

        assert result is True
        mock_split.assert_called_once()


class TestResampleTile:
    """Tests for resample_raster function (JP2/imagery use case)."""

    def test_resample_calculates_correct_transform(self):
        """Test that resampling produces correct output transform."""
        data = np.zeros((3, 1000, 1000), dtype=np.uint8)
        src_transform = Affine(0.2, 0, 350000, 0, -0.2, 5601000)
        target_size = 500

        with patch("georaffer.converters.utils.rasterio.warp.reproject") as mock_reproject:
            out_data, out_transform = resample_raster(
                data,
                src_transform,
                "EPSG:25832",
                target_size,
                num_threads=2,
                dtype=np.uint8,
                resampling=Resampling.lanczos,
            )

        # Scale factor is 1000/500 = 2
        assert out_transform.a == pytest.approx(0.4)  # 0.2 * 2
        assert out_transform.e == pytest.approx(-0.4)  # -0.2 * 2
        assert out_data.shape == (3, 500, 500)

    def test_resample_preserves_origin(self):
        """Test that resampling preserves the origin coordinates."""
        data = np.zeros((3, 1000, 1000), dtype=np.uint8)
        src_transform = Affine(0.2, 0, 350000, 0, -0.2, 5601000)

        with patch("georaffer.converters.utils.rasterio.warp.reproject"):
            _, out_transform = resample_raster(
                data,
                src_transform,
                "EPSG:25832",
                500,
                num_threads=2,
                dtype=np.uint8,
                resampling=Resampling.lanczos,
            )

        assert out_transform.c == 350000  # X origin unchanged
        assert out_transform.f == 5601000  # Y origin unchanged


class TestWriteGeotiff:
    """Tests for write_geotiff function."""

    def test_write_creates_output_directory(self, tmp_path):
        """Test that output directory is created if missing."""
        data = np.zeros((3, 100, 100), dtype=np.uint8)
        transform = Affine(0.2, 0, 350000, 0, -0.2, 5601000)
        output_path = tmp_path / "nested" / "deep" / "output.tif"

        with patch("georaffer.converters.utils.atomic_rasterio_write") as mock_write:
            mock_dst = MagicMock()
            mock_write.return_value.__enter__ = Mock(return_value=mock_dst)
            mock_write.return_value.__exit__ = Mock(return_value=False)

            write_geotiff(output_path, data, transform, "EPSG:25832")

        assert output_path.parent.exists()

    def test_write_sets_correct_profile(self, tmp_path):
        """Test that GeoTIFF profile is set correctly."""
        data = np.zeros((3, 100, 100), dtype=np.uint8)
        transform = Affine(0.2, 0, 350000, 0, -0.2, 5601000)
        output_path = tmp_path / "output.tif"

        with patch("georaffer.converters.utils.atomic_rasterio_write") as mock_write:
            mock_dst = MagicMock()
            mock_write.return_value.__enter__ = Mock(return_value=mock_dst)
            mock_write.return_value.__exit__ = Mock(return_value=False)

            write_geotiff(output_path, data, transform, "EPSG:25832")

        call_kwargs = mock_write.call_args[1]
        assert call_kwargs["driver"] == "GTiff"
        assert call_kwargs["dtype"] == "uint8"
        assert call_kwargs["count"] == 3
        assert call_kwargs["height"] == 100
        assert call_kwargs["width"] == 100
        assert call_kwargs["tiled"] is True

    def test_write_sets_area_or_point_tag(self, tmp_path):
        """Test that AREA_OR_POINT tag is set to 'Area' for orthophotos."""
        data = np.zeros((3, 100, 100), dtype=np.uint8)
        transform = Affine(0.2, 0, 350000, 0, -0.2, 5601000)
        output_path = tmp_path / "output.tif"

        with patch("georaffer.converters.utils.atomic_rasterio_write") as mock_write:
            mock_dst = MagicMock()
            mock_write.return_value.__enter__ = Mock(return_value=mock_dst)
            mock_write.return_value.__exit__ = Mock(return_value=False)

            write_geotiff(output_path, data, transform, "EPSG:25832")

        mock_dst.update_tags.assert_called_once_with(AREA_OR_POINT="Area")


class TestConvertSplitRlpJP2:
    """Tests for RLP tile splitting."""

    def test_split_creates_four_tiles(self):
        """Test that splitting creates exactly 4 tiles."""
        data = np.zeros((3, 10000, 10000), dtype=np.uint8)
        transform = Affine(0.2, 0, 362000, 0, -0.2, 5606000)
        coords = (362, 5604)
        output_paths = {None: "/output/rlp_32_362000_5604000_2023.tif"}
        resolutions = [None]

        with patch("georaffer.converters.jp2.write_geotiff") as mock_write:
            result = _convert_split_jp2(
                data,
                transform,
                "EPSG:25832",
                coords,
                output_paths,
                resolutions,
                "test.jp2",
                "RLP",
                "2023",
                2,
            )

        assert result is True
        assert mock_write.call_count == 4

    def test_split_tile_coordinates(self):
        """Test that split tiles have correct grid coordinates."""
        data = np.zeros((3, 10000, 10000), dtype=np.uint8)
        transform = Affine(0.2, 0, 362000, 0, -0.2, 5606000)
        coords = (362, 5604)
        output_paths = {None: "/output/rlp_32_362000_5604000_2023.tif"}
        resolutions = [None]

        written_paths = []
        captured_kwargs = []

        def fake_generate(p, x, y, **kwargs):
            captured_kwargs.append(kwargs)
            return f"/output/rlp_{kwargs['utm_zone']}_{kwargs['easting']}_{kwargs['northing']}_2023.tif"

        with patch("georaffer.converters.jp2.write_geotiff") as mock_write:
            with patch("georaffer.converters.jp2.generate_split_output_path") as mock_path:
                mock_path.side_effect = fake_generate
                _convert_split_jp2(
                    data,
                    transform,
                    "EPSG:25832",
                    coords,
                    output_paths,
                    resolutions,
                    "test.jp2",
                    "RLP",
                    "2023",
                    2,
                )
                written_paths = [call[0][0] for call in mock_write.call_args_list]

        # Should create tiles at (362,5604), (363,5604), (362,5605), (363,5605)
        expected_paths = [
            "/output/rlp_32_362000_5604000_2023.tif",  # SW
            "/output/rlp_32_363000_5604000_2023.tif",  # SE
            "/output/rlp_32_362000_5605000_2023.tif",  # NW
            "/output/rlp_32_363000_5605000_2023.tif",  # NE
        ]
        assert sorted(written_paths) == sorted(expected_paths)
        # UTM coordinates should be provided for all split tiles
        for kw in captured_kwargs:
            assert kw.get("easting") is not None
            assert kw.get("northing") is not None

    def test_split_tile_dimensions(self):
        """Test that each split tile has correct dimensions."""
        data = np.zeros((3, 10000, 10000), dtype=np.uint8)
        transform = Affine(0.2, 0, 362000, 0, -0.2, 5606000)
        coords = (362, 5604)
        output_paths = {None: "/output/rlp_32_362000_5604000_2023.tif"}
        resolutions = [None]

        tile_shapes = []
        with patch("georaffer.converters.jp2.write_geotiff") as mock_write:
            with patch(
                "georaffer.converters.jp2.generate_split_output_path", return_value="/out.tif"
            ):
                _convert_split_jp2(
                    data,
                    transform,
                    "EPSG:25832",
                    coords,
                    output_paths,
                    resolutions,
                    "test.jp2",
                    "RLP",
                    "2023",
                    2,
                )
                tile_shapes = [call[0][1].shape for call in mock_write.call_args_list]

        # Each tile should be 5000x5000 pixels (half of 10000)
        for shape in tile_shapes:
            assert shape == (3, 5000, 5000)

    def test_split_uses_current_resolution_half(self):
        """Split should honor the actual read size (e.g., overview)."""
        data = np.zeros((3, 312, 312), dtype=np.uint8)  # corresponds to an overview read
        transform = Affine(6.4, 0, 362000, 0, -6.4, 5606000)
        coords = (362, 5604)
        output_paths = {None: "/output/rlp_32_362000_5604000_2023.tif"}
        resolutions = [None]

        tile_shapes = []
        with patch("georaffer.converters.jp2.write_geotiff") as mock_write:
            with patch(
                "georaffer.converters.jp2.generate_split_output_path", return_value="/out.tif"
            ):
                _convert_split_jp2(
                    data,
                    transform,
                    "EPSG:25832",
                    coords,
                    output_paths,
                    resolutions,
                    "test.jp2",
                    "RLP",
                    "2023",
                    2,
                )
                tile_shapes = [call[0][1].shape for call in mock_write.call_args_list]

        # Data width 312 -> half 156 per split
        for shape in tile_shapes:
            assert shape == (3, 156, 156)
