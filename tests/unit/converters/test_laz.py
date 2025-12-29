"""Tests for LAZ converter."""

import os

# Disable numba JIT during tests for speed/determinism
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from rasterio.enums import Resampling
from rasterio.transform import Affine

from georaffer.config import DSM_NODATA
from georaffer.converters.laz import _fill_raster_numba, convert_laz
from georaffer.converters.utils import resample_raster, write_geotiff


# Replace numba JIT with pure Python for tests to avoid compilation dependency
def _fill_raster_python(
    raster,
    x_int,
    y_int,
    z_int,
    x_scale,
    y_scale,
    z_scale,
    x_offset,
    y_offset,
    z_offset,
    min_x,
    max_y,
    inv_resolution,
    height,
    width,
):
    for i in range(len(x_int)):
        x = x_int[i] * x_scale + x_offset
        y = y_int[i] * y_scale + y_offset
        z = z_int[i] * z_scale + z_offset
        col = round((x - min_x) * inv_resolution)
        row = round((max_y - y) * inv_resolution)
        if 0 <= row < height and 0 <= col < width:
            raster[row, col] = z
    return raster


# Monkeypatch the module function for tests
import georaffer.converters.laz as laz_mod

laz_mod._fill_raster_numba = _fill_raster_python
_fill_raster_numba = _fill_raster_python


class TestFillRasterNumba:
    """Tests for the numba JIT rasterization function."""

    def test_single_point_at_origin(self):
        """Test single point placed correctly."""
        x = np.array([0.0])
        y = np.array([10.0])
        z = np.array([100.0])

        raster = np.full((11, 11), DSM_NODATA, dtype=np.float32)
        raster = _fill_raster_numba(
            raster, x, y, z, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 10.0, 1.0, 11, 11
        )

        # Point at (0, 10) with resolution 1.0, origin (0, 10)
        # col = round((0 - 0) / 1) = 0
        # row = round((10 - 10) / 1) = 0
        assert raster[0, 0] == pytest.approx(100.0)

    def test_grid_of_points(self):
        """Test regular grid fills correctly."""
        # 3x3 grid of points
        x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        y = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        z = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

        raster = np.full((3, 3), DSM_NODATA, dtype=np.float32)
        raster = _fill_raster_numba(
            raster, x, y, z, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 3, 3
        )

        expected = np.array(
            [
                [1.0, 2.0, 3.0],  # y=2 (top row)
                [4.0, 5.0, 6.0],  # y=1
                [7.0, 8.0, 9.0],  # y=0 (bottom row)
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_almost_equal(raster, expected)

    def test_nodata_for_missing_cells(self):
        """Test that unfilled cells have nodata value."""
        x = np.array([0.0])
        y = np.array([0.0])
        z = np.array([100.0])

        raster = np.full((3, 3), DSM_NODATA, dtype=np.float32)
        raster = _fill_raster_numba(
            raster, x, y, z, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 3, 3
        )

        # Only one point placed, rest should be nodata
        assert np.sum(raster == DSM_NODATA) == 8
        assert raster[2, 0] == pytest.approx(100.0)  # The one filled cell

    def test_out_of_bounds_points_ignored(self):
        """Test that points outside bounds are ignored."""
        x = np.array([-100.0, 0.0, 100.0])
        y = np.array([0.0, 0.0, 0.0])
        z = np.array([1.0, 2.0, 3.0])

        raster = np.full((2, 2), DSM_NODATA, dtype=np.float32)
        raster = _fill_raster_numba(
            raster, x, y, z, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2, 2
        )

        # Only middle point should be placed
        assert np.sum(raster != DSM_NODATA) == 1

    def test_subpixel_rounding(self):
        """Test that points are rounded to nearest pixel."""
        # Point at x=0.4 should round to col 0, x=0.6 should round to col 1
        x = np.array([0.4, 0.6])
        y = np.array([0.0, 0.0])
        z = np.array([10.0, 20.0])

        raster = np.full((2, 2), DSM_NODATA, dtype=np.float32)
        raster = _fill_raster_numba(
            raster, x, y, z, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2, 2
        )

        assert raster[1, 0] == pytest.approx(10.0)  # x=0.4 -> col 0
        assert raster[1, 1] == pytest.approx(20.0)  # x=0.6 -> col 1


class TestConvertLaz:
    """Tests for main convert_laz function."""

    @pytest.fixture
    def mock_laspy(self):
        """Mock laspy.open with regular grid data."""
        with patch("georaffer.converters.laz.laspy.open") as mock_open:
            mock_las = MagicMock()
            # Create a 3x3 regular grid
            x = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
            y = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
            z = np.array([100.0] * 9)
            mock_las.x = x
            mock_las.y = y
            mock_las.z = z
            mock_las.header = MagicMock()
            mock_las.header.creation_date = MagicMock(year=2023)
            mock_las.header.mins = (0.0, 0.0, 0.0)
            mock_las.header.maxs = (1.0, 1.0, 0.0)
            mock_las.header.scales = (0.01, 0.01, 0.01)
            mock_las.header.offsets = (0.0, 0.0, 0.0)
            mock_las.header.point_count = 9
            mock_las.chunk_iterator.return_value = [
                MagicMock(X=mock_las.x, Y=mock_las.y, Z=mock_las.z)
            ]
            mock_las.__enter__ = Mock(return_value=mock_las)
            mock_las.__exit__ = Mock(return_value=False)
            mock_open.return_value = mock_las
            yield mock_open, mock_las

    @pytest.fixture
    def mock_atomic_write(self):
        """Mock atomic_rasterio_write."""
        with patch("georaffer.converters.utils.atomic_rasterio_write") as mock:
            mock_dst = MagicMock()
            mock.return_value.__enter__ = Mock(return_value=mock_dst)
            mock.return_value.__exit__ = Mock(return_value=False)
            yield mock, mock_dst

    def test_convert_success_returns_true(
        self, mock_laspy, mock_atomic_write, tmp_path
    ):
        """Test successful conversion returns True."""
        mock_read, mock_las = mock_laspy
        mock_write, _ = mock_atomic_write

        output_path = str(tmp_path / "output" / "test.tif")
        result = convert_laz("/fake/input.laz", output_path, region="NRW", resolution=0.5)

        assert result is True

    def test_convert_failure_raises_error(self, tmp_path):
        """Test failed conversion raises RuntimeError."""
        with patch("georaffer.converters.laz.laspy.open") as mock_open:
            mock_open.side_effect = Exception("File not found")

            with pytest.raises(RuntimeError, match="Failed to convert"):
                convert_laz("/nonexistent/input.laz", str(tmp_path / "output.tif"), region="NRW")

    def test_convert_handles_missing_year(
        self, mock_laspy, mock_atomic_write, tmp_path
    ):
        """Test graceful handling when year is missing in header."""
        mock_read, mock_las = mock_laspy
        mock_las.header.creation_date = None

        result = convert_laz(
            "/fake/input.laz", str(tmp_path / "output.tif"), region="NRW", resolution=0.5
        )

        assert result is True

    def test_convert_validates_regular_grid(self, tmp_path):
        """Test that non-regular grids raise RuntimeError."""
        with patch("georaffer.converters.laz.laspy.open") as mock_open:
            mock_las = MagicMock()
            # Irregular: 5 points but grid expects different count
            mock_las.x = np.array([0.0, 0.5, 1.0, 0.0, 0.5])
            mock_las.y = np.array([1.0, 1.0, 1.0, 0.5, 0.5])
            mock_las.z = np.array([100.0] * 5)
            mock_las.header = MagicMock()
            mock_las.header.creation_date = None
            mock_las.header.mins = (0.0, 0.0, 0.0)
            mock_las.header.maxs = (1.0, 1.0, 0.0)
            mock_las.header.scales = (0.01, 0.01, 0.01)
            mock_las.header.offsets = (0.0, 0.0, 0.0)
            mock_las.header.point_count = 5
            mock_las.chunk_iterator.return_value = [
                MagicMock(X=mock_las.x, Y=mock_las.y, Z=mock_las.z)
            ]
            mock_las.__enter__ = Mock(return_value=mock_las)
            mock_las.__exit__ = Mock(return_value=False)
            mock_open.return_value = mock_las

            with pytest.raises(RuntimeError, match="Non-regular point cloud"):
                convert_laz(
                    "/fake/input.laz", str(tmp_path / "output.tif"), region="NRW", resolution=0.5
                )


class TestResampleDsm:
    """Tests for resample_raster function (DSM use case)."""

    def test_resample_produces_correct_shape(self):
        """Test resampled output has correct dimensions."""
        data = np.zeros((1000, 1000), dtype=np.float32)
        transform = Affine(0.5, 0, 350000, 0, -0.5, 5601000)

        with patch("georaffer.converters.utils.rasterio.warp.reproject"):
            out_data, _ = resample_raster(
                data,
                transform,
                "EPSG:25832",
                500,
                num_threads=2,
                dtype=np.float32,
                resampling=Resampling.bilinear,
                nodata=DSM_NODATA,
            )

        assert out_data.shape == (500, 500)

    def test_resample_adjusts_transform(self):
        """Test transform is scaled correctly."""
        data = np.zeros((1000, 1000), dtype=np.float32)
        transform = Affine(0.5, 0, 350000, 0, -0.5, 5601000)

        with patch("georaffer.converters.utils.rasterio.warp.reproject"):
            _, out_transform = resample_raster(
                data,
                transform,
                "EPSG:25832",
                500,
                num_threads=2,
                dtype=np.float32,
                resampling=Resampling.bilinear,
                nodata=DSM_NODATA,
            )

        # 1000 -> 500 = scale factor 2
        assert out_transform.a == pytest.approx(1.0)
        assert out_transform.e == pytest.approx(-1.0)


class TestWriteDsmGeotiff:
    """Tests for DSM GeoTIFF writing via write_geotiff."""

    def test_write_sets_point_tag(self, tmp_path):
        """Test AREA_OR_POINT is set to 'Point' for DSM."""
        data = np.zeros((100, 100), dtype=np.float32)
        transform = Affine(0.5, 0, 350000, 0, -0.5, 5601000)

        with patch("georaffer.converters.utils.atomic_rasterio_write") as mock_write:
            mock_dst = MagicMock()
            mock_write.return_value.__enter__ = Mock(return_value=mock_dst)
            mock_write.return_value.__exit__ = Mock(return_value=False)

            write_geotiff(
                tmp_path / "out.tif",
                data,
                transform,
                "EPSG:25832",
                dtype="float32",
                count=1,
                nodata=DSM_NODATA,
                area_or_point="Point",
            )

        mock_dst.update_tags.assert_called_once_with(AREA_OR_POINT="Point")

    def test_write_sets_nodata_value(self, tmp_path):
        """Test nodata value is set in profile."""
        data = np.zeros((100, 100), dtype=np.float32)
        transform = Affine(0.5, 0, 350000, 0, -0.5, 5601000)

        with patch("georaffer.converters.utils.atomic_rasterio_write") as mock_write:
            mock_dst = MagicMock()
            mock_write.return_value.__enter__ = Mock(return_value=mock_dst)
            mock_write.return_value.__exit__ = Mock(return_value=False)

            write_geotiff(
                tmp_path / "out.tif",
                data,
                transform,
                "EPSG:25832",
                dtype="float32",
                count=1,
                nodata=DSM_NODATA,
                area_or_point="Point",
            )

        call_kwargs = mock_write.call_args[1]
        assert call_kwargs["nodata"] == DSM_NODATA


class TestConvertLazEdgeCases:
    """Edge case tests for LAZ converter."""

    def test_empty_point_cloud(self, tmp_path):
        """Test handling of empty point cloud raises RuntimeError."""
        with patch("georaffer.converters.laz.laspy.read") as mock_read:
            mock_las = MagicMock()
            mock_las.x = np.array([])
            mock_las.y = np.array([])
            mock_las.z = np.array([])
            mock_las.header = MagicMock()
            mock_las.header.creation_date = None
            mock_read.return_value = mock_las

            with pytest.raises(RuntimeError, match="Failed to convert"):
                convert_laz("/fake/empty.laz", str(tmp_path / "output.tif"), region="NRW")
