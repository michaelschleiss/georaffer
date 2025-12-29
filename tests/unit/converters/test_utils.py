"""Tests for converter utilities."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from georaffer.converters.utils import (
    atomic_rasterio_write,
    generate_split_output_path,
    parse_rlp_tile_coords,
    parse_tile_coords,
    resample_raster,
)


class TestParseTileCoords:
    """Tests for unified parse_tile_coords function."""

    def test_nrw_jp2(self):
        """NRW JP2 files have zone prefix separated by underscore."""
        coords = parse_tile_coords("dop10rgbi_32_350_5600_1_nw_2021.jp2")
        assert coords == (350, 5600)

    def test_nrw_jp2_missing_underscore_after_zone(self):
        """Historic NRW JP2 files may omit the underscore after '32'."""
        coords = parse_tile_coords("dop10rgbi_32288_5736_1_nw_2015.jp2")
        assert coords == (288, 5736)

    def test_nrw_laz_strips_zone_prefix(self):
        """NRW LAZ files concatenate zone with x-coord (32350 -> 350)."""
        coords = parse_tile_coords("bdom50_32350_5600_1_nw_2025.laz")
        assert coords == (350, 5600), "Should strip '32' zone prefix from LAZ filename"

    def test_nrw_laz_various_coords(self):
        """Test multiple NRW LAZ coordinate extractions."""
        assert parse_tile_coords("bdom50_32400_5700_1_nw_2024.laz") == (400, 5700)
        assert parse_tile_coords("bdom50_32123_5999_1_nw_2023.laz") == (123, 5999)

    def test_rlp_jp2(self):
        """RLP JP2 files."""
        coords = parse_tile_coords("dop20rgb_32_362_5604_2_rp_2023.jp2")
        assert coords == (362, 5604)

    def test_rlp_laz(self):
        """RLP LAZ files."""
        coords = parse_tile_coords("bdom20rgbi_32_364_5582_2_rp.laz")
        assert coords == (364, 5582)

    def test_bb_bdom_zip(self):
        """BB bDOM raw ZIP files."""
        coords = parse_tile_coords("bdom_33250-5888.zip")
        assert coords == (250, 5888)

    def test_bb_dop_zip(self):
        """BB DOP raw ZIP files."""
        coords = parse_tile_coords("dop_33250-5888.zip")
        assert coords == (250, 5888)

    def test_output_file_nrw(self):
        """Processed NRW output files."""
        assert parse_tile_coords("nrw_32_350000_5600000_2021.tif") == (350000, 5600000)
        assert parse_tile_coords("nrw_32_350000_5600000_2021_1000.tif") == (350000, 5600000)

    def test_output_file_rlp(self):
        """Processed RLP output files."""
        assert parse_tile_coords("rlp_32_362000_5604000_2023.tif") == (362000, 5604000)

    def test_output_file_utm_coords(self):
        """Output files with full UTM coordinates (from splits)."""
        assert parse_tile_coords("nrw_32_350500_5600000_2021.tif") == (350500, 5600000)

    def test_output_file_legacy_rejected(self):
        """Legacy grid-based output names are not accepted."""
        assert parse_tile_coords("nrw_32_350_5600_2021.tif") is None

    def test_invalid_returns_none(self):
        """Non-matching filenames return None."""
        assert parse_tile_coords("random_file.jp2") is None
        assert parse_tile_coords("something.tif") is None


class TestParseRLPCoords:
    def test_jp2_with_year(self):
        coords = parse_rlp_tile_coords("dop20rgb_32_362_5604_2_rp_2023.jp2")
        assert coords == (362, 5604)

    def test_laz(self):
        coords = parse_rlp_tile_coords("bdom20rgbi_32_364_5582_2_rp.laz")
        assert coords == (364, 5582)

    def test_invalid(self):
        coords = parse_rlp_tile_coords("random_file.jp2")
        assert coords is None


class TestGenerateSplitPath:
    def test_standard(self):
        path = generate_split_output_path(
            "/output/rlp_32_362000_5604000_2023.tif",
            363,
            5605,
            easting=363000,
            northing=5605000,
        )
        assert path == Path("/output/rlp_32_363000_5605000_2023.tif")

    def test_with_resolution(self):
        path = generate_split_output_path(
            "/output/rlp_32_362000_5604000_2023_1000.tif",
            363,
            5605,
            easting=363000,
            northing=5605000,
        )
        assert path == Path("/output/rlp_32_363000_5605000_2023.tif")

    def test_missing_utm_coords_raises(self):
        """Split outputs require explicit UTM coords."""
        with pytest.raises(ValueError):
            generate_split_output_path("/output/rlp_32_362000_5604000_2023.tif", 363, 5605)

    def test_uses_utm_when_provided(self):
        """Splits embed UTM coordinates (zone 32) to avoid collisions."""
        # Input format: {region}_{zone}_{easting}_{northing}_{year}.tif (5 parts)
        path = generate_split_output_path(
            "/output/nrw_32_350000_5600000_2023.tif",
            351,
            5600,
            easting=350500,
            northing=5600000,
        )
        assert path == Path("/output/nrw_32_350500_5600000_2023.tif")

    def test_utm_format_extracts_year_not_northing(self):
        """Verify year is extracted from parts[4], not parts[3] (northing)."""
        # This test catches the bug where northing was used as year
        path = generate_split_output_path(
            "/output/rlp_32_360000_5600000_2024.tif",
            361,
            5601,
            easting=360500,
            northing=5600500,
        )
        # Year should be 2024, not 5600000
        assert path == Path("/output/rlp_32_360500_5600500_2024.tif")
        assert "5600000" not in path.name.replace("5600500", "")  # northing not duplicated

    def test_utm_format_various_regions(self):
        """Test UTM naming works for both NRW and RLP regions."""
        # NRW
        path_nrw = generate_split_output_path(
            "/output/nrw_32_369000_5621000_2023.tif",
            370,
            5622,
            easting=369500,
            northing=5621500,
        )
        assert path_nrw == Path("/output/nrw_32_369500_5621500_2023.tif")

        # RLP
        path_rlp = generate_split_output_path(
            "/output/rlp_32_377000_5595000_2022.tif",
            378,
            5596,
            easting=377500,
            northing=5595500,
        )
        assert path_rlp == Path("/output/rlp_32_377500_5595500_2022.tif")

    def test_legacy_format_raises(self):
        """Legacy 4-part format is rejected."""
        with pytest.raises(ValueError):
            generate_split_output_path(
                "/output/rlp_362_5604_2023.tif",
                363,
                5605,
                easting=362500,
                northing=5604500,
            )

    def test_utm_format_preserves_year_with_different_values(self):
        """Ensure various year values are preserved correctly."""
        for year in ["2020", "2021", "2022", "2023", "2024", "2025"]:
            path = generate_split_output_path(
                f"/output/nrw_32_350000_5600000_{year}.tif",
                351,
                5601,
                easting=350500,
                northing=5600500,
            )
            assert path == Path(f"/output/nrw_32_350500_5600500_{year}.tif")


class TestAtomicRasterioWrite:
    """Tests for atomic file writing."""

    def test_writes_to_temp_then_renames(self, tmp_path):
        """Test that write goes to .tmp file then is renamed."""
        output_path = tmp_path / "output.tif"
        temp_path = output_path.parent / (output_path.name + ".tmp")

        with patch("georaffer.converters.utils.rasterio.open") as mock_open:
            mock_dst = MagicMock()
            mock_open.return_value.__enter__ = Mock(return_value=mock_dst)
            mock_open.return_value.__exit__ = Mock(return_value=False)

            # Simulate file creation
            def create_temp(*args, **kwargs):
                temp_path.touch()
                return mock_open.return_value

            mock_open.side_effect = create_temp

            with atomic_rasterio_write(output_path, "w", driver="GTiff") as dst:
                pass

        # After context exits, temp should be renamed to final
        assert output_path.exists()
        assert not temp_path.exists()

    def test_cleans_up_temp_on_error(self, tmp_path):
        """Test temp file is removed if write fails."""
        output_path = tmp_path / "output.tif"
        temp_path = output_path.parent / (output_path.name + ".tmp")

        with patch("georaffer.converters.utils.rasterio.open") as mock_open:
            # Create temp file then raise error
            def fail_write(*args, **kwargs):
                temp_path.touch()
                raise OSError("Write failed")

            mock_open.side_effect = fail_write

            with pytest.raises(IOError):
                with atomic_rasterio_write(output_path, "w", driver="GTiff"):
                    pass

        # Temp file should be cleaned up
        assert not temp_path.exists()
        assert not output_path.exists()

    def test_passes_args_to_rasterio(self, tmp_path):
        """Test that args/kwargs are passed through to rasterio.open."""
        output_path = tmp_path / "output.tif"
        temp_path = output_path.parent / (output_path.name + ".tmp")

        with patch("georaffer.converters.utils.rasterio.open") as mock_open:
            mock_dst = MagicMock()
            mock_open.return_value.__enter__ = Mock(return_value=mock_dst)
            mock_open.return_value.__exit__ = Mock(return_value=False)

            # Create temp file so rename succeeds
            def create_temp(*args, **kwargs):
                temp_path.touch()
                return mock_open.return_value

            mock_open.side_effect = create_temp

            with atomic_rasterio_write(output_path, "w", driver="GTiff", count=3) as dst:
                pass

        mock_open.assert_called_once()
        call_args = mock_open.call_args
        assert call_args[0][0] == temp_path
        assert call_args[0][1] == "w"
        assert call_args[1]["driver"] == "GTiff"
        assert call_args[1]["count"] == 3


class TestResampleRaster:
    def test_resample_multiband_keeps_data(self):
        data = np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4)
        transform = from_origin(100.0, 200.0, 1.0, 1.0)

        out_data, out_transform = resample_raster(
            data,
            transform,
            "EPSG:25833",
            target_size=2,
            num_threads=1,
            dtype=np.uint8,
            resampling=Resampling.nearest,
        )

        assert out_data.shape == (3, 2, 2)
        assert out_transform.a == 2.0
        assert out_transform.e == -2.0
        assert out_data.max() > 0


