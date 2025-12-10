"""Tests for pipeline module."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from georaffer.conversion import convert_tiles
from georaffer.downloading import download_files
from georaffer.grids import generate_user_grid_tiles, latlon_to_utm
from georaffer.pipeline import ProcessingStats
from georaffer.provenance import extract_year_from_filename
from georaffer.tiles import TileSet
from georaffer.workers import generate_output_name


class TestTileSet:
    """Tests for TileSet dataclass."""

    def test_default_empty(self):
        """Test TileSet initializes with empty dicts."""
        ts = TileSet()
        assert ts.jp2_count("nrw") == 0
        assert ts.laz_count("nrw") == 0
        assert ts.jp2_count("rlp") == 0
        assert ts.laz_count("rlp") == 0
        assert len(ts.missing_jp2) == 0
        assert len(ts.missing_laz) == 0

    def test_add_tiles(self):
        """Test adding tiles to TileSet."""
        ts = TileSet()
        ts.jp2["nrw"] = {(350, 5600)}
        ts.missing_laz.add((351, 5601))

        assert (350, 5600) in ts.jp2["nrw"]
        assert (351, 5601) in ts.missing_laz


class TestProcessingStats:
    """Tests for ProcessingStats dataclass."""

    def test_default_zeros(self):
        """Test ProcessingStats initializes with zeros."""
        stats = ProcessingStats()
        assert stats.downloaded == 0
        assert stats.skipped == 0
        assert stats.failed_download == 0
        assert stats.converted == 0
        assert stats.failed_convert == 0
        assert stats.jp2_converted == 0
        assert stats.jp2_failed == 0
        assert stats.laz_converted == 0
        assert stats.laz_failed == 0
        assert stats.jp2_duration == 0.0
        assert stats.laz_duration == 0.0

    def test_increment(self):
        """Test incrementing stats."""
        stats = ProcessingStats()
        stats.downloaded += 5
        stats.failed_convert += 1
        stats.jp2_converted += 2
        stats.laz_duration += 1.5

        assert stats.downloaded == 5
        assert stats.failed_convert == 1
        assert stats.jp2_converted == 2
        assert stats.laz_duration == 1.5


class TestGenerateUserGridTiles:
    """Tests for generate_user_grid_tiles function."""

    def test_margin_zero(self):
        """Test margin 0 returns only center tile."""
        tiles = generate_user_grid_tiles([(350000, 5600000)], grid_size_km=1.0, margin_km=0)
        assert tiles == {(350, 5600)}

    def test_margin_one_1km_grid(self):
        """Test margin 1 km with 1 km grid returns 9 tiles (3x3)."""
        tiles = generate_user_grid_tiles([(350000, 5600000)], grid_size_km=1.0, margin_km=1.0)

        assert len(tiles) == 9
        assert (350, 5600) in tiles  # center
        assert (349, 5599) in tiles  # SW
        assert (351, 5601) in tiles  # NE

    def test_margin_one_half_km_grid(self):
        """Test margin 1 km with 0.5 km grid samples more tiles."""
        tiles = generate_user_grid_tiles([(350000, 5600000)], grid_size_km=0.5, margin_km=1.0)

        # margin_tiles = ceil(1000m / 500m) = 2
        # grid = (-2 to 2) = 5x5 = 25 tiles
        assert len(tiles) == 25

    def test_multiple_coords_deduplicate(self):
        """Test multiple coordinates deduplicate at user grid level."""
        tiles = generate_user_grid_tiles(
            [(350000, 5600000), (350500, 5600500)],  # Both map to same 1km tile
            grid_size_km=1.0,
            margin_km=0,
        )
        assert len(tiles) == 1
        assert (350, 5600) in tiles

    def test_fractional_grid_size_0_2km(self):
        """Test 0.2km grid size works correctly."""
        tiles = generate_user_grid_tiles([(350000, 5600000)], grid_size_km=0.2, margin_km=0.2)

        # margin_tiles = ceil(200m / 200m) = 1
        # grid = (-1 to 1) = 3x3 = 9 tiles
        assert len(tiles) == 9
        # Center tile at 350000m / 200m = 1750
        assert (1750, 28000) in tiles
        assert (1749, 27999) in tiles  # SW
        assert (1751, 28001) in tiles  # NE


class TestCalculateRequiredTiles:
    """Tests for calculate_required_tiles native grid mapping behavior."""

    def test_maps_to_nrw_native_grid(self, tmp_path):
        """Test user-grid tiles map to NRW 1km native grid."""
        from georaffer.tiles import RegionCatalog, calculate_required_tiles

        # Mock downloaders
        nrw_downloader = MagicMock(raw_dir=tmp_path / "nrw")
        nrw_downloader.utm_to_grid_coords.side_effect = lambda x, y: (
            (int(x // 1000), int(y // 1000)),
            (int(x // 1000), int(y // 1000)),
        )
        rlp_downloader = MagicMock(raw_dir=tmp_path / "rlp")
        rlp_downloader.utm_to_grid_coords.side_effect = lambda x, y: (
            (int(x // 2000), int(y // 2000)),
            (int(x // 2000), int(y // 2000)),
        )

        # Build NRW catalog for 3x3 ring (9 tiles)
        nrw_jp2_catalog = {
            (350 + dx, 5600 + dy): f"http://example.com/{350 + dx}_{5600 + dy}.jp2"
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        }

        regions = [
            RegionCatalog("nrw", nrw_downloader, nrw_jp2_catalog, {}),
            RegionCatalog("rlp", rlp_downloader, {}, {}),
        ]

        # User tiles in grid coordinates (at 1km resolution) covering a 3x3 area
        user_tiles = {(350 + dx, 5600 + dy) for dx in range(-1, 2) for dy in range(-1, 2)}

        tiles, downloads = calculate_required_tiles(user_tiles, 1.0, regions)

        assert tiles.jp2_count("nrw") == 9
        assert len(downloads["nrw_jp2"]) == 9

    def test_only_downloads_available_tiles(self, tmp_path):
        """Test only available tiles are downloaded, not missing ones."""
        from georaffer.tiles import RegionCatalog, calculate_required_tiles

        nrw_downloader = MagicMock(raw_dir=tmp_path / "nrw")
        nrw_downloader.utm_to_grid_coords.side_effect = lambda x, y: (
            (int(x // 1000), int(y // 1000)),
            (int(x // 1000), int(y // 1000)),
        )
        rlp_downloader = MagicMock(raw_dir=tmp_path / "rlp")
        rlp_downloader.utm_to_grid_coords.side_effect = lambda x, y: (
            (int(x // 2000), int(y // 2000)),
            (int(x // 2000), int(y // 2000)),
        )

        # Only one tile available
        nrw_jp2_catalog = {(350, 5600): "http://example.com/tile.jp2"}

        regions = [
            RegionCatalog("nrw", nrw_downloader, nrw_jp2_catalog, {}),
            RegionCatalog("rlp", rlp_downloader, {}, {}),
        ]

        # Request 3 user tiles
        user_tiles = {(350, 5600), (351, 5601), (352, 5602)}

        tiles, downloads = calculate_required_tiles(user_tiles, 1.0, regions)

        assert tiles.jp2_count("nrw") == 1
        assert len(downloads["nrw_jp2"]) == 1
        assert len(tiles.missing_jp2) == 2

    def test_coverage_from_nrw_only(self, tmp_path):
        """Test user tile covered by NRW but not RLP is not reported as missing."""
        from georaffer.tiles import RegionCatalog, calculate_required_tiles

        nrw_downloader = MagicMock(raw_dir=tmp_path / "nrw")
        nrw_downloader.utm_to_grid_coords.side_effect = lambda x, y: (
            (int(x // 1000), int(y // 1000)),
            (int(x // 1000), int(y // 1000)),
        )

        rlp_downloader = MagicMock(raw_dir=tmp_path / "rlp")
        rlp_downloader.utm_to_grid_coords.side_effect = lambda x, y: (
            (int(x // 2000), int(y // 2000)),
            (int(x // 2000), int(y // 2000)),
        )

        # Only NRW has coverage for this location
        nrw_jp2_catalog = {(350, 5600): "http://example.com/nrw_tile.jp2"}

        regions = [
            RegionCatalog("nrw", nrw_downloader, nrw_jp2_catalog, {}),
            RegionCatalog("rlp", rlp_downloader, {}, {}),
        ]

        user_tiles = {(350, 5600)}

        tiles, downloads = calculate_required_tiles(user_tiles, 1.0, regions)

        # Should find tile in NRW
        assert tiles.jp2_count("nrw") == 1
        assert len(downloads["nrw_jp2"]) == 1

        # Should NOT report as missing since NRW covers it
        assert len(tiles.missing_jp2) == 0, (
            "User tile covered by NRW should not be reported as missing"
        )

    def test_coverage_from_rlp_only(self, tmp_path):
        """Test user tile covered by RLP but not NRW is not reported as missing."""
        from georaffer.tiles import RegionCatalog, calculate_required_tiles

        nrw_downloader = MagicMock(raw_dir=tmp_path / "nrw")
        nrw_downloader.utm_to_grid_coords.side_effect = lambda x, y: (
            (int(x // 1000), int(y // 1000)),
            (int(x // 1000), int(y // 1000)),
        )

        rlp_downloader = MagicMock(raw_dir=tmp_path / "rlp")
        rlp_downloader.utm_to_grid_coords.side_effect = lambda x, y: (
            (int(x // 2000), int(y // 2000)),
            (int(x // 2000), int(y // 2000)),
        )

        # Only RLP has coverage for this location
        rlp_jp2_catalog = {(175, 2800): "http://example.com/rlp_tile.jp2"}

        regions = [
            RegionCatalog("nrw", nrw_downloader, {}, {}),
            RegionCatalog("rlp", rlp_downloader, rlp_jp2_catalog, {}),
        ]

        user_tiles = {(350, 5600)}

        tiles, downloads = calculate_required_tiles(user_tiles, 1.0, regions)

        # Should find tile in RLP
        assert tiles.jp2_count("rlp") == 1
        assert len(downloads["rlp_jp2"]) == 1

        # Should NOT report as missing since RLP covers it
        assert len(tiles.missing_jp2) == 0, (
            "User tile covered by RLP should not be reported as missing"
        )

    def test_no_coverage_from_either_region(self, tmp_path):
        """Test user tile covered by neither region is reported as missing."""
        from georaffer.tiles import RegionCatalog, calculate_required_tiles

        nrw_downloader = MagicMock(raw_dir=tmp_path / "nrw")
        nrw_downloader.utm_to_grid_coords.side_effect = lambda x, y: (
            (int(x // 1000), int(y // 1000)),
            (int(x // 1000), int(y // 1000)),
        )

        rlp_downloader = MagicMock(raw_dir=tmp_path / "rlp")
        rlp_downloader.utm_to_grid_coords.side_effect = lambda x, y: (
            (int(x // 2000), int(y // 2000)),
            (int(x // 2000), int(y // 2000)),
        )

        # Neither region has coverage
        regions = [
            RegionCatalog("nrw", nrw_downloader, {}, {}),
            RegionCatalog("rlp", rlp_downloader, {}, {}),
        ]

        user_tiles = {(350, 5600)}

        tiles, downloads = calculate_required_tiles(user_tiles, 1.0, regions)

        # Should find nothing
        assert tiles.jp2_count("nrw") == 0
        assert tiles.jp2_count("rlp") == 0

        # Should report as missing
        assert len(tiles.missing_jp2) == 1, (
            "User tile with no coverage should be reported as missing"
        )


class TestLatLonToUtm:
    """Tests for latlon_to_utm function."""

    def test_conversion_germany(self):
        """Test conversion for coordinates in Germany."""
        # Cologne area
        easting, northing = latlon_to_utm(50.9375, 6.9603)

        # Should be in UTM zone 32N
        assert 350000 < easting < 360000
        assert 5640000 < northing < 5650000

    def test_forces_zone_32(self):
        """Test that conversion forces zone 32."""
        # Even coordinates that might be in zone 33
        with patch("georaffer.grids.utm.from_latlon") as mock_utm:
            mock_utm.return_value = (500000, 5600000, 32, "N")
            latlon_to_utm(50.0, 9.0)

            mock_utm.assert_called_once()
            call_kwargs = mock_utm.call_args[1]
            assert call_kwargs["force_zone_number"] == 32


class TestExtractYearFromFilename:
    """Tests for extract_year_from_filename function."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("dop10rgbi_32_350_5600_1_nw_2021.jp2", "2021"),
            ("bdom50_32350_5600_1_nw_2025.laz", "2025"),
            ("tile_2015.tif", "2015"),
        ],
    )
    def test_extract(self, filename, expected):
        """Test year extraction from various filenames."""
        assert extract_year_from_filename(filename) == expected

    @pytest.mark.parametrize(
        "filename",
        [
            "no_year_here.jp2",
            "",
        ],
    )
    def test_extract_optional_latest(self, filename):
        """Without require flag we still allow 'latest' fallback."""
        assert extract_year_from_filename(filename, require=False) == "latest"

    @pytest.mark.parametrize(
        "filename",
        [
            "no_year_here.jp2",
            "",
        ],
    )
    def test_extract_required_raises(self, filename):
        """Require flag should raise when year is absent."""
        with pytest.raises(RuntimeError):
            extract_year_from_filename(filename, require=True)


class TestGenerateOutputName:
    """Tests for generate_output_name function."""

    def test_nrw_image(self):
        """Test NRW image output name generation."""
        from georaffer.config import Region

        name = generate_output_name(
            "dop10rgbi_32_350_5600_1_nw_2021.jp2", Region.NRW, "2021", "image"
        )
        assert name == "nrw_32_350000_5600000_2021.tif"

    def test_rlp_pointcloud(self):
        """Test RLP pointcloud output name generation."""
        from georaffer.config import Region

        name = generate_output_name(
            "bdom20rgbi_32_362_5604_2_rp.laz", Region.RLP, None, "pointcloud"
        )
        assert name == "rlp_32_362000_5604000_latest.tif"

    def test_no_coords_match(self):
        """Test fallback when coords not found."""
        from georaffer.config import Region

        name = generate_output_name("weird_filename.jp2", Region.NRW, "2021", "image")
        assert name == "nrw_32_0_0_2021.tif"


class TestDownloadFiles:
    """Tests for download_files function."""

    def test_download_success(self, tmp_path):
        """Test successful downloads."""
        mock_downloader = Mock()
        mock_downloader.download_file.return_value = True

        downloads = [
            ("http://example.com/a.jp2", str(tmp_path / "a.jp2")),
            ("http://example.com/b.jp2", str(tmp_path / "b.jp2")),
        ]

        stats = download_files(downloads, mock_downloader)

        assert stats.downloaded == 2
        assert stats.skipped == 0
        assert stats.failed == 0

    def test_skip_existing(self, tmp_path):
        """Test skipping existing files."""
        # Create existing file
        existing = tmp_path / "existing.jp2"
        existing.touch()

        mock_downloader = Mock()

        downloads = [
            ("http://example.com/existing.jp2", str(existing)),
        ]

        stats = download_files(downloads, mock_downloader, force=False)

        assert stats.skipped == 1
        assert stats.downloaded == 0
        mock_downloader.download_file.assert_not_called()

    def test_force_redownload(self, tmp_path):
        """Test force re-download of existing files."""
        existing = tmp_path / "existing.jp2"
        existing.touch()

        mock_downloader = Mock()
        mock_downloader.download_file.return_value = True

        downloads = [
            ("http://example.com/existing.jp2", str(existing)),
        ]

        stats = download_files(downloads, mock_downloader, force=True)

        assert stats.downloaded == 1
        assert stats.skipped == 0
        mock_downloader.download_file.assert_called_once()

    def test_failed_download_raises(self, tmp_path):
        """Test that failed downloads propagate exception."""
        mock_downloader = Mock()
        mock_downloader.download_file.side_effect = RuntimeError("Download failed")

        downloads = [
            ("http://example.com/fail.jp2", str(tmp_path / "fail.jp2")),
        ]

        with pytest.raises(RuntimeError, match="Download failed"):
            download_files(downloads, mock_downloader)


class TestConvertTiles:
    """Tests for convert_tiles function."""

    def test_convert_jp2_files(self, tmp_path, monkeypatch):
        """Test JP2 conversion."""
        import georaffer.conversion as convert_mod

        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        (raw_dir / "image").mkdir(parents=True)
        (raw_dir / "dsm").mkdir(parents=True)

        # Create fake JP2 file
        (raw_dir / "image" / "dop10rgbi_32_350_5600_1_nw_2021.jp2").touch()

        def fake_jp2_worker(args):
            return True, [], args[0], 1  # success, metadata, filename, outputs

        monkeypatch.setenv("GEORAFFER_DISABLE_PROCESS_POOL", "1")
        monkeypatch.setattr(convert_mod, "convert_jp2_worker", fake_jp2_worker)

        stats = convert_tiles(str(raw_dir), str(processed_dir), [1000])

        assert not stats.interrupted
        assert stats.converted == 1
        assert stats.failed == 0

    def test_convert_laz_files(self, tmp_path, monkeypatch):
        """Test LAZ conversion."""
        import georaffer.conversion as convert_mod

        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        (raw_dir / "image").mkdir(parents=True)
        (raw_dir / "dsm").mkdir(parents=True)

        # Create fake LAZ file
        (raw_dir / "dsm" / "bdom50_32350_5600_1_nw_2025.laz").touch()

        def fake_laz_worker(args):
            return True, [], args[0], 1  # success, metadata, filename, outputs

        monkeypatch.setenv("GEORAFFER_DISABLE_PROCESS_POOL", "1")
        monkeypatch.setattr(convert_mod, "convert_laz_worker", fake_laz_worker)

        stats = convert_tiles(str(raw_dir), str(processed_dir), [1000])

        assert not stats.interrupted
        assert stats.converted == 1

    def test_laz_year_falls_back_to_header(self, tmp_path, monkeypatch):
        """When filename lacks year (RLP LAZ), use LAS header year for provenance."""
        import georaffer.workers as workers_mod

        laz_dir = tmp_path / "raw" / "dsm"
        processed_dir = tmp_path / "processed"
        laz_dir.mkdir(parents=True, exist_ok=True)

        # RLP LAZ files have no year in filename - must use LAS header
        laz_file = laz_dir / "bdom20rgbi_32_364_5582_2_rp.laz"
        laz_file.touch()

        monkeypatch.setattr(workers_mod, "get_laz_year", lambda _: "2030")
        monkeypatch.setattr(workers_mod, "convert_laz", lambda *args, **kwargs: True)

        # Use grid_size_km=2.0 to match RLP tile size (no split)
        success, metadata, fname, out_count = workers_mod.convert_laz_worker(
            (laz_file.name, str(laz_dir), str(processed_dir), [1000], 1, 2.0, False)
        )

        assert success is True
        assert out_count == 1
        assert metadata[0]["year"] == "2030"

    def test_detects_rlp_region(self, tmp_path, monkeypatch):
        """Test RLP region detection from filename."""
        import georaffer.workers as workers_mod
        from georaffer.config import Region

        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        (raw_dir / "image").mkdir(parents=True)
        (raw_dir / "pointcloud").mkdir(parents=True)

        (raw_dir / "image" / "dop20rgb_32_362_5604_2_rp_2023.jp2").touch()

        detected_region = None
        original_convert_jp2 = workers_mod.convert_jp2

        def capture_region(*args, **kwargs):
            nonlocal detected_region
            detected_region = args[2]  # region is 3rd positional arg
            return original_convert_jp2(*args, **kwargs)

        monkeypatch.setattr(workers_mod, "convert_jp2", capture_region)
        monkeypatch.setenv("GEORAFFER_DISABLE_PROCESS_POOL", "1")

        # Use the worker function directly to test region detection
        region = workers_mod.detect_region("dop20rgb_32_362_5604_2_rp_2023.jp2")
        assert region == Region.RLP

    def test_empty_directories(self, tmp_path):
        """Test handling of empty directories."""
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"

        stats = convert_tiles(str(raw_dir), str(processed_dir), [1000])

        assert stats.converted == 0
        assert stats.failed == 0

    def test_split_flag_not_set_for_multi_resolution_only(self, tmp_path, monkeypatch):
        """Multiple resolutions without spatial split should not mark split."""
        import georaffer.conversion as convert_mod

        monkeypatch.setenv("GEORAFFER_DISABLE_PROCESS_POOL", "1")

        def fake_worker(args):
            (
                filename,
                jp2_dir,
                processed_dir,
                resolutions,
                threads_per_worker,
                grid_size_km,
                profiling,
            ) = args
            return True, [], filename, len(resolutions)  # no split; outputs == resolutions

        monkeypatch.setattr(convert_mod, "convert_jp2_worker", fake_worker)

        raw_dir = tmp_path / "raw" / "image"
        raw_dir.mkdir(parents=True)
        (raw_dir / "noop.jp2").touch()

        stats = convert_tiles(
            str(tmp_path / "raw"),
            str(tmp_path / "processed"),
            resolutions=[500, 1000],
            max_workers=1,
            process_pointclouds=False,
        )

        assert stats.jp2_split_performed is False

    def test_split_flag_set_when_outputs_exceed_resolutions(self, tmp_path, monkeypatch):
        """Spatial split should mark split flag."""
        import georaffer.conversion as convert_mod

        monkeypatch.setenv("GEORAFFER_DISABLE_PROCESS_POOL", "1")

        def fake_worker(args):
            (
                filename,
                jp2_dir,
                processed_dir,
                resolutions,
                threads_per_worker,
                grid_size_km,
                profiling,
            ) = args
            return True, [], filename, len(resolutions) * 4  # simulate 2x2 split

        monkeypatch.setattr(convert_mod, "convert_jp2_worker", fake_worker)

        raw_dir = tmp_path / "raw" / "image"
        raw_dir.mkdir(parents=True)
        (raw_dir / "split.jp2").touch()

        stats = convert_tiles(
            str(tmp_path / "raw"),
            str(tmp_path / "processed"),
            resolutions=[500],
            max_workers=1,
            process_pointclouds=False,
        )

        assert stats.jp2_split_performed is True
