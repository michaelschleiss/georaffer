"""Tests for georaffer.provenance module."""

import pytest

from georaffer.config import Region
from georaffer.provenance import (
    build_metadata_rows,
    compute_split_coordinates,
    get_tile_center_utm,
)


class TestComputeSplitCoordinates:
    """Tests for compute_split_coordinates function."""

    def test_no_split_returns_single_coord(self):
        """When tile equals grid size, return single coordinate."""
        coords = compute_split_coordinates(350, 5600, tile_km=1.0, grid_size_km=1.0)
        assert coords == [(350, 5600)]

    def test_2x2_split_properties(self):
        """2×2 split produces 4 tiles covering the source tile."""
        coords = compute_split_coordinates(362, 5604, tile_km=2.0, grid_size_km=1.0)

        # Correct count
        assert len(coords) == 4

        # No duplicates
        assert len(set(coords)) == 4

        # Covers full tile bounds
        xs = [x for x, y in coords]
        ys = [y for x, y in coords]
        assert min(xs) == 362  # base_x
        assert max(xs) == 363  # base_x + ratio - 1
        assert min(ys) == 5604  # base_y
        assert max(ys) == 5605  # base_y + ratio - 1

        # All four corners present
        assert (362, 5604) in coords  # SW
        assert (363, 5604) in coords  # SE
        assert (362, 5605) in coords  # NW
        assert (363, 5605) in coords  # NE

    def test_3x3_split_properties(self):
        """3×3 split produces 9 tiles - verifies formula generalizes."""
        coords = compute_split_coordinates(300, 5600, tile_km=3.0, grid_size_km=1.0)

        # Correct count
        assert len(coords) == 9

        # No duplicates
        assert len(set(coords)) == 9

        # Covers full tile bounds
        xs = [x for x, y in coords]
        ys = [y for x, y in coords]
        assert min(xs) == 300  # base_x
        assert max(xs) == 302  # base_x + ratio - 1
        assert min(ys) == 5600  # base_y
        assert max(ys) == 5602  # base_y + ratio - 1

        # All four corners present
        assert (300, 5600) in coords  # SW
        assert (302, 5600) in coords  # SE
        assert (300, 5602) in coords  # NW
        assert (302, 5602) in coords  # NE

    def test_4x4_split_properties(self):
        """4×4 split (e.g., 2km tile to 0.5km grid) produces 16 tiles."""
        coords = compute_split_coordinates(362, 5604, tile_km=2.0, grid_size_km=0.5)

        # Correct count
        assert len(coords) == 16

        # No duplicates
        assert len(set(coords)) == 16

        # Bounds span 4 units in each direction
        xs = [x for x, y in coords]
        ys = [y for x, y in coords]
        assert max(xs) - min(xs) == 3  # 4 tiles = indices 0,1,2,3
        assert max(ys) - min(ys) == 3

    def test_none_base_x_raises(self):
        """None base_x raises ValueError."""
        with pytest.raises(ValueError, match="base_x and base_y are required"):
            compute_split_coordinates(None, 5600, tile_km=1.0, grid_size_km=1.0)

    def test_none_base_y_raises(self):
        """None base_y raises ValueError."""
        with pytest.raises(ValueError, match="base_x and base_y are required"):
            compute_split_coordinates(350, None, tile_km=1.0, grid_size_km=1.0)

    def test_both_none_raises(self):
        """Both coords None raises ValueError."""
        with pytest.raises(ValueError, match="base_x and base_y are required"):
            compute_split_coordinates(None, None, tile_km=2.0, grid_size_km=1.0)

    def test_zero_coords_valid(self):
        """Zero is a valid coordinate, not treated as None."""
        coords = compute_split_coordinates(0, 0, tile_km=2.0, grid_size_km=1.0)
        assert len(coords) == 4
        assert (0, 0) in coords
        assert (1, 1) in coords


class TestGetTileCenterUtm:
    """Tests for get_tile_center_utm function."""

    def test_nrw_1km_tile_center(self):
        """NRW 1km tile center is 500m from SW corner."""
        center_x, center_y = get_tile_center_utm(350, 5600, tile_km=1.0)
        assert center_x == 350_500.0
        assert center_y == 5_600_500.0

    def test_rlp_2km_tile_center(self):
        """RLP 2km tile center is 1km from SW corner."""
        center_x, center_y = get_tile_center_utm(362, 5604, tile_km=2.0)
        assert center_x == 363_000.0
        assert center_y == 5_605_000.0

    def test_zero_coords(self):
        """Zero coordinates produce non-zero center."""
        center_x, center_y = get_tile_center_utm(0, 0, tile_km=1.0)
        assert center_x == 500.0
        assert center_y == 500.0


class TestBuildMetadataRows:
    """Tests for build_metadata_rows function."""

    def test_single_tile_nrw(self):
        """NRW tile without split produces single metadata row."""
        rows = build_metadata_rows(
            filename="dop10rgbi_32_350_5600_1_nw_2021.jp2",
            output_path="/out/nrw_32_350000_5600000_2021.tif",
            region=Region.NRW,
            year="2021",
            file_type="orthophoto",
            grid_size_km=1.0,
        )

        assert len(rows) == 1
        assert rows[0]["source_file"] == "dop10rgbi_32_350_5600_1_nw_2021.jp2"
        assert rows[0]["source_region"] == "NRW"
        assert rows[0]["year"] == "2021"
        assert rows[0]["grid_x"] == 350
        assert rows[0]["grid_y"] == 5600

    def test_split_tile_rlp(self):
        """RLP 2km tile split to 1km grid produces 4 metadata rows."""
        rows = build_metadata_rows(
            filename="dop20rgb_32_362_5604_2_rp_2023.jp2",
            output_path="/out/rlp_32_362000_5604000_2023.tif",
            region=Region.RLP,
            year="2023",
            file_type="orthophoto",
            grid_size_km=1.0,
        )

        assert len(rows) == 4

        # All rows have same source
        assert all(r["source_file"] == "dop20rgb_32_362_5604_2_rp_2023.jp2" for r in rows)
        assert all(r["source_region"] == "RLP" for r in rows)

        # Each row has unique coordinates
        coords = [(r["grid_x"], r["grid_y"]) for r in rows]
        assert len(set(coords)) == 4

        # Coordinates cover the 2km tile
        xs = [r["grid_x"] for r in rows]
        ys = [r["grid_y"] for r in rows]
        assert min(xs) == 362
        assert max(xs) == 363
        assert min(ys) == 5604
        assert max(ys) == 5605

    def test_unparseable_filename_raises(self):
        """Unparseable filename raises ValueError."""
        with pytest.raises(ValueError, match="Cannot parse grid coordinates"):
            build_metadata_rows(
                filename="weird_file.jp2",
                output_path="/out/output.tif",
                region=Region.NRW,
                year="2021",
                file_type="orthophoto",
                grid_size_km=1.0,
            )

    def test_optional_metadata_fields(self):
        """Optional acquisition_date and metadata_source are included."""
        rows = build_metadata_rows(
            filename="dop10rgbi_32_350_5600_1_nw_2021.jp2",
            output_path="/out/nrw_32_350000_5600000_2021.tif",
            region=Region.NRW,
            year="2021",
            file_type="orthophoto",
            grid_size_km=1.0,
            acquisition_date="2021-06-15",
            metadata_source="WMS GetFeatureInfo",
        )

        assert rows[0]["acquisition_date"] == "2021-06-15"
        assert rows[0]["metadata_source"] == "WMS GetFeatureInfo"

    def test_conversion_date_is_included(self):
        """Conversion date is auto-generated and in ISO format."""
        rows = build_metadata_rows(
            filename="dop10rgbi_32_350_5600_1_nw_2021.jp2",
            output_path="/out/nrw_32_350000_5600000_2021.tif",
            region=Region.NRW,
            year="2021",
            file_type="orthophoto",
            grid_size_km=1.0,
        )

        # conversion_date should be present and in ISO format
        conversion_date = rows[0]["conversion_date"]
        assert conversion_date is not None
        # Verify it parses as valid ISO datetime
        from datetime import datetime
        datetime.fromisoformat(conversion_date)  # Will raise if invalid


class TestProvenanceCsvMerge:
    """Tests for provenance CSV merge functionality."""

    def test_provenance_csv_merges_existing(self, tmp_path):
        """Test that new provenance rows merge with existing CSV."""
        from georaffer.metadata import create_provenance_csv

        csv_path = tmp_path / "provenance.csv"

        # Create initial CSV with one row
        existing_rows = [
            {
                "processed_file": "existing_tile.tif",
                "source_file": "source_existing.jp2",
                "source_region": "NRW",
                "grid_x": 350,
                "grid_y": 5600,
                "year": "2020",
                "file_type": "orthophoto",
            }
        ]
        create_provenance_csv(existing_rows, str(csv_path))

        # Add new row
        new_rows = [
            {
                "processed_file": "new_tile.tif",
                "source_file": "source_new.jp2",
                "source_region": "RLP",
                "grid_x": 362,
                "grid_y": 5604,
                "year": "2021",
                "file_type": "orthophoto",
            }
        ]
        create_provenance_csv(new_rows, str(csv_path))

        # Read merged result
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Should have both rows
        assert len(rows) == 2
        files = {r["processed_file"] for r in rows}
        assert files == {"existing_tile.tif", "new_tile.tif"}

    def test_provenance_csv_overwrites_same_key(self, tmp_path):
        """Test that reprocessing same file overwrites existing row."""
        from georaffer.metadata import create_provenance_csv

        csv_path = tmp_path / "provenance.csv"

        # Create initial CSV
        old_rows = [
            {
                "processed_file": "tile.tif",
                "source_file": "old_source.jp2",
                "source_region": "NRW",
                "year": "2020",
            }
        ]
        create_provenance_csv(old_rows, str(csv_path))

        # Reprocess same tile with new metadata
        new_rows = [
            {
                "processed_file": "tile.tif",  # Same key
                "source_file": "new_source.jp2",  # Different metadata
                "source_region": "NRW",
                "year": "2021",
            }
        ]
        create_provenance_csv(new_rows, str(csv_path))

        # Read result
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Should have only one row with updated metadata
        assert len(rows) == 1
        assert rows[0]["source_file"] == "new_source.jp2"
        assert rows[0]["year"] == "2021"
