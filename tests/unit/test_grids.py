"""Tests for grids module - split factor calculations."""

import pytest

from georaffer.config import METERS_PER_KM, NRW_GRID_SIZE, RLP_GRID_SIZE
from georaffer.grids import compute_split_factor


class TestComputeSplitFactor:
    """Tests for compute_split_factor with various grid sizes."""

    # NRW has 1km native tiles, RLP has 2km native tiles
    NRW_TILE_KM = NRW_GRID_SIZE / METERS_PER_KM  # 1.0
    RLP_TILE_KM = RLP_GRID_SIZE / METERS_PER_KM  # 2.0

    def test_nrw_1km_target_no_split(self):
        """NRW 1km tiles with 1km target → no split (1×1 = 1)."""
        factor = compute_split_factor(self.NRW_TILE_KM, 1.0)
        assert factor == 1

    def test_nrw_0_5km_target_2x2_split(self):
        """NRW 1km tiles with 0.5km target → 2×2 split = 4."""
        factor = compute_split_factor(self.NRW_TILE_KM, 0.5)
        assert factor == 4

    def test_nrw_0_25km_target_4x4_split(self):
        """NRW 1km tiles with 0.25km target → 4×4 split = 16."""
        factor = compute_split_factor(self.NRW_TILE_KM, 0.25)
        assert factor == 16

    def test_rlp_2km_target_no_split(self):
        """RLP 2km tiles with 2km target → no split (1×1 = 1)."""
        factor = compute_split_factor(self.RLP_TILE_KM, 2.0)
        assert factor == 1

    def test_rlp_1km_target_2x2_split(self):
        """RLP 2km tiles with 1km target → 2×2 split = 4."""
        factor = compute_split_factor(self.RLP_TILE_KM, 1.0)
        assert factor == 4

    def test_rlp_0_5km_target_4x4_split(self):
        """RLP 2km tiles with 0.5km target → 4×4 split = 16."""
        factor = compute_split_factor(self.RLP_TILE_KM, 0.5)
        assert factor == 16

    def test_rlp_0_25km_target_8x8_split(self):
        """RLP 2km tiles with 0.25km target → 8×8 split = 64."""
        factor = compute_split_factor(self.RLP_TILE_KM, 0.25)
        assert factor == 64

    def test_invalid_non_integer_ratio_raises(self):
        """Non-integer ratio >= 2 (e.g., 1km tile with 0.3km grid = 3.33x) raises."""
        # 1.0 / 0.3 = 3.33 which is >= 2 but not an integer
        with pytest.raises(RuntimeError, match="Splitting isn't possible"):
            compute_split_factor(1.0, 0.3)

    def test_invalid_non_integer_ratio_rlp_raises(self):
        """Non-integer ratio for RLP (e.g., 2km tile with 0.75km grid) raises."""
        with pytest.raises(RuntimeError, match="Splitting isn't possible"):
            compute_split_factor(2.0, 0.75)

    def test_zero_grid_size_raises(self):
        """Zero grid size raises error."""
        with pytest.raises(RuntimeError, match="Grid size must be positive"):
            compute_split_factor(1.0, 0.0)

    def test_negative_grid_size_raises(self):
        """Negative grid size raises error."""
        with pytest.raises(RuntimeError, match="Grid size must be positive"):
            compute_split_factor(1.0, -1.0)

    def test_larger_grid_than_tile_no_split(self):
        """Grid larger than tile → no split."""
        # 1km tile with 2km grid → ratio < 2 → return 1
        factor = compute_split_factor(1.0, 2.0)
        assert factor == 1


class TestConversionPlanCalculations:
    """Tests verifying the conversion plan output calculations."""

    NRW_TILE_KM = 1.0
    RLP_TILE_KM = 2.0

    def test_output_count_nrw_1km_grid(self):
        """NRW with 1km grid: 154 sources → 154 outputs."""
        nrw_sources = 154
        split = compute_split_factor(self.NRW_TILE_KM, 1.0)
        assert split == 1
        assert nrw_sources * split == 154

    def test_output_count_rlp_1km_grid(self):
        """RLP with 1km grid: 95 sources → 380 outputs."""
        rlp_sources = 95
        split = compute_split_factor(self.RLP_TILE_KM, 1.0)
        assert split == 4
        assert rlp_sources * split == 380

    def test_output_count_nrw_0_5km_grid(self):
        """NRW with 0.5km grid: 154 sources → 616 outputs."""
        nrw_sources = 154
        split = compute_split_factor(self.NRW_TILE_KM, 0.5)
        assert split == 4
        assert nrw_sources * split == 616

    def test_output_count_rlp_0_5km_grid(self):
        """RLP with 0.5km grid: 104 sources → 1664 outputs."""
        rlp_sources = 104
        split = compute_split_factor(self.RLP_TILE_KM, 0.5)
        assert split == 16
        assert rlp_sources * split == 1664

    def test_split_side_calculation(self):
        """Verify split_side = sqrt(split_factor) for display."""
        for tile_km, grid_km, expected_side in [
            (1.0, 1.0, 1),  # 1×1
            (1.0, 0.5, 2),  # 2×2
            (2.0, 1.0, 2),  # 2×2
            (2.0, 0.5, 4),  # 4×4
            (2.0, 0.25, 8),  # 8×8
        ]:
            split = compute_split_factor(tile_km, grid_km)
            split_side = int(split**0.5)
            assert split_side == expected_side, (
                f"tile={tile_km}km, grid={grid_km}km: expected {expected_side}×{expected_side}, got {split_side}×{split_side}"
            )
            # Verify it squares back
            assert split_side * split_side == split
