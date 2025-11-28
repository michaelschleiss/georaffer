"""Tests for bounding box input adapter."""

from georaffer.inputs.bbox import load_from_bbox


class TestLoadFromBbox:
    """Tests for load_from_bbox function.

    Note: load_from_bbox returns UTM tile CENTER coordinates (floats),
    not km-based grid indices. For a 1km tile at 350km, the center is 350500.0m.
    """

    def test_single_tile_1km(self):
        """Test bbox covering exactly one 1km tile."""
        coords = load_from_bbox(350000, 5600000, 350500, 5600500, tile_size_m=1000)

        assert len(coords) == 1
        # Center of 1km tile starting at 350000m
        assert coords[0] == (350500.0, 5600500.0)

    def test_multiple_tiles_1km(self):
        """Test bbox covering multiple 1km tiles."""
        # 2km x 2km bbox should cover 4 tiles (1km grid)
        coords = load_from_bbox(350000, 5600000, 351999, 5601999, tile_size_m=1000)

        assert len(coords) == 4
        # Centers of 1km tiles
        assert (350500.0, 5600500.0) in coords
        assert (351500.0, 5600500.0) in coords
        assert (350500.0, 5601500.0) in coords
        assert (351500.0, 5601500.0) in coords

    def test_single_tile_2km(self):
        """Test bbox covering exactly one 2km tile."""
        coords = load_from_bbox(362000, 5604000, 363000, 5605000, tile_size_m=2000)

        assert len(coords) == 1
        # Center of 2km tile starting at 362000m
        assert coords[0] == (363000.0, 5605000.0)

    def test_2km_snaps_to_grid(self):
        """Test that coordinates are snapped to 2km grid."""
        # Point at 363500m should snap to tile starting at 362000m
        coords = load_from_bbox(363500, 5605500, 363600, 5605600, tile_size_m=2000)

        assert len(coords) == 1
        # 363500m // 2000 * 2000 = 362000m -> center at 363000m
        assert coords[0] == (363000.0, 5605000.0)

    def test_no_duplicate_tiles(self):
        """Test that overlapping points don't create duplicates."""
        # Dense grid of points within same tile
        coords = load_from_bbox(350100, 5600100, 350200, 5600200, tile_size_m=1000)

        assert len(coords) == 1

    def test_sorted_output(self):
        """Test that output is sorted."""
        coords = load_from_bbox(350000, 5600000, 352999, 5602999, tile_size_m=1000)

        assert coords == sorted(coords)

    def test_default_tile_size(self):
        """Test default tile size is 1km."""
        coords_explicit = load_from_bbox(350000, 5600000, 350500, 5600500, tile_size_m=1000)
        coords_default = load_from_bbox(350000, 5600000, 350500, 5600500)

        assert coords_explicit == coords_default

    def test_point_bbox(self):
        """Test bbox where min equals max (single point)."""
        coords = load_from_bbox(350500, 5600500, 350500, 5600500, tile_size_m=1000)

        assert len(coords) == 1
        # Point 350500m falls in tile 350000-351000m, center at 350500m
        assert coords[0] == (350500.0, 5600500.0)

    def test_subkm_tiles(self):
        """Test that sub-km tile sizes work correctly (the bug fix)."""
        # With 500m tiles, same bbox should give more tiles
        coords = load_from_bbox(350000, 5600000, 350999, 5600999, tile_size_m=500)

        assert len(coords) == 4  # 2x2 grid of 500m tiles
        # Centers should be at 250m offsets from tile origins
        assert (350250.0, 5600250.0) in coords
        assert (350750.0, 5600250.0) in coords
        assert (350250.0, 5600750.0) in coords
        assert (350750.0, 5600750.0) in coords
