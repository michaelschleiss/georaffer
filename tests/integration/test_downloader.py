"""Integration test for downloaders - downloads real tiles from NRW and RLP feeds.

Downloaded tiles are cached in ~/.cache/georaffer/test_tiles/ for reuse.

Environment variables:
    GEORAFFER_TEST_CACHE: Override cache directory (default: ~/.cache/georaffer/test_tiles)
    GEORAFFER_FORCE_DOWNLOAD: Set to "1" to force re-download even if cached
"""

import os
from pathlib import Path

import pytest

from georaffer.downloaders import NRWDownloader, RLPDownloader

CACHE_DIR = Path(
    os.environ.get("GEORAFFER_TEST_CACHE", Path.home() / ".cache" / "georaffer" / "test_tiles")
)

FORCE_DOWNLOAD = os.environ.get("GEORAFFER_FORCE_DOWNLOAD", "0") == "1"

# Known tiles that should exist in feeds
NRW_TEST_TILE = (350, 5600)
RLP_TEST_TILE = (362, 5604)  # 2km grid, km-based coordinates


# =============================================================================
# NRW Tests
# =============================================================================


@pytest.fixture(scope="module")
def nrw_downloader():
    """Create NRW downloader with cache directory."""
    cache = CACHE_DIR / "nrw"
    cache.mkdir(parents=True, exist_ok=True)
    return NRWDownloader(output_dir=str(cache))


@pytest.fixture(scope="module")
def nrw_catalogs(nrw_downloader):
    """Fetch NRW catalogs once for all tests."""
    return nrw_downloader.get_available_tiles()


@pytest.mark.network
@pytest.mark.integration
def test_nrw_fetches_feed(nrw_catalogs):
    """Test that NRW downloader can fetch and parse feeds."""
    jp2_catalog, laz_catalog = nrw_catalogs

    assert len(jp2_catalog) > 0, "NRW JP2 catalog should not be empty"
    assert len(laz_catalog) > 0, "NRW LAZ catalog should not be empty"
    assert NRW_TEST_TILE in jp2_catalog, f"Test tile {NRW_TEST_TILE} not in NRW JP2 catalog"
    assert NRW_TEST_TILE in laz_catalog, f"Test tile {NRW_TEST_TILE} not in NRW LAZ catalog"


@pytest.mark.network
@pytest.mark.integration
def test_nrw_downloads_jp2(nrw_downloader, nrw_catalogs):
    """Test downloading a NRW JP2 tile."""
    jp2_catalog, _ = nrw_catalogs

    url = jp2_catalog[NRW_TEST_TILE]
    filename = Path(url).name
    output_path = nrw_downloader.raw_dir / "image" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not FORCE_DOWNLOAD and output_path.exists() and output_path.stat().st_size > 1000:
        pytest.skip("NRW JP2 cached (set GEORAFFER_FORCE_DOWNLOAD=1 to re-download)")

    if FORCE_DOWNLOAD and output_path.exists():
        output_path.unlink()

    success = nrw_downloader.download_file(url, str(output_path))

    assert success, f"Failed to download {url}"
    assert output_path.exists(), f"Output file not created: {output_path}"
    assert output_path.stat().st_size > 1000, "Downloaded file too small"


@pytest.mark.network
@pytest.mark.integration
def test_nrw_downloads_laz(nrw_downloader, nrw_catalogs):
    """Test downloading a NRW LAZ tile."""
    _, laz_catalog = nrw_catalogs

    url = laz_catalog[NRW_TEST_TILE]
    filename = Path(url).name
    output_path = nrw_downloader.raw_dir / "dsm" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not FORCE_DOWNLOAD and output_path.exists() and output_path.stat().st_size > 1000:
        pytest.skip("NRW LAZ cached (set GEORAFFER_FORCE_DOWNLOAD=1 to re-download)")

    if FORCE_DOWNLOAD and output_path.exists():
        output_path.unlink()

    success = nrw_downloader.download_file(url, str(output_path))

    assert success, f"Failed to download {url}"
    assert output_path.exists(), f"Output file not created: {output_path}"
    assert output_path.stat().st_size > 1000, "Downloaded file too small"


@pytest.mark.network
@pytest.mark.integration
def test_nrw_cached_files_readable(nrw_downloader, nrw_catalogs):
    """Test that NRW cached files can be opened by rasterio/laspy."""
    import laspy
    import rasterio

    jp2_catalog, laz_catalog = nrw_catalogs

    jp2_path = nrw_downloader.raw_dir / "image" / Path(jp2_catalog[NRW_TEST_TILE]).name
    laz_path = nrw_downloader.raw_dir / "dsm" / Path(laz_catalog[NRW_TEST_TILE]).name

    if not jp2_path.exists() or not laz_path.exists():
        pytest.skip("Need cached files - run download tests first")

    with rasterio.open(jp2_path) as src:
        assert src.width > 0
        assert src.height > 0
        assert src.count >= 3  # RGB or RGBI

    with laspy.open(laz_path) as las:
        assert las.header.point_count > 0


# =============================================================================
# RLP Tests
# =============================================================================


@pytest.fixture(scope="module")
def rlp_downloader():
    """Create RLP downloader with cache directory."""
    cache = CACHE_DIR / "rlp"
    cache.mkdir(parents=True, exist_ok=True)
    return RLPDownloader(output_dir=str(cache))


@pytest.fixture(scope="module")
def rlp_catalogs(rlp_downloader):
    """Fetch RLP catalogs once for all tests."""
    return rlp_downloader.get_available_tiles()


@pytest.mark.network
@pytest.mark.integration
def test_rlp_fetches_feed(rlp_catalogs):
    """Test that RLP downloader can fetch and parse feeds."""
    jp2_catalog, laz_catalog = rlp_catalogs

    assert len(jp2_catalog) > 0, "RLP JP2 catalog should not be empty"
    assert len(laz_catalog) > 0, "RLP LAZ catalog should not be empty"
    assert RLP_TEST_TILE in jp2_catalog, f"Test tile {RLP_TEST_TILE} not in RLP JP2 catalog"
    assert RLP_TEST_TILE in laz_catalog, f"Test tile {RLP_TEST_TILE} not in RLP LAZ catalog"


@pytest.mark.network
@pytest.mark.integration
def test_rlp_downloads_jp2(rlp_downloader, rlp_catalogs):
    """Test downloading a RLP JP2 tile."""
    jp2_catalog, _ = rlp_catalogs

    url = jp2_catalog[RLP_TEST_TILE]
    filename = Path(url).name
    output_path = rlp_downloader.raw_dir / "image" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not FORCE_DOWNLOAD and output_path.exists() and output_path.stat().st_size > 1000:
        pytest.skip("RLP JP2 cached (set GEORAFFER_FORCE_DOWNLOAD=1 to re-download)")

    if FORCE_DOWNLOAD and output_path.exists():
        output_path.unlink()

    success = rlp_downloader.download_file(url, str(output_path))

    assert success, f"Failed to download {url}"
    assert output_path.exists(), f"Output file not created: {output_path}"
    assert output_path.stat().st_size > 1000, "Downloaded file too small"


@pytest.mark.network
@pytest.mark.integration
def test_rlp_downloads_laz(rlp_downloader, rlp_catalogs):
    """Test downloading a RLP LAZ tile."""
    _, laz_catalog = rlp_catalogs

    url = laz_catalog[RLP_TEST_TILE]
    filename = Path(url).name
    output_path = rlp_downloader.raw_dir / "dsm" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not FORCE_DOWNLOAD and output_path.exists() and output_path.stat().st_size > 1000:
        pytest.skip("RLP LAZ cached (set GEORAFFER_FORCE_DOWNLOAD=1 to re-download)")

    if FORCE_DOWNLOAD and output_path.exists():
        output_path.unlink()

    success = rlp_downloader.download_file(url, str(output_path))

    assert success, f"Failed to download {url}"
    assert output_path.exists(), f"Output file not created: {output_path}"
    assert output_path.stat().st_size > 1000, "Downloaded file too small"


@pytest.mark.network
@pytest.mark.integration
def test_rlp_cached_files_readable(rlp_downloader, rlp_catalogs):
    """Test that RLP cached files can be opened by rasterio/laspy."""
    import laspy
    import rasterio

    jp2_catalog, laz_catalog = rlp_catalogs

    jp2_path = rlp_downloader.raw_dir / "image" / Path(jp2_catalog[RLP_TEST_TILE]).name
    laz_path = rlp_downloader.raw_dir / "dsm" / Path(laz_catalog[RLP_TEST_TILE]).name

    if not jp2_path.exists() or not laz_path.exists():
        pytest.skip("Need cached files - run download tests first")

    with rasterio.open(jp2_path) as src:
        assert src.width > 0
        assert src.height > 0
        assert src.count >= 3  # RGB or RGBI

    with laspy.open(laz_path) as las:
        assert las.header.point_count > 0


# =============================================================================
# RLP WMS Historical Imagery Tests
# =============================================================================

WMS_TEST_TILE = (380, 5540)  # Known tile with coverage across multiple years


@pytest.fixture(scope="module")
def wms_source():
    """Create WMS imagery source for RLP historical imagery."""
    from georaffer.downloaders.wms import WMSImagerySource

    return WMSImagerySource(
        base_url="https://geo4.service24.rlp.de/wms/rp_hkdop20.fcgi",
        rgb_layer_pattern="rp_dop20_rgb_{year}",
        info_layer_pattern="rp_dop20_info_{year}",
    )


@pytest.mark.network
@pytest.mark.integration
def test_wms_check_coverage_returns_valid_result(wms_source):
    """Canary test: WMS coverage check returns valid result for known tile.

    This test validates that:
    1. WMS service is reachable
    2. Response format matches what our code expects
    3. Parsing extracts acquisition_date and tile_name correctly

    If this test fails, the WMS response format may have changed.
    """
    import re

    # Test multiple years to handle format variations
    test_years = [2020, 2022, 2024]
    results = {}

    for year in test_years:
        result = wms_source.check_coverage(year, *WMS_TEST_TILE)
        if result:
            results[year] = result

    # At least one year should have coverage
    assert len(results) > 0, (
        f"No coverage found for tile {WMS_TEST_TILE} in any of years {test_years}. "
        "WMS service may be down or response format changed."
    )

    # Validate result structure for all successful responses
    for year, result in results.items():
        assert isinstance(result, dict), f"Year {year}: Expected dict, got {type(result)}"
        assert "acquisition_date" in result, f"Year {year}: Missing acquisition_date"
        assert "tile_name" in result, f"Year {year}: Missing tile_name"

        # Validate date format (YYYY-MM-DD)
        date = result["acquisition_date"]
        assert re.match(r"\d{4}-\d{2}-\d{2}", date), (
            f"Year {year}: acquisition_date '{date}' doesn't match YYYY-MM-DD format"
        )

        # Validate tile_name contains expected coordinates
        tile_name = result["tile_name"]
        assert str(WMS_TEST_TILE[0]) in tile_name, (
            f"Year {year}: tile_name '{tile_name}' doesn't contain x coord {WMS_TEST_TILE[0]}"
        )
        assert str(WMS_TEST_TILE[1]) in tile_name, (
            f"Year {year}: tile_name '{tile_name}' doesn't contain y coord {WMS_TEST_TILE[1]}"
        )


@pytest.mark.network
@pytest.mark.integration
def test_wms_check_coverage_returns_none_for_invalid_tile(wms_source):
    """Test that coverage check returns None for tiles outside service area."""
    result = wms_source.check_coverage(2020, 999, 9999)
    assert result is None, "Expected None for nonexistent tile (999, 9999)"


@pytest.mark.network
@pytest.mark.integration
def test_wms_downloads_tile(wms_source, tmp_path):
    """Test that WMS can download a valid GeoTIFF tile."""
    import rasterio

    # Get URL for a known tile
    url = wms_source.get_tile_url(2020, *WMS_TEST_TILE)
    output_path = tmp_path / "wms_test_tile.tif"

    # Download the tile
    success = wms_source.download_tile(url, str(output_path))

    assert success, f"Failed to download WMS tile from {url}"
    assert output_path.exists(), "Output file not created"
    assert output_path.stat().st_size > 10000, "Downloaded file too small for a 2km tile"

    # Verify it's a valid GeoTIFF with expected properties
    with rasterio.open(output_path) as src:
        assert src.width == 10000, f"Expected 10000 pixels width, got {src.width}"
        assert src.height == 10000, f"Expected 10000 pixels height, got {src.height}"
        assert src.count >= 3, f"Expected at least 3 bands (RGB), got {src.count}"
        assert src.crs is not None, "Missing CRS"
        assert "25832" in str(src.crs), f"Expected EPSG:25832, got {src.crs}"
