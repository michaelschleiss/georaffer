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


@pytest.mark.integration
def test_nrw_fetches_feed(nrw_catalogs):
    """Test that NRW downloader can fetch and parse feeds."""
    jp2_catalog, laz_catalog = nrw_catalogs

    assert len(jp2_catalog) > 0, "NRW JP2 catalog should not be empty"
    assert len(laz_catalog) > 0, "NRW LAZ catalog should not be empty"
    assert NRW_TEST_TILE in jp2_catalog, f"Test tile {NRW_TEST_TILE} not in NRW JP2 catalog"
    assert NRW_TEST_TILE in laz_catalog, f"Test tile {NRW_TEST_TILE} not in NRW LAZ catalog"


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


@pytest.mark.integration
def test_rlp_fetches_feed(rlp_catalogs):
    """Test that RLP downloader can fetch and parse feeds."""
    jp2_catalog, laz_catalog = rlp_catalogs

    assert len(jp2_catalog) > 0, "RLP JP2 catalog should not be empty"
    assert len(laz_catalog) > 0, "RLP LAZ catalog should not be empty"
    assert RLP_TEST_TILE in jp2_catalog, f"Test tile {RLP_TEST_TILE} not in RLP JP2 catalog"
    assert RLP_TEST_TILE in laz_catalog, f"Test tile {RLP_TEST_TILE} not in RLP LAZ catalog"


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
