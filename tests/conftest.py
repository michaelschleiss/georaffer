"""Shared fixtures for georaffer tests."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

# ============ Path Fixtures ============


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_tiles_dir(fixtures_dir):
    """Path to sample tiles directory."""
    return fixtures_dir / "sample_tiles"


# ============ Temp Directory Fixtures ============


@pytest.fixture
def output_dir(tmp_path):
    """Temporary output directory for converter tests."""
    out = tmp_path / "output"
    out.mkdir()
    return out


# ============ Mock Fixtures ============


@pytest.fixture
def mock_rasterio_open():
    """Mock rasterio.open for converter tests."""
    with patch("rasterio.open") as mock:
        mock_dataset = Mock()
        mock_dataset.read.return_value = np.zeros((3, 100, 100), dtype=np.uint8)
        mock_dataset.crs = "EPSG:25832"
        mock_dataset.transform = Mock(a=0.2, b=0, c=350000, d=0, e=-0.2, f=5601000)
        mock_dataset.__enter__ = Mock(return_value=mock_dataset)
        mock_dataset.__exit__ = Mock(return_value=False)
        mock.return_value = mock_dataset
        yield mock


@pytest.fixture
def mock_laspy_read():
    """Mock laspy.read for LAZ converter tests."""
    with patch("laspy.read") as mock:
        mock_las = Mock()
        mock_las.x = np.array([350000.5, 350001.5])
        mock_las.y = np.array([5600000.5, 5600001.5])
        mock_las.z = np.array([100.0, 101.0])
        mock_las.header = Mock(point_count=2)
        mock.return_value = mock_las
        yield mock


@pytest.fixture
def mock_requests_session():
    """Mock requests.Session for downloader tests."""
    with patch("requests.Session") as mock:
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake file content"
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock.return_value = mock_session
        yield mock_session


def pytest_addoption(parser):
    """Add an opt-in flag for canary tests."""
    parser.addoption(
        "--run-canary",
        action="store_true",
        default=False,
        help="run canary tests that hit real services",
    )


def pytest_collection_modifyitems(config, items):
    """Skip canary tests unless explicitly opted in."""
    if config.getoption("--run-canary"):
        return

    skip_canary = pytest.mark.skip(reason="need --run-canary to run")
    for item in items:
        if "canary" in item.keywords:
            item.add_marker(skip_canary)


# ============ Parametrized Data ============

NRW_JP2_CASES = [
    ("dop10rgbi_32_350_5600_1_nw_2021.jp2", "NRW", 350, 5600, 2021),
    ("dop10rgbi_32_400_5700_1_nw_2020.jp2", "NRW", 400, 5700, 2020),
]

NRW_LAZ_CASES = [
    ("bdom50_32350_5600_1_nw_2025.laz", "NRW", 350, 5600, 2025),
    ("bdom50_32400_5700_1_nw_2024.laz", "NRW", 400, 5700, 2024),
]

RLP_JP2_CASES = [
    ("dop20rgb_32_362_5604_2_rp_2023.jp2", "RLP", 362, 5604, 2023),
    ("dop20rgb_32_370_5590_2_rp_2022.jp2", "RLP", 370, 5590, 2022),
]

RLP_LAZ_CASES = [
    ("bdom20rgbi_32_364_5582_2_rp.laz", "RLP", 364, 5582, None),
    ("bdom20rgbi_32_370_5590_2_rp.laz", "RLP", 370, 5590, None),
]

INVALID_FILENAMES = [
    "random_file.jp2",
    "not_a_tile.laz",
    "dop10_wrong_format.jp2",
    "",
]
