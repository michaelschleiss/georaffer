"""Integration-style checks for full convert pipeline on tiny fixtures."""

from pathlib import Path

import laspy
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from georaffer.conversion import convert_tiles


@pytest.fixture(autouse=True)
def disable_process_pool(monkeypatch):
    """Keep tests fast/deterministic: disable process pool."""
    monkeypatch.setenv("GEORAFFER_DISABLE_PROCESS_POOL", "1")
    yield


def _write_fake_jp2(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((3, 10, 10), dtype=np.uint8)
    transform = Affine(0.2, 0, 362000, 0, -0.2, 5606000)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",  # use GeoTIFF container; extension doesn't matter for rasterio
        height=data.shape[1],
        width=data.shape[2],
        count=3,
        dtype=data.dtype,
        crs="EPSG:25832",
        transform=transform,
    ) as dst:
        dst.write(data)


def _write_fake_laz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = (0.0, 0.0, 0.0)
    header.scales = (0.01, 0.01, 0.01)

    las = laspy.LasData(header)
    # 2x2 grid at 0.5m spacing -> width=3, height=3 after rasterization
    las.x = np.array([0.0, 0.5, 0.0, 0.5])
    las.y = np.array([0.5, 0.5, 0.0, 0.0])
    las.z = np.array([100.0, 100.0, 100.0, 100.0])

    las.write(path)  # writes uncompressed LAS even with .laz extension


@pytest.mark.integration
def test_convert_tiles_end_to_end_jp2(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    jp2_path = raw_dir / "image" / "dop20rgb_32_362_5604_2_rp_2023.jp2"

    _write_fake_jp2(jp2_path)

    stats = convert_tiles(str(raw_dir), str(processed_dir), resolutions=[10], max_workers=1)

    assert stats.jp2_converted == 4
    out_files = list((processed_dir / "image").rglob("*.tif"))
    assert len(out_files) == 4
    with rasterio.open(out_files[0]) as src:
        assert src.count == 3
        assert src.width == 10 and src.height == 10


@pytest.mark.integration
def test_convert_tiles_end_to_end_laz(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    laz_path = raw_dir / "dsm" / "bdom50_32350_5600_1_nw_2025.laz"

    _write_fake_laz(laz_path)

    stats = convert_tiles(str(raw_dir), str(processed_dir), resolutions=[10], max_workers=1)

    assert stats.laz_converted == 1
    out_files = list((processed_dir / "dsm").rglob("*.tif"))
    assert len(out_files) == 1
    with rasterio.open(out_files[0]) as src:
        assert src.count == 1
        assert src.width == 10 and src.height == 10


