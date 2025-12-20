"""Tests for GeoTIFF alignment helpers."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from georaffer.align import align_to_reference


def _write_raster(
    path: Path,
    data: np.ndarray,
    *,
    crs: str,
    transform,
    nodata: float | None = None,
) -> None:
    count = data.shape[0] if data.ndim == 3 else 1
    height = data.shape[-2]
    width = data.shape[-1]

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": data.dtype,
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            dst.write(data)


def test_align_to_reference_uses_highest_resolution(tmp_path):
    output_dir = tmp_path / "output"
    processed_dir = output_dir / "processed"
    image_dir_low = processed_dir / "image" / "1000"
    image_dir_high = processed_dir / "image" / "2000"
    dsm_dir = processed_dir / "dsm" / "1000"
    image_dir_low.mkdir(parents=True, exist_ok=True)
    image_dir_high.mkdir(parents=True, exist_ok=True)
    dsm_dir.mkdir(parents=True, exist_ok=True)

    ref_path = tmp_path / "ref.tif"
    ref_transform = from_origin(100.0, 200.0, 1.0, 1.0)
    ref_data = np.zeros((1, 10, 10), dtype=np.uint8)
    _write_raster(ref_path, ref_data, crs="EPSG:25832", transform=ref_transform)

    image_low = np.full((3, 10, 10), 10, dtype=np.uint8)
    image_high = np.full((3, 20, 20), 20, dtype=np.uint8)
    low_transform = from_origin(100.0, 200.0, 1.0, 1.0)
    high_transform = from_origin(100.0, 200.0, 0.5, 0.5)
    _write_raster(
        image_dir_low / "nrw_32_100000_200000_2020.tif",
        image_low,
        crs="EPSG:25832",
        transform=low_transform,
    )
    _write_raster(
        image_dir_high / "nrw_32_100000_200000_2021.tif",
        image_high,
        crs="EPSG:25832",
        transform=high_transform,
    )

    dsm_data = np.full((10, 10), 30.0, dtype=np.float32)
    _write_raster(
        dsm_dir / "nrw_32_100000_200000_2020.tif",
        dsm_data,
        crs="EPSG:25832",
        transform=low_transform,
        nodata=-9999.0,
    )

    outputs = align_to_reference(str(ref_path), str(output_dir), align_images=True, align_dsm=True)

    image_out = outputs["image"]
    dsm_out = outputs["dsm"]

    with rasterio.open(image_out) as src:
        assert src.width == 10
        assert src.height == 10
        assert src.transform == ref_transform
        assert src.crs.to_string() == "EPSG:25832"
        data = src.read()
        assert np.all(data == 20)

    with rasterio.open(dsm_out) as src:
        assert src.width == 10
        assert src.height == 10
        assert src.transform == ref_transform
        assert src.crs.to_string() == "EPSG:25832"
        data = src.read(1)
        assert np.all(data == 30.0)
