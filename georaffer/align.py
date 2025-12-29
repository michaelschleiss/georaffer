"""Align processed tiles to a reference GeoTIFF grid."""

from __future__ import annotations

import re
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import reproject, transform_bounds

from georaffer.config import DSM_NODATA


def _extract_year_from_filename(filename: str) -> str | None:
    """Extract 4-digit year from filename (e.g., 'tile_2021.tif' -> '2021')."""
    match = re.search(r"_(\d{4})\.", filename)
    return match.group(1) if match else None


def _select_highest_resolution_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"No processed outputs found at {base_dir}")

    numeric_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if numeric_dirs:
        return max(numeric_dirs, key=lambda d: int(d.name))

    native_dir = base_dir / "native"
    if native_dir.is_dir():
        return native_dir

    raise FileNotFoundError(f"No resolution directories found in {base_dir}")


def _sort_paths_latest_first(paths: list[Path]) -> list[Path]:
    def _year_key(path: Path) -> int:
        year = _extract_year_from_filename(path.name)
        return int(year) if year and year.isdigit() else 9999

    return sorted(paths, key=_year_key, reverse=True)


def _collect_geotiffs(res_dir: Path) -> list[Path]:
    tifs = sorted(res_dir.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No GeoTIFF outputs found in {res_dir}")
    return _sort_paths_latest_first(tifs)


def _ensure_single_crs(datasets: list[rasterio.DatasetReader]) -> CRS:
    crs_set = {ds.crs.to_string() if ds.crs else None for ds in datasets}
    if None in crs_set:
        raise ValueError("Processed tile missing CRS; cannot align.")
    if len(crs_set) != 1:
        raise ValueError("Processed tiles have multiple CRSs; cannot align as a single mosaic.")
    return datasets[0].crs


def _align_outputs(
    sources: list[Path],
    output_path: Path,
    *,
    ref_crs,
    ref_transform,
    ref_width: int,
    ref_height: int,
    ref_bounds: tuple[float, float, float, float],
    resampling: Resampling,
    nodata: float | None,
    area_or_point: str,
    max_bands: int | None = None,
) -> None:
    with ExitStack() as stack:
        datasets = [stack.enter_context(rasterio.open(path)) for path in sources]
        src_crs = _ensure_single_crs(datasets)

        if src_crs != ref_crs:
            src_bounds = transform_bounds(ref_crs, src_crs, *ref_bounds, densify_pts=21)
        else:
            src_bounds = ref_bounds

        merge_kwargs = {
            "bounds": src_bounds,
            "res": datasets[0].res,
            "method": "first",
        }
        if nodata is not None:
            merge_kwargs["nodata"] = nodata

        mosaic, mosaic_transform = merge(datasets, **merge_kwargs)
        if max_bands is not None and mosaic.shape[0] > max_bands:
            mosaic = mosaic[:max_bands]
        band_count = mosaic.shape[0]

        dst_nodata = nodata if nodata is not None else datasets[0].nodata
        if dst_nodata is None:
            aligned = np.zeros((band_count, ref_height, ref_width), dtype=mosaic.dtype)
        else:
            aligned = np.full((band_count, ref_height, ref_width), dst_nodata, dtype=mosaic.dtype)

        reproject_kwargs = {
            "src_transform": mosaic_transform,
            "src_crs": src_crs,
            "dst_transform": ref_transform,
            "dst_crs": ref_crs,
            "resampling": resampling,
            "init_dest_nodata": False,
        }
        if datasets[0].nodata is not None:
            reproject_kwargs["src_nodata"] = datasets[0].nodata
        if dst_nodata is not None:
            reproject_kwargs["dst_nodata"] = dst_nodata

        for band_index in range(band_count):
            reproject(
                source=mosaic[band_index],
                destination=aligned[band_index],
                **reproject_kwargs,
            )

    profile = {
        "driver": "GTiff",
        "dtype": aligned.dtype,
        "count": band_count,
        "height": ref_height,
        "width": ref_width,
        "crs": ref_crs,
        "transform": ref_transform,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if dst_nodata is not None:
        profile["nodata"] = dst_nodata

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(aligned)
        dst.update_tags(AREA_OR_POINT=area_or_point)


def align_to_reference(
    reference_path: str,
    output_dir: str,
    *,
    align_images: bool = True,
    align_dsm: bool = True,
) -> dict[str, Path]:
    """Align processed outputs to the reference GeoTIFF grid.

    Args:
        reference_path: Path to the reference GeoTIFF
        output_dir: Base output directory (contains processed/ outputs)
        align_images: Align imagery outputs
        align_dsm: Align DSM outputs

    Returns:
        Dict mapping data type to aligned output path
    """
    if not align_images and not align_dsm:
        return {}

    output_root = Path(output_dir)
    processed_dir = output_root / "processed"
    aligned_dir = output_root / "aligned"

    with rasterio.open(reference_path) as ref:
        ref_crs = ref.crs
        if ref_crs is None:
            raise ValueError(f"Reference GeoTIFF has no CRS: {reference_path}")
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
        ref_bounds = ref.bounds

    outputs: dict[str, Path] = {}
    ref_stem = Path(reference_path).stem

    if align_images:
        image_dir = _select_highest_resolution_dir(processed_dir / "image")
        image_paths = _collect_geotiffs(image_dir)
        output_path = aligned_dir / f"{ref_stem}_image.tif"
        _align_outputs(
            image_paths,
            output_path,
            ref_crs=ref_crs,
            ref_transform=ref_transform,
            ref_width=ref_width,
            ref_height=ref_height,
            ref_bounds=ref_bounds,
            resampling=Resampling.lanczos,
            nodata=None,
            area_or_point="Area",
            max_bands=3,
        )
        outputs["image"] = output_path

    if align_dsm:
        dsm_dir = _select_highest_resolution_dir(processed_dir / "dsm")
        dsm_paths = _collect_geotiffs(dsm_dir)
        output_path = aligned_dir / f"{ref_stem}_dsm.tif"
        _align_outputs(
            dsm_paths,
            output_path,
            ref_crs=ref_crs,
            ref_transform=ref_transform,
            ref_width=ref_width,
            ref_height=ref_height,
            ref_bounds=ref_bounds,
            resampling=Resampling.bilinear,
            nodata=DSM_NODATA,
            area_or_point="Point",
        )
        outputs["dsm"] = output_path

    return outputs
