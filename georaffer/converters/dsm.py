"""Raster DSM to GeoTIFF converter with optional tile splitting."""

import os
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine

from georaffer.config import (
    DSM_DTYPE,
    DSM_NODATA,
    METERS_PER_KM,
    REPROJECT_THREADS,
    Region,
    get_tile_size_km,
    utm_zone_str_for_region,
)
from georaffer.converters.utils import (
    compute_split_bounds,
    generate_split_output_path,
    parse_tile_coords,
    resample_raster,
    write_geotiff,
)
from georaffer.grids import compute_split_factor
from georaffer.metadata import add_provenance_to_geotiff


def convert_dsm_raster(
    input_path: str,
    output_paths: str | dict[int, str],
    region: Region,
    year: str | None = None,
    target_sizes: list[int] | None = None,
    num_threads: int | None = None,
    grid_size_km: float = 1.0,
    profiling: bool = False,
) -> bool:
    """Convert a raster DSM (GeoTIFF) to GeoTIFF outputs with resampling/splitting."""
    # Normalize to dict format
    if isinstance(output_paths, str):
        if target_sizes and len(target_sizes) == 1:
            output_paths = {target_sizes[0]: output_paths}
        else:
            output_paths = {None: output_paths}
            target_sizes = [None]
    elif target_sizes is None:
        target_sizes = list(output_paths.keys())

    threads = num_threads or REPROJECT_THREADS
    tile_km = get_tile_size_km(region)
    ratio_int = round(tile_km / grid_size_km) if grid_size_km > 0 else 1
    split_factor = compute_split_factor(tile_km, grid_size_km)
    utm_zone = utm_zone_str_for_region(region)

    t_read_start = time.perf_counter()
    try:
        with rasterio.open(input_path) as src:
            raster = src.read(1)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata if src.nodata is not None else DSM_NODATA
        t_read = time.perf_counter() - t_read_start

        filename = os.path.basename(input_path)

        coords = parse_tile_coords(filename)
        if split_factor > 1:
            if not coords:
                raise RuntimeError(
                    f"Splitting failed: could not read grid coordinates from '{filename}'."
                )
            return _convert_split_dsm(
                raster,
                transform,
                crs,
                coords,
                output_paths,
                target_sizes,
                nodata,
                filename,
                region,
                year,
                ratio_int,
                grid_size_km=grid_size_km,
                num_threads=threads,
                utm_zone=utm_zone,
                profiling=profiling,
                t_read=t_read,
            )

        # Build provenance metadata
        metadata = {
            "source_file": filename,
            "source_region": region,
            "file_type": "dsm",
            "metadata_source": "bdom_tif",
        }
        if year and year.isdigit():
            metadata["acquisition_date"] = year

        timings = []
        for target_size in target_sizes:
            path_str = output_paths.get(target_size)
            if not path_str:
                continue
            output_path = Path(path_str)

            if target_size:
                t_res_start = time.perf_counter()
                out_data, out_transform = resample_raster(
                    raster,
                    transform,
                    crs,
                    target_size,
                    threads,
                    dtype=np.float32,
                    resampling=Resampling.bilinear,
                    nodata=nodata,
                )
                t_res = time.perf_counter() - t_res_start
            else:
                out_data, out_transform = raster, transform
                t_res = 0.0

            t_write_start = time.perf_counter()
            year_int = int(year) if year and year.isdigit() else None
            write_geotiff(
                output_path,
                out_data,
                out_transform,
                crs,
                dtype=DSM_DTYPE,
                count=1,
                nodata=nodata,
                area_or_point="Point",
                metadata=metadata,
                year_int=year_int,
            )
            t_write = time.perf_counter() - t_write_start
            add_provenance_to_geotiff(output_path, metadata, year=year_int)
            timings.append((target_size, t_res, t_write))

        if profiling:
            total_resample = sum(t[1] for t in timings)
            total_write = sum(t[2] for t in timings)
            total = t_read + total_resample + total_write
            timing_str = "; ".join(
                f"res={ts or 'native'} resample={t_res:.3f}s write={t_w:.3f}s"
                for ts, t_res, t_w in timings
            )
            print(
                f"[dsm] {filename} region={region} split_factor={split_factor} "
                f"read={t_read:.3f}s resample_total={total_resample:.3f}s "
                f"write_total={total_write:.3f}s total={total:.3f}s | {timing_str}"
            )

        return True

    except Exception as e:
        raise RuntimeError(f"Failed to convert {input_path}: {e.__class__.__name__}: {e}") from e


def _convert_split_dsm(
    raster: np.ndarray,
    transform: Affine,
    crs,
    coords: tuple[int, int],
    output_paths: dict[int, str],
    target_sizes: list[int],
    nodata: float,
    source_file: str,
    region: Region,
    year: str | None,
    ratio: int,
    grid_size_km: float = 1.0,
    num_threads: int = REPROJECT_THREADS,
    utm_zone: str = "32",
    profiling: bool = False,
    t_read: float = 0.0,
) -> bool:
    """Split a raster DSM into ratio x ratio subtiles and write each."""
    base_x, base_y = coords
    rows, cols = raster.shape
    tile_size_m = METERS_PER_KM
    grid_size_m = round(grid_size_km * METERS_PER_KM)

    total_resample = 0.0
    total_write = 0.0

    for r_idx in range(ratio):
        row_start, row_end = compute_split_bounds(r_idx, rows, ratio)
        for c_idx in range(ratio):
            col_start, col_end = compute_split_bounds(c_idx, cols, ratio)
            tile_raster = raster[row_start:row_end, col_start:col_end]

            new_x = base_x + c_idx
            new_y = base_y + (ratio - 1 - r_idx)
            split_transform = transform * Affine.translation(col_start, row_start)

            metadata = {
                "source_file": source_file,
                "source_region": region,
                "file_type": "dsm",
                "metadata_source": "bdom_tif",
                "grid_x": new_x,
                "grid_y": new_y,
            }
            if year and year.isdigit():
                metadata["acquisition_date"] = year

            for target_size in target_sizes:
                base_path = output_paths.get(target_size)
                if not base_path:
                    continue

                easting = int(base_x * tile_size_m + c_idx * grid_size_m)
                northing = int(base_y * tile_size_m + (ratio - 1 - r_idx) * grid_size_m)

                output_path = generate_split_output_path(
                    base_path,
                    new_x,
                    new_y,
                    easting=easting,
                    northing=northing,
                    utm_zone=utm_zone,
                )

                t_res_start = time.perf_counter()
                if target_size:
                    out_data, out_transform = resample_raster(
                        tile_raster,
                        split_transform,
                        crs,
                        target_size,
                        num_threads,
                        dtype=np.float32,
                        resampling=Resampling.bilinear,
                        nodata=nodata,
                    )
                else:
                    out_data, out_transform = tile_raster, split_transform
                total_resample += time.perf_counter() - t_res_start

                t_write_start = time.perf_counter()
                year_int = int(year) if year and year.isdigit() else None
                write_geotiff(
                    output_path,
                    out_data,
                    out_transform,
                    crs,
                    dtype=DSM_DTYPE,
                    count=1,
                    nodata=nodata,
                    area_or_point="Point",
                    metadata=metadata,
                    year_int=year_int,
                )
                add_provenance_to_geotiff(output_path, metadata, year=year_int)
                total_write += time.perf_counter() - t_write_start

    if profiling:
        split_factor = ratio * ratio
        total = t_read + total_resample + total_write
        print(
            f"[dsm] {source_file} region={region} split={ratio}x{ratio}={split_factor} "
            f"read={t_read:.3f}s resample={total_resample:.3f}s "
            f"write={total_write:.3f}s total={total:.3f}s"
        )

    return True
