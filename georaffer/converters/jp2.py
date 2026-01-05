"""JP2 to GeoTIFF converter with optional RLP tile splitting."""

import os
import time
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
from rasterio.errors import NotGeoreferencedWarning
import warnings
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine, from_bounds

from georaffer.config import (
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


def convert_jp2(
    input_path: str,
    output_paths: str | dict[int, str],
    region: Region,
    year: str,
    resolutions: list[int] | None = None,
    num_threads: int | None = None,
    grid_size_km: float = 1.0,
    profiling: bool = False,
) -> bool:
    """Convert JP2 to GeoTIFF(s) with optional resampling and RLP splitting.

    Unified function handling both single and batch conversion.

    Args:
        input_path: Path to input JP2 file
        output_paths: Single output path (str) or dict mapping resolution -> path
        region: Region code ('NRW', 'RLP')
        year: Year string
        resolutions: Output tile dimensions in pixels per 1km grid. For example:
            - 2000 = 2000×2000 px per 1km tile (0.5m GSD)
            - 1000 = 1000×1000 px per 1km tile (1.0m GSD)
            - 500 = 500×500 px per 1km tile (2.0m GSD)
            None for native resolution. Must be divisible by split ratio when
            splitting tiles (e.g., RLP 2km→1km split requires even values).
        num_threads: Number of threads for reprojection (default: REPROJECT_THREADS)
        grid_size_km: User grid size to determine split factor
        profiling: Enable timing output for performance analysis

    Returns:
        True if successful

    Raises:
        RuntimeError: If JP2 cannot be read, split coordinates invalid, or GeoTIFF write fails

    Example:
        >>> # NRW: Single tile conversion
        >>> convert_jp2("raw/nrw.jp2", "out.tif", Region.NRW, "2021", [1000])
        True

        >>> # RLP: 2km tile split to four 1km tiles (grid_size_km triggers 2x2 split)
        >>> convert_jp2("raw/rlp.jp2", {2000: "out.tif"}, Region.RLP, "2023",
        ...             resolutions=[2000], grid_size_km=1.0)
        True
    """
    # Normalize to dict format
    t_read_start = time.perf_counter()
    if isinstance(output_paths, str):
        if resolutions and len(resolutions) == 1:
            output_paths = {resolutions[0]: output_paths}
        else:
            output_paths = {None: output_paths}
            resolutions = [None]
    elif resolutions is None:
        resolutions = list(output_paths.keys())

    threads = num_threads or REPROJECT_THREADS

    try:
        # GDAL environment variables for performance optimization
        env_opts = {
            # Multi-threading: enable parallel decoding across GDAL layers
            "GDAL_NUM_THREADS": "ALL_CPUS",  # GDAL core operations
            "NUM_THREADS": "ALL_CPUS",  # Rasterio/underlying drivers
            "OPJ_NUM_THREADS": "ALL_CPUS",  # OpenJPEG 2000 decoder
            # I/O optimization: reduce syscalls by reading larger chunks
            "GDAL_ONE_BIG_READ": "YES",  # Single read operation for small files
            # GeoTIFF output optimization: bypass some layers for direct writes
            "GTIFF_DIRECT_IO": "YES",  # Direct I/O when possible
            "GTIFF_VIRTUAL_MEM_IO": "YES",  # Memory-mapped I/O for GTiff
            # Metadata optimization: skip expensive directory scans
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",  # Don't scan for sidecar files
            "GDAL_PAM_ENABLED": "NO",  # Disable auxiliary metadata (.aux.xml)
            # IMPORTANT: GDAL_CACHEMAX is NOT set here
            # Setting it (e.g., "512") causes 10x+ slowdown when reading JP2 overviews
            # GDAL's default cache strategy works better for downsampled reads
        }
        open_opts = {"NUM_THREADS": "ALL_CPUS", "USE_TILE_AS_BLOCK": "YES"}

        want_native = None in resolutions
        max_req = None if want_native else max(resolutions)
        year_int_val = int(year) if isinstance(year, str) and year.isdigit() else None

        tile_km = get_tile_size_km(region)
        ratio_int = round(tile_km / grid_size_km) if grid_size_km > 0 else 1
        split_factor = compute_split_factor(tile_km, grid_size_km)

        def _choose_rlevel(size: int, target: int) -> int:
            """Pick the smallest overview level whose dimension is still >= target.

            JP2 files contain pyramidal overviews at powers of 2 (1/2, 1/4, 1/8, etc).
            Reading from an overview is much faster than downsampling full resolution.

            Strategy: Use the smallest overview that's still larger than target, then
            let rasterio downsample the small difference. Avoids decoding unnecessary pixels.

            Example: For 10000px source wanting 1200px output:
              - Level 0: 10000px (full res) - too large, slow
              - Level 3: 1250px (10000>>3) - perfect! ~5% oversample, then downsample to 1200px

            Returns: Overview level (0=full res, 1=half, 2=quarter, ..., 8=1/256th)
            """
            for lvl in range(0, 9):
                dim = size >> lvl  # Bit shift: efficient division by 2^lvl
                if dim < target:
                    return max(lvl - 1, 0)  # Previous level was last one >= target
            return 8  # Fallback: even 1/256th resolution is too big (rare)

        with rasterio.Env(**env_opts):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=NotGeoreferencedWarning,
                    message="Dataset has no geotransform",
                )
                with rasterio.open(input_path, **open_opts) as src:
                    # For splits, read enough pixels so each sub-tile has the requested resolution
                    target_full = None
                    split_max = max([r for r in resolutions if r], default=None)
                    if split_max and split_factor > 1:
                        scale = max(1, ratio_int)
                        target_full = min(src.width, split_max * scale)

                    effective_target = target_full or max_req
    
                    if (
                        effective_target
                        and isinstance(src.width, (int, float))
                        and isinstance(src.height, (int, float))
                        and effective_target < src.width
                        and effective_target < src.height
                    ):
                        rlevel = _choose_rlevel(src.width, effective_target)
                        factor = 2**rlevel
                        target = (src.count, src.height // factor, src.width // factor)
                        data = src.read(
                            out_shape=target,
                            masked=False,
                            boundless=False,
                        )
                        transform = src.transform * Affine.scale(factor, factor)
                    else:
                        data = src.read()
                        transform = src.transform
                    crs = src.crs
                    colorinterp = src.colorinterp
                    t_read = time.perf_counter() - t_read_start
    
                    filename = os.path.basename(input_path)
    
                    coords = parse_tile_coords(filename)
    
                    if (crs is None or transform == Affine.identity()) and coords:
                        tile_m = get_tile_size_km(region) * METERS_PER_KM
                        left = coords[0] * METERS_PER_KM
                        bottom = coords[1] * METERS_PER_KM
                        right = left + tile_m
                        top = bottom + tile_m
                        transform = from_bounds(left, bottom, right, top, data.shape[-1], data.shape[-2])
                        crs = (
                            "EPSG:25833"
                            if region == Region.BB
                            else "EPSG:25832"
                        )
    
                    if len(colorinterp) == 1 and colorinterp[0] == ColorInterp.palette:
                        cmap = src.colormap(1)
                        if cmap:
                            max_idx = max(cmap.keys())
                            rgba_len = len(next(iter(cmap.values())))
                            lut = np.zeros((max_idx + 1, rgba_len), dtype=np.uint8)
                            for idx, rgba in cmap.items():
                                lut[idx] = rgba
                            indices = data[0] if data.ndim == 3 else data
                            expanded = lut[indices]
                            if rgba_len == 4 and np.all(expanded[..., 3] == 255):
                                expanded = expanded[..., :3]
                            data = expanded.transpose(2, 0, 1)

                    # Strip alpha channel if present (4 bands -> 3 bands RGB)
                    # This ensures cv2 can read the output correctly
                    if data.shape[0] == 4:
                        data = data[:3]
    
                    if split_factor > 1:
                        if not coords:
                            raise RuntimeError(
                                f"Splitting failed: could not read grid coordinates from '{filename}'. "
                                "This tile name must include grid coords (e.g., …_350_5600_…)."
                            )
                        rows, cols = data.shape[1], data.shape[2]
                        if rows < ratio_int or cols < ratio_int:
                            raise RuntimeError(
                                f"Splitting failed: the data read is only {rows}x{cols} pixels, "
                                f"but a {ratio_int}x{ratio_int} split needs at least {ratio_int} pixels on each side. "
                                "Try a higher read resolution or a coarser grid."
                            )
                        if rows % ratio_int != 0 or cols % ratio_int != 0:
                            raise RuntimeError(
                                f"Splitting failed: the read window {rows}x{cols} does not divide evenly into a {ratio_int}x{ratio_int} grid. "
                                "Read at a resolution where both dimensions are multiples of the split ratio, or choose a compatible grid size."
                            )
                        return _convert_split_jp2(
                            data,
                            transform,
                            crs,
                            coords,
                            output_paths,
                            resolutions,
                            filename,
                            region,
                            year,
                            ratio_int,
                            grid_size_km=grid_size_km,
                            num_threads=threads,
                            profiling=profiling,
                            t_read=t_read,
                        )
    
                    # Standard conversion (no splitting)
                    timing = []
                    for resolution in resolutions:
                        path_str = output_paths.get(resolution)
                        if not path_str:
                            continue
                        output_path = Path(path_str)
    
                        if resolution:
                            t_res_start = time.perf_counter()
                            out_data, out_transform = resample_raster(
                                data,
                                transform,
                                crs,
                                resolution,
                                threads,
                                dtype=np.uint8,
                                resampling=Resampling.lanczos,
                            )
                            t_res = time.perf_counter() - t_res_start
                        else:
                            out_data, out_transform = data, transform
                            t_res = 0.0
    
                        t_write_start = time.perf_counter()
                        write_geotiff(
                            output_path,
                            out_data,
                            out_transform,
                            crs,
                        )
                        t_write = time.perf_counter() - t_write_start
                        timing.append((resolution, t_res, t_write))
    
                    if profiling:
                        total_resample = sum(t[1] for t in timing)
                        total_write = sum(t[2] for t in timing)
                        total = t_read + total_resample + total_write
                        timing_str = "; ".join(
                            f"res={res or 'native'} resample={t_res:.3f}s write={t_w:.3f}s"
                            for res, t_res, t_w in timing
                        )
                        print(
                            f"[jp2] {os.path.basename(input_path)} region={region} split_factor={compute_split_factor(tile_km, grid_size_km)} "
                            f"read={t_read:.3f}s resample_total={total_resample:.3f}s write_total={total_write:.3f}s total={total:.3f}s | {timing_str}"
                        )
    
        return True

    except Exception as e:
        raise RuntimeError(f"Failed to convert {input_path}: {e}") from e


def _convert_split_jp2(
    data: np.ndarray,
    transform: Affine,
    crs,
    coords: tuple[int, int],
    output_paths: dict[int, str],
    resolutions: list[int],
    source_file: str,
    region: str,
    year: str | None,
    ratio: int,
    grid_size_km: float = 1.0,
    num_threads: int = REPROJECT_THREADS,
    profiling: bool = False,
    t_read: float = 0.0,
) -> bool:
    """Split a JP2 tile evenly into ratio x ratio subtiles and convert each."""
    time.perf_counter()

    base_x, base_y = coords
    rows, cols = data.shape[1], data.shape[2]
    tile_size_m = METERS_PER_KM  # grid indices are 1km-based for both regions
    grid_size_m = round(grid_size_km * 1000)

    total_resample = 0.0
    total_write = 0.0
    utm_zone = utm_zone_str_for_region(region)

    # Split the source tile into ratio × ratio sub-tiles
    # Iterate in row-major order (top-to-bottom, left-to-right in image space)
    for r_idx in range(ratio):
        row_start, row_end = compute_split_bounds(r_idx, rows, ratio)
        for c_idx in range(ratio):
            col_start, col_end = compute_split_bounds(c_idx, cols, ratio)
            tile_data = data[:, row_start:row_end, col_start:col_end]

            # Calculate new grid coordinates for this sub-tile
            # Coordinate system explanation:
            #   Grid indices are in km (e.g., base tile 362,5604 covers 362-363km E, 5604-5605km N)
            #   X increases eastward (columns in image), Y increases northward
            #   In image/array space: row 0 = north (max Y), increasing rows go south
            #
            # Example for 2×2 split of 2km tile at base (362km, 5604km) → four 1km tiles:
            #   r_idx=0, c_idx=0: grid (362, 5605) → UTM (362000m, 5605000m) [NW, top-left]
            #   r_idx=0, c_idx=1: grid (363, 5605) → UTM (363000m, 5605000m) [NE, top-right]
            #   r_idx=1, c_idx=0: grid (362, 5604) → UTM (362000m, 5604000m) [SW, bottom-left]
            #   r_idx=1, c_idx=1: grid (363, 5604) → UTM (363000m, 5604000m) [SE, bottom-right]
            new_x = base_x + c_idx  # Eastward increment
            new_y = base_y + (ratio - 1 - r_idx)  # Northward from base (invert row index)

            # Adjust geotransform for this sub-tile's pixel offset within source image
            split_transform = transform * Affine.translation(col_start, row_start)

            for resolution in resolutions:
                base_path = output_paths.get(resolution)
                if not base_path:
                    continue

                # Calculate full UTM coordinates in meters for output filename
                # Grid indices (base_x, base_y) are in km, multiply by 1000 to get meters
                # Then add the sub-tile offset within the split
                # Example: 2km base tile (362km, 5604km) split to 1km at c_idx=1, r_idx=0:
                #   easting  = 362 * 1000 + 1 * 1000 = 363000m E
                #   northing = 5604 * 1000 + (2-1-0) * 1000 = 5605000m N
                # Output filename: rlp_32_363000_5605000_2023.tif (UTM coordinates)
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
                if resolution:
                    out_data, out_transform = resample_raster(
                        tile_data,
                        split_transform,
                        crs,
                        resolution,
                        num_threads,
                        dtype=np.uint8,
                        resampling=Resampling.lanczos,
                    )
                else:
                    out_data, out_transform = tile_data, split_transform
                total_resample += time.perf_counter() - t_res_start

                t_write_start = time.perf_counter()
                write_geotiff(output_path, out_data, out_transform, crs)
                total_write += time.perf_counter() - t_write_start

    if profiling:
        split_factor = ratio * ratio
        total = t_read + total_resample + total_write
        print(
            f"[jp2] {source_file} region={region} split={ratio}x{ratio}={split_factor} "
            f"read={t_read:.3f}s resample={total_resample:.3f}s write={total_write:.3f}s total={total:.3f}s"
        )

    return True
