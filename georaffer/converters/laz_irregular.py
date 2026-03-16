"""Irregular LAZ to raster DSM conversion via an optional PDAL backend."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import rasterio

from georaffer.converters.dsm import convert_dsm_raster


def build_irregular_pdal_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    *,
    native_resolution: float,
    default_srs: str | None = None,
) -> dict:
    """Build a PDAL pipeline for triangulated surface rasterization."""
    reader = {
        "type": "readers.las",
        "filename": str(input_path),
    }
    if default_srs:
        reader["default_srs"] = default_srs

    return {
        "pipeline": [
            reader,
            {
                "type": "filters.delaunay",
            },
            {
                "type": "filters.faceraster",
                "resolution": native_resolution,
            },
            {
                "type": "writers.raster",
                "filename": str(output_path),
            },
        ]
    }


def ensure_pdal_available() -> str:
    """Return the PDAL executable path or raise with install guidance."""
    pdal_path = shutil.which("pdal")
    if pdal_path:
        return pdal_path
    raise RuntimeError(
        "Irregular LAZ rasterization requires the optional PDAL backend. "
        "Install PDAL in a dedicated Pixi feature/environment or on the system PATH."
    )


def run_pdal_pipeline(pdal_path: str, pipeline: dict, *, cwd: str | Path | None = None) -> None:
    """Run a PDAL pipeline from a temporary JSON file."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fp:
        json.dump(pipeline, fp)
        fp.flush()
        pipeline_path = Path(fp.name)

    try:
        subprocess.run(
            [pdal_path, "pipeline", str(pipeline_path)],
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        detail = stderr or stdout or str(e)
        raise RuntimeError(f"PDAL pipeline failed: {detail}") from e
    finally:
        pipeline_path.unlink(missing_ok=True)


def convert_laz_irregular(
    input_path: str,
    output_paths: str | dict[int, str],
    region,
    *,
    year: str | None = None,
    native_resolution: float,
    default_srs: str | None = None,
    target_sizes: list[int] | None = None,
    num_threads: int | None = None,
    grid_size_km: float = 1.0,
    profiling: bool = False,
) -> bool:
    """Rasterize an irregular LAZ via PDAL, then reuse the raster DSM output path."""
    t_start = time.perf_counter()
    pdal_path = ensure_pdal_available()

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_raster = Path(tmpdir) / Path(input_path).with_suffix(".tif").name
        pipeline = build_irregular_pdal_pipeline(
            input_path,
            temp_raster,
            native_resolution=native_resolution,
            default_srs=default_srs,
        )
        run_pdal_pipeline(pdal_path, pipeline, cwd=tmpdir)

        if not temp_raster.exists():
            raise RuntimeError("PDAL completed without producing a raster output.")

        with rasterio.open(temp_raster) as src:
            if src.crs is None:
                raise RuntimeError(
                    f"PDAL raster output for {Path(input_path).name} has no CRS; "
                    "irregular LAZ inputs must carry a projected metric CRS."
                )
            if src.width == 0 or src.height == 0:
                raise RuntimeError("PDAL raster output is empty.")

        ok = convert_dsm_raster(
            str(temp_raster),
            output_paths,
            region,
            year,
            target_sizes=target_sizes,
            num_threads=num_threads,
            grid_size_km=grid_size_km,
            profiling=profiling,
        )

    if profiling:
        elapsed = time.perf_counter() - t_start
        print(
            f"[laz_irregular] {Path(input_path).name} native_resolution={native_resolution:.2f}m "
            f"total={elapsed:.3f}s"
        )

    return ok
