"""Benchmark convert_tiles using real data if available.

This uses the first JP2 and LAZ files found under REAL_DATA_ROOT. It symlinks
them into a temp raw directory, runs the full converters, and records a single
timing via pytest-benchmark.

Data sources (checked in order):
1. REAL_DATA_ROOT env var
2. Default path: /mnt/Storage/res/raw
3. Downloader test cache: ~/.cache/georaffer/test_tiles/raw
"""

import os
import time
from pathlib import Path

import pytest
import rasterio

from georaffer.conversion import convert_tiles


def _find_real_data_root():
    """Find real data directory, checking multiple locations."""
    if env_path := os.environ.get("REAL_DATA_ROOT"):
        return Path(env_path)

    candidates = [
        Path("/mnt/Storage/res/raw"),
        Path.home() / ".cache" / "georaffer" / "test_tiles" / "raw",
    ]
    for path in candidates:
        if path.exists() and (path / "image").exists():
            return path
    return candidates[0]  # Return first candidate for error message


REAL_DATA_ROOT = _find_real_data_root()


def _pick_sample_files(root: Path):
    jp2_dir = root / "image"
    laz_dirs = [root / "dsm", root / "pointcloud"]
    laz_dir = next((d for d in laz_dirs if d.exists()), laz_dirs[0])
    jp2_files = sorted(jp2_dir.glob("*.jp2"))
    laz_files = sorted(laz_dir.glob("*.laz"))
    if not jp2_files or not laz_files:
        return None, None
    # Prefer NRW (_nw_) to avoid RLP split complications
    nw_jp2 = [p for p in jp2_files if "_nw_" in p.name]
    nw_laz = [p for p in laz_files if "_nw_" in p.name]
    jp2_pick = nw_jp2[0] if nw_jp2 else jp2_files[0]
    laz_pick = nw_laz[0] if nw_laz else laz_files[0]
    return jp2_pick, laz_pick


@pytest.mark.integration
@pytest.mark.benchmark(min_rounds=1, disable_gc=True)
def test_convert_tiles_benchmark_real_data(tmp_path, benchmark, monkeypatch):
    if not REAL_DATA_ROOT.exists():
        pytest.skip(f"REAL_DATA_ROOT not found: {REAL_DATA_ROOT}")

    jp2_src, laz_src = _pick_sample_files(REAL_DATA_ROOT)
    if not jp2_src or not laz_src:
        pytest.skip("No JP2/LAZ files found in real data root")

    # Stub WMS to avoid network variability in benchmarks
    monkeypatch.delenv("GEORAFFER_DISABLE_WMS", raising=False)
    monkeypatch.setenv("GEORAFFER_DISABLE_PROCESS_POOL", "1")

    def fake_wms_metadata(*_args, **_kwargs):
        return {"acquisition_date": "2020-01-01", "metadata_source": "test"}

    monkeypatch.setattr("georaffer.metadata.get_wms_metadata_for_region", fake_wms_metadata)
    monkeypatch.setattr("georaffer.conversion.get_wms_metadata_for_region", fake_wms_metadata)
    monkeypatch.setattr("georaffer.workers.get_wms_metadata_for_region", fake_wms_metadata)

    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    (raw_dir / "image").mkdir(parents=True, exist_ok=True)
    (raw_dir / "dsm").mkdir(parents=True, exist_ok=True)

    # Symlink real data into temp raw directory
    os.symlink(jp2_src, raw_dir / "image" / jp2_src.name)
    os.symlink(laz_src, raw_dir / "dsm" / laz_src.name)

    def run():
        return convert_tiles(
            str(raw_dir),
            str(processed_dir),
            resolutions=[1000],
            max_workers=1,
        )

    stats = benchmark.pedantic(run, iterations=1, rounds=1)
    assert stats.jp2_converted == 1
    assert stats.laz_converted == 1


@pytest.mark.integration
def test_gdal_cachemax_impact(tmp_path):
    """Benchmark GDAL_CACHEMAX impact on JP2 overview reading.

    Tests the claim in jp2.py that setting GDAL_CACHEMAX causes slowdown
    when reading JP2 overviews.
    """
    if not REAL_DATA_ROOT.exists():
        pytest.skip(f"REAL_DATA_ROOT not found: {REAL_DATA_ROOT}")

    jp2_src, _ = _pick_sample_files(REAL_DATA_ROOT)
    if not jp2_src:
        pytest.skip("No JP2 files found in real data root")

    iterations = 5

    def read_jp2_with_overview(jp2_path: Path, env_opts: dict) -> float:
        """Read JP2 at reduced resolution (simulating overview read)."""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            with rasterio.Env(**env_opts):
                with rasterio.open(jp2_path) as src:
                    # Read at 1/4 resolution to trigger overview usage
                    out_shape = (src.count, src.height // 4, src.width // 4)
                    _ = src.read(out_shape=out_shape)
            times.append(time.perf_counter() - start)
        return sum(times) / len(times)

    # Baseline: without GDAL_CACHEMAX (GDAL default)
    baseline_opts = {
        "GDAL_NUM_THREADS": "ALL_CPUS",
        "NUM_THREADS": "ALL_CPUS",
        "OPJ_NUM_THREADS": "ALL_CPUS",
    }

    # With GDAL_CACHEMAX set to 512MB
    cachemax_opts = {
        **baseline_opts,
        "GDAL_CACHEMAX": 512,
    }

    # Warm up (first read is always slower due to file system caching)
    read_jp2_with_overview(jp2_src, baseline_opts)
    read_jp2_with_overview(jp2_src, cachemax_opts)

    # Actual benchmark
    time_without_cachemax = read_jp2_with_overview(jp2_src, baseline_opts)
    time_with_cachemax = read_jp2_with_overview(jp2_src, cachemax_opts)

    ratio = time_with_cachemax / time_without_cachemax

    print(f"\n{'='*60}")
    print(f"GDAL_CACHEMAX Benchmark Results ({jp2_src.name})")
    print(f"{'='*60}")
    print(f"Without GDAL_CACHEMAX: {time_without_cachemax:.3f}s (avg of {iterations})")
    print(f"With GDAL_CACHEMAX=512: {time_with_cachemax:.3f}s (avg of {iterations})")
    print(f"Ratio (with/without): {ratio:.2f}x")
    print(f"{'='*60}")

    if ratio > 1.5:
        print("CONCLUSION: GDAL_CACHEMAX causes significant slowdown")
    elif ratio < 0.7:
        print("CONCLUSION: GDAL_CACHEMAX improves performance")
    else:
        print("CONCLUSION: No significant difference")

    # Don't assert - just report results
    # The test passes regardless; we want to see the data
