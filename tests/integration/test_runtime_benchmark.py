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
from pathlib import Path

import pytest

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

    # Keep WMS off to avoid network variability
    monkeypatch.setenv("GEORAFFER_DISABLE_WMS", "1")

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
