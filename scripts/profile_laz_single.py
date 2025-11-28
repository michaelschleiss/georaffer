"""Profile single-worker LAZ conversion to identify bottlenecks."""

import os
import pathlib
import shutil
import sys
import tempfile
import time

# Ensure project import
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from georaffer.pipeline import convert_tiles

os.environ.setdefault("GEORAFFER_DISABLE_WMS", "1")


def main():
    candidates = [
        pathlib.Path("/mnt/Storage/res/raw/dsm"),
        pathlib.Path("/mnt/Storage/res/raw/pointcloud"),
    ]
    root = next((p for p in candidates if p.exists()), candidates[0])
    if not root.exists():
        print(f"No LAZ root found; checked: {', '.join(str(p) for p in candidates)}")
        return

    laz = sorted(root.glob("*_nw_*.laz"))[:20]
    if not laz:
        laz = sorted(root.glob("*.laz"))[:20]
    if not laz:
        print(f"No LAZ files found under {root}; aborting.")
        return

    base = pathlib.Path(tempfile.mkdtemp(prefix="georaffer-prof-single-"))
    raw = base / "raw" / "dsm"
    raw.mkdir(parents=True)
    for p in laz:
        os.symlink(p, raw / p.name)

    proc = base / "proc"
    t0 = time.perf_counter()
    stats = convert_tiles(
        str(raw.parent),
        str(proc),
        resolutions=[1000],
        max_workers=1,
        num_threads_per_worker=32,
        process_images=False,
        process_pointclouds=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"elapsed {elapsed:.3f}s laz={stats.laz_converted} failed={stats.failed}")
    shutil.rmtree(base)


if __name__ == "__main__":
    main()
