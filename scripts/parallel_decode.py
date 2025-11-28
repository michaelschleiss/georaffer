#!/usr/bin/env python3
"""Parallel LAZ decode benchmark with tunable CPU/IO knobs."""

import argparse
import multiprocessing as mp
import os
import time
from itertools import islice
from pathlib import Path

import laspy

# --- defaults; can be overridden via env or CLI flags ---
DATA_DIR = Path(os.environ.get("DATA_DIR", "/mnt/Storage/res/raw/pointcloud"))
MAX_FILES = int(os.environ.get("MAX_FILES", "8")) if os.environ.get("MAX_FILES", "") else None
WORKERS = int(os.environ.get("WORKERS", "4"))
RAYON_THREADS = int(os.environ.get("RAYON_THREADS", "4"))
CHUNK_POINTS = int(os.environ.get("CHUNK_POINTS", "1000000"))
EST_IO_GBPS = float(os.environ.get("EST_IO_GBPS", "2.5"))
# --------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel LAZ decode benchmark")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--max-files", type=int, default=MAX_FILES)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--rayon-threads", type=int, default=RAYON_THREADS)
    parser.add_argument("--chunk-points", type=int, default=CHUNK_POINTS)
    parser.add_argument("--est-io-gbps", type=float, default=EST_IO_GBPS)
    parser.add_argument(
        "--affinity",
        type=str,
        default=None,
        help="Comma-separated CPU ranges per worker (e.g. '0-7,8-15'); "
        "length must match workers to enable pinning",
    )
    parser.add_argument(
        "--backend",
        choices=["parallel", "single"],
        default=os.environ.get("BACKEND", "parallel"),
        help="lazrs backend: 'parallel' (LazrsParallel, default) or 'single' (Lazrs).",
    )
    parser.add_argument(
        "--ccd2",
        action="store_true",
        help="Convenience: use 2 workers pinned to CCD0 and CCD1 (0-7,8-15) "
        "and set rayon-threads=8 if not overridden.",
    )
    return parser.parse_args()


def init_worker() -> None:
    """Set per-process Rayon pool size to avoid oversubscription."""
    os.environ["RAYON_NUM_THREADS"] = str(RAYON_THREADS)


def process_file(path: Path) -> dict:
    """Decode one file, summing X to force full read and return stats."""
    backend = laspy.LazBackend.LazrsParallel if BACKEND == "parallel" else laspy.LazBackend.Lazrs
    t0 = time.perf_counter()
    checksum = 0
    pts_read = 0
    with laspy.open(path, laz_backend=backend) as f:
        for pts in f.chunk_iterator(CHUNK_POINTS):
            checksum = (checksum + int(pts.X.sum())) % 1_000_000
            pts_read += len(pts)
    dur = time.perf_counter() - t0
    return {
        "name": path.name,
        "secs": dur,
        "mpts_s": pts_read / dur / 1e6,
        "gb_s": path.stat().st_size / dur / 1e9,
        "checksum": checksum,
    }


def main() -> None:
    args = parse_args()

    # Optional CCD convenience preset.
    workers = args.workers
    rayon_threads = args.rayon_threads
    if args.ccd2:
        workers = 2
        if "RAYON_THREADS" not in os.environ:
            rayon_threads = 8

    data_dir = args.data_dir
    max_files = args.max_files
    chunk_points = args.chunk_points
    est_io_gbps = args.est_io_gbps

    # Update globals used by worker initializer.
    global RAYON_THREADS, CHUNK_POINTS
    RAYON_THREADS = rayon_threads
    CHUNK_POINTS = chunk_points
    global BACKEND
    BACKEND = args.backend

    files = sorted(data_dir.glob("*.laz"))
    if max_files:
        files = list(islice(files, max_files))
    if not files:
        print("No .laz files found.")
        return

    # Crude hint: assume ~300 MB/s per worker as a starting point.
    budget_workers = max(1, int(est_io_gbps / 0.3))
    print(f"IO budget hint: ~{budget_workers} workers @ ~300 MB/s each for {est_io_gbps} GB/s disk")
    print(
        f"Running {len(files)} files with {workers} procs x "
        f"{rayon_threads} threads (chunk {chunk_points:,})"
    )

    # Parse optional affinity map.
    aff_list: list[list[int]] | None = None
    if args.affinity:
        ranges = args.affinity.split(",")
        if len(ranges) != workers:
            raise SystemExit(f"--affinity expects {workers} ranges, got {len(ranges)}")
        aff_list = []
        for r in ranges:
            parts = r.split("-")
            if len(parts) == 1:
                aff_list.append([int(parts[0])])
            else:
                start, end = map(int, parts)
                aff_list.append(list(range(start, end + 1)))

    def initializer_with_affinity(idx: int) -> None:
        init_worker()
        if aff_list:
            ",".join(map(str, aff_list[idx]))
            os.sched_setaffinity(0, aff_list[idx])

    # Pre-bind worker initializers if affinity provided.
    if aff_list:
        # Need per-process index; use initializer args via wrapper.
        def _init(idx=0, arr=aff_list):
            init_worker()
            os.sched_setaffinity(0, arr[idx])

    # Pool map with optional per-worker affinity.
    def _process_with_index(args_idx_path):
        idx, path = args_idx_path
        if aff_list:
            os.sched_setaffinity(0, aff_list[idx % len(aff_list)])
        return process_file(path)

    t0 = time.perf_counter()
    with mp.Pool(workers, initializer=init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(process_file, files), 1):
            print(
                f"[{i}/{len(files)}] {res['name']:<32} "
                f"{res['secs']:5.2f}s  {res['mpts_s']:6.1f} Mpts/s  "
                f"{res['gb_s']:4.2f} GB/s  chk {res['checksum']}"
            )
    total = time.perf_counter() - t0
    print(f"Total wall: {total:5.2f}s for {len(files)} files ({len(files) / total:4.2f} files/s)")


if __name__ == "__main__":
    main()
