#!/usr/bin/env python3
"""
Quick, opt-in benchmark for convert_laz on one or more LAZ files.

Usage:
  python scripts/bench_convert_laz.py --inputs "/mnt/Storage/res/raw/pointcloud/*.laz" \
      --region RLP --backend parallel --chunk-points 1000000 --repeat 1

Defaults are conservative and skip if no files match.
"""

from __future__ import annotations

import argparse
import glob
import sys
import tempfile
import time
from pathlib import Path

from georaffer.config import Region
from georaffer.converters.laz import convert_laz


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark convert_laz on LAZ files")
    p.add_argument(
        "--inputs",
        required=True,
        help="Glob or single path to LAZ files (e.g., /mnt/.../*.laz)",
    )
    p.add_argument(
        "--region",
        default="RLP",
        choices=[r.name for r in Region],
        help="Region code (affects default resolution)",
    )
    p.add_argument(
        "--backend",
        choices=["parallel", "single"],
        default="parallel",
        help="lazrs backend via env BACKEND for convert_laz",
    )
    p.add_argument(
        "--stream-chunks",
        type=int,
        default=4_000_000,
        help="Chunk size for streaming rasterization; set to 0 to disable streaming",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of repetitions per file",
    )
    p.add_argument(
        "--target-sizes",
        type=str,
        default="",
        help="Comma-separated target sizes (pixels) for resampling, e.g. '500,2000,5000'. "
        "If empty, outputs native resolution only.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted(Path(p).resolve() for p in glob.glob(args.inputs))
    if not files:
        print("No files matched; exiting.")
        return 0

    region = Region[args.region]

    print(
        f"Running convert_laz benchmark on {len(files)} file(s); "
        f"backend={args.backend}, repeat={args.repeat}"
    )

    # convert_laz uses laspy backend set via env; we respect caller's env if set.
    if args.backend == "single":
        import os

        os.environ["BACKEND"] = "single"
    else:
        import os

        os.environ["BACKEND"] = "parallel"

    for path in files:
        for i in range(args.repeat):
            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / f"{path.stem}.tif"
                target_sizes = (
                    [int(x) for x in args.target_sizes.split(",") if x.strip()]
                    if args.target_sizes
                    else None
                )
                if target_sizes:
                    output_paths = {
                        sz: str(Path(td) / f"{path.stem}_{sz}.tif") for sz in target_sizes
                    }
                else:
                    output_paths = str(out)
                start = time.perf_counter()
                stream_arg = (
                    args.stream_chunks if args.stream_chunks and args.stream_chunks > 0 else None
                )
                ok = convert_laz(
                    str(path),
                    output_paths,
                    region,
                    stream_chunks=stream_arg,
                )
                dur = time.perf_counter() - start
                if target_sizes:
                    sizes = {
                        sz: (Path(output_paths[sz]).stat().st_size / 1e6) for sz in target_sizes
                    }
                    size_str = ", ".join(f"{sz}:{sizes[sz]:.1f}MB" for sz in sorted(target_sizes))
                else:
                    size_str = f"{(out.stat().st_size / 1e6) if out.exists() else 0.0:.1f}MB"
                print(
                    f"{path.name} run {i + 1}/{args.repeat}: "
                    f"ok={ok} time={dur:.3f}s out_mb={size_str}"
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
