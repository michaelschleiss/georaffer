#!/usr/bin/env python3
import argparse
import time

from georaffer.pipeline import convert_tiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()

    start = time.perf_counter()
    stats = convert_tiles(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        resolutions=[1000],
        max_workers=args.workers,
        num_threads_per_worker=None,
        process_images=False,
        process_pointclouds=True,
    )
    duration = time.perf_counter() - start
    print(f"duration {duration}")
    print(stats)


if __name__ == "__main__":
    main()
