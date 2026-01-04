#!/usr/bin/env python3
"""Fetch and summarize tile catalogs for selected regions."""

from __future__ import annotations

import argparse
import os
import sys
import time

# Allow running as standalone script without installing the package.
if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from georaffer.config import Region
from georaffer.downloaders import (
    BBDownloader,
    BWDownloader,
    BYDownloader,
    NRWDownloader,
    RLPDownloader,
)
from georaffer.reporting import print_catalog_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch full tile catalogs for selected regions."
    )
    parser.add_argument(
        "--output",
        default=".",
        help="Base output directory (default: current directory).",
    )
    parser.add_argument(
        "--region",
        nargs="+",
        choices=["nrw", "rlp", "bb", "bw", "by"],
        default=["nrw", "rlp", "bw", "by"],
        metavar="REGION",
        help="Regions to include: nrw rlp bb bw by (default: nrw rlp bw by).",
    )
    parser.add_argument(
        "--from",
        type=int,
        metavar="YEAR",
        dest="from_year",
        help="Include historic orthophotos from YEAR (availability varies).",
    )
    parser.add_argument(
        "--to",
        type=int,
        metavar="YEAR",
        dest="to_year",
        help="End year for historic orthophotos (default: present).",
    )
    parser.add_argument(
        "--refresh-catalog",
        action="store_true",
        help="Force refresh of tile catalog cache.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-region progress output.",
    )
    return parser.parse_args()


def normalize_regions(region_args: list[str]) -> list[Region]:
    region_map = {region.value.lower(): region for region in Region}
    normalized: list[Region] = []
    seen: set[Region] = set()
    for region_name in region_args:
        key = region_name.lower()
        region = region_map.get(key)
        if region is None:
            raise ValueError(
                f"Unknown region '{region_name}'. Use: nrw, rlp, bb, bw, by."
            )
        if region not in seen:
            normalized.append(region)
            seen.add(region)
    if not normalized:
        raise ValueError("At least one region is required.")
    return normalized


def _build_downloader(
    region: Region,
    output_dir: str,
    imagery_from: tuple[int, int | None] | None,
    quiet: bool,
):
    if region == Region.NRW:
        return NRWDownloader(output_dir, imagery_from=imagery_from, quiet=quiet)
    if region == Region.RLP:
        return RLPDownloader(output_dir, imagery_from=imagery_from, quiet=quiet)
    if region == Region.BB:
        return BBDownloader(output_dir, quiet=quiet)
    if region == Region.BW:
        return BWDownloader(output_dir, imagery_from=imagery_from, quiet=quiet)
    if region == Region.BY:
        return BYDownloader(output_dir, imagery_from=imagery_from, quiet=quiet)
    raise ValueError(f"Unsupported region: {region}")


def main() -> None:
    args = parse_args()

    if args.to_year is not None and args.from_year is None:
        raise ValueError("--to requires --from.")

    selected_regions = normalize_regions(args.region)
    imagery_from = (args.from_year, args.to_year) if args.from_year else None

    catalog_rows: list[tuple[str, int, int]] = []
    start = time.perf_counter()

    for region in selected_regions:
        downloader = _build_downloader(
            region, args.output, imagery_from=imagery_from, quiet=args.quiet
        )
        catalog = downloader.build_catalog(refresh=args.refresh_catalog)
        catalog_rows.append(
            (
                region.value,
                downloader.total_image_count,
                len(catalog.dsm_tiles),
            )
        )

    duration = time.perf_counter() - start
    if catalog_rows:
        print_catalog_summary(catalog_rows, duration)


if __name__ == "__main__":
    main()
