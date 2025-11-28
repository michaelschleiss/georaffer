"""Rebuild provenance.csv from existing processed outputs.

Usage:
    python scripts/regenerate_provenance.py --processed-dir /path/to/processed
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

# Allow running as standalone script without installing the package
if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from georaffer.converters.utils import parse_tile_coords
from georaffer.metadata import get_wms_metadata


def iter_tiffs(processed_dir: str) -> list[tuple[str, str]]:
    records = []
    for sub in ("image", "dsm"):
        root = os.path.join(processed_dir, sub)
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(".tif"):
                    records.append((sub, os.path.join(dirpath, f)))
    return records


def extract_region(filename: str) -> str:
    return "Region.RLP" if filename.startswith("rlp_") else "Region.NRW"


def extract_year(filename: str) -> str:
    """Extract acquisition year from processed filename.

    Filenames encode year as the LAST 4-digit token (e.g., *_2023.tif).
    Taking the last match avoids mistaking grid indices (also 4 digits)
    for the year.
    """
    stem = os.path.splitext(filename)[0]
    digits = [p for p in stem.split("_") if p.isdigit() and len(p) == 4]
    return digits[-1] if digits else "latest"


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate provenance.csv from processed outputs."
    )
    parser.add_argument(
        "--processed-dir",
        required=True,
        help="Path to processed directory (contains image/ and dsm/)",
    )
    parser.add_argument(
        "--grid-size-m", type=int, default=1000, help="Grid size in meters (default: 1000)"
    )
    parser.add_argument(
        "--output", default=None, help="Output CSV path (default: <processed-dir>/provenance.csv)"
    )
    parser.add_argument(
        "--wms-timing", action="store_true", help="Print per-tile WMS timing for NRW tiles"
    )
    args = parser.parse_args()

    processed_dir = os.path.abspath(args.processed_dir)
    out_csv = args.output or os.path.join(processed_dir, "provenance.csv")

    rows = []
    records = iter_tiffs(processed_dir)
    seen = set()
    for tile_type, path in records:
        fname = os.path.basename(path)
        if fname in seen:
            continue
        seen.add(fname)
        region = extract_region(fname)
        year = extract_year(fname)
        coords = parse_tile_coords(fname)
        gx, gy = coords if coords else (None, None)

        acquisition_date = None
        metadata_source = None
        if region == "Region.NRW" and gx is not None and gy is not None:
            import time

            t0 = time.perf_counter()
            try:
                center_x = gx * args.grid_size_m + args.grid_size_m / 2
                center_y = gy * args.grid_size_m + args.grid_size_m / 2
                wms_meta = get_wms_metadata(
                    center_x, center_y, "NRW", int(year) if year.isdigit() else None
                )
                if wms_meta:
                    acquisition_date = wms_meta.get("acquisition_date")
                    metadata_source = wms_meta.get("metadata_source")
                if args.wms_timing:
                    elapsed = time.perf_counter() - t0
                    status = acquisition_date or "none"
                    print(f"[wms] {fname} ({gx},{gy}) {elapsed:.2f}s date={status}")
            except Exception as exc:
                if args.wms_timing:
                    elapsed = time.perf_counter() - t0
                    print(f"[wms] {fname} ({gx},{gy}) {elapsed:.2f}s error={exc}")
                pass

        rows.append(
            {
                "processed_file": fname,
                "source_file": "",
                "source_region": region,
                "grid_x": gx,
                "grid_y": gy,
                "year": year,
                "acquisition_date": acquisition_date,
                "file_type": "orthophoto" if tile_type == "image" else "dsm",
                "metadata_source": metadata_source,
                "point_count": "",
                "split_from": "",
                "processing_date": "",
            }
        )

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "processed_file",
                "source_file",
                "source_region",
                "grid_x",
                "grid_y",
                "year",
                "acquisition_date",
                "file_type",
                "metadata_source",
                "point_count",
                "split_from",
                "processing_date",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
