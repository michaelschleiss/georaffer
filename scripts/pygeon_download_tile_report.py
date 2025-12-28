"""Report required and cached tiles for a pygeon (4Seasons) dataset."""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter

# Allow running as standalone script without installing the package.
if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from georaffer.config import (
    FEED_TIMEOUT,
    HTTP_POOL_MAXSIZE,
    METERS_PER_KM,
    MIN_FILE_SIZE,
    OUTPUT_TILE_SIZE_KM,
    UTM_ZONE_BY_REGION,
    Region,
)
from georaffer.downloaders import BrandenburgDownloader, NRWDownloader, RLPDownloader
from georaffer.grids import generate_tiles_by_zone, latlon_array_to_utm
from georaffer.inputs import load_from_pygeon
from georaffer.reporting import print_catalog_summary, print_table
from georaffer.tiles import RegionCatalog, calculate_required_tiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check required tiles for a pygeon dataset and verify local cache."
    )
    parser.add_argument(
        "dataset_path",
        help="Path to a 4Seasons campaign directory (ins.csv) or a parent directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory containing raw/ and processed/ tiles (used for cache checks).",
    )
    parser.add_argument(
        "--grid-size-km",
        type=float,
        default=OUTPUT_TILE_SIZE_KM,
        help=f"User grid size in km (default: {OUTPUT_TILE_SIZE_KM}).",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=0,
        help="Tile buffer around input: 0=exact, 1=+1 ring (default: 0).",
    )
    parser.add_argument(
        "--region",
        nargs="+",
        default=["nrw", "rlp", "bb"],
        metavar="REGION",
        help="Regions to include (nrw rlp bb). Defaults to all.",
    )
    parser.add_argument(
        "--from",
        type=int,
        metavar="YEAR",
        dest="from_year",
        help="Include historic orthophotos from YEAR (NRW/RLP only, availability varies).",
    )
    parser.add_argument(
        "--to",
        type=int,
        metavar="YEAR",
        dest="to_year",
        help="End year for historic orthophotos (default: present).",
    )
    parser.add_argument(
        "--type",
        dest="data_types",
        nargs="+",
        choices=["image", "dsm"],
        help="Limit report to specific data types (default: all).",
    )
    parser.add_argument(
        "--check-remote-size",
        action="store_true",
        help="Compare local file size to remote Content-Length when available.",
    )
    parser.add_argument(
        "--remote-workers",
        type=int,
        default=min(HTTP_POOL_MAXSIZE, 16),
        help="Parallel workers for remote size checks (default: 16 or HTTP pool max).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output for remote size checks and comparisons.",
    )
    parser.add_argument(
        "--max-sample",
        type=int,
        default=5,
        help="Max sample size to display for missing tiles/files (default: 5).",
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
            raise ValueError(f"Unknown region '{region_name}'. Use: nrw, rlp, bb.")
        if region not in seen:
            normalized.append(region)
            seen.add(region)
    if not normalized:
        raise ValueError("At least one region is required.")
    return normalized


def load_pygeon_coords(dataset_path: str) -> tuple[list[tuple[float, float]], int]:
    raw_coords = load_from_pygeon(dataset_path)
    coords_array = np.array(raw_coords)
    if coords_array.size == 0:
        raise ValueError("No coordinates found in pygeon dataset.")

    lons = coords_array[:, 1]
    zone_candidates = np.floor((lons + 180) / 6).astype(int) + 1
    unique_zones = set(zone_candidates.tolist())
    if len(unique_zones) > 1:
        raise ValueError("Pygeon dataset spans multiple UTM zones; split input by zone.")
    source_zone = unique_zones.pop()
    if source_zone not in (32, 33):
        raise ValueError(
            f"Pygeon dataset resolves to UTM zone {source_zone}; only zones 32 and 33 are supported."
        )

    utm_x, utm_y = latlon_array_to_utm(
        coords_array[:, 0], coords_array[:, 1], force_zone_number=source_zone
    )
    coords = list(zip(utm_x, utm_y))
    return coords, source_zone


def dedupe_coords(
    coords: list[tuple[float, float]], grid_size_km: float
) -> tuple[list[tuple[float, float]], int, int]:
    coords_array = np.array(coords)
    if coords_array.size == 0:
        return [], 0, 0
    grid_size_m = grid_size_km * METERS_PER_KM
    tile_x = (coords_array[:, 0] // grid_size_m).astype(int)
    tile_y = (coords_array[:, 1] // grid_size_m).astype(int)
    tiles = np.column_stack([tile_x, tile_y])
    _, unique_indices = np.unique(tiles, axis=0, return_index=True)
    unique_coords = coords_array[unique_indices].tolist()
    return unique_coords, len(coords_array), len(unique_coords)


def _rlp_native_coords(
    rlp_downloader: RLPDownloader,
    tiles_by_zone: dict[int, set[tuple[int, int]]],
    grid_size_km: float,
) -> set[tuple[int, int]]:
    rlp_zone = UTM_ZONE_BY_REGION[Region.RLP]
    rlp_user_tiles = tiles_by_zone.get(rlp_zone, set())
    native_coords: set[tuple[int, int]] = set()
    for x, y in rlp_user_tiles:
        utm_x = (x + 0.5) * grid_size_km * METERS_PER_KM
        utm_y = (y + 0.5) * grid_size_km * METERS_PER_KM
        rlp_coords, _ = rlp_downloader.utm_to_grid_coords(utm_x, utm_y)
        native_coords.add(rlp_coords)
    return native_coords


_THREAD_LOCAL = threading.local()


def _get_thread_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=0)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _THREAD_LOCAL.session = session
    return session


def _is_wms_url(downloader: object, url: str) -> bool:
    checker = getattr(downloader, "_is_wms_getmap", None)
    if callable(checker):
        try:
            return bool(checker(url))
        except Exception:
            return False
    lowered = url.lower()
    return "service=wms" in lowered and "request=getmap" in lowered


def _parse_content_length(headers: dict[str, str]) -> int | None:
    length = headers.get("Content-Length")
    if length:
        try:
            value = int(length)
            return value if value > 0 else None
        except ValueError:
            return None
    content_range = headers.get("Content-Range")
    if not content_range:
        return None
    match = re.search(r"/(\d+)$", content_range)
    if not match:
        return None
    return int(match.group(1))


def _fetch_remote_size(url: str, downloader: object, timeout: int) -> int | None:
    session = _get_thread_session()
    verify_ssl = getattr(downloader, "verify_ssl", True)

    response = None
    try:
        response = session.head(url, allow_redirects=True, timeout=timeout, verify=verify_ssl)
        if response.ok:
            size = _parse_content_length(response.headers)
            if size is not None:
                return size
    except requests.RequestException:
        pass
    finally:
        with suppress(Exception):
            if response is not None:
                response.close()

    response = None
    try:
        response = session.get(
            url,
            headers={"Range": "bytes=0-0"},
            stream=True,
            allow_redirects=True,
            timeout=timeout,
            verify=verify_ssl,
        )
        if response.ok:
            return _parse_content_length(response.headers)
    except requests.RequestException:
        return None
    finally:
        with suppress(Exception):
            if response is not None:
                response.close()

    return None


def _collect_remote_sizes(
    downloads_by_source: dict[str, list[tuple[str, str]]],
    downloaders: dict[str, object],
    max_workers: int,
    timeout: int,
    verbose: bool,
) -> tuple[dict[str, int | None], dict[str, int]]:
    unique_urls: dict[str, object] = {}
    total_existing = 0
    skipped_wms = 0

    for source_key, downloads in downloads_by_source.items():
        region_key, _ = source_key.split("_", maxsplit=1)
        downloader = downloaders.get(region_key)
        if downloader is None:
            continue
        for url, path in downloads:
            if not os.path.exists(path):
                continue
            total_existing += 1
            if url in unique_urls:
                continue
            if _is_wms_url(downloader, url):
                skipped_wms += 1
                continue
            unique_urls[url] = downloader

    sizes: dict[str, int | None] = {}
    if not unique_urls:
        return sizes, {
            "existing": total_existing,
            "checked": 0,
            "available": 0,
            "unknown": 0,
            "skipped_wms": skipped_wms,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_remote_size, url, downloader, timeout): url
            for url, downloader in unique_urls.items()
        }
        for fut in as_completed(futures):
            url = futures[fut]
            try:
                sizes[url] = fut.result()
            except Exception:
                sizes[url] = None
            if verbose:
                size = sizes[url]
                label = str(size) if size is not None else "unknown"
                print(f"  remote size: {label} bytes | {url}")

    available = sum(1 for size in sizes.values() if size is not None)
    unknown = len(sizes) - available
    return sizes, {
        "existing": total_existing,
        "checked": len(sizes),
        "available": available,
        "unknown": unknown,
        "skipped_wms": skipped_wms,
    }


def _check_local_files(
    downloads_by_source: dict[str, list[tuple[str, str]]],
    downloaders: dict[str, object],
    sample_limit: int,
    remote_sizes: dict[str, int | None] | None = None,
    verbose: bool = False,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for source_key, downloads in downloads_by_source.items():
        region_key, tile_type = source_key.split("_", maxsplit=1)
        downloader = downloaders.get(region_key)
        if downloader is None:
            continue

        expected = len(downloads)
        valid = 0
        invalid = 0
        missing = 0
        mismatch = 0
        invalid_samples: list[str] = []
        missing_samples: list[str] = []
        mismatch_samples: list[str] = []

        for url, path in downloads:
            if not os.path.exists(path):
                missing += 1
                if len(missing_samples) < sample_limit:
                    missing_samples.append(path)
                continue

            local_size = os.path.getsize(path)
            if local_size < MIN_FILE_SIZE:
                invalid += 1
                if len(invalid_samples) < sample_limit:
                    invalid_samples.append(path)
                continue

            try:
                with open(path, "rb") as handle:
                    is_valid = downloader._verify_file_integrity(handle, path)
            except Exception:
                is_valid = False

            if is_valid:
                remote_size = remote_sizes.get(url) if remote_sizes else None
                if remote_size is not None and remote_size != local_size:
                    mismatch += 1
                    if len(mismatch_samples) < sample_limit:
                        mismatch_samples.append(
                            f"{path} (local {local_size}, remote {remote_size})"
                        )
                    if verbose:
                        print(
                            f"  size mismatch: local {local_size} vs remote {remote_size} | {path}"
                        )
                elif remote_size is None and verbose and remote_sizes is not None:
                    print(f"  remote size unknown | {path}")
                else:
                    valid += 1
            else:
                invalid += 1
                if len(invalid_samples) < sample_limit:
                    invalid_samples.append(path)

        results[source_key] = {
            "type": tile_type,
            "expected": expected,
            "valid": valid,
            "invalid": invalid,
            "missing": missing,
            "mismatch": mismatch,
            "invalid_samples": invalid_samples,
            "missing_samples": missing_samples,
            "mismatch_samples": mismatch_samples,
        }
    return results


def main() -> None:
    args = parse_args()

    selected_regions = normalize_regions(args.region)
    imagery_from = (args.from_year, args.to_year) if args.from_year else None
    process_images = args.data_types is None or "image" in args.data_types
    process_pointclouds = args.data_types is None or "dsm" in args.data_types

    coords, source_zone = load_pygeon_coords(args.dataset_path)
    unique_coords, coord_count, unique_count = dedupe_coords(coords, args.grid_size_km)
    if not unique_coords:
        raise ValueError("No coordinates found after deduplication.")

    margin_km = args.margin * args.grid_size_km
    tiles_by_zone = generate_tiles_by_zone(unique_coords, source_zone, args.grid_size_km, margin_km)
    total_user_tiles = len(tiles_by_zone.get(source_zone, set()))

    selected_zones = {UTM_ZONE_BY_REGION[region] for region in selected_regions}
    if source_zone not in selected_zones:
        region_names = ", ".join(region.value for region in selected_regions)
        zone_list = ", ".join(str(zone) for zone in sorted(selected_zones))
        raise ValueError(
            f"Input coordinates are in UTM zone {source_zone}; selected regions "
            f"({region_names}) use zones {zone_list}. Include the matching region."
        )

    print()
    print("Pygeon Tile Report")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output dir: {args.output}")
    print(f"Grid size: {args.grid_size_km:.2f} km | Margin: {args.margin} tiles")
    print(f"Source UTM zone: {source_zone}")
    print(f"Regions: {', '.join(region.value for region in selected_regions)}")
    print()

    print_table(
        "Coordinate Summary",
        ["Metric", "Count"],
        [
            ("Raw coordinates", str(coord_count)),
            ("Unique tiles", str(unique_count)),
            ("User tiles (with margin)", str(total_user_tiles)),
        ],
    )
    print()

    output_path = Path(args.output)
    if not output_path.exists():
        print(f"Warning: output directory does not exist: {output_path}")
        print()

    downloaders: dict[str, object] = {}
    region_catalogs: list[RegionCatalog] = []
    catalog_rows: list[tuple[str, int, int]] = []

    start = time.perf_counter()

    if Region.NRW in selected_regions:
        nrw_downloader = NRWDownloader(args.output, imagery_from=imagery_from)
        nrw_jp2, nrw_laz = nrw_downloader.get_available_tiles()
        catalog_rows.append(("NRW", nrw_downloader.total_jp2_count or len(nrw_jp2), len(nrw_laz)))
        region_catalogs.append(RegionCatalog("nrw", nrw_downloader, nrw_jp2, nrw_laz))
        downloaders["nrw"] = nrw_downloader

    if Region.RLP in selected_regions:
        rlp_downloader = RLPDownloader(args.output, imagery_from=imagery_from)
        rlp_coords = _rlp_native_coords(rlp_downloader, tiles_by_zone, args.grid_size_km)
        rlp_jp2, rlp_laz = rlp_downloader.get_available_tiles(requested_coords=rlp_coords)
        catalog_rows.append(("RLP", rlp_downloader.total_jp2_count or len(rlp_jp2), len(rlp_laz)))
        region_catalogs.append(RegionCatalog("rlp", rlp_downloader, rlp_jp2, rlp_laz))
        downloaders["rlp"] = rlp_downloader

    if Region.BB in selected_regions:
        bb_downloader = BrandenburgDownloader(args.output)
        bb_jp2, bb_laz = bb_downloader.get_available_tiles()
        catalog_rows.append(("BB", len(bb_jp2), len(bb_laz)))
        region_catalogs.append(RegionCatalog("bb", bb_downloader, bb_jp2, bb_laz))
        downloaders["bb"] = bb_downloader

    catalog_duration = time.perf_counter() - start
    if catalog_rows:
        print_catalog_summary(catalog_rows, catalog_duration)

    zone_by_region = {
        region.value.lower(): UTM_ZONE_BY_REGION[region] for region in selected_regions
    }
    coords_array = np.array(unique_coords)
    tile_set, downloads_by_source = calculate_required_tiles(
        tiles_by_zone,
        args.grid_size_km,
        region_catalogs,
        zone_by_region,
        original_coords=coords_array,
        source_zone=source_zone,
    )

    coverage_rows = []
    if process_images:
        covered_jp2 = total_user_tiles - len(tile_set.missing_jp2)
        coverage_jp2_pct = (covered_jp2 / total_user_tiles * 100) if total_user_tiles else 0
        coverage_rows.append(
            (
                "Imagery (JP2)",
                str(total_user_tiles),
                str(covered_jp2),
                str(len(tile_set.missing_jp2)),
                f"{coverage_jp2_pct:.0f}%",
            )
        )
    if process_pointclouds:
        covered_laz = total_user_tiles - len(tile_set.missing_laz)
        coverage_laz_pct = (covered_laz / total_user_tiles * 100) if total_user_tiles else 0
        coverage_rows.append(
            (
                "DSM (LAZ/TIF)",
                str(total_user_tiles),
                str(covered_laz),
                str(len(tile_set.missing_laz)),
                f"{coverage_laz_pct:.0f}%",
            )
        )

    print_table(
        "Tile Coverage",
        ["Type", "User Tiles", "Available", "Missing", "Coverage"],
        coverage_rows,
    )
    print()

    download_rows = []
    for region in selected_regions:
        key = region.value.lower()
        jp2_count = len(downloads_by_source.get(f"{key}_jp2", [])) if process_images else "-"
        laz_count = len(downloads_by_source.get(f"{key}_laz", [])) if process_pointclouds else "-"
        download_rows.append((region.value, str(jp2_count), str(laz_count)))
    print_table(
        "Expected Downloads",
        ["Region", "JP2 Files", "DSM Files"],
        download_rows,
    )
    print()

    if process_images and tile_set.missing_jp2:
        sample = list(tile_set.missing_jp2)[: args.max_sample]
        print(f"Missing imagery tiles: {len(tile_set.missing_jp2)}")
        print(f"  Sample (zone, grid_x, grid_y): {sample}")
        print()
    if process_pointclouds and tile_set.missing_laz:
        sample = list(tile_set.missing_laz)[: args.max_sample]
        print(f"Missing DSM tiles: {len(tile_set.missing_laz)}")
        print(f"  Sample (zone, grid_x, grid_y): {sample}")
        print()

    remote_sizes = None
    if args.check_remote_size:
        remote_sizes, remote_stats = _collect_remote_sizes(
            downloads_by_source,
            downloaders,
            max_workers=args.remote_workers,
            timeout=FEED_TIMEOUT,
            verbose=args.verbose,
        )
        print(
            f"Remote size check: {remote_stats['available']}/{remote_stats['checked']} "
            f"sizes available ({remote_stats['unknown']} missing headers)"
        )
        if remote_stats["skipped_wms"]:
            print(f"  Skipped {remote_stats['skipped_wms']} WMS URLs")
        if remote_stats["existing"] == 0:
            print("  Note: no local files found to compare.")
        print()

    local_results = _check_local_files(
        downloads_by_source,
        downloaders,
        sample_limit=args.max_sample,
        remote_sizes=remote_sizes,
        verbose=args.verbose,
    )
    local_rows = []
    for source_key in sorted(local_results.keys()):
        region_key, tile_type = source_key.split("_", maxsplit=1)
        if tile_type == "jp2" and not process_images:
            continue
        if tile_type == "laz" and not process_pointclouds:
            continue
        result = local_results[source_key]
        label = f"{region_key.upper()} {'Imagery' if tile_type == 'jp2' else 'DSM'}"
        row = [
            label,
            str(result["expected"]),
            str(result["valid"]),
            str(result["invalid"]),
            str(result["missing"]),
        ]
        if args.check_remote_size:
            row.append(str(result["mismatch"]))
        local_rows.append(tuple(row))

    if local_rows:
        headers = ["Source", "Expected", "Valid", "Invalid", "Missing"]
        if args.check_remote_size:
            headers.append("Size Mismatch")
        print_table(
            "Local Cache Check",
            headers,
            local_rows,
        )
        print()

    for source_key in sorted(local_results.keys()):
        region_key, tile_type = source_key.split("_", maxsplit=1)
        if tile_type == "jp2" and not process_images:
            continue
        if tile_type == "laz" and not process_pointclouds:
            continue
        result = local_results[source_key]
        label = f"{region_key.upper()} {'Imagery' if tile_type == 'jp2' else 'DSM'}"
        if result["invalid_samples"]:
            print(f"Invalid files ({label}):")
            for path in result["invalid_samples"]:
                print(f"  {path}")
            print()
        if result["missing_samples"]:
            print(f"Missing files ({label}):")
            for path in result["missing_samples"]:
                print(f"  {path}")
            print()
        if result["mismatch_samples"]:
            print(f"Size mismatches ({label}):")
            for path in result["mismatch_samples"]:
                print(f"  {path}")
            print()


if __name__ == "__main__":
    main()
