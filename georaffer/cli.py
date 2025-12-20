"""Command-line interface for georaffer."""

import argparse
import signal
import sys
import warnings

import numpy as np
from PIL import Image
import utm
from rasterio.warp import transform_bounds

from georaffer import __version__

# Suppress PIL decompression bomb warnings for large aerial orthophotos
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


class QuietArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that shows clean error messages."""

    def error(self, message):
        message = (message or "").strip()
        # Show usage + positional args only (skip options and epilog)
        formatter = self._get_formatter()
        formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups)
        # Only show positional arguments (first group), skip options
        if self._action_groups:
            positional = self._action_groups[0]
            if positional._group_actions:
                formatter.start_section(positional.title)
                formatter.add_arguments(positional._group_actions)
                formatter.end_section()
        # Usage first, then error message, then help hint
        print(formatter.format_help(), file=sys.stderr)
        print(f"error: {message}", file=sys.stderr)
        print("Use --help for more", file=sys.stderr)
        sys.exit(2)


def signal_handler(signum, frame):
    """Handle Ctrl+C by signaling interrupt and raising KeyboardInterrupt."""
    from georaffer.runtime import InterruptManager

    # Just set flag and raise - let the handler print after closing progress bars
    InterruptManager.get().signal()
    raise KeyboardInterrupt()


# Install signal handler for immediate Ctrl+C response
signal.signal(signal.SIGINT, signal_handler)


def load_coordinates(args, *, return_zone: bool = False):
    """Load coordinates based on CLI arguments.

    If return_zone is True, also return the detected/source UTM zone.
    """
    from georaffer.config import METERS_PER_KM, OUTPUT_TILE_SIZE_KM, UTM_ZONE
    from georaffer.inputs import load_from_bbox, load_from_csv, load_from_geotiff

    # Use output tile size consistently for all coordinate loading
    tile_size_m = int(OUTPUT_TILE_SIZE_KM * METERS_PER_KM)

    coords = []
    utm_zone = getattr(args, "utm_zone", None)
    source_zone = utm_zone or UTM_ZONE

    def _require_utm_zone() -> int:
        if utm_zone is None:
            raise ValueError("UTM inputs require --utm-zone (32 or 33).")
        return utm_zone

    def _latlon_bbox_to_utm(
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        *,
        context: str,
    ) -> tuple[float, float, float, float, int]:
        min_x, min_y, min_zone, _ = utm.from_latlon(min_lat, min_lon)
        max_x, max_y, max_zone, _ = utm.from_latlon(max_lat, max_lon)
        if utm_zone is not None:
            raise ValueError(f"{context} lat/lon inputs do not accept --utm-zone.")
        if min_zone != max_zone:
            raise ValueError(f"{context} spans multiple UTM zones; split input by zone.")
        return min_x, min_y, max_x, max_y, min_zone

    if args.command == "csv":
        # Validate --cols format: must be exactly two comma-separated names
        parts = [p.strip() for p in args.cols.split(",")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"--cols must be two comma-separated column names (e.g. 'lon,lat'), got: '{args.cols}'"
            )
        x_col, y_col = parts
        raw_coords = load_from_csv(args.file, x_col, y_col)
        # Auto-detect lat/lon vs UTM by value ranges (consistent with bbox)
        sample = raw_coords[:100] if len(raw_coords) > 100 else raw_coords
        is_latlon = all(abs(c[0]) < 180 and abs(c[1]) < 90 for c in sample)
        if is_latlon:
            if utm_zone is not None:
                raise ValueError("CSV lat/lon inputs do not accept --utm-zone.")
            coords = []
            zones: set[int] = set()
            for lon, lat in raw_coords:
                easting, northing, zone, _ = utm.from_latlon(lat, lon)
                coords.append((easting, northing))
                zones.add(zone)
            if len(zones) > 1:
                raise ValueError("CSV coordinates span multiple UTM zones; split input by zone.")
            if zones:
                source_zone = zones.pop()
        else:
            source_zone = _require_utm_zone()
            coords = [(c[0], c[1]) for c in raw_coords]

    elif args.command == "bbox":
        # bbox format: WEST,SOUTH,EAST,NORTH (min_x, min_y, max_x, max_y)
        min_x, min_y, max_x, max_y = map(float, args.bbox.split(","))

        # Auto-detect lat/lon vs UTM based on value ranges
        # Lat/lon: small values (lat -90 to 90, lon -180 to 180)
        # UTM: large values (easting 280k-920k, northing 5.2M-6.1M for Germany)
        is_latlon = abs(min_x) < 180 and abs(max_x) < 180 and abs(min_y) < 90 and abs(max_y) < 90

        if is_latlon:
            # Convert lat/lon bbox corners to UTM
            # min_x=west_lon, min_y=south_lat, max_x=east_lon, max_y=north_lat
            min_x, min_y, max_x, max_y, source_zone = _latlon_bbox_to_utm(
                min_x, min_y, max_x, max_y, context="BBox"
            )
        else:
            source_zone = _require_utm_zone()

        # load_from_bbox returns UTM tile centers directly
        coords = load_from_bbox(min_x, min_y, max_x, max_y, tile_size_m)

    elif args.command == "tif":
        bounds, crs = load_from_geotiff(args.tif)
        min_x, min_y, max_x, max_y = bounds

        epsg = crs.to_epsg()
        if crs.is_geographic:
            min_x, min_y, max_x, max_y, source_zone = _latlon_bbox_to_utm(
                min_x, min_y, max_x, max_y, context="GeoTIFF"
            )
        elif epsg in (25832, 25833, 32632, 32633):
            detected_zone = epsg % 100  # extract utm zone from epsg code
            if utm_zone is not None and utm_zone != detected_zone:
                raise ValueError(
                    f"GeoTIFF is in UTM zone {detected_zone}; "
                    f"pass --utm-zone {detected_zone} or omit --utm-zone."
                )
            source_zone = detected_zone
        else:
            min_lon, min_lat, max_lon, max_lat = transform_bounds(
                crs, "EPSG:4326", min_x, min_y, max_x, max_y, densify_pts=21
            )
            min_x, min_y, max_x, max_y, source_zone = _latlon_bbox_to_utm(
                min_lon, min_lat, max_lon, max_lat, context="GeoTIFF"
            )

        if source_zone not in (32, 33):
            raise ValueError(
                f"GeoTIFF resolves to UTM zone {source_zone}; only zones 32 and 33 are supported."
            )

        # load_from_bbox returns UTM tile centers directly
        coords = load_from_bbox(min_x, min_y, max_x, max_y, tile_size_m)

    elif args.command == "pygeon":
        from georaffer.grids import latlon_array_to_utm
        from georaffer.inputs import load_from_pygeon

        raw_coords = load_from_pygeon(args.dataset_path)
        # pygeon returns (lat, lon, alt) - vectorized UTM conversion
        coords_array = np.array(raw_coords)
        if coords_array.size == 0:
            coords = []
        else:
            if utm_zone is not None:
                raise ValueError("Pygeon inputs do not accept --utm-zone.")
            lons = coords_array[:, 1]
            zone_candidates = np.floor((lons + 180) / 6).astype(int) + 1
            unique_zones = set(zone_candidates.tolist())
            if len(unique_zones) > 1:
                raise ValueError("Pygeon dataset spans multiple UTM zones; split input by zone.")
            source_zone = unique_zones.pop()
            utm_x, utm_y = latlon_array_to_utm(
                coords_array[:, 0], coords_array[:, 1], force_zone_number=source_zone
            )
            coords = list(zip(utm_x, utm_y))

    elif args.command == "tiles":
        from georaffer.grids import tile_to_utm_center

        source_zone = _require_utm_zone()
        # Parse tile coordinates: "350,5600 351,5601" (km indices)
        for tile_str in args.tiles:
            x, y = map(int, tile_str.split(","))
            coords.append(tile_to_utm_center(x, y, tile_size_m))

    if return_zone:
        return coords, source_zone
    return coords


def pixel_size_to_resolution(pixel_size: float, tile_size_km: float) -> int:
    """Convert pixel size (meters) to resolution (pixels per tile side)."""
    from georaffer.config import METERS_PER_KM

    tile_size_m = tile_size_km * METERS_PER_KM
    return int(tile_size_m / pixel_size)


def normalize_regions(region_args: list[str]) -> list["Region"]:
    """Normalize region CLI args to Region enums, preserving order."""
    from georaffer.config import Region

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


def validate_args(args) -> list[str]:
    """Validate parsed arguments, return list of errors."""
    from georaffer.config import METERS_PER_KM, OUTPUT_TILE_SIZE_KM

    errors = []
    tile_size_m = OUTPUT_TILE_SIZE_KM * METERS_PER_KM

    # Validate pixel-size: must be positive and not exceed tile size
    if args.pixel_size:
        for ps in args.pixel_size:
            if ps <= 0:
                errors.append(f"--pixel-size: {ps} must be positive")
            elif ps > tile_size_m:
                errors.append(f"--pixel-size: {ps}m exceeds tile size ({tile_size_m}m)")

    # Validate --from/--to: FROM <= TO, reasonable range
    if args.from_year:
        if args.from_year < 2000:
            errors.append(f"--from: {args.from_year} is before 2000 (no data available)")
        if args.to_year is not None and args.to_year < args.from_year:
            errors.append(f"--to: {args.to_year} must be >= --from year ({args.from_year})")
    if args.to_year and not args.from_year:
        errors.append("--to: requires --from")

    # Validate margin: must be non-negative
    if args.margin < 0:
        errors.append(f"--margin: {args.margin} must be non-negative")

    # Validate workers: must be positive (if specified)
    if args.workers is not None and args.workers <= 0:
        errors.append(f"--workers: {args.workers} must be positive")

    # Command-specific validation
    if args.command == "bbox":
        parts = args.bbox.split(",")
        if len(parts) != 4:
            errors.append(
                f"bbox: expected 4 comma-separated values (WEST,SOUTH,EAST,NORTH), got {len(parts)}"
            )
        else:
            try:
                min_x, min_y, max_x, max_y = map(float, parts)
                if min_x > max_x:
                    errors.append(f"bbox: WEST ({min_x}) must be <= EAST ({max_x})")
                if min_y > max_y:
                    errors.append(f"bbox: SOUTH ({min_y}) must be <= NORTH ({max_y})")
            except ValueError:
                errors.append("bbox: all values must be valid numbers")

    if args.command == "tiles":
        for i, tile_str in enumerate(args.tiles):
            parts = tile_str.split(",")
            if len(parts) != 2:
                errors.append(f"tiles[{i}]: expected 'x,y' format, got '{tile_str}'")
            else:
                try:
                    int(parts[0])
                    int(parts[1])
                except ValueError:
                    errors.append(f"tiles[{i}]: '{tile_str}' must be two integers")

    return errors


def main():
    """Main CLI entry point."""
    # Shared epilog for subcommands explaining common options
    shared_epilog = """\
Details:

  --margin: Add buffer tiles around each input tile.

            --margin 0         --margin 1

            ┌───┐              ┌───┬───┬───┐
            │ x │              │ □ │ □ │ □ │
            └───┼───┐          ├───┼───┼───┼───┐
                │ x │          │ □ │ x │ □ │ □ │
                └───┘          ├───┼───┼───┼───┤
                               │ □ │ □ │ x │ □ │
            2 tiles            └───┼───┼───┼───┤
                                   │ □ │ □ │ □ │
                                   └───┴───┴───┘
                               14 tiles

              x = requested tile
              □ = margin

  --pixel-size: Output resolution in meters. Multiple values create multiple outputs.
                Examples: --pixel-size 0.5 (50cm), --pixel-size 1 2 5 (three resolutions)

  --type:       Download only image or dsm (default: both).
                DSM: Photogrammetric surface model (bDOM) from aerial imagery.

  --from/--to:  Include historic orthophotos (NRW only, availability varies).
                Examples: --from 2015 (2015 to present), --from 2015 --to 2018
"""

    # bbox-specific epilog
    bbox_epilog = f"""\
{shared_epilog}"""

    # tif-specific epilog
    tif_epilog = f"""\
{shared_epilog}"""

    # tiles-specific epilog
    tiles_epilog = f"""\
{shared_epilog}"""

    # csv-specific epilog
    csv_epilog = f"""\
{shared_epilog}"""

    # pygeon-specific epilog
    pygeon_epilog = f"""\
{shared_epilog}"""

    parser = argparse.ArgumentParser(
        prog="georaffer",
        description="Download and convert German state geodata (orthophotos + bDOM) to GeoTIFF",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # Shared options as parent parser
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--output", required=True, metavar="DIR", help="Output directory")
    shared.add_argument(
        "--margin",
        type=int,
        default=0,
        metavar="INT",
        help="Tile buffer around input: 0=exact, 1=+1 ring (default: 0)",
    )
    shared.add_argument(
        "--pixel-size",
        type=float,
        nargs="+",
        default=[0.5],
        metavar="FLOAT",
        help="Output resolution in meters per pixel (default: 0.5)",
    )
    shared.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="INT",
        help="Parallel conversion workers (default: 4)",
    )
    shared.add_argument(
        "--reprocess",
        action="store_true",
        help="Re-download and re-convert existing files (by default, skips existing outputs)",
    )
    shared.add_argument("--profiling", action="store_true", help="Enable conversion timing output")
    shared.add_argument(
        "--from",
        type=int,
        metavar="YEAR",
        dest="from_year",
        help="Include historic orthophotos from YEAR (NRW only, availability varies)",
    )
    shared.add_argument(
        "--to",
        type=int,
        metavar="YEAR",
        dest="to_year",
        help="End year for historic orthophotos (default: present)",
    )
    shared.add_argument(
        "--type",
        dest="data_types",
        nargs="+",
        metavar="TYPE",
        choices=["image", "dsm"],
        help="Download only specific data types (default: all). "
        "Options: image (orthophotos), dsm (surface elevation)",
    )
    shared.add_argument(
        "--only",
        dest="data_types",
        nargs="+",
        metavar="TYPE",
        choices=["image", "dsm"],
        help=argparse.SUPPRESS,
    )
    shared.add_argument(
        "--utm-zone",
        type=int,
        choices=[32, 33],
        metavar="ZONE",
        help="UTM zone for UTM inputs (required for UTM inputs; invalid for lat/lon)",
    )
    shared.add_argument(
        "--region",
        nargs="+",
        choices=["nrw", "rlp", "bb"],
        default=["nrw", "rlp"],
        metavar="REGION",
        help="Regions to include: nrw rlp bb (default: nrw rlp)",
    )

    # bbox subcommand
    subparsers._parser_class = QuietArgumentParser
    bbox_parser = subparsers.add_parser(
        "bbox",
        parents=[shared],
        help="Download tiles covering a bounding box",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="georaffer bbox XMIN,YMIN,XMAX,YMAX --output DIR [options]\n\n   ex: georaffer bbox 6.9,50.9,7.1,51.1 --output ./tiles\n                      └─ lon/lat\n       georaffer bbox 350000,5600000,360000,5610000 --output ./tiles\n                      └─ UTM meters",
        epilog=bbox_epilog,
    )
    bbox_parser.add_argument(
        "bbox", metavar="BBOX", help="Bounding box: XMIN,YMIN,XMAX,YMAX (UTM meters or lon/lat)"
    )

    # tif subcommand
    tif_parser = subparsers.add_parser(
        "tif",
        parents=[shared],
        help="Download tiles covering a GeoTIFF footprint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="georaffer tif PATH --output DIR [options]\n\n   ex: georaffer tif ./area.tif --output ./tiles",
        epilog=tif_epilog,
    )
    tif_parser.add_argument(
        "tif", metavar="PATH", help="GeoTIFF path used to derive bounds (CRS required)"
    )

    # tiles subcommand
    tiles_parser = subparsers.add_parser(
        "tiles",
        parents=[shared],
        help="Download specific tiles by grid coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="georaffer tiles TILE... --output DIR [options]\n\n   ex: georaffer tiles 350,5600 --output ./tiles\n                       └─ X,Y grid index (350km E, 5600km N)\n       georaffer tiles 350,5600 351,5601 --output ./tiles\n                       └─ multiple tiles",
        epilog=tiles_epilog,
    )
    tiles_parser.add_argument(
        "tiles",
        nargs="+",
        metavar="TILE",
        help="Tile grid indices as X,Y (e.g., 350,5600 = 350km E, 5600km N)",
    )

    # csv subcommand
    csv_parser = subparsers.add_parser(
        "csv",
        parents=[shared],
        help="Download tiles from CSV coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="georaffer csv FILE --cols X,Y --output DIR [options]\n\n   ex: georaffer csv coords.csv --cols lon,lat --output ./tiles\n       georaffer csv coords.csv --cols easting,northing --output ./tiles\n\n       X,Y = column names from your CSV (auto-detects lat/lon vs UTM)",
        epilog=csv_epilog,
    )
    csv_parser.add_argument("file", metavar="FILE", help="CSV file with coordinates")
    csv_parser.add_argument(
        "--cols",
        required=True,
        metavar="X,Y",
        help="Column names from CSV (e.g. lon,lat or easting,northing). Auto-detects coordinate type.",
    )

    # pygeon subcommand
    pygeon_parser = subparsers.add_parser(
        "pygeon",
        parents=[shared],
        help="Download tiles for 4Seasons dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="georaffer pygeon PATH --output DIR [options]\n\n   ex: georaffer pygeon /data/4seasons/campaign --output ./tiles   # has ins.csv\n       georaffer pygeon /data/4seasons --output ./tiles            # has */ins.csv",
        epilog=pygeon_epilog,
    )
    pygeon_parser.add_argument(
        "dataset_path", metavar="PATH", help="Folder with ins.csv or parent of campaign subfolders"
    )

    # Don't let argparse auto-error on missing subcommand - we handle it manually
    subparsers.required = False
    args = parser.parse_args()

    # Show help + error if no command given
    if args.command is None:
        parser.print_help()
        sys.stdout.flush()
        print("\nerror: the following arguments are required: <command>", file=sys.stderr)
        sys.exit(2)

    # Validate arguments
    errors = validate_args(args)
    if errors:
        print("Validation errors:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    Image.MAX_IMAGE_PIXELS = None

    try:
        # Load coordinates
        print(f"Loading coordinates from {args.command}...", flush=True)
        coords, source_zone = load_coordinates(args, return_zone=True)

        if not coords:
            print("Error: No coordinates found", file=sys.stderr)
            sys.exit(1)

        print(f"Loaded {len(coords)} coordinates")

        # Deduplicate at output tile level
        from georaffer.config import METERS_PER_KM, OUTPUT_TILE_SIZE_KM
        from georaffer.pipeline import process_tiles

        coords_array = np.array(coords)
        grid_size_m = OUTPUT_TILE_SIZE_KM * METERS_PER_KM
        tile_x = (coords_array[:, 0] // grid_size_m).astype(int)
        tile_y = (coords_array[:, 1] // grid_size_m).astype(int)
        tiles = np.column_stack([tile_x, tile_y])
        _, unique_indices = np.unique(tiles, axis=0, return_index=True)

        unique_coords = coords_array[unique_indices].tolist()
        print(f"Deduplicated {len(coords)} coordinates to {len(unique_coords)} center tiles")

        # Build imagery_from tuple from --from/--to flags
        imagery_from = None
        if args.from_year:
            imagery_from = (args.from_year, args.to_year)

        # Convert pixel sizes to resolutions (internal format)
        resolutions = [pixel_size_to_resolution(ps, OUTPUT_TILE_SIZE_KM) for ps in args.pixel_size]
        pixel_info = ", ".join(f"{ps}m ({r}px)" for ps, r in zip(args.pixel_size, resolutions))
        print(f"Output resolution: {pixel_info} per {OUTPUT_TILE_SIZE_KM}km tile")

        # Convert margin (tile count) to km for internal API
        margin_km = args.margin * OUTPUT_TILE_SIZE_KM

        # Use config defaults if not specified on CLI
        from georaffer.config import DEFAULT_WORKERS

        workers = args.workers if args.workers is not None else DEFAULT_WORKERS

        # Parse --type into booleans for pipeline
        process_images = args.data_types is None or "image" in args.data_types
        process_pointclouds = args.data_types is None or "dsm" in args.data_types
        regions = normalize_regions(args.region)

        stats = process_tiles(
            coords=unique_coords,
            output_dir=args.output,
            resolutions=resolutions,
            grid_size_km=OUTPUT_TILE_SIZE_KM,
            margin_km=margin_km,
            imagery_from=imagery_from,
            force_download=args.reprocess,
            max_workers=workers,
            profiling=args.profiling,
            process_images=process_images,
            process_pointclouds=process_pointclouds,
            reprocess=args.reprocess,
            source_zone=source_zone,
            regions=regions,
        )

        if args.command == "tif":
            from georaffer.align import align_to_reference

            print("Aligning outputs to reference GeoTIFF...")
            aligned = align_to_reference(
                reference_path=args.tif,
                output_dir=args.output,
                align_images=process_images,
                align_dsm=process_pointclouds,
            )
            for key, path in aligned.items():
                print(f"  {key}: {path}")

        # Exit with error if there were failures
        if stats.failed_download > 0 or stats.failed_convert > 0:
            print("⚠ Pipeline completed with errors - see messages above", file=sys.stderr)
            sys.exit(1)

    except KeyboardInterrupt:
        # Signal handler already printed the interrupt message
        sys.exit(130)
    except (ValueError, FileNotFoundError) as e:
        # User input errors - exit cleanly without traceback
        print(f"\nerror: {e}", file=sys.stderr)
        print("Run with --help for usage information", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Unexpected errors - show full context for debugging
        print()
        print("=" * 80, file=sys.stderr)
        print("ERROR", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(file=sys.stderr)
        print(f"Pipeline failed: {e}", file=sys.stderr)
        if e.__cause__:
            print(f"Root cause: {e.__cause__}", file=sys.stderr)
        print(file=sys.stderr)
        print("Run with --help for usage information", file=sys.stderr)
        print(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
