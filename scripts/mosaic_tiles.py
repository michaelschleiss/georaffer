"""Build a quick mosaic from GeoTIFF tiles that share a CRS/resolution.

Usage:
    python scripts/mosaic_tiles.py --input-dir /path/to/processed/image/5000 \
        --output /tmp/mosaic_5000.tif

The script reads all GeoTIFFs in the input directory, uses their geotransforms
to place each tile at the correct position, and writes a single GeoTIFF mosaic.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from contextlib import ExitStack

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.plot import reshape_as_image

try:
    from PIL import Image, ImageColor, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional dependency
    Image = None
    ImageFont = None

import csv

# Allow running as standalone script without installing the package
if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from georaffer.converters.utils import parse_tile_coords


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mosaic GeoTIFF tiles into one image")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing GeoTIFF tiles (e.g., processed/image/5000)",
    )
    parser.add_argument("--output", required=True, help="Path to the output mosaic GeoTIFF")
    parser.add_argument(
        "--pattern",
        default="*.tif",
        help="Glob pattern to match tiles inside input-dir (default: *.tif)",
    )
    parser.add_argument(
        "--num-threads",
        default="ALL_CPUS",
        help="GDAL/RasterIO thread hint (default: ALL_CPUS)",
    )
    parser.add_argument(
        "--compression",
        default="LZW",
        help="Output compression (default: LZW)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display a quick preview after writing the mosaic (opens window if available)",
    )
    parser.add_argument(
        "--annotate-year",
        action="store_true",
        help="Overlay year labels from provenance.csv (no color coding; outlines + text)",
    )
    parser.add_argument(
        "--annotate-season",
        action="store_true",
        help="Overlay season fills (Winter/Spring/Summer/Autumn) derived from acquisition_date in provenance.csv",
    )
    parser.add_argument(
        "--provenance",
        default=None,
        help="Path to provenance.csv (defaults to <input-dir>/../provenance.csv)",
    )
    return parser.parse_args()


def find_tiles(input_dir: str, pattern: str) -> list[str]:
    tiles = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not tiles:
        raise SystemExit(f"No tiles found in {input_dir} matching '{pattern}'")
    return tiles


def _select_indexes_and_dtype(datasets: list[rasterio.io.DatasetReader]) -> tuple[list[int], str]:
    """Choose common band indexes and output dtype for a set of open datasets.

    Uses already-open datasets (so callers control lifetimes) to avoid double
    opens. Assumes at least one dataset is provided.
    """

    if not datasets:
        raise ValueError("No tiles found to mosaic.")

    counts = [ds.count for ds in datasets]
    min_count = min(counts)
    if min_count == 0:
        raise ValueError("Found dataset with zero bands; cannot mosaic.")

    if len(set(counts)) > 1:
        print(
            f"[mosaic] Warning: mixed band counts {sorted(set(counts))}; "
            f"using first {min_count} band(s) everywhere.",
            file=sys.stderr,
        )

    indexes = list(range(1, min_count + 1))
    dtype = datasets[0].dtypes[0]
    return indexes, dtype


def _show_mosaic(array):
    if Image is None:
        print("[mosaic] PIL not installed; skipping preview", file=sys.stderr)
        return

    # Expect array shape (bands, rows, cols)
    img = array
    if img.shape[0] >= 3:
        img = img[:3]  # RGB
        mode = "RGB"
    else:
        mode = "L"

    hwc = reshape_as_image(img)

    try:
        img_pil = Image.fromarray(hwc.astype("uint8"), mode=mode)
        img_pil.show()
    except Exception as exc:
        print(f"[mosaic] PIL preview failed ({exc})", file=sys.stderr)


def _load_provenance(path: str):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _parse_coords(row):
    """Return grid coords, trying provenance fields then filenames as fallback."""
    try:
        gx = int(row.get("grid_x")) if row.get("grid_x") not in (None, "", "None") else None
        gy = int(row.get("grid_y")) if row.get("grid_y") not in (None, "", "None") else None
        if gx is not None and gy is not None:
            return gx, gy
        # Fallback: parse from processed or source filename
        for key in ("processed_file", "source_file"):
            name = row.get(key)
            if name:
                parsed = parse_tile_coords(os.path.basename(name))
                if parsed:
                    return parsed
        return None, None
    except Exception:
        return None, None


def _annotate_year(mosaic_array, transform, provenance_rows, grid_size_m=1000):
    """Draw year overlays using PIL; mosaic_array shape (bands, rows, cols)."""
    if Image is None:
        print("[mosaic] PIL not installed; skipping year annotation", file=sys.stderr)
        return None

    img_hwc = (
        reshape_as_image(mosaic_array[:3])
        if mosaic_array.shape[0] >= 3
        else reshape_as_image(mosaic_array)
    )
    base = Image.fromarray(
        img_hwc.astype("uint8"), mode="RGB" if mosaic_array.shape[0] >= 3 else "L"
    ).convert("RGBA")
    draw = ImageDraw.Draw(base, "RGBA")
    # Choose a large, readable font for labels/legend
    font = None
    if ImageFont:
        for size in (28, 24, 20):
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", size=size)
                break
            except Exception:
                continue
        if font is None:
            try:
                default = ImageFont.load_default()
                # upscale default bitmap font by rendering via truetype fallback if path exists
                if hasattr(default, "path"):
                    font = ImageFont.truetype(default.path, size=60)
                else:
                    font = default
            except Exception:
                font = None

    # Build palette per year (cycles if many distinct years)
    palette = [
        "#e4572e",
        "#17bebb",
        "#ffc914",
        "#4e79a7",
        "#59a14f",
        "#edc948",
        "#b07aa1",
        "#ff9da7",
        "#9c755f",
        "#76b7b2",
    ]
    seen_years = []
    tile_years = {}  # (gx, gy) -> set(years)
    for row in provenance_rows:
        yr = row.get("year") or row.get("acquisition_date") or "unknown"
        gx, gy = _parse_coords(row)
        if gx is None or gy is None:
            continue
        tile_years.setdefault((gx, gy), set()).add(yr)
        if yr not in seen_years:
            seen_years.append(yr)
    year_colors = {}
    for idx, yr in enumerate(sorted(seen_years)):
        year_colors[yr] = palette[idx % len(palette)]

    def hex_to_rgba(hex_str, alpha):
        rgb = ImageColor.getrgb(hex_str)
        return (rgb[0], rgb[1], rgb[2], alpha)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ovr = ImageDraw.Draw(overlay, "RGBA")

    skipped = 0
    drawn = 0
    mixed = 0

    def xy_to_rc(x, y):
        col = (x - transform.c) / transform.a
        row = (transform.f - y) / -transform.e
        return round(col), round(row)

    for (gx, gy), years in tile_years.items():
        if gx is None or gy is None:
            skipped += 1
            continue

        # Compute pixel bbox from grid coords and transform
        x0 = gx * grid_size_m
        y0 = (gy + 1) * grid_size_m
        x1 = (gx + 1) * grid_size_m
        y1 = gy * grid_size_m

        c0, r0 = xy_to_rc(x0, y0)
        c1, r1 = xy_to_rc(x1, y1)
        num_rows, num_cols = mosaic_array.shape[1], mosaic_array.shape[2]
        c0 = max(0, min(num_cols - 1, c0))
        c1 = max(0, min(num_cols - 1, c1))
        r0 = max(0, min(num_rows - 1, r0))
        r1 = max(0, min(num_rows - 1, r1))

        # Ensure coordinates are ordered for PIL (y1 >= y0, x1 >= x0)
        left, right = sorted((c0, c1))
        top, bottom = sorted((r0, r1))

        rect = [left, top, right, bottom]
        years_sorted = sorted(years)
        colors = [year_colors.get(y, "#ffffff") for y in years_sorted]

        if len(colors) == 1:
            fill = hex_to_rgba(colors[0], 70)
            ovr.rectangle(rect, fill=fill, outline=hex_to_rgba("#ffffff", 200), width=2)
        else:
            mixed += 1
            stripe_w = max(6, (right - left + 1) // 15)
            for idx, x in enumerate(range(left, right + 1, stripe_w)):
                col = colors[idx % len(colors)]
                stripe_left = x
                stripe_right = min(x + stripe_w - 1, right)
                ovr.rectangle([stripe_left, top, stripe_right, bottom], fill=hex_to_rgba(col, 110))
            ovr.rectangle(rect, outline=hex_to_rgba("#ffffff", 230), width=2)

        drawn += 1

    if skipped:
        print(
            f"[mosaic] skipped {skipped} provenance rows without grid coords for annotation",
            file=sys.stderr,
        )
    print(f"[mosaic] drew {drawn} tile overlays" + (" (none drawn)" if drawn == 0 else ""))

    # Legend (bottom-left), enlarged for readability
    if year_colors:
        legend_items = sorted(year_colors.items(), key=lambda kv: kv[0])
        scale = 2.5
        pad = int(14 * scale)
        sw = int(18 * scale)
        row_h = int(30 * scale)
        text_pad = int(10 * scale)
        max_text_w = 0
        for y, _ in legend_items:
            if font and hasattr(draw, "textbbox"):
                bbox = draw.textbbox((0, 0), str(y), font=font)
                tw = bbox[2] - bbox[0]
                bbox[3] - bbox[1]
            else:
                tw = len(str(y)) * 10  # rough fallback
            max_text_w = max(max_text_w, tw)
        if font and hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), "Ag", font=font)
            row_h = max(row_h, (bbox[3] - bbox[1]) + 8)
        legend_rows = list(legend_items)
        mixed_entry = None
        if mixed:
            mixed_entry = ("mixed (multi-year)", "#666666")
            legend_rows.append(mixed_entry)

        legend_w = pad * 2 + sw + text_pad + max_text_w
        legend_h = pad * 2 + row_h * len(legend_rows)
        _img_w, img_h = base.size
        x0, y0 = 10, img_h - legend_h - 10
        x1, y1 = x0 + legend_w, y0 + legend_h
        draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 200), outline=(255, 255, 255, 220))
        for i, (year_val, col_hex) in enumerate(legend_rows):
            y_top = y0 + pad + i * row_h
            # For mixed entry, draw stripes in the swatch
            if mixed_entry and year_val == mixed_entry[0]:
                stripe_w = max(4, sw // 3)
                for idx, x in enumerate(range(x0 + pad, x0 + pad + sw, stripe_w)):
                    stripe_col = palette[idx % len(palette)]
                    draw.rectangle(
                        [x, y_top, min(x + stripe_w - 1, x0 + pad + sw), y_top + sw],
                        fill=stripe_col,
                        outline=None,
                    )
                draw.rectangle(
                    [x0 + pad, y_top, x0 + pad + sw, y_top + sw],
                    outline=(255, 255, 255, 180),
                    width=1,
                )
            else:
                draw.rectangle(
                    [x0 + pad, y_top, x0 + pad + sw, y_top + sw],
                    fill=col_hex,
                    outline=(255, 255, 255, 180),
                )
            draw.text(
                (x0 + pad + sw + text_pad, y_top),
                str(year_val),
                fill="#ffffff",
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0, 200),
            )

    # Composite overlay with fills/stripes onto base
    base = Image.alpha_composite(base, overlay)
    return base.convert("RGB")


def _season_from_date(date_str: str) -> str:
    """Return season name from YYYY-MM-DD (Northern Hemisphere)."""
    try:
        month = int(date_str.split("-")[1])
    except Exception:
        return "unknown"
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    if month in (9, 10, 11):
        return "Autumn"
    return "unknown"


def _annotate_season(mosaic_array, transform, provenance_rows, grid_size_m=1000):
    """Draw season overlays using PIL; mosaic_array shape (bands, rows, cols)."""
    if Image is None:
        print("[mosaic] PIL not installed; skipping season annotation", file=sys.stderr)
        return None

    img_hwc = (
        reshape_as_image(mosaic_array[:3])
        if mosaic_array.shape[0] >= 3
        else reshape_as_image(mosaic_array)
    )
    base = Image.fromarray(
        img_hwc.astype("uint8"), mode="RGB" if mosaic_array.shape[0] >= 3 else "L"
    ).convert("RGBA")
    draw = ImageDraw.Draw(base, "RGBA")
    font = None
    if ImageFont:
        for size in (28, 24, 20):
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", size=size)
                break
            except Exception:
                continue
        if font is None:
            try:
                default = ImageFont.load_default()
                font = default
            except Exception:
                font = None

    palette = {
        "Winter": "#64b5f6",
        "Spring": "#81c784",
        "Summer": "#ffd54f",
        "Autumn": "#ff8a65",
        "unknown": "#b0bec5",
    }

    tile_seasons = {}
    seen_seasons = []
    for row in provenance_rows:
        season = _season_from_date(row.get("acquisition_date") or "")
        gx, gy = _parse_coords(row)
        if gx is None or gy is None:
            continue
        tile_seasons.setdefault((gx, gy), set()).add(season)
        if season not in seen_seasons:
            seen_seasons.append(season)

    def hex_to_rgba(hex_str, alpha):
        rgb = ImageColor.getrgb(hex_str)
        return (rgb[0], rgb[1], rgb[2], alpha)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ovr = ImageDraw.Draw(overlay, "RGBA")

    skipped = 0
    drawn = 0
    mixed = 0

    def xy_to_rc(x, y):
        col = (x - transform.c) / transform.a
        row = (transform.f - y) / -transform.e
        return round(col), round(row)

    for (gx, gy), seasons in tile_seasons.items():
        if gx is None or gy is None:
            skipped += 1
            continue

        x0 = gx * grid_size_m
        y0 = (gy + 1) * grid_size_m
        x1 = (gx + 1) * grid_size_m
        y1 = gy * grid_size_m

        c0, r0 = xy_to_rc(x0, y0)
        c1, r1 = xy_to_rc(x1, y1)
        num_rows, num_cols = mosaic_array.shape[1], mosaic_array.shape[2]
        c0 = max(0, min(num_cols - 1, c0))
        c1 = max(0, min(num_cols - 1, c1))
        r0 = max(0, min(num_rows - 1, r0))
        r1 = max(0, min(num_rows - 1, r1))

        left, right = sorted((c0, c1))
        top, bottom = sorted((r0, r1))

        rect = [left, top, right, bottom]
        seasons_sorted = sorted(seasons)
        colors = [palette.get(s, "#b0bec5") for s in seasons_sorted]

        if len(colors) == 1:
            fill = hex_to_rgba(colors[0], 70)
            ovr.rectangle(rect, fill=fill, outline=hex_to_rgba("#ffffff", 200), width=2)
        else:
            mixed += 1
            stripe_w = max(6, (right - left + 1) // 12)
            for idx, x in enumerate(range(left, right + 1, stripe_w)):
                col = colors[idx % len(colors)]
                stripe_left = x
                stripe_right = min(x + stripe_w - 1, right)
                ovr.rectangle([stripe_left, top, stripe_right, bottom], fill=hex_to_rgba(col, 110))
            ovr.rectangle(rect, outline=hex_to_rgba("#ffffff", 230), width=2)

        drawn += 1

    if skipped:
        print(
            f"[mosaic] skipped {skipped} provenance rows without grid coords for season annotation",
            file=sys.stderr,
        )
    print(f"[mosaic] drew {drawn} tile overlays" + (" (none drawn)" if drawn == 0 else ""))

    # Legend
    if palette:
        legend_items = [(name, hexval) for name, hexval in palette.items() if name in seen_seasons]
        if not legend_items:
            legend_items = list(palette.items())
        scale = 2.0
        pad = int(14 * scale)
        sw = int(18 * scale)
        row_h = int(30 * scale)
        text_pad = int(10 * scale)
        max_text_w = 0
        for y, _ in legend_items:
            if font and hasattr(draw, "textbbox"):
                bbox = draw.textbbox((0, 0), str(y), font=font)
                tw = bbox[2] - bbox[0]
            else:
                tw = len(str(y)) * 8
            max_text_w = max(max_text_w, tw)
        legend_rows = list(legend_items)
        if mixed:
            legend_rows.append(("mixed (multi-season)", "#666666"))
        legend_w = pad * 2 + sw + text_pad + max_text_w
        legend_h = pad * 2 + row_h * len(legend_rows)
        _img_w, img_h = base.size
        x0, y0 = 10, img_h - legend_h - 10
        x1, y1 = x0 + legend_w, y0 + legend_h
        draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 200), outline=(255, 255, 255, 220))
        for i, (label, col_hex) in enumerate(legend_rows):
            y_top = y0 + pad + i * row_h
            if "mixed" in label:
                stripe_w = max(4, sw // 3)
                for idx, x in enumerate(range(x0 + pad, x0 + pad + sw, stripe_w)):
                    stripe_col = list(palette.values())[idx % len(palette)]
                    draw.rectangle(
                        [x, y_top, min(x + stripe_w - 1, x0 + pad + sw), y_top + sw],
                        fill=stripe_col,
                        outline=None,
                    )
                draw.rectangle(
                    [x0 + pad, y_top, x0 + pad + sw, y_top + sw],
                    outline=(255, 255, 255, 180),
                    width=1,
                )
            else:
                draw.rectangle(
                    [x0 + pad, y_top, x0 + pad + sw, y_top + sw],
                    fill=col_hex,
                    outline=(255, 255, 255, 180),
                )
            draw.text(
                (x0 + pad + sw + text_pad, y_top),
                str(label),
                fill="#ffffff",
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0, 200),
            )

    base = Image.alpha_composite(base, overlay)
    return base.convert("RGB")


def build_mosaic(
    tile_paths: list[str],
    output_path: str,
    num_threads: str,
    compression: str,
    show: bool = False,
    annotate_year: bool = False,
    annotate_season: bool = False,
    provenance_path: str | None = None,
) -> None:
    env_opts = {
        "GDAL_NUM_THREADS": num_threads,
        "NUM_THREADS": num_threads,
        "GDAL_CACHEMAX": 512,
    }

    with rasterio.Env(**env_opts):
        # Open all datasets under a single ExitStack so merge receives real datasets.
        with ExitStack() as stack:
            datasets = [stack.enter_context(rasterio.open(os.fspath(p))) for p in tile_paths]
            indexes, dtype = _select_indexes_and_dtype(datasets)
            base_profile = datasets[0].profile.copy()
            mosaic, transform = merge(datasets, indexes=indexes, dtype=dtype)
        profile = base_profile
        profile.update(
            {
                "count": mosaic.shape[0],
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "dtype": dtype,
            }
        )

        # Choose driver based on extension
        ext = os.path.splitext(output_path)[1].lower()
        driver = "GTiff"
        if ext in (".png",):
            driver = "PNG"
        elif ext in (".jpg", ".jpeg"):
            driver = "JPEG"
        profile["driver"] = driver

        if driver == "GTiff":
            profile["compress"] = compression
        else:
            # Drop GeoTIFF-only options some drivers reject
            for key in ("compress", "tiled", "blockxsize", "blockysize", "interleave"):
                profile.pop(key, None)
            # PNG/JPEG expect 8-bit
            if mosaic.dtype != "uint8":
                mosaic = mosaic.astype("uint8")
                profile["dtype"] = "uint8"

        # Optional year annotation: draw on the mosaic array before writing
        if annotate_year and annotate_season:
            raise SystemExit("Choose only one of --annotate-year or --annotate-season.")

        if annotate_year or annotate_season:
            prov_path = provenance_path
            if not os.path.exists(prov_path):
                print(
                    f"[mosaic] provenance not found at {prov_path}; skipping annotation",
                    file=sys.stderr,
                )
            else:
                rows = _load_provenance(prov_path)

                # Restrict provenance to the tiles actually mosaicked.
                # Prefer matching by processed filename (handles UTM-named split outputs),
                # then fall back to grid coords when filename is unavailable.
                tile_basenames = {os.path.basename(p) for p in tile_paths}
                wanted_coords = set()
                for p in tile_paths:
                    coords = parse_tile_coords(os.path.basename(p))
                    if coords:
                        wanted_coords.add(coords)

                filtered = []
                seen_files = set()
                seen_coords = set()
                for r in rows:
                    processed_name = r.get("processed_file") or r.get("source_file")
                    processed_base = os.path.basename(processed_name) if processed_name else None

                    # Primary match: exact processed filename used in this mosaic
                    if (
                        processed_base
                        and processed_base in tile_basenames
                        and processed_base not in seen_files
                    ):
                        filtered.append(r)
                        seen_files.add(processed_base)
                        # Also mark its coords to avoid double-counting via fallback
                        gx, gy = _parse_coords(r)
                        if gx is not None and gy is not None:
                            seen_coords.add((gx, gy))
                        continue

                    # Fallback: coordinate match when filename isn't present in provenance
                    gx, gy = _parse_coords(r)
                    if gx is None or gy is None:
                        continue
                    if wanted_coords and (gx, gy) not in wanted_coords:
                        continue
                    if (gx, gy) in seen_coords:
                        continue
                    filtered.append(r)
                    seen_coords.add((gx, gy))

                if len(filtered) != len(rows):
                    print(
                        f"[mosaic] filtered provenance rows to {len(filtered)}/{len(rows)} matching tiles",
                        file=sys.stderr,
                    )
                if annotate_year:
                    annotated = _annotate_year(mosaic, transform, filtered)
                    anno_label = "Year"
                else:
                    annotated = _annotate_season(mosaic, transform, filtered)
                    anno_label = "Season"
                if annotated is not None:
                    annotated_np = np.array(annotated)
                    # Convert HWC -> CHW
                    if annotated_np.ndim == 2:  # single band
                        mosaic = annotated_np[np.newaxis, ...]
                        profile["count"] = 1
                    else:
                        mosaic = annotated_np.transpose(2, 0, 1)
                        profile["count"] = mosaic.shape[0]
                    profile["dtype"] = mosaic.dtype
                    print(f"[mosaic] {anno_label} annotation applied to output (single file).")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic)

        if show:
            _show_mosaic(mosaic)


def main() -> None:
    args = parse_args()
    tiles = find_tiles(args.input_dir, args.pattern)
    print(f"Found {len(tiles)} tiles. Building mosaic -> {args.output}")
    prov_default = args.provenance
    if prov_default is None:
        # Default: provenance.csv next to the resolution folder's parent (processed/)
        prov_default = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(args.input_dir))), "provenance.csv"
        )
    build_mosaic(
        tiles,
        args.output,
        args.num_threads,
        args.compression,
        show=args.show,
        annotate_year=args.annotate_year,
        annotate_season=args.annotate_season,
        provenance_path=prov_default,
    )
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
