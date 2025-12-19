"""Console output and progress reporting for pipeline operations."""

import textwrap
from collections.abc import Iterable, Sequence
from typing import Any

from georaffer.tiles import TileSet

# ASCII logo printed once at pipeline start.
PIPELINE_LOGO = textwrap.dedent(
    r"""
      ____   _____    ___    ____       _      _____   _____   _____   ____ 
     / ___| | ____|  / _ \  |  _ \     / \    |  ___| |  ___| | ____| |  _ \
    | |  _  |  _|   | | | | | |_) |   / _ \   | |_    | |_    |  _|   | |_) |
    | |_| | | |___  | |_| | |  _ <   / ___ \  |  _|   |  _|   | |___  |  _ <
     \____| |_____|  \___/  |_| \_\ /_/   \_\ |_|     |_|     |_____| |_| \_\
    """
).lstrip("\n")


def render_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Render a simple ASCII table with aligned columns."""
    headers = list(headers)
    rows = [list(map(lambda x: "-" if x is None else str(x), row)) for row in rows]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt_row(row_vals):
        return "  ".join(val.ljust(widths[i]) for i, val in enumerate(row_vals))

    lines = [_fmt_row(headers)]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append(_fmt_row(row))
    return "\n".join(lines)


def print_table(title: str, headers: Sequence[str], rows: list[Sequence[Any]]) -> None:
    """Render and print a table with title.

    Args:
        title: Table title
        headers: Column headers
        rows: Row data
    """
    if not rows:
        return
    table_text = render_table(headers, rows)
    print(f"{title}\n{table_text}")


def print_header(text: str, char: str = "=", width: int = 80) -> None:
    """Print a header line with border characters.

    Args:
        text: Header text
        char: Border character
        width: Total width
    """
    print(char * width)
    print(text)
    print(char * width)


def print_step_header(step_num: int, title: str) -> None:
    """Print a step header.

    Args:
        step_num: Step number
        title: Step title
    """
    print("─" * 80)
    print(f"STEP {step_num}: {title}")
    print("─" * 80)


def print_pipeline_banner() -> None:
    """Print the pipeline startup banner."""
    print()
    print(PIPELINE_LOGO)
    print("=" * 80)
    print("GEORAFFER - Geospatial Tile Processing Pipeline")
    print("=" * 80)


def print_config(
    num_coords: int,
    grid_size_km: float,
    margin_km: float,
    resolutions: list[int],
    output_dir: str,
    imagery_from: tuple[int, int | None] | None = None,
) -> None:
    """Print pipeline configuration summary.

    Args:
        num_coords: Number of input coordinates
        grid_size_km: Grid resolution in km
        margin_km: Margin buffer in km
        resolutions: Output resolutions
        output_dir: Output directory path
        imagery_from: Optional (from_year, to_year) for historic imagery
    """
    print("Configuration:")
    print(f"  • Input coordinates: {num_coords} locations")
    print(f"  • Grid resolution: {grid_size_km:.2f} km")
    print(f"  • Margin buffer: {margin_km:.2f} km from tile borders")
    print(f"  • Output resolutions: {', '.join(f'{r}px' for r in resolutions)}")
    print(f"  • Output directory: {output_dir}")
    if imagery_from:
        from_year, to_year = imagery_from
        if to_year is None:
            print(f"  • Historical imagery: {from_year} to present (NRW only)")
        else:
            print(f"  • Historical imagery: {from_year} to {to_year} (NRW only)")
    print()


def print_catalog_summary(rows: Sequence[tuple[str, int, int]], duration: float) -> None:
    """Print tile catalog summary table.

    Args:
        rows: List of (region_label, imagery_count, dsm_count)
        duration: Query duration in seconds (shown in footer)
    """
    print()
    print_table(
        "Available Tiles by Region",
        ["Region", "Imagery (JP2)", "DSM (LAZ/TIF)"],
        [(region, f"{jp2:,}", f"{dsm:,}") for region, jp2, dsm in rows],
    )
    print(f"  Query time: {duration:.1f}s")
    print()


def print_coverage_analysis(
    tile_set: TileSet,
    user_tiles: set[tuple[int, int]],
    jp2_estimate: int,
    laz_estimate: int,
    jp2_split: bool,
    laz_split: bool,
) -> None:
    """Print tile coverage analysis.

    Args:
        tile_set: Resolved tile set
        user_tiles: Set of user-requested tiles
        jp2_estimate: Estimated JP2 outputs
        laz_estimate: Estimated LAZ outputs
        jp2_split: Whether JP2 splitting will occur
        laz_split: Whether LAZ splitting will occur
    """
    total_user = len(user_tiles)
    covered_jp2 = total_user - len(tile_set.missing_jp2)
    covered_laz = total_user - len(tile_set.missing_laz)

    print()
    print_table(
        "Tile Coverage Analysis",
        ["Type", "User Tiles", "Covered", "Missing", "Coverage %"],
        [
            (
                "JP2 (Imagery)",
                str(total_user),
                str(covered_jp2),
                str(len(tile_set.missing_jp2)),
                f"{100 * covered_jp2 / total_user:.1f}%" if total_user else "0%",
            ),
            (
                "LAZ (DSM)",
                str(total_user),
                str(covered_laz),
                str(len(tile_set.missing_laz)),
                f"{100 * covered_laz / total_user:.1f}%" if total_user else "0%",
            ),
        ],
    )

    print()
    print_table(
        "Download Plan",
        ["Source", "JP2 Files", "LAZ Files"],
        [
            ("NRW", str(tile_set.jp2_count("nrw")), str(tile_set.laz_count("nrw"))),
            ("RLP", str(tile_set.jp2_count("rlp")), str(tile_set.laz_count("rlp"))),
            ("TOTAL", str(tile_set.total_jp2), str(tile_set.total_laz)),
        ],
    )

    if jp2_split or laz_split:
        print()
        split_info = []
        if jp2_split:
            split_info.append(f"JP2: {tile_set.total_jp2} → {jp2_estimate} outputs")
        if laz_split:
            split_info.append(f"LAZ: {tile_set.total_laz} → {laz_estimate} outputs")
        print(f"Tile splitting enabled: {', '.join(split_info)}")

    print()


def print_download_summary(downloaded: int, skipped: int, failed: int, duration: float) -> None:
    """Print download phase summary.

    Args:
        downloaded: Number of files downloaded
        skipped: Number of files skipped (already exist)
        failed: Number of failed downloads
        duration: Download duration in seconds
    """
    print()
    print_table(
        "Download Summary",
        ["Status", "Count"],
        [
            ("Downloaded", str(downloaded)),
            ("Skipped (cached)", str(skipped)),
            ("Failed", str(failed)),
            ("Duration", f"{duration:.1f}s"),
        ],
    )
    print()


def print_conversion_summary(
    jp2_sources: int,
    jp2_converted: int,
    jp2_failed: int,
    jp2_duration: float,
    laz_sources: int,
    laz_converted: int,
    laz_failed: int,
    laz_duration: float,
) -> None:
    """Print conversion phase summary.

    Args:
        jp2_sources: Number of JP2 source files
        jp2_converted: Number of JP2 outputs created
        jp2_failed: Number of JP2 conversion failures
        jp2_duration: JP2 conversion duration
        laz_sources: Number of LAZ source files
        laz_converted: Number of LAZ outputs created
        laz_failed: Number of LAZ conversion failures
        laz_duration: LAZ conversion duration
    """
    print()
    print_table(
        "Conversion Summary",
        ["Type", "Sources", "Outputs", "Failed", "Duration"],
        [
            (
                "JP2 → GeoTIFF",
                str(jp2_sources),
                str(jp2_converted),
                str(jp2_failed),
                f"{jp2_duration:.1f}s",
            ),
            (
                "LAZ → DSM",
                str(laz_sources),
                str(laz_converted),
                str(laz_failed),
                f"{laz_duration:.1f}s",
            ),
        ],
    )
    print()


def print_final_summary(
    total_duration: float,
    downloaded: int,
    skipped: int,
    failed_download: int,
    converted: int,
    failed_convert: int,
    interrupted: bool = False,
) -> None:
    """Print final pipeline summary.

    Args:
        total_duration: Total run time in seconds
        downloaded: Files downloaded
        skipped: Files skipped
        failed_download: Download failures
        converted: Files converted
        failed_convert: Conversion failures
        interrupted: Whether run was interrupted
    """
    print()
    print("=" * 80)
    if interrupted:
        print("PIPELINE INTERRUPTED")
    else:
        print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Total duration: {total_duration:.1f}s")
    print(f"Downloads: {downloaded} new, {skipped} cached, {failed_download} failed")
    print(f"Conversions: {converted} successful, {failed_convert} failed")
    print()
