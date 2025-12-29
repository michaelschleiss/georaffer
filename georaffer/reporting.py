"""Console output and progress reporting for pipeline operations."""

import textwrap
from collections.abc import Iterable, Sequence
from typing import Any

from georaffer.config import Region

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
    regions: list[Region] | None = None,
) -> None:
    """Print pipeline configuration summary.

    Args:
        num_coords: Number of input coordinates
        grid_size_km: Grid resolution in km
        margin_km: Margin buffer in km
        resolutions: Output resolutions
        output_dir: Output directory path
        imagery_from: Optional (from_year, to_year) for historic imagery
        regions: Optional list of regions included in the run
    """
    print("Configuration:")
    print(f"  • Input coordinates: {num_coords} locations")
    print(f"  • Grid resolution: {grid_size_km:.2f} km")
    print(f"  • Margin buffer: {margin_km:.2f} km from tile borders")
    print(f"  • Output resolutions: {', '.join(f'{r}px' for r in resolutions)}")
    print(f"  • Output directory: {output_dir}")
    if regions:
        region_names = ", ".join(region.value for region in regions)
        print(f"  • Regions: {region_names}")
    if imagery_from:
        from_year, to_year = imagery_from
        if to_year is None:
            print(f"  • Historical imagery: {from_year} to present")
        else:
            print(f"  • Historical imagery: {from_year} to {to_year}")
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
