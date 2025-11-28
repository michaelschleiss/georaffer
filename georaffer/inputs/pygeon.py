"""Load coordinates from 4Seasons dataset INS files."""

import csv
from pathlib import Path

import numpy as np


def _load_ins(path: Path) -> np.ndarray:
    """Load INS CSV with lat/lon/alt columns.

    Compatible with 4Seasons dataset ins.csv format.
    Uses csv.reader for fast parsing (~6x faster than numpy.genfromtxt).
    """
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        lat_i = header.index("latitude")
        lon_i = header.index("longitude")
        alt_i = header.index("altitude")
        rows = [(float(r[lat_i]), float(r[lon_i]), float(r[alt_i])) for r in reader]

    return np.array(rows, dtype=np.float32)


def load_from_pygeon(campaign_dir: str) -> list[tuple[float, float, float]]:
    """Load coordinates from pygeon 4Seasons dataset(s).

    Auto-detects mode:
    - If campaign_dir/ins.csv exists, loads single campaign
    - Otherwise, searches for ins.csv in any immediate subdirectory (*/ins.csv)

    Args:
        campaign_dir: Path to folder with ins.csv, or parent folder containing campaign subfolders

    Returns:
        List of (latitude, longitude, altitude) tuples from all found campaigns

    Raises:
        FileNotFoundError: If no ins.csv files found
    """
    base_path = Path(campaign_dir)
    ins_path = base_path / "ins.csv"

    # Auto-detect: single campaign or multiple campaigns?
    if ins_path.exists():
        # Single campaign mode - _load_ins returns array with columns [lat, lon, alt]
        ins_data = _load_ins(ins_path)
        return [tuple(row) for row in ins_data]

    # Multi-campaign mode: search for ins.csv in any immediate subdirectory
    campaign_paths = sorted(base_path.glob("*/ins.csv"))

    if not campaign_paths:
        raise FileNotFoundError(
            f"No ins.csv files found in {campaign_dir}. "
            "Please provide a valid 4Seasons campaign directory."
        )

    all_coordinates = []
    for ins_path in campaign_paths:
        ins_data = _load_ins(ins_path)
        all_coordinates.extend(tuple(row) for row in ins_data)

    return all_coordinates
