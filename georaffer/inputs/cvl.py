"""Load coordinates from German-CVL pose CSVs."""

import csv
from pathlib import Path


def _load_pose_csv(path: Path) -> list[tuple[float, float, float]]:
    """Load lat/lon/alt coordinates from a single pose CSV."""
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            return []
        try:
            lat_i = header.index("lat")
            lon_i = header.index("lon")
            alt_i = header.index("alt")
        except ValueError as exc:
            raise ValueError(f"{path} missing required columns: lat, lon, alt") from exc

        rows: list[tuple[float, float, float]] = []
        for row_num, row in enumerate(reader, start=2):
            try:
                rows.append((float(row[lat_i]), float(row[lon_i]), float(row[alt_i])))
            except (IndexError, ValueError) as exc:
                raise ValueError(f"{path}:{row_num} invalid lat/lon/alt values") from exc
        return rows


def _collect_pose_csvs(base_path: Path) -> list[Path]:
    if base_path.is_file():
        if base_path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a .csv file, got {base_path}")
        return [base_path]

    if not base_path.exists():
        raise FileNotFoundError(base_path)

    if base_path.name == "poses":
        pose_roots = [base_path]
    elif (base_path / "poses").is_dir():
        pose_roots = [base_path / "poses"]
    else:
        pose_roots = [p / "poses" for p in sorted(base_path.iterdir()) if (p / "poses").is_dir()]

    if pose_roots:
        csv_paths: list[Path] = []
        for root in pose_roots:
            csv_paths.extend(sorted(root.rglob("*.csv")))
        return csv_paths

    return sorted(base_path.rglob("*.csv"))


def load_from_cvl(poses_path: str) -> list[tuple[float, float, float]]:
    """Load coordinates from German-CVL standardized pose CSVs.

    Accepts:
    - data/ (dataset folders containing poses/)
    - data/kitti (dataset folder containing poses/)
    - data/kitti/poses (pose directory)
    - data/kitti/poses/<sequence> (sequence directory)
    - data/kitti/poses/<sequence>/<camera>.csv (single pose CSV)
    """
    base_path = Path(poses_path)
    csv_paths = _collect_pose_csvs(base_path)
    if not csv_paths:
        raise FileNotFoundError(f"No pose CSVs found in {poses_path}")

    all_coordinates: list[tuple[float, float, float]] = []
    for csv_path in csv_paths:
        all_coordinates.extend(_load_pose_csv(csv_path))

    return all_coordinates
