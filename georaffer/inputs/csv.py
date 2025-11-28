"""Load coordinates from CSV files."""

import csv


def load_from_csv(filepath: str, x_col: str = "x", y_col: str = "y") -> list[tuple[float, float]]:
    """Load UTM coordinates from a CSV file.

    Args:
        filepath: Path to CSV file
        x_col: Column name for X/easting coordinate
        y_col: Column name for Y/northing coordinate

    Returns:
        List of (x, y) coordinate tuples

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        KeyError: If specified columns don't exist
        ValueError: If coordinate values cannot be parsed as floats
    """
    coords = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):  # +2 for header + 1-based
            try:
                x = float(row[x_col])
                y = float(row[y_col])
            except KeyError as e:
                raise KeyError(
                    f"Row {row_num}: Column {e} not found. Available: {list(row.keys())}"
                ) from e
            except ValueError as e:
                raise ValueError(f"Row {row_num}: Cannot parse coordinate - {e}") from e
            coords.append((x, y))

    if not coords:
        raise ValueError(f"No coordinates found in {filepath}")

    return coords
