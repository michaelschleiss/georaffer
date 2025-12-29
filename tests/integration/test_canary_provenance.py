"""Network canary test for provenance tracking.

Downloads real tiles from NRW and RLP to verify provenance.csv is correctly
generated with valid metadata for all permutations:
- NRW × RLP
- Current × Historical
- Image (JP2/TIF) × DSM (LAZ)
"""

import csv
import re
from pathlib import Path

import pytest

from georaffer.conversion import convert_tiles
from georaffer.downloaders import NRWDownloader, RLPDownloader
from georaffer.tiles import _filename_from_url

# Known tiles with coverage
NRW_TILE = (350, 5600)
RLP_TILE = (380, 5540)  # Known to have 2020, 2022, 2024 historical WMS coverage

# Ground truth for HISTORICAL data (fixed, won't change)
# Queried from WMS 2024-12-10
NRW_HISTORICAL_GROUND_TRUTH = {
    2019: {"acquisition_date": "2019-06-30"},
    2021: {"acquisition_date": "2021-02-25"},
}
RLP_HISTORICAL_GROUND_TRUTH = {
    2020: {"acquisition_date": "2020-08-07"},
    2022: {"acquisition_date": "2022-06-14"},
    2024: {"acquisition_date": "2024-08-06"},
}


def _download_tile(downloader, catalog, tile, subdir="image"):
    """Download tile using production downloader."""
    if tile not in catalog:
        return None
    url = catalog[tile]
    filename = _filename_from_url(url)
    output_path = downloader.raw_dir / subdir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 1000:
        return output_path

    success = downloader.download_file(url, str(output_path))
    return output_path if success else None


@pytest.mark.network
@pytest.mark.canary
@pytest.mark.slow
@pytest.mark.integration
def test_provenance_comprehensive_ground_truth(tmp_path):
    """Canary: All permutations produce correct provenance.csv with ground truth validation.

    Downloads and converts (6 source files):
    - NRW current image (JP2) → 1 row
    - NRW current DSM (LAZ) → 1 row
    - NRW historical 2021 image → 1 row, ground truth date=2021-02-25
    - RLP current image (JP2) → 4 rows (2×2 split)
    - RLP current DSM (LAZ) → 4 rows (2×2 split)
    - RLP historical 2022 WMS → 4 rows (2×2 split), ground truth date=2022-06-14

    Total expected: 1 + 1 + 1 + 4 + 4 + 4 = 15 rows in provenance.csv
    """
    raw_dir = tmp_path / "raw"
    (raw_dir / "image").mkdir(parents=True)
    (raw_dir / "dsm").mkdir(parents=True)

    downloaded = {}  # type -> (filename, expected_assertions)

    # === NRW CURRENT (image + LAZ) ===
    nrw = NRWDownloader(output_dir=str(tmp_path / "nrw"))
    jp2_cat, laz_cat = nrw.get_available_tiles()

    path = _download_tile(nrw, jp2_cat, NRW_TILE, "image")
    assert path, f"NRW current JP2 download failed for {NRW_TILE}"
    (raw_dir / "image" / path.name).write_bytes(path.read_bytes())
    downloaded["nrw_current_image"] = path.name

    path = _download_tile(nrw, laz_cat, NRW_TILE, "dsm")
    assert path, f"NRW current LAZ download failed for {NRW_TILE}"
    (raw_dir / "dsm" / path.name).write_bytes(path.read_bytes())
    downloaded["nrw_current_laz"] = path.name

    # === NRW HISTORICAL (2021) ===
    nrw_hist = NRWDownloader(output_dir=str(tmp_path / "nrw_hist"), imagery_from=(2021, 2021))
    jp2_cat_hist, _ = nrw_hist.get_available_tiles()

    path = _download_tile(nrw_hist, jp2_cat_hist, NRW_TILE, "image")
    assert path, f"NRW historical 2021 JP2 download failed for {NRW_TILE}"
    (raw_dir / "image" / path.name).write_bytes(path.read_bytes())
    downloaded["nrw_hist_2021"] = path.name

    # === RLP CURRENT (image + LAZ) ===
    rlp = RLPDownloader(output_dir=str(tmp_path / "rlp"))
    jp2_cat_rlp, laz_cat_rlp = rlp.get_available_tiles(requested_coords={RLP_TILE})

    path = _download_tile(rlp, jp2_cat_rlp, RLP_TILE, "image")
    assert path, f"RLP current JP2 download failed for {RLP_TILE}"
    (raw_dir / "image" / path.name).write_bytes(path.read_bytes())
    downloaded["rlp_current_image"] = path.name

    path = _download_tile(rlp, laz_cat_rlp, RLP_TILE, "dsm")
    assert path, f"RLP current LAZ download failed for {RLP_TILE}"
    (raw_dir / "dsm" / path.name).write_bytes(path.read_bytes())
    downloaded["rlp_current_laz"] = path.name

    # === RLP HISTORICAL WMS (2022) ===
    rlp_hist = RLPDownloader(output_dir=str(tmp_path / "rlp_hist"), imagery_from=(2022, 2022))
    jp2_cat_rlp_hist, _ = rlp_hist.get_available_tiles(requested_coords={RLP_TILE})

    path = _download_tile(rlp_hist, jp2_cat_rlp_hist, RLP_TILE, "image")
    assert path, f"RLP historical 2022 WMS download failed for {RLP_TILE}"
    (raw_dir / "image" / path.name).write_bytes(path.read_bytes())
    downloaded["rlp_hist_2022"] = path.name

    # === CONVERT ALL ===
    processed_dir = tmp_path / "processed"
    convert_tiles(str(raw_dir), str(processed_dir), resolutions=[2000], max_workers=1)

    # === VERIFY PROVENANCE.CSV ===
    csv_path = processed_dir / "provenance.csv"
    assert csv_path.exists(), "provenance.csv not created"

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # Group rows by source file
    by_source = {}
    for row in rows:
        src = row["source_file"]
        by_source.setdefault(src, []).append(row)

    # --- NRW current image (JP2) ---
    filename = downloaded["nrw_current_image"]
    assert filename in by_source, f"Missing NRW current image: {filename}"
    # Verify JP2 filename pattern: dop10rgbi_32_350_5600_YYYY.jp2
    assert filename.endswith(".jp2"), f"NRW current image not JP2: {filename}"
    assert filename.startswith("dop10rgbi_32_"), f"Wrong NRW JP2 pattern: {filename}"
    assert f"_{NRW_TILE[0]}_{NRW_TILE[1]}_" in filename, f"NRW coords not in filename: {filename}"
    file_rows = by_source[filename]
    assert len(file_rows) == 1, f"Expected 1 row for NRW 1km tile, got {len(file_rows)}"
    row = file_rows[0]
    assert row["source_region"] == "NRW", f"Wrong region: {row['source_region']}"
    assert row["file_type"] == "orthophoto", f"Wrong file_type: {row['file_type']}"
    assert int(row["grid_x"]) == NRW_TILE[0], f"Wrong grid_x: {row['grid_x']}"
    assert int(row["grid_y"]) == NRW_TILE[1], f"Wrong grid_y: {row['grid_y']}"
    year = int(row["year"])
    assert 2020 <= year <= 2030, f"Year out of range: {year}"
    assert re.match(r"\d{4}-\d{2}-\d{2}", row["acquisition_date"]), (
        f"Invalid date format: {row['acquisition_date']}"
    )

    # --- NRW current LAZ ---
    filename = downloaded["nrw_current_laz"]
    assert filename in by_source, f"Missing NRW current LAZ: {filename}"
    # Verify LAZ filename pattern: bdom50_32_350_5600_1_nw_YYYY.laz
    assert filename.endswith(".laz"), f"NRW current LAZ not .laz: {filename}"
    assert filename.startswith("bdom50_32"), f"Wrong NRW LAZ pattern: {filename}"
    file_rows = by_source[filename]
    assert len(file_rows) == 1, f"Expected 1 row for NRW 1km tile, got {len(file_rows)}"
    row = file_rows[0]
    assert row["source_region"] == "NRW", f"Wrong region: {row['source_region']}"
    assert row["file_type"] == "dsm", f"Wrong file_type: {row['file_type']}"
    assert int(row["grid_x"]) == NRW_TILE[0], f"Wrong grid_x: {row['grid_x']}"
    assert int(row["grid_y"]) == NRW_TILE[1], f"Wrong grid_y: {row['grid_y']}"

    # Expected RLP 2×2 split coordinates for tile (380, 5540)
    RLP_EXPECTED_COORDS = {(380, 5540), (381, 5540), (380, 5541), (381, 5541)}

    # --- RLP current image (JP2, 2×2 split) ---
    filename = downloaded["rlp_current_image"]
    assert filename in by_source, f"Missing RLP current image: {filename}"
    # Verify JP2 filename pattern: dop20rgb(i)_32_380_5540_... (may or may not have 'i')
    assert filename.endswith(".jp2"), f"RLP current image not JP2: {filename}"
    assert filename.startswith("dop20rgb"), f"Wrong RLP JP2 pattern: {filename}"
    assert f"_{RLP_TILE[0]}_{RLP_TILE[1]}_" in filename, f"RLP coords not in filename: {filename}"
    file_rows = by_source[filename]
    assert len(file_rows) == 4, f"Expected 4 rows for RLP 2km→1km split, got {len(file_rows)}"
    actual_coords = {(int(r["grid_x"]), int(r["grid_y"])) for r in file_rows}
    assert actual_coords == RLP_EXPECTED_COORDS, f"RLP current image coords: {actual_coords}"
    for row in file_rows:
        assert row["source_region"] == "RLP", f"Wrong region: {row['source_region']}"
        assert row["file_type"] == "orthophoto", f"Wrong file_type: {row['file_type']}"
        year = int(row["year"])
        assert 2020 <= year <= 2030, f"Year out of range: {year}"
        assert re.match(r"\d{4}-\d{2}-\d{2}", row["acquisition_date"]), (
            f"Invalid date format: {row['acquisition_date']}"
        )

    # --- RLP current LAZ (2×2 split) ---
    filename = downloaded["rlp_current_laz"]
    assert filename in by_source, f"Missing RLP current LAZ: {filename}"
    # Verify LAZ filename pattern: bdom20rgbi_32_380_5540_2_rp.laz
    assert filename.endswith(".laz"), f"RLP current LAZ not .laz: {filename}"
    assert filename.startswith("bdom20rgbi_32_"), f"Wrong RLP LAZ pattern: {filename}"
    file_rows = by_source[filename]
    assert len(file_rows) == 4, f"Expected 4 rows for RLP 2km→1km split, got {len(file_rows)}"
    actual_coords = {(int(r["grid_x"]), int(r["grid_y"])) for r in file_rows}
    assert actual_coords == RLP_EXPECTED_COORDS, f"RLP current LAZ coords: {actual_coords}"
    for row in file_rows:
        assert row["source_region"] == "RLP", f"Wrong region: {row['source_region']}"
        assert row["file_type"] == "dsm", f"Wrong file_type: {row['file_type']}"

    # --- NRW historical 2021 (JP2, ground truth) ---
    filename = downloaded["nrw_hist_2021"]
    assert filename in by_source, f"Missing NRW historical 2021: {filename}"
    # Verify JP2 filename pattern: dop10rgbi_32_350_5600_2021.jp2
    assert filename.endswith(".jp2"), f"NRW hist not JP2: {filename}"
    assert filename.startswith("dop10rgbi_32_"), f"Wrong NRW hist JP2 pattern: {filename}"
    assert "_2021.jp2" in filename, f"Year 2021 not in filename: {filename}"
    file_rows = by_source[filename]
    assert len(file_rows) == 1, f"Expected 1 row for NRW 1km tile, got {len(file_rows)}"
    row = file_rows[0]
    assert row["source_region"] == "NRW", f"Wrong region: {row['source_region']}"
    assert row["file_type"] == "orthophoto", f"Wrong file_type: {row['file_type']}"
    assert int(row["grid_x"]) == NRW_TILE[0], f"Wrong grid_x: {row['grid_x']}"
    assert int(row["grid_y"]) == NRW_TILE[1], f"Wrong grid_y: {row['grid_y']}"
    # Historical ground truth: exact year and date
    assert row["year"] == "2021", f"Wrong year: {row['year']}"
    assert row["acquisition_date"] == NRW_HISTORICAL_GROUND_TRUTH[2021]["acquisition_date"], (
        f"NRW 2021 date mismatch: {row['acquisition_date']} != {NRW_HISTORICAL_GROUND_TRUTH[2021]['acquisition_date']}"
    )

    # --- RLP historical 2022 (TIF from WMS, ground truth + 2×2 split) ---
    filename = downloaded["rlp_hist_2022"]
    assert filename in by_source, f"Missing RLP historical 2022: {filename}"
    # Verify TIF filename pattern: dop20rgb_32_380_5540_2_rp_2022.tif
    assert filename.endswith(".tif"), f"RLP hist WMS not TIF: {filename}"
    assert filename.startswith("dop20rgb_32_"), f"Wrong RLP WMS TIF pattern: {filename}"
    assert f"_{RLP_TILE[0]}_{RLP_TILE[1]}_" in filename, f"RLP coords not in filename: {filename}"
    assert "_2022.tif" in filename, f"Year 2022 not in filename: {filename}"
    file_rows = by_source[filename]
    assert len(file_rows) == 4, f"Expected 4 rows for RLP 2km→1km split, got {len(file_rows)}"
    actual_coords = {(int(r["grid_x"]), int(r["grid_y"])) for r in file_rows}
    assert actual_coords == RLP_EXPECTED_COORDS, f"RLP historical 2022 coords: {actual_coords}"
    for row in file_rows:
        assert row["source_region"] == "RLP", f"Wrong region: {row['source_region']}"
        assert row["file_type"] == "orthophoto", f"Wrong file_type: {row['file_type']}"
        # Historical ground truth: exact year and date
        assert row["year"] == "2022", f"Wrong year: {row['year']}"
        assert row["acquisition_date"] == RLP_HISTORICAL_GROUND_TRUTH[2022]["acquisition_date"], (
            f"RLP 2022 date mismatch: {row['acquisition_date']} != {RLP_HISTORICAL_GROUND_TRUTH[2022]['acquisition_date']}"
        )
