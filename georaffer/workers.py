"""Conversion worker functions for ProcessPoolExecutor.

These functions must be top-level (not methods) because ProcessPoolExecutor
uses pickle, which cannot serialize methods. Each worker processes one source
file with all its target resolutions.
"""

import os

# Suppress "omp_set_nested deprecated" warning in worker processes.
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
os.environ.setdefault("KMP_WARNINGS", "0")

import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path


def init_worker(threads_per_worker: int) -> None:
    """Initialize worker process environment.

    Sets thread limits for parallel libraries to avoid oversubscription
    when multiple workers run concurrently.

    Args:
        threads_per_worker: Maximum threads for parallel libraries (lazrs, numba)
    """
    threads_str = str(threads_per_worker)
    # lazrs uses Rayon (Rust parallel runtime)
    os.environ["RAYON_NUM_THREADS"] = threads_str
    # numba parallel loops
    os.environ["NUMBA_NUM_THREADS"] = threads_str
    # OpenMP (used by some GDAL operations)
    os.environ["OMP_NUM_THREADS"] = threads_str
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    os.environ["OMP_NESTED"] = "FALSE"
    # Suppress "omp_set_nested deprecated" warning from GDAL/rasterio internals.
    # This is an upstream issue - GDAL still uses the deprecated OpenMP API.
    # See: https://github.com/OSGeo/gdal - waiting for fix to use omp_set_max_active_levels
    os.environ["KMP_WARNINGS"] = "0"


from georaffer.config import (
    BB_BDOM_PATTERN,
    BB_DOP_PATTERN,
    BW_DOM_PATTERN,
    BW_DOP_PATTERN,
    BY_DOM_PATTERN,
    BY_DOP_PATTERN,
    FILE_TILE_SIZE_KM,
    METERS_PER_KM,
    NRW_JP2_PATTERN,
    NRW_LAZ_PATTERN,
    RLP_JP2_PATTERN,
    RLP_LAZ_PATTERN,
    TH_DOP_PATTERN,
    TH_LAZ_PATTERN,
    Region,
    utm_zone_str_for_region,
)
from georaffer.converters import convert_dsm_raster, convert_jp2, convert_laz, get_laz_year
from georaffer.converters.utils import parse_tile_coords
from georaffer.grids import compute_split_factor


def extract_year_from_filename(filename: str) -> str | None:
    """Extract 4-digit year from filename (e.g., 'tile_2021.jp2' -> '2021')."""
    match = re.search(r"_(\d{4})\.", filename)
    return match.group(1) if match else None


def detect_region(filename: str) -> Region:
    """Detect source region from filename.

    Args:
        filename: Source filename

    Returns:
        Region enum (NRW, RLP, BB, BW, or BY)
    """
    filename_lower = filename.lower()

    if RLP_JP2_PATTERN.match(filename) or RLP_LAZ_PATTERN.match(filename):
        return Region.RLP
    if BB_BDOM_PATTERN.match(filename_lower) or BB_DOP_PATTERN.match(filename_lower):
        return Region.BB
    if BW_DOP_PATTERN.match(filename_lower) or BW_DOM_PATTERN.match(filename_lower):
        return Region.BW
    if BY_DOP_PATTERN.match(filename) or BY_DOM_PATTERN.match(filename):
        return Region.BY
    if TH_DOP_PATTERN.match(filename_lower) or TH_LAZ_PATTERN.match(filename_lower):
        return Region.TH
    if NRW_JP2_PATTERN.match(filename) or NRW_LAZ_PATTERN.match(filename):
        return Region.NRW
    raise ValueError(f"Unrecognized tile filename: {filename}")


def generate_output_name(
    filename: str,
    region: Region,
    year: str,
    tile_type: str,
) -> str:
    """Generate standardized output filename.

    Output format: {region}_{zone}_{easting}_{northing}_{year}.tif

    Args:
        filename: Source filename (used to extract coordinates)
        region: Source region
        year: Year string
        tile_type: 'image' or 'dsm' (for logging only)

    Returns:
        Standardized output filename
    """
    # Extract grid coordinates from filename
    match_result = parse_tile_coords(filename)
    if not match_result:
        raise ValueError(f"Cannot parse grid coordinates from filename: {filename}")
    grid_x, grid_y = match_result

    year_str = str(year) if year is not None else ""
    if not year_str.isdigit() or len(year_str) != 4:
        raise ValueError(f"Year is required for output filenames (got '{year}').")

    easting = grid_x * METERS_PER_KM
    northing = grid_y * METERS_PER_KM

    utm_zone = utm_zone_str_for_region(region)
    return f"{region.value.lower()}_{utm_zone}_{int(easting)}_{int(northing)}_{year_str}.tif"


def convert_jp2_worker(args: tuple) -> tuple[bool, str, int]:
    """Worker function to convert a single JP2 file with all its resolutions.

    Args:
        args: Tuple of (filename, jp2_dir, processed_dir, resolutions,
              num_threads, grid_size_km, profiling)

    Returns:
        Tuple of (success, filename, outputs_count) where:
        - success: True if conversion succeeded, False otherwise
        - filename: Original JP2 filename (for logging/tracking)
        - outputs_count: Number of GeoTIFF files created (accounts for splits and resolutions)

    Raises:
        RuntimeError: If conversion fails
    """
    (
        filename,
        jp2_dir,
        processed_dir,
        resolutions,
        num_threads,
        grid_size_km,
        profiling,
        _,  # precomputed_wms (unused)
    ) = args

    input_path = Path(jp2_dir) / filename
    region = detect_region(filename)
    year = None
    if not (region == Region.BW and input_path.suffix.lower() == ".zip"):
        year = resolve_source_year(filename, input_path, data_type="image", region=region)

    try:
        # Setup output paths for each resolution
        outputs_count = 0
        tile_km = FILE_TILE_SIZE_KM[region]
        split_factor = compute_split_factor(tile_km, grid_size_km)

        if input_path.suffix.lower() == ".zip" and region == Region.BW:
            with tempfile.TemporaryDirectory() as tmpdir:
                extracted_tifs = _extract_zip_tifs(input_path, Path(tmpdir))
                for tif_path in extracted_tifs:
                    sub_name = tif_path.name
                    sub_year = resolve_source_year(
                        sub_name, tif_path, data_type="image", region=region
                    )
                    output_paths: dict[int | None, str] = {}
                    for res in resolutions:
                        output_name = generate_output_name(sub_name, region, sub_year, "image")
                        res_dir = Path(processed_dir) / "image" / (
                            "native" if res is None else str(res)
                        )
                        res_dir.mkdir(parents=True, exist_ok=True)
                        output_paths[res] = str(res_dir / output_name)
                    convert_jp2(
                        tif_path,
                        output_paths,
                        region,
                        sub_year,
                        resolutions,
                        num_threads=num_threads,
                        grid_size_km=grid_size_km,
                        profiling=profiling,
                    )
                    outputs_count += len(resolutions) * split_factor
        else:
            output_paths = {}
            for res in resolutions:
                output_name = generate_output_name(filename, region, year, "image")
                res_dir = Path(processed_dir) / "image" / ("native" if res is None else str(res))
                res_dir.mkdir(parents=True, exist_ok=True)
                output_paths[res] = str(res_dir / output_name)

            if input_path.suffix.lower() == ".zip":
                with tempfile.TemporaryDirectory() as tmpdir:
                    extracted_tif = _extract_bb_zip_tif(input_path, Path(tmpdir))
                    convert_jp2(
                        extracted_tif,
                        output_paths,
                        region,
                        year,
                        resolutions,
                        num_threads=num_threads,
                        grid_size_km=grid_size_km,
                        profiling=profiling,
                    )
            else:
                convert_jp2(
                    input_path,
                    output_paths,
                    region,
                    year,
                    resolutions,
                    num_threads=num_threads,
                    grid_size_km=grid_size_km,
                    profiling=profiling,
                )
            outputs_count += len(resolutions) * split_factor

        return (True, filename, outputs_count)

    except Exception as e:
        raise RuntimeError(
            f"JP2 conversion failed for {filename} "
            f"(region={region}, year={year}, resolutions={resolutions})"
        ) from e


def convert_dsm_worker(args: tuple) -> tuple[bool, str, int]:
    """Worker function to convert a single DSM file (.laz or raster) with resolutions.

    Args:
        args: Tuple of (filename, laz_dir, processed_dir, resolutions,
              num_threads, grid_size_km, profiling)

    Returns:
        Tuple of (success, filename, outputs_count) where:
        - success: True if conversion succeeded, False otherwise
        - filename: Original DSM filename (for logging/tracking)
        - outputs_count: Number of DSM GeoTIFF files created (accounts for splits and resolutions)

    Raises:
        RuntimeError: If conversion fails
    """
    (
        filename,
        laz_dir,
        processed_dir,
        resolutions,
        num_threads,
        grid_size_km,
        profiling,
        _,  # precomputed_wms (unused)
    ) = args

    input_path = Path(laz_dir) / filename
    region = detect_region(filename)
    year = None
    if not (region == Region.BW and input_path.suffix.lower() == ".zip"):
        year = resolve_source_year(filename, input_path, data_type="dsm", region=region)

    if input_path.suffix.lower() in (".tif", ".zip"):
        try:
            tile_km = FILE_TILE_SIZE_KM[region]
            split_factor = compute_split_factor(tile_km, grid_size_km)

            if input_path.suffix.lower() == ".zip" and region == Region.BW:
                outputs_count = 0
                with tempfile.TemporaryDirectory() as tmpdir:
                    extracted_tifs = _extract_zip_tifs(input_path, Path(tmpdir))
                    for tif_path in extracted_tifs:
                        sub_name = tif_path.name
                        sub_year = resolve_source_year(
                            sub_name, tif_path, data_type="dsm", region=region
                        )
                        output_paths: dict[int | None, str] = {}
                        for res in resolutions:
                            output_name = generate_output_name(sub_name, region, sub_year, "dsm")
                            res_dir = Path(processed_dir) / "dsm" / str(res)
                            res_dir.mkdir(parents=True, exist_ok=True)
                            output_paths[res] = str(res_dir / output_name)
                        convert_dsm_raster(
                            str(tif_path),
                            output_paths,
                            region,
                            sub_year,
                            target_sizes=resolutions,
                            num_threads=num_threads,
                            grid_size_km=grid_size_km,
                            profiling=profiling,
                        )
                        outputs_count += len(resolutions) * split_factor
                return (True, filename, outputs_count)

            output_paths: dict[int | None, str] = {}
            for res in resolutions:
                output_name = generate_output_name(filename, region, year, "dsm")
                res_dir = Path(processed_dir) / "dsm" / str(res)
                res_dir.mkdir(parents=True, exist_ok=True)
                output_paths[res] = str(res_dir / output_name)

            if input_path.suffix.lower() == ".zip":
                with tempfile.TemporaryDirectory() as tmpdir:
                    extracted_tif = _extract_bb_zip_tif(input_path, Path(tmpdir))
                    convert_dsm_raster(
                        str(extracted_tif),
                        output_paths,
                        region,
                        year,
                        target_sizes=resolutions,
                        num_threads=num_threads,
                        grid_size_km=grid_size_km,
                        profiling=profiling,
                    )
            else:
                convert_dsm_raster(
                    str(input_path),
                    output_paths,
                    region,
                    year,
                    target_sizes=resolutions,
                    num_threads=num_threads,
                    grid_size_km=grid_size_km,
                    profiling=profiling,
                )

            outputs_count = len(resolutions) * split_factor
            return (True, filename, outputs_count)
        except Exception as e:
            raise RuntimeError(
                f"DSM conversion failed for {filename} "
                f"(region={region}, year={year}, resolutions={resolutions})"
            ) from e

    # LAZ file path
    output_paths: dict[int | None, str] = {}
    for res in resolutions:
        output_name = generate_output_name(filename, region, year, "dsm")
        res_dir = Path(processed_dir) / "dsm" / str(res)
        res_dir.mkdir(parents=True, exist_ok=True)
        output_paths[res] = str(res_dir / output_name)

    try:
        convert_laz(
            input_path,
            output_paths,
            region,
            target_sizes=resolutions,
            num_threads=num_threads,
            grid_size_km=grid_size_km,
            profiling=profiling,
        )

        tile_km = FILE_TILE_SIZE_KM[region]
        split_factor = compute_split_factor(tile_km, grid_size_km)
        outputs_count = len(resolutions) * split_factor

        return (True, filename, outputs_count)

    except Exception as e:
        raise RuntimeError(
            f"LAZ conversion failed for {filename} "
            f"(region={region}, year={year}, resolutions={resolutions}): {e}"
        ) from e


def resolve_source_year(
    filename: str,
    input_path: Path,
    *,
    data_type: str,
    region: Region | None = None,
) -> str:
    """Resolve a 4-digit year for a source file or raise if unavailable."""

    def _normalize_year(value: str | None) -> str | None:
        if value is None:
            return None
        year_str = str(value)
        if year_str.isdigit() and len(year_str) == 4:
            return year_str
        raise ValueError(f"Invalid year '{value}' for {filename}.")

    region = region or detect_region(filename)
    if region == Region.BY:
        by_match = BY_DOP_PATTERN.match(filename)
        if by_match:
            by_year = _normalize_year(by_match.group(3))
            if by_year:
                return by_year
        year = None
    else:
        year = extract_year_from_filename(filename)

    normalized = _normalize_year(year)
    if normalized:
        return normalized

    if data_type == "image":
        if region in (Region.BB, Region.BW):
            if input_path.suffix.lower() != ".zip":
                raise ValueError(f"{region.value} raw tiles must be .zip: {filename}")
            meta_year = _normalize_year(_extract_zip_meta_year(input_path))
            if meta_year:
                return meta_year
            raise ValueError(f"Year not found in {region.value} metadata: {filename}")
        # BY uses direct TIFs without year in filename; use current year
        if region == Region.BY:
            from datetime import date
            return str(date.today().year)
        raise ValueError(f"Year not found in filename or metadata: {filename}")

    if data_type == "dsm":
        if input_path.suffix.lower() == ".laz":
            header_year = _normalize_year(get_laz_year(str(input_path)))
            if header_year:
                return header_year
        if input_path.suffix.lower() == ".zip":
            meta_year = _normalize_year(_extract_zip_meta_year(input_path))
            if meta_year:
                return meta_year
        if region in (Region.BB, Region.BW):
            raise ValueError(f"{region.value} raw tiles must be .zip: {filename}")
        # BY uses direct TIFs without year in filename; use current year
        if region == Region.BY:
            from datetime import date
            return str(date.today().year)
        raise ValueError(f"Year not found in filename or source metadata: {filename}")

    raise ValueError(f"Unknown data_type '{data_type}' for year resolution.")


def _extract_zip_meta_year(input_path: Path) -> str | None:
    """Extract year from ZIP metadata (BB or BW formats)."""
    if input_path.suffix.lower() != ".zip":
        return None

    texts: list[str] = []
    try:
        with zipfile.ZipFile(input_path) as zf:
            # Try XML metadata first, then HTML (BB provides .html since 2025)
            meta_name = (
                _find_zip_member(zf, "_meta.xml")
                or _find_zip_member(zf, ".xml")
                or _find_zip_member(zf, ".html")
            )
            if meta_name:
                texts.append(zf.read(meta_name).decode("utf-8", errors="ignore"))
    except Exception:
        return None

    for text in texts:
        year = _extract_zip_year_from_text(text)
        if year:
            return year
    return None


def _extract_zip_meta_date(input_path: Path) -> str | None:
    """Extract date from ZIP metadata (BB or BW formats)."""
    if input_path.suffix.lower() != ".zip":
        return None

    texts: list[str] = []
    try:
        with zipfile.ZipFile(input_path) as zf:
            meta_name = (
                _find_zip_member(zf, "_meta.xml")
                or _find_zip_member(zf, ".xml")
                or _find_zip_member(zf, ".html")
            )
            if meta_name:
                texts.append(zf.read(meta_name).decode("utf-8", errors="ignore"))
    except Exception:
        return None

    for text in texts:
        date = _extract_zip_date_from_text(text)
        if date:
            return date
    return None


def _extract_zip_year_from_text(text: str) -> str | None:
    """Extract year from ZIP metadata text (supports BB and BW formats)."""
    # BB legacy XML format: <file_creation_day_year>123/2024</file_creation_day_year>
    match = re.search(r"<file_creation_day_year>\d{1,3}/(\d{4})</file_creation_day_year>", text)
    if match:
        return match.group(1)

    # BB HTML format (2025+): Bildflugdatum:</td><td>2025-04-27</td>
    match = re.search(r"Bildflugdatum:</td><td>\s*(\d{4})-\d{2}-\d{2}", text)
    if match:
        return match.group(1)

    # BW XML format: try common date patterns
    # Look for befliegungsdatum (flight date) or erstellung (creation)
    match = re.search(r"befliegungsdatum[^0-9]*(\d{4})", text, re.IGNORECASE)
    if match:
        return match.group(1)

    year = _extract_iso_metadata_year(text, "creation")
    if year:
        return year
    return None


def _extract_zip_date_from_text(text: str) -> str | None:
    """Extract date from ZIP metadata text (supports BB and BW formats)."""
    match = re.search(
        r"Bildflugdatum[^0-9]*(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}\.\d{1,2}\.\d{4})",
        text,
        re.IGNORECASE,
    )
    if match:
        normalized = _normalize_metadata_date(match.group(1))
        if normalized:
            return normalized

    # BW: befliegungsdatum
    match = re.search(
        r"befliegungsdatum[^0-9]*(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}\.\d{1,2}\.\d{4})",
        text,
        re.IGNORECASE,
    )
    if match:
        normalized = _normalize_metadata_date(match.group(1))
        if normalized:
            return normalized

    match = re.search(
        r"erstellung[^0-9]*(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}\.\d{1,2}\.\d{4})",
        text,
        re.IGNORECASE,
    )
    if match:
        normalized = _normalize_metadata_date(match.group(1))
        if normalized:
            return normalized

    date = _extract_iso_metadata_date(text, "creation")
    if date:
        return date
    return None


def _extract_iso_metadata_year(text: str, date_type: str) -> str | None:
    try:
        root = ET.fromstring(text)
    except Exception:
        return None

    ns = {
        "gmd": "http://www.isotc211.org/2005/gmd",
        "gco": "http://www.isotc211.org/2005/gco",
    }

    for ci_date in root.findall(".//gmd:CI_Date", ns):
        type_el = ci_date.find("gmd:dateType/gmd:CI_DateTypeCode", ns)
        if type_el is None or not type_el.text:
            continue
        if type_el.text.strip() != date_type:
            continue
        date_el = ci_date.find("gmd:date/gco:DateTime", ns)
        if date_el is None:
            date_el = ci_date.find("gmd:date/gco:Date", ns)
        if date_el is None or not date_el.text:
            continue
        match = re.match(r"(19\d{2}|20\d{2})", date_el.text.strip())
        if match:
            return match.group(1)
    return None


def _extract_iso_metadata_date(text: str, date_type: str) -> str | None:
    try:
        root = ET.fromstring(text)
    except Exception:
        return None

    ns = {
        "gmd": "http://www.isotc211.org/2005/gmd",
        "gco": "http://www.isotc211.org/2005/gco",
    }

    for ci_date in root.findall(".//gmd:CI_Date", ns):
        type_el = ci_date.find("gmd:dateType/gmd:CI_DateTypeCode", ns)
        if type_el is None or not type_el.text:
            continue
        if type_el.text.strip() != date_type:
            continue
        date_el = ci_date.find("gmd:date/gco:DateTime", ns)
        if date_el is None:
            date_el = ci_date.find("gmd:date/gco:Date", ns)
        if date_el is None or not date_el.text:
            continue
        normalized = _normalize_metadata_date(date_el.text.strip())
        if normalized:
            return normalized
    return None


def _normalize_metadata_date(value: str) -> str | None:
    if not value:
        return None
    stripped = value.strip()

    match = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", stripped)
    if match:
        year, month, day = match.groups()
        normalized = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
        try:
            datetime.strptime(normalized, "%Y-%m-%d")
        except ValueError:
            return None
        return normalized

    match = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", stripped)
    if match:
        day, month, year = match.groups()
        normalized = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
        try:
            datetime.strptime(normalized, "%Y-%m-%d")
        except ValueError:
            return None
        return normalized

    return None


def _find_zip_member(zf: zipfile.ZipFile, suffix: str) -> str | None:
    for name in zf.namelist():
        if name.lower().endswith(suffix):
            return name
    return None


def _extract_bb_zip_tif(input_path: Path, temp_dir: Path) -> Path:
    with zipfile.ZipFile(input_path) as zf:
        tif_name = _find_zip_member(zf, ".tif")
        if not tif_name:
            raise RuntimeError(f"No GeoTIFF found in {input_path.name}")
        output_path = temp_dir / Path(tif_name).name
        with zf.open(tif_name) as src, open(output_path, "wb") as dst:
            dst.write(src.read())
    return output_path


def _extract_zip_tifs(input_path: Path, temp_dir: Path) -> list[Path]:
    """Extract all GeoTIFFs from a ZIP to a temp directory."""
    with zipfile.ZipFile(input_path) as zf:
        tif_names = sorted(name for name in zf.namelist() if name.lower().endswith(".tif"))
        if not tif_names:
            raise RuntimeError(f"No GeoTIFFs found in {input_path.name}")
        output_paths: list[Path] = []
        for name in tif_names:
            output_path = temp_dir / Path(name).name
            with zf.open(name) as src, open(output_path, "wb") as dst:
                dst.write(src.read())
            output_paths.append(output_path)
    return output_paths
