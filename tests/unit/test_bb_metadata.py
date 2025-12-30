import zipfile
from pathlib import Path

from georaffer.workers import _extract_zip_meta_year


def test_extract_zip_meta_year_from_zip(tmp_path: Path) -> None:
    """Test extraction of year from BB/BW ZIP metadata."""
    zip_path = tmp_path / "dop_33382-5775.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dop_33382-5775.tif", b"")
        zf.writestr(
            "dop_33382-5775.html",
            "<html><td>Bildflugdatum:</td><td>2025-04-27</td></html>",
        )

    assert _extract_zip_meta_year(zip_path) == "2025"
