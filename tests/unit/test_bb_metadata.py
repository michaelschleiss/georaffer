from pathlib import Path
import zipfile

from georaffer.workers import _extract_bb_meta_year


def test_extract_bb_meta_year_from_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "dop_33382-5775.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dop_33382-5775.tif", b"")
        zf.writestr(
            "dop_33382-5775.xml",
            "<metadata><file_creation_day_year>120/2025</file_creation_day_year></metadata>",
        )

    assert _extract_bb_meta_year(zip_path) == "2025"
