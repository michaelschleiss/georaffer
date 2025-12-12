from pathlib import Path


def test_outputs_exist_uses_laz_header_year(monkeypatch, tmp_path: Path) -> None:
    from georaffer.conversion import _outputs_exist

    # RLP LAZ filenames do not include the year; conversion derives it from the LAS header.
    laz_filename = "bdom20rgbi_32_364_5582_2_rp.laz"
    raw_dsm_dir = tmp_path / "raw" / "dsm"
    raw_dsm_dir.mkdir(parents=True)
    (raw_dsm_dir / laz_filename).write_bytes(b"")  # placeholder, header read is mocked

    monkeypatch.setattr("georaffer.conversion.get_laz_year", lambda _p: "2023")

    processed_dir = tmp_path / "processed"
    out_dir = processed_dir / "dsm" / "5000"
    out_dir.mkdir(parents=True)
    # Expected output name for RLP at (364km, 5582km) with year from header
    (out_dir / "rlp_32_364000_5582000_2023.tif").write_bytes(b"")

    assert (
        _outputs_exist(
            laz_filename,
            str(processed_dir),
            "dsm",
            [5000],
            grid_size_km=2.0,  # no split for RLP native 2km tiles
            source_dir=str(raw_dsm_dir),
        )
        is True
    )

