"""Tests for CZ (Czech Republic) downloader."""

import shutil
import zipfile
from pathlib import Path

import pytest

from georaffer.downloaders.base import RegionDownloader
from georaffer.downloaders.cz import CZDownloader


def _atom_feed(*titles: str) -> bytes:
    entries = []
    for title in titles:
        entries.append(
            f"""
            <entry>
              <title>{title}</title>
            </entry>
            """
        )
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      {''.join(entries)}
    </feed>
    """
    return xml.encode("utf-8")


class TestCZDownloaderInit:
    def test_init_creates_downloader(self, tmp_path):
        downloader = CZDownloader(str(tmp_path))
        assert downloader.region_name == "CZ"


class TestCZUtmToGridCoords:
    @pytest.fixture
    def downloader(self, tmp_path):
        return CZDownloader(str(tmp_path))

    def test_maps_to_2km_tile_origin(self, downloader):
        image_coords, dsm_coords = downloader.utm_to_grid_coords(418500, 5553500)
        assert image_coords == (418, 5552)
        assert dsm_coords == (418, 5552)


class TestCZFilenames:
    @pytest.fixture
    def downloader(self, tmp_path):
        return CZDownloader(str(tmp_path))

    def test_image_filename_from_url(self, downloader):
        url = "https://openzu.cuzk.gov.cz/opendata/OI/418_5552.zip"
        assert downloader.image_filename_from_url(url) == "oi_418_5552_2024.jp2"

    def test_dmpok_filename_from_url(self, downloader):
        url = "https://openzu.cuzk.gov.cz/opendata/DMPOK-TIFF/epsg-3045/418_5552.zip"
        assert downloader.dsm_filename_from_url(url) == "dmpok_418_5552_2025.tif"

    def test_dmp1g_filename_from_url(self, downloader):
        url = "https://openzu.cuzk.cz/opendata/DMP1G/epsg-3045/418_5552.zip"
        assert downloader.dsm_filename_from_url(url) == "dmp1g_418_5552_2013.laz"


class TestCZCatalog:
    def test_catalog_loads_imagery_and_dsm_fallback(self, tmp_path, monkeypatch):
        downloader = CZDownloader(str(tmp_path))

        oi_root = _atom_feed(
            "INSPIRE Ortofotosnimky (OI) - dlazdice: 418_5552",
            "INSPIRE Ortofotosnimky (OI) - dlazdice: 420_5552",
        )
        dmp1g_root = _atom_feed(
            "DMP 1G - mapovy list: 418_5552",
            "DMP 1G - mapovy list: 420_5552",
        )
        dmpok_root = _atom_feed("DMPOK - mapovy list: 418_5552")

        def fake_fetch(session, url, timeout=None, wrap_content=False):
            if url == CZDownloader.OI_FEED_URL:
                import xml.etree.ElementTree as ET

                return ET.fromstring(oi_root)
            if url == CZDownloader.DMP1G_FEED_URL:
                import xml.etree.ElementTree as ET

                return ET.fromstring(dmp1g_root)
            if url == CZDownloader.DMPOK_FEED_URL:
                import xml.etree.ElementTree as ET

                return ET.fromstring(dmpok_root)
            raise AssertionError(url)

        monkeypatch.setattr("georaffer.downloaders.cz.fetch_xml_feed", fake_fetch)

        catalog = downloader._load_catalog()

        assert set(catalog.image_tiles) == {(418, 5552), (420, 5552)}
        assert set(catalog.dsm_tiles) == {(418, 5552), (420, 5552)}
        assert set(catalog.dsm_tiles[(418, 5552)]) == {2013, 2025}
        assert set(catalog.dsm_tiles[(420, 5552)]) == {2013}


class TestCZDownloadExtraction:
    @pytest.fixture
    def downloader(self, tmp_path):
        return CZDownloader(str(tmp_path))

    def test_download_file_extracts_tif_payload(self, downloader, tmp_path, monkeypatch):
        sample_zip = tmp_path / "sample.zip"
        with zipfile.ZipFile(sample_zip, "w") as zf:
            zf.writestr("418_5552.tif", b"tif-data")
            zf.writestr("418_5552.tfw", b"0.5")

        def fake_download(self, url, output_path, on_progress=None):
            shutil.copyfile(sample_zip, output_path)
            return True

        monkeypatch.setattr(RegionDownloader, "download_file", fake_download)

        output_path = tmp_path / "out.tif"
        downloader.download_file(
            "https://openzu.cuzk.gov.cz/opendata/DMPOK-TIFF/epsg-3045/418_5552.zip",
            str(output_path),
        )

        assert output_path.read_bytes() == b"tif-data"
        assert not output_path.with_suffix(".zip").exists()

    def test_download_file_extracts_laz_payload(self, downloader, tmp_path, monkeypatch):
        sample_zip = tmp_path / "sample_dmp1g.zip"
        with zipfile.ZipFile(sample_zip, "w") as zf:
            zf.writestr("418_5552.laz", b"laz-data")

        def fake_download(self, url, output_path, on_progress=None):
            shutil.copyfile(sample_zip, output_path)
            return True

        monkeypatch.setattr(RegionDownloader, "download_file", fake_download)

        output_path = tmp_path / "out.laz"
        downloader.download_file(
            "https://openzu.cuzk.cz/opendata/DMP1G/epsg-3045/418_5552.zip",
            str(output_path),
        )

        assert output_path.read_bytes() == b"laz-data"
        assert not output_path.with_suffix(".zip").exists()

    def test_download_file_extracts_jp2_payload(self, downloader, tmp_path, monkeypatch):
        sample_zip = tmp_path / "sample_oi.zip"
        with zipfile.ZipFile(sample_zip, "w") as zf:
            zf.writestr("418_5552.jp2", b"jp2-data")
            zf.writestr("418_5552.gml", b"<gml />")

        def fake_download(self, url, output_path, on_progress=None):
            shutil.copyfile(sample_zip, output_path)
            return True

        monkeypatch.setattr(RegionDownloader, "download_file", fake_download)

        output_path = tmp_path / "out.jp2"
        downloader.download_file(
            "https://openzu.cuzk.gov.cz/opendata/OI/418_5552.zip",
            str(output_path),
        )

        assert output_path.read_bytes() == b"jp2-data"
        assert not output_path.with_suffix(".zip").exists()
