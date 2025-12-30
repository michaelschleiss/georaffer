"""Tests for base downloader class."""

import json
import os
from datetime import datetime, timedelta
from io import BytesIO
from unittest.mock import MagicMock, Mock, patch

import pytest

from georaffer.downloaders.base import Catalog, RegionDownloader


class ConcreteDownloader(RegionDownloader):
    """Concrete implementation for testing abstract base class."""

    def utm_to_grid_coords(self, utm_x, utm_y):
        return (int(utm_x // 1000), int(utm_y // 1000)), (int(utm_x // 1000), int(utm_y // 1000))

    def _load_catalog(self) -> Catalog:
        return Catalog()


class TestRegionDownloaderInit:
    """Tests for downloader initialization."""

    def test_init_creates_directories(self, tmp_path):
        """Test that output directories are set correctly."""
        downloader = ConcreteDownloader("TEST", str(tmp_path))

        assert downloader.region_name == "TEST"
        assert downloader.raw_dir == tmp_path / "raw"
        assert downloader.processed_dir == tmp_path / "processed"

    def test_init_accepts_custom_session(self, tmp_path):
        """Test session injection for testing."""
        mock_session = Mock()
        downloader = ConcreteDownloader("TEST", str(tmp_path), session=mock_session)

        assert downloader.session is mock_session

    def test_init_creates_default_session(self, tmp_path):
        """Test default session creation."""
        downloader = ConcreteDownloader("TEST", str(tmp_path))

        assert downloader.session is not None


class TestDownloadFile:
    """Tests for download_file method."""

    @pytest.fixture
    def downloader(self, tmp_path):
        mock_session = Mock()
        return ConcreteDownloader("TEST", str(tmp_path), session=mock_session)

    def test_download_success(self, downloader, tmp_path):
        """Test successful file download."""
        # Mock response with context manager support
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_response.headers = {"content-length": "5000"}
        mock_response.iter_content.return_value = [b"x" * 5000]
        mock_response.raise_for_status = Mock()
        downloader.session.get.return_value = mock_response

        # Mock integrity check
        with patch.object(downloader, "_verify_file_integrity", return_value=True):
            output_path = str(tmp_path / "output" / "test.jp2")
            result = downloader.download_file("http://example.com/test.jp2", output_path)

        assert result is True
        assert os.path.exists(output_path)

    def test_download_retries_on_small_file(self, downloader, tmp_path):
        """Test retry when file is too small."""
        # First response: too small, second: valid
        small_response = MagicMock()
        small_response.__enter__.return_value = small_response
        small_response.__exit__.return_value = False
        small_response.headers = {"content-length": "100"}
        small_response.iter_content.return_value = [b"x" * 100]
        small_response.raise_for_status = Mock()

        valid_response = MagicMock()
        valid_response.__enter__.return_value = valid_response
        valid_response.__exit__.return_value = False
        valid_response.headers = {"content-length": "5000"}
        valid_response.iter_content.return_value = [b"x" * 5000]
        valid_response.raise_for_status = Mock()

        downloader.session.get.side_effect = [small_response, valid_response]

        with patch.object(downloader, "_verify_file_integrity", return_value=True):
            with patch("georaffer.downloaders.base.time.sleep"):  # Skip delays
                output_path = str(tmp_path / "output" / "test.jp2")
                result = downloader.download_file("http://example.com/test.jp2", output_path)

        assert result is True
        assert downloader.session.get.call_count == 2

    def test_download_retries_on_integrity_failure(self, downloader, tmp_path):
        """Test retry when integrity check fails."""
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_response.headers = {"content-length": "5000"}
        mock_response.iter_content.return_value = [b"x" * 5000]
        mock_response.raise_for_status = Mock()
        downloader.session.get.return_value = mock_response

        # First call fails integrity, second succeeds
        with patch.object(downloader, "_verify_file_integrity", side_effect=[False, True]):
            with patch("georaffer.downloaders.base.time.sleep"):
                output_path = str(tmp_path / "output" / "test.jp2")
                result = downloader.download_file("http://example.com/test.jp2", output_path)

        assert result is True

    def test_download_fails_after_max_retries(self, downloader, tmp_path):
        """Test failure after exhausting retries raises RuntimeError."""
        import requests

        downloader.session.get.side_effect = requests.RequestException("Network error")

        with patch("georaffer.downloaders.base.time.sleep"):
            with patch("georaffer.downloaders.base.MAX_RETRIES", 3):
                output_path = str(tmp_path / "output" / "test.jp2")
                with pytest.raises(RuntimeError, match="Download failed after 3 retries"):
                    downloader.download_file("http://example.com/test.jp2", output_path)

    def test_download_atomic_write(self, downloader, tmp_path):
        """Test that download uses atomic write (temp file + rename)."""
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_response.headers = {"content-length": "5000"}
        mock_response.iter_content.return_value = [b"x" * 5000]
        mock_response.raise_for_status = Mock()
        downloader.session.get.return_value = mock_response

        with patch.object(downloader, "_verify_file_integrity", return_value=True):
            output_path = str(tmp_path / "output" / "test.jp2")
            result = downloader.download_file("http://example.com/test.jp2", output_path)

        # Temp file should not exist after successful download
        assert not os.path.exists(output_path + ".tmp")
        assert os.path.exists(output_path)


class TestVerifyFileIntegrity:
    """Tests for _verify_file_integrity method."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return ConcreteDownloader("TEST", str(tmp_path))

    def test_verify_jp2_success(self, downloader):
        """Test JP2 verification with valid image."""
        with patch("georaffer.downloaders.base.Image.open") as mock_open:
            mock_img = Mock()
            mock_img.size = (1000, 1000)
            mock_open.return_value = mock_img

            buffer = BytesIO(b"fake jp2 data")
            result = downloader._verify_file_integrity(buffer, "test.jp2")

        assert result is True

    def test_verify_jp2_invalid_dimensions(self, downloader):
        """Test JP2 verification fails with zero dimensions."""
        with patch("georaffer.downloaders.base.Image.open") as mock_open:
            mock_img = Mock()
            mock_img.size = (0, 1000)
            mock_open.return_value = mock_img

            buffer = BytesIO(b"fake jp2 data")
            result = downloader._verify_file_integrity(buffer, "test.jp2")

        assert result is False

    def test_verify_jp2_corrupt(self, downloader):
        """Test JP2 verification fails with corrupt file."""
        with patch("georaffer.downloaders.base.Image.open") as mock_open:
            mock_open.side_effect = Exception("Invalid image")

            buffer = BytesIO(b"corrupt data")
            result = downloader._verify_file_integrity(buffer, "test.jp2")

        assert result is False

    def test_verify_laz_success(self, downloader):
        """Test LAZ verification with valid file."""
        with patch("georaffer.downloaders.base.laspy.open") as mock_open:
            mock_laz = MagicMock()
            mock_laz.__enter__.return_value = mock_laz
            mock_laz.__exit__ = Mock(return_value=False)
            mock_laz.header.point_count = 10000
            mock_laz.chunk_iterator.return_value = iter([[1, 2, 3]])
            mock_open.return_value = mock_laz

            buffer = BytesIO(b"fake laz data")
            result = downloader._verify_file_integrity(buffer, "test.laz")

        assert result is True

    def test_verify_laz_zero_points(self, downloader):
        """Test LAZ verification fails with zero points."""
        with patch("georaffer.downloaders.base.laspy.open") as mock_open:
            mock_laz = MagicMock()
            mock_laz.__enter__.return_value = mock_laz
            mock_laz.__exit__ = Mock(return_value=False)
            mock_laz.header.point_count = 0
            mock_open.return_value = mock_laz

            buffer = BytesIO(b"fake laz data")
            result = downloader._verify_file_integrity(buffer, "test.laz")

        assert result is False

    def test_verify_unknown_extension(self, downloader):
        """Test unknown extension returns True (no validation)."""
        buffer = BytesIO(b"some data")
        result = downloader._verify_file_integrity(buffer, "test.unknown")

        assert result is True


class TestCatalog:
    """Tests for Catalog dataclass."""

    def test_catalog_creation(self):
        """Test creating a catalog with tiles."""
        tiles = {
            (350, 5600): {
                2020: {"url": "http://example.com/2020.jp2", "acquisition_date": None},
                2021: {"url": "http://example.com/2021.jp2", "acquisition_date": None},
            },
            (351, 5600): {2020: {"url": "http://example.com/351_2020.jp2", "acquisition_date": None}},
        }
        catalog = Catalog(image_tiles=tiles)

        assert len(catalog.image_tiles) == 2
        assert catalog.image_tiles[(350, 5600)][2020]["url"] == "http://example.com/2020.jp2"
        assert isinstance(catalog.created_at, datetime)

    def test_catalog_is_stale_fresh(self):
        """Test is_stale returns False for fresh catalog."""
        catalog = Catalog(image_tiles={}, created_at=datetime.now())

        assert catalog.is_stale(ttl_days=30) is False

    def test_catalog_is_stale_expired(self):
        """Test is_stale returns True for expired catalog."""
        old_time = datetime.now() - timedelta(days=31)
        catalog = Catalog(image_tiles={}, created_at=old_time)

        assert catalog.is_stale(ttl_days=30) is True

    def test_catalog_to_dict(self):
        """Test serialization to JSON-compatible dict."""
        tiles = {
            (350, 5600): {2020: {"url": "http://example.com/2020.jp2", "acquisition_date": None}},
        }
        dsm = {
            (350, 5600): {"url": "http://example.com/dsm.laz", "acquisition_date": None},
        }
        created = datetime(2025, 1, 15, 10, 30, 0)
        catalog = Catalog(image_tiles=tiles, dsm_tiles=dsm, created_at=created)

        result = catalog.to_dict()

        assert result["created_at"] == "2025-01-15T10:30:00"
        assert "350,5600" in result["image_tiles"]
        assert result["image_tiles"]["350,5600"]["2020"]["url"] == "http://example.com/2020.jp2"
        assert "350,5600" in result["dsm_tiles"]
        assert result["dsm_tiles"]["350,5600"]["url"] == "http://example.com/dsm.laz"

    def test_catalog_from_dict(self):
        """Test deserialization from JSON-compatible dict."""
        data = {
            "created_at": "2025-01-15T10:30:00",
            "image_tiles": {
                "350,5600": {"2020": {"url": "http://example.com/2020.jp2", "acquisition_date": None}},
                "351,5601": {
                    "2019": {"url": "http://example.com/2019.jp2", "acquisition_date": "2019-06-15"},
                    "2020": {"url": "http://example.com/2020.jp2", "acquisition_date": None},
                },
            },
            "dsm_tiles": {
                "350,5600": {"url": "http://example.com/dsm.laz", "acquisition_date": None},
            },
        }

        catalog = Catalog.from_dict(data)

        assert catalog.created_at == datetime(2025, 1, 15, 10, 30, 0)
        assert len(catalog.image_tiles) == 2
        assert catalog.image_tiles[(350, 5600)][2020]["url"] == "http://example.com/2020.jp2"
        assert catalog.image_tiles[(351, 5601)][2019]["url"] == "http://example.com/2019.jp2"
        assert catalog.image_tiles[(351, 5601)][2019]["acquisition_date"] == "2019-06-15"
        assert len(catalog.dsm_tiles) == 1
        assert catalog.dsm_tiles[(350, 5600)]["url"] == "http://example.com/dsm.laz"

    def test_catalog_roundtrip(self):
        """Test serialization roundtrip preserves data."""
        tiles = {
            (350, 5600): {
                2020: {"url": "http://example.com/2020.jp2", "acquisition_date": None},
                2021: {"url": "http://example.com/2021.jp2", "acquisition_date": "2021-07-01"},
            },
            (351, 5601): {2019: {"url": "http://example.com/2019.jp2", "acquisition_date": None}},
        }
        dsm = {
            (350, 5600): {"url": "http://example.com/dsm.laz", "acquisition_date": None},
        }
        original = Catalog(image_tiles=tiles, dsm_tiles=dsm, created_at=datetime(2025, 1, 15, 10, 30, 0))

        restored = Catalog.from_dict(original.to_dict())

        assert restored.created_at == original.created_at
        assert restored.image_tiles == original.image_tiles
        assert restored.dsm_tiles == original.dsm_tiles


class TestFetchCatalog:
    """Tests for build_catalog and cache methods."""

    @pytest.fixture
    def downloader(self, tmp_path):
        mock_session = Mock()
        dl = ConcreteDownloader("TEST", str(tmp_path), session=mock_session)
        dl._cache_path = tmp_path / "cache" / "test_catalog.json"
        return dl

    def test_build_catalog_returns_instance_cache(self, downloader):
        """Test that instance cache is returned without disk access."""
        cached = Catalog(image_tiles={(350, 5600): {2020: {"url": "http://cached.com", "acquisition_date": None}}})
        downloader._catalog = cached

        result = downloader.build_catalog()

        assert result is cached

    def test_build_catalog_reads_disk_cache(self, downloader, tmp_path):
        """Test that fresh disk cache is loaded."""
        cache_data = {
            "created_at": datetime.now().isoformat(),
            "image_tiles": {"350,5600": {"2020": {"url": "http://disk.com", "acquisition_date": None}}},
            "dsm_tiles": {},
        }
        cache_path = tmp_path / "cache" / "test_catalog.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        result = downloader.build_catalog()

        assert result.image_tiles[(350, 5600)][2020]["url"] == "http://disk.com"

    def test_build_catalog_ignores_stale_disk_cache(self, downloader, tmp_path):
        """Test that stale disk cache triggers reload."""
        old_time = datetime.now() - timedelta(days=60)
        cache_data = {
            "created_at": old_time.isoformat(),
            "image_tiles": {"350,5600": {"2020": {"url": "http://stale.com", "acquisition_date": None}}},
            "dsm_tiles": {},
        }
        cache_path = tmp_path / "cache" / "test_catalog.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        result = downloader.build_catalog()

        # Default _load_catalog returns empty catalog
        assert result.image_tiles == {}

    def test_build_catalog_refresh_bypasses_cache(self, downloader, tmp_path):
        """Test that refresh=True bypasses all caches."""
        # Set instance cache
        downloader._catalog = Catalog(image_tiles={(350, 5600): {2020: {"url": "http://instance.com", "acquisition_date": None}}})

        # Set disk cache
        cache_data = {
            "created_at": datetime.now().isoformat(),
            "image_tiles": {"350,5600": {"2020": {"url": "http://disk.com", "acquisition_date": None}}},
            "dsm_tiles": {},
        }
        cache_path = tmp_path / "cache" / "test_catalog.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        result = downloader.build_catalog(refresh=True)

        # Default _load_catalog returns empty catalog
        assert result.image_tiles == {}

    def test_build_catalog_writes_to_disk(self, downloader, tmp_path):
        """Test that loaded catalog is persisted to disk."""
        result = downloader.build_catalog()

        cache_path = tmp_path / "cache" / "test_catalog.json"
        assert cache_path.exists()

        with open(cache_path) as f:
            data = json.load(f)
        assert "created_at" in data
        assert "image_tiles" in data

    def test_read_cache_returns_none_for_missing_file(self, downloader):
        """Test _read_cache returns None when file doesn't exist."""
        result = downloader._read_cache()

        assert result is None

    def test_read_cache_returns_none_for_invalid_json(self, downloader, tmp_path):
        """Test _read_cache returns None for corrupted file."""
        cache_path = tmp_path / "cache" / "test_catalog.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            f.write("not valid json")

        result = downloader._read_cache()

        assert result is None

    def test_write_cache_creates_parent_dirs(self, downloader, tmp_path):
        """Test _write_cache creates parent directories."""
        downloader._catalog = Catalog(image_tiles={(350, 5600): {2020: {"url": "http://test.com", "acquisition_date": None}}})
        downloader._cache_path = tmp_path / "deep" / "nested" / "cache.json"

        downloader._write_cache()

        assert downloader._cache_path.exists()

    def test_write_cache_noop_without_cache_path(self, downloader):
        """Test _write_cache does nothing when _cache_path is None."""
        downloader._cache_path = None
        downloader._catalog = Catalog(image_tiles={(350, 5600): {2020: {"url": "http://test.com", "acquisition_date": None}}})

        # Should not raise
        downloader._write_cache()
