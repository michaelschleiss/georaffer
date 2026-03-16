"""Tests for irregular LAZ rasterization backend."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from georaffer.config import Region
from georaffer.converters.laz_irregular import (
    build_irregular_pdal_pipeline,
    convert_laz_irregular,
    ensure_pdal_available,
)


class TestEnsurePdalAvailable:
    def test_raises_when_missing(self, monkeypatch):
        monkeypatch.setattr("georaffer.converters.laz_irregular.shutil.which", lambda _: None)
        with pytest.raises(RuntimeError, match="PDAL"):
            ensure_pdal_available()


class TestBuildIrregularPdalPipeline:
    def test_contains_expected_stages(self):
        pipeline = build_irregular_pdal_pipeline(
            "input.laz",
            "output.tif",
            native_resolution=1.0,
        )
        stages = pipeline["pipeline"]
        assert stages[0]["type"] == "readers.las"
        assert stages[1]["type"] == "filters.delaunay"
        assert stages[2]["type"] == "filters.faceraster"
        assert stages[2]["resolution"] == 1.0
        assert stages[3]["type"] == "writers.raster"

    def test_includes_default_srs_when_provided(self):
        pipeline = build_irregular_pdal_pipeline(
            "input.laz",
            "output.tif",
            native_resolution=1.0,
            default_srs="EPSG:3045",
        )
        assert pipeline["pipeline"][0]["default_srs"] == "EPSG:3045"


class TestConvertLazIrregular:
    def test_rasterizes_then_reuses_raster_backend(self, tmp_path, monkeypatch):
        calls = {}

        def fake_run(_pdal_path, pipeline, cwd=None):
            calls["pipeline"] = pipeline
            output_path = Path(pipeline["pipeline"][-1]["filename"])
            data = np.arange(4, dtype=np.float32).reshape((2, 2))
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=2,
                width=2,
                count=1,
                dtype="float32",
                crs="EPSG:3045",
                transform=from_origin(418000, 5554000, 1, 1),
            ) as dst:
                dst.write(data, 1)

        def fake_convert_dsm_raster(
            input_path,
            output_paths,
            region,
            year=None,
            target_sizes=None,
            num_threads=None,
            grid_size_km=1.0,
            profiling=False,
        ):
            calls["input_path"] = input_path
            calls["output_paths"] = output_paths
            calls["region"] = region
            calls["year"] = year
            calls["target_sizes"] = target_sizes
            calls["grid_size_km"] = grid_size_km
            return True

        monkeypatch.setattr(
            "georaffer.converters.laz_irregular.ensure_pdal_available",
            lambda: "/usr/bin/pdal",
        )
        monkeypatch.setattr("georaffer.converters.laz_irregular.run_pdal_pipeline", fake_run)
        monkeypatch.setattr(
            "georaffer.converters.laz_irregular.convert_dsm_raster",
            fake_convert_dsm_raster,
        )

        input_path = tmp_path / "dmp1g_418_5552_2013.laz"
        input_path.touch()
        output_paths = {2000: str(tmp_path / "out.tif")}

        ok = convert_laz_irregular(
            str(input_path),
            output_paths,
            Region.CZ,
            year="2013",
            native_resolution=1.0,
            default_srs="EPSG:3045",
            target_sizes=[2000],
            grid_size_km=1.0,
        )

        assert ok is True
        assert calls["region"] == Region.CZ
        assert calls["year"] == "2013"
        assert calls["target_sizes"] == [2000]
        assert calls["grid_size_km"] == 1.0
        assert calls["output_paths"] == output_paths
        assert calls["input_path"].endswith("dmp1g_418_5552_2013.tif")
        assert calls["pipeline"]["pipeline"][0]["default_srs"] == "EPSG:3045"
