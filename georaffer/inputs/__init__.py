"""Input adapters for loading coordinates from various sources."""

from georaffer.inputs.bbox import load_from_bbox
from georaffer.inputs.csv import load_from_csv
from georaffer.inputs.geotiff import load_from_geotiff
from georaffer.inputs.cvl import load_from_cvl

__all__ = ["load_from_bbox", "load_from_csv", "load_from_geotiff", "load_from_cvl"]

# Pygeon is optional
try:
    from georaffer.inputs.pygeon import load_from_pygeon

    __all__.append("load_from_pygeon")
except ImportError:
    pass
