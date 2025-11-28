"""Apply the unit marker to every test under tests/unit."""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(session, config, items):
    """Auto-apply the unit marker to tests living in tests/unit."""
    unit_dir = Path(__file__).parent.resolve()
    for item in items:
        # Only mark tests that live under the unit directory
        try:
            item_path = Path(item.fspath).resolve()
        except (TypeError, OSError, AttributeError):
            continue

        if unit_dir in item_path.parents:
            item.add_marker(pytest.mark.unit)
