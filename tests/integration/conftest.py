"""Apply the integration marker to every test under tests/integration."""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(session, config, items):
    """Auto-apply the integration marker to tests living in tests/integration."""
    integration_dir = Path(__file__).parent.resolve()
    for item in items:
        try:
            item_path = Path(item.fspath).resolve()
        except (TypeError, OSError, AttributeError):
            continue

        if integration_dir in item_path.parents:
            item.add_marker(pytest.mark.integration)
