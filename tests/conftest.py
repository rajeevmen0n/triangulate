import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip `llm` tests when no Anthropic key is available."""
    from triangulate.config import get_settings

    if os.environ.get("ANTHROPIC_API_KEY") or get_settings().anthropic_api_key:
        return
    skip = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)
