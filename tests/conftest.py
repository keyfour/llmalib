"""Shared pytest fixtures for the adapter test suite."""

import pytest

from llmalib._testing import set_caller


@pytest.fixture(autouse=True)
def _reset_caller_seam():
    """Guarantee a clean caller seam around every test (global state)."""
    set_caller(None)
    yield
    set_caller(None)
