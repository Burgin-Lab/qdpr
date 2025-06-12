"""
Unit and regression test for the qdpr package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import qdpr


def test_qdpr_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "qdpr" in sys.modules
