"""
conftest.py – Shared pytest configuration.

Adds the src/ directory to sys.path so that all tests can import
``rating_classifier`` without installing the package.
"""

import sys
from pathlib import Path

# Insert src/ at the front of the module search path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
