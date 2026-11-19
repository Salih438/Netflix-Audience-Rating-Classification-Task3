"""
rating_classifier – Netflix audience rating classification package.

Modules
-------
config      : project-wide constants and paths
data        : data loading and label mapping
features    : feature engineering
models      : model factory and training routines
evaluate    : metrics and cross-validation helpers
visualize   : plotting utilities
pipeline    : end-to-end orchestration
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rating_classifier")
except PackageNotFoundError:
    __version__ = "0.1.0"
