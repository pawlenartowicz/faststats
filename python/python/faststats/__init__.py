"""faststats — Rust-backed statistics for Python.

Submodules: ``faststats.neuro`` (TFCE, one-sample permutation inference).
"""

from importlib.metadata import version as _version

from . import neuro

__version__ = _version("faststats")
__all__ = ["neuro", "__version__"]
