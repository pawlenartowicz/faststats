"""neurostats — TFCE and permutation inference over masked volumes and graphs.

Build a :class:`Domain` once and reuse it across maps and inferences.
"""

from importlib.metadata import version as _version

from ._inference import TfceInference, tfce_ols
from ._io import MaskedData, export_to, import_from
from ._tfce import Domain, grid_and_weights, tfce

__version__ = _version("neurostats")
__all__ = [
    "Domain",
    "MaskedData",
    "TfceInference",
    "tfce",
    "tfce_ols",
    "import_from",
    "export_to",
    "grid_and_weights",
    "__version__",
]
