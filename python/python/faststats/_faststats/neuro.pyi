import numpy as np
from numpy.typing import NDArray

class Domain:
    @staticmethod
    def from_volume(shape: tuple[int, int, int], conn: int) -> Domain: ...
    @staticmethod
    def from_mask(mask: NDArray[np.bool_], shape: tuple[int, int, int], conn: int) -> Domain: ...
    @staticmethod
    def from_csr(indptr: NDArray[np.uint32], indices: NDArray[np.uint32]) -> Domain: ...
    @property
    def n_nodes(self) -> int: ...
    @property
    def n_edges(self) -> int: ...
    def indptr(self) -> NDArray[np.uint32]: ...
    def indices(self) -> NDArray[np.uint32]: ...

def tfce_bands(
    domain: Domain,
    stat: NDArray[np.float64],
    thresholds: NDArray[np.float64],
    weights: NDArray[np.float64],
    e: float,
    strict: bool,
) -> NDArray[np.float64]: ...
def tfce_exact(
    domain: Domain, stat: NDArray[np.float64], e: float, h: float, start: float
) -> NDArray[np.float64]: ...
def tfce_one_sample(
    domain: Domain,
    x: NDArray[np.float64],
    n_subj: int,
    e: float,
    h: float,
    start: float,
    step: float,
    weighting: str,
    seed: int,
    n_perm: int,
    threads: int = ...,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]: ...
