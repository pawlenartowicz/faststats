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

def tfce(
    domain: Domain,
    stat: NDArray[np.float64],
    e: float,
    h: float,
    start: float | None,
    step: float | None,
    weighting: str,
    tail: str,
) -> NDArray[np.float64]: ...
def tfce_bands(
    domain: Domain,
    stat: NDArray[np.float64],
    thresholds: NDArray[np.float64],
    weights: NDArray[np.float64],
    e: float,
    strict: bool,
) -> NDArray[np.float64]: ...

def tfce_ols(
    domain: Domain,
    y: NDArray[np.float32] | NDArray[np.float64],
    rows: NDArray[np.uint64] | None,
    tested: NDArray[np.float64] | None,
    confounds: NDArray[np.float64] | None,
    e: float,
    h: float,
    start: float | None,
    step: float | None,
    weighting: str,
    tail: str,
    seed: int,
    n_perm: int,
    threads: int,
    extend: tuple[NDArray[np.float64], NDArray[np.uint32], NDArray[np.float64], int, int] | None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.uint32],
    int,
    int,
]: ...
