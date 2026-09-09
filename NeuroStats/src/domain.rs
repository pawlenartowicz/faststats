//! Adjacency over the nodes of a masked volume or a caller-built graph.
//!
//! Holds the voxel connectivity choice ([`Conn`]: 6, 18 or 26 neighbours) and
//! [`Domain`], the graph the TFCE operators sweep. A domain is backed by one of
//! two representations, and which one a caller picked stays observable. A
//! *lattice* ([`Domain::from_volume`], [`Domain::from_mask`]) keeps `dims`, the
//! connectivity and the in-mask voxel indices, and derives each neighbour row
//! from voxel coordinates on demand — no per-node neighbour list is stored, and
//! the sweep runs in voxel-index space. A *CSR* domain ([`Domain::from_csr`])
//! keeps the caller's `(offsets, neighbours)` arrays as given, which is how a
//! surface mesh or a scipy sparse matrix gets in. Both answer
//! [`Domain::neighbours_into`], [`Domain::to_csr`], [`Domain::n_nodes`] and
//! [`Domain::n_edges`] with the same graph, but `==` compares representations,
//! so a lattice never equals the CSR domain built from its own `to_csr`.
//!
//! The module's tests have no external oracle: lattice rows are checked against
//! a transcription of the plain `dx, dy, dz` triple loop
//! (`lattice_rows_match_transcribed_dxdydz_scan`), and the CSR round trip is
//! checked over every masked-volume fixture in `tests/fixtures/tfce_*.json`
//! (`tests/tfce_oracle.rs::from_csr_round_trips_every_fixture`). The node order
//! itself is what the MNE goldens are generated in, so it is validated wherever
//! the TFCE fixtures are.

use alloc::vec::Vec;

use crate::error::NeuroError;

/// 3-D voxel connectivity — which of the 26 grid neighbours count as adjacent.
/// Matches FSL/`fslmaths -tfce` and MNE `combine_adjacency` conventions:
/// `Face` = 6 (share a face), `Edge` = 18 (face or edge), `Vertex` = 26 (any).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conn {
    /// 6-connectivity: offsets with exactly one non-zero coordinate.
    Face = 6,
    /// 18-connectivity: offsets with one or two non-zero coordinates.
    Edge = 18,
    /// 26-connectivity: every offset in `{-1,0,1}³ \ {0}`.
    Vertex = 26,
}

impl Conn {
    /// Max number of non-zero offset coordinates that still counts as adjacent.
    fn max_nonzero(self) -> u32 {
        match self {
            Self::Face => 1,
            Self::Edge => 2,
            Self::Vertex => 3,
        }
    }
}

/// The `(dx, dy, dz)` steps `conn` counts as adjacent, `dx, dy, dz`
/// lexicographic. This order is the crate's neighbour order: it fixes the
/// column order of [`Domain::to_csr`] and of [`Domain::neighbours_into`], and
/// the TFCE sweep's merge sequence through them. Returns the filled prefix of
/// the buffer.
pub(crate) fn conn_offsets(conn: Conn) -> ([[i32; 3]; 26], usize) {
    let mut out = [[0i32; 3]; 26];
    let mut n = 0usize;
    let max_nz = conn.max_nonzero();
    for dx in -1i32..=1 {
        for dy in -1i32..=1 {
            for dz in -1i32..=1 {
                let nzc = (dx != 0) as u32 + (dy != 0) as u32 + (dz != 0) as u32;
                if nzc == 0 || nzc > max_nz {
                    continue;
                }
                out[n] = [dx, dy, dz];
                n += 1;
            }
        }
    }
    (out, n)
}

/// Calls `f` with the linear voxel index of every in-grid neighbour of the
/// voxel at linear index `lin`, in `off` order. Mask membership is *not*
/// tested — callers that need it filter.
fn for_each_neighbour_lin(
    dims: [usize; 3],
    off: &[[i32; 3]],
    lin: usize,
    mut f: impl FnMut(usize),
) {
    let [nx, ny, nz] = dims;
    let z = lin % nz;
    let t = lin / nz;
    let (y, x) = (t % ny, t / ny);
    for &[dx, dy, dz] in off {
        let (qx, qy, qz) = (
            x as i64 + dx as i64,
            y as i64 + dy as i64,
            z as i64 + dz as i64,
        );
        if qx < 0 || qy < 0 || qz < 0 || qx >= nx as i64 || qy >= ny as i64 || qz >= nz as i64 {
            continue;
        }
        f(((qx as usize) * ny + qy as usize) * nz + qz as usize);
    }
}

/// How a [`Domain`] stores its adjacency. A lattice computes its neighbours
/// from voxel coordinates instead of storing them, which is why the TFCE sweep
/// runs in voxel-index space for it (26 `u32` columns per node dominate the
/// sweep's cache traffic otherwise).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Graph {
    /// Caller-built adjacency: surface meshes, scipy sparse matrices.
    Csr {
        /// CSR row pointers, `len == n_nodes + 1`.
        offsets: Vec<u32>,
        /// CSR columns: node indices in `0..n_nodes`.
        neighbours: Vec<u32>,
    },
    /// 3-D grid over `dims` with `conn` connectivity.
    Lattice {
        /// Grid shape `[nx, ny, nz]`.
        dims: [usize; 3],
        /// Voxel connectivity.
        conn: Conn,
        /// Linear voxel index of node `i`; `len == n_nodes`, strictly increasing.
        lin_of: Vec<u32>,
    },
}

/// Undirected adjacency over the in-mask voxels of a 3-D volume (or over a
/// caller-built graph). Node `i` is the `i`-th in-mask voxel in **row-major
/// (C-order) scan of `dims`** — `linear = (x·dims[1] + y)·dims[2] + z` — the
/// same order as `numpy.ndarray.ravel()` on an array of shape `dims`. Every
/// statistic map passed to the operators must use this node order; the caller
/// owns that alignment.
///
/// Adjacency is symmetric, has no self-loops, and an edge exists only when both
/// endpoints are in-mask.
///
/// Two representations back it: [`from_volume`](Self::from_volume) and
/// [`from_mask`](Self::from_mask) keep the grid and derive neighbours from
/// coordinates; [`from_csr`](Self::from_csr) keeps the caller's neighbour list.
/// The graph is the same either way, but `==` compares representations, so a
/// lattice domain never equals the CSR domain built from its
/// [`to_csr`](Self::to_csr).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Domain {
    n_nodes: usize,
    graph: Graph,
}

impl Domain {
    /// Dense volume: every voxel of `dims` is in-mask. Equivalent to
    /// [`Domain::from_mask`] with an all-`true` mask.
    ///
    /// `dims`: grid shape `[nx, ny, nz]`, each ≥ 1 (a zero dimension yields an
    /// empty domain). Panics only if `nx·ny·nz` exceeds `u32::MAX` voxels.
    pub fn from_volume(dims: [usize; 3], conn: Conn) -> Self {
        let n = dims[0] * dims[1] * dims[2];
        let mask = alloc::vec![true; n];
        Self::from_mask(&mask, dims, conn).expect("all-true mask of the dims product fits u32")
    }

    /// In-mask voxels only — the real case (~10⁴–10⁵ nodes on a 10⁶-voxel grid).
    ///
    /// `mask`: one `bool` per voxel in row-major order of `dims`;
    /// `mask.len()` must equal `dims[0]·dims[1]·dims[2]`
    /// ([`NeuroError::MismatchedLengths`] otherwise). A grid of more than
    /// `u32::MAX` voxels returns [`NeuroError::TooManyNodes`] (checked before
    /// the mask length), which bounds the in-mask node count as well.
    ///
    /// ```
    /// use neurostats::{Conn, Domain};
    /// // 2×2×2 grid, the four voxels with z = 0 in-mask. Row-major scan, so
    /// // the mask entries are (x, y, z) = (0,0,0), (0,0,1), (0,1,0), …
    /// let mask = [true, false, true, false, true, false, true, false];
    /// let dom = Domain::from_mask(&mask, [2, 2, 2], Conn::Face).unwrap();
    /// assert_eq!(dom.n_nodes(), 4);
    /// // Node 0 is voxel (0,0,0); its in-mask face neighbours are (0,1,0) and
    /// // (1,0,0) — nodes 1 and 2. (0,0,1) is out of mask, so no edge to it.
    /// let mut nb = Vec::new();
    /// dom.neighbours_into(0, &mut nb);
    /// nb.sort_unstable();
    /// assert_eq!(nb, [1, 2]);
    /// ```
    pub fn from_mask(mask: &[bool], dims: [usize; 3], conn: Conn) -> Result<Self, NeuroError> {
        let [nx, ny, nz] = dims;
        // A lattice node stores its voxel index as a u32, so the grid must fit
        // even when few of its voxels are in-mask. Checked, not plain `*`:
        // `usize` is 32-bit on wasm32, where the product would wrap silently.
        let n_vox = match nx.checked_mul(ny).and_then(|v| v.checked_mul(nz)) {
            Some(v) if u32::try_from(v).is_ok() => v,
            _ => return Err(NeuroError::TooManyNodes),
        };
        if mask.len() != n_vox {
            return Err(NeuroError::MismatchedLengths {
                expected: n_vox,
                got: mask.len(),
            });
        }
        let mut lin_of: Vec<u32> = Vec::new();
        for (lin, &m) in mask.iter().enumerate() {
            if m {
                lin_of.push(lin as u32);
            }
        }
        Ok(Self {
            n_nodes: lin_of.len(),
            graph: Graph::Lattice { dims, conn, lin_of },
        })
    }

    /// Domain from a caller-built CSR neighbour list — the constructor for
    /// adjacencies that are not a 3-D lattice (surface meshes, a scipy sparse
    /// matrix from Python).
    ///
    /// `n = offsets.len() − 1` (empty `offsets` is an error); `offsets` must be
    /// non-decreasing with `offsets[n] == neighbours.len()`; every neighbour
    /// index `< n`; no self-loops. Symmetry is a documented precondition and is
    /// **not** checked — the TFCE sweep assumes it; the Python
    /// `Domain.from_adjacency` symmetrises before calling this. Node `i`'s
    /// neighbours are `neighbours[offsets[i]..offsets[i+1]]` in the order given.
    ///
    /// Errors: [`NeuroError::InvalidAdjacency`].
    ///
    /// ```
    /// use neurostats::Domain;
    /// // Path graph 0 — 1 — 2, given as both directions of each edge.
    /// let offsets = vec![0, 1, 3, 4];
    /// let neighbours = vec![1, 0, 2, 1];
    /// let dom = Domain::from_csr(offsets, neighbours).unwrap();
    /// assert_eq!(dom.n_nodes(), 3);
    /// assert_eq!(dom.n_edges(), 4); // directed count: each edge twice
    /// let mut nb = Vec::new();
    /// dom.neighbours_into(1, &mut nb);
    /// assert_eq!(nb, [0, 2]);
    /// ```
    pub fn from_csr(offsets: Vec<u32>, neighbours: Vec<u32>) -> Result<Self, NeuroError> {
        let Some((&last, _)) = offsets.split_last() else {
            return Err(NeuroError::InvalidAdjacency);
        };
        let n = offsets.len() - 1;
        if last as usize != neighbours.len() || offsets[0] != 0 {
            return Err(NeuroError::InvalidAdjacency);
        }
        if offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err(NeuroError::InvalidAdjacency);
        }
        for i in 0..n {
            let (lo, hi) = (offsets[i] as usize, offsets[i + 1] as usize);
            for &q in &neighbours[lo..hi] {
                if q as usize >= n || q as usize == i {
                    return Err(NeuroError::InvalidAdjacency);
                }
            }
        }
        Ok(Self {
            n_nodes: n,
            graph: Graph::Csr {
                offsets,
                neighbours,
            },
        })
    }

    pub(crate) fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Number of in-mask nodes `V`.
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Neighbours of node `i` (indices in `0..V`) in fixed offset order,
    /// written into `buf` (cleared first). A `Vec` rather than a fixed array
    /// because a CSR row may be any length.
    ///
    /// Lattice domains decode the neighbour coordinates and map each back to a
    /// node index by binary search, so this costs `O(26 log V)` per node — it
    /// is the readable path (oracles, callers inspecting the graph), not the
    /// one [`crate::tfce()`] sweeps. Panics if `i >= n_nodes()`.
    pub fn neighbours_into(&self, i: usize, buf: &mut Vec<u32>) {
        buf.clear();
        match &self.graph {
            Graph::Csr {
                offsets,
                neighbours,
            } => {
                let (lo, hi) = (offsets[i] as usize, offsets[i + 1] as usize);
                buf.extend_from_slice(&neighbours[lo..hi]);
            }
            Graph::Lattice { dims, conn, lin_of } => {
                let (off, n_off) = conn_offsets(*conn);
                self.lattice_row(i, dims, &off[..n_off], lin_of, buf);
            }
        }
    }

    /// One lattice row: node `i`'s neighbours, in `offsets` order, appended to
    /// `buf` (not cleared — callers own that). Shared by [`Self::neighbours_into`]
    /// (which computes `offsets` fresh) and [`Self::to_csr`] (which computes
    /// them once and loops this per node) so neither recomputes the 26-offset
    /// table per call.
    fn lattice_row(
        &self,
        i: usize,
        dims: &[usize; 3],
        offsets: &[[i32; 3]],
        lin_of: &[u32],
        buf: &mut Vec<u32>,
    ) {
        for_each_neighbour_lin(*dims, offsets, lin_of[i] as usize, |qlin| {
            if let Ok(q) = lin_of.binary_search(&(qlin as u32)) {
                buf.push(q as u32);
            }
        });
    }

    /// The adjacency as a CSR pair `(offsets, neighbours)` — the shape
    /// [`from_csr`](Self::from_csr) takes back, in this crate's neighbour
    /// order. A lattice materialises it (`O(edges · log V)`, allocating
    /// `4·(V + edges)` bytes); a CSR domain clones.
    ///
    /// Errors: [`NeuroError::TooManyNodes`] when the directed edge count does
    /// not fit `u32`.
    pub fn to_csr(&self) -> Result<(Vec<u32>, Vec<u32>), NeuroError> {
        match &self.graph {
            Graph::Csr {
                offsets,
                neighbours,
            } => Ok((offsets.clone(), neighbours.clone())),
            Graph::Lattice { dims, conn, lin_of } => {
                let n = self.n_nodes;
                let mut offsets = Vec::with_capacity(n + 1);
                let mut neighbours = Vec::with_capacity(n * *conn as usize);
                let (off, n_off) = conn_offsets(*conn);
                let mut buf = Vec::new();
                offsets.push(0u32);
                for i in 0..n {
                    buf.clear();
                    self.lattice_row(i, dims, &off[..n_off], lin_of, &mut buf);
                    neighbours.extend_from_slice(&buf);
                    let end =
                        u32::try_from(neighbours.len()).map_err(|_| NeuroError::TooManyNodes)?;
                    offsets.push(end);
                }
                Ok((offsets, neighbours))
            }
        }
    }

    /// Total directed edge count (each undirected edge counted twice).
    ///
    /// Free for a CSR domain; a lattice counts on every call — one pass to
    /// rebuild the mask plus the 26-offset walk, `O(voxels + 26·V)` — because
    /// caching it would need interior mutability and cost `Sync`.
    pub fn n_edges(&self) -> usize {
        match &self.graph {
            Graph::Csr { neighbours, .. } => neighbours.len(),
            Graph::Lattice { dims, conn, lin_of } => {
                let mut in_mask = alloc::vec![false; dims[0] * dims[1] * dims[2]];
                for &l in lin_of {
                    in_mask[l as usize] = true;
                }
                let (off, n_off) = conn_offsets(*conn);
                let mut count = 0usize;
                for &l in lin_of {
                    for_each_neighbour_lin(*dims, &off[..n_off], l as usize, |qlin| {
                        count += in_mask[qlin] as usize;
                    });
                }
                count
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Neighbours of node `i` as a fresh `Vec` — test-only sugar for
    /// [`Domain::neighbours_into`].
    fn nb(d: &Domain, i: usize) -> Vec<u32> {
        let mut v = Vec::new();
        d.neighbours_into(i, &mut v);
        v
    }

    #[test]
    fn cube_neighbour_counts() {
        // 2×2×2: every voxel is a corner. Face: 3 neighbours, Edge: 6, Vertex: 7.
        for (conn, want) in [(Conn::Face, 3), (Conn::Edge, 6), (Conn::Vertex, 7)] {
            let d = Domain::from_volume([2, 2, 2], conn);
            assert_eq!(d.n_nodes(), 8);
            assert_eq!(d.n_edges(), 8 * want);
            for i in 0..8 {
                assert_eq!(nb(&d, i).len(), want, "{conn:?} node {i}");
            }
        }
    }

    #[test]
    fn symmetric_no_self_loops() {
        let dims = [3, 4, 5];
        let mask: Vec<bool> = (0..60).map(|i| i % 3 != 1).collect();
        let d = Domain::from_mask(&mask, dims, Conn::Vertex).unwrap();
        assert_eq!(d.n_nodes(), mask.iter().filter(|&&m| m).count());
        for i in 0..d.n_nodes() {
            for j in nb(&d, i) {
                assert_ne!(j as usize, i);
                assert!(nb(&d, j as usize).contains(&(i as u32)));
            }
        }
    }

    #[test]
    fn mask_length_checked() {
        let e = Domain::from_mask(&[true; 5], [2, 2, 2], Conn::Face).unwrap_err();
        assert_eq!(
            e,
            NeuroError::MismatchedLengths {
                expected: 8,
                got: 5
            }
        );
    }

    /// A grid whose voxel count overflows `u32` is rejected even when the mask
    /// that would go with it is tiny — the voxel index is what must fit.
    #[test]
    fn oversized_grid_rejected() {
        let e = Domain::from_mask(&[true], [65_536, 65_536, 2], Conn::Face).unwrap_err();
        assert_eq!(e, NeuroError::TooManyNodes);
    }

    #[test]
    fn row_major_node_order() {
        // Node 1 is voxel (0,0,1); with Face conn its neighbours are (0,0,0)=0,
        // (0,1,1)=3, (1,0,1)=5 on a 2×2×2 grid.
        let d = Domain::from_volume([2, 2, 2], Conn::Face);
        let mut n = nb(&d, 1);
        n.sort_unstable();
        assert_eq!(n, alloc::vec![0, 3, 5]);
    }

    /// `neighbours_into` on a lattice reproduces, row for row, the
    /// `dx, dy, dz` scan the CSR constructor used to run. Transcribed here so
    /// the check is against the old code, not against `to_csr`, which shares
    /// the new decoding.
    #[test]
    fn lattice_rows_match_transcribed_dxdydz_scan() {
        fn reference(mask: &[bool], dims: [usize; 3], conn: Conn) -> Vec<Vec<u32>> {
            let [nx, ny, nz] = dims;
            let mut node_of = alloc::vec![u32::MAX; nx * ny * nz];
            let mut n_nodes = 0u32;
            for (lin, &m) in mask.iter().enumerate() {
                if m {
                    node_of[lin] = n_nodes;
                    n_nodes += 1;
                }
            }
            let max_nz = conn.max_nonzero();
            let mut rows = Vec::new();
            for x in 0..nx {
                for y in 0..ny {
                    for z in 0..nz {
                        let lin = (x * ny + y) * nz + z;
                        if !mask[lin] {
                            continue;
                        }
                        let mut row = Vec::new();
                        for dx in -1i64..=1 {
                            for dy in -1i64..=1 {
                                for dz in -1i64..=1 {
                                    let nzc =
                                        (dx != 0) as u32 + (dy != 0) as u32 + (dz != 0) as u32;
                                    if nzc == 0 || nzc > max_nz {
                                        continue;
                                    }
                                    let (qx, qy, qz) =
                                        (x as i64 + dx, y as i64 + dy, z as i64 + dz);
                                    if qx < 0
                                        || qy < 0
                                        || qz < 0
                                        || qx >= nx as i64
                                        || qy >= ny as i64
                                        || qz >= nz as i64
                                    {
                                        continue;
                                    }
                                    let qlin =
                                        ((qx as usize) * ny + qy as usize) * nz + qz as usize;
                                    let q = node_of[qlin];
                                    if q != u32::MAX {
                                        row.push(q);
                                    }
                                }
                            }
                        }
                        rows.push(row);
                    }
                }
            }
            rows
        }

        let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for dims in [[2, 2, 2], [3, 4, 5], [1, 7, 2], [5, 5, 5]] {
            let n_vox = dims[0] * dims[1] * dims[2];
            for fill in [1.0, 0.6] {
                let mask: Vec<bool> = (0..n_vox).map(|_| next() < fill).collect();
                for conn in [Conn::Face, Conn::Edge, Conn::Vertex] {
                    let d = Domain::from_mask(&mask, dims, conn).unwrap();
                    let want = reference(&mask, dims, conn);
                    assert_eq!(d.n_nodes(), want.len());
                    for (i, row) in want.iter().enumerate() {
                        assert_eq!(&nb(&d, i), row, "{dims:?} {conn:?} fill {fill} node {i}");
                    }
                    assert_eq!(d.n_edges(), want.iter().map(Vec::len).sum::<usize>());
                }
            }
        }
    }

    #[test]
    fn from_csr_round_trips_from_mask() {
        let dims = [3, 4, 5];
        let mask: Vec<bool> = (0..60).map(|i| i % 3 != 1).collect();
        let d = Domain::from_mask(&mask, dims, Conn::Edge).unwrap();
        let (offsets, neighbours) = d.to_csr().unwrap();
        let back = Domain::from_csr(offsets, neighbours).unwrap();
        // The two carry different representations, so compare the graphs.
        assert_eq!(back.to_csr().unwrap(), d.to_csr().unwrap());
        assert_eq!(back.n_nodes(), d.n_nodes());
        assert_eq!(back.n_edges(), d.n_edges());
    }

    #[test]
    fn from_csr_rejects_bad_input() {
        assert_eq!(
            Domain::from_csr(alloc::vec![], alloc::vec![]).unwrap_err(),
            NeuroError::InvalidAdjacency
        );
        // non-monotone offsets
        assert_eq!(
            Domain::from_csr(alloc::vec![0, 2, 1], alloc::vec![1, 0]).unwrap_err(),
            NeuroError::InvalidAdjacency
        );
        // offsets[n] != neighbours.len()
        assert_eq!(
            Domain::from_csr(alloc::vec![0, 1, 3], alloc::vec![1, 0]).unwrap_err(),
            NeuroError::InvalidAdjacency
        );
        // index out of range
        assert_eq!(
            Domain::from_csr(alloc::vec![0, 1, 2], alloc::vec![1, 5]).unwrap_err(),
            NeuroError::InvalidAdjacency
        );
        // self-loop
        assert_eq!(
            Domain::from_csr(alloc::vec![0, 1, 2], alloc::vec![0, 0]).unwrap_err(),
            NeuroError::InvalidAdjacency
        );
        // empty graph with one node is fine
        let d = Domain::from_csr(alloc::vec![0, 0], alloc::vec![]).unwrap();
        assert_eq!(d.n_nodes(), 1);
        assert_eq!(d.n_edges(), 0);
    }
}
