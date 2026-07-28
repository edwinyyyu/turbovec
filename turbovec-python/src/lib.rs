use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyType;

fn not_contiguous_err(kind: &str) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!(
        "{kind} must be C-contiguous; call np.ascontiguousarray(...) first",
    ))
}

#[pyclass]
struct TurboQuantIndex {
    inner: turbovec_core::TurboQuantIndex,
}

#[pymethods]
impl TurboQuantIndex {
    /// Construct an index. `dim` is optional: when omitted, the
    /// underlying quantized index is created lazily on the first
    /// `add` call, picking up the dimensionality from the input
    /// array's shape.
    #[new]
    #[pyo3(signature = (dim=None, bit_width=4))]
    fn new(dim: Option<usize>, bit_width: usize) -> PyResult<Self> {
        let inner = match dim {
            Some(d) => turbovec_core::TurboQuantIndex::new(d, bit_width),
            None => turbovec_core::TurboQuantIndex::new_lazy(bit_width),
        }
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn add(&mut self, vectors: PyReadonlyArray2<f32>) -> PyResult<()> {
        let arr = vectors.as_array();
        let dim = arr.ncols();
        let slice = arr.as_slice().ok_or_else(|| not_contiguous_err("vectors"))?;
        // `add_2d` handles both eager (dim must match) and lazy (locks
        // dim on first call) cases.
        self.inner
            .add_2d(slice, dim)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Run a top-`k` search against the index.
    ///
    /// `mask`, when given, is a bool array of length `len(self)`. Only slots
    /// with `mask[i] == True` contribute to the returned top-`k`. The
    /// returned result count per query is `min(k, mask.sum())`.
    #[pyo3(signature = (queries, k, *, mask=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        mask: Option<PyReadonlyArray1<bool>>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i64>>)> {
        let arr = queries.as_array();
        let nq = arr.nrows();
        let q_slice = arr.as_slice().ok_or_else(|| not_contiguous_err("queries"))?;
        // Reject wrong-dim queries cleanly. Previously the inner
        // `assert_eq!(queries.len(), nq * dim)` would fire as a Rust
        // panic and surface to Python as a PanicException, not the
        // ValueError users expect for input-shape mismatch.
        if let Some(idx_dim) = self.inner.dim_opt() {
            if arr.ncols() != idx_dim {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "query dim {} does not match index dim {}",
                    arr.ncols(),
                    idx_dim,
                )));
            }
        }

        let mask_arr = mask.as_ref().map(|m| m.as_array());
        let mask_slice: Option<&[bool]> = match mask_arr.as_ref() {
            Some(m_arr) => {
                let expected = self.inner.len();
                if m_arr.len() != expected {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "mask length {} does not match index size {}",
                        m_arr.len(),
                        expected,
                    )));
                }
                Some(m_arr.as_slice().ok_or_else(|| not_contiguous_err("mask"))?)
            }
            None => None,
        };

        let results = self.inner.search_with_mask(q_slice, k, mask_slice);
        let effective_k = results.k;

        let scores = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), results.scores)
            .unwrap()
            .into_pyarray(py);
        let indices = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), results.indices)
            .unwrap()
            .into_pyarray(py);

        Ok((scores, indices))
    }

    fn write(&self, path: &str) -> PyResult<()> {
        self.inner.write(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("{}", e))
        })
    }

    #[classmethod]
    fn load(_cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        let inner = turbovec_core::TurboQuantIndex::load(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("{}", e))
        })?;
        Ok(Self { inner })
    }

    /// Warm up the search caches (rotation matrix, Lloyd-Max centroids,
    /// SIMD-blocked code layout) so the first `search` call does not pay
    /// the one-time initialisation cost.
    fn prepare(&self) {
        self.inner.prepare();
    }

    /// Remove the vector at `idx` in O(1) by swapping with the last vector.
    ///
    /// The last vector moves into the deleted slot — order is not
    /// preserved. Returns the old index of the moved vector; equals `idx`
    /// when `idx` was already the last element.
    ///
    /// Raises ``IndexError`` if ``idx`` is out of range.
    fn swap_remove(&mut self, idx: usize) -> PyResult<usize> {
        let len = self.inner.len();
        if idx >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {idx} out of range for index of length {len}",
            )));
        }
        Ok(self.inner.swap_remove(idx))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        let dim = self
            .inner
            .dim_opt()
            .map_or_else(|| "None".to_string(), |d| d.to_string());
        format!(
            "turbovec.TurboQuantIndex(dim={}, bit_width={}, n_vectors={})",
            dim,
            self.inner.bit_width(),
            self.inner.len()
        )
    }

    /// Vector dimensionality. Returns ``None`` when the index was
    /// constructed lazily (no ``dim=``) and hasn't seen an add yet;
    /// otherwise an ``int``.
    #[getter]
    fn dim(&self) -> Option<usize> {
        self.inner.dim_opt()
    }

    #[getter]
    fn bit_width(&self) -> usize {
        self.inner.bit_width()
    }
}

#[pyclass]
struct IdMapIndex {
    inner: turbovec_core::IdMapIndex,
}

#[pymethods]
impl IdMapIndex {
    /// Construct an id-mapped index. `dim` is optional: when omitted,
    /// the underlying quantized index is created lazily on the first
    /// `add_with_ids` call, picking up dim from the input array shape.
    #[new]
    #[pyo3(signature = (dim=None, bit_width=4))]
    fn new(dim: Option<usize>, bit_width: usize) -> PyResult<Self> {
        let inner = match dim {
            Some(d) => turbovec_core::IdMapIndex::new(d, bit_width),
            None => turbovec_core::IdMapIndex::new_lazy(bit_width),
        }
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Add `n = vectors.shape[0]` vectors with the given external `ids`.
    ///
    /// `ids` must be a 1-D array of `uint64` with length equal to
    /// `vectors.shape[0]`. Raises `ValueError` if any id is already
    /// present or if the lengths don't match. On a lazy index, this
    /// call commits the dimensionality from `vectors.shape[1]`.
    fn add_with_ids(
        &mut self,
        vectors: PyReadonlyArray2<f32>,
        ids: PyReadonlyArray1<u64>,
    ) -> PyResult<()> {
        let v = vectors.as_array();
        let dim = v.ncols();
        let v_slice = v.as_slice().ok_or_else(|| not_contiguous_err("vectors"))?;
        let i = ids.as_array();
        let i_slice = i.as_slice().ok_or_else(|| not_contiguous_err("ids"))?;
        self.inner
            .add_with_ids_2d(v_slice, dim, i_slice)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Remove the vector with external id `id`. Returns `True` if it was
    /// present, `False` otherwise.
    fn remove(&mut self, id: u64) -> bool {
        self.inner.remove(id)
    }

    /// Search for the top-`k` nearest external ids for each query.
    ///
    /// `allowlist`, when given, is a `uint64` array of external ids; the
    /// returned top-`k` is restricted to ids in this list. The returned
    /// result count per query is `min(k, len(allowlist))` (after
    /// de-duplication).
    ///
    /// Returns `(scores, ids)` as `(nq, effective_k)` arrays, `ids` typed
    /// `uint64`. Raises `ValueError` for an empty allowlist and `KeyError`
    /// if any allowlist id is not present in the index.
    #[pyo3(signature = (queries, k, *, allowlist=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        allowlist: Option<PyReadonlyArray1<u64>>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u64>>)> {
        let arr = queries.as_array();
        let nq = arr.nrows();
        let q_slice = arr.as_slice().ok_or_else(|| not_contiguous_err("queries"))?;
        if let Some(idx_dim) = self.inner.dim_opt() {
            if arr.ncols() != idx_dim {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "query dim {} does not match index dim {}",
                    arr.ncols(),
                    idx_dim,
                )));
            }
        }

        let allow_arr = allowlist.as_ref().map(|a| a.as_array());
        let allow_slice: Option<&[u64]> = match allow_arr.as_ref() {
            Some(a_arr) => {
                if a_arr.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "allowlist is empty",
                    ));
                }
                let slice = a_arr.as_slice().ok_or_else(|| not_contiguous_err("allowlist"))?;
                let mut unknown: Vec<u64> = Vec::new();
                for &id in slice {
                    if !self.inner.contains(id) {
                        if unknown.len() < 5 {
                            unknown.push(id);
                        } else {
                            unknown.push(id);
                            break;
                        }
                    }
                }
                if !unknown.is_empty() {
                    let preview: Vec<u64> = unknown.iter().take(5).copied().collect();
                    return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                        "allowlist contains id(s) not present in index: {:?}{}",
                        preview,
                        if unknown.len() > 5 { ", ..." } else { "" },
                    )));
                }
                Some(slice)
            }
            None => None,
        };

        let (scores, ids) = self.inner.search_with_allowlist(q_slice, k, allow_slice);
        // For empty queries (nq=0), match TurboQuantIndex's shape
        // contract: effective_k is `min(k, n_vectors, n_allowed)`. The
        // kernel dedups the allowlist via a packed bool mask for nq>0,
        // so we have to dedup here too — otherwise `allowlist=[1, 1, 1]`
        // returns shape `(0, 3)` for empty queries but `(N, 1)` for
        // non-empty queries, a silent shape divergence.
        let effective_k = if nq == 0 {
            let n_allowed = match allow_slice {
                Some(s) => {
                    let mut seen: std::collections::HashSet<u64> =
                        std::collections::HashSet::with_capacity(s.len());
                    s.iter().filter(|id| seen.insert(**id)).count()
                }
                None => self.inner.len(),
            };
            k.min(self.inner.len()).min(n_allowed)
        } else {
            scores.len() / nq
        };

        let scores_arr = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), scores)
            .unwrap()
            .into_pyarray(py);
        let ids_arr = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), ids)
            .unwrap()
            .into_pyarray(py);
        Ok((scores_arr, ids_arr))
    }

    fn contains(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    fn prepare(&self) {
        self.inner.prepare();
    }

    /// Serialize the index and id-map side-tables to a `.tvim` file.
    fn write(&self, path: &str) -> PyResult<()> {
        self.inner.write(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("{}", e))
        })
    }

    /// Load an `IdMapIndex` from a `.tvim` file previously written by
    /// [`IdMapIndex.write`].
    #[classmethod]
    fn load(_cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        let inner = turbovec_core::IdMapIndex::load(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("{}", e))
        })?;
        Ok(Self { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        let dim = self
            .inner
            .dim_opt()
            .map_or_else(|| "None".to_string(), |d| d.to_string());
        format!(
            "turbovec.IdMapIndex(dim={}, bit_width={}, n_vectors={})",
            dim,
            self.inner.bit_width(),
            self.inner.len()
        )
    }

    fn __contains__(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    /// Vector dimensionality. Returns ``None`` when the index was
    /// constructed lazily and hasn't seen an add yet; otherwise ``int``.
    #[getter]
    fn dim(&self) -> Option<usize> {
        self.inner.dim_opt()
    }

    #[getter]
    fn bit_width(&self) -> usize {
        self.inner.bit_width()
    }
}

#[pyclass]
struct DiskIndex {
    inner: turbovec_core::DiskIndex,
}

#[pymethods]
impl DiskIndex {
    /// Construct an empty disk-primary index with no backing file yet.
    /// All vectors live in the in-RAM delta until the first `write`.
    /// `dim` is optional: when omitted, the dimensionality is committed
    /// on the first `add_with_ids` call.
    ///
    /// `target_partition_size`, when given, enables SPFresh-style
    /// partitioning at the next `write`: codes are clustered into
    /// partitions of roughly that many vectors, and searches probe only
    /// the `nprobe` nearest partitions (approximate routing).
    ///
    /// `store_vectors=True` keeps the full-precision vectors alongside the
    /// quantized codes (the delta's in RAM, the base's in a mmap section),
    /// enabling exact rescoring (on by default at depth `4 * k` once set)
    /// and `get_vectors`. Costs `4 * dim` bytes per row of file size;
    /// resident memory is unaffected. Fixed for the life of the index.
    ///
    /// `replica_epsilon`, when given, enables SPANN-style boundary
    /// multi-assignment on a partitioned index: at each `write`, a vector
    /// is also stored in every partition whose centroid is within
    /// `(1 + replica_epsilon)` of its nearest centroid's distance
    /// (RNG-rule pruned, at most 8 copies), so boundary vectors are
    /// findable from adjacent partitions at small probe counts.
    #[new]
    #[pyo3(signature = (dim=None, bit_width=4, target_partition_size=None, store_vectors=false, replica_epsilon=None))]
    fn new(
        dim: Option<usize>,
        bit_width: usize,
        target_partition_size: Option<usize>,
        store_vectors: bool,
        replica_epsilon: Option<f32>,
    ) -> PyResult<Self> {
        let mut inner = turbovec_core::DiskIndex::new(dim, bit_width)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        if target_partition_size == Some(0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_partition_size must be positive",
            ));
        }
        if let Some(epsilon) = replica_epsilon {
            if !(epsilon.is_finite() && epsilon > 0.0) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "replica_epsilon must be finite and positive",
                ));
            }
        }
        inner.set_partitioning(target_partition_size);
        inner.set_replication(replica_epsilon);
        inner.set_store_vectors(store_vectors);
        Ok(Self { inner })
    }

    /// Open a `.tvdm` file previously produced by `write`. The file is
    /// memory-mapped: searches run directly over the mapped bytes and the
    /// index stays resident only through the OS page cache.
    #[classmethod]
    fn open(_cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        let inner = turbovec_core::DiskIndex::open(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
        Ok(Self { inner })
    }

    /// Convert a `.tvim` `IdMapIndex` file into a `.tvdm` file at `dst`.
    /// Lossless — codes, scales, calibration and ids carry over, so search
    /// results are identical. `dst` may equal `src` to convert in place.
    #[classmethod]
    fn convert_id_map_file(_cls: &Bound<PyType>, src: &str, dst: &str) -> PyResult<()> {
        turbovec_core::DiskIndex::convert_id_map_file(src, dst)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))
    }

    /// Convert a `.tvdm` file back into a `.tvim` `IdMapIndex` file at
    /// `dst` — the inverse of `convert_id_map_file`, equally lossless.
    /// `dst` may equal `src` to convert in place.
    #[classmethod]
    fn convert_to_id_map_file(_cls: &Bound<PyType>, src: &str, dst: &str) -> PyResult<()> {
        turbovec_core::DiskIndex::convert_to_id_map_file(src, dst)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))
    }

    /// Add `n = vectors.shape[0]` vectors with the given external `ids`
    /// into the in-RAM delta. Same contract as `IdMapIndex.add_with_ids`.
    fn add_with_ids(
        &mut self,
        vectors: PyReadonlyArray2<f32>,
        ids: PyReadonlyArray1<u64>,
    ) -> PyResult<()> {
        let v = vectors.as_array();
        let dim = v.ncols();
        let v_slice = v.as_slice().ok_or_else(|| not_contiguous_err("vectors"))?;
        let i = ids.as_array();
        let i_slice = i.as_slice().ok_or_else(|| not_contiguous_err("ids"))?;
        self.inner
            .add_with_ids_2d(v_slice, dim, i_slice)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Remove the vector with external id `id`. Returns `True` if it was
    /// present. Base-segment removals are tombstoned in RAM and physically
    /// dropped at the next `write`.
    fn remove(&mut self, id: u64) -> bool {
        self.inner.remove(id)
    }

    /// Search for the top-`k` nearest external ids for each query.
    ///
    /// On a partitioned index, `nprobe` caps how many partitions each
    /// query scans (default `max(4, nlist / 8)`, or `nlist` when
    /// `probe_epsilon` is given); it is ignored on a flat index, where the
    /// whole segment is scanned.
    ///
    /// `probe_epsilon` enables distance-bounded adaptive probing: each
    /// query scans every partition whose centroid distance is within
    /// `(1 + probe_epsilon)` of its nearest centroid's, up to the `nprobe`
    /// cap. Boundary queries probe more partitions, confident ones fewer.
    ///
    /// `rescore_k` controls exact rescoring on an index built with
    /// `store_vectors`: the merged top `rescore_k` quantized candidates
    /// are re-ranked by exact f32 inner product (returned scores for those
    /// rows are the exact products). `None` = `4 * k` when vectors are
    /// stored, off otherwise; `0` = off.
    ///
    /// Returns `(scores, ids)` as `(nq, effective_k)` arrays with
    /// `effective_k = min(k, len(self))`, `ids` typed `uint64`.
    #[pyo3(signature = (queries, k, *, nprobe=None, probe_epsilon=None, rescore_k=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        nprobe: Option<usize>,
        probe_epsilon: Option<f32>,
        rescore_k: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u64>>)> {
        let arr = queries.as_array();
        let nq = arr.nrows();
        let q_slice = arr.as_slice().ok_or_else(|| not_contiguous_err("queries"))?;
        if let Some(idx_dim) = self.inner.dim_opt() {
            if arr.ncols() != idx_dim {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "query dim {} does not match index dim {}",
                    arr.ncols(),
                    idx_dim,
                )));
            }
        }
        if nprobe == Some(0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "nprobe must be positive",
            ));
        }
        if let Some(epsilon) = probe_epsilon {
            if !(epsilon.is_finite() && epsilon >= 0.0) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "probe_epsilon must be finite and non-negative",
                ));
            }
        }
        if matches!(rescore_k, Some(r) if r > 0) && !self.inner.stores_vectors() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rescore_k requires an index built with store_vectors=True",
            ));
        }

        // Copy the queries so the scan can run without the GIL (concurrent
        // Python threads keep working during a long search).
        let owned_queries = q_slice.to_vec();
        let index = &self.inner;
        let (scores, ids) = py.allow_threads(move || {
            index.search_with_options(
                &owned_queries,
                k,
                turbovec_core::SearchOptions {
                    nprobe,
                    probe_epsilon,
                    rescore_k,
                },
            )
        });
        let effective_k = if nq == 0 {
            k.min(self.inner.len())
        } else {
            scores.len() / nq
        };

        let scores_arr = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), scores)
            .unwrap()
            .into_pyarray(py);
        let ids_arr = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), ids)
            .unwrap()
            .into_pyarray(py);
        Ok((scores_arr, ids_arr))
    }

    fn contains(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    /// The stored full-precision vectors of the given live ids, as an
    /// `(len(ids), dim)` float32 array. Requires an index built with
    /// `store_vectors=True`; raises `KeyError` if any id is not present.
    fn get_vectors<'py>(
        &self,
        py: Python<'py>,
        ids: PyReadonlyArray1<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if !self.inner.stores_vectors() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "get_vectors requires an index built with store_vectors=True",
            ));
        }
        let ids_arr = ids.as_array();
        let ids_slice = ids_arr.as_slice().ok_or_else(|| not_contiguous_err("ids"))?;
        let dim = self.inner.dim_opt().unwrap_or(0);
        let mut out = Vec::with_capacity(ids_slice.len() * dim);
        for &id in ids_slice {
            let vector = self.inner.get_vector(id).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "id {id} is not present in the index",
                ))
            })?;
            out.extend_from_slice(&vector);
        }
        Ok(
            numpy::ndarray::Array2::from_shape_vec((ids_slice.len(), dim), out)
                .unwrap()
                .into_pyarray(py),
        )
    }

    /// Warm the query-side caches (rotation matrix, centroids, delta
    /// layout). Cheap; does not fault in the mmap-backed codes.
    fn prepare(&self) {
        self.inner.prepare();
    }

    /// Compact to `path`: stream the base segment (minus tombstones) and
    /// the in-RAM delta into a fresh `.tvdm` file, atomically replace it,
    /// re-map it as the new base, and empty the delta and tombstones.
    fn write(&mut self, path: &str) -> PyResult<()> {
        self.inner
            .write(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __contains__(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    fn __repr__(&self) -> String {
        let dim = self
            .inner
            .dim_opt()
            .map_or_else(|| "None".to_string(), |d| d.to_string());
        format!(
            "turbovec.DiskIndex(dim={}, bit_width={}, n_vectors={}, base={}, delta={}, tombstones={}, nlist={})",
            dim,
            self.inner.bit_width(),
            self.inner.len(),
            self.inner.base_len(),
            self.inner.delta_len(),
            self.inner.tombstone_count(),
            self.inner.nlist(),
        )
    }

    /// Vector dimensionality. ``None`` until a dim is committed.
    #[getter]
    fn dim(&self) -> Option<usize> {
        self.inner.dim_opt()
    }

    #[getter]
    fn bit_width(&self) -> usize {
        self.inner.bit_width()
    }

    /// Vectors in the mmap-backed base segment (including tombstoned).
    #[getter]
    fn base_len(&self) -> usize {
        self.inner.base_len()
    }

    /// Vectors in the in-RAM delta (added since the last `write`).
    #[getter]
    fn delta_len(&self) -> usize {
        self.inner.delta_len()
    }

    /// Base-segment ids hidden by tombstones (removed since last `write`).
    #[getter]
    fn tombstone_count(&self) -> usize {
        self.inner.tombstone_count()
    }

    /// Number of partitions in the base segment (1 = flat).
    #[getter]
    fn nlist(&self) -> usize {
        self.inner.nlist()
    }

    /// Partitioning target, or ``None`` when flat. Settable; takes effect
    /// at the next `write`.
    #[getter]
    fn target_partition_size(&self) -> Option<usize> {
        self.inner.partition_target()
    }

    #[setter]
    fn set_target_partition_size(&mut self, target: Option<usize>) -> PyResult<()> {
        if target == Some(0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_partition_size must be positive",
            ));
        }
        self.inner.set_partitioning(target);
        Ok(())
    }

    /// Boundary multi-assignment epsilon, or ``None`` when off. Settable;
    /// takes effect at the next `write`.
    #[getter]
    fn replica_epsilon(&self) -> Option<f32> {
        self.inner.replica_epsilon()
    }

    #[setter]
    fn set_replica_epsilon(&mut self, epsilon: Option<f32>) -> PyResult<()> {
        if let Some(e) = epsilon {
            if !(e.is_finite() && e > 0.0) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "replica_epsilon must be finite and positive",
                ));
            }
        }
        self.inner.set_replication(epsilon);
        Ok(())
    }

    /// True if the index keeps full-precision vectors (exact rescoring and
    /// `get_vectors` available).
    #[getter]
    fn store_vectors(&self) -> bool {
        self.inner.stores_vectors()
    }

    /// Closure-assignment replica rows in the base segment. Diagnostic.
    #[getter]
    fn replica_count(&self) -> usize {
        self.inner.base_replica_count()
    }

    /// Backing `.tvdm` file path, or ``None`` before the first write/open.
    #[getter]
    fn path(&self) -> Option<String> {
        self.inner
            .path()
            .map(|p| p.to_string_lossy().into_owned())
    }
}

#[pyclass]
struct FreshIndex {
    inner: turbovec_core::FreshIndex,
}

#[pymethods]
impl FreshIndex {
    /// Construct an empty incrementally-updatable index with no backing
    /// directory yet. Vectors live in RAM until the first `save(directory)`,
    /// which binds the directory, makes the index durable (write-ahead
    /// log), and appends new vectors to per-partition segment files —
    /// saves rewrite only the partitions they touch, never the whole index.
    ///
    /// Knobs are the same as `DiskIndex`: `target_partition_size` enables
    /// SPFresh partitioning (with local split/merge/reassign maintenance at
    /// each save), `store_vectors` keeps full-precision vectors for exact
    /// rescoring and `get_vectors`, `replica_epsilon` enables boundary
    /// multi-assignment.
    #[new]
    #[pyo3(signature = (dim=None, bit_width=4, target_partition_size=None, store_vectors=false, replica_epsilon=None))]
    fn new(
        dim: Option<usize>,
        bit_width: usize,
        target_partition_size: Option<usize>,
        store_vectors: bool,
        replica_epsilon: Option<f32>,
    ) -> PyResult<Self> {
        let mut inner = turbovec_core::FreshIndex::new(dim, bit_width)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        if target_partition_size == Some(0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_partition_size must be positive",
            ));
        }
        if let Some(epsilon) = replica_epsilon {
            if !(epsilon.is_finite() && epsilon > 0.0) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "replica_epsilon must be finite and positive",
                ));
            }
        }
        inner.set_partitioning(target_partition_size);
        inner.set_replication(replica_epsilon);
        inner.set_store_vectors(store_vectors);
        Ok(Self { inner })
    }

    /// Open an index directory previously produced by `save`. Replays the
    /// write-ahead log (mutations since the last save survive crashes) and
    /// cleans up after any interrupted save.
    #[classmethod]
    fn open(_cls: &Bound<PyType>, directory: &str) -> PyResult<Self> {
        let inner = turbovec_core::FreshIndex::open(directory)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
        Ok(Self { inner })
    }

    /// Build a FreshIndex directory from a `.tvdm` `DiskIndex` file.
    /// Lossless: codes, calibration, ids, partitioning, replicas and stored
    /// vectors carry over unchanged.
    #[classmethod]
    fn import_disk_index_file(
        _cls: &Bound<PyType>,
        src: &str,
        directory: &str,
    ) -> PyResult<Self> {
        let inner = turbovec_core::FreshIndex::import_disk_index_file(src, directory)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
        Ok(Self { inner })
    }

    /// Write the live vectors' codes to a `.tvim` `IdMapIndex` file
    /// (requires a saved, fully-flushed index).
    fn export_id_map_file(&self, dst: &str) -> PyResult<()> {
        self.inner
            .export_id_map_file(dst)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))
    }

    /// Add vectors with external ids (buffered in RAM + write-ahead log;
    /// appended to partitions at the next `save`).
    fn add_with_ids(
        &mut self,
        vectors: PyReadonlyArray2<f32>,
        ids: PyReadonlyArray1<u64>,
    ) -> PyResult<()> {
        let v = vectors.as_array();
        let dim = v.ncols();
        let v_slice = v.as_slice().ok_or_else(|| not_contiguous_err("vectors"))?;
        let i = ids.as_array();
        let i_slice = i.as_slice().ok_or_else(|| not_contiguous_err("ids"))?;
        self.inner
            .add_with_ids_2d(v_slice, dim, i_slice)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Remove the vector with external id `id`; every stored copy is
    /// hidden immediately and reclaimed at a later save. Returns `True`
    /// if the id was present.
    fn remove(&mut self, id: u64) -> bool {
        self.inner.remove(id)
    }

    /// Search for the top-`k` nearest external ids per query. Knobs as in
    /// `DiskIndex.search`. Releases the GIL for the scan.
    #[pyo3(signature = (queries, k, *, nprobe=None, probe_epsilon=None, rescore_k=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        nprobe: Option<usize>,
        probe_epsilon: Option<f32>,
        rescore_k: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u64>>)> {
        let arr = queries.as_array();
        let nq = arr.nrows();
        let q_slice = arr.as_slice().ok_or_else(|| not_contiguous_err("queries"))?;
        if let Some(idx_dim) = self.inner.dim_opt() {
            if arr.ncols() != idx_dim {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "query dim {} does not match index dim {}",
                    arr.ncols(),
                    idx_dim,
                )));
            }
        }
        if nprobe == Some(0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "nprobe must be positive",
            ));
        }
        if let Some(epsilon) = probe_epsilon {
            if !(epsilon.is_finite() && epsilon >= 0.0) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "probe_epsilon must be finite and non-negative",
                ));
            }
        }
        if matches!(rescore_k, Some(r) if r > 0) && !self.inner.stores_vectors() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rescore_k requires an index built with store_vectors=True",
            ));
        }

        let owned_queries = q_slice.to_vec();
        let index = &self.inner;
        let (scores, ids) = py.allow_threads(move || {
            index.search_with_options(
                &owned_queries,
                k,
                turbovec_core::SearchOptions {
                    nprobe,
                    probe_epsilon,
                    rescore_k,
                },
            )
        });
        let effective_k = if nq == 0 {
            k.min(self.inner.len())
        } else {
            scores.len() / nq
        };
        let scores_arr = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), scores)
            .unwrap()
            .into_pyarray(py);
        let ids_arr = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), ids)
            .unwrap()
            .into_pyarray(py);
        Ok((scores_arr, ids_arr))
    }

    fn contains(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    /// The stored full-precision vectors of the given live ids (requires
    /// `store_vectors=True`); raises `KeyError` if any id is not present.
    fn get_vectors<'py>(
        &self,
        py: Python<'py>,
        ids: PyReadonlyArray1<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if !self.inner.stores_vectors() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "get_vectors requires an index built with store_vectors=True",
            ));
        }
        let ids_arr = ids.as_array();
        let ids_slice = ids_arr.as_slice().ok_or_else(|| not_contiguous_err("ids"))?;
        let dim = self.inner.dim_opt().unwrap_or(0);
        let mut out = Vec::with_capacity(ids_slice.len() * dim);
        for &id in ids_slice {
            let vector = self.inner.get_vector(id).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "id {id} is not present in the index",
                ))
            })?;
            out.extend_from_slice(&vector);
        }
        Ok(
            numpy::ndarray::Array2::from_shape_vec((ids_slice.len(), dim), out)
                .unwrap()
                .into_pyarray(py),
        )
    }

    fn prepare(&self) {
        self.inner.prepare();
    }

    /// Flush to `directory` (bound on first call): buffered vectors are
    /// appended to their partitions, local maintenance runs, and a new
    /// manifest is atomically published. Only touched partitions' files
    /// change — the page cache for the rest stays valid.
    fn save(&mut self, directory: &str) -> PyResult<()> {
        self.inner
            .save(directory)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __contains__(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    fn __repr__(&self) -> String {
        let dim = self
            .inner
            .dim_opt()
            .map_or_else(|| "None".to_string(), |d| d.to_string());
        format!(
            "turbovec.FreshIndex(dim={}, bit_width={}, n_vectors={}, nlist={}, \
             memtable={}, dead={}, replicas={}, runs={}, chunks={})",
            dim,
            self.inner.bit_width(),
            self.inner.len(),
            self.inner.nlist(),
            self.inner.memtable_len(),
            self.inner.dead_count(),
            self.inner.replica_count(),
            self.inner.run_count(),
            self.inner.chunk_count(),
        )
    }

    #[getter]
    fn dim(&self) -> Option<usize> {
        self.inner.dim_opt()
    }

    #[getter]
    fn bit_width(&self) -> usize {
        self.inner.bit_width()
    }

    #[getter]
    fn nlist(&self) -> usize {
        self.inner.nlist()
    }

    #[getter]
    fn target_partition_size(&self) -> Option<usize> {
        self.inner.partition_target()
    }

    #[setter]
    fn set_target_partition_size(&mut self, target: Option<usize>) -> PyResult<()> {
        if target == Some(0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_partition_size must be positive",
            ));
        }
        self.inner.set_partitioning(target);
        Ok(())
    }

    #[getter]
    fn replica_epsilon(&self) -> Option<f32> {
        self.inner.replica_epsilon()
    }

    #[setter]
    fn set_replica_epsilon(&mut self, epsilon: Option<f32>) -> PyResult<()> {
        if let Some(e) = epsilon {
            if !(e.is_finite() && e > 0.0) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "replica_epsilon must be finite and positive",
                ));
            }
        }
        self.inner.set_replication(epsilon);
        Ok(())
    }

    #[getter]
    fn store_vectors(&self) -> bool {
        self.inner.stores_vectors()
    }

    /// Rows buffered in RAM since the last save. Diagnostic.
    #[getter]
    fn memtable_len(&self) -> usize {
        self.inner.memtable_len()
    }

    /// Dead rows awaiting compaction. Diagnostic.
    #[getter]
    fn dead_count(&self) -> usize {
        self.inner.dead_count()
    }

    /// Live closure-assignment replica rows. Diagnostic.
    #[getter]
    fn replica_count(&self) -> usize {
        self.inner.replica_count()
    }

    /// Id-run tables. Diagnostic.
    #[getter]
    fn run_count(&self) -> usize {
        self.inner.run_count()
    }

    /// Total chunks across all segment files. Diagnostic.
    #[getter]
    fn chunk_count(&self) -> usize {
        self.inner.chunk_count()
    }

    /// Backing directory, or ``None`` before the first save.
    #[getter]
    fn path(&self) -> Option<String> {
        self.inner
            .path()
            .map(|p| p.to_string_lossy().into_owned())
    }
}

#[pymodule]
fn _turbovec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TurboQuantIndex>()?;
    m.add_class::<IdMapIndex>()?;
    m.add_class::<DiskIndex>()?;
    m.add_class::<FreshIndex>()?;
    Ok(())
}