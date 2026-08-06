// `PyArrayMethods` / `PyUntypedArrayMethods` are needed for `data()` and
// `is_c_contiguous()` / `len()` in `mask_bytes`. Without the latter in
// scope, `len()` silently resolves to `PyAnyMethods::len` instead.
use numpy::{
    IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};

mod par_copy;
use par_copy::{par_copy_into, PAR_COPY_MIN_LEN};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyType};

fn not_contiguous_err(kind: &str) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!(
        "{kind} must be C-contiguous; call np.ascontiguousarray(...) first",
    ))
}

/// Name of a Python object's type, for error messages.
fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string())
}

/// Describe an argument for an array-mismatch error: "2-D float64" for
/// anything ndarray-like, otherwise the plain type name (e.g. "list").
fn array_desc(obj: &Bound<'_, PyAny>) -> String {
    let ndim = obj
        .getattr("ndim")
        .ok()
        .and_then(|n| n.extract::<usize>().ok());
    let dtype = obj
        .getattr("dtype")
        .ok()
        .and_then(|d| d.str().ok())
        .map(|s| s.to_string());
    match (ndim, dtype) {
        (Some(n), Some(d)) => format!("{n}-D {d}"),
        _ => type_name(obj),
    }
}

fn array_type_err(name: &str, expected: &str, obj: &Bound<'_, PyAny>) -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(format!(
        "{name} must be a {expected} array, got {}",
        array_desc(obj),
    ))
}

/// Extract a 2-D float32 array, replacing pyo3's opaque downcast error
/// ("'ndarray' object cannot be cast as 'ndarray'") with one that names
/// the argument and states expected vs got dtype/ndim.
fn extract_f32_2d<'py>(
    name: &str,
    obj: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray2<'py, f32>> {
    obj.extract()
        .map_err(|_| array_type_err(name, "2-D float32", obj))
}

/// Extract a 1-D uint64 array; see [`extract_f32_2d`].
fn extract_u64_1d<'py>(
    name: &str,
    obj: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray1<'py, u64>> {
    obj.extract()
        .map_err(|_| array_type_err(name, "1-D uint64", obj))
}

/// Extract a 1-D bool array; see [`extract_f32_2d`].
///
/// **The elements of the returned array must not be read as `bool`.**
/// Building this wrapper only validates the dtype and registers a
/// borrow; it does not touch the buffer. Use [`mask_bytes`] to read it —
/// see there for why.
fn extract_bool_1d<'py>(
    name: &str,
    obj: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray1<'py, bool>> {
    obj.extract()
        .map_err(|_| array_type_err(name, "1-D bool", obj))
}

/// Copy a validated 1-D numpy `bool` mask into a `Vec<bool>` by reading
/// its raw bytes.
///
/// numpy stores `bool_` in one byte and does **not** constrain that byte
/// to 0 or 1, and it treats every non-zero byte as true. Such a buffer
/// is reachable from ordinary numpy — `np.frombuffer(buf, dtype=bool)`
/// over arbitrary bytes, `np.uint8` data through `.view(bool)`, or even
/// an uninitialised `np.empty(n, dtype=bool)` — so this needs no
/// deliberate byte-forging. A Rust `bool` may only ever hold 0 or 1, so
/// forming *any* `&bool`, `&[bool]` or `ArrayView<bool>` over one is
/// undefined behavior, and it mis-filtered searches concretely: up to
/// returning none of the selected slots and only unselected ones — a
/// total filter bypass (#349). So the bytes are read through a raw
/// pointer as `u8` and compared `!= 0`, matching numpy's truthiness.
///
/// Nothing here calls into Python, deliberately. Reaching the bytes
/// through the array's own `view("uint8")` would let an `np.ndarray`
/// subclass overriding `view` decide which buffer the mask is read from
/// — the caller's *type* would choose which slots are searched, which is
/// the very leak this function exists to close. `data()` is a field of
/// the C-level array struct, so no Python method lookup is involved and
/// no subclass can influence it.
fn mask_bytes(name: &str, arr: &PyReadonlyArray1<'_, bool>) -> PyResult<Vec<bool>> {
    // Same rejection (and message) the previous `as_slice()` produced.
    if !arr.is_c_contiguous() {
        return Err(not_contiguous_err(name));
    }
    let n = arr.len();
    if n == 0 {
        // Skip the pointer entirely: an empty array need not carry a
        // dereferenceable one.
        return Ok(Vec::new());
    }
    // SAFETY: `arr` is a live, C-contiguous, 1-D numpy array of `n`
    // `bool_` elements, and numpy's `bool_` is one byte wide, so `n`
    // bytes from the data pointer lie inside the array's own allocation.
    // The GIL is held and the readonly borrow is registered for the
    // whole read, so nothing can resize or free the buffer underneath
    // it. `*mut bool -> *const u8` casts between two one-byte types
    // with the same alignment, and no `bool` reference is ever formed —
    // which is the entire point (see above).
    let bytes = unsafe { std::slice::from_raw_parts(arr.data() as *const u8, n) };
    Ok(bytes.iter().map(|&b| b != 0).collect())
}

/// Whether `obj` is integer-valued: a Python `int` of any magnitude, or
/// any object (numpy scalar, `__index__` implementor) that converts to
/// one. Used to pick a range error over a type error once the fast-path
/// fixed-width extraction has failed.
fn is_py_int(obj: &Bound<'_, PyAny>) -> bool {
    obj.extract::<i128>().is_ok() || obj.is_instance_of::<pyo3::types::PyInt>()
}

/// `str()` of an argument, for error messages.
fn int_repr(obj: &Bound<'_, PyAny>) -> String {
    obj.str()
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "<unprintable>".to_string())
}

/// Extract a non-negative count/size argument (`dim`, `bit_width`, `k`)
/// as `usize`. Values in the unsigned 64-bit range pass through untouched
/// (over-large-but-representable values stay subject to each method's own
/// range rules, e.g. `k` clamping and the core's `dim` cap). Integers
/// outside that range raise a `ValueError` naming the argument instead of
/// pyo3's bare `OverflowError`; non-integers raise `TypeError`.
fn extract_size(name: &str, obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(v) = obj.extract::<usize>() {
        return Ok(v);
    }
    if is_py_int(obj) {
        let msg = if obj.lt(0).unwrap_or(false) {
            format!(
                "{name} must be a non-negative integer, got {}",
                int_repr(obj)
            )
        } else {
            format!(
                "{name} must fit in an unsigned 64-bit integer, got {}",
                int_repr(obj)
            )
        };
        return Err(pyo3::exceptions::PyValueError::new_err(msg));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "{name} must be an integer, got {}",
        type_name(obj),
    )))
}

/// Extract an external id for membership-style calls (`contains`,
/// `__contains__`, `remove`). Integers outside the `u64` range can never
/// be present in the index, so they yield `None` ("absent") rather than
/// pyo3's bare `OverflowError`; non-integers raise `TypeError`.
fn extract_membership_id(name: &str, obj: &Bound<'_, PyAny>) -> PyResult<Option<u64>> {
    if let Ok(v) = obj.extract::<u64>() {
        return Ok(Some(v));
    }
    if is_py_int(obj) {
        return Ok(None);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "{name} must be an integer, got {}",
        type_name(obj),
    )))
}

/// Map a numpy shape error from reassembling search results into a typed
/// RuntimeError. The result dimensions are derived from the core's own
/// output, so this never fires today — but a future change to result shaping
/// would otherwise surface as an uncatchable panic instead of a catchable
/// exception.
fn shape_err(e: numpy::ndarray::ShapeError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "internal error: malformed search result shape: {e}"
    ))
}

/// Reject NaN / Inf / overflow-magnitude query coordinates with a typed
/// `ValueError`. The core `search` panics on invalid values (its documented
/// Rust contract), which would otherwise surface to Python as an uncatchable
/// `PanicException`. `add` already maps the same condition to `ValueError`;
/// this keeps `search` consistent.
///
/// The scan splits across the current rayon pool once the input exceeds one
/// chunk, so callers route it through [`with_pool`] then — see
/// [`validate_queries_pooled`].
fn validate_queries(values: &[f32], dim: usize) -> PyResult<()> {
    if let Some((vi, ci, v)) = turbovec_core::first_invalid_coord(values, dim) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid query value at query {vi}, coord {ci}: {v} \
             (must be finite and |value| < 1e16)",
        )));
    }
    Ok(())
}

/// [`validate_queries`], run in the fork-safe pool when — and only when —
/// the scan actually splits (issue #288). A splitting scan on the calling
/// thread injects work into rayon's global pool, which this module pins to
/// a one-thread sentinel whose worker is dead in a forked child: the #147
/// deadlock, reachable from an ordinary `search` with >64K query floats.
///
/// The gate is the chunk threshold rather than the search's own pooling
/// predicate: a sub-chunk buffer is a single `par_chunks` item that rayon
/// folds inline without entering any pool, so the small path is safe *and*
/// must stay free of the ~70 us `install` handoff — routing it through
/// `with_pool_if` instead measurably regresses small searches.
fn validate_queries_pooled(values: &[f32], dim: usize) -> PyResult<()> {
    if turbovec_core::validation_parallelizes(values.len()) {
        with_pool(|| validate_queries(values, dim))?
    } else {
        validate_queries(values, dim)
    }
}

/// Extract a bytes-like argument (`bytes` / `bytearray`) into an owned
/// buffer; see [`extract_f32_2d`] for the error-naming rationale. Owned
/// because the buffer must cross a GIL release (`py.detach`), where a
/// borrowed Python buffer would be mutable out from under us.
fn extract_bytes(name: &str, obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(b) = obj.cast::<PyBytes>() {
        return Ok(b.as_bytes().to_vec());
    }
    if let Ok(b) = obj.cast::<pyo3::types::PyByteArray>() {
        return Ok(b.to_vec());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "{name} must be a bytes-like object (bytes or bytearray), got {}",
        type_name(obj),
    )))
}

/// Map an `io::Error` from `from_bytes` to Python. Unlike [`load_err`]
/// there is no path to blame, and the input is an in-memory value —
/// so corrupt bytes raise `ValueError` (like other malformed-value
/// arguments) rather than `OSError`.
fn from_bytes_err(e: std::io::Error) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Shared body of both indexes' `__reduce__`: rebuild through the
/// object's own `from_bytes` classmethod applied to its `to_bytes()`
/// payload.
///
/// `pickle`, `copy.copy` and `copy.deepcopy` therefore all inherit the
/// documented `.tv` persistence contract unchanged — the same payload
/// `write` produces and the same load-time validation. Nothing here
/// widens or narrows what `to_bytes` carries; a reconstructed index is
/// exactly what `from_bytes(to_bytes())` has always produced, and is
/// fully independent of the original (#340).
///
/// Reducing to the public classmethod rather than a private rebuild
/// helper keeps `to_bytes`/`from_bytes` the single serialization path
/// (docs/api.md) and keeps the pickle stream readable.
fn reduce_via_bytes<'py>(
    slf: &Bound<'py, PyAny>,
    payload: Bound<'py, PyBytes>,
) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
    Ok((slf.get_type().getattr("from_bytes")?, (payload,)))
}

/// Map an `io::Error` from `load` or `write` to Python:
/// `ErrorKind::NotFound` becomes `FileNotFoundError` (so
/// `except FileNotFoundError:` works, matching what the integrations'
/// Python-side `open()` of their JSON side-cars raises for a missing
/// path); every other kind — including permission errors — stays plain
/// `OSError`, as before. The path is appended because the io::Error
/// alone doesn't name the file.
fn load_err(path: &str, e: std::io::Error) -> PyErr {
    let msg = format!("{e}: {path}");
    if e.kind() == std::io::ErrorKind::NotFound {
        pyo3::exceptions::PyFileNotFoundError::new_err(msg)
    } else {
        pyo3::exceptions::PyIOError::new_err(msg)
    }
}

/// Read-lock an index, recovering from poisoning. A panic that escaped
/// a previous call has already surfaced to Python as a PanicException;
/// before the GIL-release change such a panic likewise left the object
/// reachable, so poisoning must not turn every later call into a panic.
/// Python name for a [`turbovec_core::CalibrationState`].
fn calibration_state_name(state: turbovec_core::CalibrationState) -> &'static str {
    match state {
        turbovec_core::CalibrationState::Uncalibrated => "uncalibrated",
        turbovec_core::CalibrationState::Calibrated => "calibrated",
        // `CalibrationState` is `#[non_exhaustive]`; the binding is
        // versioned in lockstep with the core, so a new variant here is
        // a build that must not ship.
        _ => unreachable!("unhandled CalibrationState variant"),
    }
}

fn lock_read<T>(lock: &std::sync::RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Write-lock an index; see [`lock_read`] for the poisoning rationale.
fn lock_write<T>(lock: &std::sync::RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Write-lock for O(1) ops called while attached to the GIL: try the
/// lock without blocking (the uncontended fast path — cheaper than a
/// GIL detach round-trip), and only when another thread holds the lock
/// (e.g. a multi-second bulk add) detach before waiting, so a brief
/// removal never stalls every Python thread behind a long writer.
fn lock_write_gil_aware<'l, T: Send + Sync>(
    py: Python<'_>,
    lock: &'l std::sync::RwLock<T>,
) -> std::sync::RwLockWriteGuard<'l, T> {
    loop {
        match lock.try_write() {
            Ok(guard) => return guard,
            Err(std::sync::TryLockError::Poisoned(p)) => return p.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                // A guard cannot cross the GIL release, so wait for the
                // lock detached (acquire and immediately drop), then
                // retry attached. Another thread may slip in between —
                // the loop just waits again; each iteration makes
                // progress once the long writer finishes.
                py.detach(|| drop(lock_write(lock)));
            }
        }
    }
}

// Locking discipline (both pyclasses): the pyclass is `frozen`, so pyo3
// performs no runtime borrow-checking of its own and every method takes
// `&self`; the inner index sits behind an `RwLock` instead. Concurrent
// searches share the read lock and run in parallel; a write blocks until
// it is alone, then succeeds — the same observable outcome as when the
// GIL serialized every call, minus the serialization of reads.
//
// Every method acquires the lock INSIDE `py.detach(..)` — even trivial
// O(1) calls like `__len__` — so a thread blocked on the lock is never
// holding the GIL (a GIL-held `len()` landing during an in-flight write
// would otherwise stall every Python thread for the write's duration),
// and every index-dependent check runs under the same guard as the core
// call it protects (the pre-GIL-release code got that atomicity for
// free from the GIL). A guard is only ever released from code that does
// not need the GIL to finish, so no lock/GIL deadlock cycle exists.

/// Release a reused snapshot buffer that is far larger than the adds
/// around it need, and record this call's demand in `prev` for the next.
///
/// The same policy the core applies to its encode scratch, for the same
/// reasons — see `turbovec_core`'s `retain_scratch`. The target is the
/// previous call's length plus half again, and the buffer is only
/// touched when its capacity exceeds twice that: the slack preserves the
/// amortized growth headroom that a growing or jittering batch size
/// depends on, and the hysteresis keeps every ordinary shape at zero
/// extra allocations. A one-shot bulk load has `prev == 0` and so
/// releases the whole buffer on the call that allocated it.
///
/// `truncate` before `shrink_to` is load-bearing on that release path:
/// `shrink_to` never goes below `len`, and the buffer is left holding
/// the full snapshot, which is above the target whenever there is
/// anything to release. (`truncate` is itself a no-op when the length is
/// already at or below the target.)
fn retain_snap(snap: &mut Vec<f32>, prev: &std::sync::atomic::AtomicUsize, want: usize) {
    let prev = prev.swap(want, std::sync::atomic::Ordering::Relaxed);
    let target = prev.saturating_add(prev / 2);
    if snap.capacity() > target.saturating_mul(2) {
        snap.truncate(target);
        snap.shrink_to(target);
    }
}

// `module` makes `__module__` report the extension module the class
// actually lives in, so a class reference round-trips through pickle and
// through any registry that records `f"{cls.__module__}.{cls.__name__}"`
// (it read `builtins.TurboQuantIndex` before, which resolves nowhere).
// `weakref` lets an index sit in a `weakref.WeakValueDictionary` — the
// standard way to key a per-tenant cache without pinning the memory
// (#340). Both classes carry the same options.
#[pyclass(frozen, module = "turbovec._turbovec", weakref)]
struct TurboQuantIndex {
    /// Reusable GIL-safety snapshot buffer for `add`. Purely scratch:
    /// contents are meaningless between calls; kept so repeated adds
    /// reuse one warm allocation instead of paying a fresh multi-MB
    /// mmap + page-fault walk per call. Taken (mem::take) for the
    /// duration of one add and put back after, so concurrent adds
    /// simply fall back to a fresh buffer.
    snap: std::sync::Mutex<Vec<f32>>,
    /// Element count the *previous* `add` snapshotted, sizing
    /// [`retain_snap`]'s retention target (#333). Advisory in the same
    /// sense `snap` is: concurrent adds may interleave their updates, so
    /// the recorded demand can be a different call's. That only costs a
    /// resize — the buffer is scratch either way.
    snap_prev: std::sync::atomic::AtomicUsize,
    inner: std::sync::RwLock<turbovec_core::TurboQuantIndex>,
}

#[pymethods]
impl TurboQuantIndex {
    /// Construct an index. `dim` is optional: when omitted, the
    /// underlying quantized index is created lazily on the first
    /// `add` call, picking up the dimensionality from the input
    /// array's shape. `bit_width` defaults to 4.
    #[new]
    #[pyo3(signature = (dim=None, bit_width=None))]
    fn new(dim: Option<&Bound<'_, PyAny>>, bit_width: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let bit_width = match bit_width {
            Some(b) => extract_size("bit_width", b)?,
            None => 4,
        };
        let inner = match dim {
            Some(d) => turbovec_core::TurboQuantIndex::new(extract_size("dim", d)?, bit_width),
            None => turbovec_core::TurboQuantIndex::new_lazy(bit_width),
        }
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: std::sync::RwLock::new(inner),
            snap: std::sync::Mutex::new(Vec::new()),
            snap_prev: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    fn add(&self, py: Python<'_>, vectors: &Bound<'_, PyAny>) -> PyResult<()> {
        let vectors = extract_f32_2d("vectors", vectors)?;
        let arr = vectors.as_array();
        let dim = arr.ncols();
        let slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("vectors"))?;
        // Snapshot the numpy buffer before releasing the GIL (`detach`):
        // once released, another Python thread may write to the (possibly
        // writable) source array, and rust-numpy's borrow flags cannot
        // prevent Python-side writes. The copy is O(n·dim) against the
        // quantization kernel — but done serially it was the largest
        // single-threaded span left on the add path, so split it across
        // the rayon pool (same pool, and therefore the same
        // RAYON_NUM_THREADS clamp, as the encode kernels). The copy runs
        // under `with_pool` while the GIL is still held — the sentinel
        // global pool is single-threaded, so a bare par_iter would fold
        // serial. No index lock is taken inside the closure (see the
        // `with_pool` invariant). Small inputs skip the pool handoff
        // entirely: a single-vector add must not pay an `install` for a
        // 6 KB memcpy.
        // Reuse the per-index snapshot buffer so repeated adds keep one
        // warm allocation (a fresh multi-MB buffer per call costs a
        // page-fault walk). Taken out of the mutex for the duration of
        // this add; a concurrent add simply starts from an empty
        // buffer.
        let mut owned = std::mem::take(&mut *self.snap.lock().expect("snap lock poisoned"));
        if slice.len() < PAR_COPY_MIN_LEN || !pool_idle() {
            // Small input — or the rayon pool is busy with a long batch,
            // in which case queueing the copy behind it while holding
            // the GIL would stall every Python thread; a bounded serial
            // copy is the safe fallback (see pool_idle).
            owned.clear();
            owned.extend_from_slice(slice);
        } else {
            with_pool(|| par_copy_into(slice, &mut owned))?;
        }
        // `add_2d` handles both eager (dim must match) and lazy (locks
        // dim on first call) cases.
        //
        // Single-row adds take the same inline bypass as nq==1 search —
        // the rayon bridges in a one-row *encode* have length 1 and fold
        // on the calling thread — but two conditions must hold. The
        // input-validation scan must not split: it chunks by float
        // count, not by row, so a single wide row can exceed one chunk
        // and inject work into the global sentinel pool (issue #321;
        // #288 is the same hole on the search side). Gating on
        // `validation_parallelizes` states that dependency instead of
        // resting on `MAX_DIM` happening to be below the chunk length.
        //
        // There is deliberately NO `packed_ready()` term. It used to
        // stand for "the first mutation after a v6 load rebuilds the
        // packed rows, a payload-sized parallel job regardless of row
        // count", but an add on a loaded index now takes the core's
        // lazy-append branch, which appends to the blocked cache and
        // leaves the packed rows unset. So the probe was permanently
        // false on a loaded index and sent *every* single-row add
        // through a pool handoff costing far more than the add (#392).
        //
        // What matters for this gate is that the lazy-append branch
        // contains no rayon fan-out, which it does not. It is not free,
        // though: `pack::extract_codes_flat` rebuilds a 4 KB extract LUT
        // and allocates a `Vec<Vec<u8>>` on every call, a fixed cost a
        // one-row add pays in full — most of the ~3x a one-row add on a
        // loaded index still costs over a fresh one. That is serial work
        // to be optimized in the core, not a reason to enter the pool.
        // Probing under the write guard makes the check race-free.
        let n_rows = if dim == 0 { 0 } else { slice.len() / dim };
        let validation_splits = turbovec_core::validation_parallelizes(owned.len());
        py.detach(|| {
            let mut guard = lock_write(&self.inner);
            let inner = &mut *guard;
            let pooled = n_rows > 1 || validation_splits;
            with_pool_if(pooled, || inner.add_2d(&owned, dim))
        })?
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        // Shrink before returning the buffer, so a one-time bulk load
        // doesn't pin its full snapshot capacity for the index lifetime
        // (#333).
        retain_snap(&mut owned, &self.snap_prev, slice.len());
        *self.snap.lock().expect("snap lock poisoned") = owned;
        Ok(())
    }

    /// Fit the TQ+ per-coordinate calibration from ``sample`` — a 2D
    /// float32 array of vectors — and commit it. Every later ``add``
    /// encodes under it, and it improves recall by roughly 2.5 points
    /// of R@10 on average (up to ~8.7 on the most anisotropic data
    /// measured).
    ///
    /// The fit uses every row given. **The sample must be a uniform
    /// random, representative draw of the vectors this index will
    /// hold** — around 1024 rows is enough, and choosing them well is
    /// entirely the caller's responsibility: a sorted or clustered
    /// prefix of the same size fits a calibration that actively
    /// destroys recall.
    ///
    /// May be called at any time and repeatedly. On a populated index
    /// the stored rows are re-encoded under the new pair from their
    /// stored codes (no original vectors needed). That re-encode is a
    /// second quantization: refitting with a near-identical pair is
    /// free, but calibrating *after* a large uncalibrated ingest costs
    /// several points of recall versus calibrating first — and a badly
    /// biased earlier calibration cannot be repaired this way at all,
    /// because its clipping already destroyed the information. Rebuild
    /// from the source vectors for that. Calibrate before adding when
    /// you can.
    fn calibrate(&self, py: Python<'_>, sample: &Bound<'_, PyAny>) -> PyResult<()> {
        let sample = extract_f32_2d("sample", sample)?;
        let arr = sample.as_array();
        let dim = arr.ncols();
        let slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("sample"))?;
        // Snapshot before detaching, same as `add`: another Python
        // thread may mutate the source array once the GIL is released.
        let owned = slice.to_vec();
        py.detach(|| {
            let mut guard = lock_write(&self.inner);
            let inner = &mut *guard;
            // The fit rotates the sample and a refit re-encodes every
            // stored row through the rayon kernels — pool both.
            with_pool(|| inner.calibrate_2d(&owned, dim))
        })?
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Run a top-`k` search against the index.
    ///
    /// `mask`, when given, is a bool array of length `len(self)`. Only slots
    /// with `mask[i] == True` contribute to the returned top-`k`. The
    /// returned result count per query is `min(k, mask.sum())`.
    ///
    /// A mask names slots, and `swap_remove` renumbers them, so any
    /// mutation invalidates a mask — not only one that changes the
    /// length. The length check is not what protects you: a
    /// `swap_remove(i)` + `add` pair restores the original length while
    /// leaving a different vector in slot `i`, so a mask built before
    /// that pair passes validation and then silently selects a
    /// different set of vectors than intended. Rebuild the mask after
    /// every mutation.
    #[pyo3(signature = (queries, k, *, mask=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
        k: &Bound<'py, PyAny>,
        mask: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i64>>)> {
        let queries = extract_f32_2d("queries", queries)?;
        let k = extract_size("k", k)?;
        let mask = mask.map(|m| extract_bool_1d("mask", m)).transpose()?;
        let arr = queries.as_array();
        let nq = arr.nrows();
        let ncols = arr.ncols();
        let q_slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("queries"))?;
        // Snapshot the query (and mask) buffers before releasing the
        // GIL: once released, another Python thread may write to the
        // source arrays mid-search. Validation runs on the snapshot so
        // the searched data is exactly the data that was validated.
        let q_owned = q_slice.to_vec();
        // Read as bytes, never as `bool`: every non-zero byte is true,
        // as numpy defines it (see `mask_bytes`). Same position in the
        // call as the old `as_slice().to_vec()`, so the contiguity error
        // still fires after the query's.
        let mask_owned: Option<Vec<bool>> = match mask.as_ref() {
            Some(m) => Some(mask_bytes("mask", m)?),
            None => None,
        };

        // Index-dependent checks run under the same read guard as the
        // kernel, so a concurrent writer cannot invalidate them between
        // check and search (pre-GIL-release, holding the GIL made the
        // whole call atomic). The guard is taken on the calling thread —
        // never inside `with_pool` — see the invariant on `with_pool`.
        // nq > 1 splits over queries and runs in the fork-safe pool; nq == 1
        // never splits and runs inline unless we are in a forked child (see
        // `with_pool_if`).
        let results = py.detach(|| -> PyResult<_> {
            let guard = lock_read(&self.inner);
            let inner = &*guard;
            // Reject wrong-dim queries cleanly. Previously the inner
            // `assert_eq!(queries.len(), nq * dim)` would fire as a Rust
            // panic and surface to Python as a PanicException, not the
            // ValueError users expect for input-shape mismatch.
            if let Some(idx_dim) = inner.dim_opt() {
                if ncols != idx_dim {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "query dim {ncols} does not match index dim {idx_dim}",
                    )));
                }
            }
            validate_queries_pooled(&q_owned, ncols)?;
            if let Some(m) = mask_owned.as_deref() {
                let expected = inner.len();
                if m.len() != expected {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "mask length {} does not match index size {}",
                        m.len(),
                        expected,
                    )));
                }
            }
            // Pool nq=1 too once the index is large enough for the core's
            // block-parallel single-query path — running that inline
            // would inject parallel work into the global sentinel
            // registry (the #147 invariant). Small-index nq=1 stays
            // inline; a pooled-but-serial masked search only costs the
            // install handoff.
            with_pool_if(nq > 1 || turbovec_core::search::single_query_parallelizes(inner.len()), || {
                inner.search_with_mask(&q_owned, k, mask_owned.as_deref())
            })
        })?;
        let effective_k = results.k;

        let scores = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), results.scores)
            .map_err(shape_err)?
            .into_pyarray(py);
        let indices = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), results.indices)
            .map_err(shape_err)?
            .into_pyarray(py);

        Ok((scores, indices))
    }

    /// Persist the index. ``durable=True`` (the default) fsyncs before
    /// the atomic rename, surviving power loss; ``durable=False`` keeps
    /// the temp-file + atomic-rename protocol (the destination can never
    /// hold a torn index and the previous file survives a process crash)
    /// but skips fsync — faster, not power-loss-safe.
    ///
    /// The calibration state round-trips exactly: an uncalibrated index
    /// reloads as ``"uncalibrated"``, a calibrated one as
    /// ``"calibrated"`` with the identical fitted pair.
    #[pyo3(signature = (path, *, durable = true))]
    fn write(&self, py: Python<'_>, path: &str, durable: bool) -> PyResult<()> {
        let durability = if durable {
            turbovec_core::io::Durability::Durable
        } else {
            turbovec_core::io::Durability::Fast
        };
        // Lock on the calling thread, never inside `with_pool` (see its
        // invariant); the v6 write path parallelizes the layout
        // transform, so it must run in the fork-safe pool.
        // Probe under the SAME detached lock acquisition that does the
        // write: acquiring it while still attached to the GIL blocks every
        // Python thread for the duration of a concurrent bulk add (the
        // #289 class, see this file's lock discipline note), and a
        // separate probe could also report a state another thread has
        // already moved on from. Warn once back under the GIL (#354).
        let (state, len, result) = py.detach(|| {
            // The post-commit fsync can warn (#365) from inside the pool,
            // with this guard still held — queue it rather than let it run
            // a user `showwarning` that could re-enter the index (#360).
            let _defer = DeferCoreWarnings::new();
            let guard = lock_read(&self.inner);
            let state = guard.calibration_state();
            let len = guard.len();
            let result = with_pool(|| guard.write_with_durability(path, durability));
            (state, len, result)
        });
        flush_core_warnings(py);
        let _ = (state, len);
        result?
            // `load_err` names the path and narrows a missing directory to
            // FileNotFoundError, matching `load` (#329).
            .map_err(|e| load_err(path, e))
    }

    /// Incrementally persist the index to ``path``. The first sync of a
    /// fresh path writes the whole file; every later sync to the same
    /// path appends only what changed since the last one — added
    /// vectors and applied removals — and commits, so saving after a
    /// small batch costs kilobytes, not the file. A crash at any byte
    /// leaves the previous commit intact. Every sync is durable: when it
    /// returns, the commit is on stable storage — there is no fast
    /// mode. Re-calibrating (or heavy churn)
    /// makes the next sync compact the file by rewriting it whole.
    ///
    /// ``load`` recognises synced files, and a loaded index keeps
    /// syncing forward incrementally. ``write`` keeps its meaning; a
    /// sync after a ``write`` to the same path rebuilds the synced
    /// format once, then appends again.
    fn sync(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        // A sync mutates the cursor, so it takes the write lock; detach
        // first for the same #289 reason as `write`, and queue core
        // warnings (#360/#365) rather than running user Python under
        // the lock.
        let result = py.detach(|| {
            let _defer = DeferCoreWarnings::new();
            let mut guard = lock_write(&self.inner);
            let index = &mut *guard;
            with_pool(|| index.sync(path))
        });
        flush_core_warnings(py);
        result?.map_err(|e| load_err(path, e))
    }

    #[classmethod]
    fn load(cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        // The v6 load parallelizes the layout transform — run it in the
        // fork-safe pool.
        let inner = cls
            .py()
            .detach(|| with_pool(|| turbovec_core::TurboQuantIndex::load(path)))?
            .map_err(|e| load_err(path, e))?;
        Ok(Self {
            inner: std::sync::RwLock::new(inner),
            snap: std::sync::Mutex::new(Vec::new()),
            snap_prev: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Serialize the index to ``bytes`` in the ``.tv`` format —
    /// byte-identical to the file ``write(path)`` produces. Pairs with
    /// ``from_bytes`` for in-memory persistence (caches, databases,
    /// pickling) without a filesystem round-trip.
    ///
    /// The calibration state round-trips exactly — this is the path
    /// ``pickle``, ``copy.copy`` and ``copy.deepcopy`` take, and a copy
    /// is byte-for-byte what ``write`` would have produced.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        // Serialize under the read lock with the GIL released (the
        // payload scales with the index size); wrap into a PyBytes
        // only once back under the GIL.
        // See `write`: the calibration probe must not acquire the lock
        // while attached to the GIL (#354).
        let (state, len, buf) = py.detach(|| {
            let guard = lock_read(&self.inner);
            let state = guard.calibration_state();
            let len = guard.len();
            let buf = with_pool(|| guard.to_bytes());
            (state, len, buf)
        });
        let _ = (state, len);
        let buf = buf?;
        Ok(PyBytes::new(py, &buf))
    }

    /// Deserialize an index from ``bytes`` produced by ``to_bytes`` (or
    /// read out of a ``.tv`` file). Applies exactly the same validation
    /// as ``load`` — corrupt or drifted payloads raise ``ValueError``.
    #[classmethod]
    fn from_bytes(cls: &Bound<PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Snapshot the buffer before releasing the GIL (a bytearray can
        // be mutated by another Python thread once it is released).
        let owned = extract_bytes("data", data)?;
        let inner = cls
            .py()
            .detach(|| with_pool(|| turbovec_core::TurboQuantIndex::from_bytes(&owned)))?
            .map_err(from_bytes_err)?;
        Ok(Self {
            inner: std::sync::RwLock::new(inner),
            snap: std::sync::Mutex::new(Vec::new()),
            snap_prev: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Support ``pickle``, ``copy.copy`` and ``copy.deepcopy`` by
    /// reducing to ``from_bytes(to_bytes())`` — so an index can cross a
    /// ``spawn`` process boundary (the default start method on macOS and
    /// Windows) and a user container holding one can be deep-copied.
    /// See [`reduce_via_bytes`] for what that inherits.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let payload = slf.get().to_bytes(slf.py())?;
        reduce_via_bytes(slf.as_any(), payload)
    }

    /// Warm up the search caches (rotation matrix, Lloyd-Max centroids,
    /// SIMD-blocked code layout) so the first `search` call does not pay
    /// the one-time initialisation cost.
    fn prepare(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let guard = lock_read(&self.inner);
            let inner = &*guard;
            with_pool(|| inner.prepare())
        })
    }

    /// Remove the vector at `idx` in O(1) by swapping with the last vector.
    ///
    /// The last vector moves into the deleted slot — order is not
    /// preserved. Returns the old index of the moved vector; equals `idx`
    /// when `idx` was already the last element.
    ///
    /// Raises ``IndexError`` if ``idx`` is out of range. Negative indices
    /// are out of range: Python-style indexing from the end is not
    /// supported.
    fn swap_remove(&self, py: Python<'_>, idx: &Bound<'_, PyAny>) -> PyResult<usize> {
        if let Ok(i) = idx.extract::<usize>() {
            // Bounds check and removal share one write guard, so a
            // concurrent writer cannot shrink the index in between.
            //
            // Always the uncontended fast path — no GIL detach, no pool
            // handoff. `swap_remove` maintains whichever code layout is
            // materialized in O(dim) lane ops and never triggers the lazy
            // O(n·dim) packed rebuild, so there is no slow case to route
            // around. Nor would a `packed_ready()` probe find one: no
            // mutation on a v6-loaded index materializes the packed rows
            // — not `add` (lazy-append branch), not `swap_remove` — so
            // the probe is permanently false there and would send every
            // remove through a pool handoff costing far more than the
            // removal (#392; see `TurboQuantIndex::packed_ready`).
            let removed = {
                let mut inner = lock_write_gil_aware(py, &self.inner);
                let len = inner.len();
                if i < len {
                    Ok(inner.swap_remove(i))
                } else {
                    Err(len)
                }
            };
            match removed {
                Ok(moved) => return Ok(moved),
                Err(len) => {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "index {} out of range for index of length {len}",
                        int_repr(idx),
                    )))
                }
            }
        }
        if !is_py_int(idx) {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "idx must be an integer, got {}",
                type_name(idx),
            )));
        }
        // Any integer that isn't a valid slot — negative, or of any
        // magnitude past the end — is out of range.
        let len = py.detach(|| lock_read(&self.inner).len());
        Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "index {} out of range for index of length {len}",
            int_repr(idx),
        )))
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        py.detach(|| lock_read(&self.inner).len())
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let (dim, bit_width, len) = py.detach(|| {
            let inner = lock_read(&self.inner);
            (inner.dim_opt(), inner.bit_width(), inner.len())
        });
        let dim = dim.map_or_else(|| "None".to_string(), |d| d.to_string());
        format!("turbovec.TurboQuantIndex(dim={dim}, bit_width={bit_width}, n_vectors={len})")
    }

    /// Vector dimensionality. Returns ``None`` when the index was
    /// constructed lazily (no ``dim=``) and hasn't seen an add yet;
    /// otherwise an ``int``.
    #[getter]
    fn dim(&self, py: Python<'_>) -> Option<usize> {
        py.detach(|| lock_read(&self.inner).dim_opt())
    }

    /// TQ+ calibration state: ``"uncalibrated"`` (no calibration
    /// committed — plain TurboQuant; call ``calibrate`` with a
    /// representative sample to add the TQ+ recall gain) or
    /// ``"calibrated"`` (a calibration is committed and every stored
    /// row is encoded under it).
    #[getter]
    fn calibration_state(&self, py: Python<'_>) -> &'static str {
        py.detach(|| calibration_state_name(lock_read(&self.inner).calibration_state()))
    }

    #[getter]
    fn bit_width(&self, py: Python<'_>) -> usize {
        py.detach(|| lock_read(&self.inner).bit_width())
    }

    /// Element capacity currently retained by the `add` snapshot buffer.
    ///
    /// Internal, for tests only, and deliberately not part of the public
    /// API: the buffer is private scratch with no Python-visible handle,
    /// and macOS RSS does not move when it is freed, so `retain_snap`'s
    /// effect on a *real* `add` cannot be observed from Python any other
    /// way (#333). `tests/test_snap_retention.py` is the only caller.
    fn _snap_capacity(&self) -> usize {
        self.snap.lock().expect("snap lock poisoned").capacity()
    }
}

// See `TurboQuantIndex` for `module` / `weakref`.
#[pyclass(frozen, module = "turbovec._turbovec", weakref)]
struct IdMapIndex {
    /// See `TurboQuantIndex::snap`.
    snap: std::sync::Mutex<Vec<f32>>,
    /// See `TurboQuantIndex::snap_prev`.
    snap_prev: std::sync::atomic::AtomicUsize,
    inner: std::sync::RwLock<turbovec_core::IdMapIndex>,
}

#[pymethods]
impl IdMapIndex {
    /// Construct an id-mapped index. `dim` is optional: when omitted,
    /// the underlying quantized index is created lazily on the first
    /// `add_with_ids` call, picking up dim from the input array shape.
    /// `bit_width` defaults to 4.
    #[new]
    #[pyo3(signature = (dim=None, bit_width=None))]
    fn new(dim: Option<&Bound<'_, PyAny>>, bit_width: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let bit_width = match bit_width {
            Some(b) => extract_size("bit_width", b)?,
            None => 4,
        };
        let inner = match dim {
            Some(d) => turbovec_core::IdMapIndex::new(extract_size("dim", d)?, bit_width),
            None => turbovec_core::IdMapIndex::new_lazy(bit_width),
        }
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: std::sync::RwLock::new(inner),
            snap: std::sync::Mutex::new(Vec::new()),
            snap_prev: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Add `n = vectors.shape[0]` vectors with the given external `ids`.
    ///
    /// `ids` must be a 1-D array of `uint64` with length equal to
    /// `vectors.shape[0]`. Raises `ValueError` if any id is already
    /// present in the index, if an id is repeated within this batch, or
    /// if the lengths don't match. On a lazy index, this call commits
    /// the dimensionality from `vectors.shape[1]`.
    fn add_with_ids(
        &self,
        py: Python<'_>,
        vectors: &Bound<'_, PyAny>,
        ids: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let vectors = extract_f32_2d("vectors", vectors)?;
        let ids = extract_u64_1d("ids", ids)?;
        let v = vectors.as_array();
        let dim = v.ncols();
        let v_slice = v.as_slice().ok_or_else(|| not_contiguous_err("vectors"))?;
        let i = ids.as_array();
        let i_slice = i.as_slice().ok_or_else(|| not_contiguous_err("ids"))?;
        // Snapshot both numpy buffers before releasing the GIL (another
        // Python thread may write to them once it is released); the core
        // then validates and quantizes the snapshot. The vector snapshot
        // reuses the per-index buffer (see `TurboQuantIndex::add`).
        let mut v_owned =
            std::mem::take(&mut *self.snap.lock().expect("snap lock poisoned"));
        if v_slice.len() < PAR_COPY_MIN_LEN || !pool_idle() {
            // Small input, or the pool is busy with a long batch —
            // bounded serial copy while attached (see TurboQuantIndex::add).
            v_owned.clear();
            v_owned.extend_from_slice(v_slice);
        } else {
            with_pool(|| par_copy_into(v_slice, &mut v_owned))?;
        }
        let i_owned = i_slice.to_vec();
        py.detach(|| {
            let mut guard = lock_write(&self.inner);
            let inner = &mut *guard;
            with_pool(|| inner.add_with_ids_2d(&v_owned, dim, &i_owned))
        })?
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        // Same shrink policy as TurboQuantIndex::add.
        let mut v_owned = v_owned;
        retain_snap(&mut v_owned, &self.snap_prev, v_slice.len());
        *self.snap.lock().expect("snap lock poisoned") = v_owned;
        Ok(())
    }

    /// Remove the vector with external id `id`. Returns `True` if it was
    /// present, `False` otherwise. Integers outside the `uint64` range are
    /// never present, so they return `False`.
    fn remove(&self, py: Python<'_>, id: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(match extract_membership_id("id", id)? {
            // Fast path: the id → slot map is already materialized — the
            // removal is then O(dim) lane ops plus two hash operations,
            // so it takes the uncontended GIL-aware path with no detach
            // and no pool handoff. The packed rows are deliberately NOT
            // probed: no mutation on a v6-loaded index materializes them
            // (`add` lazy-appends to the blocked cache, `swap_remove`
            // patches it with lane ops), so `packed_ready()` stays false
            // for such an index's whole lifetime and gating on it sent
            // every remove through the pool forever (#392; see
            // `TurboQuantIndex::packed_ready`).
            //
            // Slow path, once per loaded index: the first remove builds
            // the id → slot map (serial O(n) — issue #319), so it
            // detaches rather than stall every Python thread on the GIL.
            // No pool: that build is serial. The probe only goes
            // false→true, so a race merely sends a ready index down the
            // slow path harmlessly, and it runs detached because `read()`
            // blocks for the duration of a concurrent bulk add and doing
            // that with the GIL held stalls every Python thread (#289).
            Some(v) => {
                let ready = py.detach(|| lock_read(&self.inner).slots_ready());
                if ready {
                    lock_write_gil_aware(py, &self.inner).remove(v)
                } else {
                    py.detach(|| lock_write(&self.inner).remove(v))
                }
            }
            None => false,
        })
    }

    /// Search for the top-`k` nearest external ids for each query.
    ///
    /// `allowlist`, when given, is a `uint64` array of external ids; the
    /// returned top-`k` is restricted to ids in this list. The allowlist
    /// is deduplicated: the returned result count per query is
    /// `min(k, number of unique ids in allowlist)`, so repeated ids
    /// don't widen the result.
    ///
    /// Returns `(scores, ids)` as `(nq, effective_k)` arrays, `ids` typed
    /// `uint64`. Raises `ValueError` for an empty allowlist and `KeyError`
    /// if any allowlist id is not present in the index.
    #[pyo3(signature = (queries, k, *, allowlist=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
        k: &Bound<'py, PyAny>,
        allowlist: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u64>>)> {
        let queries = extract_f32_2d("queries", queries)?;
        let k = extract_size("k", k)?;
        let allowlist = allowlist
            .map(|a| extract_u64_1d("allowlist", a))
            .transpose()?;
        let arr = queries.as_array();
        let nq = arr.nrows();
        let ncols = arr.ncols();
        let q_slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("queries"))?;
        // Snapshot the query (and allowlist) buffers before releasing
        // the GIL: once released, another Python thread may write to the
        // source arrays mid-search. Validation runs on the snapshot so
        // the searched data is exactly the data that was validated.
        let q_owned = q_slice.to_vec();

        let allow_arr = allowlist.as_ref().map(|a| a.as_array());
        let allow_owned: Option<Vec<u64>> = match allow_arr.as_ref() {
            Some(a_arr) => {
                if a_arr.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "allowlist is empty",
                    ));
                }
                Some(
                    a_arr
                        .as_slice()
                        .ok_or_else(|| not_contiguous_err("allowlist"))?
                        .to_vec(),
                )
            }
            None => None,
        };

        // Index-dependent checks (dim, allowlist membership) run under
        // the same read guard as the kernel, so a concurrent writer
        // cannot invalidate them between check and search. The guard is
        // taken on the calling thread — never inside `with_pool` — see
        // the invariant on `with_pool`. `len` is captured under the same
        // guard for the nq == 0 shape contract below.
        let (scores, ids, len_at_search) = py.detach(|| -> PyResult<_> {
            let guard = lock_read(&self.inner);
            let inner = &*guard;
            if let Some(idx_dim) = inner.dim_opt() {
                if ncols != idx_dim {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "query dim {ncols} does not match index dim {idx_dim}",
                    )));
                }
            }
            validate_queries_pooled(&q_owned, ncols)?;
            if let Some(allow) = allow_owned.as_deref() {
                let mut unknown: Vec<u64> = Vec::new();
                for &id in allow {
                    if !inner.contains(id) {
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
            }
            // See search: nq=1 on a large index must run pooled for the
            // core's block-parallel single-query path.
            let (scores, ids) = with_pool_if(nq > 1 || turbovec_core::search::single_query_parallelizes(inner.len()), || {
                // The allowlist was already validated above, so this
                // cannot fail; map it anyway to the same exceptions that
                // validation raises rather than unwrapping.
                inner
                    .search_with_allowlist(&q_owned, k, allow_owned.as_deref())
                    .map_err(|e| match e {
                        turbovec_core::SearchError::UnknownId(_) => {
                            pyo3::exceptions::PyKeyError::new_err(e.to_string())
                        }
                        _ => pyo3::exceptions::PyValueError::new_err(e.to_string()),
                    })
            })??;
            Ok((scores, ids, inner.len()))
        })?;
        // For empty queries (nq=0), match TurboQuantIndex's shape
        // contract: effective_k is `min(k, n_vectors, n_allowed)`. The
        // kernel dedups the allowlist via a packed bool mask for nq>0,
        // so we have to dedup here too — otherwise `allowlist=[1, 1, 1]`
        // returns shape `(0, 3)` for empty queries but `(N, 1)` for
        // non-empty queries, a silent shape divergence.
        let effective_k = if nq == 0 {
            let n_allowed = match allow_owned.as_deref() {
                Some(s) => {
                    let mut seen: std::collections::HashSet<u64> =
                        std::collections::HashSet::with_capacity(s.len());
                    s.iter().filter(|id| seen.insert(**id)).count()
                }
                None => len_at_search,
            };
            k.min(len_at_search).min(n_allowed)
        } else {
            scores.len() / nq
        };

        let scores_arr = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), scores)
            .map_err(shape_err)?
            .into_pyarray(py);
        let ids_arr = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), ids)
            .map_err(shape_err)?
            .into_pyarray(py);
        Ok((scores_arr, ids_arr))
    }

    /// Return `True` if external id `id` is present. Integers outside the
    /// `uint64` range are never present, so they return `False`.
    fn contains(&self, py: Python<'_>, id: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(match extract_membership_id("id", id)? {
            Some(v) => py.detach(|| lock_read(&self.inner).contains(v)),
            None => false,
        })
    }

    /// Warm up the search caches (rotation matrix, Lloyd-Max centroids,
    /// SIMD-blocked code layout) **and** the lazy id → slot map, so
    /// neither the first `search` nor the first `search(..., allowlist=)`,
    /// `contains` or `remove` pays a one-time initialisation cost.
    fn prepare(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let guard = lock_read(&self.inner);
            let inner = &*guard;
            with_pool(|| inner.prepare())
        })
    }

    /// Fit the TQ+ per-coordinate calibration from ``sample`` — a 2D
    /// float32 array of vectors — and commit it. Every later ``add_with_ids``
    /// encodes under it, and it improves recall by roughly 2.5 points
    /// of R@10 on average (up to ~8.7 on the most anisotropic data
    /// measured).
    ///
    /// The fit uses every row given. **The sample must be a uniform
    /// random, representative draw of the vectors this index will
    /// hold** — around 1024 rows is enough, and choosing them well is
    /// entirely the caller's responsibility: a sorted or clustered
    /// prefix of the same size fits a calibration that actively
    /// destroys recall.
    ///
    /// May be called at any time and repeatedly. On a populated index
    /// the stored rows are re-encoded under the new pair from their
    /// stored codes (no original vectors needed). That re-encode is a
    /// second quantization: refitting with a near-identical pair is
    /// free, but calibrating *after* a large uncalibrated ingest costs
    /// several points of recall versus calibrating first — and a badly
    /// biased earlier calibration cannot be repaired this way at all,
    /// because its clipping already destroyed the information. Rebuild
    /// from the source vectors for that. Calibrate before adding when
    /// you can.
    fn calibrate(&self, py: Python<'_>, sample: &Bound<'_, PyAny>) -> PyResult<()> {
        let sample = extract_f32_2d("sample", sample)?;
        let arr = sample.as_array();
        let dim = arr.ncols();
        let slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("sample"))?;
        // Snapshot before detaching, same as `add`: another Python
        // thread may mutate the source array once the GIL is released.
        let owned = slice.to_vec();
        py.detach(|| {
            let mut guard = lock_write(&self.inner);
            let inner = &mut *guard;
            // The fit rotates the sample and a refit re-encodes every
            // stored row through the rayon kernels — pool both.
            with_pool(|| inner.calibrate_2d(&owned, dim))
        })?
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Serialize the index and id-map side-tables to a ``.tvim`` file.
    /// ``durable=True`` (the default) fsyncs before
    /// the atomic rename, surviving power loss; ``durable=False`` keeps
    /// the temp-file + atomic-rename protocol (the destination can never
    /// hold a torn index and the previous file survives a process crash)
    /// but skips fsync — faster, not power-loss-safe.
    ///
    /// The calibration state round-trips exactly: an uncalibrated index
    /// reloads as ``"uncalibrated"``, a calibrated one as
    /// ``"calibrated"`` with the identical fitted pair.
    #[pyo3(signature = (path, *, durable = true))]
    fn write(&self, py: Python<'_>, path: &str, durable: bool) -> PyResult<()> {
        let durability = if durable {
            turbovec_core::io::Durability::Durable
        } else {
            turbovec_core::io::Durability::Fast
        };
        // Lock on the calling thread, never inside `with_pool` (see its
        // invariant); the v6 write path parallelizes the layout
        // transform, so it must run in the fork-safe pool.
        // Probe under the SAME detached lock acquisition that does the
        // write: acquiring it while still attached to the GIL blocks every
        // Python thread for the duration of a concurrent bulk add (the
        // #289 class, see this file's lock discipline note), and a
        // separate probe could also report a state another thread has
        // already moved on from. Warn once back under the GIL (#354).
        let (state, len, result) = py.detach(|| {
            // See `TurboQuantIndex::write`: the core's post-commit
            // durability warning must not run user Python under this
            // guard (#360, #365).
            let _defer = DeferCoreWarnings::new();
            let guard = lock_read(&self.inner);
            let state = guard.calibration_state();
            let len = guard.len();
            let result = with_pool(|| guard.write_with_durability(path, durability));
            (state, len, result)
        });
        flush_core_warnings(py);
        let _ = (state, len);
        result?
            // `load_err` names the path and narrows a missing directory to
            // FileNotFoundError, matching `load` (#329).
            .map_err(|e| load_err(path, e))
    }

    /// Incrementally persist the index to ``path``. The first sync of a
    /// fresh path writes the whole file; every later sync to the same
    /// path appends only what changed since the last one — added
    /// vectors and applied removals — and commits, so saving after a
    /// small batch costs kilobytes, not the file. A crash at any byte
    /// leaves the previous commit intact. Every sync is durable: when it
    /// returns, the commit is on stable storage — there is no fast
    /// mode. Re-calibrating (or heavy churn)
    /// makes the next sync compact the file by rewriting it whole.
    ///
    /// ``load`` recognises synced files, and a loaded index keeps
    /// syncing forward incrementally. ``write`` keeps its meaning; a
    /// sync after a ``write`` to the same path rebuilds the synced
    /// format once, then appends again.
    fn sync(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        // A sync mutates the cursor, so it takes the write lock; detach
        // first for the same #289 reason as `write`, and queue core
        // warnings (#360/#365) rather than running user Python under
        // the lock.
        let result = py.detach(|| {
            let _defer = DeferCoreWarnings::new();
            let mut guard = lock_write(&self.inner);
            let index = &mut *guard;
            with_pool(|| index.sync(path))
        });
        flush_core_warnings(py);
        result?.map_err(|e| load_err(path, e))
    }

    /// Load an ``IdMapIndex`` from a ``.tvim`` file previously written
    /// by ``IdMapIndex.write``, or a synced file written by
    /// ``IdMapIndex.sync``.
    #[classmethod]
    fn load(cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        // The v6 load parallelizes the layout transform — run it in the
        // fork-safe pool.
        let inner = cls
            .py()
            .detach(|| with_pool(|| turbovec_core::IdMapIndex::load(path)))?
            .map_err(|e| load_err(path, e))?;
        Ok(Self {
            inner: std::sync::RwLock::new(inner),
            snap: std::sync::Mutex::new(Vec::new()),
            snap_prev: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Serialize the index (and its id-map side-tables) to ``bytes`` in
    /// the ``.tvim`` format — byte-identical to the file ``write(path)``
    /// produces. Pairs with ``from_bytes`` for in-memory persistence
    /// (caches, databases, pickling) without a filesystem round-trip.
    ///
    /// The calibration state round-trips exactly — this is the path
    /// ``pickle``, ``copy.copy`` and ``copy.deepcopy`` take, and a copy
    /// is byte-for-byte what ``write`` would have produced.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        // Serialize under the read lock with the GIL released (the
        // payload scales with the index size); wrap into a PyBytes
        // only once back under the GIL.
        // See `write`: the calibration probe must not acquire the lock
        // while attached to the GIL (#354).
        let (state, len, buf) = py.detach(|| {
            let guard = lock_read(&self.inner);
            let state = guard.calibration_state();
            let len = guard.len();
            let buf = with_pool(|| guard.to_bytes());
            (state, len, buf)
        });
        let _ = (state, len);
        let buf = buf?;
        Ok(PyBytes::new(py, &buf))
    }

    /// Deserialize an index from ``bytes`` produced by ``to_bytes`` (or
    /// read out of a ``.tvim`` file). Applies exactly the same
    /// validation as ``load`` — corrupt or drifted payloads raise
    /// ``ValueError``.
    #[classmethod]
    fn from_bytes(cls: &Bound<PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Snapshot the buffer before releasing the GIL (a bytearray can
        // be mutated by another Python thread once it is released).
        let owned = extract_bytes("data", data)?;
        let inner = cls
            .py()
            .detach(|| with_pool(|| turbovec_core::IdMapIndex::from_bytes(&owned)))?
            .map_err(from_bytes_err)?;
        Ok(Self {
            inner: std::sync::RwLock::new(inner),
            snap: std::sync::Mutex::new(Vec::new()),
            snap_prev: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// See ``TurboQuantIndex.__reduce__``.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let payload = slf.get().to_bytes(slf.py())?;
        reduce_via_bytes(slf.as_any(), payload)
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        py.detach(|| lock_read(&self.inner).len())
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let (dim, bit_width, len) = py.detach(|| {
            let inner = lock_read(&self.inner);
            (inner.dim_opt(), inner.bit_width(), inner.len())
        });
        let dim = dim.map_or_else(|| "None".to_string(), |d| d.to_string());
        format!("turbovec.IdMapIndex(dim={dim}, bit_width={bit_width}, n_vectors={len})")
    }

    fn __contains__(&self, py: Python<'_>, id: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(match extract_membership_id("id", id)? {
            Some(v) => py.detach(|| lock_read(&self.inner).contains(v)),
            None => false,
        })
    }

    /// Vector dimensionality. Returns ``None`` when the index was
    /// constructed lazily and hasn't seen an add yet; otherwise ``int``.
    #[getter]
    fn dim(&self, py: Python<'_>) -> Option<usize> {
        py.detach(|| lock_read(&self.inner).dim_opt())
    }

    /// TQ+ calibration state: ``"uncalibrated"`` (no calibration
    /// committed — plain TurboQuant; call ``calibrate`` with a
    /// representative sample to add the TQ+ recall gain) or
    /// ``"calibrated"`` (a calibration is committed and every stored
    /// row is encoded under it).
    #[getter]
    fn calibration_state(&self, py: Python<'_>) -> &'static str {
        py.detach(|| calibration_state_name(lock_read(&self.inner).calibration_state()))
    }

    #[getter]
    fn bit_width(&self, py: Python<'_>) -> usize {
        py.detach(|| lock_read(&self.inner).bit_width())
    }

    /// See `TurboQuantIndex::_snap_capacity`. Internal, tests only.
    fn _snap_capacity(&self) -> usize {
        self.snap.lock().expect("snap lock poisoned").capacity()
    }
}

// ===========================================================================
// Fork safety (issue #147)
// ===========================================================================
//
// rayon's global thread pool does not survive `fork()`: the worker threads
// live only in the parent, so the first parallel op a forked child runs
// injects work into a registry whose workers are gone and blocks on the
// latch forever. This is the default failure mode of `multiprocessing`
// (fork start method, the Linux default through 3.13), gunicorn
// `--preload`, Celery prefork, and PyTorch `DataLoader(num_workers>0)`.
//
// Fix: route every rayon-using kernel entry through `with_pool`, which
// installs a *process-local* `ThreadPool`. A forked child detects the fork
// and transparently rebuilds that pool, so its parallel ops run on live
// workers instead of the parent's dead ones.
//
// Fork detection (defence in depth, most authoritative first):
//   1. Python's `os.register_at_fork(after_in_child=...)` bumps
//      FORK_GENERATION. This is the authoritative signal and fires for
//      every CPython-mediated fork (os.fork, multiprocessing, billiard).
//      It is immune to the glibc `getpid()` userspace cache that would
//      otherwise let a child on manylinux2014 (glibc 2.17, cache not
//      removed until 2.25) observe the parent's PID and skip the rebuild —
//      the exact silent hang this fix targets (review finding F1).
//   2. A `pthread_atfork` child handler (Unix) also bumps FORK_GENERATION,
//      catching forks issued by non-Python native code that never runs the
//      Python hook.
//   3. A `getpid()` secondary confirm in `current_pool`: even if both
//      handlers somehow miss, a changed PID forces a rebuild on any modern
//      glibc where getpid is a live syscall.
//
// The inline fast paths (see `with_pool_if`) never enter any pool, so a
// missed detection there is still safe: length-1 parallel bridges fold on
// the calling thread. Every workload that can actually split enters a pool
// and goes through `current_pool`'s generation+PID gate — nq>1 search,
// prepare, and any add beyond a single row. The two remaining bypasses,
// nq==1 search and a single-row add into a packed-ready index, are gated
// so that they also pool whenever the input-validation scan would split
// (`validation_parallelizes`); that gate is what closed #288/#321, where
// the scan's chunking is by float count and does not follow row count.
// The single-row add is additionally gated on `add_parallelizes`, since
// the add that crosses the TQ+ warm-up threshold does work proportional
// to the *buffered* rows rather than to the added ones (#364).

/// Bumped by every fork-detecting handler. A child's copy is CoW-inherited
/// from the parent then incremented, so within a fork lineage the child's
/// generation is always strictly greater than the pool cell built by its
/// parent — the comparison in `current_pool` cannot false-negative.
static FORK_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Generation the currently-published pool was built at, mirrored out of
/// the `PoolCell` so the nq==1 bypass can decide "are we in a fresh child?"
/// with two relaxed atomic loads and no pointer dereference. `u64::MAX`
/// means "no pool built yet".
static POOL_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

/// The process-local pool plus the identity of the process/generation that
/// built it.
struct PoolCell {
    /// FORK_GENERATION value when this pool was built.
    gen: u64,
    /// PID that built this pool (secondary fork confirm, review F1).
    pid: u32,
    /// Thread count this pool was built with. A forked child rebuilds at
    /// the *parent's* recorded count rather than re-reading
    /// `available_parallelism`, so a child under a different cgroup CPU
    /// quota / `taskset` cannot silently rebuild at a different thread
    /// count (which would be a fresh source of the #206 rotation/encode
    /// thread-count sensitivity — review finding F2).
    threads: usize,
    pool: rayon::ThreadPool,
}

/// The live pool cell, or null before the first build.
///
/// Leak-not-drop is *structural* here: cells are `Box::leak`ed and this
/// pointer is only ever overwritten, never used to free. A `ThreadPool`
/// inherited across fork must never be dropped — `ThreadPool::drop`
/// terminates the registry and wakes workers, deadlocking on pthread
/// mutexes inherited in an undefined state (observed on macOS via `sample`:
/// `Arc::drop_slow` -> `Sleep::wake_specific_thread` ->
/// `_pthread_mutex_firstfit_lock_slow`, hung forever). Never freeing any
/// cell sidesteps that entirely and bounds the leak to <=1 pool per forked
/// process (a process's generation changes at most once per fork it
/// performs; a parent that forks N children never leaks, each child leaks
/// exactly the parent pool it inherited — a few KB, CoW-cheap).
static POOL_PTR: std::sync::atomic::AtomicPtr<PoolCell> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Record that we are now running in a forked child (called from the fork
/// handlers). Only a single atomic increment — async-signal-safe, so it is
/// legal from a `pthread_atfork` child handler.
fn note_fork_in_child() {
    FORK_GENERATION.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// pthread_atfork child callback (Unix). Must do nothing but async-signal-
/// safe work.
#[cfg(unix)]
extern "C" fn atfork_child_handler() {
    note_fork_in_child();
}

/// Python-callable shim registered via `os.register_at_fork`.
#[pyfunction]
fn _note_fork_in_child() {
    note_fork_in_child();
}

/// Thread count for the process-local pool: `RAYON_NUM_THREADS` (clamped to
/// [`rayon_thread_cap`], mirroring the eager-init rules) when set, else the
/// hardware default rayon itself would choose.
fn desired_threads() -> usize {
    match std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(n) if n > 0 => n.min(rayon_thread_cap()),
        _ => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    }
}

/// Build a process-local pool of `threads` workers. A build failure (thread
/// spawn refused, e.g. a resource-exhausted process) is retried once at a
/// single thread; if even that fails the error is returned so callers can
/// surface a catchable `RuntimeError` (never an uncatchable panic) or, at
/// import, degrade gracefully.
fn try_build_pool(threads: usize) -> Result<rayon::ThreadPool, rayon::ThreadPoolBuildError> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .or_else(|_| rayon::ThreadPoolBuilder::new().num_threads(1).build())
}

/// A pool-build failure surfaced to Python. `PyRuntimeError` subclasses
/// `Exception`, so ordinary `except Exception:` handling catches it (unlike a
/// Rust panic, which arrives as an uncatchable `PanicException`).
fn pool_build_error(threads: usize, e: &rayon::ThreadPoolBuildError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "turbovec: could not create its thread pool ({e}). The process may be \
         out of thread resources (see `ulimit -u`, cgroup pids limits, or a \
         too-large RAYON_NUM_THREADS={threads}). If this is a forked worker, \
         use the 'spawn' start method or fork before the parent's first \
         turbovec operation."
    ))
}

/// Return the pool for the current process, rebuilding it if we have forked
/// since it was built. Fast path: one acquire load + a generation/PID
/// compare (no lock), so concurrent readers under #208's shared read lock
/// never serialize here. Returns `Err` only when a fresh pool cannot be
/// built at all (see `try_build_pool`).
fn current_pool() -> PyResult<&'static PoolCell> {
    use std::sync::atomic::Ordering;
    let gen = FORK_GENERATION.load(Ordering::Acquire);
    let pid = std::process::id();
    let p = POOL_PTR.load(Ordering::Acquire);
    if !p.is_null() {
        // SAFETY: cells are `Box::leak`ed and never freed, so any non-null
        // pointer we ever store is valid for the process lifetime.
        let cell = unsafe { &*p };
        if cell.gen == gen && cell.pid == pid {
            return Ok(cell);
        }
    }
    rebuild_pool(gen, pid)
}

/// Cold path: build and publish a fresh cell. Lock-free (a CAS publish), so
/// it cannot itself deadlock even if a fork races another thread's rebuild.
#[cold]
fn rebuild_pool(gen: u64, pid: u32) -> PyResult<&'static PoolCell> {
    use std::sync::atomic::Ordering;
    // Reuse the parent's recorded thread count across a fork (F2); only the
    // very first build in a process (null pointer) reads the environment.
    let prev = POOL_PTR.load(Ordering::Acquire);
    let threads = if prev.is_null() {
        desired_threads()
    } else {
        // SAFETY: see current_pool — leaked, never freed.
        unsafe { &*prev }.threads
    };
    let pool = try_build_pool(threads).map_err(|e| pool_build_error(threads, &e))?;
    let cell: &'static PoolCell = Box::leak(Box::new(PoolCell {
        gen,
        pid,
        threads,
        pool,
    }));
    loop {
        let cur = POOL_PTR.load(Ordering::Acquire);
        if !cur.is_null() {
            // SAFETY: leaked, never freed.
            let c = unsafe { &*cur };
            if c.gen == gen && c.pid == pid {
                // Another thread already published an equivalent cell (a
                // startup race — a fresh fork has only the calling thread,
                // so this is only reachable at process start). Adopt theirs;
                // our freshly built `cell` leaks. Its workers are alive in
                // this process but idle forever: bounded and rare.
                return Ok(c);
            }
        }
        match POOL_PTR.compare_exchange(
            cur,
            cell as *const PoolCell as *mut PoolCell,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                POOL_GENERATION.store(gen, Ordering::Release);
                return Ok(cell);
            }
            // Lost the race; retry. `cell` stays valid (leaked).
            Err(_) => continue,
        }
    }
}

/// True when we appear to be in a forked child that has not yet rebuilt its
/// pool — used to route the nq==1 fast path through `install` so the child
/// rebuilds its pool promptly on first use. Two relaxed loads, no getpid, no
/// dereference: this is on the hot single-query search path.
fn in_forked_child() -> bool {
    use std::sync::atomic::Ordering;
    FORK_GENERATION.load(Ordering::Acquire) != POOL_GENERATION.load(Ordering::Acquire)
}

/// Run `f` inside the process-local, fork-safe pool so every `par_iter`
/// beneath it uses that pool instead of rayon's global (fork-unsafe) one.
///
/// INVARIANT: `f` must not acquire the index `RwLock` (or any lock another
/// `with_pool` closure can hold). `install` runs `f` on a pool WORKER
/// thread, and a worker parked at a `join` point work-steals other injected
/// closures — so a lock taken inside `f` can be re-entered on a thread that
/// already holds it (read-while-holding-write across two stolen ops),
/// deadlocking permanently. Call sites therefore lock on the calling thread
/// (which blocks on rayon's latch and never steals) and move only a plain
/// `&T` / `&mut T` into `f`.
fn with_pool<R: Send>(f: impl FnOnce() -> R + Send) -> PyResult<R> {
    // Drop guard so an unwinding `f` (or rayon worker) still decrements
    // the counter — a leaked increment would make `pool_idle()` false
    // forever and silently route every add through the serial copy
    // path (#300).
    struct InFlight;
    impl Drop for InFlight {
        fn drop(&mut self) {
            POOL_IN_FLIGHT.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    POOL_IN_FLIGHT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let _in_flight = InFlight;
    current_pool().map(|c| c.pool.install(f))
}

/// Number of `with_pool` jobs currently running or queued. Used by
/// GIL-attached callers (the add snapshot copy) to avoid queueing behind
/// a long-running batch while holding the GIL: when the pool is busy
/// they fall back to bounded serial work instead of waiting. The check
/// is advisory — a race simply means one snapshot copies serially (or
/// briefly queues), never an error.
static POOL_IN_FLIGHT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// True when no `with_pool` job is currently in flight.
fn pool_idle() -> bool {
    POOL_IN_FLIGHT.load(std::sync::atomic::Ordering::Relaxed) == 0
}

/// Hybrid dispatch. `parallel == false` (single-query search) runs inline on
/// the calling thread with no ~70 us pool handoff — every rayon site in the
/// search path parallelizes over queries, so a single query never splits.
///
/// The inline bypass is taken only when we are NOT in a fresh forked child,
/// so a child's first op routes through `install` and rebuilds its pool
/// promptly. Note that once a child HAS rebuilt, `in_forked_child` is false
/// again and its single-query searches go inline just like the parent's —
/// which is still safe, because a length-1 parallel iterator folds on the
/// calling thread and never enters (the child's own live, or the dead
/// inherited) pool. Inline single-query correctness thus rests on rayon's
/// no-split-at-length-1 behavior in every process; `test_nq1_*` guards it and
/// the rayon pin (see Cargo.toml) holds the version it was measured on.
fn with_pool_if<R: Send>(parallel: bool, f: impl FnOnce() -> R + Send) -> PyResult<R> {
    if parallel || in_forked_child() {
        with_pool(f)
    } else {
        Ok(f())
    }
}

/// Pin this extension's rayon *global* pool to a single sentinel thread. All
/// real parallelism runs in the process-local pool via `with_pool`; the
/// global registry is touched only by the nq==1 inline path, which reads its
/// thread count but never injects work. Without this, that read lazily
/// builds a full-size global pool — a whole redundant set of idle workers.
/// turbovec's rayon-core is statically linked and private to this extension,
/// so the sentinel cannot affect other rayon users (polars, tokenizers) in
/// the process.
fn pin_global_pool_sentinel() {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();
}

/// Register the fork-detection handlers. The Python hook is authoritative;
/// pthread_atfork is a Unix belt-and-suspenders for non-Python forks.
fn register_fork_handlers(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(unix)]
    // SAFETY: pthread_atfork with a child handler that only does an atomic
    // increment (async-signal-safe). Registered once at module init.
    unsafe {
        libc::pthread_atfork(None, None, Some(atfork_child_handler));
    }
    // os.register_at_fork exists on platforms with fork (Unix); on Windows
    // it is absent and there is no fork, so skipping it is correct.
    let os = py.import("os")?;
    if let Ok(register) = os.getattr("register_at_fork") {
        let func = module.getattr("_note_fork_in_child")?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("after_in_child", func)?;
        register.call((), Some(&kwargs))?;
    }
    Ok(())
}

/// Cap applied to an explicit `RAYON_NUM_THREADS` request. Threads
/// beyond a small multiple of the hardware parallelism add no
/// throughput for turbovec's CPU-bound kernels, and a request past the
/// OS thread limit (`ulimit -u`) makes rayon's lazy pool construction
/// panic — surfacing as an opaque, uncatchable `PanicException` on the
/// first `add`/`search` (issue #158). 4x leaves room for deliberate
/// oversubscription; 1024 is the fallback when the hardware
/// parallelism cannot be determined.
fn rayon_thread_cap() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_mul(4))
        .unwrap_or(1024)
}

/// Build the global rayon pool eagerly at module init when
/// `RAYON_NUM_THREADS` is set, clamping over-large values to
/// [`rayon_thread_cap`] (with a `RuntimeWarning` naming the variable).
///
/// When the variable is unset — or holds a value rayon itself would
/// ignore (unparseable) or treat as "auto" (`0`) — this does nothing,
/// so rayon's lazy auto-sized initialization is preserved exactly.
/// Values at or under the cap eagerly build the pool with the same
/// thread count rayon's lazy path would have chosen, so results are
/// unchanged. Pool-build failure never fails the import.
fn init_rayon_pool(py: Python<'_>) -> PyResult<()> {
    let Ok(raw) = std::env::var("RAYON_NUM_THREADS") else {
        return Ok(());
    };
    // Mirror rayon's own parsing (plain `usize::from_str`, no trim):
    // anything rayon would ignore, we ignore; 0 means "auto" to rayon.
    let Ok(requested) = raw.parse::<usize>() else {
        return Ok(());
    };
    if requested == 0 {
        return Ok(());
    }
    let cap = rayon_thread_cap();
    let n = requested.min(cap);
    if n < requested {
        let warnings = py.import("warnings")?;
        warnings.call_method1(
            "warn",
            (
                format!(
                    "RAYON_NUM_THREADS={requested} exceeds turbovec's thread cap \
                     of {cap} (4x available parallelism); capping at {cap} threads",
                ),
                py.get_type::<pyo3::exceptions::PyRuntimeWarning>(),
            ),
        )?;
    }
    // Eagerly build the process-*local* pool (issue #147). The pre-#147
    // code built rayon's *global* pool here, which is exactly what poisoned
    // a child forked after import when `RAYON_NUM_THREADS` was set: the dead
    // global pool was inherited and the child's first parallel op hung. The
    // local pool is fork-safe — a child rebuilds it on first use — and it
    // honors `RAYON_NUM_THREADS` via `desired_threads` (`n` is validated and
    // clamped above only to drive the warning; `desired_threads` re-derives
    // the same count).
    //
    // A build failure here is deliberately SWALLOWED so import never fails
    // (preserving #158's "import never fails" guarantee): if the pool cannot
    // be built now, nothing is published and the next kernel call retries via
    // `current_pool`, raising a catchable `RuntimeError` then if it still
    // fails — at call time, not import time.
    let _ = n;
    let _ = current_pool();
    Ok(())
}

/// Route the core crate's non-fatal diagnostics — currently only the
/// post-commit durability shortfall (#365) — into Python's `warnings`
/// machinery instead of the crate's stderr default. That makes them
/// filterable (`warnings.simplefilter`), capturable by
/// `logging.captureWarnings(True)`, and visible to `pytest.warns`, none
/// of which a bare `eprintln!` from a library can be.
///
/// Attaching to the GIL here is safe from any thread: this file's lock
/// discipline (see the note above `TurboQuantIndex`) is that the index
/// lock is only ever *waited on* with the GIL released, so no thread can
/// be holding the GIL while blocked on the lock we hold.
///
/// What is *not* safe is delivering while the save still holds the read
/// guard, which is exactly when the core emits this: `warnings.warn`
/// dispatches through the filter chain into a user-replaceable
/// `showwarning`, and a handler that touches the same index asks for the
/// write lock while the save read-holds it — an unconditional deadlock,
/// the same defect the warm-up warning had (#360). So while a save is in
/// flight the message is queued and [`flush_core_warnings`] delivers it
/// once the guard is gone. Outside a save (no deferral open) there is no
/// guard to wait on and the message goes straight out.
fn emit_core_warning(message: &str) {
    if CORE_WARNINGS_DEFERRED.load(std::sync::atomic::Ordering::Acquire) > 0 {
        pending_core_warnings().push(message.to_owned());
        return;
    }
    Python::attach(|py| deliver_core_warning(py, message));
}

/// Emit one queued core message, routing a filter-raised error to
/// `sys.unraisablehook`: turning it into an exception is the caller's
/// business — it must not corrupt an already-committed save, and there is
/// no `PyResult` to return it through from here.
fn deliver_core_warning(py: Python<'_>, message: &str) {
    if let Err(e) = emit_runtime_warning(py, message) {
        e.write_unraisable(py, None);
    }
}

/// Core messages emitted while a save held the index read lock, waiting
/// for [`flush_core_warnings`].
static PENDING_CORE_WARNINGS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// Number of [`DeferCoreWarnings`] regions currently open, across all
/// threads. A count rather than a flag because saves run concurrently.
static CORE_WARNINGS_DEFERRED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// The pending queue, recovering from poisoning for the same reason
/// [`lock_read`] does: a panic that escaped an earlier warning must not
/// turn every later save into a panic.
fn pending_core_warnings() -> std::sync::MutexGuard<'static, Vec<String>> {
    PENDING_CORE_WARNINGS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// While alive, [`emit_core_warning`] queues instead of delivering. Held
/// across the whole detached section of a save — that is, for exactly as
/// long as the index read guard exists.
struct DeferCoreWarnings;

impl DeferCoreWarnings {
    fn new() -> Self {
        CORE_WARNINGS_DEFERRED.fetch_add(1, std::sync::atomic::Ordering::Release);
        Self
    }
}

impl Drop for DeferCoreWarnings {
    fn drop(&mut self) {
        CORE_WARNINGS_DEFERRED.fetch_sub(1, std::sync::atomic::Ordering::Release);
    }
}

/// Deliver everything [`emit_core_warning`] queued, from a caller that
/// holds no index guard. Called by every save that opened a deferral,
/// before it warns about warm-up, which keeps the order a save's two
/// warnings arrive in (durability first, then warm-up).
///
/// Nothing can be stranded in the queue: the only producer is the
/// post-commit fsync inside `write_with_durability`, which runs inside
/// the `with_pool` call the saving thread is blocked on, so a queued
/// message is always pushed before that save's own flush. Concurrent
/// saves share one queue, so a message can be delivered on a different
/// save's thread than the one that produced it — it names its own path,
/// and both saves flush, so it is delivered exactly once either way.
fn flush_core_warnings(py: Python<'_>) {
    let queued = std::mem::take(&mut *pending_core_warnings());
    for message in queued {
        deliver_core_warning(py, &message);
    }
}

/// Number of stack frames [`emit_runtime_warning`] is willing to walk out
/// of looking for the caller. Every real chain is two or three frames
/// (`store.dump` → `_persist.atomic_save` → the binding), so this only
/// bounds the pathological case.
const MAX_STACKLEVEL: usize = 16;

/// Emit `message` as a `RuntimeWarning` through Python's `warnings`
/// machinery, attributed to the first frame *outside* the `turbovec`
/// package.
///
/// The single emitter for every diagnostic this extension raises —
/// the core crate's hook (see [`emit_core_warning`]) and the warm-up
/// serialization warning both come through here, so they are uniformly
/// filterable (`warnings.simplefilter`), capturable by
/// `logging.captureWarnings(True)`, and visible to `pytest.warns`, none
/// of which a bare `eprintln!` from a library can be.
///
/// A Rust extension frame is not a `stacklevel`, so the default
/// `stacklevel=1` credits the nearest *Python* frame. For a save routed
/// through `turbovec._persist.atomic_save` — which is how all four
/// integration stores persist — that is a turbovec internal the user
/// never wrote, and `__warningregistry__` gets keyed there too, so a
/// per-module `once` filter dedupes across unrelated call sites (#366).
/// Walking out to the first non-turbovec frame points the warning at the
/// `dump()` / `persist()` / `write()` call the user actually made, and
/// falls back to `stacklevel=1` (today's behaviour) whenever the walk
/// cannot resolve a frame.
fn emit_runtime_warning(py: Python<'_>, message: &str) -> PyResult<()> {
    let warnings = py.import("warnings")?;
    let category = py.get_type::<pyo3::exceptions::PyRuntimeWarning>();
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("stacklevel", caller_stacklevel(py))?;
    warnings.call_method("warn", (message, category), Some(&kwargs))?;
    Ok(())
}

/// The `stacklevel` that names the first frame outside the `turbovec`
/// package, counting from 1 for the immediate Python caller.
///
/// Returns 1 — the previous unconditional behaviour — if `sys`, the
/// package location, or a frame cannot be resolved, so a warning is never
/// lost to a failure of the attribution machinery.
fn caller_stacklevel(py: Python<'_>) -> usize {
    // Every failure path falls through to the `1` below rather than
    // returning its own value, so there is exactly one fallback.
    if let (Some(pkg_dir), Ok(sys)) = (turbovec_package_dir(py), py.import("sys")) {
        for level in 1..=MAX_STACKLEVEL {
            // `_getframe(0)` is the innermost Python frame, i.e.
            // `stacklevel=1`. It raises once the walk runs off the top of
            // the stack, and on a thread with no Python frames at all —
            // a rayon worker emitting the core's durability warning.
            let filename = sys
                .call_method1("_getframe", (level - 1,))
                .and_then(|f| f.getattr("f_code"))
                .and_then(|c| c.getattr("co_filename"))
                .and_then(|f| f.extract::<String>());
            let Ok(filename) = filename else { break };
            if !filename.starts_with(&pkg_dir) {
                return level;
            }
        }
    }
    1
}

/// Directory holding the installed `turbovec` package, with a trailing
/// separator so it only prefix-matches files *inside* it. `None` when the
/// package has no resolvable filesystem location (a namespace package, a
/// zipimport, a frozen build).
fn turbovec_package_dir(py: Python<'_>) -> Option<String> {
    let file: String = py
        .import("turbovec")
        .and_then(|m| m.getattr("__file__"))
        .and_then(|f| f.extract())
        .ok()?;
    let parent = std::path::Path::new(&file).parent()?;
    let mut dir = parent.to_str()?.to_string();
    dir.push(std::path::MAIN_SEPARATOR);
    Some(dir)
}

// ---- the partitioned incremental index, ported from the prototype ----
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
    #[pyo3(signature = (dim=None, bit_width=4, target_partition_size=None, store_vectors=false, replica_epsilon=None, replica_prune=true))]
    fn new(
        dim: Option<usize>,
        bit_width: usize,
        target_partition_size: Option<usize>,
        store_vectors: bool,
        replica_epsilon: Option<f32>,
        replica_prune: bool,
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
        inner.set_replica_prune(replica_prune);
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
        let (scores, ids) = py.detach(move || {
            // The process-local pool, not the global one: module init pins the
            // global pool to a single thread, so a bare detach leaves the whole
            // scan single-threaded.
            with_pool(move || {
                index.search_with_options(
                    &owned_queries,
                    k,
                    turbovec_core::SearchOptions {
                        nprobe,
                        probe_epsilon,
                        rescore_k,
                    },
                )
            })
        })?;
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

impl FreshIndex {
    fn lock_read(&self) -> std::sync::RwLockReadGuard<'_, turbovec_core::FreshIndex> {
        self.inner.read().expect("FreshIndex lock poisoned")
    }

    fn lock_write(&self) -> std::sync::RwLockWriteGuard<'_, turbovec_core::FreshIndex> {
        self.inner.write().expect("FreshIndex lock poisoned")
    }
}
#[pyclass]
struct FreshIndex {
    /// Behind an RwLock so a query and a maintenance pass overlap.
    /// pyo3's &self/&mut self rules otherwise make them mutually
    /// exclusive, and the failure is a Python 'already borrowed'
    /// exception rather than a wait -- worse than blocking.
    inner: std::sync::Arc<std::sync::RwLock<turbovec_core::FreshIndex>>,
    /// The read path does not go through `inner` at all. A reader tracks the
    /// index's publication cell, so a query loads the current snapshot and
    /// scans it while a save or maintain pass is midway through building the
    /// next one. Holding the writer's read lock for the duration of a scan --
    /// what this used to do -- meant every query still waited out whatever
    /// bounded chunk of maintenance happened to be running.
    reader: turbovec_core::FreshReader,
}

impl FreshIndex {
    fn wrap(inner: turbovec_core::FreshIndex) -> Self {
        let reader = inner.reader();
        Self {
            inner: std::sync::Arc::new(std::sync::RwLock::new(inner)),
            reader,
        }
    }
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
    /// The maintenance knobs (`reassign_neighbors`, `balanced_split`,
    /// `resplit_after_reassign`, `replica_prune`) default to the values the
    /// index has always used; they exist so the choices can be measured on
    /// this implementation rather than on a reimplementation of it.
    #[new]
    #[pyo3(signature = (dim=None, bit_width=4, target_partition_size=None, store_vectors=false, replica_epsilon=None, reassign_neighbors=None, balanced_split=None, resplit_after_reassign=None, replica_prune=None, split_enabled=None, dissolve_enabled=None, reassign_enabled=None, rebootstrap_enabled=None, gc_max_chunks=None, gc_dead_ratio=None, tier_merge_enabled=None, gc_garbage_ratio=None, staging_threshold=None, hierarchical_assign=None, max_rewrites_per_flush=None, max_reassign_partitions=None, defer_maintenance=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: Option<usize>,
        bit_width: usize,
        target_partition_size: Option<usize>,
        store_vectors: bool,
        replica_epsilon: Option<f32>,
        reassign_neighbors: Option<usize>,
        balanced_split: Option<bool>,
        resplit_after_reassign: Option<bool>,
        replica_prune: Option<bool>,
        split_enabled: Option<bool>,
        dissolve_enabled: Option<bool>,
        reassign_enabled: Option<bool>,
        rebootstrap_enabled: Option<bool>,
        gc_max_chunks: Option<usize>,
        gc_dead_ratio: Option<f64>,
        tier_merge_enabled: Option<bool>,
        gc_garbage_ratio: Option<f64>,
        staging_threshold: Option<usize>,
        hierarchical_assign: Option<bool>,
        max_rewrites_per_flush: Option<usize>,
        max_reassign_partitions: Option<usize>,
        defer_maintenance: Option<bool>,
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
        if reassign_neighbors == Some(0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "reassign_neighbors must be positive",
            ));
        }
        let mut tuning = inner.tuning();
        if let Some(v) = reassign_neighbors {
            tuning.reassign_neighbors = v;
        }
        if let Some(v) = balanced_split {
            tuning.balanced_split = v;
        }
        if let Some(v) = resplit_after_reassign {
            tuning.resplit_after_reassign = v;
        }
        if let Some(v) = replica_prune {
            tuning.replica_prune = v;
        }
        if let Some(v) = split_enabled {
            tuning.split_enabled = v;
        }
        if let Some(v) = dissolve_enabled {
            tuning.dissolve_enabled = v;
        }
        if let Some(v) = reassign_enabled {
            tuning.reassign_enabled = v;
        }
        if let Some(v) = rebootstrap_enabled {
            tuning.rebootstrap_enabled = v;
        }
        if let Some(v) = gc_max_chunks {
            tuning.gc_max_chunks = v;
        }
        if let Some(v) = hierarchical_assign {
            tuning.hierarchical_assign = v;
        }
        if let Some(v) = staging_threshold {
            tuning.staging_threshold = v;
        }
        if let Some(v) = tier_merge_enabled {
            tuning.tier_merge_enabled = v;
        }
        if let Some(v) = gc_garbage_ratio {
            tuning.gc_garbage_ratio = v;
        }
        if let Some(v) = gc_dead_ratio {
            tuning.gc_dead_ratio = v;
        }
        if let Some(v) = max_rewrites_per_flush {
            tuning.max_rewrites_per_flush = v;
        }
        if let Some(v) = max_reassign_partitions {
            tuning.max_reassign_partitions = v;
        }
        if let Some(v) = defer_maintenance {
            tuning.defer_maintenance = v;
        }
        inner.set_tuning(tuning);
        inner.set_partitioning(target_partition_size);
        inner.set_replication(replica_epsilon);
        inner.set_store_vectors(store_vectors);
        Ok(Self::wrap(inner))
    }

    /// Open an index directory previously produced by `save`. Replays the
    /// write-ahead log (mutations since the last save survive crashes) and
    /// cleans up after any interrupted save.
    #[classmethod]
    fn open(_cls: &Bound<PyType>, directory: &str) -> PyResult<Self> {
        let inner = turbovec_core::FreshIndex::open(directory)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
        Ok(Self::wrap(inner))
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
        Ok(Self::wrap(inner))
    }

    /// Write the live vectors' codes to a `.tvim` `IdMapIndex` file
    /// (requires a saved, fully-flushed index).
    fn export_id_map_file(&self, dst: &str) -> PyResult<()> {
        self.lock_read()
            .export_id_map_file(dst)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))
    }

    /// Add vectors with external ids (buffered in RAM + write-ahead log;
    /// appended to partitions at the next `save`).
    fn add_with_ids(
        &self,
        py: Python<'_>,
        vectors: PyReadonlyArray2<f32>,
        ids: PyReadonlyArray1<u64>,
    ) -> PyResult<()> {
        let v = vectors.as_array();
        let dim = v.ncols();
        let v_slice = v.as_slice().ok_or_else(|| not_contiguous_err("vectors"))?;
        let i = ids.as_array();
        let i_slice = i.as_slice().ok_or_else(|| not_contiguous_err("ids"))?;
        // Copy out of numpy first, then drop the GIL for the encode. Holding
        // it here would block every concurrent query for the whole add, which
        // is a Python-level stall no amount of lock granularity can fix.
        let owned_v = v_slice.to_vec();
        let owned_i = i_slice.to_vec();
        let inner = std::sync::Arc::clone(&self.inner);
        py.detach(move || {
            with_pool(move || {
                inner
                    .write()
                    .expect("FreshIndex lock poisoned")
                    .add_with_ids_2d(&owned_v, dim, &owned_i)
            })
        })?
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Remove the vector with external id `id`; every stored copy is
    /// hidden immediately and reclaimed at a later save. Returns `True`
    /// if the id was present.
    fn remove(&self, py: Python<'_>, id: u64) -> bool {
        let inner = std::sync::Arc::clone(&self.inner);
        py.detach(move || inner.write().expect("FreshIndex lock poisoned").remove(id))
    }

    /// Remove many ids in one call, syncing the write-ahead log once.
    ///
    /// Same durability at the call boundary as `remove` -- everything is on
    /// disk when this returns -- but one fsync instead of one per id. A loop
    /// of single `remove` calls spends 88% of its time in fsync.
    /// Returns how many of the ids were present.
    fn remove_many(&self, py: Python<'_>, ids: PyReadonlyArray1<u64>) -> PyResult<usize> {
        let i = ids.as_array();
        let owned: Vec<u64> = i
            .as_slice()
            .ok_or_else(|| not_contiguous_err("ids"))?
            .to_vec();
        let inner = std::sync::Arc::clone(&self.inner);
        Ok(py.detach(move || {
            inner
                .write()
                .expect("FreshIndex lock poisoned")
                .remove_many(&owned)
        }))
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
        if let Some(idx_dim) = self.reader.dim_opt() {
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
        if matches!(rescore_k, Some(r) if r > 0) && !self.reader.stores_vectors() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rescore_k requires an index built with store_vectors=True",
            ));
        }

        let owned_queries = q_slice.to_vec();
        // The reader is cloned, not the index: no lock on `inner` is taken at
        // any point, so this cannot wait behind a save or a maintain.
        let reader = self.reader.clone();
        let (scores, ids) = py.detach(move || {
            // The process-local pool, not the global one: module init pins the
            // global pool to a single thread, so a bare detach leaves the whole
            // scan single-threaded.
            with_pool(move || {
                reader.search_with_options(
                    &owned_queries,
                    k,
                    turbovec_core::SearchOptions {
                        nprobe,
                        probe_epsilon,
                        rescore_k,
                    },
                )
            })
        })?;
        let effective_k = if nq == 0 {
            k.min(self.reader.len())
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
        self.reader.contains(id)
    }

    /// The stored full-precision vectors of the given live ids (requires
    /// `store_vectors=True`); raises `KeyError` if any id is not present.
    fn get_vectors<'py>(
        &self,
        py: Python<'py>,
        ids: PyReadonlyArray1<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if !self.reader.stores_vectors() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "get_vectors requires an index built with store_vectors=True",
            ));
        }
        let ids_arr = ids.as_array();
        let ids_slice = ids_arr.as_slice().ok_or_else(|| not_contiguous_err("ids"))?;
        let dim = self.reader.dim_opt().unwrap_or(0);
        let mut out = Vec::with_capacity(ids_slice.len() * dim);
        for &id in ids_slice {
            let vector = self.reader.get_vector(id).ok_or_else(|| {
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
        self.reader.prepare();
    }

    /// Flush to `directory` (bound on first call): buffered vectors are
    /// appended to their partitions, local maintenance runs, and a new
    /// manifest is atomically published. Only touched partitions' files
    /// change — the page cache for the rest stays valid.
    /// Run one bounded unit of deferred maintenance and publish it.
    ///
    /// Takes the write lock only for this bounded chunk, so concurrent
    /// searches wait at most one chunk rather than a whole pass. Returns True
    /// while work may remain, so callers can loop until it returns False.
    fn maintain(&self, py: Python<'_>) -> PyResult<bool> {
        let inner = std::sync::Arc::clone(&self.inner);
        py.detach(move || {
            with_pool(move || inner.write().expect("FreshIndex lock poisoned").maintain())
        })?
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))
    }

    /// True when some partition is outside the size bound.
    #[getter]
    fn needs_maintenance(&self) -> bool {
        self.lock_read().needs_maintenance()
    }

    fn save(&self, py: Python<'_>, directory: &str) -> PyResult<()> {
        let inner = std::sync::Arc::clone(&self.inner);
        let dir = directory.to_string();
        py.detach(move || {
            // `with_pool`, not the bare closure: module init pins rayon's
            // GLOBAL pool to one thread and runs real work in a process-local
            // one. Without this the whole write path — assign, decode,
            // maintenance — is single-threaded. Measured on the assign GEMM:
            // 24 GMAC/s against 130 for the same shape.
            with_pool(move || {
                inner
                    .write()
                    .expect("FreshIndex lock poisoned")
                    .save(&dir)
            })
        })?
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))
    }

    /// Test-only: drop mapped segment pages so the next search pays real I/O.
    /// Cold latency is otherwise unmeasurable without privileges to purge the
    /// system page cache.
    fn evict_page_cache(&self) {
        self.reader.evict_page_cache();
    }

    fn __len__(&self) -> usize {
        self.reader.len()
    }

    fn __contains__(&self, id: u64) -> bool {
        self.reader.contains(id)
    }

    fn __repr__(&self) -> String {
        let dim = self
            .reader
            .dim_opt()
            .map_or_else(|| "None".to_string(), |d| d.to_string());
        format!(
            "turbovec.FreshIndex(dim={}, bit_width={}, n_vectors={}, nlist={}, \
             memtable={}, dead={}, replicas={}, runs={}, chunks={})",
            dim,
            self.reader.bit_width(),
            self.reader.len(),
            self.reader.nlist(),
            self.reader.memtable_len(),
            self.reader.dead_count(),
            self.reader.replica_count(),
            self.reader.run_count(),
            self.reader.chunk_count(),
        )
    }

    #[getter]
    fn dim(&self) -> Option<usize> {
        self.reader.dim_opt()
    }

    #[getter]
    fn bit_width(&self) -> usize {
        self.reader.bit_width()
    }

    #[getter]
    fn nlist(&self) -> usize {
        self.reader.nlist()
    }

    #[getter]
    fn target_partition_size(&self) -> Option<usize> {
        self.lock_read().partition_target()
    }

    #[setter]
    fn set_target_partition_size(&self, target: Option<usize>) -> PyResult<()> {
        if target == Some(0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_partition_size must be positive",
            ));
        }
        self.lock_write().set_partitioning(target);
        Ok(())
    }

    #[getter]
    fn replica_epsilon(&self) -> Option<f32> {
        self.lock_read().replica_epsilon()
    }

    #[setter]
    fn set_replica_epsilon(&self, epsilon: Option<f32>) -> PyResult<()> {
        if let Some(e) = epsilon {
            if !(e.is_finite() && e > 0.0) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "replica_epsilon must be finite and positive",
                ));
            }
        }
        self.lock_write().set_replication(epsilon);
        Ok(())
    }

    #[getter]
    fn store_vectors(&self) -> bool {
        self.reader.stores_vectors()
    }

    /// Rows buffered in RAM since the last save. Diagnostic.
    #[getter]
    fn memtable_len(&self) -> usize {
        self.reader.memtable_len()
    }

    /// Dead rows awaiting compaction. Diagnostic.
    #[getter]
    fn dead_count(&self) -> usize {
        self.reader.dead_count()
    }

    /// Live closure-assignment replica rows. Diagnostic.
    #[getter]
    fn replica_count(&self) -> usize {
        self.reader.replica_count()
    }

    /// Live primary rows per partition. The mean is pinned near the target
    /// by construction; the max and upper percentiles are what show whether
    /// splitting keeps postings balanced. Diagnostic.
    #[getter]
    fn partition_sizes(&self) -> Vec<usize> {
        self.reader.partition_sizes()
    }

    /// What maintenance actually did, cumulatively — so that "this knob
    /// changed nothing" can be told apart from "the work this knob governs
    /// never happened". Diagnostic.
    #[getter]
    fn maintenance_stats(&self) -> std::collections::HashMap<String, f64> {
        let s = self.lock_read().maintenance_stats();
        [
            ("splits_attempted", s.splits_attempted as f64),
            ("splits_done", s.splits_done as f64),
            ("splits_degenerate", s.splits_degenerate as f64),
            ("balance_fired", s.balance_fired as f64),
            ("split_ratio_raw_sum", s.split_ratio_raw_sum),
            ("split_ratio_final_sum", s.split_ratio_final_sum),
            ("dissolves", s.dissolves as f64),
            ("reassign_passes", s.reassign_passes as f64),
            ("reassign_partitions", s.reassign_partitions as f64),
            ("reassign_rows_examined", s.reassign_rows_examined as f64),
            ("reassign_rows_moved", s.reassign_rows_moved as f64),
            ("resplit_passes", s.resplit_passes as f64),
            ("rebootstraps", s.rebootstraps as f64),
            ("compactions", s.compactions as f64),
            ("us_drain", s.us_drain as f64),
            ("us_assign", s.us_assign as f64),
            ("us_append", s.us_append as f64),
            ("us_maintenance", s.us_maintenance as f64),
            ("us_publish", s.us_publish as f64),
            ("us_add_lock", s.us_add_lock as f64),
            ("us_memtable_scan", s.us_memtable_scan as f64),
            ("us_decode", s.us_decode as f64),
            ("us_kmeans_gemm", s.us_kmeans_gemm as f64),
            ("run_merges", s.run_merges as f64),
            ("us_run_merge", s.us_run_merge as f64),
            ("bytes_ingest", s.bytes_ingest as f64),
            ("bytes_compact", s.bytes_compact as f64),
            ("bytes_split", s.bytes_split as f64),
            ("bytes_replica", s.bytes_replica as f64),
            ("bytes_import", s.bytes_import as f64),
            ("rows_ingest", s.rows_ingest as f64),
            ("rows_compact", s.rows_compact as f64),
            ("rows_split", s.rows_split as f64),
            ("rows_replica", s.rows_replica as f64),
            ("chunks_ingest", s.chunks_ingest as f64),
            ("chunks_compact", s.chunks_compact as f64),
            ("chunks_split", s.chunks_split as f64),
            ("chunks_replica", s.chunks_replica as f64),
            ("bytes_tier", s.bytes_tier as f64),
            ("rows_tier", s.rows_tier as f64),
            ("chunks_tier", s.chunks_tier as f64),
            ("tier_merges", s.tier_merges as f64),
            ("staged_saves", s.staged_saves as f64),
            ("us_assign_decode", s.us_assign_decode as f64),
            ("us_assign_gemm", s.us_assign_gemm as f64),
            ("assign_rows", s.assign_rows as f64),
            ("assign_nlist", s.assign_nlist as f64),
            ("coarse_rebuilds", s.coarse_rebuilds as f64),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }

    /// Id-run tables. Diagnostic.
    #[getter]
    fn run_count(&self) -> usize {
        self.reader.run_count()
    }

    /// Total chunks across all segment files. Diagnostic.
    #[getter]
    fn chunk_count(&self) -> usize {
        self.reader.chunk_count()
    }

    /// Backing directory, or ``None`` before the first save.
    #[getter]
    fn path(&self) -> Option<String> {
        self.lock_read()
            .path()
            .map(|p| p.to_string_lossy().into_owned())
    }
}


#[pymodule]
fn _turbovec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Pin the 1-thread global sentinel BEFORE anything can touch rayon, so
    // the nq==1 inline path never lazily builds a full global pool.
    pin_global_pool_sentinel();
    turbovec_core::set_warning_hook(Some(emit_core_warning));
    m.add_function(wrap_pyfunction!(_note_fork_in_child, m)?)?;
    register_fork_handlers(m.py(), m)?;
    init_rayon_pool(m.py())?;
    m.add_class::<TurboQuantIndex>()?;
    m.add_class::<IdMapIndex>()?;
    m.add_class::<DiskIndex>()?;
    m.add_class::<FreshIndex>()?;
    Ok(())
}

#[cfg(test)]
mod snap_retention_tests {
    //! Policy tests for [`retain_snap`] (#333): they drive it through the
    //! buffer sequence `add` performs, over the batch-size shapes the
    //! retention target has to get right (one-shot bulk, steady, growing,
    //! jittering, step-up, spike). They deliberately do *not* cover the
    //! wiring — deleting the `retain_snap` call in `add` leaves every test
    //! here green, because none of them go through `add`.
    //!
    //! That wiring is pinned from Python instead, in
    //! `tests/test_snap_retention.py`, via `_snap_capacity`: reaching the
    //! real call site from a Rust test would need a live interpreter and a
    //! numpy round trip inside a crate built as an extension module, and
    //! the buffer has no Python-visible handle otherwise.

    use super::retain_snap;
    use std::sync::atomic::AtomicUsize;

    /// One `add`'s worth of buffer handling: the snapshot is refilled to
    /// the input length, then passed to `retain_snap` on the way back
    /// into the mutex.
    fn one_add(snap: &mut Vec<f32>, prev: &AtomicUsize, len: usize) {
        snap.clear();
        snap.reserve(len);
        snap.resize(len, 0.0);
        retain_snap(snap, prev, len);
    }

    #[test]
    fn one_shot_bulk_add_releases_the_snapshot() {
        let big = 8 << 20;
        let prev = AtomicUsize::new(0);
        let mut snap = Vec::new();
        one_add(&mut snap, &prev, big);
        assert!(
            snap.capacity() < big / 4,
            "one-shot bulk add retained {} snapshot elements (input was {big})",
            snap.capacity(),
        );
    }

    #[test]
    fn repeated_same_size_adds_keep_the_snapshot_warm() {
        let big = 8 << 20;
        let prev = AtomicUsize::new(0);
        let mut snap = Vec::new();
        for _ in 0..3 {
            one_add(&mut snap, &prev, big);
        }
        assert!(
            snap.capacity() >= big,
            "steady same-size adds dropped the warm snapshot to {} elements (need {big})",
            snap.capacity(),
        );
    }

    /// Shrinking to the bare previous length would leave a growing batch
    /// size with no headroom, so every add would grow and be shrunk
    /// again. See the core's `growing_batch_sizes_keep_their_growth_headroom`.
    #[test]
    fn growing_batch_sizes_keep_their_growth_headroom() {
        let prev = AtomicUsize::new(0);
        let mut snap = Vec::new();
        let mut len = 1 << 20;
        let mut last = 0;
        for _ in 0..6 {
            one_add(&mut snap, &prev, len);
            last = len;
            len += len / 20; // +5% per batch
        }
        assert!(
            snap.capacity() >= last,
            "a growing batch size left only {} snapshot elements after a \
             {last}-element add",
            snap.capacity(),
        );
    }

    /// A batch that steps up sharply and then holds must not have the
    /// step shrunk away underneath it — the hysteresis alone does not
    /// cover a 3x step. See the core's
    /// `a_step_up_in_batch_size_is_not_shrunk_back`.
    #[test]
    fn a_step_up_in_batch_size_is_not_shrunk_back() {
        let small = 2 << 20;
        let big = 3 * small;
        let prev = AtomicUsize::new(0);
        let mut snap = Vec::new();
        one_add(&mut snap, &prev, small);
        one_add(&mut snap, &prev, big);
        assert!(
            snap.capacity() >= big,
            "a {small}->{big} step left only {} snapshot elements",
            snap.capacity(),
        );
    }

    /// Retention must not equal the largest single add ever made: a
    /// spike in a run of smaller batches has to be given back. See the
    /// core's `a_one_off_spike_does_not_stay_pinned`.
    #[test]
    fn a_one_off_spike_does_not_stay_pinned() {
        let n = 2 << 20;
        let prev = AtomicUsize::new(0);
        let mut snap = Vec::new();
        one_add(&mut snap, &prev, n);
        one_add(&mut snap, &prev, 4 * n);
        one_add(&mut snap, &prev, n);
        assert!(
            snap.capacity() < 4 * n,
            "a 4x spike left {} snapshot elements pinned after the batch \
             size dropped back to {n}",
            snap.capacity(),
        );
    }

    #[test]
    fn jittering_batch_sizes_keep_their_growth_headroom() {
        let prev = AtomicUsize::new(0);
        let mut snap = Vec::new();
        let sizes = [
            7_200_000, 8_800_000, 7_600_000, 8_640_000, 7_360_000, 8_320_000,
        ];
        for len in sizes {
            one_add(&mut snap, &prev, len);
        }
        assert!(
            snap.capacity() >= 8_320_000,
            "a jittering batch size left only {} snapshot elements",
            snap.capacity(),
        );
    }
}
