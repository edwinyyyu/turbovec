//! Incrementally-updatable disk index: the SPFresh storage model on top of
//! turbovec's quantized scan.
//!
//! [`FreshIndex`] answers the same queries as [`DiskIndex`](crate::DiskIndex)
//! with the same kernels, routing, and recall levers, but replaces the
//! monolithic `.tvdm` file with a **directory** in which every partition is
//! its own append-only segment file. The unit of rewrite shrinks from the
//! whole index to one partition:
//!
//! * **Insert**: buffered in an in-RAM memtable (searched exhaustively, so
//!   recent vectors have *exact* recall) and logged to a write-ahead log
//!   for crash durability; a flush ([`FreshIndex::save`]) appends each
//!   buffered vector to its nearest partition's segment as a new chunk.
//! * **Delete**: the id's physical copies are located through the id-run
//!   tables and dead-marked in per-partition bitmaps — no global tombstone
//!   set, no overfetch creep; dead rows are dropped the next time their
//!   partition is rewritten.
//! * **Maintenance is local** (LIRE): a partition that outgrows
//!   `2 * target` is split; one that shrinks below `target / 4` is
//!   dissolved into its neighbors; members of changed partitions and their
//!   nearest neighbors are reassigned when strictly closer; centroids of
//!   changed partitions are refreshed; partitions with too many dead rows
//!   or too many small chunks are compacted — each operation rewrites only
//!   the partitions involved. The re-clustering escape hatch runs as a
//!   gated background-style check (held-out acceptance test) once enough
//!   churn has accumulated.
//!
//! Because untouched partitions' segment files are never rewritten (and
//! appends never move existing bytes), the OS page cache stays valid across
//! flushes — the post-save cold window of the monolithic design disappears.
//!
//! # On-disk layout (directory)
//!
//! ```text
//! <dir>/manifest                  authoritative state, atomically replaced
//!                                 (epoch, calibration, centroids, partition
//!                                 table + chunk tables + dead bitmaps,
//!                                 id-run list; crc32-checked)
//! <dir>/wal                       append-only add/remove log since the
//!                                 manifest's epoch (crc per record)
//! <dir>/segment-<id>-<gen>        one per partition: a sequence of chunks,
//!                                 each holding blocked codes, scales, ids,
//!                                 a replica bitmap, and (optionally) f32
//!                                 vectors — mmap'd and scanned in place
//! <dir>/ids-<gen>                 sorted (id -> partition, row) runs; the
//!                                 LSM-style membership structure
//! ```
//!
//! Crash consistency: segment appends and run files are written and synced
//! *before* the manifest is atomically renamed over; a crash before the
//! rename leaves the old manifest pointing at the old state (orphaned tail
//! bytes are truncated at open). The WAL is tagged with the manifest epoch
//! it applies to: after a successful flush the old WAL is obsolete and
//! recreated; after a crash a matching-epoch WAL is replayed.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use memmap2::Mmap;

use crate::disk::{
    self, closure_assignments_for_vectors, clustering_sample, f32_bytes, route_queries,
    BlockStreamer, SearchOptions, CLUSTERING_CHUNK, DEFAULT_EPSILON_NPROBE_DIVISOR,
    DEFAULT_NPROBE_DIVISOR, DEFAULT_NPROBE_MIN, DEFAULT_RESCORE_MULTIPLIER, HOLDOUT_STRIDE,
    KMEANS_ITERATIONS, KMEANS_SEED, MERGE_DIVISOR, MIN_PARTITIONS, NEIGHBOR_PARTITIONS,
    REBOOTSTRAP_CANDIDATES, REBOOTSTRAP_DISTORTION_RATIO, SPLIT_FACTOR,
};
use crate::id_map::IdMapIndex;
use crate::{
    codebook, decode, first_invalid_coord, kmeans, pack, rotation, AddError, ConstructError,
    TurboQuantIndex, BLOCK,
};

const MANIFEST_MAGIC: &[u8; 4] = b"TVFM";
const WAL_MAGIC: &[u8; 4] = b"TVFW";
const RUN_MAGIC: &[u8; 4] = b"TVFR";
const CHUNK_MAGIC: &[u8; 4] = b"TVFC";
const FORMAT_VERSION: u8 = 1;
const FLAG_HAS_VECTORS: u8 = 1;
const FLAG_CLUSTERED: u8 = 2;

const MANIFEST_FILE: &str = "manifest";
const MANIFEST_TEMP_FILE: &str = "manifest.tmp";
const WAL_FILE: &str = "wal";

/// Section alignment inside segment files; also the chunk header size.
const CHUNK_ALIGN: usize = 64;
const SECTION_ALIGN: usize = 8;

/// Merge the id-run tables down to one once more than this many exist.
const MAX_RUNS: usize = 6;
/// Compact a partition once this fraction of its rows is dead...
const GC_DEAD_RATIO: f64 = 0.25;
/// ...or once it has accumulated this many chunks (block padding and scan
/// setup overhead grow with chunk count).
const GC_MAX_CHUNKS: usize = 16;
/// Run the re-clustering acceptance check after external churn (adds +
/// removes — maintenance's own moves measure repair, not drift) reaching
/// this fraction of the corpus since the last check.
const REBOOTSTRAP_CHURN_FRACTION: f64 = 0.25;
/// WAL record kinds.
const WAL_ADD: u8 = 1;
const WAL_REMOVE: u8 = 2;

const RUN_HEADER_BYTES: usize = 16;
const RUN_ENTRY_BYTES: usize = 24;
const ENTRY_FLAG_REPLICA: u32 = 1;

fn invalid_data(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn crc32(bytes: &[u8]) -> u32 {
    static TABLE: OnceLock<[u32; 256]> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut table = [0u32; 256];
        for (i, slot) in table.iter_mut().enumerate() {
            let mut value = i as u32;
            for _ in 0..8 {
                value = if value & 1 != 0 {
                    0xEDB8_8320 ^ (value >> 1)
                } else {
                    value >> 1
                };
            }
            *slot = value;
        }
        table
    });
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        crc = table[((crc ^ byte as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

/// One chunk's byte layout within a segment file, relative to chunk start.
struct ChunkLayout {
    codes: (usize, usize),
    scales: (usize, usize),
    ids: (usize, usize),
    replica_bits: (usize, usize),
    vectors: (usize, usize),
    total_len: usize,
}

fn chunk_layout(bit_width: usize, dim: usize, n_rows: usize, has_vectors: bool) -> ChunkLayout {
    let n_byte_groups = pack::n_byte_groups(bit_width, dim);
    let n_blocks = (n_rows + BLOCK - 1) / BLOCK;
    let mut offset = CHUNK_ALIGN; // header
    let mut section = |len: usize| {
        let start = offset;
        offset = disk::align_up(start + len, SECTION_ALIGN);
        (start, len)
    };
    let layout = ChunkLayout {
        codes: section(n_blocks * n_byte_groups * BLOCK),
        scales: section(4 * n_rows),
        ids: section(8 * n_rows),
        replica_bits: section((n_rows + 7) / 8),
        vectors: section(if has_vectors { 4 * n_rows * dim } else { 0 }),
        total_len: 0,
    };
    ChunkLayout {
        total_len: disk::align_up(offset, CHUNK_ALIGN),
        ..layout
    }
}

#[derive(Clone, Copy)]
struct ChunkMeta {
    offset: u64,
    n_rows: u32,
}

struct PartitionState {
    partition_id: u32,
    generation: u32,
    n_rows: u64,
    live_rows: u64,
    live_primary: u64,
    chunks: Vec<ChunkMeta>,
    /// Dead bitmap over rows (1 = dead). Mutated in RAM, persisted with the
    /// manifest.
    dead: Vec<u8>,
}

impl PartitionState {
    fn is_dead(&self, row: u64) -> bool {
        self.dead[(row / 8) as usize] & (1 << (row % 8)) != 0
    }

    fn file_len(&self, bit_width: usize, dim: usize, has_vectors: bool) -> u64 {
        match self.chunks.last() {
            None => 0,
            Some(chunk) => {
                chunk.offset
                    + chunk_layout(bit_width, dim, chunk.n_rows as usize, has_vectors)
                        .total_len as u64
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct CopyLocation {
    partition_id: u32,
    row: u32,
    is_replica: bool,
}

#[derive(Clone, Copy)]
struct RunEntry {
    id: u64,
    partition_id: u32,
    generation: u32,
    row: u32,
    flags: u32,
}

/// A batch of rows in transit (flush, move, split, compaction): group-byte
/// codes plus per-row metadata, vectors included when the index stores them.
#[derive(Default)]
struct RowBatch {
    n: usize,
    group_rows: Vec<u8>,
    scales: Vec<f32>,
    ids: Vec<u64>,
    replica: Vec<bool>,
    vectors: Vec<f32>,
}

impl RowBatch {
    fn push_row(
        &mut self,
        group_row: &[u8],
        scale: f32,
        id: u64,
        is_replica: bool,
        vector: Option<&[f32]>,
    ) {
        self.group_rows.extend_from_slice(group_row);
        self.scales.push(scale);
        self.ids.push(id);
        self.replica.push(is_replica);
        if let Some(vector) = vector {
            self.vectors.extend_from_slice(vector);
        }
        self.n += 1;
    }

    fn group_row(&self, i: usize, n_byte_groups: usize) -> &[u8] {
        &self.group_rows[i * n_byte_groups..(i + 1) * n_byte_groups]
    }

    fn vector(&self, i: usize, dim: usize) -> Option<&[f32]> {
        if self.vectors.is_empty() {
            None
        } else {
            Some(&self.vectors[i * dim..(i + 1) * dim])
        }
    }
}

/// Incrementally-updatable, directory-backed TurboQuant index. See the
/// module docs for the storage model.
pub struct FreshIndex {
    directory: Option<PathBuf>,
    bit_width: usize,
    store_vectors: bool,
    replica_epsilon: Option<f32>,
    partition_target: Option<usize>,
    clustered: bool,
    epoch: u64,
    next_partition_id: u32,
    next_run_generation: u64,
    churn_since_check: u64,
    tqplus_shift: Vec<f32>,
    tqplus_scale: Vec<f32>,
    centroids: Vec<f32>,
    partitions: Vec<PartitionState>,
    runs: Vec<u64>,
    memtable: IdMapIndex,
    memtable_originals: HashMap<u64, Box<[f32]>>,
    wal: Option<File>,
    /// dim recorded in the live WAL's header. A WAL created before the
    /// index committed a dim says 0 and must be re-headered before the
    /// first add record (it is empty at that point by construction —
    /// add records imply a committed dim, and no-op removes are not
    /// logged).
    wal_dim: usize,
    segment_maps: Mutex<HashMap<u32, Arc<Mmap>>>,
    run_maps: Mutex<HashMap<u64, Arc<Mmap>>>,
    rotation_cache: OnceLock<Vec<f32>>,
    codebook_cache: OnceLock<Vec<f32>>,
}

impl FreshIndex {
    /// Construct an empty index with no backing directory yet. Vectors live
    /// in RAM (memtable) until the first [`Self::save`], which binds the
    /// directory, makes the index durable, and starts the write-ahead log.
    pub fn new(dim: Option<usize>, bit_width: usize) -> Result<Self, ConstructError> {
        let memtable = match dim {
            Some(d) => IdMapIndex::new(d, bit_width)?,
            None => IdMapIndex::new_lazy(bit_width)?,
        };
        Ok(Self {
            directory: None,
            bit_width,
            store_vectors: false,
            replica_epsilon: None,
            partition_target: None,
            clustered: false,
            epoch: 0,
            next_partition_id: 0,
            next_run_generation: 0,
            churn_since_check: 0,
            tqplus_shift: Vec::new(),
            tqplus_scale: Vec::new(),
            centroids: Vec::new(),
            partitions: Vec::new(),
            runs: Vec::new(),
            memtable,
            memtable_originals: HashMap::new(),
            wal: None,
            wal_dim: 0,
            segment_maps: Mutex::new(HashMap::new()),
            run_maps: Mutex::new(HashMap::new()),
            rotation_cache: OnceLock::new(),
            codebook_cache: OnceLock::new(),
        })
    }

    /// Open an index directory previously produced by [`Self::save`].
    /// Replays the write-ahead log into the memtable, truncates any
    /// orphaned bytes from a crashed flush, and removes unreferenced files.
    pub fn open(directory: impl AsRef<Path>) -> io::Result<Self> {
        let directory = directory.as_ref().to_path_buf();
        let mut index = Self::read_manifest(&directory)?;
        index.cleanup_directory()?;
        index.replay_or_reset_wal()?;
        Ok(index)
    }

    // ------------------------------------------------------------------
    // Configuration (same contracts as DiskIndex)
    // ------------------------------------------------------------------

    /// Enable (`Some(target)`) or disable (`None`) partitioning; takes
    /// effect as the corpus crosses `2 * target` at a flush. Disabling
    /// after clustering is not supported (partitions already exist).
    pub fn set_partitioning(&mut self, target_partition_size: Option<usize>) {
        assert!(
            target_partition_size != Some(0),
            "target_partition_size must be positive",
        );
        if self.clustered && target_partition_size.is_none() {
            panic!("cannot disable partitioning on an already-clustered FreshIndex");
        }
        self.partition_target = target_partition_size;
    }

    pub fn partition_target(&self) -> Option<usize> {
        self.partition_target
    }

    /// See [`crate::DiskIndex::set_replication`].
    pub fn set_replication(&mut self, epsilon: Option<f32>) {
        if let Some(e) = epsilon {
            assert!(
                e.is_finite() && e > 0.0,
                "replica epsilon must be finite and positive, got {e}",
            );
        }
        self.replica_epsilon = epsilon;
    }

    pub fn replica_epsilon(&self) -> Option<f32> {
        self.replica_epsilon
    }

    /// See [`crate::DiskIndex::set_store_vectors`]. Must be set while the
    /// index is empty.
    pub fn set_store_vectors(&mut self, store_vectors: bool) {
        if store_vectors == self.store_vectors {
            return;
        }
        assert!(
            self.partitions.is_empty() && self.memtable.is_empty(),
            "store_vectors must be set while the index is empty",
        );
        self.store_vectors = store_vectors;
    }

    pub fn stores_vectors(&self) -> bool {
        self.store_vectors
    }

    // ------------------------------------------------------------------
    // Introspection
    // ------------------------------------------------------------------

    pub fn dim_opt(&self) -> Option<usize> {
        self.memtable.dim_opt()
    }

    pub fn dim(&self) -> usize {
        self.dim_opt().unwrap_or(0)
    }

    pub fn bit_width(&self) -> usize {
        self.bit_width
    }

    /// Live vectors (distinct ids): base primaries plus the memtable.
    pub fn len(&self) -> usize {
        self.partitions
            .iter()
            .map(|p| p.live_primary as usize)
            .sum::<usize>()
            + self.memtable.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of partitions (0 before any flush, 1 while unclustered).
    pub fn nlist(&self) -> usize {
        self.partitions.len()
    }

    /// Physical rows on disk, including dead rows and replicas. Diagnostic.
    pub fn base_len(&self) -> usize {
        self.partitions.iter().map(|p| p.n_rows as usize).sum()
    }

    /// Rows buffered in the in-RAM memtable. Diagnostic.
    pub fn memtable_len(&self) -> usize {
        self.memtable.len()
    }

    /// Dead (removed or moved-away) rows awaiting compaction. Diagnostic.
    pub fn dead_count(&self) -> usize {
        self.partitions
            .iter()
            .map(|p| (p.n_rows - p.live_rows) as usize)
            .sum()
    }

    /// Live closure-assignment replica rows. Diagnostic.
    pub fn replica_count(&self) -> usize {
        self.partitions
            .iter()
            .map(|p| (p.live_rows - p.live_primary) as usize)
            .sum()
    }

    /// Number of id-run tables. Diagnostic.
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    /// Total chunks across all segments. Diagnostic.
    pub fn chunk_count(&self) -> usize {
        self.partitions.iter().map(|p| p.chunks.len()).sum()
    }

    /// Backing directory, or `None` before the first save.
    pub fn path(&self) -> Option<&Path> {
        self.directory.as_deref()
    }

    /// Warm the query-side caches. Cheap.
    pub fn prepare(&self) {
        let Some(dim) = self.dim_opt() else { return };
        self.rotation_for(dim);
        self.codebook_for(dim);
        self.memtable.prepare();
    }

    // ------------------------------------------------------------------
    // Mutation
    // ------------------------------------------------------------------

    /// Add `vectors` with the given external ids. Same contract as
    /// [`crate::DiskIndex::add_with_ids_2d`]; rows land in the memtable
    /// (and the write-ahead log when a directory is bound).
    pub fn add_with_ids_2d(
        &mut self,
        vectors: &[f32],
        dim: usize,
        ids: &[u64],
    ) -> Result<(), AddError> {
        for &id in ids {
            if !self.memtable.contains(id) && !self.live_copies(id).is_empty() {
                return Err(AddError::IdAlreadyPresent(id));
            }
        }
        self.memtable.add_with_ids_2d(vectors, dim, ids)?;
        if self.wal.is_some() && self.wal_dim != dim {
            self.reset_wal().expect("write-ahead log reset failed");
        }
        for (i, &id) in ids.iter().enumerate() {
            let vector = &vectors[i * dim..(i + 1) * dim];
            if self.store_vectors {
                self.memtable_originals.insert(id, vector.into());
            }
            self.wal_append(WAL_ADD, id, vector)
                .expect("write-ahead log append failed");
        }
        self.wal_sync();
        Ok(())
    }

    /// Add with the already-committed dim.
    pub fn add_with_ids(&mut self, vectors: &[f32], ids: &[u64]) -> Result<(), AddError> {
        let dim = self.dim_opt().expect(
            "FreshIndex dim is not set; use add_with_ids_2d(vectors, dim, ids) on the \
             first add or construct with FreshIndex::new(Some(dim), bit_width)",
        );
        self.add_with_ids_2d(vectors, dim, ids)
    }

    /// Remove the vector with the given external id. Base copies are
    /// dead-marked immediately (all of them, replicas included) and
    /// physically dropped when their partition is next compacted.
    pub fn remove(&mut self, id: u64) -> bool {
        let removed = self.remove_internal(id);
        if removed {
            self.wal_append(WAL_REMOVE, id, &[])
                .expect("write-ahead log append failed");
            self.wal_sync();
        }
        removed
    }

    fn remove_internal(&mut self, id: u64) -> bool {
        if self.memtable.remove(id) {
            self.memtable_originals.remove(&id);
            return true;
        }
        let copies = self.live_copies(id);
        if copies.is_empty() {
            return false;
        }
        for copy in copies {
            self.dead_mark(copy);
        }
        self.churn_since_check += 1;
        true
    }

    /// True if a vector with this external id is live.
    pub fn contains(&self, id: u64) -> bool {
        self.memtable.contains(id) || !self.live_copies(id).is_empty()
    }

    /// The stored full-precision vector of a live id. Panics unless the
    /// index stores vectors.
    pub fn get_vector(&self, id: u64) -> Option<Vec<f32>> {
        assert!(
            self.store_vectors,
            "get_vector requires an index built with store_vectors",
        );
        if let Some(vector) = self.memtable_originals.get(&id) {
            return Some(vector.to_vec());
        }
        let copies = self.live_copies(id);
        let copy = copies.first()?;
        Some(self.copy_vector(*copy))
    }

    // ------------------------------------------------------------------
    // Search
    // ------------------------------------------------------------------

    /// Search with default options. See
    /// [`crate::DiskIndex::search_with_options`] — semantics are identical.
    pub fn search(&self, queries: &[f32], k: usize) -> (Vec<f32>, Vec<u64>) {
        self.search_with_options(queries, k, SearchOptions::default())
    }

    /// Search with explicit options (same contract as
    /// [`crate::DiskIndex::search_with_options`]).
    pub fn search_with_options(
        &self,
        queries: &[f32],
        k: usize,
        options: SearchOptions,
    ) -> (Vec<f32>, Vec<u64>) {
        let Some(dim) = self.dim_opt() else {
            return (Vec::new(), Vec::new());
        };
        let nq = queries.len() / dim;
        assert_eq!(
            queries.len(),
            nq * dim,
            "queries length must be a multiple of dim"
        );
        if let Some((vi, ci, v)) = first_invalid_coord(queries, dim) {
            panic!(
                "invalid query value at query {vi}, coord {ci}: {v} \
                 (must be finite and |value| < 1e16 to avoid f32 overflow)",
            );
        }
        if let Some(epsilon) = options.probe_epsilon {
            assert!(
                epsilon.is_finite() && epsilon >= 0.0,
                "probe_epsilon must be finite and non-negative, got {epsilon}",
            );
        }

        let k_eff = k.min(self.len());
        if nq == 0 || k_eff == 0 {
            return (Vec::new(), Vec::new());
        }

        let rescore_k = match options.rescore_k {
            Some(0) => None,
            Some(depth) => {
                assert!(
                    self.store_vectors,
                    "rescore_k requires an index built with store_vectors",
                );
                Some(depth.max(k_eff))
            }
            None if self.store_vectors => {
                Some((DEFAULT_RESCORE_MULTIPLIER * k_eff).max(k_eff))
            }
            None => None,
        };
        let fetch_k = rescore_k.unwrap_or(k_eff).min(self.len());

        let base_live: u64 = self.partitions.iter().map(|p| p.live_rows).sum();
        let base_candidates: Vec<Vec<(f32, u64)>> = if base_live > 0 {
            let prepared = crate::search::prepare(
                queries,
                nq,
                self.rotation_for(dim),
                self.codebook_for(dim),
                &self.tqplus_shift,
                &self.tqplus_scale,
                self.bit_width,
                dim,
            );
            let nlist = self.partitions.len();
            let routes: Vec<Vec<u32>> = if self.clustered && nlist > 1 {
                let nprobe_cap = options
                    .nprobe
                    .unwrap_or_else(|| {
                        if options.probe_epsilon.is_some() {
                            (nlist / DEFAULT_EPSILON_NPROBE_DIVISOR).max(DEFAULT_NPROBE_MIN)
                        } else {
                            (nlist / DEFAULT_NPROBE_DIVISOR).max(DEFAULT_NPROBE_MIN)
                        }
                    })
                    .clamp(1, nlist);
                route_queries(
                    queries,
                    nq,
                    dim,
                    &self.centroids,
                    nlist,
                    nprobe_cap,
                    options.probe_epsilon,
                )
            } else {
                vec![(0..nlist as u32).collect(); nq]
            };
            (0..nq)
                .map(|qi| {
                    let single = prepared.single(qi);
                    let mut candidates = Vec::new();
                    for &position in &routes[qi] {
                        self.scan_partition(
                            position as usize,
                            &single,
                            fetch_k,
                            &mut candidates,
                        );
                    }
                    candidates
                })
                .collect()
        } else {
            vec![Vec::new(); nq]
        };

        let (memtable_scores, memtable_ids) = if self.memtable.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            self.memtable.search(queries, fetch_k)
        };
        let k_memtable = memtable_scores.len() / nq.max(1);

        let mut out_scores = Vec::with_capacity(nq * k_eff);
        let mut out_ids = Vec::with_capacity(nq * k_eff);
        let mut candidates: Vec<(f32, u64)> = Vec::new();
        let mut seen: HashSet<u64> = HashSet::new();
        for (qi, base_list) in base_candidates.into_iter().enumerate() {
            candidates.clear();
            candidates.extend(base_list);
            for j in 0..k_memtable {
                candidates.push((
                    memtable_scores[qi * k_memtable + j],
                    memtable_ids[qi * k_memtable + j],
                ));
            }
            candidates.sort_unstable_by(|a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            seen.clear();
            candidates.retain(|&(_, id)| seen.insert(id));
            candidates.truncate(fetch_k);

            if rescore_k.is_some() {
                let query = &queries[qi * dim..(qi + 1) * dim];
                for candidate in candidates.iter_mut() {
                    candidate.0 = self.exact_score(candidate.1, query);
                }
                candidates.sort_unstable_by(|a, b| {
                    b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            candidates.truncate(k_eff);

            for &(score, id) in &candidates {
                out_scores.push(score);
                out_ids.push(id);
            }
            for _ in candidates.len()..k_eff {
                out_scores.push(f32::NEG_INFINITY);
                out_ids.push(0);
            }
        }
        (out_scores, out_ids)
    }

    fn exact_score(&self, id: u64, query: &[f32]) -> f32 {
        if let Some(vector) = self.memtable_originals.get(&id) {
            return query.iter().zip(vector.iter()).map(|(&q, &v)| q * v).sum();
        }
        let copies = self.live_copies(id);
        let copy = copies
            .first()
            .expect("rescore candidates are live ids surfaced by the scan");
        let vector = self.copy_vector(*copy);
        query.iter().zip(&vector).map(|(&q, &v)| q * v).sum()
    }

    /// Scan one partition with a single prepared query, pushing
    /// (score, id) candidates for live rows.
    fn scan_partition(
        &self,
        position: usize,
        prepared: &crate::search::PreparedQueries,
        fetch_k: usize,
        out: &mut Vec<(f32, u64)>,
    ) {
        let partition = &self.partitions[position];
        if partition.live_rows == 0 {
            return;
        }
        let dim = self.dim();
        let map = match self.segment_map(partition.partition_id, partition.generation) {
            Ok(map) => map,
            Err(_) => return,
        };
        let dead_total = (partition.n_rows - partition.live_rows) as usize;
        let mut base_row = 0u64;
        for chunk in &partition.chunks {
            let n_rows = chunk.n_rows as usize;
            let layout = chunk_layout(self.bit_width, dim, n_rows, self.store_vectors);
            let start = chunk.offset as usize;
            let codes = &map[start + layout.codes.0..start + layout.codes.0 + layout.codes.1];
            let scales = f32_slice(&map[start + layout.scales.0
                ..start + layout.scales.0 + layout.scales.1]);
            let ids = u64_slice(
                &map[start + layout.ids.0..start + layout.ids.0 + layout.ids.1],
            );
            let k_chunk = (fetch_k + dead_total).min(n_rows);
            let (scores, slots) = crate::search::scan(
                prepared,
                codes,
                scales,
                self.bit_width,
                dim,
                n_rows,
                (n_rows + BLOCK - 1) / BLOCK,
                k_chunk,
                None,
            );
            for j in 0..scores.len() {
                let slot = slots[j] as usize;
                let row = base_row + slot as u64;
                if partition.is_dead(row) {
                    continue;
                }
                out.push((scores[j], ids[slot]));
            }
            base_row += chunk.n_rows as u64;
        }
    }

    // ------------------------------------------------------------------
    // Persistence: save / flush
    // ------------------------------------------------------------------

    /// Bind to `directory` (first call) and flush: append the memtable to
    /// its partitions, run local maintenance, and atomically publish a new
    /// manifest. Untouched partitions' files are not rewritten — their
    /// page-cache contents stay valid across the save.
    pub fn save(&mut self, directory: impl AsRef<Path>) -> io::Result<()> {
        let directory = directory.as_ref();
        match &self.directory {
            None => {
                fs::create_dir_all(directory)?;
                if directory.join(MANIFEST_FILE).exists() {
                    return Err(invalid_data(format!(
                        "directory {} already contains a FreshIndex; open it instead",
                        directory.display(),
                    )));
                }
                self.directory = Some(directory.to_path_buf());
                self.reset_wal()?;
            }
            Some(bound) => {
                if bound != directory {
                    return Err(invalid_data(format!(
                        "FreshIndex is bound to {}, cannot save to {}",
                        bound.display(),
                        directory.display(),
                    )));
                }
            }
        }
        self.flush()
    }

    fn flush(&mut self) -> io::Result<()> {
        let dim = match self.dim_opt() {
            Some(dim) => dim,
            None => {
                // Nothing ever added: publish an empty manifest so open works.
                self.epoch += 1;
                self.write_manifest()?;
                self.reset_wal()?;
                return Ok(());
            }
        };
        self.commit_calibration(dim);

        let mut entries: Vec<RunEntry> = Vec::new();
        let mut dropped_files: Vec<PathBuf> = Vec::new();

        // Drain the memtable into per-partition appends.
        let batch = self.drain_memtable(dim);
        if batch.n > 0 {
            if self.partitions.is_empty() {
                self.create_partition(&vec![0.0; dim], &mut entries)?;
            }
            let destinations = self.assign_batch(&batch, dim);
            self.append_batch(&batch, &destinations, dim, &mut entries)?;
            self.churn_since_check += batch.n as u64;
        }

        self.maintenance(dim, &mut entries, &mut dropped_files)?;

        // Publish: run file, manifest, WAL reset, then cleanup.
        if !entries.is_empty() {
            self.write_run(&entries)?;
        }
        if self.runs.len() > MAX_RUNS {
            self.merge_runs(&mut dropped_files)?;
        }
        self.epoch += 1;
        self.write_manifest()?;
        self.reset_wal()?;
        self.memtable = self.fresh_memtable(dim)?;
        self.memtable_originals.clear();
        for path in dropped_files {
            fs::remove_file(path).ok();
        }
        Ok(())
    }

    fn commit_calibration(&mut self, dim: usize) {
        if !self.tqplus_shift.is_empty() {
            return;
        }
        let shift = self.memtable.inner().tqplus_shift();
        if shift.is_empty() {
            self.tqplus_shift = vec![0.0; dim];
            self.tqplus_scale = vec![1.0; dim];
        } else {
            self.tqplus_shift = shift.to_vec();
            self.tqplus_scale = self.memtable.inner().tqplus_scale().to_vec();
        }
    }

    fn fresh_memtable(&self, dim: usize) -> io::Result<IdMapIndex> {
        let inner = TurboQuantIndex::from_parts(
            Some(dim),
            self.bit_width,
            0,
            Vec::new(),
            Vec::new(),
            self.tqplus_shift.clone(),
            self.tqplus_scale.clone(),
        );
        Ok(IdMapIndex::from_inner(inner))
    }

    fn drain_memtable(&self, dim: usize) -> RowBatch {
        let n = self.memtable.len();
        let mut batch = RowBatch::default();
        if n == 0 {
            return batch;
        }
        let n_byte_groups = pack::n_byte_groups(self.bit_width, dim);
        let group_rows = pack::group_bytes(
            self.memtable.inner().packed_codes(),
            n,
            self.bit_width,
            dim,
        );
        let scales = self.memtable.inner().scales();
        let ids = self.memtable.slot_to_id_slice();
        for i in 0..n {
            let vector = if self.store_vectors {
                Some(
                    self.memtable_originals
                        .get(&ids[i])
                        .expect("store_vectors invariant: every memtable row has an original")
                        .as_ref(),
                )
            } else {
                None
            };
            batch.push_row(
                &group_rows[i * n_byte_groups..(i + 1) * n_byte_groups],
                scales[i],
                ids[i],
                false,
                vector,
            );
        }
        batch
    }

    /// Primary partition position for each batch row (plus replica
    /// positions when closure assignment is on).
    fn assign_batch(&self, batch: &RowBatch, dim: usize) -> Vec<Vec<u32>> {
        let nlist = self.partitions.len();
        if !self.clustered || nlist <= 1 {
            return vec![vec![0]; batch.n];
        }
        let vectors = self.batch_vectors(batch, dim);
        let (assignments, _) =
            kmeans::assign(&vectors, batch.n, dim, &self.centroids, nlist);
        let replica_lists = match self.replica_epsilon {
            Some(epsilon) => closure_assignments_for_vectors(
                &vectors,
                batch.n,
                dim,
                &self.centroids,
                &assignments,
                epsilon,
            ),
            None => vec![Vec::new(); batch.n],
        };
        assignments
            .into_iter()
            .zip(replica_lists)
            .map(|(primary, replicas)| {
                let mut destinations = Vec::with_capacity(1 + replicas.len());
                destinations.push(primary);
                destinations.extend(replicas);
                destinations
            })
            .collect()
    }

    /// Append batch rows to their destination partitions (first destination
    /// = primary, rest = replicas), one chunk per touched partition.
    fn append_batch(
        &mut self,
        batch: &RowBatch,
        destinations: &[Vec<u32>],
        dim: usize,
        entries: &mut Vec<RunEntry>,
    ) -> io::Result<()> {
        let n_byte_groups = pack::n_byte_groups(self.bit_width, dim);
        let mut per_partition: HashMap<u32, RowBatch> = HashMap::new();
        for (i, destination_list) in destinations.iter().enumerate().take(batch.n) {
            for (d, &position) in destination_list.iter().enumerate() {
                per_partition.entry(position).or_default().push_row(
                    batch.group_row(i, n_byte_groups),
                    batch.scales[i],
                    batch.ids[i],
                    d > 0,
                    batch.vector(i, dim),
                );
            }
        }
        let mut positions: Vec<u32> = per_partition.keys().copied().collect();
        positions.sort_unstable();
        for position in positions {
            let rows = &per_partition[&position];
            self.append_chunk(position as usize, rows, dim, entries)?;
        }
        Ok(())
    }

    /// The batch's vectors: exact originals when stored, decoded
    /// approximations otherwise.
    fn batch_vectors(&self, batch: &RowBatch, dim: usize) -> Vec<f32> {
        if self.store_vectors && !batch.vectors.is_empty() {
            return batch.vectors.clone();
        }
        self.decode_group_rows(&batch.group_rows, &batch.scales, batch.n, dim)
    }

    fn decode_group_rows(
        &self,
        group_rows: &[u8],
        scales: &[f32],
        n: usize,
        dim: usize,
    ) -> Vec<f32> {
        if n == 0 {
            return Vec::new();
        }
        let packed = pack::packed_from_group_bytes(group_rows, n, self.bit_width, dim);
        decode::decode(
            &packed,
            scales,
            n,
            dim,
            self.bit_width,
            self.rotation_for(dim),
            self.codebook_for(dim),
            &self.tqplus_shift,
            &self.tqplus_scale,
        )
    }

    // ------------------------------------------------------------------
    // Maintenance (LIRE on the incremental substrate)
    // ------------------------------------------------------------------

    fn maintenance(
        &mut self,
        dim: usize,
        entries: &mut Vec<RunEntry>,
        dropped_files: &mut Vec<PathBuf>,
    ) -> io::Result<()> {
        let Some(target) = self.partition_target else {
            self.collect_garbage(dim, entries, dropped_files)?;
            return Ok(());
        };

        // Bootstrap: k-way split of the catch-all partition once big
        // enough, then FALL THROUGH — the split/merge/reassign passes below
        // balance the raw k-means result within the same flush, so the
        // next flush starts from a healthy structure.
        if !self.clustered {
            let total: u64 = self.partitions.iter().map(|p| p.live_primary).sum();
            if (total as usize) < MIN_PARTITIONS * target {
                self.collect_garbage(dim, entries, dropped_files)?;
                return Ok(());
            }
            self.rebuild_all(dim, target, None, entries, dropped_files)?;
            self.clustered = true;
        }

        // `changed` holds PARTITION IDS (stable across the structural
        // mutations below), never positions — positions shift whenever a
        // partition is dropped.
        let mut changed: HashSet<u32> = HashSet::new();
        let mut replica_refresh: HashSet<u64> = HashSet::new();

        // Split oversized partitions.
        while let Some(position) = self
            .partitions
            .iter()
            .position(|p| p.live_primary as usize > SPLIT_FACTOR * target)
        {
            if !self.split_partition(
                position,
                dim,
                &mut changed,
                &mut replica_refresh,
                entries,
                dropped_files,
            )? {
                break; // degenerate split; do not loop forever
            }
        }

        // Dissolve undersized partitions.
        while self.partitions.len() > MIN_PARTITIONS {
            let Some(position) = self
                .partitions
                .iter()
                .position(|p| (p.live_primary as usize) < target / MERGE_DIVISOR)
            else {
                break;
            };
            self.dissolve_partition(
                position,
                dim,
                &mut changed,
                &mut replica_refresh,
                entries,
                dropped_files,
            )?;
        }

        // LIRE reassignment around changed partitions, then centroid
        // refresh for everything whose membership moved.
        if !changed.is_empty() {
            self.reassign_pass(dim, &changed, &mut replica_refresh, entries)?;
        }

        // Re-cover primaries whose replica copies were dropped or whose
        // primary moved (closure sets changed with the centroids).
        if self.replica_epsilon.is_some() && !replica_refresh.is_empty() {
            let ids: Vec<u64> = replica_refresh.iter().copied().collect();
            self.refresh_replicas(&ids, dim, entries)?;
        }

        self.collect_garbage(dim, entries, dropped_files)?;
        self.rebootstrap_if_due(dim, target, entries, dropped_files)?;
        Ok(())
    }

    /// All live rows of a partition (optionally primaries only), with
    /// their source row indices.
    fn read_live(
        &self,
        position: usize,
        primaries_only: bool,
        dim: usize,
    ) -> io::Result<(RowBatch, Vec<u64>)> {
        let partition = &self.partitions[position];
        let n_byte_groups = pack::n_byte_groups(self.bit_width, dim);
        let mut batch = RowBatch::default();
        let mut source_rows = Vec::new();
        if partition.n_rows == 0 {
            return Ok((batch, source_rows));
        }
        let map = self.segment_map(partition.partition_id, partition.generation)?;
        let mut block_rows = vec![0u8; BLOCK * n_byte_groups];
        let mut base_row = 0u64;
        for chunk in &partition.chunks {
            let n_rows = chunk.n_rows as usize;
            let layout = chunk_layout(self.bit_width, dim, n_rows, self.store_vectors);
            let start = chunk.offset as usize;
            let codes = &map[start + layout.codes.0..start + layout.codes.0 + layout.codes.1];
            let scales = f32_slice(
                &map[start + layout.scales.0..start + layout.scales.0 + layout.scales.1],
            );
            let ids =
                u64_slice(&map[start + layout.ids.0..start + layout.ids.0 + layout.ids.1]);
            let replica_bits = &map[start + layout.replica_bits.0
                ..start + layout.replica_bits.0 + layout.replica_bits.1];
            let vectors = if self.store_vectors {
                Some(f32_slice(
                    &map[start + layout.vectors.0
                        ..start + layout.vectors.0 + layout.vectors.1],
                ))
            } else {
                None
            };
            let block_bytes = n_byte_groups * BLOCK;
            for block_idx in 0..(n_rows + BLOCK - 1) / BLOCK {
                pack::unpack_block_rows(
                    &codes[block_idx * block_bytes..(block_idx + 1) * block_bytes],
                    n_byte_groups,
                    &mut block_rows,
                );
                let lanes = (n_rows - block_idx * BLOCK).min(BLOCK);
                for lane in 0..lanes {
                    let local = block_idx * BLOCK + lane;
                    let row = base_row + local as u64;
                    if partition.is_dead(row) {
                        continue;
                    }
                    let is_replica = replica_bits[local / 8] & (1 << (local % 8)) != 0;
                    if primaries_only && is_replica {
                        continue;
                    }
                    batch.push_row(
                        &block_rows[lane * n_byte_groups..(lane + 1) * n_byte_groups],
                        scales[local],
                        ids[local],
                        is_replica,
                        vectors.map(|v| &v[local * dim..(local + 1) * dim]),
                    );
                    source_rows.push(row);
                }
            }
            base_row += chunk.n_rows as u64;
        }
        Ok((batch, source_rows))
    }

    /// Split `position` into two by 2-means over its live primaries.
    /// Replica rows hosted by the partition are dropped (their primaries
    /// are queued for re-replication). Returns false on a degenerate split.
    fn split_partition(
        &mut self,
        position: usize,
        dim: usize,
        changed: &mut HashSet<u32>,
        replica_refresh: &mut HashSet<u64>,
        entries: &mut Vec<RunEntry>,
        dropped_files: &mut Vec<PathBuf>,
    ) -> io::Result<bool> {
        let (live, _) = self.read_live(position, false, dim)?;
        let mut primaries = RowBatch::default();
        let n_byte_groups = pack::n_byte_groups(self.bit_width, dim);
        for i in 0..live.n {
            if live.replica[i] {
                replica_refresh.insert(live.ids[i]);
            } else {
                primaries.push_row(
                    live.group_row(i, n_byte_groups),
                    live.scales[i],
                    live.ids[i],
                    false,
                    live.vector(i, dim),
                );
            }
        }
        let vectors = self.batch_vectors(&primaries, dim);
        let (child_centroids, child_assignments) = kmeans::kmeans(
            &vectors,
            primaries.n,
            dim,
            2,
            KMEANS_ITERATIONS,
            KMEANS_SEED ^ self.partitions[position].partition_id as u64,
        );
        let child_one = child_assignments.iter().filter(|&&c| c == 1).count();
        if child_centroids.len() / dim < 2 || child_one == 0 || child_one == primaries.n {
            return Ok(false);
        }
        let mut children = [RowBatch::default(), RowBatch::default()];
        for i in 0..primaries.n {
            children[child_assignments[i] as usize].push_row(
                primaries.group_row(i, n_byte_groups),
                primaries.scales[i],
                primaries.ids[i],
                false,
                primaries.vector(i, dim),
            );
        }
        self.drop_partition(position, dim, dropped_files);
        for (child, rows) in children.iter().enumerate() {
            let centroid = &child_centroids[child * dim..(child + 1) * dim];
            let new_position = self.create_partition(centroid, entries)?;
            self.append_chunk(new_position, rows, dim, entries)?;
            changed.insert(self.partitions[new_position].partition_id);
        }
        Ok(true)
    }

    /// Dissolve `position`: live primaries move to their nearest other
    /// partition; replica rows are dropped (primaries queued for refresh).
    fn dissolve_partition(
        &mut self,
        position: usize,
        dim: usize,
        changed: &mut HashSet<u32>,
        replica_refresh: &mut HashSet<u64>,
        entries: &mut Vec<RunEntry>,
        dropped_files: &mut Vec<PathBuf>,
    ) -> io::Result<()> {
        let (live, _) = self.read_live(position, false, dim)?;
        let n_byte_groups = pack::n_byte_groups(self.bit_width, dim);
        let mut primaries = RowBatch::default();
        for i in 0..live.n {
            if live.replica[i] {
                replica_refresh.insert(live.ids[i]);
            } else {
                primaries.push_row(
                    live.group_row(i, n_byte_groups),
                    live.scales[i],
                    live.ids[i],
                    false,
                    live.vector(i, dim),
                );
            }
        }
        self.drop_partition(position, dim, dropped_files);
        if primaries.n == 0 {
            return Ok(());
        }
        let vectors = self.batch_vectors(&primaries, dim);
        let nlist = self.partitions.len();
        let (assignments, _) =
            kmeans::assign(&vectors, primaries.n, dim, &self.centroids, nlist);
        let destinations: Vec<Vec<u32>> =
            assignments.iter().map(|&a| vec![a]).collect();
        self.append_batch(&primaries, &destinations, dim, entries)?;
        for &a in &assignments {
            changed.insert(self.partitions[a as usize].partition_id);
        }
        Ok(())
    }

    /// LIRE reassignment: members of changed partitions (given by stable
    /// partition id) and their nearest neighbors move when a different
    /// centroid is strictly closer, then every partition whose membership
    /// changed gets its centroid refreshed to the mean of its live
    /// primaries.
    fn reassign_pass(
        &mut self,
        dim: usize,
        changed: &HashSet<u32>,
        replica_refresh: &mut HashSet<u64>,
        entries: &mut Vec<RunEntry>,
    ) -> io::Result<()> {
        let nlist = self.partitions.len();
        if nlist <= 1 {
            return Ok(());
        }
        // Translate ids to current positions (dropped partitions vanish),
        // then add each one's nearest neighbor partitions. No structural
        // mutation happens inside this pass, so positions stay valid.
        let changed_positions: HashSet<usize> = changed
            .iter()
            .filter_map(|&id| self.partition_position(id))
            .collect();
        let mut affected: HashSet<usize> = changed_positions.clone();
        for &p in &changed_positions {
            let center = &self.centroids[p * dim..(p + 1) * dim];
            let mut ranked: Vec<(f32, usize)> = (0..nlist)
                .filter(|&c| c != p)
                .map(|c| {
                    let other = &self.centroids[c * dim..(c + 1) * dim];
                    let dist: f32 = center
                        .iter()
                        .zip(other)
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum();
                    (dist, c)
                })
                .collect();
            ranked.sort_unstable_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            affected.extend(ranked.iter().take(NEIGHBOR_PARTITIONS).map(|&(_, c)| c));
        }

        let n_byte_groups = pack::n_byte_groups(self.bit_width, dim);
        let mut membership_changed: HashSet<usize> = changed_positions;
        let mut positions: Vec<usize> = affected.into_iter().collect();
        positions.sort_unstable();
        for position in positions {
            let (live, source_rows) = self.read_live(position, true, dim)?;
            if live.n == 0 {
                continue;
            }
            let vectors = self.batch_vectors(&live, dim);
            let (best, best_distances) =
                kmeans::assign(&vectors, live.n, dim, &self.centroids, nlist);
            let current_center =
                &self.centroids[position * dim..(position + 1) * dim].to_vec();
            let mut moved = RowBatch::default();
            let mut moved_destinations: Vec<Vec<u32>> = Vec::new();
            for i in 0..live.n {
                let proposed = best[i];
                if proposed as usize == position {
                    continue;
                }
                let row_vector = &vectors[i * dim..(i + 1) * dim];
                let current_distance: f32 = row_vector
                    .iter()
                    .zip(current_center.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                if best_distances[i] + 1e-6 < current_distance {
                    moved.push_row(
                        live.group_row(i, n_byte_groups),
                        live.scales[i],
                        live.ids[i],
                        false,
                        live.vector(i, dim),
                    );
                    moved_destinations.push(vec![proposed]);
                    self.dead_mark(CopyLocation {
                        partition_id: self.partitions[position].partition_id,
                        row: source_rows[i] as u32,
                        is_replica: false,
                    });
                    membership_changed.insert(position);
                    membership_changed.insert(proposed as usize);
                    replica_refresh.insert(live.ids[i]);
                }
            }
            if moved.n > 0 {
                self.append_batch(&moved, &moved_destinations, dim, entries)?;
            }
        }

        // Centroid refresh.
        for position in 0..self.partitions.len() {
            if !membership_changed.contains(&position) {
                continue;
            }
            let (live, _) = self.read_live(position, true, dim)?;
            if live.n == 0 {
                continue;
            }
            let vectors = self.batch_vectors(&live, dim);
            let mut mean = vec![0.0f64; dim];
            for i in 0..live.n {
                for (m, &v) in mean.iter_mut().zip(&vectors[i * dim..(i + 1) * dim]) {
                    *m += v as f64;
                }
            }
            let centroid = &mut self.centroids[position * dim..(position + 1) * dim];
            for (c, &m) in centroid.iter_mut().zip(&mean) {
                *c = (m / live.n as f64) as f32;
            }
        }
        Ok(())
    }

    /// Ensure each id's closure-assignment coverage matches the current
    /// centroids: compute the closure set of its primary and append replica
    /// copies to partitions that lack one.
    fn refresh_replicas(
        &mut self,
        ids: &[u64],
        dim: usize,
        entries: &mut Vec<RunEntry>,
    ) -> io::Result<()> {
        let Some(epsilon) = self.replica_epsilon else {
            return Ok(());
        };
        let nlist = self.partitions.len();
        if nlist <= 1 {
            return Ok(());
        }
        let mut additions: HashMap<u32, RowBatch> = HashMap::new();
        for &id in ids {
            let copies = self.live_copies(id);
            let Some(primary) = copies.iter().find(|c| !c.is_replica) else {
                continue; // id was removed meanwhile
            };
            let (group_row, scale, vector_exact) = self.copy_row_data(*primary, dim)?;
            let vector = match &vector_exact {
                Some(v) => v.clone(),
                None => self.decode_group_rows(&group_row, &[scale], 1, dim),
            };
            let primary_position = self
                .partition_position(primary.partition_id)
                .expect("live copy implies live partition") as u32;
            let closure = closure_assignments_for_vectors(
                &vector,
                1,
                dim,
                &self.centroids,
                &[primary_position],
                epsilon,
            );
            let existing: HashSet<u32> = copies
                .iter()
                .filter_map(|c| self.partition_position(c.partition_id))
                .map(|p| p as u32)
                .collect();
            for &replica_position in &closure[0] {
                if existing.contains(&replica_position) {
                    continue;
                }
                additions.entry(replica_position).or_default().push_row(
                    &group_row,
                    scale,
                    id,
                    true,
                    vector_exact.as_deref(),
                );
            }
        }
        let mut positions: Vec<u32> = additions.keys().copied().collect();
        positions.sort_unstable();
        for position in positions {
            let rows = additions.remove(&position).expect("position from keys");
            self.append_chunk(position as usize, &rows, dim, entries)?;
        }
        Ok(())
    }

    /// Compact partitions with too many dead rows or chunks: rewrite the
    /// live rows into a single chunk in a new generation file.
    fn collect_garbage(
        &mut self,
        dim: usize,
        entries: &mut Vec<RunEntry>,
        dropped_files: &mut Vec<PathBuf>,
    ) -> io::Result<()> {
        for position in 0..self.partitions.len() {
            let partition = &self.partitions[position];
            let dead = partition.n_rows - partition.live_rows;
            let needs_gc = (partition.n_rows > 0
                && dead as f64 / partition.n_rows as f64 > GC_DEAD_RATIO)
                || partition.chunks.len() > GC_MAX_CHUNKS;
            if !needs_gc {
                continue;
            }
            let (live, _) = self.read_live(position, false, dim)?;
            let partition = &mut self.partitions[position];
            let old_path = segment_path(
                self.directory.as_deref().expect("flush requires a directory"),
                partition.partition_id,
                partition.generation,
            );
            dropped_files.push(old_path);
            partition.generation += 1;
            partition.n_rows = 0;
            partition.live_rows = 0;
            partition.live_primary = 0;
            partition.chunks.clear();
            partition.dead.clear();
            self.invalidate_segment_map(self.partitions[position].partition_id);
            self.append_chunk(position, &live, dim, entries)?;
        }
        Ok(())
    }

    /// Gated re-clustering escape hatch: once enough churn has
    /// accumulated, run the held-out acceptance test and rebuild the
    /// partitioning wholesale if a fresh clustering is clearly better.
    fn rebootstrap_if_due(
        &mut self,
        dim: usize,
        target: usize,
        entries: &mut Vec<RunEntry>,
        dropped_files: &mut Vec<PathBuf>,
    ) -> io::Result<()> {
        let total: u64 = self.partitions.iter().map(|p| p.live_primary).sum();
        if total == 0
            || (self.churn_since_check as f64)
                < REBOOTSTRAP_CHURN_FRACTION * total as f64
        {
            return Ok(());
        }
        self.churn_since_check = 0;

        // Materialize a sample of live primaries with their current
        // partition positions.
        let candidate_k = (total as usize / target).max(MIN_PARTITIONS);
        let sample_rows = clustering_sample(total as usize, candidate_k);
        let sample_set: HashSet<usize> = sample_rows.iter().copied().collect();
        let mut sample_vectors: Vec<f32> = Vec::with_capacity(sample_rows.len() * dim);
        let mut sample_positions: Vec<u32> = Vec::with_capacity(sample_rows.len());
        let mut global_row = 0usize;
        for position in 0..self.partitions.len() {
            let (live, _) = self.read_live(position, true, dim)?;
            if live.n == 0 {
                continue;
            }
            let wanted: Vec<usize> = (0..live.n)
                .filter(|i| sample_set.contains(&(global_row + i)))
                .collect();
            if !wanted.is_empty() {
                let vectors = self.batch_vectors(&live, dim);
                for &i in &wanted {
                    sample_vectors.extend_from_slice(&vectors[i * dim..(i + 1) * dim]);
                    sample_positions.push(position as u32);
                }
            }
            global_row += live.n;
        }
        let n_sample = sample_positions.len();
        if n_sample < MIN_PARTITIONS {
            return Ok(());
        }

        let mut fit_data = Vec::new();
        let mut holdout_data = Vec::new();
        let mut holdout_positions = Vec::new();
        for j in 0..n_sample {
            let row = &sample_vectors[j * dim..(j + 1) * dim];
            if j % HOLDOUT_STRIDE == 0 {
                holdout_data.extend_from_slice(row);
                holdout_positions.push(sample_positions[j]);
            } else {
                fit_data.extend_from_slice(row);
            }
        }
        let n_fit = fit_data.len() / dim;
        let n_holdout = holdout_data.len() / dim;
        if n_fit < MIN_PARTITIONS || n_holdout == 0 {
            return Ok(());
        }
        let current_distortion = disk::mean_distortion_for(
            &holdout_data,
            &holdout_positions,
            &self.centroids,
            dim,
        );
        let (candidate, candidate_distortion) = (0..REBOOTSTRAP_CANDIDATES)
            .map(|attempt| {
                let (candidate, _) = kmeans::kmeans(
                    &fit_data,
                    n_fit,
                    dim,
                    candidate_k,
                    KMEANS_ITERATIONS,
                    KMEANS_SEED ^ self.epoch ^ (attempt as u64) << 32,
                );
                let candidate_nlist = candidate.len() / dim;
                let (_, distances) = kmeans::assign(
                    &holdout_data,
                    n_holdout,
                    dim,
                    &candidate,
                    candidate_nlist,
                );
                let distortion = distances.iter().map(|&d| d as f64).sum::<f64>()
                    / n_holdout.max(1) as f64;
                (candidate, distortion as f32)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("REBOOTSTRAP_CANDIDATES > 0");
        if candidate_distortion < REBOOTSTRAP_DISTORTION_RATIO * current_distortion {
            self.rebuild_all(dim, target, Some(candidate), entries, dropped_files)?;
        }
        Ok(())
    }

    /// Re-cluster the whole corpus: bootstrap (centroids = None, fit on a
    /// sample) or adopt the given candidate centroids; rewrite every
    /// partition. The one global operation — used at the flat-to-clustered
    /// transition and by the drift escape hatch.
    fn rebuild_all(
        &mut self,
        dim: usize,
        target: usize,
        candidate_centroids: Option<Vec<f32>>,
        entries: &mut Vec<RunEntry>,
        dropped_files: &mut Vec<PathBuf>,
    ) -> io::Result<()> {
        // Gather all live primaries.
        let n_byte_groups = pack::n_byte_groups(self.bit_width, dim);
        let mut all = RowBatch::default();
        for position in 0..self.partitions.len() {
            let (live, _) = self.read_live(position, true, dim)?;
            for i in 0..live.n {
                all.push_row(
                    live.group_row(i, n_byte_groups),
                    live.scales[i],
                    live.ids[i],
                    false,
                    live.vector(i, dim),
                );
            }
        }
        let vectors = self.batch_vectors(&all, dim);

        let centroids = match candidate_centroids {
            Some(centroids) => centroids,
            None => {
                let nlist = (all.n / target).max(MIN_PARTITIONS);
                let sample = clustering_sample(all.n, nlist);
                let mut sample_data = Vec::with_capacity(sample.len() * dim);
                for &i in &sample {
                    sample_data.extend_from_slice(&vectors[i * dim..(i + 1) * dim]);
                }
                let (centroids, _) = kmeans::kmeans(
                    &sample_data,
                    sample.len(),
                    dim,
                    nlist,
                    KMEANS_ITERATIONS,
                    KMEANS_SEED,
                );
                centroids
            }
        };
        let nlist = centroids.len() / dim;
        let mut assignments = vec![0u32; all.n];
        for (chunk_start, chunk) in (0..all.n)
            .collect::<Vec<_>>()
            .chunks(CLUSTERING_CHUNK)
            .map(|c| (c[0], c))
        {
            let chunk_data = &vectors[chunk_start * dim..(chunk_start + chunk.len()) * dim];
            let (chunk_assignments, _) =
                kmeans::assign(chunk_data, chunk.len(), dim, &centroids, nlist);
            assignments[chunk_start..chunk_start + chunk.len()]
                .copy_from_slice(&chunk_assignments);
        }
        let replica_lists = match self.replica_epsilon {
            Some(epsilon) if nlist > 1 => closure_assignments_for_vectors(
                &vectors,
                all.n,
                dim,
                &centroids,
                &assignments,
                epsilon,
            ),
            _ => vec![Vec::new(); all.n],
        };

        // Drop every old partition, create the new set, append everything.
        while !self.partitions.is_empty() {
            self.drop_partition(0, dim, dropped_files);
        }
        self.centroids = Vec::new();
        for c in 0..nlist {
            self.create_partition(&centroids[c * dim..(c + 1) * dim], entries)?;
        }
        let destinations: Vec<Vec<u32>> = assignments
            .iter()
            .zip(&replica_lists)
            .map(|(&primary, replicas)| {
                let mut destinations = Vec::with_capacity(1 + replicas.len());
                destinations.push(primary);
                destinations.extend(replicas.iter().copied());
                destinations
            })
            .collect();
        self.append_batch(&all, &destinations, dim, entries)?;
        // A full re-cluster starts from a fresh structure: accumulated
        // churn no longer measures drift against it.
        self.churn_since_check = 0;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Partition bookkeeping
    // ------------------------------------------------------------------

    fn partition_position(&self, partition_id: u32) -> Option<usize> {
        self.partitions
            .iter()
            .position(|p| p.partition_id == partition_id)
    }

    fn create_partition(
        &mut self,
        centroid: &[f32],
        _entries: &mut [RunEntry],
    ) -> io::Result<usize> {
        let partition_id = self.next_partition_id;
        self.next_partition_id += 1;
        self.partitions.push(PartitionState {
            partition_id,
            generation: 0,
            n_rows: 0,
            live_rows: 0,
            live_primary: 0,
            chunks: Vec::new(),
            dead: Vec::new(),
        });
        self.centroids.extend_from_slice(centroid);
        Ok(self.partitions.len() - 1)
    }

    fn drop_partition(
        &mut self,
        position: usize,
        dim: usize,
        dropped_files: &mut Vec<PathBuf>,
    ) {
        let partition = self.partitions.remove(position);
        if let Some(directory) = self.directory.as_deref() {
            dropped_files.push(segment_path(
                directory,
                partition.partition_id,
                partition.generation,
            ));
        }
        self.invalidate_segment_map(partition.partition_id);
        self.centroids.drain(position * dim..(position + 1) * dim);
    }

    /// Append `rows` to `position`'s segment file as one chunk, fsync, and
    /// record run entries for every row.
    fn append_chunk(
        &mut self,
        position: usize,
        rows: &RowBatch,
        dim: usize,
        entries: &mut Vec<RunEntry>,
    ) -> io::Result<()> {
        if rows.n == 0 {
            return Ok(());
        }
        let directory = self
            .directory
            .as_deref()
            .expect("flush requires a bound directory")
            .to_path_buf();
        let n_byte_groups = pack::n_byte_groups(self.bit_width, dim);
        let partition = &mut self.partitions[position];
        let path = segment_path(&directory, partition.partition_id, partition.generation);
        let append_offset =
            partition.file_len(self.bit_width, dim, self.store_vectors);

        let layout = chunk_layout(self.bit_width, dim, rows.n, self.store_vectors);
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)?;
        file.set_len(append_offset)?; // truncate any orphaned crashed tail
        file.seek(SeekFrom::Start(append_offset))?;
        let mut writer = BufWriter::new(&mut file);

        // Header.
        let mut header = [0u8; CHUNK_ALIGN];
        header[0..4].copy_from_slice(CHUNK_MAGIC);
        header[4..8].copy_from_slice(&(rows.n as u32).to_le_bytes());
        header[8..12]
            .copy_from_slice(&(((rows.n + BLOCK - 1) / BLOCK) as u32).to_le_bytes());
        writer.write_all(&header)?;
        let mut position_bytes = CHUNK_ALIGN;

        // Blocked codes.
        debug_assert_eq!(position_bytes, layout.codes.0);
        let mut streamer = BlockStreamer::new(&mut writer, n_byte_groups);
        for i in 0..rows.n {
            streamer.push_row(rows.group_row(i, n_byte_groups))?;
        }
        position_bytes += streamer.finish()?;
        position_bytes = pad_to(&mut writer, position_bytes, layout.scales.0)?;

        writer.write_all(f32_bytes(&rows.scales))?;
        position_bytes += 4 * rows.n;
        position_bytes = pad_to(&mut writer, position_bytes, layout.ids.0)?;

        for &id in &rows.ids {
            writer.write_all(&id.to_le_bytes())?;
        }
        position_bytes += 8 * rows.n;
        position_bytes = pad_to(&mut writer, position_bytes, layout.replica_bits.0)?;

        let mut replica_bitmap = vec![0u8; (rows.n + 7) / 8];
        for (i, &is_replica) in rows.replica.iter().enumerate() {
            if is_replica {
                replica_bitmap[i / 8] |= 1 << (i % 8);
            }
        }
        writer.write_all(&replica_bitmap)?;
        position_bytes += replica_bitmap.len();

        if self.store_vectors {
            position_bytes = pad_to(&mut writer, position_bytes, layout.vectors.0)?;
            writer.write_all(f32_bytes(&rows.vectors))?;
            position_bytes += 4 * rows.n * dim;
        }
        position_bytes = pad_to(&mut writer, position_bytes, layout.total_len)?;
        debug_assert_eq!(position_bytes, layout.total_len);
        writer.flush()?;
        drop(writer);
        file.sync_all()?;

        // Bookkeeping + run entries.
        let base_row = partition.n_rows;
        partition.chunks.push(ChunkMeta {
            offset: append_offset,
            n_rows: rows.n as u32,
        });
        partition.n_rows += rows.n as u64;
        partition.live_rows += rows.n as u64;
        let new_bits = ((partition.n_rows + 7) / 8) as usize;
        partition.dead.resize(new_bits, 0);
        for i in 0..rows.n {
            if !rows.replica[i] {
                partition.live_primary += 1;
            }
            entries.push(RunEntry {
                id: rows.ids[i],
                partition_id: partition.partition_id,
                generation: partition.generation,
                row: (base_row + i as u64) as u32,
                flags: if rows.replica[i] { ENTRY_FLAG_REPLICA } else { 0 },
            });
        }
        self.invalidate_segment_map(self.partitions[position].partition_id);
        Ok(())
    }

    fn dead_mark(&mut self, copy: CopyLocation) {
        let Some(position) = self.partition_position(copy.partition_id) else {
            return;
        };
        let partition = &mut self.partitions[position];
        let byte = (copy.row / 8) as usize;
        let bit = 1u8 << (copy.row % 8);
        if partition.dead[byte] & bit != 0 {
            return;
        }
        partition.dead[byte] |= bit;
        partition.live_rows -= 1;
        if !copy.is_replica {
            partition.live_primary -= 1;
        }
    }

    // ------------------------------------------------------------------
    // Id-run membership
    // ------------------------------------------------------------------

    /// All currently-valid live copies of `id`: entries from the run
    /// tables whose partition still exists at the recorded generation and
    /// whose row is not dead-marked.
    fn live_copies(&self, id: u64) -> Vec<CopyLocation> {
        let mut found: Vec<CopyLocation> = Vec::new();
        let mut seen: HashSet<(u32, u32)> = HashSet::new();
        for &generation in self.runs.iter().rev() {
            let Ok(map) = self.run_map(generation) else {
                continue;
            };
            let count = run_entry_count(&map);
            // Binary search for the first entry with this id.
            let mut lo = 0usize;
            let mut hi = count;
            while lo < hi {
                let mid = (lo + hi) / 2;
                if run_entry_id(&map, mid) < id {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            let mut idx = lo;
            while idx < count && run_entry_id(&map, idx) == id {
                let entry = run_entry(&map, idx);
                idx += 1;
                let Some(position) = self.partition_position(entry.partition_id) else {
                    continue;
                };
                let partition = &self.partitions[position];
                if partition.generation != entry.generation
                    || entry.row as u64 >= partition.n_rows
                    || partition.is_dead(entry.row as u64)
                {
                    continue;
                }
                if seen.insert((entry.partition_id, entry.row)) {
                    found.push(CopyLocation {
                        partition_id: entry.partition_id,
                        row: entry.row,
                        is_replica: entry.flags & ENTRY_FLAG_REPLICA != 0,
                    });
                }
            }
        }
        found
    }

    /// The full-precision vector at a live copy (store_vectors only).
    fn copy_vector(&self, copy: CopyLocation) -> Vec<f32> {
        let dim = self.dim();
        let (_, _, vector) = self
            .copy_row_data(copy, dim)
            .expect("live copy must be readable");
        vector.expect("copy_vector requires store_vectors")
    }

    /// Raw row data of a copy: group bytes, scale, and (when stored) the
    /// exact vector.
    fn copy_row_data(
        &self,
        copy: CopyLocation,
        dim: usize,
    ) -> io::Result<(Vec<u8>, f32, Option<Vec<f32>>)> {
        let position = self
            .partition_position(copy.partition_id)
            .ok_or_else(|| invalid_data("copy in unknown partition".to_string()))?;
        let partition = &self.partitions[position];
        let map = self.segment_map(partition.partition_id, partition.generation)?;
        let n_byte_groups = pack::n_byte_groups(self.bit_width, dim);
        let mut base_row = 0u64;
        for chunk in &partition.chunks {
            if (copy.row as u64) < base_row + chunk.n_rows as u64 {
                let local = (copy.row as u64 - base_row) as usize;
                let n_rows = chunk.n_rows as usize;
                let layout = chunk_layout(self.bit_width, dim, n_rows, self.store_vectors);
                let start = chunk.offset as usize;
                let codes =
                    &map[start + layout.codes.0..start + layout.codes.0 + layout.codes.1];
                let block_bytes = n_byte_groups * BLOCK;
                let block_idx = local / BLOCK;
                let mut block_rows = vec![0u8; BLOCK * n_byte_groups];
                pack::unpack_block_rows(
                    &codes[block_idx * block_bytes..(block_idx + 1) * block_bytes],
                    n_byte_groups,
                    &mut block_rows,
                );
                let lane = local % BLOCK;
                let group_row =
                    block_rows[lane * n_byte_groups..(lane + 1) * n_byte_groups].to_vec();
                let scales = f32_slice(
                    &map[start + layout.scales.0
                        ..start + layout.scales.0 + layout.scales.1],
                );
                let vector = if self.store_vectors {
                    let vectors = f32_slice(
                        &map[start + layout.vectors.0
                            ..start + layout.vectors.0 + layout.vectors.1],
                    );
                    Some(vectors[local * dim..(local + 1) * dim].to_vec())
                } else {
                    None
                };
                return Ok((group_row, scales[local], vector));
            }
            base_row += chunk.n_rows as u64;
        }
        Err(invalid_data("copy row out of range".to_string()))
    }

    fn write_run(&mut self, entries: &[RunEntry]) -> io::Result<()> {
        let directory = self
            .directory
            .as_deref()
            .expect("flush requires a bound directory");
        let mut sorted: Vec<&RunEntry> = entries.iter().collect();
        sorted.sort_unstable_by_key(|e| e.id);
        let generation = self.next_run_generation;
        self.next_run_generation += 1;
        let path = run_path(directory, generation);
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(RUN_MAGIC)?;
        writer.write_all(&[FORMAT_VERSION, 0, 0, 0])?;
        writer.write_all(&(sorted.len() as u64).to_le_bytes())?;
        for entry in sorted {
            writer.write_all(&entry.id.to_le_bytes())?;
            writer.write_all(&entry.partition_id.to_le_bytes())?;
            writer.write_all(&entry.generation.to_le_bytes())?;
            writer.write_all(&entry.row.to_le_bytes())?;
            writer.write_all(&entry.flags.to_le_bytes())?;
        }
        writer.flush()?;
        writer.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        self.runs.push(generation);
        Ok(())
    }

    /// Merge every run into one, keeping only currently-valid entries.
    fn merge_runs(&mut self, dropped_files: &mut Vec<PathBuf>) -> io::Result<()> {
        let directory = self
            .directory
            .as_deref()
            .expect("flush requires a bound directory")
            .to_path_buf();
        let mut merged: Vec<RunEntry> = Vec::new();
        let mut seen: HashSet<(u32, u32)> = HashSet::new();
        for &generation in self.runs.iter().rev() {
            let map = self.run_map(generation)?;
            let count = run_entry_count(&map);
            for idx in 0..count {
                let entry = run_entry(&map, idx);
                let Some(position) = self.partition_position(entry.partition_id) else {
                    continue;
                };
                let partition = &self.partitions[position];
                if partition.generation != entry.generation
                    || entry.row as u64 >= partition.n_rows
                    || partition.is_dead(entry.row as u64)
                {
                    continue;
                }
                if seen.insert((entry.partition_id, entry.row)) {
                    merged.push(entry);
                }
            }
        }
        for &generation in &self.runs {
            dropped_files.push(run_path(&directory, generation));
        }
        self.runs.clear();
        self.run_maps.lock().expect("run map lock").clear();
        self.write_run(&merged)
    }

    // ------------------------------------------------------------------
    // Mmap caches
    // ------------------------------------------------------------------

    fn segment_map(&self, partition_id: u32, generation: u32) -> io::Result<Arc<Mmap>> {
        let mut cache = self.segment_maps.lock().expect("segment map lock");
        if let Some(map) = cache.get(&partition_id) {
            return Ok(Arc::clone(map));
        }
        let directory = self
            .directory
            .as_deref()
            .ok_or_else(|| invalid_data("index has no backing directory".to_string()))?;
        let file = File::open(segment_path(directory, partition_id, generation))?;
        // SAFETY: segments are append-only and replaced whole (new
        // generation file) — written bytes are never mutated; same contract
        // as the .tvdm mapping.
        let map = Arc::new(unsafe { Mmap::map(&file)? });
        cache.insert(partition_id, Arc::clone(&map));
        Ok(map)
    }

    fn invalidate_segment_map(&self, partition_id: u32) {
        self.segment_maps
            .lock()
            .expect("segment map lock")
            .remove(&partition_id);
    }

    fn run_map(&self, generation: u64) -> io::Result<Arc<Mmap>> {
        let mut cache = self.run_maps.lock().expect("run map lock");
        if let Some(map) = cache.get(&generation) {
            return Ok(Arc::clone(map));
        }
        let directory = self
            .directory
            .as_deref()
            .ok_or_else(|| invalid_data("index has no backing directory".to_string()))?;
        let file = File::open(run_path(directory, generation))?;
        // SAFETY: run files are immutable once written.
        let map = Arc::new(unsafe { Mmap::map(&file)? });
        cache.insert(generation, Arc::clone(&map));
        Ok(map)
    }

    fn rotation_for(&self, dim: usize) -> &[f32] {
        self.rotation_cache
            .get_or_init(|| rotation::make_rotation_matrix(dim))
    }

    fn codebook_for(&self, dim: usize) -> &[f32] {
        self.codebook_cache.get_or_init(|| {
            let (_, centroids) = codebook::codebook(self.bit_width, dim);
            centroids
        })
    }

    // ------------------------------------------------------------------
    // WAL
    // ------------------------------------------------------------------

    fn wal_append(&mut self, kind: u8, id: u64, vector: &[f32]) -> io::Result<()> {
        let Some(wal) = self.wal.as_mut() else {
            return Ok(()); // no directory bound yet: RAM-only mode
        };
        let payload = f32_bytes(vector);
        let mut crc_input = Vec::with_capacity(9 + payload.len());
        crc_input.push(kind);
        crc_input.extend_from_slice(&id.to_le_bytes());
        crc_input.extend_from_slice(payload);
        let crc = crc32(&crc_input);
        let mut record = Vec::with_capacity(16 + payload.len());
        record.push(kind);
        record.extend_from_slice(&[0u8; 3]);
        record.extend_from_slice(&crc.to_le_bytes());
        record.extend_from_slice(&id.to_le_bytes());
        record.extend_from_slice(payload);
        wal.write_all(&record)
    }

    fn wal_sync(&mut self) {
        if let Some(wal) = self.wal.as_mut() {
            wal.sync_data().ok();
        }
    }

    fn reset_wal(&mut self) -> io::Result<()> {
        let Some(directory) = self.directory.as_deref() else {
            return Ok(());
        };
        let path = directory.join(WAL_FILE);
        let mut file = File::create(path)?;
        file.write_all(WAL_MAGIC)?;
        file.write_all(&[FORMAT_VERSION, 0, 0, 0])?;
        file.write_all(&(self.dim() as u32).to_le_bytes())?;
        file.write_all(&[0u8; 4])?;
        file.write_all(&self.epoch.to_le_bytes())?;
        file.sync_all()?;
        self.wal = Some(OpenOptions::new().append(true).open(directory.join(WAL_FILE))?);
        self.wal_dim = self.dim();
        Ok(())
    }

    fn replay_or_reset_wal(&mut self) -> io::Result<()> {
        let directory = self
            .directory
            .clone()
            .expect("open always binds a directory");
        let path = directory.join(WAL_FILE);
        let bytes = fs::read(&path).unwrap_or_default();
        let current_epoch = Self::parse_wal_epoch(&bytes) == Some(self.epoch);
        if !current_epoch {
            // Stale (pre-flush) or absent log: its records are already in
            // the manifest state (or it never existed). Start fresh —
            // appending to a stale-epoch log would silently discard the
            // appended records at the next open.
            return self.reset_wal();
        }
        for (kind, id, vector) in Self::parse_wal(&bytes, self.epoch, self.dim()) {
            match kind {
                WAL_ADD => {
                    let dim = vector.len();
                    // Records were validated when first applied; a failure
                    // here means the log disagrees with the manifest.
                    self.memtable
                        .add_with_ids_2d(&vector, dim, &[id])
                        .map_err(|e| invalid_data(format!("WAL replay: {e}")))?;
                    if self.store_vectors {
                        self.memtable_originals.insert(id, vector.into());
                    }
                }
                WAL_REMOVE => {
                    self.remove_internal(id);
                }
                _ => unreachable!("parse_wal filters kinds"),
            }
        }
        // Keep appending to the same log: its records replay idempotently
        // against this manifest epoch.
        self.wal = Some(OpenOptions::new().append(true).open(&path)?);
        self.wal_dim = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        Ok(())
    }

    fn parse_wal_epoch(bytes: &[u8]) -> Option<u64> {
        if bytes.len() < 24 || &bytes[0..4] != WAL_MAGIC || bytes[4] != FORMAT_VERSION {
            return None;
        }
        Some(u64::from_le_bytes(bytes[16..24].try_into().unwrap()))
    }

    /// Parse WAL records for `epoch`; stops at the first corrupt record.
    fn parse_wal(bytes: &[u8], epoch: u64, dim: usize) -> Vec<(u8, u64, Vec<f32>)> {
        let mut records = Vec::new();
        if bytes.len() < 24 || &bytes[0..4] != WAL_MAGIC || bytes[4] != FORMAT_VERSION {
            return records;
        }
        let wal_dim = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let wal_epoch = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        if wal_epoch != epoch || (dim != 0 && wal_dim != dim) {
            return records;
        }
        let mut offset = 24usize;
        while offset + 16 <= bytes.len() {
            let kind = bytes[offset];
            let crc = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
            let id = u64::from_le_bytes(bytes[offset + 8..offset + 16].try_into().unwrap());
            let payload_len = match kind {
                WAL_ADD => 4 * wal_dim,
                WAL_REMOVE => 0,
                _ => break,
            };
            if offset + 16 + payload_len > bytes.len() {
                break;
            }
            let payload = &bytes[offset + 16..offset + 16 + payload_len];
            let mut crc_input = Vec::with_capacity(9 + payload.len());
            crc_input.push(kind);
            crc_input.extend_from_slice(&id.to_le_bytes());
            crc_input.extend_from_slice(payload);
            if crc32(&crc_input) != crc {
                break;
            }
            let vector: Vec<f32> = payload
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            records.push((kind, id, vector));
            offset += 16 + payload_len;
        }
        records
    }

    // ------------------------------------------------------------------
    // Manifest
    // ------------------------------------------------------------------

    fn write_manifest(&self) -> io::Result<()> {
        let directory = self
            .directory
            .as_deref()
            .expect("flush requires a bound directory");
        let dim = self.dim();
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(MANIFEST_MAGIC);
        let mut flags = 0u8;
        if self.store_vectors {
            flags |= FLAG_HAS_VECTORS;
        }
        if self.clustered {
            flags |= FLAG_CLUSTERED;
        }
        out.push(FORMAT_VERSION);
        out.push(self.bit_width as u8);
        out.push(flags);
        out.push(0);
        out.extend_from_slice(&(dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.tqplus_shift.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.epoch.to_le_bytes());
        out.extend_from_slice(&(self.partition_target.unwrap_or(0) as u32).to_le_bytes());
        out.extend_from_slice(&self.replica_epsilon.unwrap_or(0.0).to_le_bytes());
        out.extend_from_slice(&self.next_partition_id.to_le_bytes());
        out.extend_from_slice(&(self.partitions.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.next_run_generation.to_le_bytes());
        out.extend_from_slice(&self.churn_since_check.to_le_bytes());
        out.extend_from_slice(&(self.runs.len() as u32).to_le_bytes());
        out.extend_from_slice(&[0u8; 4]);
        out.extend_from_slice(f32_bytes(&self.tqplus_shift));
        out.extend_from_slice(f32_bytes(&self.tqplus_scale));
        for &generation in &self.runs {
            out.extend_from_slice(&generation.to_le_bytes());
        }
        for partition in &self.partitions {
            out.extend_from_slice(&partition.partition_id.to_le_bytes());
            out.extend_from_slice(&partition.generation.to_le_bytes());
            out.extend_from_slice(&partition.n_rows.to_le_bytes());
            out.extend_from_slice(&partition.live_rows.to_le_bytes());
            out.extend_from_slice(&partition.live_primary.to_le_bytes());
            out.extend_from_slice(&(partition.chunks.len() as u32).to_le_bytes());
            out.extend_from_slice(&[0u8; 4]);
            for chunk in &partition.chunks {
                out.extend_from_slice(&chunk.offset.to_le_bytes());
                out.extend_from_slice(&chunk.n_rows.to_le_bytes());
                out.extend_from_slice(&[0u8; 4]);
            }
            let bitmap_len = ((partition.n_rows + 7) / 8) as usize;
            debug_assert_eq!(bitmap_len, partition.dead.len());
            out.extend_from_slice(&partition.dead);
        }
        out.extend_from_slice(f32_bytes(&self.centroids));
        let crc = crc32(&out);
        out.extend_from_slice(&crc.to_le_bytes());

        let temp_path = directory.join(MANIFEST_TEMP_FILE);
        let mut file = File::create(&temp_path)?;
        file.write_all(&out)?;
        file.sync_all()?;
        fs::rename(&temp_path, directory.join(MANIFEST_FILE))?;
        if let Ok(dir_handle) = File::open(directory) {
            dir_handle.sync_all().ok();
        }
        Ok(())
    }

    fn read_manifest(directory: &Path) -> io::Result<Self> {
        let bytes = fs::read(directory.join(MANIFEST_FILE))?;
        if bytes.len() < 52 + 4 || &bytes[0..4] != MANIFEST_MAGIC {
            return Err(invalid_data("not a FreshIndex manifest".to_string()));
        }
        if bytes[4] != FORMAT_VERSION {
            return Err(invalid_data(format!(
                "unsupported FreshIndex manifest version {}",
                bytes[4],
            )));
        }
        let (body, crc_bytes) = bytes.split_at(bytes.len() - 4);
        let stored_crc = u32::from_le_bytes(crc_bytes.try_into().unwrap());
        if crc32(body) != stored_crc {
            return Err(invalid_data("corrupt FreshIndex manifest (crc)".to_string()));
        }
        let bit_width = bytes[5] as usize;
        if !(2..=4).contains(&bit_width) {
            return Err(invalid_data(format!("invalid bit_width {bit_width}")));
        }
        let flags = bytes[6];
        let store_vectors = flags & FLAG_HAS_VECTORS != 0;
        let clustered = flags & FLAG_CLUSTERED != 0;
        let mut cursor = Cursor::new(body, 8);
        let dim = cursor.u32()? as usize;
        let n_calib = cursor.u32()? as usize;
        let epoch = cursor.u64()?;
        let target = cursor.u32()? as usize;
        let replica_epsilon_raw = cursor.f32()?;
        let next_partition_id = cursor.u32()?;
        let n_partitions = cursor.u32()? as usize;
        let next_run_generation = cursor.u64()?;
        let churn_since_check = cursor.u64()?;
        let n_runs = cursor.u32()? as usize;
        cursor.skip(4)?;
        let tqplus_shift = cursor.f32s(n_calib)?;
        let tqplus_scale = cursor.f32s(n_calib)?;
        let mut runs = Vec::with_capacity(n_runs);
        for _ in 0..n_runs {
            runs.push(cursor.u64()?);
        }
        let mut partitions = Vec::with_capacity(n_partitions);
        for _ in 0..n_partitions {
            let partition_id = cursor.u32()?;
            let generation = cursor.u32()?;
            let n_rows = cursor.u64()?;
            let live_rows = cursor.u64()?;
            let live_primary = cursor.u64()?;
            let n_chunks = cursor.u32()? as usize;
            cursor.skip(4)?;
            let mut chunks = Vec::with_capacity(n_chunks);
            for _ in 0..n_chunks {
                let offset = cursor.u64()?;
                let chunk_rows = cursor.u32()?;
                cursor.skip(4)?;
                chunks.push(ChunkMeta {
                    offset,
                    n_rows: chunk_rows,
                });
            }
            let bitmap_len = ((n_rows + 7) / 8) as usize;
            let dead = cursor.bytes(bitmap_len)?.to_vec();
            partitions.push(PartitionState {
                partition_id,
                generation,
                n_rows,
                live_rows,
                live_primary,
                chunks,
                dead,
            });
        }
        let centroids = cursor.f32s(n_partitions * dim)?;

        let memtable = if dim > 0 {
            if n_calib > 0 {
                let inner = TurboQuantIndex::from_parts(
                    Some(dim),
                    bit_width,
                    0,
                    Vec::new(),
                    Vec::new(),
                    tqplus_shift.clone(),
                    tqplus_scale.clone(),
                );
                IdMapIndex::from_inner(inner)
            } else {
                IdMapIndex::new(dim, bit_width)
                    .map_err(|e| invalid_data(format!("invalid manifest parameters: {e}")))?
            }
        } else {
            IdMapIndex::new_lazy(bit_width)
                .map_err(|e| invalid_data(format!("invalid manifest parameters: {e}")))?
        };

        Ok(Self {
            directory: Some(directory.to_path_buf()),
            bit_width,
            store_vectors,
            replica_epsilon: if replica_epsilon_raw > 0.0 {
                Some(replica_epsilon_raw)
            } else {
                None
            },
            partition_target: if target > 0 { Some(target) } else { None },
            clustered,
            epoch,
            next_partition_id,
            next_run_generation,
            churn_since_check,
            tqplus_shift,
            tqplus_scale,
            centroids,
            partitions,
            runs,
            memtable,
            memtable_originals: HashMap::new(),
            wal: None,
            wal_dim: 0,
            segment_maps: Mutex::new(HashMap::new()),
            run_maps: Mutex::new(HashMap::new()),
            rotation_cache: OnceLock::new(),
            codebook_cache: OnceLock::new(),
        })
    }

    /// Remove files the manifest does not reference (crashed flushes), and
    /// truncate segment files to their recorded lengths.
    fn cleanup_directory(&self) -> io::Result<()> {
        let directory = self
            .directory
            .as_deref()
            .expect("open always binds a directory");
        let dim = self.dim();
        let known_segments: HashMap<u32, u32> = self
            .partitions
            .iter()
            .map(|p| (p.partition_id, p.generation))
            .collect();
        let known_runs: HashSet<u64> = self.runs.iter().copied().collect();
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == MANIFEST_FILE || name == WAL_FILE {
                continue;
            }
            if let Some(rest) = name.strip_prefix("segment-") {
                let mut parts = rest.splitn(2, '-');
                let id = parts.next().and_then(|s| s.parse::<u32>().ok());
                let generation = parts.next().and_then(|s| s.parse::<u32>().ok());
                if let (Some(id), Some(generation)) = (id, generation) {
                    if known_segments.get(&id) == Some(&generation) {
                        // Truncate any orphaned tail from a crashed append.
                        if let Some(partition) =
                            self.partitions.iter().find(|p| p.partition_id == id)
                        {
                            let expected =
                                partition.file_len(self.bit_width, dim, self.store_vectors);
                            if let Ok(metadata) = entry.metadata() {
                                if metadata.len() > expected {
                                    OpenOptions::new()
                                        .write(true)
                                        .open(entry.path())?
                                        .set_len(expected)?;
                                }
                            }
                        }
                        continue;
                    }
                }
            } else if let Some(rest) = name.strip_prefix("ids-") {
                if let Ok(generation) = rest.parse::<u64>() {
                    if known_runs.contains(&generation) {
                        continue;
                    }
                }
            }
            fs::remove_file(entry.path()).ok();
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Migration
    // ------------------------------------------------------------------

    /// Build a FreshIndex directory from a `.tvdm` [`crate::DiskIndex`]
    /// file. Lossless: codes, scales, calibration, ids, replica flags,
    /// stored vectors, centroids and the partitioning carry over unchanged.
    pub fn import_disk_index_file(
        src: impl AsRef<Path>,
        directory: impl AsRef<Path>,
    ) -> io::Result<Self> {
        let contents = disk::read_tvdm_contents(src.as_ref())?;
        let directory = directory.as_ref();
        fs::create_dir_all(directory)?;
        if directory.join(MANIFEST_FILE).exists() {
            return Err(invalid_data(format!(
                "directory {} already contains a FreshIndex",
                directory.display(),
            )));
        }
        let mut index = Self::new(
            if contents.dim > 0 {
                Some(contents.dim)
            } else {
                None
            },
            contents.bit_width,
        )
        .map_err(|e| invalid_data(format!("invalid .tvdm parameters: {e}")))?;
        index.directory = Some(directory.to_path_buf());
        index.store_vectors = contents.store_vectors;
        index.replica_epsilon = contents.replica_epsilon;
        index.partition_target = contents.partition_target;
        index.tqplus_shift = contents.tqplus_shift;
        index.tqplus_scale = contents.tqplus_scale;
        index.clustered = contents.partitions.len() > 1;

        let dim = contents.dim;
        let mut entries: Vec<RunEntry> = Vec::new();
        for (p, partition) in contents.partitions.iter().enumerate() {
            let centroid: Vec<f32> = if contents.centroids.is_empty() {
                vec![0.0; dim]
            } else {
                contents.centroids[p * dim..(p + 1) * dim].to_vec()
            };
            let position = index.create_partition(&centroid, &mut entries)?;
            let mut batch = RowBatch::default();
            let n_byte_groups = pack::n_byte_groups(contents.bit_width, dim.max(1));
            for i in 0..partition.ids.len() {
                batch.push_row(
                    &partition.group_rows[i * n_byte_groups..(i + 1) * n_byte_groups],
                    partition.scales[i],
                    partition.ids[i],
                    partition.replica[i],
                    partition
                        .vectors
                        .as_ref()
                        .map(|v| &v[i * dim..(i + 1) * dim]),
                );
            }
            index.append_chunk(position, &batch, dim, &mut entries)?;
        }
        if !entries.is_empty() {
            index.write_run(&entries)?;
        }
        index.epoch = 1;
        index.write_manifest()?;
        index.reset_wal()?;
        Ok(index)
    }

    /// Write the live primaries to a `.tvim` [`IdMapIndex`] file —
    /// lossless for codes/scales/calibration/ids (partitioning, replicas
    /// and stored vectors have no `.tvim` representation). Convert onward
    /// with [`crate::DiskIndex::convert_id_map_file`] for a `.tvdm`.
    pub fn export_id_map_file(&self, dst: impl AsRef<Path>) -> io::Result<()> {
        let dim = self.dim();
        if self.memtable_len() > 0 {
            return Err(invalid_data(
                "export requires a flushed index (save first)".to_string(),
            ));
        }
        let n_byte_groups = pack::n_byte_groups(self.bit_width, dim.max(1));
        let mut all = RowBatch::default();
        for position in 0..self.partitions.len() {
            let (live, _) = self.read_live(position, true, dim)?;
            for i in 0..live.n {
                all.push_row(
                    live.group_row(i, n_byte_groups),
                    live.scales[i],
                    live.ids[i],
                    false,
                    None,
                );
            }
        }
        let packed =
            pack::packed_from_group_bytes(&all.group_rows, all.n, self.bit_width, dim);
        crate::io::write_id_map(
            dst.as_ref(),
            self.bit_width,
            dim,
            all.n,
            &packed,
            &all.scales,
            &self.tqplus_shift,
            &self.tqplus_scale,
            &all.ids,
        )
    }
}

impl std::fmt::Debug for FreshIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FreshIndex")
            .field("bit_width", &self.bit_width)
            .field("dim", &self.dim_opt())
            .field("directory", &self.directory)
            .field("len", &self.len())
            .field("nlist", &self.nlist())
            .field("memtable_len", &self.memtable_len())
            .field("dead_count", &self.dead_count())
            .field("replica_count", &self.replica_count())
            .field("run_count", &self.run_count())
            .field("chunk_count", &self.chunk_count())
            .field("epoch", &self.epoch)
            .finish()
    }
}

// ----------------------------------------------------------------------
// Free helpers
// ----------------------------------------------------------------------

fn segment_path(directory: &Path, partition_id: u32, generation: u32) -> PathBuf {
    directory.join(format!("segment-{partition_id:08}-{generation:08}"))
}

fn run_path(directory: &Path, generation: u64) -> PathBuf {
    directory.join(format!("ids-{generation:016}"))
}

fn pad_to<W: Write>(writer: &mut W, position: usize, target: usize) -> io::Result<usize> {
    debug_assert!(target >= position, "pad_to target before position");
    let zeros = [0u8; CHUNK_ALIGN];
    let mut remaining = target - position;
    while remaining > 0 {
        let step = remaining.min(CHUNK_ALIGN);
        writer.write_all(&zeros[..step])?;
        remaining -= step;
    }
    Ok(target)
}

fn f32_slice(bytes: &[u8]) -> &[f32] {
    if bytes.is_empty() {
        return &[];
    }
    debug_assert_eq!(bytes.as_ptr() as usize % std::mem::align_of::<f32>(), 0);
    // SAFETY: section offsets are 8-byte aligned within page-aligned
    // mappings, lengths are multiples of 4, f32 has no invalid patterns.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4) }
}

fn u64_slice(bytes: &[u8]) -> &[u64] {
    if bytes.is_empty() {
        return &[];
    }
    debug_assert_eq!(bytes.as_ptr() as usize % std::mem::align_of::<u64>(), 0);
    // SAFETY: as above with 8-byte units.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u64>(), bytes.len() / 8) }
}

fn run_entry_count(map: &Mmap) -> usize {
    if map.len() < RUN_HEADER_BYTES {
        return 0;
    }
    u64::from_le_bytes(map[8..16].try_into().unwrap()) as usize
}

fn run_entry_id(map: &Mmap, idx: usize) -> u64 {
    let offset = RUN_HEADER_BYTES + idx * RUN_ENTRY_BYTES;
    u64::from_le_bytes(map[offset..offset + 8].try_into().unwrap())
}

fn run_entry(map: &Mmap, idx: usize) -> RunEntry {
    let offset = RUN_HEADER_BYTES + idx * RUN_ENTRY_BYTES;
    RunEntry {
        id: u64::from_le_bytes(map[offset..offset + 8].try_into().unwrap()),
        partition_id: u32::from_le_bytes(map[offset + 8..offset + 12].try_into().unwrap()),
        generation: u32::from_le_bytes(map[offset + 12..offset + 16].try_into().unwrap()),
        row: u32::from_le_bytes(map[offset + 16..offset + 20].try_into().unwrap()),
        flags: u32::from_le_bytes(map[offset + 20..offset + 24].try_into().unwrap()),
    }
}

/// Sequential little-endian reader over a byte slice.
struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8], offset: usize) -> Self {
        Self { bytes, offset }
    }

    fn bytes(&mut self, len: usize) -> io::Result<&'a [u8]> {
        if self.offset + len > self.bytes.len() {
            return Err(invalid_data("truncated FreshIndex manifest".to_string()));
        }
        let slice = &self.bytes[self.offset..self.offset + len];
        self.offset += len;
        Ok(slice)
    }

    fn skip(&mut self, len: usize) -> io::Result<()> {
        self.bytes(len).map(|_| ())
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    fn f32s(&mut self, count: usize) -> io::Result<Vec<f32>> {
        let bytes = self.bytes(4 * count)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
}
