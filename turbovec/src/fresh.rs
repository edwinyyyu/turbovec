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

use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use memmap2::{Advice, Mmap};

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
// v3 moves the centroid blob out of the manifest into `centroids-<gen>`,
// referenced by generation. The manifest is rewritten every save and carries
// `nlist * dim * 4` bytes of centroids; measured, 73% of those saves change no
// centroid at all, and at 10M x 768d the blob is ~30 MB against ~4 MB of new
// row data. Splitting it makes the per-save manifest O(partitions + N/8)
// instead of O(nlist * dim).
//
// v2 adds `file_bytes` per partition to the manifest. A tier merge
// appends the merged chunk and abandons the originals in place, so the
// physical end of file no longer follows from the chunk table and has to
// be recorded. No v1 reader can locate the append point in a v2 file, so
// this is a breaking bump rather than an optional trailer.
const FORMAT_VERSION: u8 = 3;
const FLAG_HAS_VECTORS: u8 = 1;
const FLAG_CLUSTERED: u8 = 2;

const MANIFEST_FILE: &str = "manifest";
const MANIFEST_TEMP_FILE: &str = "manifest.tmp";
/// Centroids live beside the manifest, not inside it, and are rewritten only
/// when they change. Named by generation so a reader's manifest always points
/// at the blob it was written against.
const CENTROIDS_PREFIX: &str = "centroids-";
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

#[derive(Clone)]
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
    /// Physical end of the segment file, which is NOT the end of the last live
    /// chunk once a tier merge has run.
    ///
    /// A merge folds trailing chunks into one by APPENDING the merged chunk
    /// and dropping the originals from the table. The originals' bytes have to
    /// stay where they are: a reader holding an older snapshot still has them
    /// mapped and still addresses rows by those offsets, so truncating them
    /// away would hand it garbage — or SIGBUS. Appending is safe for exactly
    /// the reason ingest appends are: nothing a live snapshot references ever
    /// moves.
    ///
    /// So the file grows past its live content, and the append offset must
    /// come from here rather than from the chunk table. `garbage_bytes` is the
    /// difference, and a full compaction (new generation) reclaims it.
    file_bytes: u64,
}

impl PartitionState {
    fn is_dead(&self, row: u64) -> bool {
        self.dead[(row / 8) as usize] & (1 << (row % 8)) != 0
    }

    /// Dead rows in `[base, base + n)`. Popcount over the bitmap, masking the
    /// partial bytes at each end so rows outside the range never count.
    fn dead_in(&self, base: u64, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        let end = base + n as u64;
        let (first, last) = ((base / 8) as usize, ((end - 1) / 8) as usize);
        if first >= self.dead.len() {
            return 0;
        }
        let last = last.min(self.dead.len() - 1);
        let mut total = 0u32;
        for (i, &byte) in self.dead[first..=last].iter().enumerate() {
            let mut b = byte;
            if first + i == first {
                b &= 0xFFu8 << (base % 8);
            }
            if first + i == last && end % 8 != 0 {
                b &= (1u8 << (end % 8)) - 1;
            }
            total += b.count_ones();
        }
        total as usize
    }

    /// Bytes of this segment file that the CURRENT chunk table actually
    /// references. Below `file_bytes` whenever a tier merge has abandoned
    /// chunks in place.
    fn live_bytes(&self, bit_width: usize, dim: usize, has_vectors: bool) -> u64 {
        self.chunks
            .iter()
            .map(|c| chunk_layout(bit_width, dim, c.n_rows as usize, has_vectors).total_len as u64)
            .sum()
    }

    /// Bytes in the file no live chunk points at — the cost of merging by
    /// appending rather than rewriting. Reclaimed by a full compaction.
    fn garbage_bytes(&self, bit_width: usize, dim: usize, has_vectors: bool) -> u64 {
        self.file_bytes
            .saturating_sub(self.live_bytes(bit_width, dim, has_vectors))
    }
}

/// Which store a removal touched. Only a partition dead-mark changes state
/// the snapshot owns, so only that one has to publish.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Removed {
    No,
    Memtable,
    Partitions,
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
    /// Size every backing vector for `n` rows up front. The append path used
    /// to grow them one row at a time, reallocating repeatedly.
    fn reserve(&mut self, n: usize, n_byte_groups: usize, dim: usize, vectors: bool) {
        self.group_rows.reserve(n * n_byte_groups);
        self.scales.reserve(n);
        self.ids.reserve(n);
        self.replica.reserve(n);
        if vectors {
            self.vectors.reserve(n * dim);
        }
    }

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
/// Maintenance knobs that SPFresh/SPANN fix by fiat and this index has so
/// far hard-coded. Exposed so the choices can be measured on the shipped
/// implementation instead of on a reimplementation of it. Defaults reproduce
/// the pre-existing behaviour exactly, so no stored index changes meaning.
#[derive(Clone, Copy, Debug)]
pub struct MaintenanceTuning {
    /// LIRE's `r_c`: how many neighbouring partitions around a changed one
    /// get their members re-tested. SPFresh's tuned default is 64.
    pub reassign_neighbors: usize,
    /// Balance the two children of a split so neither is much larger, as
    /// SPANN's multi-constraint balanced clustering does. When false a plain
    /// 2-means split is accepted whatever the resulting size ratio.
    pub balanced_split: bool,
    /// Re-run the split pass after reassignment. Reassignment can push a
    /// partition over the size bound, and without this the violation stands
    /// until the next flush.
    pub resplit_after_reassign: bool,
    /// Apply SPANN's RNG pruning rule when choosing replica targets. When
    /// false every partition inside the `(1 + epsilon)` bound is kept.
    pub replica_prune: bool,
    /// Split partitions that exceed `SPLIT_FACTOR * target`. Disabling this
    /// lets postings grow without bound, which is what plain IVF does.
    pub split_enabled: bool,
    /// Dissolve partitions that fall below `target / MERGE_DIVISOR`.
    pub dissolve_enabled: bool,
    /// Re-test the assignment of rows around a changed partition. This is
    /// the expensive half of LIRE and the part plain IVF omits.
    pub reassign_enabled: bool,
    /// Allow `rebootstrap_if_due` to globally re-cluster when a fresh k-means
    /// beats current distortion. This is a FULL REBUILD: it also rewrites every
    /// partition into one chunk and drops all dead rows. Any arm that gets one
    /// is therefore compared on a freshly compacted layout, which silently
    /// dominates a maintenance-policy comparison.
    pub rebootstrap_enabled: bool,
    /// Compact a partition once it holds more than this many chunks. The
    /// shipped 16 is far above what our workload reaches (~5), so GC almost
    /// never fires and every incremental write accumulates -- which is what
    /// makes reassignment look expensive on the latency axis.
    pub gc_max_chunks: usize,
    /// Hold arrivals in the memtable across saves until it reaches this many
    /// rows, then distribute them to partitions in one pass. 0 drains every
    /// save (the original behaviour).
    ///
    /// This is the staging tier, and it needs no new storage: the memtable is
    /// already durable (every `add` appends to the WAL and syncs, and the log
    /// replays on open) and already scanned by every query. Staging is
    /// therefore a gate on the DRAIN, not a new subsystem.
    ///
    /// It exists because distribution cost is dominated by how many rows each
    /// partition receives at once. A 10k batch over 810 partitions writes
    /// ~12-row chunks, and a chunk pads its codes to a whole 32-row block, so
    /// most of what is written is padding. Measured over +200k rows at
    /// deep 1.6M, raising the distribution size moves segment amplification
    /// 21.1x (5k) -> 11.1x (10k) -> 4.8x (50k) -> 3.4x (200k), with the knee
    /// around 50-100k.
    ///
    /// The cost is query-side: a staged row is found by scanning the memtable,
    /// which at 768d runs at ~0.64 ms per 50k rows, so a 100k staging tier
    /// adds ~1.3 ms to every query. Bound it accordingly.
    pub staging_threshold: usize,
    /// Route ingest assignment through a two-level coarse quantizer instead of
    /// scanning every centroid (`kmeans::CoarseIndex`).
    ///
    /// Assignment is O(nlist·dim) per row and nlist grows with N, so bulk
    /// construction is O(N²) — measured N^2.03, and off the field at scale
    /// (~670 h projected at 100M against DiskANN's ~1.4 h for 1B). The
    /// hierarchy makes it ~O(√nlist·dim), which is what every billion-scale
    /// system does.
    ///
    /// OFF by default because it makes assignment APPROXIMATE: a row can land
    /// in a partition that is not its true nearest. That is a recall question,
    /// not a correctness one — every row is still findable, and reassignment
    /// corrects drift — but it must be measured per corpus before it is
    /// trusted, and the flat path stays the reference.
    pub hierarchical_assign: bool,
    /// Fold trailing small chunks by appending a merged chunk instead of
    /// rewriting the partition (`tier_merge`). Off reproduces the pre-fix
    /// compaction behaviour, which is the only reason the flag exists.
    pub tier_merge_enabled: bool,
    /// Compact a partition once this fraction of its file is bytes no live
    /// chunk points at. Tier merges abandon their inputs in place, so without
    /// this the file would grow without bound.
    pub gc_garbage_ratio: f64,
    /// Compact a partition once this fraction of its rows are dead.
    pub gc_dead_ratio: f64,
    /// Keep maintenance OFF the save path entirely.
    ///
    /// `save()` then only makes data durable (~0.06 s for a 10k insert against
    /// ~0.58 s with maintenance), and the caller drives repair by calling
    /// `maintain()` when it suits -- idle, backgrounded, charging. Combined
    /// with the caps, each `maintain()` call is a bounded unit of work, so a
    /// concurrent reader is blocked for one bounded chunk rather than for
    /// however long a full pass happens to take.
    ///
    /// The size bound is NOT self-maintaining in this mode: with maintenance
    /// never run, max posting reached 5,962 against a 2,048 limit. Callers
    /// must actually call `maintain()`.
    pub defer_maintenance: bool,
    /// Cap on how many partitions one flush may REASSIGN (0 = unlimited).
    ///
    /// Capping splits bounds the tail but not the median: even a handful of
    /// splits drags `r_c` neighbours each into the repair pool, so a 10k
    /// insert can examine 100k rows. This caps the pool itself. Repairs that
    /// do not fit are not lost -- those partitions are revisited the next time
    /// they change -- but the nearest-partition invariant is then approximate
    /// for longer, which costs recall (measured worth +0.8 to +1.8 pp).
    pub max_reassign_partitions: usize,
    /// Cap on how many partitions one flush may rewrite (0 = unlimited).
    ///
    /// Segments are immutable and generation-stamped: a rewritten partition
    /// becomes a NEW file and the old one is unlinked only after the manifest
    /// rename publishes the replacement. That is what lets readers work
    /// without locks, but both generations coexist for the whole flush --
    /// measured peak on-disk is 2.4x steady state during a bulk ingest, which
    /// is a hard constraint on a phone.
    ///
    /// Capping rewrites per flush bounds the peak near
    /// `steady + cap * partition_bytes` instead of `2 * steady`. Deferred work
    /// is not skipped, it lands in the next flush, so the index still
    /// converges -- it just takes more saves.
    pub max_rewrites_per_flush: usize,
}

impl Default for MaintenanceTuning {
    fn default() -> Self {
        Self {
            reassign_neighbors: NEIGHBOR_PARTITIONS,
            balanced_split: false,
            // ON by default: reassignment moves vectors INTO partitions and can
            // push one back over the size bound, and nothing else catches it
            // before the flush ends. Measured at 1M: 0 over-limit partitions
            // against 1-2 per round without it, at no recall or maintenance cost.
            resplit_after_reassign: true,
            replica_prune: true,
            split_enabled: true,
            dissolve_enabled: true,
            reassign_enabled: true,
            rebootstrap_enabled: true,
            // 16 was far above what real workloads reach (~3.5 chunks per
            // partition), so GC essentially never fired and every incremental
            // write accumulated. At 1M, 4 cuts chunks 2256 -> 664 and dead rows
            // 13812 -> 397 for +5% maintenance time, and it is what lets
            // reassignment's quality gain reach the latency axis at all.
            gc_max_chunks: 4,
            staging_threshold: 0,
            hierarchical_assign: false,
            tier_merge_enabled: true,
            gc_garbage_ratio: 0.5,
            max_rewrites_per_flush: 0,
            max_reassign_partitions: 0,
            defer_maintenance: false,
            gc_dead_ratio: GC_DEAD_RATIO,
        }
    }
}

/// Counters for what maintenance actually DID, as opposed to what it was
/// configured to do. A knob that changes no outcome is indistinguishable from
/// a knob whose work never happens, and only these separate the two: if
/// reassignment moves no rows then `reassign_neighbors` cannot matter, and if
/// the balance pass never fires then `balanced_split` cannot either.
/// Cumulative over the life of the index.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaintenanceStats {
    pub splits_attempted: u64,
    pub splits_done: u64,
    pub splits_degenerate: u64,
    /// Splits where the balance pass actually moved at least one row.
    pub balance_fired: u64,
    /// Summed size ratio (larger child / smaller child) as 2-means produced
    /// it, before any balancing. Divide by `splits_attempted` for the mean.
    pub split_ratio_raw_sum: f64,
    /// Same ratio after the balance pass (equal to raw when balancing is off).
    pub split_ratio_final_sum: f64,
    pub dissolves: u64,
    pub reassign_passes: u64,
    /// Partitions pulled into the repair pool (the r_c neighbourhood).
    pub reassign_partitions: u64,
    /// Rows whose assignment was re-tested.
    pub reassign_rows_examined: u64,
    /// Rows that actually moved to a different partition.
    pub reassign_rows_moved: u64,
    pub resplit_passes: u64,
    pub rebootstraps: u64,
    pub compactions: u64,
    /// Wall microseconds inside each phase of the last flush. Guessing which
    /// phase dominates has been wrong twice (per-file overhead, then routing),
    /// so the split is measured.
    pub us_drain: u64,
    pub us_assign: u64,
    pub us_append: u64,
    pub us_maintenance: u64,
    pub us_publish: u64,
    /// Within maintenance: read-only PLANNING (decide what moves) versus
    /// mutating APPLY (write it). Only apply needs exclusive access, so this
    /// split decides whether a short exclusive window is worth building.
    pub us_plan: u64,
    pub us_apply: u64,
    /// Microseconds the last `add` held the memtable write lock, and the last
    /// memtable scan held the read lock. A concurrent query waits on these
    /// two and nothing else once reads are lock-free, so guessing which one
    /// dominates -- and I guessed the encode, wrongly -- is not good enough.
    pub us_add_lock: u64,
    pub us_memtable_scan: u64,
    /// Cumulative microseconds inside `decode_group_rows` and inside the
    /// assignment GEMM, summed over every thread. Maintenance is dominated by
    /// these two and the arithmetic did not settle which — measuring did.
    pub us_decode: u64,
    pub us_kmeans_gemm: u64,
    /// Times the id-run tables were merged down to one, and the wall
    /// microseconds that took. A merge rewrites every run file, so it is a
    /// periodic spike in the publish phase independent of maintenance.
    pub run_merges: u64,
    pub us_run_merge: u64,
    /// Segment bytes written, split by WHY they were written. Total write
    /// amplification measured 17.7x at 400k and 36.8x at 1.6M with segments
    /// 66-95% of it, but "segment" is four different operations and the fix
    /// differs per operation, so the bytes are attributed at the call site
    /// rather than inferred from which counters moved.
    ///
    /// `bytes_ingest` is the only class that carries new information. Every
    /// other class is a rewrite of rows already on disk.
    pub bytes_ingest: u64,
    pub bytes_compact: u64,
    pub bytes_split: u64,
    pub bytes_replica: u64,
    pub bytes_import: u64,
    pub bytes_tier: u64,
    /// Rows written per class, so the per-row overhead of a chunk (block
    /// padding to 32 rows, a 64-byte header, five section alignments) is
    /// separable from the number of rows moved.
    pub rows_ingest: u64,
    pub rows_compact: u64,
    pub rows_split: u64,
    pub rows_replica: u64,
    pub rows_tier: u64,
    /// Chunks appended, by class. `bytes_ingest / rows_ingest` versus the
    /// ideal row size is the padding tax; `rows_ingest / chunks_ingest` is
    /// the mean chunk size that causes it.
    pub chunks_ingest: u64,
    pub chunks_compact: u64,
    pub chunks_split: u64,
    pub chunks_replica: u64,
    pub chunks_tier: u64,
    /// Times `tier_merge` folded a run of trailing chunks.
    pub tier_merges: u64,
    /// Saves that published without draining the memtable, because staging
    /// had not filled. The rows stayed in the WAL and the memtable.
    pub staged_saves: u64,
    /// The ingest assign phase, split. `us_kmeans_gemm` and `us_decode` are
    /// GLOBAL counters that maintenance also increments, so attributing them
    /// to ingest overstates it -- these two are written only from the ingest
    /// assign path and are per-flush gauges like the rest.
    pub us_assign_decode: u64,
    pub us_assign_gemm: u64,
    /// The SHAPE the ingest assign GEMM actually ran at. Inferring it from the
    /// batch size the caller passed is how a 5x discrepancy went unnoticed.
    pub assign_rows: u64,
    pub assign_nlist: u64,
    /// Times the two-level coarse quantizer was rebuilt.
    pub coarse_rebuilds: u64,
}

/// Why a chunk is being written. Attribution only -- it does not change what
/// is written.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum WriteReason {
    /// New rows arriving from the memtable. The only class that adds
    /// information; everything else re-writes rows already on disk.
    Ingest,
    /// A partition rewritten whole by `collect_garbage`.
    Compact,
    /// A child partition materialized by a split or a dissolve.
    Split,
    /// Replica rows refreshed after the partitioning moved.
    Replica,
    /// Bulk import at load time; not part of steady-state amplification.
    Import,
    /// Trailing small chunks folded into one by `tier_merge`.
    TierMerge,
}

/// Chunk-count ceiling that still forces a full rewrite when tier merging is
/// on. Merging holds the count at O(log rows added) -- around 4-6 in practice
/// -- so this only fires if merging is somehow not keeping up, and exists so a
/// bug there degrades to the old behaviour instead of unbounded chunk growth.
const TIER_MERGE_CHUNK_BACKSTOP: usize = 24;

/// Run-count ceiling that still forces an all-runs merge. Tiering holds the
/// count at O(log entries); this only fires if it is not keeping up.
const RUN_TIER_BACKSTOP: usize = 24;

/// Below this many partitions a flat scan is already cheap and the hierarchy's
/// own build cost dominates, so it stays off.
const HIERARCHY_MIN_NLIST: usize = 256;

/// Supers probed per row in the two-level assignment. The accuracy dial:
/// agreement with exact assignment is ~33% at 1, ~85% at 8 (nlist 1024).
/// Cost is `(1 + probe)·√nlist` comparisons against `nlist` flat, so the
/// win still grows with nlist — ~9x at 6,300 partitions, ~35x at 97,656.
const HIERARCHY_PROBE_SUPER: usize = 8;

/// Largest tolerated size ratio between the two children of a split.
const SPLIT_BALANCE_RATIO: f32 = 1.5;
/// Cap on rebalancing rounds; each round halves the imbalance.
const SPLIT_BALANCE_ROUNDS: usize = 4;

/// Even out a 2-means split by moving the least-committed members of the
/// larger child to the smaller one, until the sizes are within
/// [`SPLIT_BALANCE_RATIO`].
///
/// This approximates SPANN's multi-constraint balanced clustering, which
/// solves a size-constrained assignment rather than post-hoc repairing an
/// unconstrained one. The cheap version is enough to answer whether balance
/// matters at all here; if it does, the constrained solve is the next step.
///
/// "Least committed" is by margin `d^2(x, c_small) - d^2(x, c_big)`: the
/// members that gain least from staying on the large side move first.
fn balance_split(
    vectors: &[f32],
    n: usize,
    dim: usize,
    child_centroids: &[f32],
    assignments: &mut [u32],
) {
    let sq =
        |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(&x, &y)| (x - y) * (x - y)).sum() };
    for _ in 0..SPLIT_BALANCE_ROUNDS {
        let ones = assignments.iter().filter(|&&c| c == 1).count();
        let zeros = n - ones;
        if zeros.min(ones) > 0
            && zeros.max(ones) as f32 <= SPLIT_BALANCE_RATIO * zeros.min(ones) as f32
        {
            return;
        }
        let big: u32 = if zeros > ones { 0 } else { 1 };
        let small = 1 - big;
        let big_center = &child_centroids[big as usize * dim..(big as usize + 1) * dim];
        let small_center = &child_centroids[small as usize * dim..(small as usize + 1) * dim];
        let mut ranked: Vec<(f32, usize)> = (0..n)
            .filter(|&i| assignments[i] == big)
            .map(|i| {
                let row = &vectors[i * dim..(i + 1) * dim];
                (sq(row, small_center) - sq(row, big_center), i)
            })
            .collect();
        if ranked.len() < 2 {
            return;
        }
        ranked.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let n_move = (zeros.abs_diff(ones) / 2).max(1).min(ranked.len() - 1);
        for &(_, i) in ranked.iter().take(n_move) {
            assignments[i] = small;
        }
    }
}

/// Mmap and codebook caches, shared by every snapshot of one index.
///
/// Segment maps are keyed by `(partition_id, generation)` rather than by
/// `partition_id` alone: once readers run without the writer's lock, a reader
/// holding an older snapshot may ask for a generation the writer has already
/// superseded, and a partition-only key would hand it the wrong file's bytes.
#[derive(Default)]
struct Caches {
    segment_maps: Mutex<HashMap<(u32, u32), Arc<Mmap>>>,
    run_maps: Mutex<HashMap<u64, Arc<Mmap>>>,
    rotation: OnceLock<rotation::Rotation>,
    codebook: OnceLock<Vec<f32>>,
}

/// The in-RAM write buffer: unflushed rows plus, when the index stores
/// vectors, their originals.
///
/// Shared live between the writer and every snapshot, rather than copied into
/// each one. Copying it per publish made a memtable-resident `remove` cost
/// O(unflushed rows) -- 0.32 ms at 200k against 0.00 ms before the snapshot
/// rewrite -- and deleting k unflushed rows quadratic.
///
/// Sharing it is safe because a flush does not CLEAR this cell: it builds a
/// fresh one and publishes it with the new partitions, atomically. A reader on
/// the older snapshot keeps the old cell, which still holds every row that
/// flush moved into partitions; a reader on the newer snapshot gets the empty
/// cell and the partitions that now hold them. Neither can see a row in
/// neither place, which is what clearing in place would allow.
#[derive(Default)]
struct MemtableCell {
    index: Option<IdMapIndex>,
    originals: HashMap<u64, Box<[f32]>>,
}

impl MemtableCell {
    fn new(index: IdMapIndex) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self {
            index: Some(index),
            originals: HashMap::new(),
        }))
    }

    fn get(&self) -> &IdMapIndex {
        self.index
            .as_ref()
            .expect("memtable cell is always populated")
    }

    fn get_mut(&mut self) -> &mut IdMapIndex {
        self.index
            .as_mut()
            .expect("memtable cell is always populated")
    }
}

/// Everything a query reads, and nothing a query does not.
///
/// The writer keeps one of these as its working copy and publishes cheap
/// clones of it as immutable [`Snapshot`]s. A clone costs `nlist` pointer
/// copies plus a handful of `Arc` bumps — the per-partition state, the
/// centroids, the run list and the memtable all sit behind `Arc`, so
/// publishing copies no vector data, and a writer that touches one partition
/// pays to copy only that partition (`Arc::make_mut`). That is the
/// per-partition granularity SPFresh gets from per-posting locks, without the
/// locks: a reader never blocks and never has to be woken.
#[derive(Clone)]
struct IndexState {
    directory: Option<PathBuf>,
    bit_width: usize,
    store_vectors: bool,
    clustered: bool,
    tqplus_shift: Vec<f32>,
    tqplus_scale: Vec<f32>,
    centroids: Arc<Vec<f32>>,
    partitions: Vec<Arc<PartitionState>>,
    runs: Arc<Vec<u64>>,
    memtable: Arc<RwLock<MemtableCell>>,
    caches: Arc<Caches>,
    /// High-water mark of one memtable scan, in microseconds. Shared so a
    /// reader thread's cost is visible to whoever asks for the stats.
    memtable_scan_us: Arc<AtomicU64>,
    decode_us: Arc<AtomicU64>,
}

/// Deferred unlink of segment and run files that a publish made unreachable.
///
/// A reader holding an older snapshot may still name a file the writer has
/// replaced, and may not have mapped it yet — so unlinking at publish time can
/// make a scan silently return nothing for that partition. Deletion therefore
/// waits until no snapshot that could name the file is still alive.
///
/// "Could name it" is a range, not a single snapshot: a partition untouched
/// for several flushes is referenced by every snapshot in between. Retiring a
/// file against the snapshot that last referenced it is not enough, because an
/// even older snapshot held by a long-running query can outlive that one. So
/// each file is queued with the sequence number of the last snapshot to
/// reference it, and is unlinked once the OLDEST live snapshot is newer.
#[derive(Default)]
struct Retirement {
    inner: Mutex<RetirementState>,
}

#[derive(Default)]
struct RetirementState {
    /// Sequence numbers of every snapshot currently alive.
    live: BTreeSet<u64>,
    /// `(last sequence that referenced this file, path)`.
    pending: Vec<(u64, PathBuf)>,
}

impl Retirement {
    fn register(&self, seq: u64) {
        self.inner.lock().expect("retire lock").live.insert(seq);
    }

    fn release(&self, seq: u64) {
        let mut state = self.inner.lock().expect("retire lock");
        state.live.remove(&seq);
        Self::sweep(&mut state);
    }

    fn retire(&self, seq: u64, paths: Vec<PathBuf>) {
        if paths.is_empty() {
            return;
        }
        let mut state = self.inner.lock().expect("retire lock");
        state.pending.extend(paths.into_iter().map(|p| (seq, p)));
        Self::sweep(&mut state);
    }

    fn sweep(state: &mut RetirementState) {
        let oldest = state.live.iter().next().copied().unwrap_or(u64::MAX);
        state.pending.retain(|(seq, path)| {
            if *seq < oldest {
                fs::remove_file(path).ok();
                false
            } else {
                true
            }
        });
    }

    /// Files queued but not yet unlinked. Diagnostic: this is the transient
    /// disk a slow reader is holding open.
    fn pending_len(&self) -> usize {
        self.inner.lock().expect("retire lock").pending.len()
    }
}

/// A published, immutable view of the index.
///
/// Readers hold one for the duration of a query; the writer holds the newest.
/// Dropping the last reference is what tells [`Retirement`] that the files
/// this snapshot pinned can go — the reader-side half of the copy-on-write
/// story that already governs the on-disk layout.
pub struct Snapshot {
    seq: u64,
    state: IndexState,
    retirement: Arc<Retirement>,
}

impl Snapshot {
    /// Query this exact snapshot.
    ///
    /// A `FreshReader` picks up the newest snapshot per call, which is what a
    /// single query wants. Holding one and querying it repeatedly pins the
    /// PARTITION state: maintenance, splits and compaction cannot change what
    /// these queries see on disk, and every segment named stays readable.
    ///
    /// It does NOT pin unflushed rows. The memtable cell is shared live, so
    /// adds and removes that have not been flushed yet do show through. A
    /// flush is the boundary: it swaps in a fresh cell alongside the new
    /// partitions, so a pinned snapshot keeps the cell it had, holding
    /// exactly the rows the flush moved into partitions it cannot see.
    pub fn search(&self, queries: &[f32], k: usize) -> (Vec<f32>, Vec<u64>) {
        self.state.search(queries, k)
    }

    pub fn search_with_options(
        &self,
        queries: &[f32],
        k: usize,
        options: SearchOptions,
    ) -> (Vec<f32>, Vec<u64>) {
        self.state.search_with_options(queries, k, options)
    }

    pub fn contains(&self, id: u64) -> bool {
        self.state.contains(id)
    }

    pub fn len(&self) -> usize {
        self.state.len()
    }

    pub fn is_empty(&self) -> bool {
        self.state.len() == 0
    }

    pub fn nlist(&self) -> usize {
        self.state.nlist()
    }

    fn new(seq: u64, state: IndexState, retirement: Arc<Retirement>) -> Arc<Self> {
        retirement.register(seq);
        Arc::new(Self {
            seq,
            state,
            retirement,
        })
    }
}

impl Drop for Snapshot {
    fn drop(&mut self) {
        self.retirement.release(self.seq);
    }
}

/// A lock-free read handle onto a [`FreshIndex`].
///
/// Holds the publication cell rather than the index, so a query never
/// contends with `save`/`maintain` for the writer's lock: it loads the current
/// snapshot (one uncontended read-lock acquire around an `Arc` clone) and
/// scans that. Maintenance running concurrently builds a new snapshot and
/// swaps it in; queries already in flight finish against the old one.
#[derive(Clone)]
pub struct FreshReader {
    published: Arc<RwLock<Arc<Snapshot>>>,
}

impl IndexState {
    pub fn stores_vectors(&self) -> bool {
        self.store_vectors
    }

    /// Read guard on the shared memtable cell. Held only for the length of
    /// one read -- never across a partition scan.
    fn mem(&self) -> std::sync::RwLockReadGuard<'_, MemtableCell> {
        self.memtable.read().expect("memtable lock poisoned")
    }

    pub fn dim_opt(&self) -> Option<usize> {
        self.mem().get().dim_opt()
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
            + self.mem().get().len()
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
        self.mem().get().len()
    }

    /// Dead (removed or moved-away) rows awaiting compaction. Diagnostic.
    pub fn dead_count(&self) -> usize {
        self.partitions
            .iter()
            .map(|p| (p.n_rows - p.live_rows) as usize)
            .sum()
    }

    /// Live primary rows per partition. The MEAN of this is uninformative —
    /// it is pinned near the target by construction — so the statistics that
    /// actually show whether splitting keeps postings balanced are the max
    /// and the upper percentiles. Diagnostic.
    pub fn partition_sizes(&self) -> Vec<usize> {
        self.partitions
            .iter()
            .map(|p| p.live_primary as usize)
            .collect()
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
        self.mem().get().prepare();
    }

    /// True if a vector with this external id is live.
    pub fn contains(&self, id: u64) -> bool {
        self.mem().get().contains(id) || !self.live_copies(id).is_empty()
    }

    /// The stored full-precision vector of a live id. Panics unless the
    /// index stores vectors.
    pub fn get_vector(&self, id: u64) -> Option<Vec<f32>> {
        assert!(
            self.store_vectors,
            "get_vector requires an index built with store_vectors",
        );
        if let Some(vector) = self.mem().originals.get(&id) {
            return Some(vector.to_vec());
        }
        let copies = self.live_copies(id);
        let copy = copies.first()?;
        Some(self.copy_vector(*copy))
    }

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
            None if self.store_vectors => Some((DEFAULT_RESCORE_MULTIPLIER * k_eff).max(k_eff)),
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
                    // Get every read in flight before touching the first one.
                    self.prefetch_partitions(&routes[qi]);
                    let mut candidates = Vec::new();
                    for &position in &routes[qi] {
                        self.scan_partition(position as usize, &single, fetch_k, &mut candidates);
                    }
                    candidates
                })
                .collect()
        } else {
            vec![Vec::new(); nq]
        };

        let t_mem = std::time::Instant::now();
        let (memtable_scores, memtable_ids) = {
            let mem = self.mem();
            if mem.get().is_empty() {
                (Vec::new(), Vec::new())
            } else {
                mem.get().search(queries, fetch_k)
            }
        };
        self.memtable_scan_us
            .fetch_max(t_mem.elapsed().as_micros() as u64, Ordering::Relaxed);
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
        if let Some(vector) = self.mem().originals.get(&id) {
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
    /// Advise the kernel about every partition this query will read, BEFORE
    /// reading any of them.
    ///
    /// Without this the scan walks partitions in order and faults each one's
    /// pages on demand, so a query whose pages are not resident pays one full
    /// I/O round trip per partition at queue depth 1. The same defect was
    /// measured on the flat disk index at a p95 of 622 ms under memory
    /// pressure; advising the whole probed set first lets the device see all
    /// of the reads at once and overlap them.
    ///
    /// This is the cold-latency lever for IVF specifically: unlike a graph's
    /// hops, probed partitions are INDEPENDENT, so their reads have no
    /// dependency chain and can all be in flight together.
    ///
    /// Advisory only — `MADV_WILLNEED` cannot fail the query, only fail to
    /// help, so errors are deliberately discarded.
    fn prefetch_partitions(&self, positions: &[u32]) {
        if positions.len() < 2 {
            return; // one partition has nothing to overlap with
        }
        for &position in positions {
            let Some(partition) = self.partitions.get(position as usize) else {
                continue;
            };
            if partition.live_rows == 0 {
                continue;
            }
            let Ok(map) = self.segment_map(partition.partition_id, partition.generation) else {
                continue;
            };
            let len = (partition.file_bytes as usize).min(map.len());
            if len > 0 {
                let _ = map.advise_range(Advice::WillNeed, 0, len);
            }
        }
    }

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
        let mut base_row = 0u64;
        for chunk in &partition.chunks {
            let n_rows = chunk.n_rows as usize;
            let layout = chunk_layout(self.bit_width, dim, n_rows, self.store_vectors);
            let start = chunk.offset as usize;
            let codes = &map[start + layout.codes.0..start + layout.codes.0 + layout.codes.1];
            let scales =
                f32_slice(&map[start + layout.scales.0..start + layout.scales.0 + layout.scales.1]);
            let ids = u64_slice(&map[start + layout.ids.0..start + layout.ids.0 + layout.ids.1]);
            // Over-fetch to cover rows this scan will discard as dead -- but
            // only by THIS chunk's dead count, not the partition's. Charging
            // every chunk the whole partition's dead rows inflates k on the
            // large base chunk, and the candidates it returns all have to be
            // merged afterwards: with tier merging the partition accumulates
            // dead rows between compactions, which turned k=10 into k=34 on
            // every chunk and multiplied the merge set ~13x at nprobe=80.
            // A chunk cannot contain more dead rows than it has rows.
            let dead_here = partition.dead_in(base_row, n_rows);
            let k_chunk = (fetch_k + dead_here).min(n_rows);
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

    fn partition_position(&self, partition_id: u32) -> Option<usize> {
        self.partitions
            .iter()
            .position(|p| p.partition_id == partition_id)
    }

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
                let codes = &map[start + layout.codes.0..start + layout.codes.0 + layout.codes.1];
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
                    &map[start + layout.scales.0..start + layout.scales.0 + layout.scales.1],
                );
                let vector = if self.store_vectors {
                    let vectors = f32_slice(
                        &map[start + layout.vectors.0..start + layout.vectors.0 + layout.vectors.1],
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

    /// Map a partition segment, caching by `(partition_id, generation)`.
    ///
    /// The generation MUST be part of the key. A reader holding an older
    /// snapshot legitimately asks for a generation the writer has already
    /// replaced; with a partition-only key it would be handed the newer
    /// file's bytes and read rows at offsets its snapshot does not describe.
    fn segment_map(&self, partition_id: u32, generation: u32) -> io::Result<Arc<Mmap>> {
        let key = (partition_id, generation);
        let mut cache = self.caches.segment_maps.lock().expect("segment map lock");
        if let Some(map) = cache.get(&key) {
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
        cache.insert(key, Arc::clone(&map));
        Ok(map)
    }

    /// Drop every cached mapping of a partition, whatever its generation.
    ///
    /// A reader mid-scan keeps its own `Arc<Mmap>` alive, and the file itself
    /// survives until the snapshot naming it is dropped, so this only evicts
    /// the cache entry — it never pulls bytes out from under a query.
    fn invalidate_segment_map(&self, partition_id: u32) {
        self.caches
            .segment_maps
            .lock()
            .expect("segment map lock")
            .retain(|&(pid, _), _| pid != partition_id);
    }

    fn run_map(&self, generation: u64) -> io::Result<Arc<Mmap>> {
        let mut cache = self.caches.run_maps.lock().expect("run map lock");
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

    fn rotation_for(&self, dim: usize) -> &rotation::Rotation {
        self.caches
            .rotation
            .get_or_init(|| rotation::Rotation::new(dim))
    }

    fn codebook_for(&self, dim: usize) -> &[f32] {
        self.caches.codebook.get_or_init(|| {
            let (_, centroids) = codebook::codebook(self.bit_width, dim);
            centroids
        })
    }
}

pub struct FreshIndex {
    /// The writer's working copy. Published as an immutable [`Snapshot`] at
    /// every commit point.
    state: IndexState,
    published: Arc<RwLock<Arc<Snapshot>>>,
    retirement: Arc<Retirement>,
    next_snapshot_seq: u64,
    replica_epsilon: Option<f32>,
    tuning: MaintenanceTuning,
    stats: MaintenanceStats,
    partition_target: Option<usize>,
    epoch: u64,
    /// Generation of the on-disk centroid blob the manifest references, and
    /// whether the in-memory centroids have diverged from it since.
    centroids_generation: u64,
    centroids_dirty: bool,
    /// Previous centroid blob, unlinked once the manifest naming its
    /// replacement has been renamed into place.
    retired_centroids: Option<PathBuf>,
    /// Two-level quantizer over the centroids, rebuilt when nlist has moved
    /// materially rather than on every centroid touch — it clusters a summary,
    /// so a slightly stale level costs a little accuracy, not correctness.
    coarse: Option<kmeans::CoarseIndex>,
    next_partition_id: u32,
    next_run_generation: u64,
    churn_since_check: u64,
    wal: Option<File>,
    /// dim recorded in the live WAL's header. A WAL created before the
    /// index committed a dim says 0 and must be re-headered before the
    /// first add record (it is empty at that point by construction —
    /// add records imply a committed dim, and no-op removes are not
    /// logged).
    wal_dim: usize,
    /// Segment files written during the current flush that still need an
    /// fsync. `append_chunk` used to fsync every chunk it wrote, which at 1M
    /// meant ~2,400 serialized write->fsync round-trips inside one save and
    /// left the process at 0.31 of one core while queries were blocked. Only
    /// the MANIFEST RENAME is the commit point, so segment data merely has to
    /// be durable BEFORE that rename -- one batched pass, same guarantee.
    pending_syncs: Vec<PathBuf>,
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
        Ok(Self::from_state(IndexState {
            directory: None,
            bit_width,
            store_vectors: false,
            clustered: false,
            tqplus_shift: Vec::new(),
            tqplus_scale: Vec::new(),
            centroids: Arc::new(Vec::new()),
            partitions: Vec::new(),
            runs: Arc::new(Vec::new()),
            memtable: MemtableCell::new(memtable),
            caches: Arc::new(Caches::default()),
            memtable_scan_us: Arc::new(AtomicU64::new(0)),
            decode_us: Arc::new(AtomicU64::new(0)),
        }))
    }

    fn from_state(state: IndexState) -> Self {
        let retirement = Arc::new(Retirement::default());
        let published = Arc::new(RwLock::new(Snapshot::new(
            0,
            state.clone(),
            Arc::clone(&retirement),
        )));
        Self {
            state,
            published,
            retirement,
            next_snapshot_seq: 1,
            replica_epsilon: None,
            tuning: MaintenanceTuning::default(),
            stats: MaintenanceStats::default(),
            partition_target: None,
            epoch: 0,
            centroids_generation: 0,
            centroids_dirty: true,
            retired_centroids: None,
            coarse: None,
            next_partition_id: 0,
            next_run_generation: 0,
            churn_since_check: 0,
            wal: None,
            wal_dim: 0,
            pending_syncs: Vec::new(),
        }
    }

    /// A lock-free read handle. Cloneable, cheap, and valid for the life of
    /// the index: it tracks the publication cell, not any one snapshot.
    pub fn reader(&self) -> FreshReader {
        FreshReader {
            published: Arc::clone(&self.published),
        }
    }

    /// Make the working state visible to readers and queue the files this
    /// publish orphaned for deletion.
    ///
    /// The swap itself is what a query can contend with, and it is a pointer
    /// move under a write lock — not the maintenance that produced the new
    /// state, which ran entirely on the writer's private copy. Cost is one
    /// `IndexState` clone (`nlist` pointer copies), cheap enough to call at
    /// the end of every mutating method.
    fn publish_with(&mut self, dropped_files: Vec<PathBuf>) {
        let seq = self.next_snapshot_seq;
        self.next_snapshot_seq += 1;
        let next = Snapshot::new(seq, self.state.clone(), Arc::clone(&self.retirement));
        let previous = {
            let mut slot = self.published.write().expect("snapshot lock");
            std::mem::replace(&mut *slot, next)
        };
        // `previous` is the last snapshot that could still name the dropped
        // files, so that is the sequence they are queued against. Dropping
        // our reference to it here may be what frees them — or a query still
        // holding it, or an even older one, finishes later and frees them
        // then.
        let previous_seq = previous.seq;
        drop(previous);
        self.retirement.retire(previous_seq, dropped_files);
    }

    fn mem_read(&self) -> std::sync::RwLockReadGuard<'_, MemtableCell> {
        self.state.memtable.read().expect("memtable lock poisoned")
    }

    fn mem_write(&self) -> std::sync::RwLockWriteGuard<'_, MemtableCell> {
        self.state.memtable.write().expect("memtable lock poisoned")
    }

    fn publish(&mut self) {
        self.publish_with(Vec::new());
    }

    /// Files unlinked-but-for a reader still holding an older snapshot.
    /// Diagnostic.
    pub fn retired_pending(&self) -> usize {
        self.retirement.pending_len()
    }

    // ------------------------------------------------------------------
    // Read path. Forwarded to the working state; a `FreshReader` runs the
    // very same code against a published snapshot instead.
    // ------------------------------------------------------------------

    pub fn stores_vectors(&self) -> bool {
        self.state.stores_vectors()
    }

    pub fn dim_opt(&self) -> Option<usize> {
        self.state.dim_opt()
    }

    pub fn dim(&self) -> usize {
        self.state.dim()
    }

    pub fn bit_width(&self) -> usize {
        self.state.bit_width()
    }

    pub fn len(&self) -> usize {
        self.state.len()
    }

    pub fn is_empty(&self) -> bool {
        self.state.is_empty()
    }

    pub fn nlist(&self) -> usize {
        self.state.nlist()
    }

    pub fn base_len(&self) -> usize {
        self.state.base_len()
    }

    pub fn memtable_len(&self) -> usize {
        self.state.memtable_len()
    }

    pub fn dead_count(&self) -> usize {
        self.state.dead_count()
    }

    pub fn partition_sizes(&self) -> Vec<usize> {
        self.state.partition_sizes()
    }

    pub fn replica_count(&self) -> usize {
        self.state.replica_count()
    }

    pub fn run_count(&self) -> usize {
        self.state.run_count()
    }

    pub fn chunk_count(&self) -> usize {
        self.state.chunk_count()
    }

    pub fn path(&self) -> Option<&Path> {
        self.state.path()
    }

    pub fn prepare(&self) {
        self.state.prepare()
    }

    pub fn contains(&self, id: u64) -> bool {
        self.state.contains(id)
    }

    pub fn get_vector(&self, id: u64) -> Option<Vec<f32>> {
        self.state.get_vector(id)
    }

    pub fn search(&self, queries: &[f32], k: usize) -> (Vec<f32>, Vec<u64>) {
        self.state.search(queries, k)
    }

    pub fn search_with_options(
        &self,
        queries: &[f32],
        k: usize,
        options: SearchOptions,
    ) -> (Vec<f32>, Vec<u64>) {
        self.state.search_with_options(queries, k, options)
    }

    fn exact_score(&self, id: u64, query: &[f32]) -> f32 {
        self.state.exact_score(id, query)
    }

    fn scan_partition(
        &self,
        position: usize,
        prepared: &crate::search::PreparedQueries,
        fetch_k: usize,
        out: &mut Vec<(f32, u64)>,
    ) {
        self.state.scan_partition(position, prepared, fetch_k, out)
    }

    fn partition_position(&self, partition_id: u32) -> Option<usize> {
        self.state.partition_position(partition_id)
    }

    fn live_copies(&self, id: u64) -> Vec<CopyLocation> {
        self.state.live_copies(id)
    }

    fn copy_vector(&self, copy: CopyLocation) -> Vec<f32> {
        self.state.copy_vector(copy)
    }

    fn copy_row_data(
        &self,
        copy: CopyLocation,
        dim: usize,
    ) -> io::Result<(Vec<u8>, f32, Option<Vec<f32>>)> {
        self.state.copy_row_data(copy, dim)
    }

    fn segment_map(&self, partition_id: u32, generation: u32) -> io::Result<Arc<Mmap>> {
        self.state.segment_map(partition_id, generation)
    }

    fn invalidate_segment_map(&self, partition_id: u32) {
        self.state.invalidate_segment_map(partition_id)
    }

    fn run_map(&self, generation: u64) -> io::Result<Arc<Mmap>> {
        self.state.run_map(generation)
    }

    fn rotation_for(&self, dim: usize) -> &rotation::Rotation {
        self.state.rotation_for(dim)
    }

    fn codebook_for(&self, dim: usize) -> &[f32] {
        self.state.codebook_for(dim)
    }

    /// Run one bounded unit of maintenance and publish the result.
    ///
    /// This is the other half of `defer_maintenance`: `save()` gets data
    /// durable fast, and this does the repair, in chunks the caller can
    /// schedule. Bounded by `max_rewrites_per_flush` and
    /// `max_reassign_partitions`, so the exclusive window is capped no matter
    /// how much work has accumulated.
    ///
    /// Returns true while work may remain -- i.e. a partition still exceeds
    /// the size bound -- so a caller can loop until it returns false.
    pub fn maintain(&mut self) -> io::Result<bool> {
        let Some(dim) = self.dim_opt() else {
            return Ok(false);
        };
        if self.state.directory.is_none() {
            return Ok(false);
        }
        let mut entries: Vec<RunEntry> = Vec::new();
        let mut dropped_files: Vec<PathBuf> = Vec::new();
        self.maintenance(dim, &mut entries, &mut dropped_files)?;
        if !entries.is_empty() {
            self.write_run(&entries)?;
        }
        if self.state.runs.len() > MAX_RUNS {
            self.tier_merge_runs(&mut dropped_files)?;
        }
        self.sync_pending()?;
        self.epoch += 1;
        self.write_manifest()?;
        self.publish_with(dropped_files);
        Ok(self.needs_maintenance())
    }

    /// True when some partition is outside the size bound, i.e. `maintain()`
    /// still has work to do.
    pub fn needs_maintenance(&self) -> bool {
        let Some(target) = self.partition_target else {
            return false;
        };
        self.state.partitions.iter().any(|p| {
            let over = p.live_primary as usize > SPLIT_FACTOR * target;
            let under = self.state.partitions.len() > MIN_PARTITIONS
                && (p.live_primary as usize) < target / MERGE_DIVISOR;
            over || under || p.chunks.len() > self.tuning.gc_max_chunks
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
        index.publish();
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
        if self.state.clustered && target_partition_size.is_none() {
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

    /// Override the maintenance knobs. Takes effect from the next flush;
    /// affects nothing already written.
    pub fn set_tuning(&mut self, tuning: MaintenanceTuning) {
        assert!(
            tuning.reassign_neighbors > 0,
            "reassign_neighbors must be positive, got {}",
            tuning.reassign_neighbors,
        );
        self.tuning = tuning;
    }

    pub fn tuning(&self) -> MaintenanceTuning {
        self.tuning
    }

    /// What maintenance actually did, cumulatively. Diagnostic.
    pub fn maintenance_stats(&self) -> MaintenanceStats {
        let mut stats = self.stats;
        stats.us_memtable_scan = self.state.memtable_scan_us.load(Ordering::Relaxed);
        stats.us_decode = self.state.decode_us.load(Ordering::Relaxed);
        stats.us_kmeans_gemm = crate::kmeans::assign_gemm_micros();
        stats
    }

    /// See [`crate::DiskIndex::set_store_vectors`]. Must be set while the
    /// index is empty.
    pub fn set_store_vectors(&mut self, store_vectors: bool) {
        if store_vectors == self.state.store_vectors {
            return;
        }
        assert!(
            self.state.partitions.is_empty() && self.state.memtable_len() == 0,
            "store_vectors must be set while the index is empty",
        );
        self.state.store_vectors = store_vectors;
        self.publish();
    }

    // ------------------------------------------------------------------
    // Introspection
    // ------------------------------------------------------------------

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
            if !self.mem_read().get().contains(id) && !self.live_copies(id).is_empty() {
                return Err(AddError::IdAlreadyPresent(id));
            }
        }
        let t_lock = std::time::Instant::now();
        self.mem_write()
            .get_mut()
            .add_with_ids_2d(vectors, dim, ids)?;
        self.stats.us_add_lock = t_lock.elapsed().as_micros() as u64;
        if self.wal.is_some() && self.wal_dim != dim {
            self.reset_wal().expect("write-ahead log reset failed");
        }
        for (i, &id) in ids.iter().enumerate() {
            let vector = &vectors[i * dim..(i + 1) * dim];
            if self.state.store_vectors {
                self.mem_write().originals.insert(id, vector.into());
            }
            self.wal_append(WAL_ADD, id, vector)
                .expect("write-ahead log append failed");
        }
        self.wal_sync().expect("write-ahead log sync failed");
        // No publish: rows land in the shared memtable cell, which the live
        // snapshot already names, so they are visible the moment they land.
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
    /// Remove one id. Durable when this returns, which costs one write-ahead
    /// log fsync -- so removing a batch this way costs one fsync per id.
    /// [`Self::remove_many`] is the same operation for a batch and is ~1300x
    /// faster; there is no reason to loop this.
    ///
    /// Delegates, so the single and batch paths cannot drift apart.
    pub fn remove(&mut self, id: u64) -> bool {
        self.remove_many(&[id]) == 1
    }

    /// Remove many ids, syncing the write-ahead log ONCE.
    ///
    /// `remove` fsyncs per call so that a mutation is durable when the call
    /// returns. Looping over it therefore costs one fsync per id -- measured
    /// at 2.6 ms per delete against 0.32 ms for the same work with no WAL, so
    /// 88% of a bulk delete is fsync. This keeps the same contract at the call
    /// boundary (everything durable on return, nothing partially applied) and
    /// pays for one fsync, exactly as a batched `add_with_ids` already does.
    ///
    /// Also publishes once rather than once per id.
    ///
    /// Returns how many ids were actually present.
    pub fn remove_many(&mut self, ids: &[u64]) -> usize {
        let mut removed = 0usize;
        let mut touched_partitions = false;
        for &id in ids {
            let outcome = self.remove_internal(id);
            if outcome != Removed::No {
                self.wal_append(WAL_REMOVE, id, &[])
                    .expect("write-ahead log append failed");
                removed += 1;
                touched_partitions |= outcome == Removed::Partitions;
            }
        }
        if removed > 0 {
            self.wal_sync().expect("write-ahead log sync failed");
            if touched_partitions {
                self.publish();
            }
        }
        removed
    }

    fn remove_internal(&mut self, id: u64) -> Removed {
        {
            let mut mem = self.mem_write();
            if mem.get().contains(id) {
                mem.get_mut().remove(id);
                mem.originals.remove(&id);
                // No publish: the memtable cell is shared with the live
                // snapshot, so this removal is already visible to readers.
                return Removed::Memtable;
            }
        }
        let copies = self.live_copies(id);
        if copies.is_empty() {
            return Removed::No;
        }
        for copy in copies {
            self.dead_mark(copy);
        }
        self.churn_since_check += 1;
        Removed::Partitions
    }

    // ------------------------------------------------------------------
    // Search
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // Persistence: save / flush
    // ------------------------------------------------------------------

    /// Bind to `directory` (first call) and flush: append the memtable to
    /// its partitions, run local maintenance, and atomically publish a new
    /// manifest. Untouched partitions' files are not rewritten — their
    /// page-cache contents stay valid across the save.
    pub fn save(&mut self, directory: impl AsRef<Path>) -> io::Result<()> {
        let directory = directory.as_ref();
        match &self.state.directory {
            None => {
                fs::create_dir_all(directory)?;
                if directory.join(MANIFEST_FILE).exists() {
                    return Err(invalid_data(format!(
                        "directory {} already contains a FreshIndex; open it instead",
                        directory.display(),
                    )));
                }
                self.state.directory = Some(directory.to_path_buf());
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
                self.publish();
                return Ok(());
            }
        };
        self.commit_calibration(dim);

        // Phase timers are GAUGES for this flush, and each is written only on
        // the path that runs it. A staging save skips the drain entirely, so
        // without this reset `us_assign`/`us_append` keep the last DRAINING
        // flush's values and a caller summing them across saves counts that
        // drain once per save. Zero them up front so "did not run" reads as 0.
        self.stats.us_drain = 0;
        self.stats.us_assign = 0;
        self.stats.us_append = 0;
        self.stats.us_maintenance = 0;
        self.stats.us_publish = 0;
        self.stats.us_assign_decode = 0;
        self.stats.us_assign_gemm = 0;
        self.stats.assign_rows = 0;
        self.stats.assign_nlist = 0;

        let mut entries: Vec<RunEntry> = Vec::new();
        let mut dropped_files: Vec<PathBuf> = Vec::new();

        // Drain the memtable into per-partition appends -- unless staging is
        // on and the memtable has not filled yet, in which case the rows stay
        // put and this save only publishes.
        //
        // Skipping the drain means the WAL must NOT be reset below: the rows
        // it holds are still the only durable copy. That coupling is the whole
        // correctness argument for staging, so both branches key off this one
        // flag rather than recomputing the condition.
        let staged = self.memtable_len();
        let draining = self.tuning.staging_threshold == 0 || staged >= self.tuning.staging_threshold;
        if !draining {
            self.stats.staged_saves += 1;
        }
        let t_drain = std::time::Instant::now();
        let batch = if draining {
            self.drain_memtable(dim)
        } else {
            RowBatch::default()
        };
        self.stats.us_drain = t_drain.elapsed().as_micros() as u64;
        if batch.n > 0 {
            if self.state.partitions.is_empty() {
                self.create_partition(&vec![0.0; dim], &mut entries)?;
            }
            let t_assign = std::time::Instant::now();
            if self.replica_epsilon.is_none() {
                let destinations = self.assign_batch_single(&batch, dim);
                self.stats.us_assign = t_assign.elapsed().as_micros() as u64;
                let t_append = std::time::Instant::now();
                self.append_batch_single(&batch, &destinations, dim, &mut entries)?;
                self.stats.us_append = t_append.elapsed().as_micros() as u64;
            } else {
                let destinations = self.assign_batch(&batch, dim);
                self.stats.us_assign = t_assign.elapsed().as_micros() as u64;
                let t_append = std::time::Instant::now();
                self.append_batch(&batch, &destinations, dim, &mut entries)?;
                self.stats.us_append = t_append.elapsed().as_micros() as u64;
            }
            self.churn_since_check += batch.n as u64;
        }

        let t_maint = std::time::Instant::now();
        if !self.tuning.defer_maintenance {
            self.maintenance(dim, &mut entries, &mut dropped_files)?;
        }
        self.stats.us_maintenance = t_maint.elapsed().as_micros() as u64;
        let t_publish = std::time::Instant::now();

        // Publish: run file, manifest, WAL reset, then cleanup.
        if !entries.is_empty() {
            self.write_run(&entries)?;
        }
        if self.state.runs.len() > MAX_RUNS {
            self.tier_merge_runs(&mut dropped_files)?;
        }
        self.sync_pending()?;
        // The epoch is the WAL generation counter (plus seed entropy for
        // splits): `replay_or_reset_wal` replays a log only when its recorded
        // epoch matches the manifest's, on the reasoning that a stale log's
        // records are already in the manifest state.
        //
        // A staging save breaks that reasoning -- its records are NOT in the
        // manifest -- so it must not advance the epoch. Bumping it here while
        // leaving the WAL alone makes the log look stale and silently discards
        // every staged row at the next open.
        //
        // Both orders are crash-safe with this in place. Crash during a
        // staging save: manifest and WAL both at E, so the log replays. Crash
        // during a draining save between the manifest and `reset_wal`:
        // manifest at E+1, WAL at E, so the log is discarded -- correctly,
        // because its rows are in the segments the manifest now names.
        if draining {
            self.epoch += 1;
        }
        self.write_manifest()?;
        if draining {
            self.reset_wal()?;
        }
        // A FRESH cell, not a cleared one: readers still on the previous
        // snapshot keep the old cell and every row it holds, while readers on
        // the new snapshot find those rows in the partitions instead.
        if draining {
            self.state.memtable = MemtableCell::new(self.fresh_memtable(dim)?);
        }
        // Only here does the flush become visible: readers ran the whole
        // drain/append/maintenance against the previous snapshot.
        self.publish_with(dropped_files);
        self.stats.us_publish = t_publish.elapsed().as_micros() as u64;
        Ok(())
    }

    fn commit_calibration(&mut self, dim: usize) {
        if !self.state.tqplus_shift.is_empty() {
            return;
        }
        let calibration = {
            let mem = self.mem_read();
            let shift = mem.get().inner().tqplus_shift();
            if shift.is_empty() {
                None
            } else {
                Some((shift.to_vec(), mem.get().inner().tqplus_scale().to_vec()))
            }
        };
        match calibration {
            None => {
                self.state.tqplus_shift = vec![0.0; dim];
                self.state.tqplus_scale = vec![1.0; dim];
            }
            Some((shift, scale)) => {
                self.state.tqplus_shift = shift;
                self.state.tqplus_scale = scale;
            }
        }
    }

    fn fresh_memtable(&self, dim: usize) -> io::Result<IdMapIndex> {
        let inner = TurboQuantIndex::from_parts(
            Some(dim),
            self.state.bit_width,
            0,
            Vec::new(),
            Vec::new(),
            self.state.tqplus_shift.clone(),
            self.state.tqplus_scale.clone(),
        );
        Ok(IdMapIndex::from_inner(inner.map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, e.to_string())
        })?))
    }

    fn drain_memtable(&self, dim: usize) -> RowBatch {
        let mem = self.mem_read();
        let n = mem.get().len();
        let mut batch = RowBatch::default();
        if n == 0 {
            return batch;
        }
        let n_byte_groups = pack::n_byte_groups(self.state.bit_width, dim);
        let group_rows = pack::group_bytes(
            mem.get().inner().packed_codes(),
            n,
            self.state.bit_width,
            dim,
        );
        let scales = mem.get().inner().scales();
        let ids = mem.get().slot_to_id_slice();
        for i in 0..n {
            let vector = if self.state.store_vectors {
                Some(
                    mem.originals
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

    /// Primary partition position per row, as a flat slice.
    ///
    /// The general [`Self::assign_batch`] returns `Vec<Vec<u32>>` so a row can
    /// carry replica destinations, which costs ONE HEAP ALLOCATION PER ROW --
    /// 600k of them for a 600k-row batch, and measurably the larger half of a
    /// durable save (phase timing put `assign` at 81% of an append-only save,
    /// of which the allocation floor is the part that does not scale with
    /// nlist). Replication is off in the recommended design, so the common
    /// path does not need the nesting.
    fn assign_batch_single(&mut self, batch: &RowBatch, dim: usize) -> Vec<u32> {
        let nlist = self.state.partitions.len();
        if !self.state.clustered || nlist <= 1 {
            return vec![0u32; batch.n];
        }
        let t_dec = std::time::Instant::now();
        let vectors = self.batch_vectors(batch, dim);
        self.stats.us_assign_decode = t_dec.elapsed().as_micros() as u64;
        let t_gemm = std::time::Instant::now();
        let assignments = if self.tuning.hierarchical_assign && nlist >= HIERARCHY_MIN_NLIST {
            // Rebuild when nlist has moved more than 20%: the level clusters
            // centroids, which are themselves a summary, so it tolerates being
            // a little stale far better than it tolerates being rebuilt every
            // save.
            let stale = match &self.coarse {
                None => true,
                Some(c) => {
                    let grown = nlist.max(c.built_for) as f64;
                    let shrunk = nlist.min(c.built_for) as f64;
                    grown / shrunk.max(1.0) > 1.2
                }
            };
            if stale {
                self.coarse = Some(kmeans::CoarseIndex::build(
                    &self.state.centroids,
                    nlist,
                    dim,
                    KMEANS_SEED ^ nlist as u64,
                ));
                self.stats.coarse_rebuilds += 1;
            }
            let coarse = self.coarse.as_ref().expect("just built");
            // Agreement with exact assignment at probe=1 is only ~33-37%
            // (Voronoi boundaries: a row's nearest centroid often sits in the
            // neighbouring super). 8 lifts it to 85-95%, and recall tolerates
            // more than agreement does because search probes many partitions
            // anyway.
            coarse.assign(&vectors, batch.n, &self.state.centroids, HIERARCHY_PROBE_SUPER)
        } else {
            // Top of a flush, one large batch, not inside any parallel loop:
            // take every core. The ambient check cannot see this, because the
            // extension runs the whole save inside an installed pool.
            kmeans::assign_ex(
                &vectors,
                batch.n,
                dim,
                &self.state.centroids,
                nlist,
                Some(true),
            )
            .0
        };
        self.stats.us_assign_gemm = t_gemm.elapsed().as_micros() as u64;
        self.stats.assign_rows = batch.n as u64;
        self.stats.assign_nlist = nlist as u64;
        assignments
    }

    /// Append rows that each have exactly one destination. Counts per
    /// partition first so every `RowBatch` is allocated once at its final
    /// size, and indexes partitions by position instead of hashing per row.
    fn append_batch_single(
        &mut self,
        batch: &RowBatch,
        destinations: &[u32],
        dim: usize,
        entries: &mut Vec<RunEntry>,
    ) -> io::Result<()> {
        let n_byte_groups = pack::n_byte_groups(self.state.bit_width, dim);
        let nlist = self.state.partitions.len().max(1);
        let mut counts = vec![0usize; nlist];
        for &position in destinations.iter().take(batch.n) {
            counts[position as usize] += 1;
        }
        let mut per_partition: Vec<Option<RowBatch>> = (0..nlist).map(|_| None).collect();
        for (position, &count) in counts.iter().enumerate() {
            if count > 0 {
                let mut rows = RowBatch::default();
                rows.reserve(count, n_byte_groups, dim, self.state.store_vectors);
                per_partition[position] = Some(rows);
            }
        }
        for (i, &position) in destinations.iter().enumerate().take(batch.n) {
            if let Some(rows) = per_partition[position as usize].as_mut() {
                rows.push_row(
                    batch.group_row(i, n_byte_groups),
                    batch.scales[i],
                    batch.ids[i],
                    false,
                    batch.vector(i, dim),
                );
            }
        }
        for position in 0..nlist {
            if let Some(rows) = per_partition[position].take() {
                if rows.n > 0 {
                    self.append_chunk(position, &rows, dim, entries, WriteReason::Ingest)?;
                }
            }
        }
        Ok(())
    }

    /// Primary partition position for each batch row (plus replica
    /// positions when closure assignment is on).
    fn assign_batch(&self, batch: &RowBatch, dim: usize) -> Vec<Vec<u32>> {
        let nlist = self.state.partitions.len();
        if !self.state.clustered || nlist <= 1 {
            return vec![vec![0]; batch.n];
        }
        let vectors = self.batch_vectors(batch, dim);
        let (assignments, _) = kmeans::assign(&vectors, batch.n, dim, &self.state.centroids, nlist);
        let replica_lists = match self.replica_epsilon {
            Some(epsilon) => closure_assignments_for_vectors(
                &vectors,
                batch.n,
                dim,
                &self.state.centroids,
                &assignments,
                epsilon,
                self.tuning.replica_prune,
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
        let n_byte_groups = pack::n_byte_groups(self.state.bit_width, dim);
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
            self.append_chunk(position as usize, rows, dim, entries, WriteReason::Ingest)?;
        }
        Ok(())
    }

    /// The batch's vectors: exact originals when stored, decoded
    /// approximations otherwise.
    fn batch_vectors(&self, batch: &RowBatch, dim: usize) -> Vec<f32> {
        if self.state.store_vectors && !batch.vectors.is_empty() {
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
        let t = std::time::Instant::now();
        let packed = pack::packed_from_group_bytes(group_rows, n, self.state.bit_width, dim);
        let out = decode::decode(
            &packed,
            scales,
            n,
            dim,
            self.state.bit_width,
            self.rotation_for(dim),
            self.codebook_for(dim),
            &self.state.tqplus_shift,
            &self.state.tqplus_scale,
        );
        self.state
            .decode_us
            .fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        out
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
        if !self.state.clustered {
            let total: u64 = self.state.partitions.iter().map(|p| p.live_primary).sum();
            if (total as usize) < MIN_PARTITIONS * target {
                self.collect_garbage(dim, entries, dropped_files)?;
                return Ok(());
            }
            self.rebuild_all(dim, target, None, entries, dropped_files)?;
            self.state.clustered = true;
        }

        // `changed` holds PARTITION IDS (stable across the structural
        // mutations below), never positions — positions shift whenever a
        // partition is dropped.
        let mut changed: HashSet<u32> = HashSet::new();
        let mut replica_refresh: HashSet<u64> = HashSet::new();

        // Split oversized partitions.
        self.split_oversized(
            target,
            dim,
            &mut changed,
            &mut replica_refresh,
            entries,
            dropped_files,
        )?;

        // Dissolve undersized partitions.
        while self.tuning.dissolve_enabled && self.state.partitions.len() > MIN_PARTITIONS {
            let Some(position) = self
                .state
                .partitions
                .iter()
                .position(|p| (p.live_primary as usize) < target / MERGE_DIVISOR)
            else {
                break;
            };
            self.stats.dissolves += 1;
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
        if self.tuning.reassign_enabled && !changed.is_empty() {
            self.reassign_pass(dim, &changed, &mut replica_refresh, entries)?;

            // Reassignment moves vectors INTO partitions, so it can push one
            // back over the size bound. Without this pass the violation
            // stands until the next flush, and the size bound is supposed to
            // be an invariant rather than a per-flush best effort.
            if self.tuning.resplit_after_reassign {
                self.stats.resplit_passes += 1;
                self.split_oversized(
                    target,
                    dim,
                    &mut changed,
                    &mut replica_refresh,
                    entries,
                    dropped_files,
                )?;
            }
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
        self.read_live_from(position, 0, primaries_only, dim)
    }

    /// `read_live`, restricted to chunks `from_chunk..`. A tier merge folds
    /// only a suffix of the chunk table, so it must decode only that suffix —
    /// decoding the whole partition is the cost the merge exists to avoid.
    fn read_live_from(
        &self,
        position: usize,
        from_chunk: usize,
        primaries_only: bool,
        dim: usize,
    ) -> io::Result<(RowBatch, Vec<u64>)> {
        let partition = &self.state.partitions[position];
        let n_byte_groups = pack::n_byte_groups(self.state.bit_width, dim);
        let mut batch = RowBatch::default();
        let mut source_rows = Vec::new();
        if partition.n_rows == 0 || from_chunk >= partition.chunks.len() {
            return Ok((batch, source_rows));
        }
        let map = self.segment_map(partition.partition_id, partition.generation)?;
        let mut block_rows = vec![0u8; BLOCK * n_byte_groups];
        let mut base_row: u64 = partition.chunks[..from_chunk]
            .iter()
            .map(|c| c.n_rows as u64)
            .sum();
        for chunk in &partition.chunks[from_chunk..] {
            let n_rows = chunk.n_rows as usize;
            let layout = chunk_layout(self.state.bit_width, dim, n_rows, self.state.store_vectors);
            let start = chunk.offset as usize;
            let codes = &map[start + layout.codes.0..start + layout.codes.0 + layout.codes.1];
            let scales =
                f32_slice(&map[start + layout.scales.0..start + layout.scales.0 + layout.scales.1]);
            let ids = u64_slice(&map[start + layout.ids.0..start + layout.ids.0 + layout.ids.1]);
            let replica_bits = &map[start + layout.replica_bits.0
                ..start + layout.replica_bits.0 + layout.replica_bits.1];
            let vectors = if self.state.store_vectors {
                Some(f32_slice(
                    &map[start + layout.vectors.0..start + layout.vectors.0 + layout.vectors.1],
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
    /// Split every partition above `SPLIT_FACTOR * target`, to fixpoint.
    fn split_oversized(
        &mut self,
        target: usize,
        dim: usize,
        changed: &mut HashSet<u32>,
        replica_refresh: &mut HashSet<u64>,
        entries: &mut Vec<RunEntry>,
        dropped_files: &mut Vec<PathBuf>,
    ) -> io::Result<()> {
        if !self.tuning.split_enabled {
            return Ok(());
        }
        let mut budget = match self.tuning.max_rewrites_per_flush {
            0 => usize::MAX,
            cap => cap,
        };
        while let Some(position) = self
            .state
            .partitions
            .iter()
            .position(|p| p.live_primary as usize > SPLIT_FACTOR * target)
        {
            if budget == 0 {
                break; // deferred to the next flush, not skipped
            }
            budget -= 1;
            if !self.split_partition(
                position,
                dim,
                changed,
                replica_refresh,
                entries,
                dropped_files,
            )? {
                break; // degenerate split; do not loop forever
            }
        }
        Ok(())
    }

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
        let n_byte_groups = pack::n_byte_groups(self.state.bit_width, dim);
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
            KMEANS_SEED ^ self.state.partitions[position].partition_id as u64,
        );
        let mut child_assignments = child_assignments;
        self.stats.splits_attempted += 1;
        let raw_one = child_assignments.iter().filter(|&&c| c == 1).count();
        let ratio = |ones: usize, n: usize| -> f64 {
            let (a, b) = (ones, n - ones);
            if a.min(b) == 0 {
                f64::INFINITY
            } else {
                a.max(b) as f64 / a.min(b) as f64
            }
        };
        let raw_ratio = ratio(raw_one, primaries.n);
        if raw_ratio.is_finite() {
            self.stats.split_ratio_raw_sum += raw_ratio;
        }
        if self.tuning.balanced_split && child_centroids.len() / dim >= 2 {
            balance_split(
                &vectors,
                primaries.n,
                dim,
                &child_centroids,
                &mut child_assignments,
            );
        }
        let child_one = child_assignments.iter().filter(|&&c| c == 1).count();
        if child_one != raw_one {
            self.stats.balance_fired += 1;
        }
        let final_ratio = ratio(child_one, primaries.n);
        if final_ratio.is_finite() {
            self.stats.split_ratio_final_sum += final_ratio;
        }
        if child_centroids.len() / dim < 2 || child_one == 0 || child_one == primaries.n {
            self.stats.splits_degenerate += 1;
            return Ok(false);
        }
        self.stats.splits_done += 1;
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
            self.append_chunk(new_position, rows, dim, entries, WriteReason::Split)?;
            changed.insert(self.state.partitions[new_position].partition_id);
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
        let n_byte_groups = pack::n_byte_groups(self.state.bit_width, dim);
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
        let nlist = self.state.partitions.len();
        let (assignments, _) =
            kmeans::assign(&vectors, primaries.n, dim, &self.state.centroids, nlist);
        self.append_batch_single(&primaries, &assignments, dim, entries)?;
        for &a in &assignments {
            changed.insert(self.state.partitions[a as usize].partition_id);
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
        let nlist = self.state.partitions.len();
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
            let center = &self.state.centroids[p * dim..(p + 1) * dim];
            let mut ranked: Vec<(f32, usize)> = (0..nlist)
                .filter(|&c| c != p)
                .map(|c| {
                    let other = &self.state.centroids[c * dim..(c + 1) * dim];
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
            affected.extend(
                ranked
                    .iter()
                    .take(self.tuning.reassign_neighbors)
                    .map(|&(_, c)| c),
            );
        }

        let n_byte_groups = pack::n_byte_groups(self.state.bit_width, dim);
        let mut membership_changed: HashSet<usize> = changed_positions;
        let mut positions: Vec<usize> = affected.into_iter().collect();
        positions.sort_unstable();
        if self.tuning.max_reassign_partitions > 0 {
            positions.truncate(self.tuning.max_reassign_partitions);
        }
        self.stats.reassign_passes += 1;
        self.stats.reassign_partitions += positions.len() as u64;

        // Two phases. Deciding WHICH rows move is the dominant cost of a save
        // -- a bulk add at 1.6M examines 1.26M rows across 953 partitions,
        // each needing an assignment against every centroid -- and it is
        // read-only and independent per partition. `kmeans::assign` is
        // internally parallel but only over one partition's ~1.3k rows, which
        // parallelises badly; hoisting the parallelism to the outer loop uses
        // the machine. Phase two applies the moves sequentially because
        // dead-marking, appending and the run-entry vector all mutate `self`.
        struct Reassignment {
            position: usize,
            examined: u64,
            moved: RowBatch,
            destinations: Vec<u32>,
            source_rows: Vec<u32>,
            ids: Vec<u64>,
        }

        let t_plan = std::time::Instant::now();
        let planned: Vec<io::Result<Option<Reassignment>>> = positions
            .par_iter()
            .map(|&position| {
                let (live, source_rows) = self.read_live(position, true, dim)?;
                if live.n == 0 {
                    return Ok(None);
                }
                let vectors = self.batch_vectors(&live, dim);
                let (best, best_distances) =
                    kmeans::assign(&vectors, live.n, dim, &self.state.centroids, nlist);
                let current_center = &self.state.centroids[position * dim..(position + 1) * dim];
                let mut moved = RowBatch::default();
                let mut destinations: Vec<u32> = Vec::new();
                let mut moved_rows: Vec<u32> = Vec::new();
                let mut moved_ids: Vec<u64> = Vec::new();
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
                        destinations.push(proposed);
                        moved_rows.push(source_rows[i] as u32);
                        moved_ids.push(live.ids[i]);
                    }
                }
                Ok(Some(Reassignment {
                    position,
                    examined: live.n as u64,
                    moved,
                    destinations,
                    source_rows: moved_rows,
                    ids: moved_ids,
                }))
            })
            .collect();

        self.stats.us_plan += t_plan.elapsed().as_micros() as u64;
        let t_apply = std::time::Instant::now();
        for plan in planned {
            let Some(plan) = plan? else { continue };
            self.stats.reassign_rows_examined += plan.examined;
            for (k, &row) in plan.source_rows.iter().enumerate() {
                self.dead_mark(CopyLocation {
                    partition_id: self.state.partitions[plan.position].partition_id,
                    row,
                    is_replica: false,
                });
                membership_changed.insert(plan.position);
                membership_changed.insert(plan.destinations[k] as usize);
                replica_refresh.insert(plan.ids[k]);
                self.stats.reassign_rows_moved += 1;
            }
            if plan.moved.n > 0 {
                self.append_batch_single(&plan.moved, &plan.destinations, dim, entries)?;
            }
        }

        self.stats.us_apply += t_apply.elapsed().as_micros() as u64;

        // Centroid refresh.
        for position in 0..self.state.partitions.len() {
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
            let centroid =
                &mut self.centroids_mut()[position * dim..(position + 1) * dim];
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
        let nlist = self.state.partitions.len();
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
            let primary_position =
                self.partition_position(primary.partition_id)
                    .expect("live copy implies live partition") as u32;
            let closure = closure_assignments_for_vectors(
                &vector,
                1,
                dim,
                &self.state.centroids,
                &[primary_position],
                epsilon,
                self.tuning.replica_prune,
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
            self.append_chunk(position as usize, &rows, dim, entries, WriteReason::Replica)?;
        }
        Ok(())
    }

    /// Fold each partition's trailing small chunks into one, without touching
    /// the large chunks in front of them.
    ///
    /// The problem this solves: trickle ingest appends one chunk per partition
    /// per save, and a batch spread over `nlist` partitions makes those chunks
    /// tiny (measured 11.4 rows at 1.6M/805 partitions; ~1 row at the 10M x
    /// 768d target). Compacting on chunk COUNT then rewrites the whole
    /// partition — its ~1024-row base chunk included — to absorb ~44 new rows,
    /// which measured 87.7% of all segment bytes and 21x row amplification.
    ///
    /// The fix is the logarithmic method: merge the last two chunks while the
    /// newer is at least as large as the older. Chunk sizes then form a
    /// decreasing sequence, so a row is rewritten O(log n) times instead of
    /// once per compaction, and the base chunk is only ever touched by a merge
    /// that has already accumulated an equal number of rows.
    ///
    /// The merged chunk is APPENDED and the originals abandoned in place
    /// (`file_bytes`), so no reader's offsets move and no generation bump is
    /// needed. The abandoned bytes are reclaimed by `collect_garbage`.
    fn tier_merge(&mut self, dim: usize, entries: &mut Vec<RunEntry>) -> io::Result<()> {
        if !self.tuning.tier_merge_enabled {
            return Ok(());
        }
        // Every partition, not just the ones maintenance touched: ingest
        // appends land before maintenance runs and are the main source of
        // small chunks. The check is two integer comparisons for a partition
        // with nothing to do, so scanning all of them is cheaper than tracking
        // which ones were appended to.
        for position in 0..self.state.partitions.len() {
            loop {
                let chunks = &self.state.partitions[position].chunks;
                if chunks.len() < 2 {
                    break;
                }
                // How far back does the merge reach? Extend while the run of
                // trailing chunks is at least as big as the chunk in front of
                // it -- that is the carry in the binary-counter analogy.
                let mut from = chunks.len() - 1;
                let mut run_rows = chunks[from].n_rows as u64;
                while from > 0 && run_rows >= chunks[from - 1].n_rows as u64 {
                    from -= 1;
                    run_rows += chunks[from].n_rows as u64;
                }
                if from == chunks.len() - 1 {
                    break; // trailing chunk is smaller than its predecessor
                }
                // Merging a single chunk into itself would loop forever, and
                // merging everything is a full rewrite by another name -- that
                // is `collect_garbage`'s job, and it reclaims the bytes.
                if from == 0 && self.state.partitions[position].chunks.len() > 1 {
                    let live = self.state.partitions[position].live_rows;
                    let n_rows = self.state.partitions[position].n_rows;
                    if live * 2 < n_rows {
                        break; // mostly dead: let compaction reclaim instead
                    }
                }
                let (rows, _) = self.read_live_from(position, from, false, dim)?;
                let partition = Arc::make_mut(&mut self.state.partitions[position]);
                let kept_rows: u64 = partition.chunks[..from]
                    .iter()
                    .map(|c| c.n_rows as u64)
                    .sum();
                let dropped_live: u64 = (kept_rows..partition.n_rows)
                    .filter(|&r| !partition.is_dead(r))
                    .count() as u64;
                let dropped_primary: u64 = rows.replica.iter().filter(|&&r| !r).count() as u64;
                partition.chunks.truncate(from);
                partition.n_rows = kept_rows;
                partition.live_rows -= dropped_live;
                partition.live_primary -= dropped_primary;
                partition.dead.truncate(((kept_rows + 7) / 8) as usize);
                // `dead` is a bitmap: truncating to a byte boundary can leave
                // stale high bits inside the final byte for rows that no
                // longer exist. Clear them, or the next resize reads them back.
                if kept_rows % 8 != 0 {
                    if let Some(last) = partition.dead.last_mut() {
                        *last &= (1u8 << (kept_rows % 8)) - 1;
                    }
                }
                self.stats.tier_merges += 1;
                self.append_chunk(position, &rows, dim, entries, WriteReason::TierMerge)?;
            }
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
        // Two phases. Reading and decoding every compacting partition is the
        // CPU half and is independent per partition, so it runs in parallel;
        // the metadata mutation and the append are sequential because they
        // touch `self.state.partitions` and the shared run-entry vector. After the
        // fsync batching the save path is compute-bound at ~0.9 of one core,
        // so this is where the remaining headroom is.
        // Merge first: it can retire chunks that would otherwise look like a
        // chunk-count violation, and it is the cheap fix.
        self.tier_merge(dim, entries)?;

        // Three reasons to rewrite a partition whole, and only the first two
        // are about reclaiming space:
        //
        //   dead ratio     rows removed by the caller
        //   garbage ratio  chunks abandoned in place by `tier_merge`
        //   chunk count    the pre-tier-merge trigger
        //
        // The chunk-count trigger is what made compaction 87.7% of segment
        // bytes: it fires on trickle-ingest chunk accumulation, which is not a
        // space problem at all, and pays a whole-partition rewrite to fix it.
        // With tier merging on, chunk count is bounded logarithmically and this
        // trigger is redundant, so it is left only as a backstop far above
        // where merging holds it.
        let chunk_ceiling = if self.tuning.tier_merge_enabled {
            self.tuning.gc_max_chunks.max(TIER_MERGE_CHUNK_BACKSTOP)
        } else {
            self.tuning.gc_max_chunks
        };
        let victims: Vec<usize> = (0..self.state.partitions.len())
            .filter(|&position| {
                let partition = &self.state.partitions[position];
                let dead = partition.n_rows - partition.live_rows;
                let dead_ratio = partition.n_rows > 0
                    && dead as f64 / partition.n_rows as f64 > self.tuning.gc_dead_ratio;
                let garbage = partition.file_bytes > 0
                    && partition.garbage_bytes(self.state.bit_width, dim, self.state.store_vectors)
                        as f64
                        / partition.file_bytes as f64
                        > self.tuning.gc_garbage_ratio;
                dead_ratio || garbage || partition.chunks.len() > chunk_ceiling
            })
            .collect();
        if victims.is_empty() {
            return Ok(());
        }
        let victims: Vec<usize> = match self.tuning.max_rewrites_per_flush {
            0 => victims,
            cap => victims.into_iter().take(cap).collect(),
        };
        let t_plan = std::time::Instant::now();
        let decoded: Vec<io::Result<(RowBatch, Vec<u64>)>> = victims
            .par_iter()
            .map(|&position| self.read_live(position, false, dim))
            .collect();

        self.stats.us_plan += t_plan.elapsed().as_micros() as u64;
        let t_apply = std::time::Instant::now();
        for (position, live) in victims.into_iter().zip(decoded) {
            let (live, _) = live?;
            self.stats.compactions += 1;
            let partition = Arc::make_mut(&mut self.state.partitions[position]);
            let old_path = segment_path(
                self.state
                    .directory
                    .as_deref()
                    .expect("flush requires a directory"),
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
            // New generation, new file: the abandoned merge bytes are exactly
            // what this rewrite reclaims.
            partition.file_bytes = 0;
            self.invalidate_segment_map(self.state.partitions[position].partition_id);
            self.append_chunk(position, &live, dim, entries, WriteReason::Compact)?;
        }
        self.stats.us_apply += t_apply.elapsed().as_micros() as u64;
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
        if !self.tuning.rebootstrap_enabled {
            return Ok(());
        }
        let total: u64 = self.state.partitions.iter().map(|p| p.live_primary).sum();
        if total == 0 || (self.churn_since_check as f64) < REBOOTSTRAP_CHURN_FRACTION * total as f64
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
        for position in 0..self.state.partitions.len() {
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
            &self.state.centroids,
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
                let (_, distances) =
                    kmeans::assign(&holdout_data, n_holdout, dim, &candidate, candidate_nlist);
                let distortion =
                    distances.iter().map(|&d| d as f64).sum::<f64>() / n_holdout.max(1) as f64;
                (candidate, distortion as f32)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("REBOOTSTRAP_CANDIDATES > 0");
        if candidate_distortion < REBOOTSTRAP_DISTORTION_RATIO * current_distortion {
            self.stats.rebootstraps += 1;
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
        let n_byte_groups = pack::n_byte_groups(self.state.bit_width, dim);
        let mut all = RowBatch::default();
        for position in 0..self.state.partitions.len() {
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
            assignments[chunk_start..chunk_start + chunk.len()].copy_from_slice(&chunk_assignments);
        }
        let replica_lists = match self.replica_epsilon {
            Some(epsilon) if nlist > 1 => closure_assignments_for_vectors(
                &vectors,
                all.n,
                dim,
                &centroids,
                &assignments,
                epsilon,
                self.tuning.replica_prune,
            ),
            _ => vec![Vec::new(); all.n],
        };

        // Drop every old partition, create the new set, append everything.
        while !self.state.partitions.is_empty() {
            self.drop_partition(0, dim, dropped_files);
        }
        self.state.centroids = Arc::new(Vec::new());
        self.centroids_dirty = true;
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

    /// Mutable centroids. Every mutation goes through here so the dirty flag
    /// cannot be missed -- a missed flag means the manifest keeps pointing at
    /// a stale blob and the index silently routes against old centroids after
    /// a reload.
    fn centroids_mut(&mut self) -> &mut Vec<f32> {
        self.centroids_dirty = true;
        Arc::make_mut(&mut self.state.centroids)
    }

    fn create_partition(
        &mut self,
        centroid: &[f32],
        _entries: &mut [RunEntry],
    ) -> io::Result<usize> {
        let partition_id = self.next_partition_id;
        self.next_partition_id += 1;
        self.state.partitions.push(Arc::new(PartitionState {
            partition_id,
            generation: 0,
            n_rows: 0,
            live_rows: 0,
            live_primary: 0,
            chunks: Vec::new(),
            dead: Vec::new(),
            file_bytes: 0,
        }));
        self.centroids_mut().extend_from_slice(centroid);
        Ok(self.state.partitions.len() - 1)
    }

    fn drop_partition(&mut self, position: usize, dim: usize, dropped_files: &mut Vec<PathBuf>) {
        let partition = self.state.partitions.remove(position);
        if let Some(directory) = self.state.directory.as_deref() {
            dropped_files.push(segment_path(
                directory,
                partition.partition_id,
                partition.generation,
            ));
        }
        self.invalidate_segment_map(partition.partition_id);
        self.centroids_mut().drain(position * dim..(position + 1) * dim);
    }

    /// Append `rows` to `position`'s segment file as one chunk, fsync, and
    /// record run entries for every row.
    /// fsync every segment file touched since the last commit. Called once,
    /// immediately before the manifest rename that makes the new state
    /// visible, so a crash at any earlier point still leaves the previous
    /// manifest pointing at the previous (fully durable) state.
    fn sync_pending(&mut self) -> io::Result<()> {
        if self.pending_syncs.is_empty() {
            return Ok(());
        }
        let mut paths = std::mem::take(&mut self.pending_syncs);
        paths.sort_unstable();
        paths.dedup();

        // ONE device barrier for the whole batch, not one per file.
        //
        // `F_FULLFSYNC` tells the DRIVE to flush its write cache; it is a
        // device-level command, so a single call covers every byte already
        // handed to the device. `fsync(2)` is what hands a file's bytes over,
        // and it is comparatively cheap because it stops there.
        //
        // Issuing the device barrier per file made durability cost scale with
        // the number of partitions a batch touched, which grows with nlist --
        // measured at 800k x 768d, that was 92% of save wall (6,233 ms against
        // 496 ms with barriers off) and it was superlinear, because a wider
        // index spreads the same batch over more files. Per-file `fsync` plus
        // one barrier gives the same guarantee: every file's data is at the
        // device before the barrier, and the barrier flushes all of it.
        let mut last = None;
        for path in &paths {
            // A file may have been dropped by a later split/dissolve in the
            // same flush; nothing to make durable in that case.
            if let Ok(file) = OpenOptions::new().write(true).open(path) {
                file.sync_all()?;
                last = Some(file);
            }
        }
        if let Some(file) = last {
            full_sync(&file)?;
        }
        Ok(())
    }

    fn append_chunk(
        &mut self,
        position: usize,
        rows: &RowBatch,
        dim: usize,
        entries: &mut Vec<RunEntry>,
        reason: WriteReason,
    ) -> io::Result<()> {
        if rows.n == 0 {
            return Ok(());
        }
        {
            let bytes = chunk_layout(self.state.bit_width, dim, rows.n, self.state.store_vectors)
                .total_len as u64;
            let n = rows.n as u64;
            let s = &mut self.stats;
            match reason {
                WriteReason::Ingest => {
                    s.bytes_ingest += bytes;
                    s.rows_ingest += n;
                    s.chunks_ingest += 1;
                }
                WriteReason::Compact => {
                    s.bytes_compact += bytes;
                    s.rows_compact += n;
                    s.chunks_compact += 1;
                }
                WriteReason::Split => {
                    s.bytes_split += bytes;
                    s.rows_split += n;
                    s.chunks_split += 1;
                }
                WriteReason::Replica => {
                    s.bytes_replica += bytes;
                    s.rows_replica += n;
                    s.chunks_replica += 1;
                }
                WriteReason::TierMerge => {
                    s.bytes_tier += bytes;
                    s.rows_tier += n;
                    s.chunks_tier += 1;
                }
                WriteReason::Import => s.bytes_import += bytes,
            }
        }
        let directory = self
            .state
            .directory
            .as_deref()
            .expect("flush requires a bound directory")
            .to_path_buf();
        let n_byte_groups = pack::n_byte_groups(self.state.bit_width, dim);
        let partition = Arc::make_mut(&mut self.state.partitions[position]);
        let path = segment_path(&directory, partition.partition_id, partition.generation);
        // Physical end of file, not end-of-last-chunk: a tier merge leaves
        // abandoned bytes behind that older snapshots still read.
        let append_offset = partition.file_bytes;

        let layout = chunk_layout(self.state.bit_width, dim, rows.n, self.state.store_vectors);
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
        header[8..12].copy_from_slice(&(((rows.n + BLOCK - 1) / BLOCK) as u32).to_le_bytes());
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

        if self.state.store_vectors {
            position_bytes = pad_to(&mut writer, position_bytes, layout.vectors.0)?;
            writer.write_all(f32_bytes(&rows.vectors))?;
            position_bytes += 4 * rows.n * dim;
        }
        position_bytes = pad_to(&mut writer, position_bytes, layout.total_len)?;
        debug_assert_eq!(position_bytes, layout.total_len);
        writer.flush()?;
        drop(writer);
        // Durability is owed at the manifest rename, not here (see
        // `pending_syncs`); batching turns thousands of fsyncs into one pass.
        self.pending_syncs.push(path.clone());

        // Bookkeeping + run entries.
        let base_row = partition.n_rows;
        partition.chunks.push(ChunkMeta {
            offset: append_offset,
            n_rows: rows.n as u32,
        });
        partition.file_bytes = append_offset + layout.total_len as u64;
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
                flags: if rows.replica[i] {
                    ENTRY_FLAG_REPLICA
                } else {
                    0
                },
            });
        }
        self.invalidate_segment_map(self.state.partitions[position].partition_id);
        Ok(())
    }

    fn dead_mark(&mut self, copy: CopyLocation) {
        let Some(position) = self.partition_position(copy.partition_id) else {
            return;
        };
        let partition = Arc::make_mut(&mut self.state.partitions[position]);
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

    fn write_run(&mut self, entries: &[RunEntry]) -> io::Result<()> {
        let directory = self
            .state
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
        writer
            .into_inner()
            .map_err(|e| e.into_error())?
            .sync_all()?;
        Arc::make_mut(&mut self.state.runs).push(generation);
        Ok(())
    }

    /// Merge every run into one, keeping only currently-valid entries.
    fn merge_runs(&mut self, dropped_files: &mut Vec<PathBuf>) -> io::Result<()> {
        let t_merge = std::time::Instant::now();
        let out = self.merge_runs_inner(dropped_files);
        self.stats.run_merges += 1;
        self.stats.us_run_merge += t_merge.elapsed().as_micros() as u64;
        out
    }

    fn merge_runs_inner(&mut self, dropped_files: &mut Vec<PathBuf>) -> io::Result<()> {
        let n = self.state.runs.len();
        self.merge_run_range(0, n, dropped_files)
    }

    /// Merge runs `[from, to)` into one, preserving their position in the
    /// newest-first lookup order.
    ///
    /// Lookup scans runs newest-first and takes the first hit, so a newer run
    /// shadows an older one for the same id. Merging a CONTIGUOUS range keeps
    /// that: within the range the same newest-first walk decides which entry
    /// survives, and the result sits where the range was.
    fn merge_run_range(
        &mut self,
        from: usize,
        to: usize,
        dropped_files: &mut Vec<PathBuf>,
    ) -> io::Result<()> {
        let directory = self
            .state
            .directory
            .as_deref()
            .expect("flush requires a bound directory")
            .to_path_buf();
        let mut merged: Vec<RunEntry> = Vec::new();
        let mut seen: HashSet<(u32, u32)> = HashSet::new();
        let range: Vec<u64> = self.state.runs[from..to].to_vec();
        let tail: Vec<u64> = self.state.runs[to..].to_vec();
        for &generation in range.iter().rev() {
            let map = self.run_map(generation)?;
            let count = run_entry_count(&map);
            for idx in 0..count {
                let entry = run_entry(&map, idx);
                let Some(position) = self.partition_position(entry.partition_id) else {
                    continue;
                };
                let partition = &self.state.partitions[position];
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
        for &generation in range.iter() {
            dropped_files.push(run_path(&directory, generation));
        }
        Arc::make_mut(&mut self.state.runs).truncate(from);
        self.state
            .caches
            .run_maps
            .lock()
            .expect("run map lock")
            .clear();
        self.write_run(&merged)?;
        // Runs newer than the merged range keep their order after it.
        Arc::make_mut(&mut self.state.runs).extend_from_slice(&tail);
        Ok(())
    }

    /// Merge runs in size tiers instead of collapsing all of them.
    ///
    /// The all-runs merge rewrites every live entry, so its cost is O(N) and
    /// it fires every `MAX_RUNS` saves -- measured as the largest single term
    /// in `publish`, which grew 16x over an 8.2x rise in nlist while every
    /// other phase grew sublinearly.
    ///
    /// The same carry rule `tier_merge` uses for chunks applies here: merge
    /// the newest run of runs while it is at least as large as the run before
    /// it. Sizes then decrease from the front, so an entry is rewritten
    /// O(log n) times rather than once per merge cycle, and the large oldest
    /// run is only touched by a merge that has already accumulated as many
    /// entries as it holds.
    fn tier_merge_runs(&mut self, dropped_files: &mut Vec<PathBuf>) -> io::Result<()> {
        loop {
            let n = self.state.runs.len();
            if n < 2 {
                break;
            }
            let mut sizes = Vec::with_capacity(n);
            for &generation in self.state.runs.iter() {
                let map = self.run_map(generation)?;
                sizes.push(run_entry_count(&map) as u64);
            }
            let mut from = n - 1;
            let mut acc = sizes[from];
            while from > 0 && acc >= sizes[from - 1] {
                from -= 1;
                acc += sizes[from];
            }
            if from == n - 1 {
                break; // newest run is smaller than its predecessor
            }
            self.stats.run_merges += 1;
            let t = std::time::Instant::now();
            self.merge_run_range(from, n, dropped_files)?;
            self.stats.us_run_merge += t.elapsed().as_micros() as u64;
        }
        // Backstop: tiering bounds the count logarithmically, so this only
        // fires if that is somehow not keeping up, and degrades to the old
        // behaviour rather than letting lookup walk an unbounded list.
        if self.state.runs.len() > RUN_TIER_BACKSTOP {
            self.merge_runs(dropped_files)?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Mmap caches
    // ------------------------------------------------------------------

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

    /// Make the write-ahead log durable. Errors were swallowed here; a failed
    /// sync means an `add` or `remove` that returned is not actually on disk,
    /// which is exactly the case the log exists for.
    fn wal_sync(&mut self) -> io::Result<()> {
        if let Some(wal) = self.wal.as_ref() {
            full_sync(wal)?;
        }
        Ok(())
    }

    fn reset_wal(&mut self) -> io::Result<()> {
        let Some(directory) = self.state.directory.as_deref() else {
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
        self.wal = Some(
            OpenOptions::new()
                .append(true)
                .open(directory.join(WAL_FILE))?,
        );
        self.wal_dim = self.dim();
        Ok(())
    }

    fn replay_or_reset_wal(&mut self) -> io::Result<()> {
        let directory = self
            .state
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
                    self.mem_write()
                        .get_mut()
                        .add_with_ids_2d(&vector, dim, &[id])
                        .map_err(|e| invalid_data(format!("WAL replay: {e}")))?;
                    if self.state.store_vectors {
                        self.mem_write().originals.insert(id, vector.into());
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

    fn write_manifest(&mut self) -> io::Result<()> {
        let directory = self
            .state
            .directory
            .as_deref()
            .expect("flush requires a bound directory");
        let dim = self.dim();
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(MANIFEST_MAGIC);
        let mut flags = 0u8;
        if self.state.store_vectors {
            flags |= FLAG_HAS_VECTORS;
        }
        if self.state.clustered {
            flags |= FLAG_CLUSTERED;
        }
        out.push(FORMAT_VERSION);
        out.push(self.state.bit_width as u8);
        out.push(flags);
        out.push(0);
        out.extend_from_slice(&(dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.state.tqplus_shift.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.epoch.to_le_bytes());
        out.extend_from_slice(&(self.partition_target.unwrap_or(0) as u32).to_le_bytes());
        out.extend_from_slice(&self.replica_epsilon.unwrap_or(0.0).to_le_bytes());
        out.extend_from_slice(&self.next_partition_id.to_le_bytes());
        out.extend_from_slice(&(self.state.partitions.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.next_run_generation.to_le_bytes());
        out.extend_from_slice(&self.churn_since_check.to_le_bytes());
        out.extend_from_slice(&(self.state.runs.len() as u32).to_le_bytes());
        out.extend_from_slice(&[0u8; 4]);
        out.extend_from_slice(f32_bytes(&self.state.tqplus_shift));
        out.extend_from_slice(f32_bytes(&self.state.tqplus_scale));
        for &generation in self.state.runs.iter() {
            out.extend_from_slice(&generation.to_le_bytes());
        }
        for partition in &self.state.partitions {
            out.extend_from_slice(&partition.partition_id.to_le_bytes());
            out.extend_from_slice(&partition.generation.to_le_bytes());
            out.extend_from_slice(&partition.n_rows.to_le_bytes());
            out.extend_from_slice(&partition.live_rows.to_le_bytes());
            out.extend_from_slice(&partition.live_primary.to_le_bytes());
            out.extend_from_slice(&partition.file_bytes.to_le_bytes());
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
        // Centroids by reference, not by value. Rewriting them costs
        // `nlist * dim * 4` on every save -- ~30 MB at 10M x 768d -- and 73%
        // of saves change none of them.
        let stale = directory.join(format!(
            "{}{:016}",
            CENTROIDS_PREFIX, self.centroids_generation
        ));
        if self.centroids_dirty || !stale.exists() {
            // A NEW generation, never an overwrite: the old manifest still
            // names the old generation until the rename below lands, and
            // rewriting that file in place would change what an unreplaced
            // manifest resolves to.
            self.centroids_generation += 1;
            let path = directory.join(format!(
                "{}{:016}",
                CENTROIDS_PREFIX, self.centroids_generation
            ));
            // Durable BEFORE the manifest rename that names it, so a manifest
            // never references a blob that is not on disk.
            let temp = directory.join(format!("{}tmp", CENTROIDS_PREFIX));
            let mut file = File::create(&temp)?;
            file.write_all(f32_bytes(&self.state.centroids))?;
            full_sync(&file)?;
            fs::rename(&temp, &path)?;
            self.centroids_dirty = false;
            // The superseded blob is retired only after the manifest rename
            // publishes its replacement; an orphan left by a crash in between
            // is swept at the next open.
            self.retired_centroids = Some(stale);
        }
        out.extend_from_slice(&self.centroids_generation.to_le_bytes());
        let crc = crc32(&out);
        out.extend_from_slice(&crc.to_le_bytes());

        // The commit point. Write the new manifest beside the old one, make it
        // durable, then rename over the old one -- rename is atomic, so a
        // crash at any instant leaves a directory referring entirely to the
        // old generation or entirely to the new one, never a mixture. The
        // final fsync is on the DIRECTORY: without it the rename itself can
        // be lost, which would silently roll the index back one epoch.
        let temp_path = directory.join(MANIFEST_TEMP_FILE);
        let mut file = File::create(&temp_path)?;
        file.write_all(&out)?;
        full_sync(&file)?;
        fs::rename(&temp_path, directory.join(MANIFEST_FILE))?;
        let dir_handle = File::open(directory)?;
        full_sync(&dir_handle)?;
        if let Some(stale) = self.retired_centroids.take() {
            fs::remove_file(stale).ok();
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
            return Err(invalid_data(
                "corrupt FreshIndex manifest (crc)".to_string(),
            ));
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
            let file_bytes = cursor.u64()?;
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
                file_bytes,
            });
        }
        let centroids_generation = cursor.u64()?;
        let centroids = {
            let path = directory.join(format!(
                "{}{:016}",
                CENTROIDS_PREFIX, centroids_generation
            ));
            let bytes = fs::read(&path).map_err(|e| {
                invalid_data(format!(
                    "manifest references {} which cannot be read: {e}",
                    path.display(),
                ))
            })?;
            let want = n_partitions * dim * 4;
            if bytes.len() != want {
                return Err(invalid_data(format!(
                    "centroid blob {} is {} bytes, expected {want}",
                    path.display(),
                    bytes.len(),
                )));
            }
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>()
        };

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
                IdMapIndex::from_inner(
                    inner.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?,
                )
            } else {
                IdMapIndex::new(dim, bit_width)
                    .map_err(|e| invalid_data(format!("invalid manifest parameters: {e}")))?
            }
        } else {
            IdMapIndex::new_lazy(bit_width)
                .map_err(|e| invalid_data(format!("invalid manifest parameters: {e}")))?
        };

        let mut index = Self::from_state(IndexState {
            directory: Some(directory.to_path_buf()),
            bit_width,
            store_vectors,
            clustered,
            tqplus_shift,
            tqplus_scale,
            centroids: Arc::new(centroids),
            partitions: partitions.into_iter().map(Arc::new).collect(),
            runs: Arc::new(runs),
            memtable: MemtableCell::new(memtable),
            caches: Arc::new(Caches::default()),
            memtable_scan_us: Arc::new(AtomicU64::new(0)),
            decode_us: Arc::new(AtomicU64::new(0)),
        });
        index.replica_epsilon = if replica_epsilon_raw > 0.0 {
            Some(replica_epsilon_raw)
        } else {
            None
        };
        // Tuning is a runtime choice, not a property of the stored index,
        // so a reopened index starts from the defaults.
        index.partition_target = if target > 0 { Some(target) } else { None };
        index.epoch = epoch;
        // Restore which centroid blob this manifest names, and mark it clean:
        // the in-memory centroids came FROM that blob, so nothing has diverged
        // yet. Leaving the generation at its constructor default makes the
        // orphan sweep delete the very file the manifest references -- and it
        // does so on the SECOND open, because the first still reads the
        // manifest before sweeping.
        index.centroids_generation = centroids_generation;
        index.centroids_dirty = false;
        index.next_partition_id = next_partition_id;
        index.next_run_generation = next_run_generation;
        index.churn_since_check = churn_since_check;
        index.publish();
        Ok(index)
    }

    /// Remove files the manifest does not reference (crashed flushes), and
    /// truncate segment files to their recorded lengths.
    fn cleanup_directory(&self) -> io::Result<()> {
        let directory = self
            .state
            .directory
            .as_deref()
            .expect("open always binds a directory");
        let dim = self.dim();
        let known_segments: HashMap<u32, u32> = self
            .state
            .partitions
            .iter()
            .map(|p| (p.partition_id, p.generation))
            .collect();
        let known_runs: HashSet<u64> = self.state.runs.iter().copied().collect();
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == MANIFEST_FILE || name == WAL_FILE {
                continue;
            }
            if let Some(rest) = name.strip_prefix(CENTROIDS_PREFIX) {
                // Keep the generation the manifest names; every other blob is
                // a superseded one, or an orphan from a crash mid-write.
                if rest.parse::<u64>() == Ok(self.centroids_generation) {
                    continue;
                }
            }
            if let Some(rest) = name.strip_prefix("segment-") {
                let mut parts = rest.splitn(2, '-');
                let id = parts.next().and_then(|s| s.parse::<u32>().ok());
                let generation = parts.next().and_then(|s| s.parse::<u32>().ok());
                if let (Some(id), Some(generation)) = (id, generation) {
                    if known_segments.get(&id) == Some(&generation) {
                        // Truncate any orphaned tail from a crashed append.
                        if let Some(partition) =
                            self.state.partitions.iter().find(|p| p.partition_id == id)
                        {
                            // `file_bytes` is the committed physical end. Bytes
                            // past it are a crashed append; bytes between the
                            // last live chunk and it are merge garbage and must
                            // be kept, so the chunk table cannot be the bound.
                            let expected = partition.file_bytes;
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
        index.state.directory = Some(directory.to_path_buf());
        index.state.store_vectors = contents.store_vectors;
        index.replica_epsilon = contents.replica_epsilon;
        index.partition_target = contents.partition_target;
        index.state.tqplus_shift = contents.tqplus_shift;
        index.state.tqplus_scale = contents.tqplus_scale;
        index.state.clustered = contents.partitions.len() > 1;

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
            index.append_chunk(position, &batch, dim, &mut entries, WriteReason::Import)?;
        }
        if !entries.is_empty() {
            index.write_run(&entries)?;
        }
        index.epoch = 1;
        index.write_manifest()?;
        index.reset_wal()?;
        index.publish();
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
        let n_byte_groups = pack::n_byte_groups(self.state.bit_width, dim.max(1));
        let mut all = RowBatch::default();
        for position in 0..self.state.partitions.len() {
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
            pack::packed_from_group_bytes(&all.group_rows, all.n, self.state.bit_width, dim);
        // Upstream's writer takes the blocked-SEQUENTIAL layout, padded to
        // whole 32-lane blocks; `packed_from_group_bytes` yields the unpadded
        // bit-plane form. Writing the latter silently produced a file whose
        // code section was short by the padding of the final partial block.
        let packed = pack::repack_seq(&packed, all.n, self.state.bit_width, dim);
        // Upstream's writer now persists the codebook alongside the codes,
        // so a reader never has to re-derive it. It is a pure function of
        // (bit_width, dim), so this is the same table the encoder used.
        let (boundaries, centroids) = codebook::codebook(self.state.bit_width, dim);
        crate::io::write_id_map(
            dst.as_ref(),
            self.state.bit_width,
            dim,
            all.n,
            &packed,
            &boundaries,
            &centroids,
            &all.scales,
            &self.state.tqplus_shift,
            &self.state.tqplus_scale,
            &all.ids,
        )
    }
}

impl std::fmt::Debug for FreshIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FreshIndex")
            .field("bit_width", &self.state.bit_width)
            .field("dim", &self.dim_opt())
            .field("directory", &self.state.directory)
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

/// Flush a file all the way to stable storage.
///
/// `File::sync_all` is `fsync(2)`. On Darwin — which is macOS AND iOS, i.e.
/// half the deployment target — `fsync` hands the data to the drive but does
/// NOT make the drive flush its own write cache, so a power cut can still lose
/// writes that `fsync` reported durable. `F_FULLFSYNC` is the call that does,
/// and there is no way to reach it through `std`.
///
/// ON by default, because it is what makes the manifest-rename commit mean
/// what it claims on the target platform, and because it measured **free**:
/// 8 x 10k saves into a 400k index averaged 196.3 ms with it against 198.7 ms
/// without (`results/writer_profile.jsonl`). That is a result for APFS on this
/// NVMe and may not hold on phone flash, so `TURBOVEC_FULLSYNC=0` disables it.
///
/// Linux `fsync` already flushes the device cache on mainstream filesystems,
/// so this is a no-op difference there.
fn full_sync(file: &File) -> io::Result<()> {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        if fullsync_enabled() {
            use std::os::unix::io::AsRawFd;
            const F_FULLFSYNC: i32 = 51;
            extern "C" {
                fn fcntl(fd: i32, cmd: i32, ...) -> i32;
            }
            // SAFETY: `file` owns a valid descriptor for the duration of the
            // call, and F_FULLFSYNC takes no argument.
            let rc = unsafe { fcntl(file.as_raw_fd(), F_FULLFSYNC, 0) };
            if rc == -1 {
                // Not every filesystem implements it; fall back rather than
                // fail the commit.
                return file.sync_all();
            }
            return Ok(());
        }
    }
    file.sync_all()
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn fullsync_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("TURBOVEC_FULLSYNC")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
            .unwrap_or(true)
    })
}

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

impl FreshReader {
    /// The snapshot current at this instant. Held for one query.
    ///
    /// This is the only synchronisation on the read path: an `RwLock` read
    /// acquire around an `Arc` clone. It is never held while scanning, so a
    /// writer's publish — which takes the same lock for the length of one
    /// pointer swap — cannot stall a query behind maintenance.
    pub fn snapshot(&self) -> Arc<Snapshot> {
        Arc::clone(&self.published.read().expect("snapshot lock"))
    }

    pub fn search(&self, queries: &[f32], k: usize) -> (Vec<f32>, Vec<u64>) {
        self.snapshot().state.search(queries, k)
    }

    pub fn search_with_options(
        &self,
        queries: &[f32],
        k: usize,
        options: SearchOptions,
    ) -> (Vec<f32>, Vec<u64>) {
        self.snapshot()
            .state
            .search_with_options(queries, k, options)
    }

    pub fn contains(&self, id: u64) -> bool {
        self.snapshot().state.contains(id)
    }

    pub fn get_vector(&self, id: u64) -> Option<Vec<f32>> {
        self.snapshot().state.get_vector(id)
    }

    pub fn prepare(&self) {
        self.snapshot().state.prepare();
    }

    pub fn dim_opt(&self) -> Option<usize> {
        self.snapshot().state.dim_opt()
    }

    pub fn dim(&self) -> usize {
        self.snapshot().state.dim()
    }

    pub fn bit_width(&self) -> usize {
        self.snapshot().state.bit_width()
    }

    pub fn stores_vectors(&self) -> bool {
        self.snapshot().state.stores_vectors()
    }

    pub fn len(&self) -> usize {
        self.snapshot().state.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn nlist(&self) -> usize {
        self.snapshot().state.nlist()
    }

    pub fn base_len(&self) -> usize {
        self.snapshot().state.base_len()
    }

    pub fn memtable_len(&self) -> usize {
        self.snapshot().state.memtable_len()
    }

    pub fn dead_count(&self) -> usize {
        self.snapshot().state.dead_count()
    }

    pub fn partition_sizes(&self) -> Vec<usize> {
        self.snapshot().state.partition_sizes()
    }

    pub fn replica_count(&self) -> usize {
        self.snapshot().state.replica_count()
    }

    pub fn run_count(&self) -> usize {
        self.snapshot().state.run_count()
    }

    pub fn chunk_count(&self) -> usize {
        self.snapshot().state.chunk_count()
    }
}

#[cfg(test)]
mod dead_in_tests {
    use super::*;

    /// `dead_in` is masked bit arithmetic over a range that need not be byte
    /// aligned at either end, so it is checked against the obvious loop over
    /// `is_dead` for every range of a partition with an awkward row count.
    #[test]
    fn dead_in_matches_per_row_counting() {
        for n_rows in [1u64, 7, 8, 9, 63, 64, 65, 100] {
            let mut p = PartitionState {
                partition_id: 0,
                generation: 0,
                n_rows,
                live_rows: n_rows,
                live_primary: n_rows,
                chunks: Vec::new(),
                dead: vec![0u8; ((n_rows + 7) / 8) as usize],
                file_bytes: 0,
            };
            // A deterministic scatter of dead rows, including row 0 and the
            // last row, which are the ends the masks operate on.
            for r in 0..n_rows {
                if r % 3 == 0 || r == n_rows - 1 {
                    p.dead[(r / 8) as usize] |= 1 << (r % 8);
                }
            }
            for base in 0..n_rows {
                for len in 0..=(n_rows - base) {
                    let want = (base..base + len).filter(|&r| p.is_dead(r)).count();
                    let got = p.dead_in(base, len as usize);
                    assert_eq!(
                        got, want,
                        "n_rows={n_rows} base={base} len={len}: got {got}, want {want}",
                    );
                }
            }
        }
    }
}
