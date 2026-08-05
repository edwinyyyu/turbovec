//! Disk-primary index: an mmap-backed base segment plus a small in-RAM
//! delta, with optional SPFresh-style partitioning.
//!
//! [`DiskIndex`] answers the same queries as [`IdMapIndex`](crate::IdMapIndex)
//! but keeps the bulk of its data — quantized codes, per-vector scales and
//! the id tables — in a `.tvdm` file that is memory-mapped rather than read
//! into the heap. The codes are stored in the *same SIMD-blocked layout*
//! the scoring kernel consumes, so [`DiskIndex::search`] runs
//! [`search::scan`] directly over the mapped bytes: no per-query
//! deserialization, no repacking, and no resident copy of the index.
//!
//! Resident memory at query time is therefore the OS page cache (evictable,
//! shared, demand-paged) plus:
//! * the in-RAM **delta** — an [`IdMapIndex`] holding vectors added since
//!   the last [`DiskIndex::write`], encoded with the base segment's TQ+
//!   calibration so base and delta scores are directly comparable, and
//! * the **tombstone set** — external ids removed from the base segment
//!   since the last write.
//!
//! [`DiskIndex::write`] compacts: it streams the base segment (minus
//! tombstones) and the delta into a fresh `.tvdm` file, atomically replaces
//! the old file, re-maps it, and empties the delta and tombstones.
//!
//! # Partitioning (opt-in)
//!
//! With [`DiskIndex::set_partitioning`], compaction clusters the codes into
//! partitions of roughly `target_partition_size` vectors (k-means over
//! approximate reconstructions — see `decode.rs`), each stored as its own
//! contiguous blocked segment. Search then ranks partition centroids and
//! scans only the `nprobe` nearest partitions, making queries touch
//! `~nprobe/nlist` of the file instead of all of it — both faster warm and
//! drastically cheaper under memory pressure, at the cost of approximate
//! routing (a true neighbor in an unprobed partition is missed).
//!
//! Updates follow a lightweight version of SPFresh's LIRE protocol: new
//! vectors are assigned to the nearest existing partition at compaction;
//! partitions that outgrow `2 * target` are split by 2-means; partitions
//! that shrink below `target / 4` are dissolved into their nearest
//! neighbors. All repair work is local to the affected partitions.
//!
//! Three recall levers beyond plain routed search, all opt-in:
//!
//! * **Adaptive probing** ([`SearchOptions::probe_epsilon`]): instead of a
//!   fixed partition count, each query probes every partition whose
//!   centroid is within `(1 + epsilon)` of its nearest centroid's distance
//!   (capped by `nprobe`). Queries that land near partition boundaries —
//!   exactly the ones fixed-`nprobe` routing starves — automatically open
//!   more partitions.
//! * **Boundary multi-assignment** ([`DiskIndex::set_replication`]):
//!   SPANN-style closure assignment at compaction. A vector lands in its
//!   nearest partition and is *replicated* into every partition whose
//!   centroid is within `(1 + epsilon)` of the nearest (RNG-rule pruned,
//!   at most [`MAX_ASSIGNMENT_COPIES`] copies), so boundary vectors are
//!   findable from either side of a partition boundary. Replicas are
//!   marked in a per-slot bitmap; maintenance, compaction and conversion
//!   operate on primary copies only, and search de-duplicates by id.
//! * **Exact rescoring** (`store_vectors` + [`SearchOptions::rescore_k`]):
//!   the full-precision vectors are kept in a final mmap section. Searches
//!   over-fetch quantized candidates and re-rank the best `rescore_k` of
//!   them by exact f32 inner product (a handful of random page reads per
//!   query), lifting the quantization ceiling without resident cost. Also
//!   enables [`DiskIndex::get_vector`], and clustering maintenance runs on
//!   the exact vectors instead of decoded approximations.
//!
//! # File format (`.tvdm`, version 3)
//!
//! A 64-byte header followed by 64-byte-aligned sections:
//!
//! ```text
//! header:   magic "TVDM" | version u8 | bit_width u8 | layout u8 | flags u8
//!           dim u32 | n_calib u32 | n_slots u64 | nlist u32 |
//!           target_partition_size u32 | n_unique u64 |
//!           replica_epsilon f32 | zero pad to 64
//! sections: partition table     (nlist * { n u64, slot_base u64, codes_off u64 })
//!           partition centroids (f32 * nlist * dim)
//!           blocked codes       (per-partition segments, each padded to BLOCK)
//!           scales              (f32 * n_slots, in partition order)
//!           tqplus_shift        (f32 * n_calib)
//!           tqplus_scale        (f32 * n_calib)
//!           slot_to_id          (u64 * n_slots, in partition order)
//!           sorted_ids          (u64 * n_slots, non-decreasing)
//!           sorted_slots        (u64 * n_slots, slot of sorted_ids[i])
//!           replica_flags       (bitmap, ceil(n_slots / 8) bytes)
//!           vectors             (f32 * n_slots * dim; only with flags bit 0)
//! ```
//!
//! `n_slots` counts physical rows (primaries plus replicas); `n_unique`
//! counts distinct external ids. Without replication the two are equal and
//! `sorted_ids` is strictly increasing. `flags` bit 0 records whether the
//! `vectors` section is present.
//!
//! Sections are read in place as `&[u8]` / `&[f32]` /
//! `&[u64]` slices, which is why values are little-endian (the module is
//! compiled out on big-endian targets) and why the blocked-codes layout is
//! architecture-flavored: the `layout` header byte records which flavor was
//! written (0 = sequential / NEON, 1 = x86 perm0-interleaved) and `open`
//! refuses a file written for the other flavor with a rebuild hint.
//!
//! `sorted_ids` / `sorted_slots` exist so id membership tests
//! (`contains`, duplicate-id rejection on add, remove) are a binary search
//! over the mapping instead of an O(n)-RAM hash table.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use memmap2::{Advice, Mmap};

use crate::id_map::IdMapIndex;
use crate::{
    codebook, decode, first_invalid_coord, kmeans, pack, rotation, search, AddError,
    ConstructError, TurboQuantIndex, BLOCK,
};

const TVDM_MAGIC: &[u8; 4] = b"TVDM";
const TVDM_VERSION: u8 = 3;
/// Header flag bit: the file carries a full-precision `vectors` section.
const FLAG_HAS_VECTORS: u8 = 1;
/// Header bytes before the first section; also the section alignment.
const SECTION_ALIGN: usize = 64;
const PARTITION_RECORD_BYTES: usize = 24;
/// Closure assignment writes at most this many copies of a vector
/// (1 primary + up to `MAX_ASSIGNMENT_COPIES - 1` replicas) — SPANN's cap.
pub const MAX_ASSIGNMENT_COPIES: usize = 8;
/// Default exact-rescore depth as a multiple of `k` when the index stores
/// full-precision vectors and [`SearchOptions::rescore_k`] is `None`.
pub(crate) const DEFAULT_RESCORE_MULTIPLIER: usize = 4;

/// Partitioning only engages once the corpus is at least this many targets
/// big — below that a single flat segment is both faster and exact.
pub(crate) const MIN_PARTITIONS: usize = 2;
/// Split a partition once it exceeds `SPLIT_FACTOR * target`.
pub(crate) const SPLIT_FACTOR: usize = 2;
/// Dissolve a partition once it falls below `target / MERGE_DIVISOR`.
pub(crate) const MERGE_DIVISOR: usize = 4;
/// Lloyd iterations for bootstrap clustering and 2-means splits.
pub(crate) const KMEANS_ITERATIONS: usize = 8;
pub(crate) const KMEANS_SEED: u64 = 0x7459_0001;
/// Bootstrap k-means runs on at most this many decoded vectors; assignment
/// of the full corpus then streams in chunks of the same size.
pub(crate) const CLUSTERING_CHUNK: usize = 65_536;
/// Default probe count: `max(4, nlist / 8)`, clamped to nlist.
pub(crate) const DEFAULT_NPROBE_MIN: usize = 4;
pub(crate) const DEFAULT_NPROBE_DIVISOR: usize = 8;
/// Default probe cap when `probe_epsilon` is set and `nprobe` is not:
/// `nlist / 2`. The epsilon rule alone is unbounded for out-of-distribution
/// queries (a query far from every centroid has near-tied distances to all
/// of them, so the bound admits the whole index); the cap keeps such
/// queries from degenerating into full scans while leaving in-distribution
/// boundary queries unconstrained in practice.
pub(crate) const DEFAULT_EPSILON_NPROBE_DIVISOR: usize = 2;
/// Bootstrap / re-clustering sample sizing: at least this many sample
/// vectors per centroid (k-means cannot estimate `k` centers from a sample
/// of comparable size), within [CLUSTERING_CHUNK, MAX_BOOTSTRAP_SAMPLE].
pub(crate) const MIN_SAMPLES_PER_CENTROID: usize = 64;
/// Hard cap on the bootstrap sample (memory bound: the sample is
/// materialized as f32, `cap * dim * 4` bytes transiently).
pub(crate) const MAX_BOOTSTRAP_SAMPLE: usize = 524_288;
/// Of the bootstrap sample, every 4th vector is held out: candidates are
/// fit on the rest and the re-clustering acceptance test compares current
/// vs candidate distortion on the held-out vectors only. Evaluating a
/// candidate on its own training sample is biased — as `k` approaches the
/// sample size a candidate overfits toward zero distortion and would be
/// adopted spuriously.
pub(crate) const HOLDOUT_STRIDE: usize = 4;
/// LIRE maintenance scope: members of a changed partition and of its
/// this-many nearest neighbor partitions get their assignment re-checked.
pub(crate) const NEIGHBOR_PARTITIONS: usize = 8;
/// Re-clustering acceptance test: every compaction runs a cheap sample
/// k-means (best of [`REBOOTSTRAP_CANDIDATES`] random inits); the fresh
/// clustering replaces the incrementally-maintained one only when its
/// sample distortion is below this fraction of the current structure's.
/// Local LIRE repairs handle steady-state churn; this is the escape hatch
/// for distribution drift that local repair cannot unwind (incremental
/// Lloyd converges to the nearest local optimum, which under heavy drift
/// is a bad one). The margin is small because within-cluster variance
/// dominates the objective — structural incoherence only moves it by
/// ~10-15% — and adoption is self-limiting: once adopted, the next
/// candidate cannot beat the structure by the margin again.
pub(crate) const REBOOTSTRAP_DISTORTION_RATIO: f32 = 0.95;
pub(crate) const REBOOTSTRAP_CANDIDATES: usize = 3;

#[cfg(target_arch = "x86_64")]
const LAYOUT_TAG: u8 = 1;
#[cfg(not(target_arch = "x86_64"))]
const LAYOUT_TAG: u8 = 0;

pub(crate) fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) / align * align
}

fn invalid_data(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Byte offsets and lengths of every section, derived from the header
/// fields. Shared by the writer and the reader so they cannot drift.
struct SectionLayout {
    partition_table: (usize, usize),
    centroids: (usize, usize),
    codes: (usize, usize),
    scales: (usize, usize),
    tqplus_shift: (usize, usize),
    tqplus_scale: (usize, usize),
    slot_to_id: (usize, usize),
    sorted_ids: (usize, usize),
    sorted_slots: (usize, usize),
    replica_flags: (usize, usize),
    vectors: (usize, usize),
    total_len: usize,
}

#[allow(clippy::too_many_arguments)]
fn section_layout(
    bit_width: usize,
    dim: usize,
    n_slots: usize,
    n_calib: usize,
    nlist: usize,
    total_blocks: usize,
    has_vectors: bool,
) -> SectionLayout {
    let block_bytes = if dim > 0 {
        pack::n_byte_groups(bit_width, dim) * BLOCK
    } else {
        0
    };

    let mut offset = SECTION_ALIGN; // header occupies the first 64 bytes
    let mut section = |len: usize| {
        let start = offset;
        offset = align_up(start + len, SECTION_ALIGN);
        (start, len)
    };

    SectionLayout {
        partition_table: section(PARTITION_RECORD_BYTES * nlist),
        centroids: section(4 * nlist * dim),
        codes: section(total_blocks * block_bytes),
        scales: section(4 * n_slots),
        tqplus_shift: section(4 * n_calib),
        tqplus_scale: section(4 * n_calib),
        slot_to_id: section(8 * n_slots),
        sorted_ids: section(8 * n_slots),
        sorted_slots: section(8 * n_slots),
        replica_flags: section((n_slots + 7) / 8),
        vectors: section(if has_vectors { 4 * n_slots * dim } else { 0 }),
        total_len: offset,
    }
}

#[derive(Clone, Copy)]
struct PartitionMeta {
    n: usize,
    slot_base: usize,
    /// Byte offset of this partition's blocked segment within the codes
    /// section.
    codes_offset: usize,
}

impl PartitionMeta {
    fn n_blocks(&self) -> usize {
        (self.n + BLOCK - 1) / BLOCK
    }
}

/// The immutable mmap-backed part of a [`DiskIndex`].
struct BaseSegment {
    mmap: Mmap,
    bit_width: usize,
    /// 0 means the file was written from a lazy index that never committed
    /// a dim (and then `n == 0`).
    dim: usize,
    /// Physical rows (primaries plus replicas).
    n: usize,
    /// Distinct external ids. Equals `n` without replication.
    n_unique: usize,
    n_calib: usize,
    /// Per-partition metadata, in slot order. An unpartitioned file has
    /// exactly one partition covering everything.
    partitions: Vec<PartitionMeta>,
    /// `target_partition_size` recorded in the file; 0 = unpartitioned.
    file_partition_target: usize,
    /// Closure-assignment epsilon recorded in the file; `None` = off.
    file_replica_epsilon: Option<f32>,
    /// The file carries a full-precision `vectors` section.
    has_vectors: bool,
    /// Primary (non-replica) rows per partition; sums to `n_unique`.
    primary_counts: Vec<usize>,
    layout: SectionLayout,
}

impl BaseSegment {
    fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        // SAFETY: we map the file read-only and never resize or mutate it
        // through this mapping. Truncation of the backing file by another
        // process while mapped is undefined behavior on POSIX — same
        // contract as every mmap-backed index format; `write` replaces the
        // file via rename (new inode) rather than truncating in place.
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < SECTION_ALIGN {
            return Err(invalid_data("not a .tvdm file: too short".to_string()));
        }
        if &mmap[0..4] != TVDM_MAGIC {
            return Err(invalid_data("not a .tvdm file: wrong magic".to_string()));
        }
        let version = mmap[4];
        if version != TVDM_VERSION {
            return Err(invalid_data(format!(
                "unsupported .tvdm format version: {version} (this build supports {TVDM_VERSION})",
            )));
        }
        let bit_width = mmap[5] as usize;
        if !(2..=4).contains(&bit_width) {
            return Err(invalid_data(format!(
                "invalid .tvdm bit_width: {bit_width} (must be 2, 3, or 4)",
            )));
        }
        let layout_tag = mmap[6];
        if layout_tag != LAYOUT_TAG {
            return Err(invalid_data(format!(
                ".tvdm file uses blocked-code layout {layout_tag}, but this \
                 architecture expects layout {LAYOUT_TAG} (the blocked layout is \
                 architecture-flavored). Rebuild the index on this machine from \
                 the source vectors.",
            )));
        }
        let flags = mmap[7];
        if flags & !FLAG_HAS_VECTORS != 0 {
            return Err(invalid_data(format!(
                "corrupt .tvdm header: unknown flags {flags:#04x}",
            )));
        }
        let has_vectors = flags & FLAG_HAS_VECTORS != 0;
        let dim = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;
        let n_calib = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        let n = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let nlist = u32::from_le_bytes(mmap[24..28].try_into().unwrap()) as usize;
        let file_partition_target = u32::from_le_bytes(mmap[28..32].try_into().unwrap()) as usize;
        let n_unique = u64::from_le_bytes(mmap[32..40].try_into().unwrap()) as usize;
        let replica_epsilon_raw = f32::from_le_bytes(mmap[40..44].try_into().unwrap());
        if !replica_epsilon_raw.is_finite() || replica_epsilon_raw < 0.0 {
            return Err(invalid_data(format!(
                "corrupt .tvdm header: replica_epsilon = {replica_epsilon_raw}",
            )));
        }
        let file_replica_epsilon = if replica_epsilon_raw > 0.0 {
            Some(replica_epsilon_raw)
        } else {
            None
        };
        if n_unique > n {
            return Err(invalid_data(format!(
                "corrupt .tvdm header: n_unique={n_unique} exceeds n_slots={n}",
            )));
        }

        if dim == 0 {
            if n != 0 || n_calib != 0 {
                return Err(invalid_data(format!(
                    "corrupt .tvdm header: dim=0 (lazy) with n_vectors={n}, n_calib={n_calib}",
                )));
            }
        } else {
            if dim % 8 != 0 {
                return Err(invalid_data(format!(
                    "invalid .tvdm dim: {dim} (must be a positive multiple of 8)",
                )));
            }
            let expected_calib = if n > 0 { dim } else { 0 };
            if n_calib != expected_calib {
                return Err(invalid_data(format!(
                    "corrupt .tvdm header: n_calib={n_calib}, expected {expected_calib} \
                     for n_vectors={n}",
                )));
            }
        }
        if nlist == 0 {
            return Err(invalid_data(format!("corrupt .tvdm header: nlist={nlist}")));
        }

        // Parse the partition table.
        let block_bytes = if dim > 0 {
            pack::n_byte_groups(bit_width, dim) * BLOCK
        } else {
            0
        };
        let table_offset = SECTION_ALIGN;
        if mmap.len() < table_offset + PARTITION_RECORD_BYTES * nlist {
            return Err(invalid_data(
                "corrupt .tvdm file: truncated partition table".to_string(),
            ));
        }
        let partitions: Vec<PartitionMeta> = (0..nlist)
            .map(|p| {
                let record = table_offset + p * PARTITION_RECORD_BYTES;
                PartitionMeta {
                    n: u64::from_le_bytes(mmap[record..record + 8].try_into().unwrap()) as usize,
                    slot_base: u64::from_le_bytes(mmap[record + 8..record + 16].try_into().unwrap())
                        as usize,
                    codes_offset: u64::from_le_bytes(
                        mmap[record + 16..record + 24].try_into().unwrap(),
                    ) as usize,
                }
            })
            .collect();

        // Structural validation: partitions tile [0, n) in slot order and
        // their code segments tile the codes section.
        let mut expected_slot_base = 0usize;
        let mut expected_codes_offset = 0usize;
        for partition in &partitions {
            if partition.slot_base != expected_slot_base
                || partition.codes_offset != expected_codes_offset
            {
                return Err(invalid_data(
                    "corrupt .tvdm file: partition table is not contiguous".to_string(),
                ));
            }
            expected_slot_base += partition.n;
            expected_codes_offset += partition.n_blocks() * block_bytes;
        }
        if expected_slot_base != n {
            return Err(invalid_data(format!(
                "corrupt .tvdm file: partition sizes sum to {expected_slot_base}, expected {n}",
            )));
        }
        let total_blocks = expected_codes_offset / block_bytes.max(1);

        let layout = section_layout(bit_width, dim, n, n_calib, nlist, total_blocks, has_vectors);
        if mmap.len() != layout.total_len {
            return Err(invalid_data(format!(
                "corrupt .tvdm file: length {} does not match expected {} \
                 for header (bit_width={bit_width}, dim={dim}, n_slots={n}, nlist={nlist})",
                mmap.len(),
                layout.total_len,
            )));
        }

        let mut segment = Self {
            mmap,
            bit_width,
            dim,
            n,
            n_unique,
            n_calib,
            partitions,
            file_partition_target,
            file_replica_epsilon,
            has_vectors,
            primary_counts: Vec::new(),
            layout,
        };

        // sorted_ids must be non-decreasing (it's the binary-search
        // membership structure), and the number of distinct ids must match
        // the header's n_unique. Without replication this degenerates to
        // the old strictly-increasing uniqueness invariant.
        let sorted_ids = segment.sorted_ids();
        if sorted_ids.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(invalid_data(
                "corrupt .tvdm file: sorted_ids section is not sorted".to_string(),
            ));
        }
        let distinct = if sorted_ids.is_empty() {
            0
        } else {
            1 + sorted_ids
                .windows(2)
                .filter(|pair| pair[0] < pair[1])
                .count()
        };
        if distinct != n_unique {
            return Err(invalid_data(format!(
                "corrupt .tvdm file: sorted_ids has {distinct} distinct ids, \
                 header says n_unique={n_unique}",
            )));
        }
        if segment.sorted_slots().iter().any(|&slot| slot >= n as u64) {
            return Err(invalid_data(
                "corrupt .tvdm file: sorted_slots entry out of range".to_string(),
            ));
        }

        // Replica bitmap: primaries (zero bits) must number n_unique, and
        // per-partition primary counts feed the maintenance bookkeeping.
        let mut primary_counts = vec![0usize; segment.partitions.len()];
        for (p, meta) in segment.partitions.iter().enumerate() {
            primary_counts[p] = (meta.slot_base..meta.slot_base + meta.n)
                .filter(|&slot| !segment.is_replica(slot))
                .count();
        }
        if primary_counts.iter().sum::<usize>() != n_unique {
            return Err(invalid_data(format!(
                "corrupt .tvdm file: replica bitmap marks {} primaries, \
                 header says n_unique={n_unique}",
                primary_counts.iter().sum::<usize>(),
            )));
        }
        segment.primary_counts = primary_counts;

        Ok(segment)
    }

    fn bytes(&self, section: (usize, usize)) -> &[u8] {
        &self.mmap[section.0..section.0 + section.1]
    }

    fn f32s(&self, section: (usize, usize)) -> &[f32] {
        let bytes = self.bytes(section);
        if bytes.is_empty() {
            return &[];
        }
        debug_assert_eq!(bytes.as_ptr() as usize % std::mem::align_of::<f32>(), 0);
        // SAFETY: the section offset is 64-byte aligned within a
        // page-aligned mapping, the length is a multiple of 4 by
        // construction, f32 has no invalid bit patterns, and the borrow is
        // tied to &self (the mapping outlives the slice).
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4) }
    }

    fn u64s(&self, section: (usize, usize)) -> &[u64] {
        let bytes = self.bytes(section);
        if bytes.is_empty() {
            return &[];
        }
        debug_assert_eq!(bytes.as_ptr() as usize % std::mem::align_of::<u64>(), 0);
        // SAFETY: same argument as `f32s`, with 8-byte units.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u64>(), bytes.len() / 8) }
    }

    fn nlist(&self) -> usize {
        self.partitions.len()
    }

    fn block_bytes(&self) -> usize {
        pack::n_byte_groups(self.bit_width, self.dim) * BLOCK
    }

    /// Partition centroids, `nlist * dim` row-major. Empty for v1 files.
    fn partition_centroids(&self) -> &[f32] {
        self.f32s(self.layout.centroids)
    }

    /// The blocked code segment of partition `p`.
    /// Ask the kernel to start reading every probed partition's code bytes
    /// NOW, before the scan loop touches them one at a time.
    ///
    /// IVF's structural advantage over a graph is that routing yields every
    /// address up front -- there is no serial hop chain. The scan loop did not
    /// use it: it walked partitions sequentially and faulted each one's pages
    /// on demand, so a query whose pages were not resident paid one full I/O
    /// round trip per partition at queue depth 1 (measured p95 of 622 ms under
    /// memory pressure). Advising the whole probed set first lets the device
    /// see all of the reads at once.
    ///
    /// Advisory only: `MADV_WILLNEED` cannot fail the query, only fail to
    /// help, so errors are deliberately discarded.
    fn prefetch_partitions(&self, partition_ids: &[u32]) {
        if partition_ids.len() < 2 {
            return; // one partition has nothing to overlap with
        }
        let codes = self.layout.codes;
        let block_bytes = self.block_bytes();
        for &p in partition_ids {
            let meta = match self.partitions.get(p as usize) {
                Some(m) if m.n > 0 => m,
                _ => continue,
            };
            let start = codes.0 + meta.codes_offset;
            let len = meta.n_blocks() * block_bytes;
            if start + len <= self.mmap.len() {
                let _ = self.mmap.advise_range(Advice::WillNeed, start, len);
            }
        }
    }

    fn partition_codes(&self, p: usize) -> &[u8] {
        let meta = &self.partitions[p];
        let codes = self.bytes(self.layout.codes);
        let len = meta.n_blocks() * self.block_bytes();
        &codes[meta.codes_offset..meta.codes_offset + len]
    }

    /// The scales of partition `p`, locally indexed.
    fn partition_scales(&self, p: usize) -> &[f32] {
        let meta = &self.partitions[p];
        &self.scales()[meta.slot_base..meta.slot_base + meta.n]
    }

    fn scales(&self) -> &[f32] {
        self.f32s(self.layout.scales)
    }

    fn tqplus_shift(&self) -> &[f32] {
        self.f32s(self.layout.tqplus_shift)
    }

    fn tqplus_scale(&self) -> &[f32] {
        self.f32s(self.layout.tqplus_scale)
    }

    fn slot_to_id(&self) -> &[u64] {
        self.u64s(self.layout.slot_to_id)
    }

    fn sorted_ids(&self) -> &[u64] {
        self.u64s(self.layout.sorted_ids)
    }

    fn sorted_slots(&self) -> &[u64] {
        self.u64s(self.layout.sorted_slots)
    }

    fn replica_flags(&self) -> &[u8] {
        self.bytes(self.layout.replica_flags)
    }

    /// True if `slot` holds a closure-assignment replica (a duplicate copy
    /// of a vector whose primary lives in another partition).
    fn is_replica(&self, slot: usize) -> bool {
        self.replica_flags()[slot / 8] & (1 << (slot % 8)) != 0
    }

    /// Full-precision vectors, `n * dim` row-major in slot order. Empty
    /// unless the file was written with `store_vectors`.
    fn vectors(&self) -> &[f32] {
        self.f32s(self.layout.vectors)
    }

    /// The full-precision vector stored at `slot`.
    fn vector_row(&self, slot: usize) -> &[f32] {
        &self.vectors()[slot * self.dim..(slot + 1) * self.dim]
    }

    /// A slot of `id` in this segment, via binary search over the sorted id
    /// table. O(log n) page touches, no resident lookup structure. With
    /// replication an id may occupy several slots holding identical data;
    /// any of them answers membership and vector lookups.
    fn slot_of(&self, id: u64) -> Option<usize> {
        let sorted_ids = self.sorted_ids();
        let position = sorted_ids.binary_search(&id).ok()?;
        Some(self.sorted_slots()[position] as usize)
    }
}

/// Streams group-byte rows into the writer as blocked code blocks,
/// accumulating [`BLOCK`] rows at a time so compaction never materializes
/// a second full copy of the codes section.
pub(crate) struct BlockStreamer<'w, W: Write> {
    writer: &'w mut W,
    n_byte_groups: usize,
    row_accumulator: Vec<u8>,
    rows_accumulated: usize,
    block_out: Vec<u8>,
    bytes_written: usize,
}

impl<'w, W: Write> BlockStreamer<'w, W> {
    pub(crate) fn new(writer: &'w mut W, n_byte_groups: usize) -> Self {
        Self {
            writer,
            n_byte_groups,
            row_accumulator: vec![0u8; BLOCK * n_byte_groups],
            rows_accumulated: 0,
            block_out: vec![0u8; n_byte_groups * BLOCK],
            bytes_written: 0,
        }
    }

    pub(crate) fn push_row(&mut self, row: &[u8]) -> io::Result<()> {
        debug_assert_eq!(row.len(), self.n_byte_groups);
        let start = self.rows_accumulated * self.n_byte_groups;
        self.row_accumulator[start..start + self.n_byte_groups].copy_from_slice(row);
        self.rows_accumulated += 1;
        if self.rows_accumulated == BLOCK {
            self.flush_block()?;
        }
        Ok(())
    }

    fn flush_block(&mut self) -> io::Result<()> {
        if self.rows_accumulated == 0 {
            return Ok(());
        }
        pack::pack_block_rows(
            &self.row_accumulator[..self.rows_accumulated * self.n_byte_groups],
            self.rows_accumulated,
            self.n_byte_groups,
            &mut self.block_out,
        );
        self.writer.write_all(&self.block_out)?;
        self.bytes_written += self.block_out.len();
        self.rows_accumulated = 0;
        Ok(())
    }

    pub(crate) fn finish(mut self) -> io::Result<usize> {
        self.flush_block()?;
        Ok(self.bytes_written)
    }
}

/// All live rows of an index gathered for compaction (primary copies only —
/// replicas are derived data, recomputed at every write): per-row group
/// bytes, scale, external id, the partition each base row came from
/// (`u32::MAX` marks delta rows, which have no partition yet), and the base
/// slot the row was gathered from (`u64::MAX` marks delta rows), so the
/// full-precision vector can be streamed from the old mapping at emission.
struct GatheredRows {
    rows: Vec<u8>,
    scales: Vec<f32>,
    ids: Vec<u64>,
    source_partition: Vec<u32>,
    base_slot: Vec<u64>,
    n_byte_groups: usize,
}

const NO_PARTITION: u32 = u32::MAX;
const NO_SLOT: u64 = u64::MAX;

impl GatheredRows {
    fn len(&self) -> usize {
        self.scales.len()
    }

    fn row(&self, i: usize) -> &[u8] {
        &self.rows[i * self.n_byte_groups..(i + 1) * self.n_byte_groups]
    }

    /// Decode rows `indices` into approximate original vectors (flat
    /// `indices.len() * dim`).
    #[allow(clippy::too_many_arguments)]
    fn decode_rows(
        &self,
        indices: &[usize],
        dim: usize,
        bit_width: usize,
        rotation: &crate::rotation::Rotation,
        codebook_centroids: &[f32],
        tqplus_shift: &[f32],
        tqplus_scale: &[f32],
    ) -> Vec<f32> {
        let mut group_rows = Vec::with_capacity(indices.len() * self.n_byte_groups);
        let mut scales = Vec::with_capacity(indices.len());
        for &i in indices {
            group_rows.extend_from_slice(self.row(i));
            scales.push(self.scales[i]);
        }
        let packed = pack::packed_from_group_bytes(&group_rows, indices.len(), bit_width, dim);
        decode::decode(
            &packed,
            &scales,
            indices.len(),
            dim,
            bit_width,
            rotation,
            codebook_centroids,
            tqplus_shift,
            tqplus_scale,
        )
    }
}

/// Clustering-time view of the gathered rows as vectors: exact
/// full-precision rows when the index stores them, decoded approximations
/// otherwise. All maintenance (bootstrap, LIRE, split/merge, replication)
/// goes through this, so a vector-storing index clusters on true geometry.
struct RowSource<'a> {
    gathered: &'a GatheredRows,
    index: &'a DiskIndex,
    dim: usize,
    tqplus_shift: &'a [f32],
    tqplus_scale: &'a [f32],
}

impl RowSource<'_> {
    /// The vectors of `indices` (flat `indices.len() * dim`).
    fn vectors(&self, indices: &[usize]) -> Vec<f32> {
        if !self.index.store_vectors {
            return self.gathered.decode_rows(
                indices,
                self.dim,
                self.index.bit_width,
                self.index.rotation_for(self.dim),
                self.index.codebook_centroids_for(self.dim),
                self.tqplus_shift,
                self.tqplus_scale,
            );
        }
        let mut out = Vec::with_capacity(indices.len() * self.dim);
        for &i in indices {
            match self.gathered.base_slot[i] {
                NO_SLOT => {
                    let id = self.gathered.ids[i];
                    let vector = self
                        .index
                        .delta_originals
                        .get(&id)
                        .expect("store_vectors invariant: every delta row has an original");
                    out.extend_from_slice(vector);
                }
                slot => {
                    let base = self
                        .index
                        .base
                        .as_ref()
                        .expect("gathered base row implies a base segment");
                    out.extend_from_slice(base.vector_row(slot as usize));
                }
            }
        }
        out
    }
}

/// Disk-primary, id-addressed TurboQuant index. See the module docs for the
/// storage model and file format.
pub struct DiskIndex {
    bit_width: usize,
    /// Backing file of `base`; `None` until the first `write` or `open`.
    path: Option<PathBuf>,
    base: Option<BaseSegment>,
    /// Vectors added since the last write. Encoded with the base segment's
    /// TQ+ calibration (inherited at open/write) so scores are comparable.
    delta: IdMapIndex,
    /// External ids removed from the base segment since the last write.
    /// Every entry is present in `base` and absent from `delta`'s live set
    /// unless the id was re-added after removal (then `delta` holds the
    /// live copy and the tombstone keeps hiding the stale base copy).
    tombstones: HashSet<u64>,
    /// Target partition size; `None` = flat (exact) single-segment index.
    partition_target: Option<usize>,
    /// Closure-assignment epsilon; `None` = one copy per vector.
    replica_epsilon: Option<f32>,
    /// Apply SPANN's RNG pruning rule when choosing replica targets. True
    /// reproduces the original behaviour; false keeps every partition inside
    /// the `(1 + epsilon)` bound, which is what SPANN's reported replica
    /// factors look like on their data.
    replica_prune: bool,
    /// Keep full-precision vectors (in RAM for the delta, in the file's
    /// `vectors` section for the base). Fixed for the life of the index:
    /// either every row has its original or none does.
    store_vectors: bool,
    /// Originals of the delta's live vectors, keyed by external id.
    /// Populated only when `store_vectors`.
    delta_originals: HashMap<u64, Box<[f32]>>,
    rotation: OnceLock<rotation::Rotation>,
    centroids: OnceLock<Vec<f32>>,
}

/// Per-search tuning knobs for [`DiskIndex::search_with_options`]. The
/// `Default` value reproduces [`DiskIndex::search`]'s behavior.
#[derive(Clone, Copy, Debug, Default)]
pub struct SearchOptions {
    /// Maximum partitions each query scans on a partitioned base.
    /// `None` = `max(4, nlist / 8)`, or `max(4, nlist / 2)` when
    /// `probe_epsilon` is set (the distance bound is the primary rule and
    /// the cap is the guard against out-of-distribution queries, whose
    /// near-tied centroid distances would otherwise admit every partition).
    pub nprobe: Option<usize>,
    /// Distance-bounded adaptive probing: scan every partition whose
    /// centroid distance is within `(1 + epsilon)` of the query's nearest
    /// centroid distance, up to the `nprobe` cap. Queries near partition
    /// boundaries automatically probe more partitions; confident queries
    /// probe fewer. `None` = fixed `nprobe` routing.
    pub probe_epsilon: Option<f32>,
    /// Exact-rescore depth: the top `rescore_k` quantized candidates are
    /// re-ranked by exact f32 inner product against the stored originals.
    /// `None` = `4 * k` when the index stores vectors, off otherwise;
    /// `Some(0)` = off. Values below `k` are raised to `k`. Requires
    /// `store_vectors`.
    pub rescore_k: Option<usize>,
}

impl DiskIndex {
    /// Construct an empty index with no backing file yet. All vectors live
    /// in the in-RAM delta until the first [`Self::write`].
    ///
    /// `dim = None` defers committing a dimensionality until the first add,
    /// matching [`IdMapIndex::new_lazy`].
    pub fn new(dim: Option<usize>, bit_width: usize) -> Result<Self, ConstructError> {
        let delta = match dim {
            Some(d) => IdMapIndex::new(d, bit_width)?,
            None => IdMapIndex::new_lazy(bit_width)?,
        };
        Ok(Self {
            bit_width,
            path: None,
            base: None,
            delta,
            tombstones: HashSet::new(),
            partition_target: None,
            replica_epsilon: None,
            replica_prune: true,
            store_vectors: false,
            delta_originals: HashMap::new(),
            rotation: OnceLock::new(),
            centroids: OnceLock::new(),
        })
    }

    /// Open a `.tvdm` file previously produced by [`Self::write`]. The file
    /// is memory-mapped; nothing beyond the header, partition table and the
    /// sorted id table is read eagerly. Partitioning configuration recorded
    /// in the file carries over.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let base = BaseSegment::open(path)?;
        let delta = Self::fresh_delta(&base)?;
        let partition_target = match base.file_partition_target {
            0 => None,
            target => Some(target),
        };
        let replica_epsilon = base.file_replica_epsilon;
        let store_vectors = base.has_vectors;
        Ok(Self {
            bit_width: base.bit_width,
            path: Some(path.to_path_buf()),
            base: Some(base),
            delta,
            tombstones: HashSet::new(),
            partition_target,
            replica_epsilon,
            replica_prune: true,
            store_vectors,
            delta_originals: HashMap::new(),
            rotation: OnceLock::new(),
            centroids: OnceLock::new(),
        })
    }

    /// Enable (`Some(target)`) or disable (`None`) partitioning. Takes
    /// effect at the next [`Self::write`]: enabling re-clusters the corpus
    /// into partitions of roughly `target` vectors; disabling collapses it
    /// back to one flat segment. Search behavior follows the *file*: an
    /// open partitioned base is probed regardless of this setting.
    pub fn set_partitioning(&mut self, target_partition_size: Option<usize>) {
        assert!(
            target_partition_size != Some(0),
            "target_partition_size must be positive",
        );
        self.partition_target = target_partition_size;
    }

    /// Current partitioning target, if enabled.
    pub fn partition_target(&self) -> Option<usize> {
        self.partition_target
    }

    /// Enable (`Some(epsilon)`) or disable (`None`) boundary
    /// multi-assignment. Takes effect at the next [`Self::write`] on a
    /// partitioned base: each vector is also stored in every partition
    /// whose centroid is within `(1 + epsilon)` of its nearest centroid's
    /// distance (RNG-rule pruned, at most [`MAX_ASSIGNMENT_COPIES`] copies
    /// total), so boundary vectors are findable from adjacent partitions
    /// at small probe counts. Costs the replication factor in file size;
    /// no effect on a flat base. Persisted in the file.
    ///
    /// # Panics
    ///
    /// Panics if `epsilon` is not finite and positive.
    pub fn set_replication(&mut self, epsilon: Option<f32>) {
        if let Some(e) = epsilon {
            assert!(
                e.is_finite() && e > 0.0,
                "replica epsilon must be finite and positive, got {e}",
            );
        }
        self.replica_epsilon = epsilon;
    }

    /// Current closure-assignment epsilon, if enabled.
    pub fn replica_epsilon(&self) -> Option<f32> {
        self.replica_epsilon
    }

    /// Apply SPANN's RNG pruning rule when selecting replica targets
    /// (default true). On embedding corpora whose centroids sit in a narrow
    /// cone the rule can reject essentially every candidate, so measuring
    /// replication's value at all requires being able to turn it off.
    pub fn set_replica_prune(&mut self, prune: bool) {
        self.replica_prune = prune;
    }

    pub fn replica_prune(&self) -> bool {
        self.replica_prune
    }

    /// Keep full-precision vectors alongside the quantized codes (the
    /// delta's in RAM, the base's in the file's `vectors` section). Enables
    /// exact rescoring ([`SearchOptions::rescore_k`], on by default once
    /// set) and [`Self::get_vector`], and clustering maintenance uses the
    /// exact vectors instead of decoded approximations. Costs `4 * dim`
    /// bytes per stored row of file size; resident memory is unaffected
    /// (rescoring touches a handful of mapped pages per query).
    ///
    /// The setting is fixed for the life of the index — either every row
    /// has its original or none does — so it must be chosen while the
    /// index is empty; opening a file adopts what the file records.
    ///
    /// # Panics
    ///
    /// Panics if the index already holds any vectors (live or tombstoned).
    pub fn set_store_vectors(&mut self, store_vectors: bool) {
        if store_vectors == self.store_vectors {
            return;
        }
        assert!(
            self.base.is_none() && self.delta.is_empty() && self.tombstones.is_empty(),
            "store_vectors must be set while the index is empty",
        );
        self.store_vectors = store_vectors;
    }

    /// True if the index keeps full-precision vectors for exact rescoring
    /// and [`Self::get_vector`].
    pub fn stores_vectors(&self) -> bool {
        self.store_vectors
    }

    /// Number of partitions in the base segment (1 = flat).
    pub fn nlist(&self) -> usize {
        self.base.as_ref().map_or(1, BaseSegment::nlist)
    }

    /// Empty delta for `base`: dim-committed when the base is, and carrying
    /// the base's TQ+ calibration when the base has vectors, so new adds
    /// encode into the same calibrated coordinate system.
    fn fresh_delta(base: &BaseSegment) -> io::Result<IdMapIndex> {
        let construct_err =
            |e: ConstructError| invalid_data(format!("invalid .tvdm parameters: {e}"));
        if base.dim == 0 {
            return IdMapIndex::new_lazy(base.bit_width).map_err(construct_err);
        }
        if base.n_calib > 0 {
            let inner = TurboQuantIndex::from_parts(
                Some(base.dim),
                base.bit_width,
                0,
                Vec::new(),
                Vec::new(),
                base.tqplus_shift().to_vec(),
                base.tqplus_scale().to_vec(),
            );
            return Ok(IdMapIndex::from_inner(inner.map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, e.to_string())
            })?));
        }
        IdMapIndex::new(base.dim, base.bit_width).map_err(construct_err)
    }

    /// Number of live vectors (distinct base ids minus tombstones, plus
    /// delta). Replicas do not count: they are extra copies, not vectors.
    pub fn len(&self) -> usize {
        let base_unique = self.base.as_ref().map_or(0, |base| base.n_unique);
        base_unique - self.tombstones.len() + self.delta.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn bit_width(&self) -> usize {
        self.bit_width
    }

    /// Vector dimensionality as an [`Option`], where `None` means no dim
    /// has been committed yet (lazy construction, no adds, no dim-committed
    /// base file).
    pub fn dim_opt(&self) -> Option<usize> {
        // The delta is constructed dim-committed whenever the base is, so
        // its dim state is authoritative for the whole index.
        self.delta.dim_opt()
    }

    /// Vector dimensionality, or 0 when uncommitted (matches
    /// [`crate::TurboQuantIndex::dim`] semantics).
    pub fn dim(&self) -> usize {
        self.dim_opt().unwrap_or(0)
    }

    /// Number of physical rows in the mmap-backed base segment, including
    /// tombstoned vectors and closure-assignment replicas. Diagnostic.
    pub fn base_len(&self) -> usize {
        self.base.as_ref().map_or(0, |base| base.n)
    }

    /// Number of closure-assignment replica rows in the base segment
    /// (physical rows minus distinct ids). Diagnostic.
    pub fn base_replica_count(&self) -> usize {
        self.base.as_ref().map_or(0, |base| base.n - base.n_unique)
    }

    /// Number of vectors in the in-RAM delta. Diagnostic.
    pub fn delta_len(&self) -> usize {
        self.delta.len()
    }

    /// Number of base-segment ids hidden by tombstones. Diagnostic.
    pub fn tombstone_count(&self) -> usize {
        self.tombstones.len()
    }

    /// Backing file path, or `None` before the first write/open.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// True if a vector with this external id is live in the index.
    pub fn contains(&self, id: u64) -> bool {
        if self.delta.contains(id) {
            return true;
        }
        if self.tombstones.contains(&id) {
            return false;
        }
        self.base
            .as_ref()
            .is_some_and(|base| base.slot_of(id).is_some())
    }

    /// Add `vectors` of dimensionality `dim` with the given external ids.
    /// Same contract as [`IdMapIndex::add_with_ids_2d`]; new vectors land
    /// in the in-RAM delta until the next [`Self::write`].
    pub fn add_with_ids_2d(
        &mut self,
        vectors: &[f32],
        dim: usize,
        ids: &[u64],
    ) -> Result<(), AddError> {
        // Reject ids that are live in the base segment up front, before the
        // delta mutates (the delta validates its own ids and in-call
        // duplicates before mutating, so failure leaves the index intact).
        if let Some(base) = &self.base {
            for &id in ids {
                if !self.tombstones.contains(&id) && base.slot_of(id).is_some() {
                    return Err(AddError::IdAlreadyPresent(id));
                }
            }
        }
        self.delta.add_with_ids_2d(vectors, dim, ids)?;
        if self.store_vectors {
            for (i, &id) in ids.iter().enumerate() {
                self.delta_originals
                    .insert(id, vectors[i * dim..(i + 1) * dim].into());
            }
        }
        Ok(())
    }

    /// Add with the already-committed dim. Same contract as
    /// [`IdMapIndex::add_with_ids`].
    pub fn add_with_ids(&mut self, vectors: &[f32], ids: &[u64]) -> Result<(), AddError> {
        let dim = self.dim_opt().expect(
            "DiskIndex dim is not set; use add_with_ids_2d(vectors, dim, ids) \
             on the first add or construct with DiskIndex::new(Some(dim), bit_width)",
        );
        self.add_with_ids_2d(vectors, dim, ids)
    }

    /// Remove the vector with the given external id. Returns `true` if the
    /// id was live and is now removed. Base-segment removals are tombstoned
    /// in RAM and physically dropped at the next [`Self::write`].
    pub fn remove(&mut self, id: u64) -> bool {
        if self.delta.remove(id) {
            // A re-added id may also have a tombstoned base copy; the
            // tombstone must stay so the stale base copy remains hidden.
            self.delta_originals.remove(&id);
            return true;
        }
        if self.tombstones.contains(&id) {
            return false;
        }
        let in_base = self
            .base
            .as_ref()
            .is_some_and(|base| base.slot_of(id).is_some());
        if in_base {
            self.tombstones.insert(id);
        }
        in_base
    }

    /// The stored full-precision vector of a live id, or `None` when the
    /// id is not present.
    ///
    /// # Panics
    ///
    /// Panics unless the index stores vectors ([`Self::set_store_vectors`]).
    pub fn get_vector(&self, id: u64) -> Option<Vec<f32>> {
        assert!(
            self.store_vectors,
            "get_vector requires an index built with store_vectors",
        );
        if let Some(vector) = self.delta_originals.get(&id) {
            return Some(vector.to_vec());
        }
        if self.tombstones.contains(&id) {
            return None;
        }
        let base = self.base.as_ref()?;
        base.slot_of(id).map(|slot| base.vector_row(slot).to_vec())
    }

    /// Default probe count for a partitioned base.
    fn default_nprobe(nlist: usize) -> usize {
        (nlist / DEFAULT_NPROBE_DIVISOR)
            .max(DEFAULT_NPROBE_MIN)
            .min(nlist)
    }

    /// Search for the top-`k` nearest external ids for each query with
    /// default options: default probe count, fixed routing, and exact
    /// rescoring at depth `4 * k` when the index stores vectors. See
    /// [`Self::search_with_options`].
    pub fn search(&self, queries: &[f32], k: usize) -> (Vec<f32>, Vec<u64>) {
        self.search_with_options(queries, k, SearchOptions::default())
    }

    /// Search with an explicit probe count and otherwise-default options.
    /// See [`Self::search_with_options`].
    pub fn search_with_nprobe(
        &self,
        queries: &[f32],
        k: usize,
        nprobe: Option<usize>,
    ) -> (Vec<f32>, Vec<u64>) {
        self.search_with_options(
            queries,
            k,
            SearchOptions {
                nprobe,
                ..SearchOptions::default()
            },
        )
    }

    /// Search for the top-`k` nearest external ids for each query.
    ///
    /// Returns `(scores, ids)` flattened row-major with row width
    /// `min(k, len())`. On a flat base the whole segment is scanned in
    /// place (exact over the quantized scores). On a partitioned base each
    /// query scans only its routed partitions — a fixed `nprobe` count
    /// (default `max(4, nlist/8)`), or the distance-bounded set when
    /// [`SearchOptions::probe_epsilon`] is given — which is approximate: a
    /// true neighbor in an unprobed partition is missed. The in-RAM delta
    /// is always scanned exhaustively and merged by score. Duplicate
    /// candidates from replicated rows are merged by id. When rescoring is
    /// active the merged top [`SearchOptions::rescore_k`] candidates are
    /// re-ranked by exact f32 inner product before the final truncation,
    /// and the returned scores for those rows are the exact products.
    ///
    /// # Panics
    ///
    /// Panics if `queries.len()` is not a multiple of the committed dim,
    /// if any query coordinate is non-finite or has magnitude `>= 1e16`
    /// (same contract as [`IdMapIndex::search`]), if
    /// `options.probe_epsilon` is negative or non-finite, or if
    /// `options.rescore_k` is `Some(r > 0)` on an index that does not
    /// store vectors.
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

        // Rescore depth, and how many candidates to pull from every source
        // so the exact re-ranking has real candidates to promote.
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

        // Per-query candidate lists from the base segment(s).
        let base_candidates: Vec<Vec<(f32, u64)>> = match self.base.as_ref() {
            Some(base) if base.n > 0 => {
                let prepared = search::prepare(
                    queries,
                    nq,
                    self.rotation_for(dim),
                    self.codebook_centroids_for(dim),
                    base.tqplus_shift(),
                    base.tqplus_scale(),
                    self.bit_width,
                    dim,
                );
                // Over-fetch by the tombstone count so hidden vectors cannot
                // displace live results.
                let k_base = fetch_k + self.tombstones.len();
                if base.nlist() <= 1 {
                    self.scan_partitions(base, &prepared, &[0], nq, k_base)
                } else {
                    let nprobe_cap = options
                        .nprobe
                        .unwrap_or_else(|| {
                            if options.probe_epsilon.is_some() {
                                (base.nlist() / DEFAULT_EPSILON_NPROBE_DIVISOR)
                                    .max(DEFAULT_NPROBE_MIN)
                            } else {
                                Self::default_nprobe(base.nlist())
                            }
                        })
                        .clamp(1, base.nlist());
                    let routes = route_queries(
                        queries,
                        nq,
                        dim,
                        base.partition_centroids(),
                        base.nlist(),
                        nprobe_cap,
                        options.probe_epsilon,
                    );
                    (0..nq)
                        .map(|qi| {
                            let single = prepared.single(qi);
                            self.scan_partitions(base, &single, &routes[qi], 1, k_base)
                                .pop()
                                .expect("one query yields one candidate list")
                        })
                        .collect()
                }
            }
            _ => vec![Vec::new(); nq],
        };

        // Delta: same calibrated scoring, so scores merge directly.
        let (delta_scores, delta_ids) = if self.delta.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            self.delta.search(queries, fetch_k)
        };
        let k_delta = delta_scores.len() / nq.max(1);

        let mut out_scores = Vec::with_capacity(nq * k_eff);
        let mut out_ids = Vec::with_capacity(nq * k_eff);
        let mut candidates: Vec<(f32, u64)> = Vec::new();
        let mut seen: HashSet<u64> = HashSet::new();
        for (qi, base_list) in base_candidates.into_iter().enumerate() {
            candidates.clear();
            candidates.extend(base_list);
            for j in 0..k_delta {
                candidates.push((delta_scores[qi * k_delta + j], delta_ids[qi * k_delta + j]));
            }
            candidates.sort_unstable_by(|a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            // Replicated rows surface the same id (with the same quantized
            // score) once per probed copy; keep the best occurrence.
            seen.clear();
            candidates.retain(|&(_, id)| seen.insert(id));
            candidates.truncate(fetch_k);

            if rescore_k.is_some() {
                let query = &queries[qi * dim..(qi + 1) * dim];
                for candidate in candidates.iter_mut() {
                    let exact = self.exact_score(candidate.1, query);
                    candidate.0 = exact;
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
            // Keep rows rectangular when routing surfaced fewer than k_eff
            // live candidates (possible with a small nprobe), padding like
            // `search::search` does.
            for _ in candidates.len()..k_eff {
                out_scores.push(f32::NEG_INFINITY);
                out_ids.push(0);
            }
        }
        (out_scores, out_ids)
    }

    /// Exact f32 inner product of `query` with the stored original of a
    /// live `id`. Only called while rescoring, so the original must exist.
    fn exact_score(&self, id: u64, query: &[f32]) -> f32 {
        let vector: &[f32] = if let Some(vector) = self.delta_originals.get(&id) {
            vector
        } else {
            let base = self
                .base
                .as_ref()
                .expect("rescore candidate not in delta implies a base segment");
            let slot = base
                .slot_of(id)
                .expect("rescore candidates are live ids surfaced by the scan");
            base.vector_row(slot)
        };
        query.iter().zip(vector).map(|(&q, &v)| q * v).sum()
    }

    /// Scan the given partitions with prepared queries; returns one
    /// tombstone-filtered candidate list per query.
    fn scan_partitions(
        &self,
        base: &BaseSegment,
        prepared: &search::PreparedQueries,
        partition_ids: &[u32],
        nq: usize,
        k_per_partition: usize,
    ) -> Vec<Vec<(f32, u64)>> {
        base.prefetch_partitions(partition_ids);
        let slot_to_id = base.slot_to_id();
        let mut per_query: Vec<Vec<(f32, u64)>> = vec![Vec::new(); nq];
        for &p in partition_ids {
            let meta = &base.partitions[p as usize];
            if meta.n == 0 {
                continue;
            }
            let k_partition = k_per_partition.min(meta.n);
            let (scores, slots) = search::scan(
                prepared,
                base.partition_codes(p as usize),
                base.partition_scales(p as usize),
                self.bit_width,
                base.dim,
                meta.n,
                meta.n_blocks(),
                k_partition,
                None,
            );
            let width = scores.len().checked_div(nq).unwrap_or(0);
            for qi in 0..nq {
                for j in 0..width {
                    let local_slot = slots[qi * width + j] as usize;
                    let id = slot_to_id[meta.slot_base + local_slot];
                    if self.tombstones.contains(&id) {
                        continue;
                    }
                    per_query[qi].push((scores[qi * width + j], id));
                }
            }
        }
        per_query
    }

    fn rotation_for(&self, dim: usize) -> &rotation::Rotation {
        self.rotation.get_or_init(|| rotation::Rotation::new(dim))
    }

    fn codebook_centroids_for(&self, dim: usize) -> &[f32] {
        self.centroids.get_or_init(|| {
            let (_, centroids) = codebook::codebook(self.bit_width, dim);
            centroids
        })
    }

    /// Eagerly populate the query-side caches (rotation matrix, centroids,
    /// and the delta's blocked layout). Cheap; does not fault in the
    /// mmap-backed codes.
    pub fn prepare(&self) {
        let Some(dim) = self.dim_opt() else { return };
        self.rotation_for(dim);
        self.codebook_centroids_for(dim);
        self.delta.prepare();
    }

    /// Convert a `.tvim` [`IdMapIndex`] file into a `.tvdm` file at `dst`.
    ///
    /// Lossless: the quantized codes, per-vector scales, TQ+ calibration
    /// and external ids carry over unchanged — search results over the
    /// converted file are identical to the source index's. The source is
    /// read into RAM once (conversion-time only); `dst` may equal `src`'s
    /// path to convert in place (the write is via temp file + rename).
    pub fn convert_id_map_file(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> io::Result<()> {
        let delta = IdMapIndex::load(src)?;
        let mut index = Self {
            bit_width: delta.bit_width(),
            path: None,
            base: None,
            delta,
            tombstones: HashSet::new(),
            partition_target: None,
            replica_epsilon: None,
            replica_prune: true,
            store_vectors: false,
            delta_originals: HashMap::new(),
            rotation: OnceLock::new(),
            centroids: OnceLock::new(),
        };
        index.write(dst)
    }

    /// Convert a `.tvdm` file into a `.tvim` [`IdMapIndex`] file at `dst` —
    /// the inverse of [`Self::convert_id_map_file`], equally lossless. The
    /// codes are materialized in RAM once (conversion-time only); `dst` may
    /// equal `src`'s path (the write is via temp file + rename, so the
    /// source mapping is never truncated).
    pub fn convert_to_id_map_file(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> io::Result<()> {
        let dst = dst.as_ref();
        let base = BaseSegment::open(src.as_ref())?;
        let temp_path = temp_sibling(dst);

        if base.dim == 0 {
            crate::io::write_id_map(
                &temp_path,
                base.bit_width,
                0,
                0,
                &[],
                &[],
                &[],
                &[],
                &[],
                &[],
                &[],
            )?;
            return std::fs::rename(&temp_path, dst);
        }

        let n_byte_groups = pack::n_byte_groups(base.bit_width, base.dim);
        let block_bytes = base.block_bytes();
        let slot_to_id = base.slot_to_id();
        let scales = base.scales();
        // Primary copies only: closure-assignment replicas are duplicate
        // rows and an IdMapIndex requires unique ids. The full-precision
        // vectors section (if any) has no .tvim counterpart and is dropped.
        let mut rows = Vec::with_capacity(base.n_unique * n_byte_groups);
        let mut out_ids = Vec::with_capacity(base.n_unique);
        let mut out_scales = Vec::with_capacity(base.n_unique);
        let mut block_rows = vec![0u8; BLOCK * n_byte_groups];
        for p in 0..base.nlist() {
            let meta = base.partitions[p];
            let codes = base.partition_codes(p);
            for block_idx in 0..meta.n_blocks() {
                pack::unpack_block_rows(
                    &codes[block_idx * block_bytes..(block_idx + 1) * block_bytes],
                    n_byte_groups,
                    &mut block_rows,
                );
                let lanes = (meta.n - block_idx * BLOCK).min(BLOCK);
                for lane in 0..lanes {
                    let slot = meta.slot_base + block_idx * BLOCK + lane;
                    if base.is_replica(slot) {
                        continue;
                    }
                    rows.extend_from_slice(
                        &block_rows[lane * n_byte_groups..(lane + 1) * n_byte_groups],
                    );
                    out_ids.push(slot_to_id[slot]);
                    out_scales.push(scales[slot]);
                }
            }
        }
        let packed = pack::packed_from_group_bytes(&rows, base.n_unique, base.bit_width, base.dim);

        let (boundaries, centroids) = codebook::codebook(base.bit_width, base.dim);
        crate::io::write_id_map(
            &temp_path,
            base.bit_width,
            base.dim,
            base.n_unique,
            &packed,
            &boundaries,
            &centroids,
            &out_scales,
            base.tqplus_shift(),
            base.tqplus_scale(),
            &out_ids,
        )?;
        std::fs::rename(&temp_path, dst)
    }

    /// Compact to `path`: stream the base segment (minus tombstones) and
    /// the delta into a fresh `.tvdm` file, atomically replace `path`,
    /// re-map it as the new base, and empty the delta and tombstones. With
    /// partitioning enabled this is also where clustering maintenance
    /// happens (bootstrap, delta assignment, splits, merges).
    pub fn write(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        let temp_path = temp_sibling(path);
        self.write_compacted(&temp_path)?;
        std::fs::rename(&temp_path, path)?;

        let base = BaseSegment::open(path)?;
        self.delta = Self::fresh_delta(&base)?;
        self.delta_originals.clear();
        self.tombstones.clear();
        self.base = Some(base);
        self.path = Some(path.to_path_buf());
        Ok(())
    }

    /// Gather all live rows (base survivors in slot order, then delta) with
    /// their scales, ids, source partitions, and base slots. Replica slots
    /// are skipped: replication is derived data, recomputed at every write
    /// from the final assignments, so each id is gathered exactly once (its
    /// primary copy).
    fn gather_rows(&self, dim: usize) -> GatheredRows {
        let n_byte_groups = pack::n_byte_groups(self.bit_width, dim);
        let n_out = self.len();
        let mut gathered = GatheredRows {
            rows: Vec::with_capacity(n_out * n_byte_groups),
            scales: Vec::with_capacity(n_out),
            ids: Vec::with_capacity(n_out),
            source_partition: Vec::with_capacity(n_out),
            base_slot: Vec::with_capacity(n_out),
            n_byte_groups,
        };

        if let Some(base) = self.base.as_ref().filter(|base| base.n > 0) {
            let block_bytes = base.block_bytes();
            let slot_to_id = base.slot_to_id();
            let scales = base.scales();
            let mut block_rows = vec![0u8; BLOCK * n_byte_groups];
            for p in 0..base.nlist() {
                let meta = base.partitions[p];
                let codes = base.partition_codes(p);
                for block_idx in 0..meta.n_blocks() {
                    pack::unpack_block_rows(
                        &codes[block_idx * block_bytes..(block_idx + 1) * block_bytes],
                        n_byte_groups,
                        &mut block_rows,
                    );
                    let lanes = (meta.n - block_idx * BLOCK).min(BLOCK);
                    for lane in 0..lanes {
                        let slot = meta.slot_base + block_idx * BLOCK + lane;
                        if base.is_replica(slot) {
                            continue;
                        }
                        let id = slot_to_id[slot];
                        if self.tombstones.contains(&id) {
                            continue;
                        }
                        gathered.rows.extend_from_slice(
                            &block_rows[lane * n_byte_groups..(lane + 1) * n_byte_groups],
                        );
                        gathered.scales.push(scales[slot]);
                        gathered.ids.push(id);
                        gathered.source_partition.push(p as u32);
                        gathered.base_slot.push(slot as u64);
                    }
                }
            }
        }

        let delta_n = self.delta.len();
        if delta_n > 0 {
            let delta_rows = pack::group_bytes(
                self.delta.inner().packed_codes(),
                delta_n,
                self.bit_width,
                dim,
            );
            gathered.rows.extend_from_slice(&delta_rows);
            gathered
                .scales
                .extend_from_slice(self.delta.inner().scales());
            gathered
                .ids
                .extend_from_slice(self.delta.slot_to_id_slice());
            gathered
                .source_partition
                .extend(std::iter::repeat(NO_PARTITION).take(delta_n));
            gathered
                .base_slot
                .extend(std::iter::repeat(NO_SLOT).take(delta_n));
        }

        gathered
    }

    /// The TQ+ calibration the stored vectors were encoded with: the base
    /// segment's when committed there, else the delta's first-add fit, else
    /// identity (a first batch too small to fit calibration encodes as
    /// identity — mirrors `TurboQuantIndex::from_parts`).
    fn calibration_for_write(&self, dim: usize) -> (Vec<f32>, Vec<f32>) {
        if let Some(base) = self.base.as_ref().filter(|base| base.n_calib > 0) {
            return (base.tqplus_shift().to_vec(), base.tqplus_scale().to_vec());
        }
        let delta_shift = self.delta.inner().tqplus_shift();
        if !delta_shift.is_empty() {
            return (
                delta_shift.to_vec(),
                self.delta.inner().tqplus_scale().to_vec(),
            );
        }
        (vec![0.0; dim], vec![1.0; dim])
    }

    /// One LIRE maintenance iteration: re-establish the nearest-partition
    /// invariant for members of `changed` partitions and their
    /// [`NEIGHBOR_PARTITIONS`] nearest neighbors (a row moves only if a
    /// different centroid is strictly closer), then refresh the centroid of
    /// every partition whose membership changed to the mean of its decoded
    /// members.
    ///
    /// Work is bounded by the affected partitions' member counts, never the
    /// whole corpus, and exactly one iteration runs per call: convergence
    /// amortizes across compactions (the same argument LIRE makes for its
    /// incremental rebalancing).
    fn lire_maintenance(
        &self,
        rows: &RowSource,
        centroids: &mut [f32],
        assignments: &mut [u32],
        changed: &HashSet<u32>,
    ) {
        let dim = rows.dim;
        let nlist = centroids.len() / dim;
        if nlist <= 1 || changed.is_empty() {
            return;
        }
        let n = rows.gathered.len();
        let decode_rows = |indices: &[usize]| rows.vectors(indices);

        // Affected = changed partitions plus each one's nearest neighbors
        // (by centroid distance).
        let mut affected: HashSet<u32> = changed.clone();
        for &p in changed {
            let center = &centroids[p as usize * dim..(p as usize + 1) * dim];
            let mut ranked: Vec<(f32, u32)> = (0..nlist)
                .filter(|&c| c != p as usize)
                .map(|c| {
                    let other = &centroids[c * dim..(c + 1) * dim];
                    let dist: f32 = center
                        .iter()
                        .zip(other)
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum();
                    (dist, c as u32)
                })
                .collect();
            ranked.sort_unstable_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            affected.extend(ranked.iter().take(NEIGHBOR_PARTITIONS).map(|&(_, c)| c));
        }

        // Reassign: members of affected partitions move to the globally
        // nearest centroid when it is strictly closer than their current
        // one (strictness prevents tie ping-pong across compactions).
        let candidates: Vec<usize> = (0..n)
            .filter(|&i| affected.contains(&assignments[i]))
            .collect();
        let mut membership_changed: HashSet<u32> = changed.clone();
        for chunk in candidates.chunks(CLUSTERING_CHUNK) {
            let decoded = decode_rows(chunk);
            let (best, best_distances) =
                kmeans::assign(&decoded, chunk.len(), dim, centroids, nlist);
            for (offset, &i) in chunk.iter().enumerate() {
                let current = assignments[i];
                let proposed = best[offset];
                if proposed == current {
                    continue;
                }
                let row = &decoded[offset * dim..(offset + 1) * dim];
                let center = &centroids[current as usize * dim..(current as usize + 1) * dim];
                let current_distance: f32 = row
                    .iter()
                    .zip(center)
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                if best_distances[offset] + 1e-6 < current_distance {
                    assignments[i] = proposed;
                    membership_changed.insert(current);
                    membership_changed.insert(proposed);
                }
            }
        }

        // Refresh: centroids of partitions whose membership changed become
        // the mean of their (decoded) members.
        let refresh_members: Vec<usize> = (0..n)
            .filter(|&i| membership_changed.contains(&assignments[i]))
            .collect();
        let mut sums = vec![0.0f64; nlist * dim];
        let mut counts = vec![0usize; nlist];
        for chunk in refresh_members.chunks(CLUSTERING_CHUNK) {
            let decoded = decode_rows(chunk);
            for (offset, &i) in chunk.iter().enumerate() {
                let p = assignments[i] as usize;
                counts[p] += 1;
                for d in 0..dim {
                    sums[p * dim + d] += decoded[offset * dim + d] as f64;
                }
            }
        }
        for &p in &membership_changed {
            let p = p as usize;
            if counts[p] == 0 {
                continue; // emptied; the merge pass will dissolve it
            }
            for d in 0..dim {
                centroids[p * dim + d] = (sums[p * dim + d] / counts[p] as f64) as f32;
            }
        }
    }

    /// Partition assignment for every gathered row, plus the centroid set.
    /// Implements the LIRE maintenance loop; called only when partitioning
    /// is enabled and the corpus is big enough.
    fn assign_partitions(&self, rows: &RowSource, target: usize) -> (Vec<f32>, Vec<u32>) {
        let gathered = rows.gathered;
        let dim = rows.dim;
        let n = gathered.len();
        let decode_rows = |indices: &[usize]| rows.vectors(indices);

        let base_partitioned = self.base.as_ref().is_some_and(|base| base.nlist() > 1);

        let (mut centroids, mut assignments) = if base_partitioned {
            // Incremental: base rows keep their partition; only delta rows
            // are assigned (to the existing centroids).
            let base = self.base.as_ref().expect("base_partitioned implies base");
            let mut centroids = base.partition_centroids().to_vec();
            let nlist = base.nlist();
            let mut assignments: Vec<u32> = gathered.source_partition.clone();
            let unassigned: Vec<usize> =
                (0..n).filter(|&i| assignments[i] == NO_PARTITION).collect();
            let mut delta_recipients: HashSet<u32> = HashSet::new();
            for chunk in unassigned.chunks(CLUSTERING_CHUNK) {
                let decoded = decode_rows(chunk);
                let (chunk_assignments, _) =
                    kmeans::assign(&decoded, chunk.len(), dim, &centroids, nlist);
                for (&i, &a) in chunk.iter().zip(&chunk_assignments) {
                    assignments[i] = a;
                    delta_recipients.insert(a);
                }
            }
            // Tombstoned removals also change membership: partitions whose
            // rows disappeared since the last write need refreshing too.
            // gather_rows dropped them already, so approximate "shrank" by
            // including every source partition that lost at least one
            // primary row (replicas are not gathered, so they are compared
            // against the primary count, not the physical row count).
            if !self.tombstones.is_empty() {
                let mut survivor_counts = vec![0usize; nlist];
                for &p in &gathered.source_partition {
                    if p != NO_PARTITION {
                        survivor_counts[p as usize] += 1;
                    }
                }
                for (p, &primaries) in base.primary_counts.iter().enumerate() {
                    if survivor_counts[p] < primaries {
                        delta_recipients.insert(p as u32);
                    }
                }
            }
            self.lire_maintenance(rows, &mut centroids, &mut assignments, &delta_recipients);

            // Drift escape hatch: candidate re-clustering with an
            // acceptance test on a held-out split of the sample (fitting
            // and judging on the same vectors over-credits the candidate;
            // see HOLDOUT_STRIDE).
            let candidate_k = (n / target).max(MIN_PARTITIONS);
            let sample = clustering_sample(n, candidate_k);
            let decoded_sample = decode_rows(&sample);
            let mut fit_data = Vec::new();
            let mut holdout_data = Vec::new();
            let mut holdout_current_assignments = Vec::new();
            for (j, &i) in sample.iter().enumerate() {
                let row = &decoded_sample[j * dim..(j + 1) * dim];
                if j % HOLDOUT_STRIDE == 0 {
                    holdout_data.extend_from_slice(row);
                    holdout_current_assignments.push(assignments[i]);
                } else {
                    fit_data.extend_from_slice(row);
                }
            }
            let n_fit = fit_data.len() / dim;
            let n_holdout = holdout_data.len() / dim;
            let current_distortion =
                mean_distortion_for(&holdout_data, &holdout_current_assignments, &centroids, dim);
            let (candidate_centroids, candidate_distortion) = (0..REBOOTSTRAP_CANDIDATES)
                .map(|attempt| {
                    let (candidate, _) = kmeans::kmeans(
                        &fit_data,
                        n_fit,
                        dim,
                        candidate_k,
                        KMEANS_ITERATIONS,
                        KMEANS_SEED ^ n as u64 ^ (attempt as u64) << 32,
                    );
                    let candidate_nlist = candidate.len() / dim;
                    let (_, holdout_distances) =
                        kmeans::assign(&holdout_data, n_holdout, dim, &candidate, candidate_nlist);
                    let distortion = holdout_distances.iter().map(|&d| d as f64).sum::<f64>()
                        / n_holdout.max(1) as f64;
                    (candidate, distortion as f32)
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .expect("REBOOTSTRAP_CANDIDATES > 0");
            if candidate_distortion < REBOOTSTRAP_DISTORTION_RATIO * current_distortion {
                centroids = candidate_centroids;
                let nlist = centroids.len() / dim;
                let all: Vec<usize> = (0..n).collect();
                for chunk in all.chunks(CLUSTERING_CHUNK) {
                    let decoded = decode_rows(chunk);
                    let (chunk_assignments, _) =
                        kmeans::assign(&decoded, chunk.len(), dim, &centroids, nlist);
                    for (&i, &a) in chunk.iter().zip(&chunk_assignments) {
                        assignments[i] = a;
                    }
                }
            }
            (centroids, assignments)
        } else {
            // Bootstrap: cluster a sample, then assign everything.
            let nlist = (n / target).max(MIN_PARTITIONS);
            let sample = clustering_sample(n, nlist);
            let decoded_sample = decode_rows(&sample);
            let (centroids, _) = kmeans::kmeans(
                &decoded_sample,
                sample.len(),
                dim,
                nlist,
                KMEANS_ITERATIONS,
                KMEANS_SEED,
            );
            let nlist = centroids.len() / dim;
            let mut assignments = vec![0u32; n];
            let all: Vec<usize> = (0..n).collect();
            for chunk in all.chunks(CLUSTERING_CHUNK) {
                let decoded = decode_rows(chunk);
                let (chunk_assignments, _) =
                    kmeans::assign(&decoded, chunk.len(), dim, &centroids, nlist);
                for (&i, &a) in chunk.iter().zip(&chunk_assignments) {
                    assignments[i] = a;
                }
            }
            (centroids, assignments)
        };

        // Split pass: any partition above SPLIT_FACTOR * target is 2-means
        // split (repeatedly, until all fit). Splits redistribute only their
        // own members here; the post-split maintenance pass below then
        // re-checks neighbors — LIRE's split-then-reassign step.
        let mut split_changed: HashSet<u32> = HashSet::new();
        loop {
            let nlist = centroids.len() / dim;
            let mut counts = vec![0usize; nlist];
            for &a in &assignments {
                counts[a as usize] += 1;
            }
            let Some(oversized) = counts.iter().position(|&c| c > SPLIT_FACTOR * target) else {
                break;
            };
            let members: Vec<usize> = (0..n)
                .filter(|&i| assignments[i] as usize == oversized)
                .collect();
            let decoded = decode_rows(&members);
            let (child_centroids, child_assignments) = kmeans::kmeans(
                &decoded,
                members.len(),
                dim,
                2,
                KMEANS_ITERATIONS,
                KMEANS_SEED ^ oversized as u64,
            );
            let child_one_count = child_assignments.iter().filter(|&&c| c == 1).count();
            if child_centroids.len() / dim < 2
                || child_one_count == 0
                || child_one_count == members.len()
            {
                // Degenerate split (e.g. all-identical members): both
                // children collapse onto one side, which would loop forever.
                break;
            }
            // Child 0 replaces the parent; child 1 appends.
            let new_partition = nlist as u32;
            centroids[oversized * dim..(oversized + 1) * dim]
                .copy_from_slice(&child_centroids[0..dim]);
            centroids.extend_from_slice(&child_centroids[dim..2 * dim]);
            for (&member, &child) in members.iter().zip(&child_assignments) {
                if child == 1 {
                    assignments[member] = new_partition;
                }
            }
            split_changed.insert(oversized as u32);
            split_changed.insert(new_partition);
        }
        self.lire_maintenance(rows, &mut centroids, &mut assignments, &split_changed);

        // Merge pass: dissolve partitions below target / MERGE_DIVISOR by
        // reassigning their members to the nearest surviving centroid.
        loop {
            let nlist = centroids.len() / dim;
            if nlist <= MIN_PARTITIONS {
                break;
            }
            let mut counts = vec![0usize; nlist];
            for &a in &assignments {
                counts[a as usize] += 1;
            }
            let Some(undersized) = counts.iter().position(|&c| c < target / MERGE_DIVISOR) else {
                break;
            };
            // Remove the centroid (swap-remove style: last centroid moves
            // into its slot) and reassign its members.
            let last = nlist - 1;
            let members: Vec<usize> = (0..n)
                .filter(|&i| assignments[i] as usize == undersized)
                .collect();
            if undersized != last {
                let (head, tail) = centroids.split_at_mut(last * dim);
                head[undersized * dim..(undersized + 1) * dim].copy_from_slice(&tail[..dim]);
                for a in assignments.iter_mut() {
                    if *a as usize == last {
                        *a = undersized as u32;
                    }
                }
            }
            centroids.truncate(last * dim);
            let survivors = centroids.len() / dim;
            for chunk in members.chunks(CLUSTERING_CHUNK) {
                let decoded = decode_rows(chunk);
                let (chunk_assignments, _) =
                    kmeans::assign(&decoded, chunk.len(), dim, &centroids, survivors);
                for (&i, &a) in chunk.iter().zip(&chunk_assignments) {
                    assignments[i] = a;
                }
            }
        }

        (centroids, assignments)
    }

    fn write_compacted(&self, path: &Path) -> io::Result<()> {
        let n_out = self.len();
        let dim = self.dim_opt().unwrap_or(0);
        let n_calib = if n_out > 0 { dim } else { 0 };
        let (shift, scale_tq) = if n_calib > 0 {
            self.calibration_for_write(dim)
        } else {
            (Vec::new(), Vec::new())
        };

        // Gather all live rows, then (optionally) partition them.
        let gathered = if dim > 0 {
            self.gather_rows(dim)
        } else {
            GatheredRows {
                rows: Vec::new(),
                scales: Vec::new(),
                ids: Vec::new(),
                source_partition: Vec::new(),
                base_slot: Vec::new(),
                n_byte_groups: 0,
            }
        };
        debug_assert_eq!(gathered.len(), n_out);
        let rows = RowSource {
            gathered: &gathered,
            index: self,
            dim,
            tqplus_shift: &shift,
            tqplus_scale: &scale_tq,
        };

        let partitioning_active = self
            .partition_target
            .is_some_and(|target| n_out >= MIN_PARTITIONS * target);
        let (partition_centroids, assignments) = if partitioning_active {
            let target = self.partition_target.expect("partitioning_active");
            self.assign_partitions(&rows, target)
        } else {
            (
                vec![0.0f32; if n_out > 0 { dim } else { 0 }],
                vec![0u32; n_out],
            )
        };
        let nlist = if n_out > 0 {
            (partition_centroids.len() / dim.max(1)).max(1)
        } else {
            1
        };

        // Closure assignment: each row's primary copy plus its boundary
        // replicas, grouped by partition (stable within one, so insertion
        // order is preserved).
        let replica_lists = match self.replica_epsilon {
            Some(epsilon) if partitioning_active && nlist > 1 => closure_assignments(
                &rows,
                &partition_centroids,
                &assignments,
                epsilon,
                self.replica_prune,
            ),
            _ => vec![Vec::new(); n_out],
        };
        let mut emission: Vec<(u32, usize, bool)> =
            Vec::with_capacity(n_out + replica_lists.iter().map(Vec::len).sum::<usize>());
        for (i, &assignment) in assignments.iter().enumerate() {
            emission.push((assignment, i, false));
            for &replica_partition in &replica_lists[i] {
                emission.push((replica_partition, i, true));
            }
        }
        emission.sort_by_key(|&(partition, _, _)| partition);
        let n_slots = emission.len();

        // Per-partition metadata.
        let mut partition_counts = vec![0usize; nlist];
        for &(partition, _, _) in &emission {
            partition_counts[partition as usize] += 1;
        }
        let n_byte_groups = if dim > 0 {
            pack::n_byte_groups(self.bit_width, dim)
        } else {
            0
        };
        let block_bytes = n_byte_groups * BLOCK;
        let mut slot_base = 0usize;
        let mut codes_offset = 0usize;
        let mut partition_meta = Vec::with_capacity(nlist);
        for &count in &partition_counts {
            partition_meta.push(PartitionMeta {
                n: count,
                slot_base,
                codes_offset,
            });
            slot_base += count;
            codes_offset += ((count + BLOCK - 1) / BLOCK) * block_bytes;
        }
        let total_blocks: usize = partition_counts
            .iter()
            .map(|&count| (count + BLOCK - 1) / BLOCK)
            .sum();

        let layout = section_layout(
            self.bit_width,
            dim,
            n_slots,
            n_calib,
            nlist,
            total_blocks,
            self.store_vectors,
        );

        let mut writer = BufWriter::new(File::create(path)?);
        let mut position = 0usize;

        // Header.
        let flags = if self.store_vectors {
            FLAG_HAS_VECTORS
        } else {
            0
        };
        writer.write_all(TVDM_MAGIC)?;
        writer.write_all(&[TVDM_VERSION, self.bit_width as u8, LAYOUT_TAG, flags])?;
        writer.write_all(&(dim as u32).to_le_bytes())?;
        writer.write_all(&(n_calib as u32).to_le_bytes())?;
        writer.write_all(&(n_slots as u64).to_le_bytes())?;
        writer.write_all(&(nlist as u32).to_le_bytes())?;
        writer.write_all(&(self.partition_target.unwrap_or(0) as u32).to_le_bytes())?;
        writer.write_all(&(n_out as u64).to_le_bytes())?;
        writer.write_all(&self.replica_epsilon.unwrap_or(0.0).to_le_bytes())?;
        position += 44;
        position = pad_to(&mut writer, position, SECTION_ALIGN)?;

        // Partition table.
        debug_assert_eq!(position, layout.partition_table.0);
        for meta in &partition_meta {
            writer.write_all(&(meta.n as u64).to_le_bytes())?;
            writer.write_all(&(meta.slot_base as u64).to_le_bytes())?;
            writer.write_all(&(meta.codes_offset as u64).to_le_bytes())?;
        }
        position += PARTITION_RECORD_BYTES * nlist;
        position = pad_to(&mut writer, position, SECTION_ALIGN)?;

        // Partition centroids.
        debug_assert_eq!(position, layout.centroids.0);
        writer.write_all(f32_bytes(&partition_centroids))?;
        position += 4 * partition_centroids.len();
        position = pad_to(&mut writer, position, SECTION_ALIGN)?;

        // Codes: streamed per partition, each segment block-padded.
        debug_assert_eq!(position, layout.codes.0);
        if dim > 0 {
            let mut cursor = 0usize;
            for &count in &partition_counts {
                let mut streamer = BlockStreamer::new(&mut writer, n_byte_groups);
                for &(_, i, _) in &emission[cursor..cursor + count] {
                    streamer.push_row(gathered.row(i))?;
                }
                position += streamer.finish()?;
                cursor += count;
            }
        }
        position = pad_to(&mut writer, position, SECTION_ALIGN)?;

        // Scales (in emission order).
        debug_assert_eq!(position, layout.scales.0);
        for &(_, i, _) in &emission {
            writer.write_all(&gathered.scales[i].to_le_bytes())?;
        }
        position += 4 * n_slots;
        position = pad_to(&mut writer, position, SECTION_ALIGN)?;

        // TQ+ calibration.
        if n_calib > 0 {
            debug_assert_eq!(position, layout.tqplus_shift.0);
            writer.write_all(f32_bytes(&shift))?;
            position += 4 * shift.len();
            position = pad_to(&mut writer, position, SECTION_ALIGN)?;
            writer.write_all(f32_bytes(&scale_tq))?;
            position += 4 * scale_tq.len();
            position = pad_to(&mut writer, position, SECTION_ALIGN)?;
        }

        // Id tables (in emission order).
        debug_assert_eq!(position, layout.slot_to_id.0);
        for &(_, i, _) in &emission {
            writer.write_all(&gathered.ids[i].to_le_bytes())?;
        }
        position += 8 * n_slots;
        position = pad_to(&mut writer, position, SECTION_ALIGN)?;

        let mut sorted_pairs: Vec<(u64, u64)> = emission
            .iter()
            .enumerate()
            .map(|(slot, &(_, i, _))| (gathered.ids[i], slot as u64))
            .collect();
        sorted_pairs.sort_unstable();
        for &(id, _) in &sorted_pairs {
            writer.write_all(&id.to_le_bytes())?;
        }
        position += 8 * sorted_pairs.len();
        position = pad_to(&mut writer, position, SECTION_ALIGN)?;
        for &(_, slot) in &sorted_pairs {
            writer.write_all(&slot.to_le_bytes())?;
        }
        position += 8 * sorted_pairs.len();
        position = pad_to(&mut writer, position, SECTION_ALIGN)?;

        // Replica bitmap (in emission order).
        debug_assert_eq!(position, layout.replica_flags.0);
        let mut bitmap = vec![0u8; (n_slots + 7) / 8];
        for (slot, &(_, _, is_replica)) in emission.iter().enumerate() {
            if is_replica {
                bitmap[slot / 8] |= 1 << (slot % 8);
            }
        }
        writer.write_all(&bitmap)?;
        position += bitmap.len();
        position = pad_to(&mut writer, position, SECTION_ALIGN)?;

        // Full-precision vectors (in emission order; replicas repeat their
        // primary's vector so every slot stays directly addressable).
        if self.store_vectors {
            debug_assert_eq!(position, layout.vectors.0);
            for &(_, i, _) in &emission {
                let vector: &[f32] = match gathered.base_slot[i] {
                    NO_SLOT => self
                        .delta_originals
                        .get(&gathered.ids[i])
                        .expect("store_vectors invariant: every delta row has an original"),
                    slot => self
                        .base
                        .as_ref()
                        .expect("gathered base row implies a base segment")
                        .vector_row(slot as usize),
                };
                writer.write_all(f32_bytes(vector))?;
            }
            position += 4 * n_slots * dim;
            position = pad_to(&mut writer, position, SECTION_ALIGN)?;
        }

        debug_assert_eq!(position, layout.total_len);
        writer.flush()?;
        Ok(())
    }
}

/// The raw little-endian bytes of an f32 slice. This module is compiled
/// only on little-endian targets, so the in-memory representation already
/// matches the file's layout.
pub(crate) fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and the target is little-endian; the
    // borrow is tied to the input slice.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) }
}

/// SPANN closure assignment: for every row, the partitions beyond its
/// primary whose centroids are within `(1 + epsilon)` of the row's nearest
/// centroid distance — pruned by SPANN's RNG rule (a candidate partition is
/// skipped when some already-selected partition's centroid is closer to the
/// candidate's centroid than the row itself is: the existing copy already
/// covers queries from that direction) and capped at
/// [`MAX_ASSIGNMENT_COPIES`] total copies. Returns the per-row replica
/// partition lists (primaries not included).
fn closure_assignments(
    rows: &RowSource,
    centroids: &[f32],
    assignments: &[u32],
    epsilon: f32,
    prune: bool,
) -> Vec<Vec<u32>> {
    let dim = rows.dim;
    let n = rows.gathered.len();
    let mut replica_lists: Vec<Vec<u32>> = Vec::with_capacity(n);
    let all: Vec<usize> = (0..n).collect();
    for chunk in all.chunks(CLUSTERING_CHUNK) {
        let vectors = rows.vectors(chunk);
        let chunk_assignments: Vec<u32> = chunk.iter().map(|&i| assignments[i]).collect();
        replica_lists.extend(closure_assignments_for_vectors(
            &vectors,
            chunk.len(),
            dim,
            centroids,
            &chunk_assignments,
            epsilon,
            prune,
        ));
    }
    replica_lists
}

/// The batch-level core of closure assignment over already-materialized
/// vectors; `assignments[i]` is the primary partition of `vectors[i]`. See
/// [`closure_assignments`] for the rule.
pub(crate) fn closure_assignments_for_vectors(
    vectors: &[f32],
    n: usize,
    dim: usize,
    centroids: &[f32],
    assignments: &[u32],
    epsilon: f32,
    prune: bool,
) -> Vec<Vec<u32>> {
    let nlist = centroids.len() / dim;
    let mut replica_lists: Vec<Vec<u32>> = vec![Vec::new(); n];
    if nlist <= 1 || n == 0 {
        return replica_lists;
    }

    let centroid_sq_norms: Vec<f32> = (0..nlist)
        .map(|c| {
            centroids[c * dim..(c + 1) * dim]
                .iter()
                .map(|&v| v * v)
                .sum()
        })
        .collect();
    let sq_l2 =
        |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(&x, &y)| (x - y) * (x - y)).sum() };

    let bound_factor = (1.0 + epsilon) * (1.0 + epsilon);
    let products = crate::linalg::matmul_nt(vectors, n, dim, centroids, nlist);
    for i in 0..n {
        let row = &vectors[i * dim..(i + 1) * dim];
        let row_sq_norm: f32 = row.iter().map(|&v| v * v).sum();
        let mut ranked: Vec<(f32, u32)> = (0..nlist)
            .map(|c| {
                let sq_dist =
                    (row_sq_norm + centroid_sq_norms[c] - 2.0 * products[i * nlist + c]).max(0.0);
                (sq_dist, c as u32)
            })
            .collect();
        ranked.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let bound = ranked[0].0 * bound_factor;
        let primary = assignments[i];
        let mut selected: Vec<u32> = vec![primary];
        for &(sq_dist, candidate) in &ranked {
            if sq_dist > bound || selected.len() >= MAX_ASSIGNMENT_COPIES {
                break;
            }
            if candidate == primary {
                continue;
            }
            let candidate_center =
                &centroids[candidate as usize * dim..(candidate as usize + 1) * dim];
            let redundant = prune
                && selected.iter().any(|&s| {
                    let selected_center = &centroids[s as usize * dim..(s as usize + 1) * dim];
                    sq_l2(candidate_center, selected_center) < sq_dist
                });
            if redundant {
                continue;
            }
            selected.push(candidate);
            replica_lists[i].push(candidate);
        }
    }
    replica_lists
}

/// One partition's raw contents read from a `.tvdm` file, for lossless
/// crate-internal migration (codes, scales, ids, replica flags, and the
/// full-precision vectors when stored).
pub(crate) struct TvdmPartition {
    pub group_rows: Vec<u8>,
    pub scales: Vec<f32>,
    pub ids: Vec<u64>,
    pub replica: Vec<bool>,
    pub vectors: Option<Vec<f32>>,
}

/// Full raw contents of a `.tvdm` file. See [`read_tvdm_contents`].
pub(crate) struct TvdmContents {
    pub dim: usize,
    pub bit_width: usize,
    pub store_vectors: bool,
    pub partition_target: Option<usize>,
    pub replica_epsilon: Option<f32>,
    pub tqplus_shift: Vec<f32>,
    pub tqplus_scale: Vec<f32>,
    /// `partitions.len() * dim`, row i = centroid of `partitions[i]`.
    pub centroids: Vec<f32>,
    pub partitions: Vec<TvdmPartition>,
}

/// Read a `.tvdm` file's complete contents into RAM (migration-time only).
pub(crate) fn read_tvdm_contents(path: &Path) -> io::Result<TvdmContents> {
    let base = BaseSegment::open(path)?;
    let n_byte_groups = if base.dim > 0 {
        pack::n_byte_groups(base.bit_width, base.dim)
    } else {
        0
    };
    let block_bytes = base.block_bytes();
    let slot_to_id = base.slot_to_id();
    let scales = base.scales();
    let mut partitions = Vec::with_capacity(base.nlist());
    let mut block_rows = vec![0u8; BLOCK * n_byte_groups];
    for p in 0..base.nlist() {
        let meta = base.partitions[p];
        let codes = base.partition_codes(p);
        let mut partition = TvdmPartition {
            group_rows: Vec::with_capacity(meta.n * n_byte_groups),
            scales: Vec::with_capacity(meta.n),
            ids: Vec::with_capacity(meta.n),
            replica: Vec::with_capacity(meta.n),
            vectors: base
                .has_vectors
                .then(|| Vec::with_capacity(meta.n * base.dim)),
        };
        for block_idx in 0..meta.n_blocks() {
            pack::unpack_block_rows(
                &codes[block_idx * block_bytes..(block_idx + 1) * block_bytes],
                n_byte_groups,
                &mut block_rows,
            );
            let lanes = (meta.n - block_idx * BLOCK).min(BLOCK);
            for lane in 0..lanes {
                let slot = meta.slot_base + block_idx * BLOCK + lane;
                partition.group_rows.extend_from_slice(
                    &block_rows[lane * n_byte_groups..(lane + 1) * n_byte_groups],
                );
                partition.scales.push(scales[slot]);
                partition.ids.push(slot_to_id[slot]);
                partition.replica.push(base.is_replica(slot));
                if let Some(vectors) = partition.vectors.as_mut() {
                    vectors.extend_from_slice(base.vector_row(slot));
                }
            }
        }
        partitions.push(partition);
    }
    Ok(TvdmContents {
        dim: base.dim,
        bit_width: base.bit_width,
        store_vectors: base.has_vectors,
        partition_target: match base.file_partition_target {
            0 => None,
            target => Some(target),
        },
        replica_epsilon: base.file_replica_epsilon,
        tqplus_shift: base.tqplus_shift().to_vec(),
        tqplus_scale: base.tqplus_scale().to_vec(),
        centroids: base.partition_centroids().to_vec(),
        partitions,
    })
}

/// Row indices to cluster `k` centroids from: at least
/// [`MIN_SAMPLES_PER_CENTROID`] per centroid, within
/// `[CLUSTERING_CHUNK, MAX_BOOTSTRAP_SAMPLE]`, strided over the corpus so
/// the sample spans the full insertion history.
pub(crate) fn clustering_sample(n: usize, k: usize) -> Vec<usize> {
    let wanted = (MIN_SAMPLES_PER_CENTROID * k)
        .clamp(CLUSTERING_CHUNK, MAX_BOOTSTRAP_SAMPLE)
        .min(n.max(1));
    let step = (n / wanted).max(1);
    (0..n).step_by(step).collect()
}

/// Mean squared distance of vectors to their assigned centroids — the
/// k-means objective, used by the re-clustering acceptance test.
pub(crate) fn mean_distortion_for(
    data: &[f32],
    assignments: &[u32],
    centroids: &[f32],
    dim: usize,
) -> f32 {
    if assignments.is_empty() {
        return 0.0;
    }
    let total: f64 = assignments
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            let row = &data[i * dim..(i + 1) * dim];
            let center = &centroids[a as usize * dim..(a as usize + 1) * dim];
            row.iter()
                .zip(center)
                .map(|(&x, &c)| ((x - c) as f64).powi(2))
                .sum::<f64>()
        })
        .sum();
    (total / assignments.len() as f64) as f32
}

/// Rank partitions per query by squared L2 distance to partition centroids;
/// returns up to `nprobe` nearest partition ids per query. With
/// `probe_epsilon`, the ranked list is additionally cut at the SPANN
/// distance bound: only partitions whose centroid distance is within
/// `(1 + epsilon)` of the query's nearest centroid distance are probed —
/// the comparison runs on squared distances against `(1 + epsilon)^2`.
pub(crate) fn route_queries(
    queries: &[f32],
    nq: usize,
    dim: usize,
    centroids: &[f32],
    nlist: usize,
    nprobe: usize,
    probe_epsilon: Option<f32>,
) -> Vec<Vec<u32>> {
    let centroid_sq_norms: Vec<f32> = (0..nlist)
        .map(|c| {
            centroids[c * dim..(c + 1) * dim]
                .iter()
                .map(|&v| v * v)
                .sum()
        })
        .collect();

    let products = crate::linalg::matmul_nt(queries, nq, dim, centroids, nlist);

    (0..nq)
        .map(|qi| {
            let row = &products[qi * nlist..(qi + 1) * nlist];
            // The epsilon cut needs true (squared) distances, so the
            // query's own norm cannot be dropped the way pure ranking
            // allows; clamp at zero against rounding.
            let query_sq_norm: f32 = queries[qi * dim..(qi + 1) * dim]
                .iter()
                .map(|&v| v * v)
                .sum();
            let mut ranked: Vec<(f32, u32)> = (0..nlist)
                .map(|c| {
                    let sq_dist = (query_sq_norm + centroid_sq_norms[c] - 2.0 * row[c]).max(0.0);
                    (sq_dist, c as u32)
                })
                .collect();
            ranked.sort_unstable_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            ranked.truncate(nprobe);
            if let Some(epsilon) = probe_epsilon {
                let bound = ranked[0].0 * (1.0 + epsilon) * (1.0 + epsilon);
                let cut = ranked.partition_point(|&(sq_dist, _)| sq_dist <= bound);
                ranked.truncate(cut.max(1));
            }
            ranked.into_iter().map(|(_, c)| c).collect()
        })
        .collect()
}

impl std::fmt::Debug for DiskIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskIndex")
            .field("bit_width", &self.bit_width)
            .field("dim", &self.dim_opt())
            .field("path", &self.path)
            .field("base_len", &self.base_len())
            .field("delta_len", &self.delta_len())
            .field("tombstone_count", &self.tombstone_count())
            .field("nlist", &self.nlist())
            .field("partition_target", &self.partition_target)
            .field("replica_epsilon", &self.replica_epsilon)
            .field("store_vectors", &self.store_vectors)
            .finish()
    }
}

fn temp_sibling(path: &Path) -> PathBuf {
    let mut file_name = path
        .file_name()
        .map_or_else(|| "index".into(), |name| name.to_os_string());
    file_name.push(".tmp");
    path.with_file_name(file_name)
}

fn pad_to<W: Write>(writer: &mut W, position: usize, align: usize) -> io::Result<usize> {
    let target = align_up(position, align);
    let zeros = [0u8; SECTION_ALIGN];
    writer.write_all(&zeros[..target - position])?;
    Ok(target)
}
