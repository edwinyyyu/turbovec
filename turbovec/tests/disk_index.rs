//! Tests for the disk-primary [`DiskIndex`].
//!
//! The load-bearing property is *score parity with [`IdMapIndex`]*: a
//! compacted `.tvdm` file stores the same blocked-code layout the in-RAM
//! kernel cache uses, so searching the mmap must return byte-identical
//! scores for identically-constructed data. The remaining tests cover the
//! delta/tombstone state machine and file-format rejection.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use turbovec::{DiskIndex, IdMapIndex};

const DIM: usize = 64;
const BITS: usize = 4;

fn temp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-disk-{}-{}", nonce, name));
    p
}

fn gaussian_vectors(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    (0..n * DIM)
        .map(|_| <StandardNormal as Distribution<f32>>::sample(&StandardNormal, &mut rng))
        .collect()
}

fn ids(range: std::ops::Range<u64>) -> Vec<u64> {
    range.collect()
}

#[test]
fn compacted_base_matches_id_map_index_exactly() {
    let path = temp_path("parity.tvdm");
    let n = 100;
    let k = 10;
    let vectors = gaussian_vectors(n, 1);
    let queries = gaussian_vectors(5, 2);
    let all_ids = ids(1000..1000 + n as u64);

    let mut reference = IdMapIndex::new(DIM, BITS).unwrap();
    reference.add_with_ids(&vectors, &all_ids).unwrap();

    let mut disk = DiskIndex::new(Some(DIM), BITS).unwrap();
    disk.add_with_ids(&vectors, &all_ids).unwrap();
    disk.write(&path).unwrap();
    assert_eq!(disk.base_len(), n);
    assert_eq!(disk.delta_len(), 0);

    let (reference_scores, reference_ids) = reference.search(&queries, k);
    let (disk_scores, disk_ids) = disk.search(&queries, k);
    assert_eq!(disk_ids, reference_ids);
    assert_eq!(disk_scores, reference_scores);

    // A freshly-opened handle (nothing in RAM but the header and id table)
    // must agree too.
    let reopened = DiskIndex::open(&path).unwrap();
    assert_eq!(reopened.len(), n);
    let (reopened_scores, reopened_ids) = reopened.search(&queries, k);
    assert_eq!(reopened_ids, reference_ids);
    assert_eq!(reopened_scores, reference_scores);

    std::fs::remove_file(&path).ok();
}

#[test]
fn base_plus_delta_matches_incrementally_built_id_map_index() {
    let path = temp_path("merge.tvdm");
    let n_base = 80;
    let n_delta = 40;
    let k = 12;
    let base_vectors = gaussian_vectors(n_base, 3);
    let delta_vectors = gaussian_vectors(n_delta, 4);
    let queries = gaussian_vectors(4, 5);
    let base_ids = ids(0..n_base as u64);
    let delta_ids = ids(500..500 + n_delta as u64);

    // Reference built with the same two-batch history, so TQ+ calibration
    // (fitted on the first batch, reused for the second) matches.
    let mut reference = IdMapIndex::new(DIM, BITS).unwrap();
    reference.add_with_ids(&base_vectors, &base_ids).unwrap();
    reference.add_with_ids(&delta_vectors, &delta_ids).unwrap();

    let mut disk = DiskIndex::new(Some(DIM), BITS).unwrap();
    disk.add_with_ids(&base_vectors, &base_ids).unwrap();
    disk.write(&path).unwrap();
    disk.add_with_ids(&delta_vectors, &delta_ids).unwrap();
    assert_eq!(disk.base_len(), n_base);
    assert_eq!(disk.delta_len(), n_delta);
    assert_eq!(disk.len(), n_base + n_delta);

    let (reference_scores, reference_ids) = reference.search(&queries, k);
    let (disk_scores, disk_ids) = disk.search(&queries, k);
    assert_eq!(disk_ids, reference_ids);
    assert_eq!(disk_scores, reference_scores);

    // Compacting the delta into the base must not change results.
    disk.write(&path).unwrap();
    assert_eq!(disk.base_len(), n_base + n_delta);
    assert_eq!(disk.delta_len(), 0);
    let (compacted_scores, compacted_ids) = disk.search(&queries, k);
    assert_eq!(compacted_ids, reference_ids);
    assert_eq!(compacted_scores, reference_scores);

    std::fs::remove_file(&path).ok();
}

#[test]
fn tombstoned_base_vectors_are_hidden_and_compacted_away() {
    let path = temp_path("tombstone.tvdm");
    let n = 60;
    let vectors = gaussian_vectors(n, 6);
    let queries = gaussian_vectors(3, 7);
    let all_ids = ids(0..n as u64);

    let mut disk = DiskIndex::new(Some(DIM), BITS).unwrap();
    disk.add_with_ids(&vectors, &all_ids).unwrap();
    disk.write(&path).unwrap();

    // Remove every third id from the base.
    let removed: Vec<u64> = all_ids.iter().copied().filter(|id| id % 3 == 0).collect();
    for &id in &removed {
        assert!(disk.remove(id));
        assert!(!disk.remove(id), "second remove of {id} must be a no-op");
    }
    let n_live = n - removed.len();
    assert_eq!(disk.len(), n_live);
    assert_eq!(disk.tombstone_count(), removed.len());
    for &id in &removed {
        assert!(!disk.contains(id));
    }

    // Tombstoned ids never appear, even with k = full live count.
    let (_, result_ids) = disk.search(&queries, n_live);
    assert_eq!(result_ids.len() / 3, n_live, "row width must be live count");
    for id in &result_ids {
        assert!(!removed.contains(id), "tombstoned id {id} surfaced");
    }

    // Survivors are unaffected by compaction.
    let (before_scores, before_ids) = disk.search(&queries, 10);
    disk.write(&path).unwrap();
    assert_eq!(disk.base_len(), n_live);
    assert_eq!(disk.tombstone_count(), 0);
    let (after_scores, after_ids) = disk.search(&queries, 10);
    assert_eq!(after_ids, before_ids);
    assert_eq!(after_scores, before_scores);

    std::fs::remove_file(&path).ok();
}

#[test]
fn removed_id_can_be_re_added_and_re_removed() {
    let path = temp_path("readd.tvdm");
    let n = 40;
    let vectors = gaussian_vectors(n, 8);
    let all_ids = ids(0..n as u64);

    let mut disk = DiskIndex::new(Some(DIM), BITS).unwrap();
    disk.add_with_ids(&vectors, &all_ids).unwrap();
    disk.write(&path).unwrap();

    // Adding a live base id must be rejected without corrupting state.
    let one = gaussian_vectors(1, 9);
    assert!(disk.add_with_ids(&one, &[5]).is_err());
    assert_eq!(disk.len(), n);

    // Remove from base, re-add into the delta: live again with new data.
    assert!(disk.remove(5));
    assert!(!disk.contains(5));
    disk.add_with_ids(&one, &[5]).unwrap();
    assert!(disk.contains(5));
    assert_eq!(disk.len(), n);

    // Removing the re-added id removes the delta copy; the stale base copy
    // must stay hidden behind its tombstone.
    assert!(disk.remove(5));
    assert!(!disk.contains(5));
    assert_eq!(disk.len(), n - 1);

    // Compaction preserves all of the above.
    disk.write(&path).unwrap();
    assert!(!disk.contains(5));
    assert_eq!(disk.len(), n - 1);

    std::fs::remove_file(&path).ok();
}

#[test]
fn lazy_index_commits_dim_on_first_add_and_survives_empty_write() {
    let path = temp_path("lazy.tvdm");

    let mut disk = DiskIndex::new(None, BITS).unwrap();
    assert_eq!(disk.dim_opt(), None);
    assert!(disk.is_empty());
    let (scores, result_ids) = disk.search(&gaussian_vectors(1, 10), 5);
    assert!(scores.is_empty() && result_ids.is_empty());

    // Empty lazy write round-trips.
    disk.write(&path).unwrap();
    let mut reopened = DiskIndex::open(&path).unwrap();
    assert_eq!(reopened.dim_opt(), None);
    assert!(reopened.is_empty());

    // First add commits dim; write and reopen preserve it.
    let vectors = gaussian_vectors(10, 11);
    reopened
        .add_with_ids_2d(&vectors, DIM, &ids(0..10))
        .unwrap();
    assert_eq!(reopened.dim_opt(), Some(DIM));
    reopened.write(&path).unwrap();
    let reopened_again = DiskIndex::open(&path).unwrap();
    assert_eq!(reopened_again.dim_opt(), Some(DIM));
    assert_eq!(reopened_again.len(), 10);

    std::fs::remove_file(&path).ok();
}

#[test]
fn search_row_width_is_clamped_to_live_count() {
    let n = 6;
    let vectors = gaussian_vectors(n, 12);
    let queries = gaussian_vectors(2, 13);

    let mut disk = DiskIndex::new(Some(DIM), BITS).unwrap();
    disk.add_with_ids(&vectors, &ids(0..n as u64)).unwrap();
    let (scores, result_ids) = disk.search(&queries, 50);
    assert_eq!(scores.len(), 2 * n);
    assert_eq!(result_ids.len(), 2 * n);
}

#[test]
fn open_rejects_wrong_magic_and_missing_file() {
    let path = temp_path("bad-magic.tvdm");
    let mut file = File::create(&path).unwrap();
    file.write_all(b"NOPE").unwrap();
    file.write_all(&[0u8; 60]).unwrap();
    drop(file);
    let err = DiskIndex::open(&path).unwrap_err();
    assert!(err.to_string().contains("wrong magic"), "{err}");
    std::fs::remove_file(&path).ok();

    assert!(DiskIndex::open(temp_path("does-not-exist.tvdm")).is_err());
}

#[test]
fn open_rejects_truncated_file() {
    let path = temp_path("truncated.tvdm");
    let n = 50;
    let vectors = gaussian_vectors(n, 14);

    let mut disk = DiskIndex::new(Some(DIM), BITS).unwrap();
    disk.add_with_ids(&vectors, &ids(0..n as u64)).unwrap();
    disk.write(&path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() - 64]).unwrap();
    let err = DiskIndex::open(&path).unwrap_err();
    assert!(err.to_string().contains("length"), "{err}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn two_bit_index_round_trips() {
    let path = temp_path("2bit.tvdm");
    let n = 50;
    let k = 5;
    let vectors = gaussian_vectors(n, 15);
    let queries = gaussian_vectors(3, 16);
    let all_ids = ids(0..n as u64);

    let mut reference = IdMapIndex::new(DIM, 2).unwrap();
    reference.add_with_ids(&vectors, &all_ids).unwrap();

    let mut disk = DiskIndex::new(Some(DIM), 2).unwrap();
    disk.add_with_ids(&vectors, &all_ids).unwrap();
    disk.write(&path).unwrap();

    let (reference_scores, reference_ids) = reference.search(&queries, k);
    let (disk_scores, disk_ids) = DiskIndex::open(&path).unwrap().search(&queries, k);
    assert_eq!(disk_ids, reference_ids);
    assert_eq!(disk_scores, reference_scores);

    std::fs::remove_file(&path).ok();
}

#[test]
fn convert_id_map_file_preserves_results_exactly() {
    let tvim_path = temp_path("convert-src.tvim");
    let tvdm_path = temp_path("convert-dst.tvdm");
    let n = 90;
    let k = 10;
    let vectors = gaussian_vectors(n, 17);
    let queries = gaussian_vectors(4, 18);
    let all_ids = ids(2000..2000 + n as u64);

    let mut source = IdMapIndex::new(DIM, BITS).unwrap();
    source.add_with_ids(&vectors, &all_ids).unwrap();
    source.write(&tvim_path).unwrap();

    DiskIndex::convert_id_map_file(&tvim_path, &tvdm_path).unwrap();
    let converted = DiskIndex::open(&tvdm_path).unwrap();
    assert_eq!(converted.len(), n);

    let (source_scores, source_ids) = source.search(&queries, k);
    let (converted_scores, converted_ids) = converted.search(&queries, k);
    assert_eq!(converted_ids, source_ids);
    assert_eq!(converted_scores, source_scores);

    std::fs::remove_file(&tvim_path).ok();
    std::fs::remove_file(&tvdm_path).ok();
}

#[test]
fn tvim_tvdm_round_trip_is_lossless() {
    let tvim_a = temp_path("rt-a.tvim");
    let tvdm = temp_path("rt.tvdm");
    let tvim_b = temp_path("rt-b.tvim");
    let n = 75;
    let k = 10;
    let vectors = gaussian_vectors(n, 19);
    let queries = gaussian_vectors(4, 20);
    let all_ids = ids(3000..3000 + n as u64);

    let mut source = IdMapIndex::new(DIM, BITS).unwrap();
    source.add_with_ids(&vectors, &all_ids).unwrap();
    source.write(&tvim_a).unwrap();

    DiskIndex::convert_id_map_file(&tvim_a, &tvdm).unwrap();
    DiskIndex::convert_to_id_map_file(&tvdm, &tvim_b).unwrap();

    // Byte-identical files: codes, scales, calibration and ids all survive
    // the round trip exactly.
    let bytes_a = std::fs::read(&tvim_a).unwrap();
    let bytes_b = std::fs::read(&tvim_b).unwrap();
    assert_eq!(bytes_a, bytes_b, "round-tripped .tvim differs from source");

    let restored = IdMapIndex::load(&tvim_b).unwrap();
    let (source_scores, source_ids) = source.search(&queries, k);
    let (restored_scores, restored_ids) = restored.search(&queries, k);
    assert_eq!(restored_ids, source_ids);
    assert_eq!(restored_scores, source_scores);

    for path in [&tvim_a, &tvdm, &tvim_b] {
        std::fs::remove_file(path).ok();
    }
}
