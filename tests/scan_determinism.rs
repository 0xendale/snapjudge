//! The report does not depend on discovery order (redesign §15 "Determinism").

use std::path::Path;

use snapjudge::scan::walk::{SourceFile, source_files};
use snapjudge::scan::{scan, scan_files};

const ROOTS: &[&str] = &["tests/fixtures_wrap", "tests/fixtures"];

/// Deterministic Fisher-Yates shuffle driven by a small LCG.
fn shuffled(mut files: Vec<SourceFile>, mut seed: u64) -> Vec<SourceFile> {
    for i in (1..files.len()).rev() {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let j = (seed >> 33) as usize % (i + 1);
        files.swap(i, j);
    }
    files
}

#[test]
fn file_order_does_not_change_the_report() {
    for root in ROOTS {
        let root = Path::new(root);
        let want = scan(root);
        let (files, too_big) = source_files(root);

        let mut reversed = files.clone();
        reversed.reverse();
        assert_eq!(scan_files(root, reversed, too_big), want, "reversed");
        for seed in [1, 7, 42, 2026] {
            let permuted = shuffled(files.clone(), seed);
            assert_eq!(scan_files(root, permuted, too_big), want, "seed {seed}");
        }
    }
}

#[test]
fn repeated_runs_give_the_same_report() {
    let root = Path::new("tests/fixtures_wrap");
    let first = scan(root);
    for _ in 0..3 {
        assert_eq!(scan(root), first);
    }
    let ids: Vec<&str> = first.sites.iter().map(|s| s.id.as_str()).collect();
    let mut unique = ids.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), ids.len(), "ids are unique after dedupe");
}
