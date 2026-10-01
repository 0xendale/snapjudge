use std::path::Path;

use typed_arena::Arena;

use super::*;
use crate::decision;
use crate::scan::{source_scan, walk};

const ROOT: &str = "tests/fixtures_wrap";

#[test]
fn via_caps_labels_and_counts_other_alternatives() {
    let labels: Vec<String> = (0..7).map(|i| format!("w.py:{i} f{i}")).collect();
    assert_eq!(via(&labels[..2], 0), labels[..2].to_vec());
    let capped = via(&labels, 3);
    assert_eq!(capped.len(), MAX_VIA_LABELS + 1);
    assert_eq!(capped[..MAX_VIA_LABELS], labels[..MAX_VIA_LABELS]);
    assert_eq!(capped[MAX_VIA_LABELS], "+3 alternatives");
}

/// Run the analysis after warming the workspace caches in a given order.
fn analyze_after(warm: impl Fn(&Workspace)) -> (crate::model::ScanReport, Trace) {
    let (mut files, too_big) = walk::source_files(Path::new(ROOT));
    files.sort_by(|a, b| a.rel.cmp(&b.rel));
    let arena = Arena::new();
    let workspace = Workspace::new(files, &arena);
    warm(&workspace);
    let analysis = analyze(&workspace);
    let scan = source_scan(Path::new(ROOT), &workspace, too_big, analysis.records);
    (
        decision::legacy_report(&decision::from_scan(scan)),
        analysis.trace,
    )
}

#[test]
fn cache_warm_up_order_does_not_change_the_result() {
    let cold = analyze_after(|_| {});
    let reversed = analyze_after(|workspace| {
        for id in (0..workspace.files.len()).rev() {
            workspace.scope(id);
        }
    });
    let interleaved = analyze_after(|workspace| {
        let count = workspace.files.len();
        let odd: Vec<usize> = (0..count).filter(|id| id % 2 == 1).collect();
        workspace.ensure_parsed(&odd);
        for id in (0..count).filter(|id| id % 3 == 0) {
            workspace.imports(id);
            workspace.definitions(id);
        }
    });
    assert_eq!(cold, reversed);
    assert_eq!(cold, interleaved);
    assert!(!cold.1.edges.is_empty());
}
