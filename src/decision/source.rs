//! Source sites from a finished scan, and the legacy report projection (plan
//! "Construction and projection").

use std::collections::HashSet;

use super::{
    DecisionSite, DecisionSiteReport, Evidence, GenericSummary, Origin, Provenance, SCHEMA_NAME,
    SCHEMA_VERSION, SOURCE_PREFIX, SiteKind, SourceLocation, is_legacy,
};
use crate::model::{CallSite, ScanReport};
use crate::scan::{id, summarize};

/// One emitted or hidden occurrence of a scan, owned.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceRecord {
    /// The site as the legacy report renders it, with its raw (not deduplicated) id.
    pub site: CallSite,
    /// `WrapperDefinition` for a hidden wrapper, `Occurrence` for an emitted site.
    pub kind: SiteKind,
    pub byte_offset: usize,
    pub evidence: Evidence,
    pub provenance: Provenance,
}

/// The owned output of one scan, in analysis order.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceScan {
    pub root: String,
    pub files_scanned: usize,
    pub files_skipped: usize,
    pub records: Vec<SourceRecord>,
}

/// Generic report of a scan. Occurrence ids and order are the legacy ones (sort by
/// (file, line, raw id), then `-N` suffixes); wrapper definitions then take the next free
/// suffix in (file, line, byte) order and are merged in by (file, line, id).
pub fn from_scan(scan: SourceScan) -> DecisionSiteReport {
    let (mut occurrences, mut definitions): (Vec<SourceRecord>, Vec<SourceRecord>) = scan
        .records
        .into_iter()
        .partition(|record| record.kind == SiteKind::Occurrence);
    // Stable sort: ties keep analysis order, exactly like the legacy report.
    occurrences.sort_by(|a, b| {
        (&a.site.file, a.site.line, &a.site.id).cmp(&(&b.site.file, b.site.line, &b.site.id))
    });
    let mut ids: Vec<String> = occurrences.iter().map(|r| r.site.id.clone()).collect();
    id::dedupe_ids(&mut ids);
    let mut used: HashSet<String> = ids.iter().cloned().collect();
    let occurrences: Vec<DecisionSite> = occurrences
        .into_iter()
        .zip(ids)
        .map(|(record, id)| site(record, id))
        .collect();

    definitions.sort_by(|a, b| {
        (&a.site.file, a.site.line, a.byte_offset).cmp(&(&b.site.file, b.site.line, b.byte_offset))
    });
    let mut definitions: Vec<DecisionSite> = definitions
        .into_iter()
        .map(|record| {
            let base = record.site.id.clone();
            let id = std::iter::once(base.clone())
                .chain((2..).map(|n| format!("{base}-{n}")))
                .find(|id| !used.contains(id))
                .unwrap_or(base);
            used.insert(id.clone());
            site(record, id)
        })
        .collect();
    definitions.sort_by(|a, b| key(a).cmp(&key(b)));

    let legacy: Vec<CallSite> = occurrences.iter().filter_map(project).collect();
    let summary = GenericSummary {
        legacy: summarize(&legacy),
        wrapper_definitions: definitions.len(),
    };
    let mut sites = Vec::with_capacity(occurrences.len() + definitions.len());
    let mut definitions = definitions.into_iter().peekable();
    for occurrence in occurrences {
        while definitions
            .peek()
            .is_some_and(|definition| key(definition) < key(&occurrence))
        {
            sites.extend(definitions.next());
        }
        sites.push(occurrence);
    }
    sites.extend(definitions);
    DecisionSiteReport {
        schema: SCHEMA_NAME.to_string(),
        schema_version: SCHEMA_VERSION.to_string(),
        root: scan.root,
        files_scanned: scan.files_scanned,
        files_skipped: scan.files_skipped,
        summary,
        sites,
    }
}

/// Merge key of a Source site: (file, line, id).
fn key(site: &DecisionSite) -> (&str, usize, &str) {
    let (file, line) = site
        .source
        .as_ref()
        .map_or(("", 0), |source| (source.file.as_str(), source.line));
    (file, line, site.id.as_str())
}

fn site(record: SourceRecord, id: String) -> DecisionSite {
    let revision = record.evidence.revision();
    let CallSite {
        file,
        line,
        lang,
        sdk,
        api,
        via,
        model,
        tier,
        outputs,
        reasons,
        prompt,
        max_tokens,
        drafts,
        ..
    } = record.site;
    DecisionSite {
        schema_version: SCHEMA_VERSION.to_string(),
        id: format!("{SOURCE_PREFIX}{id}"),
        source_revision: Some(revision),
        origin: Origin::Source,
        source: Some(SourceLocation {
            file,
            line,
            byte_offset: record.byte_offset,
            lang,
            sdk,
            api,
            max_tokens,
            via,
        }),
        agent: None,
        model,
        prompt,
        outputs,
        kind: record.kind,
        tier,
        reasons,
        drafts,
        input_schema: None,
        provenance: record.provenance,
        jev_review: None,
    }
}

fn project(site: &DecisionSite) -> Option<CallSite> {
    CallSite::try_from(site).ok()
}

/// The legacy report: every Source occurrence projected to a `CallSite`, in report order,
/// and the legacy summary of exactly those sites.
pub fn legacy_report(report: &DecisionSiteReport) -> ScanReport {
    let sites: Vec<CallSite> = report
        .sites
        .iter()
        .filter(|site| is_legacy(site))
        .filter_map(project)
        .collect();
    ScanReport {
        root: report.root.clone(),
        files_scanned: report.files_scanned,
        files_skipped: report.files_skipped,
        summary: summarize(&sites),
        sites,
    }
}
