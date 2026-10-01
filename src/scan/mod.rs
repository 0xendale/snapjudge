//! Static, keyless scan for LLM call sites.

pub mod candidate;
pub mod draft;
pub mod extract;
pub mod function;
pub mod id;
pub mod imports;
pub mod modules;
mod pipeline;
pub mod python;
mod review;
pub mod schema;
pub mod schema_json;
pub mod signals;
pub mod syntax;
pub mod tier;
pub mod trace;
pub mod typescript;
pub mod walk;
pub mod workspace;

use std::path::Path;

use typed_arena::Arena;

use crate::decision::{self, DecisionSiteReport, SourceRecord, SourceScan};
use crate::model::{CallSite, OutputField, ScanReport, Sdk, Summary, Tier};
use extract::RawCall;
use schema::Schema;
use trace::Trace;
use walk::SourceFile;
use workspace::Workspace;

pub use review::{JevReviewer, ReviewError, notice as jev_notice};

/// The legacy report: the `CallSite` projection of [`scan_decision_sites`].
pub fn scan(root: &Path) -> ScanReport {
    decision::legacy_report(&scan_decision_sites(root))
}

/// Generic `decision-site-v1` report of `root` (redesign §4): every Source occurrence of
/// the legacy report plus the hidden wrapper definitions, with provenance.
pub fn scan_decision_sites(root: &Path) -> DecisionSiteReport {
    let (files, too_big) = walk::source_files(root);
    scan_files_generic(root, files, too_big).0
}

/// Generic schema 1.1 report with diagnostic Jev suggestions on Review-tier sites.
pub fn scan_decision_sites_with_jev(
    root: &Path,
    reviewer: &JevReviewer<'_>,
) -> Result<DecisionSiteReport, ReviewError> {
    let mut report = scan_decision_sites(root);
    review::review_report(root, &mut report, reviewer)?;
    Ok(report)
}

/// `scan` plus its provenance (every occurrence, caller edge and disposition). The
/// provenance is internal: it is not part of the legacy JSON report.
pub fn scan_with_trace(root: &Path) -> (ScanReport, Trace) {
    let (files, too_big) = walk::source_files(root);
    let (report, trace) = scan_files_generic(root, files, too_big);
    (decision::legacy_report(&report), trace)
}

/// Scan the given files (`too_big` counts files skipped before this call). The result
/// does not depend on the order of `files`.
pub fn scan_files(root: &Path, files: Vec<SourceFile>, too_big: usize) -> ScanReport {
    decision::legacy_report(&scan_files_generic(root, files, too_big).0)
}

/// Generic report of the given files (see [`scan_files`]).
pub fn scan_files_decision_sites(
    root: &Path,
    files: Vec<SourceFile>,
    too_big: usize,
) -> DecisionSiteReport {
    scan_files_generic(root, files, too_big).0
}

fn scan_files_generic(
    root: &Path,
    mut files: Vec<SourceFile>,
    too_big: usize,
) -> (DecisionSiteReport, Trace) {
    // File ids (and so occurrence identities) follow the relative path, never the input order.
    files.sort_by(|a, b| a.rel.cmp(&b.rel));
    files.dedup_by(|a, b| a.rel == b.rel);
    let arena = Arena::new();
    let workspace = Workspace::new(files, &arena);
    let analysis = pipeline::analyze(&workspace);
    (
        decision::from_scan(source_scan(root, &workspace, too_big, analysis.records)),
        analysis.trace,
    )
}

/// Literal `temperature`, `top_p` and `seed` of the SDK call at `source` (under `root`), for
/// eval to mirror; nothing when no SDK call starts there or the file cannot be read.
/// Internal: no scan report carries these values.
pub fn call_params(root: &Path, source: &decision::SourceLocation) -> extract::CallParams {
    let path = root.join(&source.file);
    let (Some(grammar), Ok(text)) = (walk::grammar_for(&path), walk::read_source(&path)) else {
        return extract::CallParams::default();
    };
    let lang = grammar.lang();
    let sdks = imports::imported_sdks(&text, lang);
    let ast = syntax::parse(&text, grammar);
    let node = ast.root();
    let idx = syntax::index(&node);
    let found = match lang {
        crate::model::Lang::Python => python::candidates(&node, &sdks, &idx),
        _ => typescript::candidates(&node, &sdks, &idx, lang),
    };
    found
        .iter()
        .find(|c| c.call.range().start == source.byte_offset)
        .map(|c| extract::call_params(&c.named))
        .unwrap_or_default()
}

/// The owned scan result of `workspace` (records in analysis order).
fn source_scan(
    root: &Path,
    workspace: &Workspace,
    too_big: usize,
    records: Vec<SourceRecord>,
) -> SourceScan {
    let unreadable = workspace.texts.iter().filter(|text| text.is_none()).count();
    SourceScan {
        root: root.display().to_string(),
        files_scanned: workspace.files.len() - unreadable,
        files_skipped: too_big + unreadable,
        records,
    }
}

/// Tier, reasons and output fields of one evaluation. Parameter conflicts are appended
/// to the reasons and force Review.
pub(crate) fn assess(r: &RawCall) -> (Tier, Vec<String>, Vec<OutputField>) {
    let signals = r
        .prompt
        .as_ref()
        .and_then(|p| p.text.as_deref())
        .map(signals::prompt_signals)
        .unwrap_or_default();
    let (mut tier, mut reasons) =
        tier::assign(&r.schema, r.prompt.as_ref(), &signals, r.max_tokens);
    if !r.conflicts.is_empty() {
        tier = Tier::Review;
        reasons.extend(r.conflicts.iter().cloned());
    }
    let outputs: Vec<OutputField> = match (&r.schema, tier) {
        (Schema::Resolved { fields, .. }, _) => fields.clone(),
        (Schema::None, Tier::Likely) => signals::implied_space(&signals)
            .map(|space| {
                vec![OutputField {
                    name: None,
                    description: None,
                    space,
                }]
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    };
    if tier == Tier::Likely && outputs.is_empty() {
        reasons.push("answer labels not found in the prompt".into());
    }
    (tier, reasons, outputs)
}

fn to_site(r: RawCall, f: &SourceFile) -> CallSite {
    let (tier, reasons, outputs) = assess(&r);
    let drafts = if tier == Tier::NotDecision {
        Vec::new()
    } else {
        draft::rough_drafts(&outputs, r.prompt.as_ref())
    };
    CallSite {
        id: id::site_id(&f.rel, &r.call_text),
        file: f.rel.clone(),
        line: r.line,
        lang: f.grammar.lang(),
        sdk: r.sdk,
        api: r.api,
        via: r.via,
        model: r.model,
        tier,
        outputs,
        reasons,
        prompt: r.prompt,
        max_tokens: r.max_tokens,
        drafts,
    }
}

pub fn summarize(sites: &[CallSite]) -> Summary {
    let count = |t: Tier| sites.iter().filter(|s| s.tier == t).count();
    let counted: Vec<&CallSite> = sites.iter().filter(|s| s.sdk != Sdk::Langchain).collect();
    let decisions = counted
        .iter()
        .filter(|s| matches!(s.tier, Tier::Sure | Tier::Likely))
        .count();
    let decision_pct = if counted.is_empty() {
        0.0
    } else {
        (decisions as f64 * 1000.0 / counted.len() as f64).round() / 10.0
    };
    Summary {
        call_sites: sites.len(),
        sure: count(Tier::Sure),
        likely: count(Tier::Likely),
        review: count(Tier::Review),
        not_decision: count(Tier::NotDecision),
        langchain_excluded: sites.len() - counted.len(),
        counted: counted.len(),
        decisions,
        decision_pct,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::model::AnswerSpace;

    fn write(dir: &Path, rel: &str, body: &str) {
        let p = dir.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, body).unwrap();
    }

    #[test]
    fn scans_tiers_and_summarizes() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write(
            root,
            "a.py",
            "from openai import OpenAI\nfrom typing import Literal\nfrom pydantic import BaseModel\n\nclass V(BaseModel):\n    label: Literal['spam', 'ham']\n\nr = client.chat.completions.parse(model='gpt-4o', messages=[], response_format=V)\n",
        );
        write(
            root,
            "b.ts",
            "import { generateText } from 'ai';\nconst { text } = await generateText({ model: m, prompt: 'Write a haiku' });\n",
        );
        write(root, "c.py", "print('no llm here')\n");
        write(
            root,
            "d.py",
            "from langchain_openai import ChatOpenAI\ns = ChatOpenAI(model='x').with_structured_output(V)\n",
        );
        write(
            root,
            "e.py",
            "import openai\nr = openai.chat.completions.create(model='x', messages=[{'role': 'user', 'content': 'Answer yes or no'}], max_tokens=1)\n",
        );
        fs::write(root.join("bad.py"), [0xff, 0xfe, 0x00]).unwrap();

        let r = scan(root);
        assert_eq!(r.files_scanned, 5);
        assert_eq!(r.files_skipped, 1);
        let got: Vec<(&str, Tier)> = r.sites.iter().map(|s| (s.file.as_str(), s.tier)).collect();
        assert_eq!(
            got,
            vec![
                ("a.py", Tier::Sure),
                ("b.ts", Tier::NotDecision),
                ("d.py", Tier::Review),
                ("e.py", Tier::Likely)
            ]
        );
        // Likely without schema: answer space implied by the prompt, one rough Noul draft
        let e = &r.sites[3];
        assert_eq!(e.outputs[0].space, AnswerSpace::Noul);
        assert_eq!(e.drafts.len(), 1);
        assert!(r.sites[1].drafts.is_empty());
        assert_eq!(
            r.summary,
            Summary {
                call_sites: 4,
                sure: 1,
                likely: 1,
                review: 1,
                not_decision: 1,
                langchain_excluded: 1,
                counted: 3,
                decisions: 2,
                decision_pct: 66.7
            }
        );
    }

    /// The pre-6B legacy list, computed directly from the emitted records: sort by
    /// (file, line, raw id), `-N` suffixes, summary.
    fn direct_legacy(root: &Path) -> ScanReport {
        let (mut files, too_big) = walk::source_files(root);
        files.sort_by(|a, b| a.rel.cmp(&b.rel));
        let arena = Arena::new();
        let workspace = Workspace::new(files, &arena);
        let scan = source_scan(
            root,
            &workspace,
            too_big,
            pipeline::analyze(&workspace).records,
        );
        let mut sites: Vec<CallSite> = scan
            .records
            .into_iter()
            .filter(|record| record.kind == decision::SiteKind::Occurrence)
            .map(|record| record.site)
            .collect();
        sites.sort_by(|a, b| (&a.file, a.line, &a.id).cmp(&(&b.file, b.line, &b.id)));
        let mut ids: Vec<String> = sites.iter().map(|s| s.id.clone()).collect();
        id::dedupe_ids(&mut ids);
        for (site, id) in sites.iter_mut().zip(ids) {
            site.id = id;
        }
        ScanReport {
            root: scan.root,
            files_scanned: scan.files_scanned,
            files_skipped: scan.files_skipped,
            summary: summarize(&sites),
            sites,
        }
    }

    #[test]
    fn legacy_report_is_the_projection_of_the_generic_report() {
        let dir = tempfile::tempdir().unwrap();
        // Two identical calls on one line: same raw id, same line.
        let call = "client.chat.completions.create(model='x', messages=m)";
        write(
            dir.path(),
            "a.py",
            &format!("import openai\npair = [{call}, {call}]\n{call}\n"),
        );
        for root in [
            Path::new("tests/fixtures"),
            Path::new("tests/fixtures_wrap"),
            dir.path(),
        ] {
            let generic = scan_decision_sites(root);
            assert_eq!(generic.validate(), Ok(()), "{}", root.display());
            let projected = decision::legacy_report(&generic);
            let direct = direct_legacy(root);
            assert_eq!(projected, direct, "{}", root.display());
            assert_eq!(scan(root), direct, "{}", root.display());
            assert_eq!(
                serde_json::to_string_pretty(&projected).unwrap(),
                serde_json::to_string_pretty(&direct).unwrap()
            );
            // Occurrences map one-to-one, in order, onto the legacy sites.
            let occurrences: Vec<&decision::DecisionSite> = generic
                .sites
                .iter()
                .filter(|site| site.kind == decision::SiteKind::Occurrence)
                .collect();
            assert_eq!(occurrences.len(), direct.sites.len());
            for (site, legacy) in occurrences.iter().zip(&direct.sites) {
                assert_eq!(site.id, format!("source:{}", legacy.id));
            }
        }
        let ids: Vec<String> = scan(dir.path()).sites.into_iter().map(|s| s.id).collect();
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[1], format!("{}-2", ids[0]));
    }

    #[test]
    fn duplicate_calls_in_one_file_get_distinct_ids() {
        let dir = tempfile::tempdir().unwrap();
        let call = "client.chat.completions.create(model='x', messages=m)\n";
        write(dir.path(), "a.py", &format!("import openai\n{call}{call}"));
        let r = scan(dir.path());
        assert_eq!(r.sites.len(), 2);
        assert_eq!(r.sites[1].id, format!("{}-2", r.sites[0].id));
    }

    #[test]
    fn conflicts_force_review_and_append_reasons() {
        let source = "from typing import Literal\nfrom pydantic import BaseModel\nclass V(BaseModel):\n    label: Literal['spam', 'ham']\nr = client.chat.completions.parse(model='gpt-4o', messages=[], response_format=V)\n";
        let ast = syntax::parse(source, walk::Grammar::Python);
        let root = ast.root();
        let index = syntax::index(&root);
        let mut raw = python::detect(&root, &[Sdk::Openai], &index).remove(0);
        let file = SourceFile {
            path: "a.py".into(),
            rel: "a.py".into(),
            grammar: walk::Grammar::Python,
        };
        assert_eq!(to_site(raw.clone(), &file).tier, Tier::Sure);

        raw.conflicts = vec!["conflicting values for parameter schema".into()];
        let site = to_site(raw, &file);
        assert_eq!(site.tier, Tier::Review);
        assert_eq!(
            site.reasons.last().map(String::as_str),
            Some("conflicting values for parameter schema")
        );
        assert_eq!(site.outputs.len(), 1);
    }

    #[test]
    fn call_params_read_literal_sampling_parameters_of_the_site_call() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "a.ts",
            "import OpenAI from 'openai';\nconst c = new OpenAI();\nconst t = 0.4;\nexport const r = await c.chat.completions.create({ model: 'gpt-4o', messages: [{ role: 'user', content: 'Answer yes or no' }], max_tokens: 1, topP: 0.2, temperature: t, seed: 3 });\n",
        );
        write(
            dir.path(),
            "b.py",
            "import openai\nr = openai.chat.completions.create(model='x', messages=[{'role': 'user', 'content': 'Answer yes or no'}], max_tokens=1, temperature=0)\n",
        );
        let report = scan_decision_sites(dir.path());
        let params = |file: &str| {
            let site = report
                .sites
                .iter()
                .find(|s| s.source.as_ref().is_some_and(|src| src.file == file))
                .unwrap();
            call_params(dir.path(), site.source.as_ref().unwrap())
        };
        let ts = params("a.ts");
        assert_eq!(
            (ts.temperature, ts.top_p, ts.seed),
            (None, Some(0.2), Some(3))
        );
        let py = params("b.py");
        assert_eq!((py.temperature, py.top_p, py.seed), (Some(0.0), None, None));
    }
}
