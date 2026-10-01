//! Generic `decision-site-v1` contract (redesign §4, §14 row 6B, §15 Core): schema
//! validation, round trips, provenance, counting, ids and `source_revision`.

use std::fs;
use std::path::Path;

use serde_json::Value;
use snapjudge::decision::{
    DecisionSite, DecisionSiteReport, Disposition, EdgeKind, Origin, SiteKind, legacy_report,
};
use snapjudge::model::Tier;
use snapjudge::scan::{scan, scan_decision_sites, scan_files_decision_sites, walk};

const SCHEMA: &str = "schemas/decision-site-v1.schema.json";
const EXAMPLES: &str = "schemas/examples";
const ROOTS: &[&str] = &[
    "tests/fixtures",
    "tests/fixtures_wrap",
    "tests/fixtures_site",
];

fn validator() -> jsonschema::Validator {
    let schema: Value = serde_json::from_str(&fs::read_to_string(SCHEMA).unwrap()).unwrap();
    jsonschema::validator_for(&schema).unwrap()
}

fn schema_errors(validator: &jsonschema::Validator, value: &Value) -> Vec<String> {
    validator
        .iter_errors(value)
        .map(|e| format!("{} at {}", e, e.instance_path()))
        .collect()
}

fn pretty<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string_pretty(value).unwrap() + "\n"
}

fn example(name: &str) -> String {
    fs::read_to_string(format!("{EXAMPLES}/decision-site-v1.{name}.json")).unwrap()
}

fn write(dir: &Path, rel: &str, body: &str) {
    let path = dir.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, body).unwrap();
}

fn site<'r>(report: &'r DecisionSiteReport, file: &str, line: usize) -> &'r DecisionSite {
    let found: Vec<&DecisionSite> = report
        .sites
        .iter()
        .filter(|s| {
            s.source
                .as_ref()
                .is_some_and(|l| l.file == file && l.line == line)
        })
        .collect();
    assert_eq!(found.len(), 1, "{file}:{line}: {found:#?}");
    found[0]
}

fn revision(report: &DecisionSiteReport, file: &str, line: usize) -> String {
    site(report, file, line).source_revision.clone().unwrap()
}

// ---- Schema, examples and round trips (Task 4) ----

#[test]
fn report_example_is_the_scan_of_the_site_fixture() {
    let generated = pretty(&scan_decision_sites(Path::new("tests/fixtures_site")));
    assert!(
        generated == example("report"),
        "schemas/examples/decision-site-v1.report.json is stale"
    );
}

#[test]
fn report_example_covers_every_documented_case() {
    let report = DecisionSiteReport::from_json(&example("report")).unwrap();
    let disposition = |file: &str, line: usize| site(&report, file, line).provenance.disposition;
    assert_eq!(disposition("app.py", 20), Some(Disposition::Direct));
    let traced = site(&report, "app.py", 16);
    assert_eq!(traced.provenance.disposition, Some(Disposition::Distinct));
    assert_eq!(traced.source.as_ref().unwrap().via, ["llm.py:6 ask"]);
    let hidden = site(&report, "llm.py", 7);
    assert_eq!(hidden.kind, SiteKind::WrapperDefinition);
    assert_eq!(hidden.provenance.disposition, Some(Disposition::Hidden));
    assert_eq!(disposition("unused.py", 7), Some(Disposition::NoCallers));
    let ambiguous = site(&report, "classifiers.py", 32);
    assert_eq!(
        ambiguous.provenance.disposition,
        Some(Disposition::Ambiguous)
    );
    assert_eq!(ambiguous.provenance.alternatives.len(), 2);
}

#[test]
fn generated_reports_and_examples_match_the_json_schema() {
    let validator = validator();
    for root in ROOTS {
        let value = serde_json::to_value(scan_decision_sites(Path::new(root))).unwrap();
        assert_eq!(
            schema_errors(&validator, &value),
            Vec::<String>::new(),
            "{root}"
        );
    }
    for name in ["report", "agent", "runtime"] {
        let value: Value = serde_json::from_str(&example(name)).unwrap();
        assert_eq!(
            schema_errors(&validator, &value),
            Vec::<String>::new(),
            "{name}"
        );
    }
}

#[test]
fn reports_and_examples_round_trip() {
    for root in ROOTS {
        let report = scan_decision_sites(Path::new(root));
        let text = pretty(&report);
        let back = DecisionSiteReport::from_json(&text).unwrap();
        assert_eq!(back, report, "{root}");
        assert_eq!(pretty(&back), text, "{root}");
    }
    let report = DecisionSiteReport::from_json(&example("report")).unwrap();
    assert_eq!(pretty(&report), example("report"));
    for (name, origin) in [("agent", Origin::AgentHook), ("runtime", Origin::Runtime)] {
        let site = DecisionSite::from_json(&example(name)).unwrap();
        assert_eq!(site.origin, origin);
        assert_eq!(pretty(&site), example(name), "{name}");
    }
}

/// Each origin rule is enforced alike by `validate()` and by the JSON Schema.
#[test]
fn schema_and_validate_reject_the_same_origin_violations() {
    let validator = validator();
    let agent: Value = serde_json::from_str(&example("agent")).unwrap();
    let runtime: Value = serde_json::from_str(&example("runtime")).unwrap();
    let report: Value = serde_json::from_str(&example("report")).unwrap();
    let source = report["sites"][0].clone();
    let source_location = source["source"].clone();
    let agent_origin = agent["agent"].clone();
    type Change = Box<dyn Fn(&mut Value)>;
    let cases: Vec<(&str, &Value, Change)> = vec![
        (
            "source without source",
            &source,
            Box::new(|v: &mut Value| {
                v.as_object_mut().unwrap().remove("source");
            }),
        ),
        (
            "source without revision",
            &source,
            Box::new(|v: &mut Value| {
                v.as_object_mut().unwrap().remove("source_revision");
            }),
        ),
        (
            "source with agent",
            &source,
            Box::new(move |v: &mut Value| {
                v["agent"] = agent_origin.clone();
            }),
        ),
        (
            "source with agent id",
            &source,
            Box::new(|v: &mut Value| v["id"] = "agent:x".into()),
        ),
        (
            "agent without agent",
            &agent,
            Box::new(|v: &mut Value| {
                v.as_object_mut().unwrap().remove("agent");
            }),
        ),
        (
            "agent with source",
            &agent,
            Box::new(move |v: &mut Value| {
                v["source"] = source_location.clone();
            }),
        ),
        (
            "agent with revision",
            &agent,
            Box::new(|v: &mut Value| {
                v["source_revision"] = "0".repeat(64).into();
            }),
        ),
        (
            "agent wrapper definition",
            &agent,
            Box::new(|v: &mut Value| {
                v["kind"] = "wrapper_definition".into();
            }),
        ),
        (
            "agent with provenance",
            &agent,
            Box::new(|v: &mut Value| {
                v["provenance"] = serde_json::json!({"depth": 0});
            }),
        ),
        (
            "agent with source id",
            &agent,
            Box::new(|v: &mut Value| v["id"] = "source:x".into()),
        ),
        (
            "runtime with agent id",
            &runtime,
            Box::new(|v: &mut Value| v["id"] = "agent:x".into()),
        ),
        (
            "runtime wrapper definition",
            &runtime,
            Box::new(|v: &mut Value| {
                v["kind"] = "wrapper_definition".into();
            }),
        ),
        (
            "runtime with revision",
            &runtime,
            Box::new(|v: &mut Value| {
                v["source_revision"] = "0".repeat(64).into();
            }),
        ),
        (
            "major version 2",
            &runtime,
            Box::new(|v: &mut Value| {
                v["schema_version"] = "2.0".into();
            }),
        ),
    ];
    for (name, base, change) in cases {
        let mut value = base.clone();
        change(&mut value);
        assert!(!validator.is_valid(&value), "schema accepts: {name}");
        assert!(
            DecisionSite::from_json(&value.to_string()).is_err(),
            "validate accepts: {name}"
        );
    }
    // Omission rule: an empty provenance list is omitted, never serialized.
    let mut value = source;
    value["provenance"]["callers"] = serde_json::json!([]);
    assert!(!validator.is_valid(&value));
    assert!(DecisionSite::from_json(&value.to_string()).is_err());
}

#[test]
fn generic_wrapper_fixture_report_snapshot() {
    insta::assert_json_snapshot!(scan_decision_sites(Path::new("tests/fixtures_wrap")));
}

// ---- Provenance, counting and ids (Tasks 4 and 5) ----

#[test]
fn wrapper_definitions_are_generic_only() {
    for root in ROOTS {
        let generic = scan_decision_sites(Path::new(root));
        let legacy = scan(Path::new(root));
        let definitions: Vec<&DecisionSite> = generic
            .sites
            .iter()
            .filter(|s| s.kind == SiteKind::WrapperDefinition)
            .collect();
        assert_eq!(generic.summary.wrapper_definitions, definitions.len());
        // Never in the legacy output or its summary.
        for definition in &definitions {
            let id = definition.id.strip_prefix("source:").unwrap();
            assert!(legacy.sites.iter().all(|s| s.id != id), "{root}: {id}");
            assert_eq!(definition.provenance.disposition, Some(Disposition::Hidden));
        }
        assert_eq!(generic.summary.legacy, legacy.summary, "{root}");
        assert_eq!(
            generic.sites.len(),
            legacy.sites.len() + definitions.len(),
            "{root}"
        );
        assert_eq!(legacy_report(&generic), legacy);
        let text = serde_json::to_string(&legacy).unwrap();
        assert!(!text.contains("wrapper_definition"));
    }
    let wrap = scan_decision_sites(Path::new("tests/fixtures_wrap"));
    assert!(wrap.summary.wrapper_definitions > 0);
    assert_eq!(wrap.summary.legacy.counted, wrap.summary.legacy.call_sites);
}

#[test]
fn generic_sites_are_sorted_occurrences_with_definitions_merged_in() {
    for root in ROOTS {
        let report = scan_decision_sites(Path::new(root));
        let legacy = scan(Path::new(root));
        let occurrences: Vec<String> = report
            .sites
            .iter()
            .filter(|s| s.kind == SiteKind::Occurrence)
            .map(|s| s.id.clone())
            .collect();
        let want: Vec<String> = legacy
            .sites
            .iter()
            .map(|s| format!("source:{}", s.id))
            .collect();
        assert_eq!(occurrences, want, "{root}");
        let mut ids: Vec<&str> = report.sites.iter().map(|s| s.id.as_str()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), report.sites.len(), "{root}: duplicate ids");
        // A definition precedes the first occurrence with a greater (file, line, id).
        let key = |s: &DecisionSite| {
            let l = s.source.as_ref().unwrap();
            (l.file.clone(), l.line, s.id.clone())
        };
        for (at, definition) in report.sites.iter().enumerate() {
            if definition.kind != SiteKind::WrapperDefinition {
                continue;
            }
            for other in &report.sites[..at] {
                if other.kind == SiteKind::Occurrence {
                    assert!(key(other) < key(definition), "{root}");
                }
            }
        }
    }
}

#[test]
fn provenance_preserves_every_alternative_edge_and_disposition() {
    let report = scan_decision_sites(Path::new("tests/fixtures_wrap"));

    // Ambiguous: every alternative keeps what it decides.
    let ambiguous = site(&report, "py/app/classifiers.py", 32);
    let p = &ambiguous.provenance;
    assert_eq!(p.disposition, Some(Disposition::Ambiguous));
    assert_eq!(p.reason_codes, ["ambiguous_wrapper_match"]);
    let models: Vec<Option<&str>> = p.alternatives.iter().map(|a| a.model.as_deref()).collect();
    assert_eq!(models, [Some("gpt-4o-mini"), Some("gpt-4o")]);
    assert!(
        p.alternatives
            .iter()
            .all(|a| a.edge.kind == EdgeKind::Ambiguous)
    );
    assert!(
        p.alternatives
            .iter()
            .all(|a| a.tier == Tier::Sure && a.outputs.len() == 1)
    );
    assert_eq!(ambiguous.model, None);

    // The display cap never truncates stored alternatives.
    let ensemble = site(&report, "py/app/ensemble.py", 24);
    assert_eq!(ensemble.provenance.alternatives.len(), 6);
    assert_eq!(ensemble.source.as_ref().unwrap().via.len(), 2);
    assert_eq!(
        ensemble.provenance.reason_codes,
        ["wrapper_multiple_decisions"]
    );
    assert_eq!(
        ensemble.provenance.disposition,
        Some(Disposition::MultipleDecisions)
    );

    // Hidden wrapper definition: callers are its representing edges.
    let hidden = site(&report, "py/app/llm.py", 7);
    assert_eq!(hidden.kind, SiteKind::WrapperDefinition);
    assert!(!hidden.provenance.callers.is_empty());
    assert!(hidden.provenance.callers.iter().all(|e| e.callee.line == 7));
    // Folded chain links (triage.py, tickets.py, routing.py) are provenance of the wrapper.
    let folded: Vec<&str> = hidden
        .provenance
        .folded
        .iter()
        .map(|o| o.file.as_str())
        .collect();
    assert_eq!(
        folded,
        ["py/app/routing.py", "py/app/tickets.py", "py/app/triage.py"]
    );

    // D6A-3: no callers keeps the legacy evaluation; the disposition is generic-only.
    let unused = site(&report, "py/app/unused.py", 7);
    assert_eq!(unused.provenance.disposition, Some(Disposition::NoCallers));
    assert_eq!(unused.provenance.reason_codes, ["no_callers"]);
    assert_eq!(unused.kind, SiteKind::Occurrence);
    let represented = site(&report, "py/app/summaries.py", 7);
    assert_eq!(
        represented.provenance.disposition,
        Some(Disposition::Represented)
    );
    assert_eq!(
        represented.provenance.folded[0].file,
        "py/tests/test_summaries.py"
    );

    // Depth limit: the fifth edge is blocked, never an alternative.
    let deep: Vec<&DecisionSite> = report
        .sites
        .iter()
        .filter(|s| s.provenance.disposition == Some(Disposition::DepthExceeded))
        .collect();
    assert_eq!(deep.len(), 1);
    assert!(deep[0].provenance.alternatives.is_empty());
    assert_eq!(deep[0].provenance.blocked.len(), 1);
    assert_eq!(deep[0].provenance.blocked[0].depth, 5);
    assert_eq!(deep[0].provenance.reason_codes, ["trace_depth_exceeded"]);
    let hub = site(&report, "py/deep/moderation_chain.py", 51);
    assert_eq!(hub.provenance.blocked[0].kind, EdgeKind::DepthExceeded);
    assert_eq!(hub.provenance.reason_codes, ["trace_depth_exceeded"]);
    // Recursion unrolled to depth 5 is blocked but gets no depth reason code.
    let requeue = site(&report, "py/app/requeue.py", 18);
    assert_eq!(requeue.provenance.blocked.len(), 1);
    assert!(requeue.provenance.reason_codes.is_empty());

    // Edge ids: unique per edge, shared by the caller's and the callee's view.
    let mut edges: Vec<(String, String)> = Vec::new();
    for s in &report.sites {
        let p = &s.provenance;
        let all = p
            .alternatives
            .iter()
            .map(|a| &a.edge)
            .chain(&p.blocked)
            .chain(&p.callers);
        for edge in all {
            assert!(!edge.origins.is_empty());
            assert_eq!(edge.origins[0], edge.callee);
            edges.push((edge.id.clone(), serde_json::to_string(edge).unwrap()));
        }
    }
    edges.sort();
    edges.dedup();
    let mut ids: Vec<&String> = edges.iter().map(|(id, _)| id).collect();
    let total = ids.len();
    ids.dedup();
    assert_eq!(ids.len(), total, "one edge per id");
    let shared = site(&report, "py/app/main.py", 20).provenance.alternatives[0]
        .edge
        .clone();
    assert!(hidden.provenance.callers.contains(&shared));

    // Every Source site carries its location in provenance; bindings use slot encoding.
    for s in &report.sites {
        let occurrence = s.provenance.occurrence.as_ref().unwrap();
        let location = s.source.as_ref().unwrap();
        assert_eq!(occurrence.file, location.file);
        assert_eq!(occurrence.byte_offset, location.byte_offset);
    }
    let slots: Vec<&str> = shared.bindings.iter().map(|b| b.slot.as_str()).collect();
    assert_eq!(slots, ["param:0", "param:1"]);
}

const SENTIMENT: &str = "from typing import Literal\n\nfrom openai import OpenAI\nfrom pydantic import BaseModel\n\nclient = OpenAI()\n\n\nclass Sentiment(BaseModel):\n    mood: Literal[\"happy\", \"sad\"]\n\n\ndef classify(text, model):\n    return client.chat.completions.parse(\n        model=model,\n        messages=[\n            {\"role\": \"system\", \"content\": \"Classify the sentiment.\"},\n            {\"role\": \"user\", \"content\": text},\n        ],\n        response_format=Sentiment,\n    )\n";
const CALLER: &str = "from llm import classify\n\n\ndef handle(review):\n    return classify(review.text, \"gpt-4o\")\n";

/// A caller site (app.py:5) through a wrapper whose schema and prompt live in llm.py.
fn sentiment_repo(wrapper: &str, caller: &str) -> (tempfile::TempDir, DecisionSiteReport) {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "llm.py", wrapper);
    write(dir.path(), "app.py", caller);
    let report = scan_decision_sites(dir.path());
    (dir, report)
}

#[test]
fn source_revision_is_stable_across_runs_file_order_and_moves() {
    let (dir, report) = sentiment_repo(SENTIMENT, CALLER);
    let caller = site(&report, "app.py", 5);
    assert_eq!(caller.provenance.disposition, Some(Disposition::Distinct));
    assert_eq!(caller.tier, Tier::Sure);
    let base = revision(&report, "app.py", 5);
    assert_eq!(base.len(), 64);
    assert_eq!(scan_decision_sites(dir.path()), report);

    let (mut files, too_big) = walk::source_files(dir.path());
    files.reverse();
    assert_eq!(
        scan_files_decision_sites(dir.path(), files, too_big),
        report
    );

    for root in ROOTS {
        let root = Path::new(root);
        let (mut files, too_big) = walk::source_files(root);
        files.reverse();
        assert_eq!(
            scan_files_decision_sites(root, files, too_big),
            scan_decision_sites(root)
        );
    }

    // Moved call: lines shift in the caller and in the wrapper file.
    let shifted = format!("# moved\n\n\n{CALLER}");
    let wrapper = SENTIMENT.replace("client = OpenAI()\n", "client = OpenAI()\n\n# note\n");
    let (_dir, moved) = sentiment_repo(&wrapper, &shifted);
    assert_eq!(site(&moved, "app.py", 8).source.as_ref().unwrap().line, 8);
    assert_eq!(revision(&moved, "app.py", 8), base);
    // Whitespace inside the call never changes it either.
    let spaced = CALLER.replace(
        "classify(review.text, \"gpt-4o\")",
        "classify(\n        review.text,\n        \"gpt-4o\"\n    )",
    );
    let (_dir, spaced) = sentiment_repo(SENTIMENT, &spaced);
    assert_eq!(revision(&spaced, "app.py", 5), base);
}

#[test]
fn source_revision_changes_with_the_wrapper_schema_prompt_or_model() {
    let (_dir, report) = sentiment_repo(SENTIMENT, CALLER);
    let base = revision(&report, "app.py", 5);
    let edits = [
        SENTIMENT.replace("\"happy\", \"sad\"", "\"happy\", \"sad\", \"angry\""),
        SENTIMENT.replace("Classify the sentiment.", "Classify the review sentiment."),
    ];
    for wrapper in edits {
        let (_dir, edited) = sentiment_repo(&wrapper, CALLER);
        assert_ne!(revision(&edited, "app.py", 5), base);
    }
    let (_dir, edited) = sentiment_repo(SENTIMENT, &CALLER.replace("gpt-4o", "gpt-4.1"));
    assert_ne!(revision(&edited, "app.py", 5), base);
}

#[test]
fn source_revision_changes_when_one_alternative_of_an_ambiguous_site_changes() {
    let classifiers = fs::read_to_string("tests/fixtures_site/classifiers.py").unwrap();
    let scan_with = |text: &str| {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "classifiers.py", text);
        scan_decision_sites(dir.path())
    };
    let report = scan_with(&classifiers);
    let ambiguous = site(&report, "classifiers.py", 32);
    assert_eq!(
        ambiguous.provenance.disposition,
        Some(Disposition::Ambiguous)
    );
    let base = revision(&report, "classifiers.py", 32);
    // The Careful alternative switches model; the ambiguous call itself is unchanged.
    let edited = scan_with(&classifiers.replacen("model=\"gpt-4o\"", "model=\"gpt-4.1\"", 1));
    assert_eq!(
        site(&edited, "classifiers.py", 32).provenance.disposition,
        Some(Disposition::Ambiguous)
    );
    assert_ne!(revision(&edited, "classifiers.py", 32), base);
    assert_eq!(scan_with(&classifiers).sites, report.sites);
}

#[test]
fn conflicting_parameter_values_are_provenance() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "llm.py",
        "from openai import OpenAI\n\nclient = OpenAI()\n\n\ndef ask(text, schema, model):\n    return client.chat.completions.parse(model=model, messages=[{\"role\": \"user\", \"content\": text}], response_format=schema)\n",
    );
    write(
        dir.path(),
        "app.py",
        "from typing import Literal\n\nfrom pydantic import BaseModel\n\nfrom llm import ask\n\n\nclass A(BaseModel):\n    a: Literal[\"x\", \"y\"]\n\n\ndef go(t):\n    return ask(t, A, \"gpt-4o\", schema=A)\n",
    );
    let report = scan_decision_sites(dir.path());
    let caller = site(&report, "app.py", 13);
    assert_eq!(caller.tier, Tier::Review);
    assert_eq!(caller.provenance.conflicts.len(), 1, "{caller:#?}");
    assert!(caller.reasons.contains(&caller.provenance.conflicts[0]));
    assert_eq!(
        caller.provenance.reason_codes,
        ["conflicting_parameter_values"]
    );
}

#[test]
fn hidden_wrappers_render_as_if_emitted_with_the_depth_reason() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "chain.py",
        "from typing import Literal\n\nfrom openai import OpenAI\nfrom pydantic import BaseModel\n\nclient = OpenAI()\n\n\nclass Decision(BaseModel):\n    action: Literal[\"keep\", \"hide\"]\n\n\ndef screen(c, s, m):\n    return client.chat.completions.parse(model=m, messages=[{\"role\": \"user\", \"content\": c}], response_format=s)\n\n\ndef p1(c, s, m):\n    return screen(c, s, m)\n\n\ndef p2(c, s, m):\n    return p1(c, s, m)\n\n\ndef p3(c, s, m):\n    return p2(c, s, m)\n\n\ndef hub(c, s, m, deep):\n    if deep:\n        return p3(c, s, m)\n    return screen(c, s, m)\n\n\ndef moderate(c, s):\n    return hub(c, s, \"gpt-4o\", True)\n\n\ndef top(c):\n    return moderate(c, Decision)\n",
    );
    let report = scan_decision_sites(dir.path());
    let moderate = site(&report, "chain.py", 36);
    assert_eq!(moderate.kind, SiteKind::WrapperDefinition, "{report:#?}");
    assert_eq!(moderate.provenance.disposition, Some(Disposition::Hidden));
    assert_eq!(
        moderate.source.as_ref().unwrap().via,
        ["chain.py:29 hub", "chain.py:13 screen"]
    );
    assert_eq!(moderate.model.as_deref(), Some("gpt-4o"));
    assert_eq!(moderate.provenance.blocked.len(), 1);
    assert_eq!(moderate.provenance.reason_codes, ["trace_depth_exceeded"]);
    assert_eq!(
        moderate.reasons.last().map(String::as_str),
        Some("trace_depth_exceeded: wrapper chain deeper than 4 levels")
    );
    assert_eq!(site(&report, "chain.py", 40).tier, Tier::Sure);
    assert!(scan(dir.path()).sites.iter().all(|s| s.line != 36));
}

/// Deviation (plan "WrapperDefinition ids"): occurrence ids are fixed first, so a hidden
/// wrapper earlier in a file can carry a higher suffix than a later emitted duplicate,
/// and its id changes when it switches between Hidden and NoCallers.
#[test]
fn wrapper_definition_ids_take_the_next_free_suffix() {
    let call =
        "client.chat.completions.parse(model=\"m\", messages=messages, response_format=schema)";
    let llm = format!(
        "from openai import OpenAI\n\nclient = OpenAI()\n\n\ndef ask(messages, schema):\n    return {call}\n\n\nmessages, schema = [], None\nr = {call}\n"
    );
    let app = "from typing import Literal\n\nfrom pydantic import BaseModel\n\nfrom llm import ask\n\n\nclass A(BaseModel):\n    a: Literal[\"x\", \"y\"]\n\n\ndef go(t):\n    return ask([{\"role\": \"user\", \"content\": t}], A)\n";
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "llm.py", &llm);
    write(dir.path(), "app.py", app);
    let report = scan_decision_sites(dir.path());
    let hidden = site(&report, "llm.py", 7);
    let direct = site(&report, "llm.py", 11);
    assert_eq!(hidden.kind, SiteKind::WrapperDefinition);
    assert_eq!(direct.kind, SiteKind::Occurrence);
    assert_eq!(hidden.id, format!("{}-2", direct.id));

    // Without the caller the wrapper is a NoCallers occurrence and takes the base id.
    fs::remove_file(dir.path().join("app.py")).unwrap();
    let report = scan_decision_sites(dir.path());
    let unrepresented = site(&report, "llm.py", 7);
    assert_eq!(
        unrepresented.provenance.disposition,
        Some(Disposition::NoCallers)
    );
    assert_eq!(unrepresented.id, direct.id);
    assert_eq!(site(&report, "llm.py", 11).id, format!("{}-2", direct.id));
}

const DEEP: &str = "from typing import Literal\n\nfrom openai import OpenAI\nfrom pydantic import BaseModel\n\nclient = OpenAI()\n\n\nclass Category(BaseModel):\n    kind: Literal[\"billing\", \"bug\"]\n\n\ndef screen(text, schema):\n    return client.chat.completions.parse(model=\"gpt-4o\", messages=[{\"role\": \"user\", \"content\": text}], response_format=schema)\n\n\ndef one(text, schema):\n    return screen(text, schema)\n\n\ndef two(text, schema):\n    return one(text, schema)\n\n\ndef three(text, schema):\n    return two(text, schema)\n\n\nclass Fast:\n    def categorize(self, text, schema):\n        return three(text, schema)\n\n\nclass Careful:\n    def categorize(self, text, schema):\n        return three(text, schema)\n\n\ndef label(classifier, ticket):\n    return classifier.categorize(ticket.body, Category)\n";

#[test]
fn depth_exceeded_revision_hashes_every_blocked_chain() {
    let scan_with = |text: &str| {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "deep.py", text);
        scan_decision_sites(dir.path())
    };
    let report = scan_with(DEEP);
    let deep = site(&report, "deep.py", 40);
    assert_eq!(
        deep.provenance.disposition,
        Some(Disposition::DepthExceeded)
    );
    assert_eq!(deep.provenance.blocked.len(), 2);
    let base = revision(&report, "deep.py", 40);
    assert_eq!(scan_with(DEEP).sites, report.sites);
    // Only the second (Careful) wrapper's call changes; the depth-5 call is untouched.
    let careful = "class Careful:\n    def categorize(self, text, schema):\n        return three(text, schema)\n";
    let edited = DEEP.replace(
        careful,
        "class Careful:\n    def categorize(self, text, schema):\n        return three(text.strip(), schema)\n",
    );
    assert_ne!(edited, DEEP);
    let edited = scan_with(&edited);
    assert_eq!(
        site(&edited, "deep.py", 40).provenance.disposition,
        Some(Disposition::DepthExceeded)
    );
    assert_ne!(revision(&edited, "deep.py", 40), base);
}

/// Rust validation is strict and accepts exactly what the schema accepts: every case below
/// is either accepted by both or rejected by both.
#[test]
fn schema_and_from_json_agree_on_strictness() {
    use serde_json::json;
    let validator = validator();
    let agent: Value = serde_json::from_str(&example("agent")).unwrap();
    let runtime: Value = serde_json::from_str(&example("runtime")).unwrap();
    let report: Value = serde_json::from_str(&example("report")).unwrap();
    let source = report["sites"][0].clone();
    assert!(source["provenance"]["alternatives"][0]["edge"]["bindings"][0].is_object());
    let with_codes = report["sites"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["provenance"].get("reason_codes").is_some())
        .unwrap()
        .clone();
    let edge = "/provenance/alternatives/0/edge";
    let set = |pointer: &str, value: Value| {
        let pointer = pointer.to_string();
        Box::new(move |v: &mut Value| *v.pointer_mut(&pointer).unwrap() = value.clone()) as Change
    };
    let insert = |pointer: &str, key: &str, value: Value| {
        let (pointer, key) = (pointer.to_string(), key.to_string());
        Box::new(move |v: &mut Value| {
            v.pointer_mut(&pointer)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .insert(key.clone(), value.clone());
        }) as Change
    };
    type Change = Box<dyn Fn(&mut Value)>;
    let site_cases: Vec<(&str, &Value, Change, bool)> = vec![
        (
            "version 1.00",
            &runtime,
            set("/schema_version", json!("1.00")),
            false,
        ),
        (
            "version 01.0",
            &runtime,
            set("/schema_version", json!("01.0")),
            false,
        ),
        (
            "version 1.01",
            &runtime,
            set("/schema_version", json!("1.01")),
            false,
        ),
        (
            "version 1.12",
            &runtime,
            set("/schema_version", json!("1.12")),
            false,
        ),
        (
            "version with a huge minor",
            &runtime,
            set("/schema_version", json!("1.123456789012345678901234567890")),
            false,
        ),
        (
            "runtime empty callers",
            &runtime,
            insert("/provenance", "callers", json!([])),
            false,
        ),
        (
            "agent empty folded",
            &agent,
            insert("/provenance", "folded", json!([])),
            false,
        ),
        (
            "source empty callers",
            &source,
            insert("/provenance", "callers", json!([])),
            false,
        ),
        (
            "unknown top-level field",
            &runtime,
            insert("", "extra", json!(1)),
            false,
        ),
        (
            "unknown source field",
            &source,
            insert("/source", "extra", json!(1)),
            false,
        ),
        (
            "unknown agent field",
            &agent,
            insert("/agent", "extra", json!(1)),
            false,
        ),
        (
            "unknown provenance field",
            &source,
            insert("/provenance", "extra", json!(1)),
            false,
        ),
        (
            "unknown edge field",
            &source,
            insert(edge, "extra", json!(1)),
            false,
        ),
        (
            "unknown output field",
            &runtime,
            insert("/outputs/0", "extra", json!(1)),
            false,
        ),
        (
            "unknown label field",
            &runtime,
            insert("/outputs/0/space/options/0", "extra", json!(1)),
            false,
        ),
        (
            "unknown noul field",
            &runtime,
            set("/outputs/0/space", json!({"kind": "noul", "extra": 1})),
            false,
        ),
        (
            "noul space",
            &runtime,
            set("/outputs/0/space", json!({"kind": "noul"})),
            true,
        ),
        (
            "unknown input_schema field",
            &runtime,
            insert("", "input_schema", json!({"fields": [], "extra": 1})),
            false,
        ),
        (
            "unknown input field",
            &runtime,
            insert(
                "",
                "input_schema",
                json!({"fields": [{"name": "a", "description": "", "kind": "text", "required": true, "extra": 1}]}),
            ),
            false,
        ),
        ("model null", &runtime, set("/model", json!(null)), false),
        (
            "runtime agent null",
            &runtime,
            insert("", "agent", json!(null)),
            false,
        ),
        (
            "runtime source null",
            &runtime,
            insert("", "source", json!(null)),
            false,
        ),
        (
            "prompt null",
            &runtime,
            insert("", "prompt", json!(null)),
            false,
        ),
        (
            "output name null",
            &runtime,
            set("/outputs/0/name", json!(null)),
            false,
        ),
        (
            "disposition null",
            &source,
            set("/provenance/disposition", json!(null)),
            false,
        ),
        (
            "source max_tokens null",
            &source,
            insert("/source", "max_tokens", json!(null)),
            false,
        ),
        (
            "revision abc",
            &source,
            set("/source_revision", json!("abc")),
            false,
        ),
        (
            "revision uppercase",
            &source,
            set("/source_revision", json!("A".repeat(64))),
            false,
        ),
        (
            "revision 64 hex",
            &source,
            set("/source_revision", json!("0".repeat(64))),
            true,
        ),
        (
            "source line 0",
            &source,
            set("/source/line", json!(0)),
            false,
        ),
        (
            "occurrence line 0",
            &source,
            set("/provenance/occurrence/line", json!(0)),
            false,
        ),
        (
            "edge caller line 0",
            &source,
            set(&format!("{edge}/caller/line"), json!(0)),
            false,
        ),
        (
            "edge depth 0",
            &source,
            set(&format!("{edge}/depth"), json!(0)),
            false,
        ),
        (
            "edge id not hex",
            &source,
            set(&format!("{edge}/id"), json!("xyz")),
            false,
        ),
        (
            "edge id suffix",
            &source,
            set(&format!("{edge}/id"), json!("0123456789ab-2")),
            true,
        ),
        (
            "binding slot",
            &source,
            set(&format!("{edge}/bindings/0/slot"), json!("param:x")),
            false,
        ),
        (
            "binding prop slot with newline",
            &source,
            set(&format!("{edge}/bindings/0/slot"), json!("prop:0:a\nb")),
            false,
        ),
        (
            "binding prop slot",
            &source,
            set(&format!("{edge}/bindings/0/slot"), json!("prop:0:a b")),
            true,
        ),
        (
            "unknown reason code",
            &with_codes,
            set("/provenance/reason_codes", json!(["whatever"])),
            false,
        ),
        (
            "empty reason codes",
            &with_codes,
            set("/provenance/reason_codes", json!([])),
            false,
        ),
        (
            "id newline token",
            &source,
            set("/id", json!("source:\n")),
            false,
        ),
        (
            "id empty token",
            &runtime,
            set("/id", json!("runtime:")),
            false,
        ),
        (
            "id with a space",
            &runtime,
            set("/id", json!("runtime:a b")),
            false,
        ),
        (
            "id with a tab",
            &agent,
            set("/id", json!("agent:a\tb")),
            false,
        ),
        (
            "id with DEL",
            &runtime,
            set("/id", json!("runtime:a\u{7f}")),
            false,
        ),
        (
            "id with NEL",
            &runtime,
            set("/id", json!("runtime:a\u{85}")),
            false,
        ),
        (
            "id with nbsp",
            &runtime,
            set("/id", json!("runtime:a\u{a0}")),
            false,
        ),
        (
            "id with BOM",
            &runtime,
            set("/id", json!("runtime:a\u{feff}")),
            false,
        ),
        (
            "id with ideographic space",
            &runtime,
            set("/id", json!("runtime:a\u{3000}")),
            false,
        ),
        (
            "id with non-ASCII letters",
            &runtime,
            set("/id", json!("runtime:catégorie-1")),
            true,
        ),
        (
            "id with punctuation",
            &agent,
            set("/id", json!("agent:claude-code/triage.v2")),
            true,
        ),
    ];
    for (name, base, change, valid) in site_cases {
        let mut value = base.clone();
        change(&mut value);
        let rust = DecisionSite::from_json(&value.to_string());
        assert_eq!(validator.is_valid(&value), valid, "schema: {name}");
        assert_eq!(rust.is_ok(), valid, "from_json: {name}: {rust:?}");
    }

    let report_cases: Vec<(&str, Change)> = vec![
        ("unknown report field", insert("", "extra", json!(1))),
        (
            "unknown summary field",
            insert("/summary", "extra", json!(1)),
        ),
        ("report version 1.00", set("/schema_version", json!("1.00"))),
        ("site model null", set("/sites/1/model", json!(null))),
    ];
    for (name, change) in report_cases {
        let mut value = report.clone();
        change(&mut value);
        assert!(!validator.is_valid(&value), "schema: {name}");
        assert!(
            DecisionSiteReport::from_json(&value.to_string()).is_err(),
            "from_json: {name}"
        );
    }

    // The valid examples and generated reports still pass both.
    for name in ["agent", "runtime"] {
        let text = example(name);
        assert!(
            validator.is_valid(&serde_json::from_str(&text).unwrap()),
            "{name}"
        );
        DecisionSite::from_json(&text).unwrap();
    }
    assert!(validator.is_valid(&report));
    DecisionSiteReport::from_json(&example("report")).unwrap();
    for root in ROOTS {
        let text = pretty(&scan_decision_sites(Path::new(root)));
        assert!(
            validator.is_valid(&serde_json::from_str(&text).unwrap()),
            "{root}"
        );
        DecisionSiteReport::from_json(&text).unwrap();
    }
}
