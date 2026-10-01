use indexmap::IndexMap;
use serde_json::json;

use super::*;
use crate::model::{AnswerSpace, Label};

fn label_field() -> OutputField {
    OutputField {
        name: Some("label".into()),
        description: None,
        space: AnswerSpace::Choice {
            options: vec![Label::new("spam"), Label::new("ham")],
            nullable: false,
        },
    }
}

fn evidence() -> Evidence {
    Evidence {
        file: "a.py".into(),
        call_text: "ask(text)".into(),
        lang: Lang::Python,
        sdk: Sdk::Openai,
        api: "chat.completions.parse".into(),
        model: Some("gpt-4o".into()),
        max_tokens: None,
        prompt_parts: vec![PromptEvidence {
            key: "messages".into(),
            text: Some("Is this spam?".into()),
            dynamic: true,
        }],
        schema_fields: SchemaEvidence::Resolved {
            fields: vec![label_field()],
            free_text: vec![],
        },
        chain: vec![ChainLink {
            file: "llm.py".into(),
            function: "ask".into(),
            call_text: "client.chat.completions.parse(model='gpt-4o')".into(),
        }],
        alternatives: vec![],
        blocked: vec![],
    }
}

fn call_site(file: &str, line: usize, id: &str) -> CallSite {
    CallSite {
        id: id.into(),
        file: file.into(),
        line,
        lang: Lang::Python,
        sdk: Sdk::Openai,
        api: "chat.completions.parse".into(),
        via: vec![],
        model: None,
        tier: Tier::Sure,
        outputs: vec![label_field()],
        reasons: vec!["closed-set schema: 1 field(s)".into()],
        prompt: None,
        max_tokens: None,
        drafts: vec![],
    }
}

fn record(file: &str, line: usize, byte: usize, id: &str, kind: SiteKind) -> SourceRecord {
    SourceRecord {
        site: call_site(file, line, id),
        kind,
        byte_offset: byte,
        evidence: evidence(),
        provenance: Provenance::default(),
    }
}

fn source_site() -> DecisionSite {
    let report = from_scan(SourceScan {
        root: "repo".into(),
        files_scanned: 1,
        files_skipped: 0,
        records: vec![record("a.py", 3, 40, "abcdefabcdef", SiteKind::Occurrence)],
    });
    report.sites.into_iter().next().unwrap()
}

fn agent_site() -> DecisionSite {
    DecisionSite {
        schema_version: "1.0".into(),
        id: "agent:claude-code:task-route".into(),
        source_revision: None,
        origin: Origin::AgentHook,
        source: None,
        agent: Some(AgentOrigin {
            host: "claude-code".into(),
            adapter_version: "0.1.0".into(),
            event: "PreToolUse".into(),
            definition_id: "task-route".into(),
        }),
        model: None,
        prompt: None,
        outputs: vec![label_field()],
        kind: SiteKind::Occurrence,
        tier: Tier::Sure,
        reasons: vec![],
        drafts: vec![],
        input_schema: Some(InputSchema {
            fields: vec![InputField {
                name: "task".into(),
                description: "Task description".into(),
                kind: "string".into(),
                required: true,
            }],
        }),
        provenance: Provenance::default(),
        jev_review: None,
    }
}

fn runtime_site() -> DecisionSite {
    DecisionSite {
        id: "runtime:ticket-category".into(),
        origin: Origin::Runtime,
        agent: None,
        ..agent_site()
    }
}

#[test]
fn valid_sites_of_every_origin() {
    for site in [source_site(), agent_site(), runtime_site()] {
        assert_eq!(site.validate(), Ok(()), "{site:?}");
    }
}

#[test]
fn source_rules() {
    let err = |change: fn(&mut DecisionSite)| {
        let mut site = source_site();
        change(&mut site);
        site.validate().unwrap_err()
    };
    let origin = Origin::Source;
    assert_eq!(
        err(|s| s.source = None),
        SiteError::MissingField {
            origin,
            field: "source"
        }
    );
    assert_eq!(
        err(|s| s.agent = agent_site().agent),
        SiteError::UnexpectedField {
            origin,
            field: "agent"
        }
    );
    assert_eq!(
        err(|s| s.source_revision = None),
        SiteError::MissingField {
            origin,
            field: "source_revision"
        }
    );
    assert!(matches!(
        err(|s| s.id = "agent:x".into()),
        SiteError::IdPrefix { .. }
    ));
    assert!(matches!(
        err(|s| s.id = "source:".into()),
        SiteError::IdPrefix { .. }
    ));
    // A wrapper definition is a valid Source site.
    let mut site = source_site();
    site.kind = SiteKind::WrapperDefinition;
    assert_eq!(site.validate(), Ok(()));
}

#[test]
fn agent_hook_rules() {
    let err = |change: fn(&mut DecisionSite)| {
        let mut site = agent_site();
        change(&mut site);
        site.validate().unwrap_err()
    };
    let origin = Origin::AgentHook;
    assert_eq!(
        err(|s| s.agent = None),
        SiteError::MissingField {
            origin,
            field: "agent"
        }
    );
    assert_eq!(
        err(|s| s.source = source_site().source),
        SiteError::UnexpectedField {
            origin,
            field: "source"
        }
    );
    assert_eq!(
        err(|s| s.source_revision = Some("00".into())),
        SiteError::UnexpectedField {
            origin,
            field: "source_revision"
        }
    );
    assert!(matches!(
        err(|s| s.id = "runtime:x".into()),
        SiteError::IdPrefix { .. }
    ));
    assert_eq!(
        err(|s| s.kind = SiteKind::WrapperDefinition),
        SiteError::KindNotAllowed(origin)
    );
    assert_eq!(
        err(|s| s.provenance.reason_codes = vec!["no_callers".into()]),
        SiteError::UnexpectedProvenance(origin)
    );
}

#[test]
fn runtime_rules() {
    let err = |change: fn(&mut DecisionSite)| {
        let mut site = runtime_site();
        change(&mut site);
        site.validate().unwrap_err()
    };
    let origin = Origin::Runtime;
    assert_eq!(
        err(|s| s.agent = agent_site().agent),
        SiteError::UnexpectedField {
            origin,
            field: "agent"
        }
    );
    assert_eq!(
        err(|s| s.source = source_site().source),
        SiteError::UnexpectedField {
            origin,
            field: "source"
        }
    );
    assert_eq!(
        err(|s| s.source_revision = Some("00".into())),
        SiteError::UnexpectedField {
            origin,
            field: "source_revision"
        }
    );
    assert!(matches!(
        err(|s| s.id = "source:x".into()),
        SiteError::IdPrefix { .. }
    ));
    assert_eq!(
        err(|s| s.kind = SiteKind::WrapperDefinition),
        SiteError::KindNotAllowed(origin)
    );
    assert_eq!(
        err(|s| s.provenance.depth = Some(0)),
        SiteError::UnexpectedProvenance(origin)
    );
}

#[test]
fn versions_accept_supported_minor_versions_only() {
    for ok in ["1.0", "1.1"] {
        assert_eq!(check_site_version(ok), Ok(()), "{ok}");
    }
    for bad in ["2.0", "0.9", "1.2", "1.7", "1.12"] {
        assert_eq!(
            check_site_version(bad),
            Err(SiteError::IncompatibleVersion(bad.into()))
        );
    }
    for bad in [
        "1", "", "1.", ".0", "v1.0", "1.x", "+1.0", "1.00", "01.0", "1.01", "02.0", "1.0.0",
    ] {
        assert_eq!(
            check_site_version(bad),
            Err(SiteError::InvalidVersion(bad.into()))
        );
    }
    let mut site = source_site();
    site.schema_version = "2.0".into();
    assert_eq!(
        site.validate(),
        Err(SiteError::IncompatibleVersion("2.0".into()))
    );
}

#[test]
fn jev_review_requires_schema_minor_one() {
    let mut site = source_site();
    site.jev_review = Some(JevReview {
        probability: 0.8,
        suggestion: "likely".into(),
        model: "jev-1.13.0".into(),
        request_id: Some("req-1".into()),
        questions: vec![JevSuggestion {
            subject: JevReviewSubject::BoundedDecision,
            answer: "yes".into(),
            probabilities: IndexMap::from([("yes".into(), 0.8), ("no".into(), 0.2)]),
        }],
    });
    assert!(site.validate().is_err());

    site.schema_version = "1.1".into();
    assert_eq!(site.validate(), Ok(()));
    let text = serde_json::to_string(&site).unwrap();
    assert_eq!(DecisionSite::from_json(&text), Ok(site));
}

#[test]
fn from_json_rejects_other_majors_before_reading_the_shape() {
    let v2 = json!({"schema_version": "2.0", "id": "source:x", "shape": "unknown"}).to_string();
    assert_eq!(
        DecisionSite::from_json(&v2),
        Err(SiteError::IncompatibleVersion("2.0".into()))
    );
    let report = json!({"schema": "decision-site-v1", "schema_version": "3.1"}).to_string();
    assert_eq!(
        DecisionSiteReport::from_json(&report),
        Err(SiteError::IncompatibleVersion("3.1".into()))
    );
    let site = source_site();
    let text = serde_json::to_string(&site).unwrap();
    assert_eq!(DecisionSite::from_json(&text), Ok(site));
    let mut invalid = serde_json::to_value(agent_site()).unwrap();
    invalid["source_revision"] = json!("00");
    assert!(matches!(
        DecisionSite::from_json(&invalid.to_string()),
        Err(SiteError::UnexpectedField { .. })
    ));
    assert!(matches!(
        DecisionSite::from_json("{"),
        Err(SiteError::Json(_))
    ));
}

#[test]
fn serde_omits_absent_fields_and_keeps_required_vectors() {
    let mut site = runtime_site();
    site.outputs.clear();
    site.input_schema = None;
    let value = serde_json::to_value(&site).unwrap();
    let keys: Vec<&str> = value
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        [
            "schema_version",
            "id",
            "origin",
            "outputs",
            "kind",
            "tier",
            "reasons",
            "drafts",
            "provenance"
        ]
    );
    assert_eq!(value["provenance"], json!({}));
    assert_eq!(value["origin"], json!("runtime"));

    let source = serde_json::to_value(source_site()).unwrap();
    assert_eq!(source["kind"], json!("occurrence"));
    assert_eq!(source["source"]["via"], json!([]));
    assert!(source["source"].get("max_tokens").is_none());
    assert!(source.get("agent").is_none());
    let agent = serde_json::to_value(agent_site()).unwrap();
    assert_eq!(agent["origin"], json!("agent_hook"));
    assert_eq!(agent["input_schema"]["fields"][0]["kind"], json!("string"));

    let provenance = Provenance {
        disposition: Some(Disposition::MultipleDecisions),
        reason_codes: vec!["wrapper_multiple_decisions".into()],
        ..Provenance::default()
    };
    assert_eq!(
        serde_json::to_value(&provenance).unwrap(),
        json!({"disposition": "multiple_decisions", "reason_codes": ["wrapper_multiple_decisions"]})
    );
    assert_eq!(
        serde_json::to_value(SiteKind::WrapperDefinition).unwrap(),
        json!("wrapper_definition")
    );
    assert_eq!(
        serde_json::to_value(EdgeKind::DepthExceeded).unwrap(),
        json!("depth_exceeded")
    );
}

#[test]
fn owned_serialization_round_trips() {
    for site in [source_site(), agent_site(), runtime_site()] {
        let text = serde_json::to_string_pretty(&site).unwrap();
        let back: DecisionSite = serde_json::from_str(&text).unwrap();
        assert_eq!(back, site);
    }
}

#[test]
fn source_ids_are_namespaced_legacy_ids_with_duplicate_suffixes() {
    let report = from_scan(SourceScan {
        root: "repo".into(),
        files_scanned: 2,
        files_skipped: 0,
        records: vec![
            record("b.py", 9, 90, "111111111111", SiteKind::Occurrence),
            // Hidden wrapper earlier in the file than the emitted duplicates.
            record("a.py", 2, 10, "aaaaaaaaaaaa", SiteKind::WrapperDefinition),
            record("a.py", 5, 50, "aaaaaaaaaaaa", SiteKind::Occurrence),
            record("a.py", 5, 70, "aaaaaaaaaaaa", SiteKind::Occurrence),
            record("a.py", 7, 80, "222222222222", SiteKind::WrapperDefinition),
        ],
    });
    let got: Vec<(&str, SiteKind)> = report
        .sites
        .iter()
        .map(|s| (s.id.as_str(), s.kind))
        .collect();
    assert_eq!(
        got,
        [
            ("source:aaaaaaaaaaaa-3", SiteKind::WrapperDefinition),
            ("source:aaaaaaaaaaaa", SiteKind::Occurrence),
            ("source:aaaaaaaaaaaa-2", SiteKind::Occurrence),
            ("source:222222222222", SiteKind::WrapperDefinition),
            ("source:111111111111", SiteKind::Occurrence),
        ]
    );
    // Equal (file, line, raw id) keeps analysis order: byte 50 before byte 70.
    assert_eq!(report.sites[1].source.as_ref().unwrap().byte_offset, 50);
    assert_eq!(report.validate(), Ok(()));

    let legacy = legacy_report(&report);
    let ids: Vec<&str> = legacy.sites.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(ids, ["aaaaaaaaaaaa", "aaaaaaaaaaaa-2", "111111111111"]);
    assert_eq!(legacy.summary.call_sites, 3);
    assert_eq!(report.summary.wrapper_definitions, 2);
    assert_eq!(report.summary.legacy, legacy.summary);
}

#[test]
fn hook_and_runtime_sites_are_never_counted_or_projected() {
    let mut report = from_scan(SourceScan {
        root: "repo".into(),
        files_scanned: 1,
        files_skipped: 0,
        records: vec![record("a.py", 3, 40, "abcdefabcdef", SiteKind::Occurrence)],
    });
    let before = report.summary.clone();
    report.sites.push(agent_site());
    report.sites.push(runtime_site());
    assert_eq!(report.validate(), Ok(()));
    assert_eq!(report.summary, before);
    assert_eq!(legacy_report(&report).sites.len(), 1);
    assert!(matches!(
        CallSite::try_from(&agent_site()),
        Err(SiteError::NotLegacy(_))
    ));

    let mut wrong = report.clone();
    wrong.summary.legacy.sure += 1;
    assert_eq!(wrong.validate(), Err(SiteError::SummaryMismatch));
    let mut wrong = report.clone();
    wrong.summary.wrapper_definitions = 1;
    assert_eq!(wrong.validate(), Err(SiteError::SummaryMismatch));
    let mut wrong = report;
    wrong.schema = "legacy".into();
    assert_eq!(
        wrong.validate(),
        Err(SiteError::UnknownSchema("legacy".into()))
    );
}

#[test]
fn generic_summary_flattens_the_legacy_fields() {
    let report = from_scan(SourceScan {
        root: "repo".into(),
        files_scanned: 1,
        files_skipped: 0,
        records: vec![
            record("a.py", 3, 40, "abcdefabcdef", SiteKind::Occurrence),
            record("a.py", 1, 5, "000000000000", SiteKind::WrapperDefinition),
        ],
    });
    let value = serde_json::to_value(&report.summary).unwrap();
    let legacy = serde_json::to_value(&report.summary.legacy).unwrap();
    let mut expected = legacy.as_object().unwrap().clone();
    expected.insert("wrapper_definitions".into(), json!(1));
    assert_eq!(value, serde_json::Value::Object(expected));
    let back: GenericSummary = serde_json::from_value(value).unwrap();
    assert_eq!(back, report.summary);
}

#[test]
fn revision_is_evidence_only() {
    let base = evidence();
    let revision = base.revision();
    assert_eq!(revision.len(), 64);
    assert!(
        revision
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
    );
    // The hash input holds no scanner judgment and no positions.
    let input: serde_json::Value = serde_json::from_str(&base.canonical_json()).unwrap();
    let keys: Vec<&str> = input
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        [
            "file",
            "call_text",
            "lang",
            "sdk",
            "api",
            "model",
            "max_tokens",
            "prompt_parts",
            "schema_fields",
            "chain",
            "alternatives"
        ]
    );
    let changed = |change: fn(&mut Evidence)| {
        let mut other = evidence();
        change(&mut other);
        other.revision()
    };
    assert_ne!(changed(|e| e.model = Some("gpt-4o-mini".into())), revision);
    assert_ne!(
        changed(|e| e.prompt_parts[0].text = Some("Is this ham?".into())),
        revision
    );
    assert_ne!(changed(|e| e.chain[0].call_text.push('x')), revision);
    assert_ne!(
        changed(|e| e.schema_fields = SchemaEvidence::None),
        revision
    );
}

#[test]
fn revision_ignores_alternative_order() {
    let alternative = |model: &str| AlternativeEvidence {
        sdk: Sdk::Openai,
        api: "chat.completions.parse".into(),
        model: Some(model.into()),
        max_tokens: None,
        prompt_parts: vec![],
        schema_fields: SchemaEvidence::Unresolved,
        chain: vec![],
        bindings: vec![BindingRecord {
            key: "response_format".into(),
            role: BindingRole::Schema,
            slot: "param:1".into(),
        }],
    };
    let mut a = evidence();
    a.alternatives = vec![alternative("x"), alternative("y")];
    let mut b = evidence();
    b.alternatives = vec![alternative("y"), alternative("x")];
    assert_eq!(a.revision(), b.revision());
    b.alternatives[0].model = Some("z".into());
    assert_ne!(a.revision(), b.revision());
}

#[test]
fn edge_ids_are_text_based() {
    let chain = [("llm.py", "client.parse(model = 'x')")];
    let id = edge_id(("a.py", "ask(t, S)"), &chain);
    assert_eq!(id.len(), 12);
    assert_eq!(id, edge_id(("a.py", "ask( t,\n S )"), &chain));
    assert_ne!(id, edge_id(("b.py", "ask(t, S)"), &chain));
    assert_ne!(id, edge_id(("a.py", "ask(t, S)"), &[]));
    assert_ne!(
        id,
        edge_id(
            ("a.py", "ask(t, S)"),
            &[("llm.py", "client.parse(model='y')")]
        )
    );
}

#[test]
fn values_outside_the_schema_are_rejected() {
    let err = |change: fn(&mut DecisionSite)| {
        let mut site = source_site();
        change(&mut site);
        site.validate().unwrap_err()
    };
    assert!(matches!(
        err(|s| s.source_revision = Some("abc".into())),
        SiteError::InvalidValue {
            field: "source_revision",
            ..
        }
    ));
    assert!(matches!(
        err(|s| s.source.as_mut().unwrap().line = 0),
        SiteError::InvalidValue {
            field: "source.line",
            ..
        }
    ));
    assert!(matches!(
        err(|s| s.provenance.reason_codes = vec!["whatever".into()]),
        SiteError::InvalidValue {
            field: "reason_codes",
            ..
        }
    ));
    for id in [
        "source:\n",
        "source:a b",
        "source:a\u{feff}",
        "source:\u{85}",
    ] {
        let mut site = source_site();
        site.id = id.into();
        assert!(
            matches!(site.validate(), Err(SiteError::IdPrefix { .. })),
            "{id:?}"
        );
    }
    for code in REASON_CODES {
        let mut site = source_site();
        site.provenance.reason_codes = vec![code.to_string()];
        assert_eq!(site.validate(), Ok(()), "{code}");
    }
}
