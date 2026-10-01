//! The canonical eval export fixture (Task 8 frozen decision 2): `schemas/examples/eval-export/
//! ticket-triage/` is the export of a hand-made, deterministic `SiteRun` (no provider, no
//! cache). The committed files must equal a fresh export, the policy must pass the
//! `policy-v1` schema and the Task 7 validators, `eval --verify` logic must accept the
//! directory, and the report's honest numbers are snapshotted. Set `SNAPJUDGE_BLESS=1` to
//! rewrite the fixture after an intended change.

use std::fs;
use std::path::Path;
use std::time::{Duration, UNIX_EPOCH};

use indexmap::IndexMap;
use serde_json::{Value, json};
use snapjudge::decision::{DecisionDefinition, EvidenceKind, GatePolicy, jcs};
use snapjudge::eval::answers;
use snapjudge::eval::export::{self, Context, Options, PriceRow, RunInfo};
use snapjudge::eval::inputs::{InputOrigin, Split, input_hash};
use snapjudge::eval::ledger::Totals;
use snapjudge::eval::pipeline::Estimate;
use snapjudge::eval::run::{
    Call, Failure, FailureKind, InputCounts, InputRecord, JevAnswer, Outcome, ReferenceAnswer,
    ReferenceSetup, ReferenceSource, SiteRun, SiteStatus, StageCall,
};
use snapjudge::judge::Answer;
use snapjudge::llm::catalogue::Catalogue;

const FIXTURE: &str = "schemas/examples/eval-export/ticket-triage";
const FILES: [&str; 5] = [
    "dataset.jsonl",
    "results.json",
    "policy.json",
    "report.md",
    "run.json",
];
const TEACHER: &str = "openai/gpt-4o-mini";
const DESIGNER: &str = "anthropic/claude-sonnet-4.5";
const JEV: &str = "jev-1.13.0";
const TEAMS: [Option<&str>; 3] = [Some("billing"), Some("support"), None];
const LEVELS: [f64; 5] = [-2.0, -1.0, 0.0, 1.0, 2.0];

fn definition() -> DecisionDefinition {
    DecisionDefinition::from_json(
        &fs::read_to_string("schemas/examples/definition-v1.ticket-triage.json").unwrap(),
    )
    .unwrap()
}

fn call(model: &str, cost: f64, ms: u64, id: String) -> Call {
    Call {
        model: model.into(),
        served_model: model.into(),
        request_id: Some(id),
        usage: None,
        cost_usd: Some(cost),
        duration_ms: ms,
        cache_hit: false,
        price_usd_per_mtok: None,
    }
}

/// Jev's normalized answers: `team` (None = the nullable option), `urgent`, the tags, the
/// sentiment level, all at gate confidence `gate`.
fn jev_answers(team: Option<&str>, urgent: bool, tags: &[&str], level: usize, gate: f64) -> Value {
    let label = |name: &str, cutoff: f64| {
        let on = tags.contains(&name);
        json!({"value": on, "probability_yes": if on { gate } else { 1.0 - gate },
               "cutoff": cutoff, "gate_confidence": gate, "passed": true})
    };
    let mut probabilities =
        json!({"billing": 0.05, "support": 0.05, "sales": 0.05, "none_of_the_above": 0.05});
    probabilities[team.unwrap_or("none_of_the_above")] = json!(gate);
    json!({
        "team": {"type": "choice", "value": team, "probabilities": probabilities,
                 "confidence": gate, "gate_confidence": gate, "passed": true, "reasons": []},
        "urgent": {"type": "noul", "value": urgent,
                   "probability_yes": if urgent { gate } else { 1.0 - gate }, "cutoff": 0.5,
                   "gate_confidence": gate, "passed": true, "reasons": []},
        "tags": {"type": "multilabel", "value": tags,
                 "labels": {"billing": label("billing", 0.5), "bug": label("bug", 0.5),
                            "security": label("security", 0.3)},
                 "passed": true, "reasons": []},
        "sentiment": {"type": "score", "value": LEVELS[level], "score": level as f64,
                      "level": level, "probabilities": {}, "confidence": gate,
                      "gate_confidence": gate, "passed": true, "reasons": []},
    })
}

/// 80 synthetic tickets: a third of them (by index) answered wrongly by Jev on `team` at
/// gate confidence 0.55, the rest right at 0.6–0.9; one refused reference and one failed
/// Jev call.
fn fixture_run() -> SiteRun {
    let definition = definition();
    let mut records: Vec<InputRecord> = (0..80)
        .map(|i| {
            let category = i % 3;
            let team = TEAMS[category];
            let urgent = i % 4 == 0;
            let tags: Vec<&str> = match category {
                0 => vec!["billing"],
                1 => vec!["bug"],
                _ => vec![],
            };
            let level = i % 5;
            let input = json!({
                "subject": format!("Ticket {i:02}"),
                "body": format!("{} ticket body {i:02}", team.unwrap_or("general")),
            });
            let reference_values = answers::parse_values(
                &definition,
                &json!({"team": team, "urgent": urgent, "tags": tags, "sentiment": LEVELS[level]}),
            )
            .unwrap();
            let wrong = (i / 3) % 3 == 0;
            let (jev_team, gate) = if wrong {
                (
                    if team == Some("billing") {
                        Some("support")
                    } else {
                        Some("billing")
                    },
                    0.55,
                )
            } else {
                (team, 0.6 + (i % 4) as f64 * 0.1)
            };
            let answers: IndexMap<String, Answer> =
                serde_json::from_value(jev_answers(jev_team, urgent, &tags, level, gate)).unwrap();
            let reference = if i == 79 {
                Outcome::Failed(Failure::new(
                    FailureKind::TeacherInvalid,
                    "no answer content",
                ))
            } else {
                Outcome::Ok(ReferenceAnswer {
                    values: reference_values.clone(),
                    call: Some(call(
                        TEACHER,
                        0.000_2,
                        800 + i as u64,
                        format!("gen-{i:02}"),
                    )),
                })
            };
            let jev = if i == 78 {
                Outcome::Failed(Failure::new(FailureKind::JevFailed, "request timed out"))
            } else {
                Outcome::Ok(JevAnswer {
                    values: answers::jev_values(&answers),
                    gate_confidence: answers::row_gate_confidence(&definition, &answers),
                    call: Call {
                        price_usd_per_mtok: Some(0.042),
                        ..call(JEV, 0.000_01, 120 + i as u64, format!("ts-{i:02}"))
                    },
                    answers: answers.clone(),
                })
            };
            let agrees = (i != 78 && i != 79)
                .then(|| answers::jev_agrees(&definition, &reference_values, &answers));
            InputRecord {
                input_hash: input_hash(&input),
                input,
                origin: InputOrigin::Synthetic,
                occurrences: 1,
                conflicting_labels: false,
                split: Split::HeldOut,
                reference: Some(reference),
                jev: Some(jev),
                agrees,
                self_agreement: None,
                adjudication: None,
            }
        })
        .collect();
    records.sort_by(|a, b| a.input_hash.cmp(&b.input_hash));
    for record in &mut records[..40] {
        record.split = Split::Calibration;
    }
    SiteRun {
        site_id: definition.site_id.clone(),
        definition_id: definition.id.clone(),
        status: SiteStatus::Completed,
        message: None,
        definition: Some(definition),
        jev_model: JEV.into(),
        reference: Some(ReferenceSetup {
            source: ReferenceSource::Model,
            model: Some(TEACHER.into()),
            detected_model: Some("gpt-4o-mini".into()),
            teacher_assumed: false,
            teacher_override: false,
            reconstructed: false,
            max_completion_tokens: 300,
            temperature: Some(0.0),
            top_p: None,
            seed: None,
        }),
        adjudicator: None,
        target: 0.8,
        min_accepted: 20,
        inputs: Some(InputOrigin::Synthetic),
        counts: InputCounts {
            rows: 80,
            invalid: 0,
            duplicates: 0,
            unique: 80,
            calibration: 40,
            held_out: 40,
        },
        invalid_inputs: Vec::new(),
        stage_calls: vec![StageCall {
            stage: "designer".into(),
            outcome: Outcome::Ok(call(DESIGNER, 0.003, 4_000, "gen-designer".into())),
        }],
        records,
    }
}

fn catalogue() -> Catalogue {
    serde_json::from_value(json!({
        "retrieved": "2026-09-29",
        "source": "https://openrouter.ai/api/v1/models",
        "models": [
            {"id": DESIGNER, "prompt_usd_per_token": 0.000003, "completion_usd_per_token": 0.000015,
             "supported_parameters": ["max_tokens", "response_format", "structured_outputs"]},
            {"id": TEACHER, "prompt_usd_per_token": 0.00000015, "completion_usd_per_token": 0.0000006,
             "supported_parameters": ["max_tokens", "response_format", "structured_outputs", "temperature"]},
        ],
    }))
    .unwrap()
}

fn price(role: &'static str, model: &str, catalogue: &Catalogue) -> PriceRow {
    let info = catalogue.model(model).unwrap();
    PriceRow {
        role,
        model: model.into(),
        currency: "USD",
        per: "token",
        prompt: info.prompt_usd_per_token.unwrap(),
        completion: info.completion_usd_per_token,
        source: "catalogue",
        retrieved: Some(catalogue.retrieved.clone()),
        usage_basis: "usage.cost reported by the provider",
    }
}

fn context() -> Context {
    let catalogue = catalogue();
    Context {
        catalogue_source: catalogue.source.clone(),
        catalogue_retrieved: catalogue.retrieved.clone(),
        catalogue_sha256: catalogue.sha256().unwrap(),
        prices: vec![
            price("designer", DESIGNER, &catalogue),
            price("reference", TEACHER, &catalogue),
            PriceRow {
                role: "jev",
                model: JEV.into(),
                currency: "USD",
                per: "million_input_tokens",
                prompt: 0.042,
                completion: None,
                source: "configuration",
                retrieved: None,
                usage_basis: "usage.input_tokens × input_price_usd_per_mtok",
            },
        ],
        estimate: Estimate {
            usd: 0.025,
            worst_usd: 0.09,
            requests: 162,
            ..Estimate::default()
        },
        options: Options::default(),
    }
}

fn run_info() -> RunInfo {
    let started = UNIX_EPOCH + Duration::from_secs(1_790_640_000);
    RunInfo {
        started,
        finished: started + Duration::from_millis(12_345),
        llm_key: None,
        jev_key: None,
        replay_only: true,
        ledger: Totals {
            budget_usd: 1.0,
            ..Totals::default()
        },
    }
}

#[test]
fn canonical_export_fixture_is_current_valid_and_verifies() {
    let run = fixture_run();
    let export = export::build(&run, &context()).unwrap();
    let dir = Path::new(FIXTURE);
    if std::env::var_os("SNAPJUDGE_BLESS").is_some() {
        fs::create_dir_all(dir).unwrap();
        export::write(dir, &export, &export::run_json(&run_info(), &run)).unwrap();
    }
    let fresh = tempfile::tempdir().unwrap();
    export::write(fresh.path(), &export, &export::run_json(&run_info(), &run)).unwrap();
    for file in FILES {
        assert!(
            fs::read(fresh.path().join(file)).unwrap() == fs::read(dir.join(file)).unwrap(),
            "{FIXTURE}/{file} is stale (rerun with SNAPJUDGE_BLESS=1)"
        );
    }

    // The honest numbers the fixture documents.
    let results: Value =
        serde_json::from_slice(&fs::read(dir.join("results.json")).unwrap()).unwrap();
    assert_eq!(results["metrics"]["gate"]["k"], 12);
    let held = &results["metrics"]["held_out"];
    assert_eq!(held["agreement"]["count"], held["agreement"]["n"]);
    assert_eq!(held["agreement"]["held_out"], true);
    assert_eq!(held["agreement"]["inputs"], "synthetic");
    assert_eq!(results["policy"]["evidence"], "measured");

    // policy.json: the policy-v1 schema, the Task 7 validators, bound to the export.
    let text = fs::read_to_string(dir.join("policy.json")).unwrap();
    let schema: Value =
        serde_json::from_str(&fs::read_to_string("schemas/policy-v1.schema.json").unwrap())
            .unwrap();
    let validator = jsonschema::validator_for(&schema).unwrap();
    let value: Value = serde_json::from_str(&text).unwrap();
    let errors: Vec<String> = validator
        .iter_errors(&value)
        .map(|e| e.to_string())
        .collect();
    assert!(errors.is_empty(), "{errors:?}");
    let policy = GatePolicy::from_json(&text).unwrap();
    policy.validate_against(&definition()).unwrap();
    assert_eq!(policy.evidence, EvidenceKind::Measured);
    assert_eq!(policy.definition_revision, definition().revision());
    // Canonical JSON orders the keys.
    assert_eq!(
        policy.thresholds.keys().collect::<Vec<_>>(),
        [
            "sentiment",
            "tags.billing",
            "tags.bug",
            "tags.security",
            "team",
            "urgent"
        ]
    );
    assert!(policy.thresholds.values().all(|t| *t == 0.6));
    assert_eq!(
        policy.evidence_revision.as_deref(),
        results["evidence"]["revision"].as_str()
    );
    let dataset = fs::read(dir.join("dataset.jsonl")).unwrap();
    assert_eq!(
        policy.dataset.as_deref(),
        Some(jcs::sha256_hex(&dataset).as_str())
    );
    export::verify(dir).unwrap();
}

#[test]
fn report_of_the_canonical_fixture() {
    let export = export::build(&fixture_run(), &context()).unwrap();
    insta::assert_snapshot!(export.report);
}
