//! Task 7a contracts (redesign §4, §8; §14 row 7 "Protocol schemas"): definition, policy,
//! judge request and response schemas with their canonical examples, schema/Rust parity in
//! both directions, RFC 8785 revisions and the `state` validator.

use std::fs;

use serde_json::{Value, json};
use snapjudge::decision::{ContractError, DecisionDefinition, GatePolicy, jcs};
use snapjudge::judge::{JudgeRequest, JudgeResponse, ReasonCode, Status};

const EXAMPLES: &str = "schemas/examples";

#[derive(Clone, Copy, Debug, PartialEq)]
enum Doc {
    Definition,
    Policy,
    Request,
    Response,
}

impl Doc {
    fn schema(self) -> &'static str {
        match self {
            Doc::Definition => "definition-v1",
            Doc::Policy => "policy-v1",
            Doc::Request => "judge-request-v1",
            Doc::Response => "judge-response-v1",
        }
    }

    fn rust_accepts(self, value: &Value) -> bool {
        match self {
            Doc::Definition => DecisionDefinition::from_value(value.clone()).is_ok(),
            Doc::Policy => GatePolicy::from_value(value.clone()).is_ok(),
            Doc::Request => JudgeRequest::parse(&value.to_string()).is_ok(),
            Doc::Response => JudgeResponse::from_json(&value.to_string()).is_ok(),
        }
    }
}

fn load(path: &str) -> Value {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn validator(doc: Doc) -> jsonschema::Validator {
    let definition = load("schemas/definition-v1.schema.json");
    let registry = jsonschema::Registry::new()
        .add("urn:snapjudge:schema:definition-v1", definition)
        .unwrap()
        .prepare()
        .unwrap();
    let schema = load(&format!("schemas/{}.schema.json", doc.schema()));
    jsonschema::options()
        .with_registry(&registry)
        .build(&schema)
        .unwrap()
}

fn errors(validator: &jsonschema::Validator, value: &Value) -> Vec<String> {
    validator
        .iter_errors(value)
        .map(|e| format!("{e} at {}", e.instance_path()))
        .collect()
}

fn text(name: &str) -> String {
    fs::read_to_string(format!("{EXAMPLES}/{name}.json")).unwrap()
}

fn example(name: &str) -> Value {
    serde_json::from_str(&text(name)).unwrap()
}

fn pretty<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string_pretty(value).unwrap() + "\n"
}

const EXAMPLE_FILES: &[(Doc, &str)] = &[
    (Doc::Definition, "definition-v1.task-route"),
    (Doc::Definition, "definition-v1.ticket-triage"),
    (Doc::Policy, "policy-v1.task-route"),
    (Doc::Policy, "policy-v1.ticket-triage"),
    (Doc::Request, "judge-request-v1.ref"),
    (Doc::Request, "judge-request-v1.inline"),
    (Doc::Response, "judge-response-v1.accepted"),
    (Doc::Response, "judge-response-v1.deferred"),
    (Doc::Response, "judge-response-v1.error"),
];

// ---- Schemas, examples and round trips ----

#[test]
fn examples_match_their_schema_and_rust() {
    for (doc, name) in EXAMPLE_FILES {
        let value = example(name);
        assert_eq!(
            errors(&validator(*doc), &value),
            Vec::<String>::new(),
            "{name}"
        );
        assert!(doc.rust_accepts(&value), "{name}");
    }
}

#[test]
fn examples_round_trip_byte_identically() {
    for (doc, name) in EXAMPLE_FILES {
        let original = text(name);
        let again = match doc {
            Doc::Definition => pretty(&DecisionDefinition::from_json(&original).unwrap()),
            Doc::Policy => pretty(&GatePolicy::from_json(&original).unwrap()),
            Doc::Request => pretty(&JudgeRequest::parse(&original).unwrap()),
            Doc::Response => pretty(&JudgeResponse::from_json(&original).unwrap()),
        };
        assert_eq!(again, original, "{name}");
    }
}

#[test]
fn examples_cover_every_status_and_answer_shape() {
    let mut statuses = Vec::new();
    let mut shapes = Vec::new();
    for name in ["accepted", "deferred", "error"] {
        let response = example(&format!("judge-response-v1.{name}"));
        statuses.push(response["status"].as_str().unwrap().to_string());
        for answer in response["answers"].as_object().unwrap().values() {
            shapes.push(answer["type"].as_str().unwrap().to_string());
        }
    }
    assert_eq!(statuses, ["accepted", "deferred", "error"]);
    for shape in ["choice", "noul", "multilabel", "score"] {
        assert!(shapes.iter().any(|s| s == shape), "{shape}");
    }
    let definition = example("definition-v1.ticket-triage");
    let outputs: Vec<&str> = definition["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o["shape"].as_str().unwrap())
        .collect();
    assert_eq!(outputs, ["choice", "noul", "multilabel", "score"]);
    // A non-required output that fails its gate keeps its value and does not defer.
    let accepted = example("judge-response-v1.accepted");
    assert_eq!(accepted["answers"]["sentiment"]["passed"], false);
    assert_eq!(
        accepted["answers"]["sentiment"]["reasons"],
        json!(["low_confidence"])
    );
    // A confident Noul "no" passes.
    assert_eq!(accepted["answers"]["urgent"]["value"], false);
    assert_eq!(accepted["answers"]["urgent"]["passed"], true);
    assert!(accepted["answers"]["urgent"].get("confidence").is_none());
}

#[test]
fn examples_are_bound_to_each_other() {
    let route = DecisionDefinition::from_json(&text("definition-v1.task-route")).unwrap();
    let triage = DecisionDefinition::from_json(&text("definition-v1.ticket-triage")).unwrap();
    let route_policy = GatePolicy::from_json(&text("policy-v1.task-route")).unwrap();
    let triage_policy = GatePolicy::from_json(&text("policy-v1.ticket-triage")).unwrap();
    assert_eq!(route_policy.definition_revision, route.revision());
    assert_eq!(triage_policy.definition_revision, triage.revision());
    route_policy.validate_against(&route).unwrap();
    triage_policy.validate_against(&triage).unwrap();
    let by_ref = JudgeRequest::parse(&text("judge-request-v1.ref")).unwrap();
    let reference = by_ref.definition_ref.unwrap();
    assert_eq!(reference.definition_revision, route.revision());
    route.input_schema.validate_state(&by_ref.state).unwrap();
    let inline = JudgeRequest::parse(&text("judge-request-v1.inline")).unwrap();
    assert_eq!(inline.definition.as_ref().unwrap(), &triage);
    triage.input_schema.validate_state(&inline.state).unwrap();
}

/// Reason codes: the schema's two enums are exactly the Rust enum, split by status as in
/// frozen decision 1.
#[test]
fn reason_codes_and_statuses_match_the_schema() {
    let schema = load("schemas/judge-response-v1.schema.json");
    let codes = |def: &str| -> Vec<String> {
        schema["$defs"][def]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect()
    };
    let deferred = codes("deferredReason");
    let error = codes("errorReason");
    assert_eq!(deferred.len() + error.len(), ReasonCode::ALL.len());
    for code in ReasonCode::ALL {
        let name = serde_json::to_value(code)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        let expected = match code.status() {
            Status::Deferred => &deferred,
            Status::Error => &error,
            Status::Accepted => panic!("no reason accepts"),
        };
        assert!(expected.contains(&name), "{name}");
    }
    assert_eq!(Status::Accepted.exit_code(), 0);
    assert_eq!(Status::Deferred.exit_code(), 0);
    assert_eq!(Status::Error.exit_code(), 1);
}

// ---- Schema/Rust parity (both directions) ----

type Change = Box<dyn Fn(&mut Value)>;

fn set(pointer: &'static str, value: Value) -> Change {
    Box::new(move |v: &mut Value| *v.pointer_mut(pointer).unwrap() = value.clone())
}

fn remove(pointer: &'static str, key: &'static str) -> Change {
    Box::new(move |v: &mut Value| {
        v.pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove(key);
    })
}

fn insert(pointer: &'static str, key: &'static str, value: Value) -> Change {
    Box::new(move |v: &mut Value| {
        v.pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert(key.into(), value.clone());
    })
}

fn many_options(n: usize) -> Value {
    Value::Object(
        (0..n)
            .map(|i| (format!("o{i}"), Value::String("x".into())))
            .collect(),
    )
}

/// Cases the JSON Schema expresses: both must reject.
fn schema_cases() -> Vec<(Doc, &'static str, &'static str, Change)> {
    use Doc::*;
    let def = "definition-v1.ticket-triage";
    let pol = "policy-v1.ticket-triage";
    let exp = "policy-v1.task-route";
    let req = "judge-request-v1.ref";
    let inl = "judge-request-v1.inline";
    let acc = "judge-response-v1.accepted";
    let dfr = "judge-response-v1.deferred";
    let err = "judge-response-v1.error";
    vec![
        (
            Definition,
            def,
            "unknown field",
            insert("", "extra", json!(1)),
        ),
        (
            Definition,
            def,
            "missing questions",
            remove("", "questions"),
        ),
        (
            Definition,
            def,
            "major 2",
            set("/schema_version", json!("2.0")),
        ),
        (
            Definition,
            def,
            "version 1.01",
            set("/schema_version", json!("1.01")),
        ),
        (Definition, def, "uppercase id", set("/id", json!("Ticket"))),
        (Definition, def, "path id", set("/id", json!("../x"))),
        (
            Definition,
            def,
            "site id without namespace",
            set("/site_id", json!("ticket")),
        ),
        (
            Definition,
            def,
            "site id with space",
            set("/site_id", json!("runtime:a b")),
        ),
        (
            Definition,
            def,
            "site id too long",
            set("/site_id", json!(format!("runtime:{}", "a".repeat(250)))),
        ),
        (
            Definition,
            def,
            "short revision",
            set("/definition_revision", json!("abc")),
        ),
        (
            Definition,
            def,
            "null revision",
            set("/definition_revision", Value::Null),
        ),
        (
            Definition,
            def,
            "unknown input kind",
            set("/input_schema/fields/0/kind", json!("date")),
        ),
        (
            Definition,
            def,
            "input field extra key",
            insert("/input_schema/fields/0", "format", json!("x")),
        ),
        (
            Definition,
            def,
            "input field bad name",
            set("/input_schema/fields/0/name", json!("1subject")),
        ),
        (
            Definition,
            def,
            "no questions",
            set("/questions", json!({})),
        ),
        (
            Definition,
            def,
            "dotted question key",
            insert(
                "/questions",
                "a.b",
                json!({"type": "noul", "instructions": "x"}),
            ),
        ),
        (
            Definition,
            def,
            "choice without options",
            set("/questions/team/criteria", json!({})),
        ),
        (
            Definition,
            def,
            "choice with 256 options",
            set("/questions/team/criteria", many_options(256)),
        ),
        (
            Definition,
            def,
            "empty option key",
            set("/questions/team/criteria", json!({"": "x"})),
        ),
        (
            Definition,
            def,
            "numeric option description",
            set("/questions/team/criteria/sales", json!(1)),
        ),
        (
            Definition,
            def,
            "score with one level",
            set("/questions/sentiment/criteria", json!(["only"])),
        ),
        (
            Definition,
            def,
            "score with 11 levels",
            set(
                "/questions/sentiment/criteria",
                json!(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]),
            ),
        ),
        (
            Definition,
            def,
            "empty instructions",
            set("/questions/urgent/instructions", json!("")),
        ),
        (
            Definition,
            def,
            "long instructions",
            set("/questions/urgent/instructions", json!("x".repeat(16385))),
        ),
        (
            Definition,
            def,
            "object instructions",
            set("/questions/urgent/instructions", json!({"text": "x"})),
        ),
        (
            Definition,
            def,
            "noul criteria without false",
            remove("/questions/urgent/criteria", "false"),
        ),
        (
            Definition,
            def,
            "unknown question type",
            set("/questions/urgent/type", json!("rank")),
        ),
        (
            Definition,
            def,
            "unknown output shape",
            set("/outputs/0/shape", json!("ranking")),
        ),
        (
            Definition,
            def,
            "output without required",
            remove("/outputs/0", "required"),
        ),
        (
            Definition,
            def,
            "null nullable_option",
            set("/outputs/0/nullable_option", Value::Null),
        ),
        (
            Definition,
            def,
            "cutoff above 1",
            set("/outputs/1/cutoff", json!(1.5)),
        ),
        (
            Definition,
            def,
            "multilabel without labels",
            set("/outputs/2/labels", json!([])),
        ),
        (
            Definition,
            def,
            "label extra key",
            insert("/outputs/2/labels/0", "threshold", json!(0.5)),
        ),
        (
            Definition,
            def,
            "unbounded level value",
            set("/outputs/3/level_values/0", json!(1e16)),
        ),
        (Definition, def, "no outputs", set("/outputs", json!([]))),
        (
            Policy,
            pol,
            "latest alias",
            set("/model", json!("jev-latest")),
        ),
        (
            Policy,
            pol,
            "preview alias",
            set("/model", json!("jev-preview")),
        ),
        (
            Policy,
            pol,
            "model with space",
            set("/model", json!("jev 1")),
        ),
        (Policy, pol, "empty model", set("/model", json!(""))),
        (
            Policy,
            pol,
            "unknown evidence",
            set("/evidence", json!("guess")),
        ),
        (
            Policy,
            pol,
            "measured without metrics",
            remove("", "metrics"),
        ),
        (
            Policy,
            pol,
            "measured without evidence revision",
            remove("", "evidence_revision"),
        ),
        (Policy, pol, "measured without target", remove("", "target")),
        (
            Policy,
            pol,
            "measured without min_accepted",
            remove("", "min_accepted"),
        ),
        (
            Policy,
            pol,
            "metrics without wilson_lower",
            remove("/metrics", "wilson_lower"),
        ),
        (
            Policy,
            pol,
            "negative accepted",
            set("/metrics/accepted", json!(-1)),
        ),
        (
            Policy,
            pol,
            "wilson_lower above 1",
            set("/metrics/wilson_lower", json!(1.2)),
        ),
        (
            Policy,
            pol,
            "threshold above 1",
            set("/thresholds/team", json!(1.2)),
        ),
        (
            Policy,
            pol,
            "three-part threshold key",
            insert("/thresholds", "tags.bug.x", json!(0.5)),
        ),
        (
            Policy,
            pol,
            "unknown field",
            insert("", "budget_usd", json!(1)),
        ),
        (
            Policy,
            pol,
            "min_accepted above 2^53 - 1",
            set("/min_accepted", json!(9007199254740992u64)),
        ),
        (
            Policy,
            pol,
            "bad policy revision",
            set("/policy_revision", json!("x")),
        ),
        (
            Policy,
            pol,
            "bad definition id",
            set("/definition_id", json!("Ticket")),
        ),
        (
            Policy,
            pol,
            "long dataset",
            set("/dataset", json!("d".repeat(1025))),
        ),
        (
            Policy,
            pol,
            "fractional min_accepted",
            set("/min_accepted", json!(200.5)),
        ),
        (
            Policy,
            pol,
            "fractional metrics.n",
            set("/metrics/n", json!(500.5)),
        ),
        (
            Policy,
            pol,
            "metrics.accepted above 2^53 - 1",
            set("/metrics/accepted", json!(1e16)),
        ),
        (
            Policy,
            exp,
            "null target",
            insert("", "target", Value::Null),
        ),
        (
            Request,
            req,
            "protocol 2",
            set("/protocol_version", json!(2)),
        ),
        (
            Request,
            req,
            "protocol as string",
            set("/protocol_version", json!("1")),
        ),
        (
            Request,
            req,
            "both definition fields",
            insert("", "definition", example(def)),
        ),
        (
            Request,
            req,
            "neither definition field",
            remove("", "definition_ref"),
        ),
        (Request, req, "array state", set("/state", json!(["x"]))),
        (Request, req, "missing state", remove("", "state")),
        (
            Request,
            req,
            "empty request id",
            set("/request_id", json!("")),
        ),
        (
            Request,
            req,
            "long request id",
            set("/request_id", json!("r".repeat(129))),
        ),
        (
            Request,
            req,
            "site id without namespace",
            set("/site_id", json!("task-route")),
        ),
        (
            Request,
            req,
            "unknown field",
            insert("", "threshold", json!(0.1)),
        ),
        (
            Request,
            req,
            "short ref revision",
            set("/definition_ref/definition_revision", json!("abc")),
        ),
        (
            Request,
            req,
            "bad policy id",
            set("/policy_id", json!("Policy")),
        ),
        (Request, req, "zero timeout", set("/timeout_ms", json!(0))),
        (
            Request,
            req,
            "fractional timeout",
            set("/timeout_ms", json!(1500.5)),
        ),
        (
            Request,
            req,
            "negative float timeout",
            set("/timeout_ms", json!(-1.0)),
        ),
        (
            Request,
            req,
            "fractional protocol",
            set("/protocol_version", json!(1.5)),
        ),
        (
            Request,
            req,
            "null policy id",
            set("/policy_id", Value::Null),
        ),
        (
            Request,
            inl,
            "inline unknown input kind",
            set("/definition/input_schema/fields/0/kind", json!("date")),
        ),
        (
            Request,
            inl,
            "inline alias field",
            insert("/definition", "model", json!("jev-latest")),
        ),
        (
            Response,
            acc,
            "accepted with fallback",
            set("/fallback_recommended", json!(true)),
        ),
        (
            Response,
            acc,
            "accepted with reasons",
            set("/gate/reasons", json!(["low_confidence"])),
        ),
        (
            Response,
            acc,
            "accepted with error",
            set("/error", example(err)["error"].clone()),
        ),
        (
            Response,
            acc,
            "accepted without provider",
            set("/provider", Value::Null),
        ),
        (
            Response,
            acc,
            "accepted without policy",
            set("/gate/policy_id", Value::Null),
        ),
        (
            Response,
            acc,
            "accepted with null request id",
            set("/request_id", Value::Null),
        ),
        (
            Response,
            acc,
            "accepted with failed gate",
            set("/gate/passed", json!(false)),
        ),
        (
            Response,
            acc,
            "noul with confidence",
            insert("/answers/urgent", "confidence", json!(0.9)),
        ),
        (
            Response,
            acc,
            "probability above 1",
            set("/answers/team/probabilities/billing", json!(1.5)),
        ),
        (
            Response,
            acc,
            "score level 10",
            set("/answers/sentiment/level", json!(10)),
        ),
        (
            Response,
            acc,
            "score probability key",
            insert("/answers/sentiment/probabilities", "10", json!(0.0)),
        ),
        (
            Response,
            acc,
            "fractional score level",
            set("/answers/sentiment/level", json!(1.5)),
        ),
        (
            Response,
            acc,
            "negative float duration",
            set("/metrics/duration_ms", json!(-1.0)),
        ),
        (
            Response,
            acc,
            "protocol 2.0",
            set("/protocol_version", json!(2.0)),
        ),
        (
            Response,
            acc,
            "multilabel bad label",
            set("/answers/tags/value", json!(["a b"])),
        ),
        (
            Response,
            acc,
            "unknown answer type",
            set("/answers/team/type", json!("rank")),
        ),
        (Response, acc, "missing error field", remove("", "error")),
        (
            Response,
            acc,
            "other fallback owner",
            set("/fallback_owner", json!("agent")),
        ),
        (
            Response,
            acc,
            "other provider",
            set("/provider/name", json!("openai")),
        ),
        (
            Response,
            acc,
            "protocol 2",
            set("/protocol_version", json!(2)),
        ),
        (
            Response,
            dfr,
            "deferred without reasons",
            set("/gate/reasons", json!([])),
        ),
        (
            Response,
            dfr,
            "deferred with error reason",
            set("/gate/reasons", json!(["invalid_input"])),
        ),
        (
            Response,
            dfr,
            "deferred without fallback",
            set("/fallback_recommended", json!(false)),
        ),
        (
            Response,
            dfr,
            "unknown reason",
            set("/gate/reasons", json!(["maybe"])),
        ),
        (
            Response,
            err,
            "error without error object",
            set("/error", Value::Null),
        ),
        (
            Response,
            err,
            "error with deferred reason",
            set("/gate/reasons", json!(["timeout"])),
        ),
        (
            Response,
            err,
            "error code of deferral",
            set("/error/code", json!("timeout")),
        ),
        (
            Response,
            err,
            "long error message",
            set("/error/message", json!("m".repeat(513))),
        ),
        (
            Response,
            err,
            "error passed gate",
            set("/gate/passed", json!(true)),
        ),
    ]
}

#[test]
fn schema_and_rust_reject_the_same_documents() {
    for (doc, base, name, change) in schema_cases() {
        let mut value = example(base);
        change(&mut value);
        assert!(!validator(doc).is_valid(&value), "schema accepts: {name}");
        assert!(!doc.rust_accepts(&value), "Rust accepts: {name}");
    }
}

/// Mutations both must still accept (boundaries, omittable fields, nulls where allowed).
#[test]
fn schema_and_rust_accept_the_same_boundaries() {
    use Doc::*;
    let def = "definition-v1.ticket-triage";
    let cases: Vec<(Doc, &str, &str, Change)> = vec![
        (
            Definition,
            def,
            "no revision",
            remove("", "definition_revision"),
        ),
        (
            Definition,
            def,
            "minor version",
            set("/schema_version", json!("1.7")),
        ),
        (
            Definition,
            def,
            "255 options",
            Box::new(|v: &mut Value| {
                v["questions"]["team"]["criteria"] = many_options(255);
                v["outputs"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("nullable_option");
                v.as_object_mut().unwrap().remove("definition_revision");
            }),
        ),
        (
            Definition,
            def,
            "cutoff omitted",
            Box::new(|v: &mut Value| {
                v["outputs"][1].as_object_mut().unwrap().remove("cutoff");
                v.as_object_mut().unwrap().remove("definition_revision");
            }),
        ),
        (
            Definition,
            def,
            "noul without criteria",
            Box::new(|v: &mut Value| {
                v["questions"]["urgent"]
                    .as_object_mut()
                    .unwrap()
                    .remove("criteria");
                v.as_object_mut().unwrap().remove("definition_revision");
            }),
        ),
        (
            Definition,
            def,
            "decreasing level values",
            Box::new(|v: &mut Value| {
                v["outputs"][3]["level_values"] = json!([2, 1, 0, -1, -2]);
                v.as_object_mut().unwrap().remove("definition_revision");
            }),
        ),
        (
            Policy,
            "policy-v1.task-route",
            "no revision",
            remove("", "policy_revision"),
        ),
        (
            Policy,
            "policy-v1.task-route",
            "fixture evidence",
            Box::new(|v: &mut Value| {
                v["evidence"] = json!("fixture");
                v.as_object_mut().unwrap().remove("policy_revision");
            }),
        ),
        (
            Policy,
            "policy-v1.ticket-triage",
            "extensible metrics",
            Box::new(|v: &mut Value| {
                v["metrics"]["note"] = Value::Null;
                v.as_object_mut().unwrap().remove("policy_revision");
            }),
        ),
        (
            Request,
            "judge-request-v1.ref",
            "no policy and timeout",
            Box::new(|v: &mut Value| {
                v.as_object_mut().unwrap().remove("policy_id");
                v.as_object_mut().unwrap().remove("timeout_ms");
            }),
        ),
        (
            Request,
            "judge-request-v1.ref",
            "timeout above the cap",
            set("/timeout_ms", json!(60_000)),
        ),
        (
            Request,
            "judge-request-v1.ref",
            "integral float protocol and timeout",
            Box::new(|v: &mut Value| {
                v["protocol_version"] = json!(1.0);
                v["timeout_ms"] = json!(1500.0);
            }),
        ),
        (
            Policy,
            "policy-v1.ticket-triage",
            "integral float counts keep the revision",
            Box::new(|v: &mut Value| {
                v["min_accepted"] = json!(200.0);
                v["metrics"]["accepted"] = json!(412.0);
                v["metrics"]["n"] = json!(500.0);
            }),
        ),
        (
            Policy,
            "policy-v1.ticket-triage",
            "accepted equal to n",
            Box::new(|v: &mut Value| {
                v["metrics"]["accepted"] = json!(500);
                v.as_object_mut().unwrap().remove("policy_revision");
            }),
        ),
        (
            Policy,
            "policy-v1.ticket-triage",
            "extra integers at 2^53 - 1",
            Box::new(|v: &mut Value| {
                v["metrics"]["seed"] =
                    json!([9_007_199_254_740_991u64, {"low": -9_007_199_254_740_991i64}]);
                v.as_object_mut().unwrap().remove("policy_revision");
            }),
        ),
        (
            Response,
            "judge-response-v1.accepted",
            "integral float protocol, duration and level",
            Box::new(|v: &mut Value| {
                v["protocol_version"] = json!(1.0);
                v["metrics"]["duration_ms"] = json!(412.0);
                v["answers"]["sentiment"]["level"] = json!(1.0);
            }),
        ),
        (
            Request,
            "judge-request-v1.inline",
            "inline without revision",
            remove("/definition", "definition_revision"),
        ),
        (
            Response,
            "judge-response-v1.error",
            "unparseable request",
            Box::new(|v: &mut Value| {
                v["request_id"] = Value::Null;
                v["site_id"] = Value::Null;
                v["gate"]["reasons"] = json!(["invalid_request"]);
                v["error"]["code"] = json!("invalid_request");
            }),
        ),
        (
            Response,
            "judge-response-v1.accepted",
            "nullable choice value",
            set("/answers/team/value", Value::Null),
        ),
    ];
    for (doc, base, name, change) in cases {
        let mut value = example(base);
        change(&mut value);
        assert_eq!(
            errors(&validator(doc), &value),
            Vec::<String>::new(),
            "{name}"
        );
        assert!(doc.rust_accepts(&value), "Rust rejects: {name}");
    }
}

/// Rules JSON Schema cannot express (cross-references, uniqueness, revision recomputation,
/// cross-field comparisons, state size): the schema accepts, Rust rejects. Each is stated in
/// the schema's description.
#[test]
fn rust_only_rules_are_the_documented_ones() {
    use Doc::*;
    let def = "definition-v1.ticket-triage";
    let pol = "policy-v1.ticket-triage";
    let cases: Vec<(Doc, &str, &str, Change)> = vec![
        (
            Definition,
            def,
            "unknown question",
            set("/outputs/0/question", json!("missing")),
        ),
        (
            Definition,
            def,
            "question of another type",
            set("/outputs/1/question", json!("team")),
        ),
        (
            Definition,
            def,
            "label on a choice question",
            set("/outputs/2/labels/0/question", json!("team")),
        ),
        (
            Definition,
            def,
            "duplicate output names",
            set("/outputs/1/name", json!("team")),
        ),
        (
            Definition,
            def,
            "duplicate label names",
            set("/outputs/2/labels/1/name", json!("billing")),
        ),
        (
            Definition,
            def,
            "duplicate input names",
            set("/input_schema/fields/1/name", json!("subject")),
        ),
        (
            Definition,
            def,
            "nullable option not a key",
            set("/outputs/0/nullable_option", json!("nobody")),
        ),
        (
            Definition,
            def,
            "level values count",
            set("/outputs/3/level_values", json!([1, 2])),
        ),
        (
            Definition,
            def,
            "non-monotonic level values",
            set("/outputs/3/level_values", json!([0, 2, 1, 3, 4])),
        ),
        (
            Definition,
            def,
            "stated revision mismatch",
            set("/definition_revision", json!("0".repeat(64))),
        ),
        (
            Policy,
            pol,
            "wilson_lower below target",
            set("/metrics/wilson_lower", json!(0.9)),
        ),
        (
            Policy,
            pol,
            "accepted below min_accepted",
            set("/metrics/accepted", json!(10)),
        ),
        (
            Policy,
            pol,
            "stated revision mismatch",
            set("/policy_revision", json!("0".repeat(64))),
        ),
        (
            Policy,
            pol,
            "metrics.accepted above metrics.n",
            Box::new(|v: &mut Value| {
                v["metrics"]["accepted"] = json!(501);
                v.as_object_mut().unwrap().remove("policy_revision");
            }),
        ),
        (
            Policy,
            pol,
            "metrics.target differs from target",
            Box::new(|v: &mut Value| {
                v["metrics"]["target"] = json!(0.9);
                v.as_object_mut().unwrap().remove("policy_revision");
            }),
        ),
        (
            Policy,
            pol,
            "metrics.target without a policy target",
            Box::new(|v: &mut Value| {
                v["evidence"] = json!("experimental");
                v.as_object_mut().unwrap().remove("target");
                v.as_object_mut().unwrap().remove("policy_revision");
            }),
        ),
        (
            Policy,
            pol,
            "unsafe integer in metrics extras",
            Box::new(|v: &mut Value| {
                v["metrics"]["seed"] = json!({"nested": [9_007_199_254_740_992u64]});
                v.as_object_mut().unwrap().remove("policy_revision");
            }),
        ),
        (
            Policy,
            pol,
            "unsafe negative integer in metrics extras",
            Box::new(|v: &mut Value| {
                v["metrics"]["seed"] = json!(-9_007_199_254_740_992i64);
                v.as_object_mut().unwrap().remove("policy_revision");
            }),
        ),
        (
            Request,
            "judge-request-v1.inline",
            "site binding mismatch",
            set("/site_id", json!("runtime:other")),
        ),
        (
            Request,
            "judge-request-v1.ref",
            "state over 96 KiB",
            set("/state/task", json!("x".repeat(96 * 1024))),
        ),
    ];
    for (doc, base, name, change) in cases {
        let mut value = example(base);
        change(&mut value);
        assert_eq!(
            errors(&validator(doc), &value),
            Vec::<String>::new(),
            "{name}"
        );
        assert!(!doc.rust_accepts(&value), "Rust accepts: {name}");
    }
}

// ---- Request reason codes (frozen decision 1) ----

fn reason(text: &str) -> (ReasonCode, Option<String>) {
    let error = JudgeRequest::parse(text).unwrap_err();
    assert!(error.message.chars().count() <= 512);
    (error.reason, error.request_id)
}

#[test]
fn request_rejections_carry_the_frozen_reason_codes() {
    let base = example("judge-request-v1.ref");
    let with = |f: &dyn Fn(&mut Value)| {
        let mut v = base.clone();
        f(&mut v);
        v.to_string()
    };
    let id = Some("demo-route-001".to_string());
    assert_eq!(reason("{not json"), (ReasonCode::InvalidRequest, None));
    assert_eq!(reason("[]"), (ReasonCode::InvalidRequest, None));
    let oversized = format!("{{\"pad\":\"{}\"}}", "x".repeat(1 << 20));
    assert_eq!(reason(&oversized), (ReasonCode::InvalidRequest, None));
    assert_eq!(
        reason(&with(&|v| v["protocol_version"] = json!(2))),
        (ReasonCode::ProtocolMismatch, id.clone())
    );
    // Another major is reported as such even when the rest has another shape.
    assert_eq!(
        reason(&json!({"protocol_version": 3, "request_id": "r", "anything": true}).to_string()),
        (ReasonCode::ProtocolMismatch, Some("r".into()))
    );
    assert_eq!(
        reason(&with(
            &|v| v["definition"] = example("definition-v1.task-route")
        )),
        (ReasonCode::InvalidRequest, id.clone())
    );
    assert_eq!(
        reason(&with(&|v| {
            v.as_object_mut().unwrap().remove("definition_ref");
        })),
        (ReasonCode::InvalidRequest, id.clone())
    );
    assert_eq!(
        reason(&with(&|v| v["state"] = json!("text"))),
        (ReasonCode::InvalidInput, id.clone())
    );
    assert_eq!(
        reason(&with(&|v| v["protocol_version"] = json!(2.0))),
        (ReasonCode::ProtocolMismatch, id.clone())
    );
    // A non-object `definition` is an envelope error, with or without a reference.
    assert_eq!(
        reason(&with(&|v| v["definition"] = json!("x"))),
        (ReasonCode::InvalidRequest, id.clone())
    );
    assert_eq!(
        reason(&with(&|v| {
            v["definition"] = json!("x");
            v.as_object_mut().unwrap().remove("definition_ref");
        })),
        (ReasonCode::InvalidRequest, id.clone())
    );
    // An invalid definition next to a reference is an envelope error, not a definition one.
    assert_eq!(
        reason(&with(&|v| {
            let mut definition = example("definition-v1.task-route");
            definition["input_schema"]["fields"][0]["kind"] = json!("date");
            v["definition"] = definition;
        })),
        (ReasonCode::InvalidRequest, id.clone())
    );
    let inline = example("judge-request-v1.inline");
    let inline_with = |f: &dyn Fn(&mut Value)| {
        let mut v = inline.clone();
        f(&mut v);
        v.to_string()
    };
    let ticket = Some("ticket-4812".to_string());
    assert_eq!(
        reason(&inline_with(&|v| v["site_id"] = json!("runtime:other"))),
        (ReasonCode::InvalidDefinition, ticket.clone())
    );
    assert_eq!(
        reason(&inline_with(
            &|v| v["definition"]["definition_revision"] = json!("f".repeat(64))
        )),
        (ReasonCode::InvalidDefinition, ticket.clone())
    );
    assert_eq!(
        reason(&inline_with(
            &|v| v["definition"]["input_schema"]["fields"][0]["kind"] = json!("date")
        )),
        (ReasonCode::InvalidDefinition, ticket.clone())
    );
    assert_eq!(
        reason(&inline_with(&|v| v["request_id"] = json!(""))),
        (ReasonCode::InvalidRequest, None)
    );
    // Envelope errors come before inline-definition errors.
    let broken = |f: &dyn Fn(&mut Value)| {
        inline_with(&|v| {
            v["definition"]["input_schema"]["fields"][0]["kind"] = json!("date");
            f(v);
        })
    };
    assert_eq!(
        reason(&broken(&|_| {})),
        (ReasonCode::InvalidDefinition, ticket.clone())
    );
    assert_eq!(
        reason(&broken(&|v| {
            v.as_object_mut().unwrap().remove("request_id");
        })),
        (ReasonCode::InvalidRequest, None)
    );
    assert_eq!(
        reason(&broken(&|v| {
            v.as_object_mut().unwrap().remove("state");
        })),
        (ReasonCode::InvalidRequest, ticket.clone())
    );
    assert_eq!(
        reason(&broken(&|v| v["policy_id"] = json!("Bad"))),
        (ReasonCode::InvalidRequest, ticket.clone())
    );
}

#[test]
fn inline_definition_revision_is_computed_not_trusted() {
    let mut value = example("judge-request-v1.inline");
    let revision = value["definition"]["definition_revision"].clone();
    value["definition"]
        .as_object_mut()
        .unwrap()
        .remove("definition_revision");
    let request = JudgeRequest::parse(&value.to_string()).unwrap();
    assert_eq!(
        request.definition.unwrap().definition_revision.as_deref(),
        revision.as_str()
    );
}

// ---- Revisions: RFC 8785 JCS + SHA-256 (frozen decision 5) ----

#[test]
fn jcs_matches_the_rfc_8785_published_examples() {
    // RFC 8785 section 3.2.2.
    let input = r#"{
  "numbers": [333333333.33333329, 1E30, 4.50,
              2e-3, 0.000000000000000000000000001],
  "string": "~u20ac$~u000F~u000aA'~u0042~u0022~u005c~~~"~/",
  "literals": [null, true, false]
}"#
    .replace('~', "\\");
    assert_eq!(
        jcs::canonicalize_text(&input).unwrap(),
        r#"{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],"string":"€$\u000f\nA'B\"\\\\\"/"}"#
    );
    // RFC 8785 section 3.2.3: members sorted by UTF-16 code units.
    let sorting = r#"{"~u20ac":"Euro Sign","~r":"Carriage Return","~ufb33":"Hebrew Letter Dalet With Dagesh","1":"One","~ud83d~ude00":"Emoji: Grinning Face","~u0080":"Control","~u00f6":"Latin Small Letter O With Diaeresis"}"#
        .replace('~', "\\");
    let keys: Vec<String> = {
        let canonical = jcs::canonicalize_text(&sorting).unwrap();
        let value: serde_json::Map<String, Value> = serde_json::from_str(&canonical).unwrap();
        value.keys().cloned().collect()
    };
    assert_eq!(
        keys,
        [
            "\r",
            "1",
            "\u{80}",
            "\u{f6}",
            "\u{20ac}",
            "\u{1f600}",
            "\u{fb33}"
        ]
    );
    // RFC 8785 Appendix B number samples (IEEE-754 bits → canonical text).
    for (bits, expected) in [
        (0x0000000000000000u64, "0"),
        (0x8000000000000000, "0"),
        (0x0000000000000001, "5e-324"),
        (0x8000000000000001, "-5e-324"),
        (0x7fefffffffffffff, "1.7976931348623157e+308"),
        (0xffefffffffffffff, "-1.7976931348623157e+308"),
        (0x4340000000000000, "9007199254740992"),
        (0xc340000000000000, "-9007199254740992"),
        (0x4430000000000000, "295147905179352830000"),
        (0x44b52d02c7e14af5, "9.999999999999997e+22"),
        (0x44b52d02c7e14af6, "1e+23"),
        (0x44b52d02c7e14af7, "1.0000000000000001e+23"),
        (0x444b1ae4d6e2ef4e, "999999999999999700000"),
        (0x444b1ae4d6e2ef4f, "999999999999999900000"),
        (0x444b1ae4d6e2ef50, "1e+21"),
        (0x3eb0c6f7a0b5ed8c, "9.999999999999997e-7"),
        (0x3eb0c6f7a0b5ed8d, "0.000001"),
        (0x41b3de4355555553, "333333333.3333332"),
        (0x41b3de4355555554, "333333333.33333325"),
        (0x41b3de4355555555, "333333333.3333333"),
        (0x41b3de4355555556, "333333333.3333334"),
        (0x41b3de4355555557, "333333333.33333343"),
        (0xbecbf647612f3696, "-0.0000033333333333333333"),
        (0x43143ff3c1cb0959, "1424953923781206.2"),
    ] {
        assert_eq!(
            jcs::canonical_json(&f64::from_bits(bits)).unwrap(),
            expected,
            "{bits:#018x}"
        );
    }
    assert!(jcs::canonical_json(&f64::NAN).is_err());
}

#[test]
fn revision_vectors_file_recomputes() {
    let file = example("revision-vectors");
    let vectors = file["vectors"].as_array().unwrap();
    assert!(vectors.len() >= 8);
    for vector in vectors {
        let name = vector["name"].as_str().unwrap();
        let input = vector["input_text"].as_str().unwrap();
        let canonical = vector["canonical"].as_str().unwrap();
        let sha = vector["sha256"].as_str().unwrap();
        assert_eq!(jcs::sha256_hex(canonical.as_bytes()), sha, "{name}");
        match vector["kind"].as_str().unwrap() {
            "jcs" => assert_eq!(jcs::canonicalize_text(input).unwrap(), canonical, "{name}"),
            "definition" => {
                let definition = DecisionDefinition::from_json(input).unwrap();
                assert_eq!(definition.revision(), sha, "{name}");
                let value = serde_json::to_value(&definition).unwrap();
                let content = json!({
                    "site_id": value["site_id"],
                    "input_schema": value["input_schema"],
                    "questions": value["questions"],
                    "outputs": value["outputs"],
                });
                assert_eq!(jcs::canonical_json(&content).unwrap(), canonical, "{name}");
            }
            "policy" => {
                let policy = GatePolicy::from_json(input).unwrap();
                assert_eq!(policy.revision(), sha, "{name}");
            }
            other => panic!("unknown kind {other}"),
        }
    }
    // The published RFC sample and the key-order/number-spelling variants are present.
    let shas: Vec<&str> = vectors
        .iter()
        .map(|v| v["sha256"].as_str().unwrap())
        .collect();
    assert_eq!(shas[2], shas[3], "key order changes a definition revision");
    assert_eq!(
        shas[4], shas[5],
        "number spelling or defaults change a revision"
    );
}

#[test]
fn revisions_ignore_key_order_and_number_spelling_but_not_content() {
    let text = fs::read_to_string(format!("{EXAMPLES}/definition-v1.ticket-triage.json")).unwrap();
    let base = DecisionDefinition::from_json(&text).unwrap();
    let respelled = text
        .replace("0.5", "5e-1")
        .replace("-2.0", "-2")
        .replace("2.0\n", "2E0\n");
    assert_ne!(respelled, text);
    let mut value: Value = serde_json::from_str(&respelled).unwrap();
    value.as_object_mut().unwrap().remove("definition_revision");
    // Reorder top-level keys: revision content does not depend on member order.
    let reordered: serde_json::Map<String, Value> = value
        .as_object()
        .unwrap()
        .clone()
        .into_iter()
        .rev()
        .collect();
    let again = DecisionDefinition::from_value(Value::Object(reordered)).unwrap();
    assert_eq!(again.revision(), base.revision());
    // id and schema_version are not definition content; instructions are.
    let mut renamed = serde_json::to_value(&base).unwrap();
    renamed["id"] = json!("ticket-triage-copy");
    renamed["schema_version"] = json!("1.3");
    renamed
        .as_object_mut()
        .unwrap()
        .remove("definition_revision");
    assert_eq!(
        DecisionDefinition::from_value(renamed).unwrap().revision(),
        base.revision()
    );
    let mut edited = serde_json::to_value(&base).unwrap();
    edited["questions"]["urgent"]["instructions"] = json!("Is it urgent?");
    edited
        .as_object_mut()
        .unwrap()
        .remove("definition_revision");
    assert_ne!(
        DecisionDefinition::from_value(edited).unwrap().revision(),
        base.revision()
    );
    // A policy revision covers everything but itself.
    let policy = GatePolicy::from_json(&text_of("policy-v1.task-route")).unwrap();
    let mut changed = serde_json::to_value(&policy).unwrap();
    changed["thresholds"]["route"] = json!(0.9);
    changed.as_object_mut().unwrap().remove("policy_revision");
    assert_ne!(
        GatePolicy::from_value(changed).unwrap().revision(),
        policy.revision()
    );
}

fn text_of(name: &str) -> String {
    text(name)
}

#[test]
fn caller_supplied_revisions_that_differ_are_rejected() {
    let mut definition = example("definition-v1.task-route");
    definition["definition_revision"] = json!("a".repeat(64));
    assert!(matches!(
        DecisionDefinition::from_value(definition),
        Err(ContractError::RevisionMismatch {
            field: "definition_revision",
            ..
        })
    ));
    let mut policy = example("policy-v1.task-route");
    policy["policy_revision"] = json!("a".repeat(64));
    assert!(matches!(
        GatePolicy::from_value(policy),
        Err(ContractError::RevisionMismatch {
            field: "policy_revision",
            ..
        })
    ));
}

// ---- Policies against their definition (frozen decisions 6, 7) ----

#[test]
fn thresholds_must_cover_every_output_and_label() {
    let definition = DecisionDefinition::from_json(&text("definition-v1.ticket-triage")).unwrap();
    let policy = GatePolicy::from_json(&text("policy-v1.ticket-triage")).unwrap();
    let check = |f: &dyn Fn(&mut GatePolicy)| {
        let mut p = policy.clone();
        f(&mut p);
        p.validate_against(&definition)
    };
    assert!(check(&|_| {}).is_ok());
    // Required or not (`sentiment`, `tags.security` are optional), every output and label
    // needs a threshold.
    for missing in [
        "team",
        "urgent",
        "sentiment",
        "tags.billing",
        "tags.bug",
        "tags.security",
    ] {
        assert!(
            check(&|p| {
                p.thresholds.shift_remove(missing);
            })
            .is_err(),
            "{missing}"
        );
    }
    for unknown in ["tags", "team.billing", "nothing", "tags.nothing"] {
        assert!(
            check(&|p| {
                p.thresholds.insert(unknown.into(), 0.5);
            })
            .is_err(),
            "{unknown}"
        );
    }
    assert!(check(&|p| p.definition_id = "task-route".into()).is_err());
}

#[test]
fn experimental_and_fixture_policies_need_no_evidence() {
    let mut value = example("policy-v1.task-route");
    value.as_object_mut().unwrap().remove("policy_revision");
    for evidence in ["experimental", "fixture"] {
        value["evidence"] = json!(evidence);
        assert!(GatePolicy::from_value(value.clone()).is_ok(), "{evidence}");
    }
    value["evidence"] = json!("measured");
    assert!(GatePolicy::from_value(value).is_err());
}

// ---- state validation (frozen decision 10) ----

#[test]
fn state_validator_enforces_the_input_schema() {
    let definition = DecisionDefinition::from_json(&text("definition-v1.ticket-triage")).unwrap();
    let schema = &definition.input_schema;
    let ok = json!({"subject": "s", "body": "b"});
    assert!(schema.validate_state(&ok).is_ok());
    assert!(
        schema
            .validate_state(&json!({"subject": "s", "body": "b", "attachments": 2}))
            .is_ok()
    );
    assert!(
        schema
            .validate_state(&json!({"subject": "s", "body": "b", "attachments": 2.0}))
            .is_ok()
    );
    for (name, state) in [
        ("not an object", json!("s")),
        ("array", json!([])),
        ("missing required", json!({"subject": "s"})),
        (
            "extra key",
            json!({"subject": "s", "body": "b", "secret": "x"}),
        ),
        ("wrong kind", json!({"subject": 1, "body": "b"})),
        (
            "fractional integer",
            json!({"subject": "s", "body": "b", "attachments": 1.5}),
        ),
        (
            "null optional",
            json!({"subject": "s", "body": "b", "attachments": null}),
        ),
        (
            "oversized",
            json!({"subject": "s", "body": "x".repeat(96 * 1024)}),
        ),
    ] {
        let error = schema.validate_state(&state).unwrap_err().to_string();
        assert!(
            !error.contains("secret"),
            "{name}: state content in {error}"
        );
        let _ = name;
    }
    // Exactly at the bound is accepted.
    let overhead = serde_json::to_vec(&json!({"subject": "s", "body": ""}))
        .unwrap()
        .len();
    let at_bound = json!({"subject": "s", "body": "x".repeat(96 * 1024 - overhead)});
    assert_eq!(serde_json::to_vec(&at_bound).unwrap().len(), 96 * 1024);
    assert!(schema.validate_state(&at_bound).is_ok());
}

#[test]
fn every_state_kind_is_checked() {
    let mut definition = example("definition-v1.task-route");
    definition
        .as_object_mut()
        .unwrap()
        .remove("definition_revision");
    for (kind, good, bad) in [
        ("string", json!("x"), json!(1)),
        ("number", json!(1.5), json!("1.5")),
        ("integer", json!(-3), json!(0.5)),
        ("boolean", json!(true), json!("true")),
        ("object", json!({"a": 1}), json!([1])),
        ("array", json!([1]), json!({"a": 1})),
    ] {
        definition["input_schema"]["fields"][0]["kind"] = json!(kind);
        let parsed = DecisionDefinition::from_value(definition.clone()).unwrap();
        assert!(
            parsed
                .input_schema
                .validate_state(&json!({"task": good}))
                .is_ok(),
            "{kind}"
        );
        assert!(
            parsed
                .input_schema
                .validate_state(&json!({"task": bad}))
                .is_err(),
            "{kind}"
        );
    }
}
