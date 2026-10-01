//! Task 7b §15 Judge, Protocol and Cache rows: normalization of all four shapes (redesign
//! §8 answer semantics), gating (frozen decision 7), answer validation (frozen decision 8),
//! policy cascade, shape subcommands, stdout purity on every path (each run is checked by
//! `support::Run::check`), and the opt-in answer cache (frozen decision 4).

mod support;

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use serde_json::{Value, json};
use support::*;

fn triage(project: &Project, answers: Value, policy: &str) -> Run {
    let mock = Mock::start(vec![Scripted::ok(MODEL, answers)]);
    project
        .judge()
        .url(&mock.url)
        .budget()
        .send(&triage_request(project, Some(policy)))
}

fn close(a: &Value, b: f64) -> bool {
    (a.as_f64().unwrap() - b).abs() < 1e-9
}

// ---- Judge ----

#[test]
fn all_four_shapes_normalize_and_accept() {
    let project = Project::examples();
    let run = triage(&project, triage_answers(), "ticket-triage-measured");
    run.expect("accepted", None);
    let r = &run.response;
    assert_eq!(r["request_id"], "ticket-1");
    assert_eq!(r["site_id"], "runtime:ticket-triage");
    assert_eq!(r["fallback_recommended"], false);
    assert_eq!(r["fallback_owner"], "host");
    assert_eq!(
        r["gate"],
        json!({"policy_id": "ticket-triage-measured", "evidence": "measured", "passed": true, "reasons": []})
    );
    assert_eq!(r["provider"], json!({"name": "typesafe", "model": MODEL}));
    assert_eq!(r["metrics"]["cache_hit"], false);
    let a = &r["answers"];
    assert_eq!(
        a.as_object().unwrap().keys().collect::<Vec<_>>(),
        ["team", "urgent", "tags", "sentiment"]
    );
    // Choice: selected label, full probabilities in canonical order, provider confidence.
    assert_eq!(a["team"]["type"], "choice");
    assert_eq!(a["team"]["value"], "billing");
    assert_eq!(
        a["team"]["probabilities"]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        ["billing", "none_of_the_above", "sales", "support"]
    );
    assert_eq!(a["team"]["confidence"], 0.93);
    assert_eq!(a["team"]["gate_confidence"], 0.93);
    // Noul: probability_yes, value at the cutoff, no provider confidence.
    assert_eq!(a["urgent"]["type"], "noul");
    assert!(a["urgent"].get("confidence").is_none());
    // MultiLabel: exact set.
    assert_eq!(a["tags"]["value"], json!(["billing"]));
    // Score: index score, argmax level, mapped value.
    assert_eq!(a["sentiment"]["level"], 3);
    assert!(close(&a["sentiment"]["score"], 2.85));
    assert!(close(&a["sentiment"]["value"], 0.85));
}

#[test]
fn nullable_choice_maps_to_null() {
    let project = Project::examples();
    let mock = Mock::start(vec![Scripted::ok(
        MODEL,
        route_answers("none_of_the_above", 0.95),
    )]);
    let run = project
        .judge()
        .url(&mock.url)
        .budget()
        .send(&route_request(&project, Some("task-route-demo")));
    run.expect("accepted", None);
    let route = &run.response["answers"]["route"];
    assert_eq!(route["value"], Value::Null);
    assert!(close(&route["probabilities"]["none_of_the_above"], 0.96));
}

#[test]
fn confident_noul_no_is_accepted() {
    let project = Project::examples();
    let mut answers = triage_answers();
    answers["urgent"] = noul(0.02);
    let run = triage(&project, answers, "ticket-triage-measured");
    run.expect("accepted", None);
    let urgent = &run.response["answers"]["urgent"];
    assert_eq!(urgent["value"], false);
    assert_eq!(urgent["probability_yes"], 0.02);
    assert_eq!(urgent["cutoff"], 0.5);
    assert!(close(&urgent["gate_confidence"], 0.98));
    assert_eq!(urgent["passed"], true);
}

#[test]
fn multilabel_is_the_exact_set_with_per_label_gates() {
    let project = Project::examples();
    // security (cutoff 0.3, not required, threshold 0.7): p = 0.4 is true but uncertain.
    let mut answers = triage_answers();
    answers["tag_bug"] = noul(0.9);
    answers["tag_security"] = noul(0.4);
    let run = triage(&project, answers.clone(), "ticket-triage-measured");
    run.expect("accepted", None);
    let tags = &run.response["answers"]["tags"];
    assert_eq!(tags["value"], json!(["billing", "bug", "security"]));
    assert_eq!(tags["labels"]["security"]["value"], true);
    assert_eq!(tags["labels"]["security"]["cutoff"], 0.3);
    assert_eq!(tags["labels"]["security"]["passed"], false);
    assert_eq!(tags["labels"]["bug"]["passed"], true);
    assert_eq!(tags["passed"], false);
    assert_eq!(tags["reasons"], json!(["low_confidence"]));

    // bug is required: an uncertain bug label defers the request.
    answers["tag_bug"] = noul(0.6);
    let run = triage(&project, answers, "ticket-triage-measured");
    run.expect("deferred", Some("low_confidence"));
    let tags = &run.response["answers"]["tags"];
    assert_eq!(tags["labels"]["bug"]["passed"], false);
    assert_eq!(tags["value"], json!(["billing", "bug", "security"]));
}

#[test]
fn score_uses_argmax_level_and_mapped_value() {
    let project = Project::examples();
    let mut answers = triage_answers();
    answers["sentiment"] = score(
        &[0.05, 0.6, 0.3, 0.05, 0.0],
        0.7,
        &["Angry", "Unhappy", "Neutral", "Satisfied", "Delighted"],
    );
    let run = triage(&project, answers, "ticket-triage-measured");
    run.expect("accepted", None);
    let s = &run.response["answers"]["sentiment"];
    assert_eq!(s["type"], "score");
    assert_eq!(s["level"], 1);
    assert!(close(&s["score"], 1.35));
    // -2*0.05 + -1*0.6 + 0*0.3 + 1*0.05 + 2*0 = -0.65
    assert!(close(&s["value"], -0.65));
    assert_eq!(s["confidence"], 0.7);
    assert_eq!(
        s["probabilities"]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        ["0", "1", "2", "3", "4"]
    );
}

#[test]
fn score_mapped_value_is_the_weighted_mean_of_unnormalized_probabilities() {
    let project = Project::examples();
    let legend = ["Angry", "Unhappy", "Neutral", "Satisfied", "Delighted"];
    let mut answers = triage_answers();
    answers["sentiment"] = score(&[0.0, 0.5, 0.0, 0.5, 0.5], 0.7, &legend);
    let run = triage(&project, answers, "ticket-triage-measured");
    run.expect("accepted", None);
    let s = &run.response["answers"]["sentiment"];
    assert_eq!(s["level"], 1);
    // (-1*0.5 + 1*0.5 + 2*0.5) / 1.5
    assert!(close(&s["value"], 1.0 / 1.5), "{s}");

    let mut answers = triage_answers();
    answers["sentiment"] = score(&[0.0; 5], 0.7, &legend);
    let run = triage(&project, answers, "ticket-triage-measured");
    run.expect("deferred", Some("provider_unavailable"));
    assert_eq!(run.response["answers"], json!({}));
}

#[test]
fn required_outputs_gate_the_request_and_optional_ones_do_not() {
    let project = Project::examples();
    // sentiment is not required (threshold 0.6): low confidence keeps the value.
    let mut answers = triage_answers();
    answers["sentiment"] = score(
        &[0.0, 0.05, 0.1, 0.8, 0.05],
        0.3,
        &["Angry", "Unhappy", "Neutral", "Satisfied", "Delighted"],
    );
    let run = triage(&project, answers.clone(), "ticket-triage-measured");
    run.expect("accepted", None);
    let s = &run.response["answers"]["sentiment"];
    assert_eq!(s["passed"], false);
    assert_eq!(s["reasons"], json!(["low_confidence"]));
    assert_eq!(s["level"], 3);

    // team is required (threshold 0.9): low confidence defers; answers kept for diagnostics.
    answers["team"]["confidence"] = json!(0.5);
    let run = triage(&project, answers, "ticket-triage-measured");
    run.expect("deferred", Some("low_confidence"));
    assert_eq!(run.response["answers"]["team"]["passed"], false);
    assert_eq!(run.response["answers"]["urgent"]["passed"], true);
    assert_eq!(run.response["fallback_recommended"], true);
    assert_eq!(run.response["error"], Value::Null);
}

#[test]
fn missing_stale_and_experimental_policies() {
    let project = Project::examples();
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.95))]);
    let judge = || project.judge().url(&mock.url).budget();

    let run = judge().send(&route_request(&project, None));
    run.expect("deferred", Some("policy_missing"));
    assert_eq!(run.response["gate"]["evidence"], Value::Null);
    assert_eq!(run.response["provider"], Value::Null);
    let run = judge().send(&route_request(&project, Some("no-such-policy")));
    run.expect("deferred", Some("policy_missing"));
    assert_eq!(mock.count(), 0, "no provider call without a policy");

    let run = judge().send(&route_request(&project, Some("task-route-demo")));
    run.expect("accepted", None);
    assert_eq!(run.response["gate"]["evidence"], "experimental");
    assert_eq!(run.response["gate"]["policy_id"], "task-route-demo");

    // Reinstalling a changed definition leaves the policy bound to the old revision.
    let mut definition = example("definition-v1.task-route");
    definition["questions"]["route"]["instructions"] = json!("Which route?");
    project.install_definition(&definition);
    let run = judge().send(&route_request(&project, Some("task-route-demo")));
    run.expect("deferred", Some("policy_stale"));
    assert_eq!(run.response["gate"]["evidence"], Value::Null);
    assert_eq!(mock.count(), 1, "stale policies never execute");
}

#[test]
fn inline_definition_bound_to_another_revision_is_policy_stale() {
    let project = Project::examples();
    let mut definition = example("definition-v1.task-route");
    definition
        .as_object_mut()
        .unwrap()
        .remove("definition_revision");
    definition["questions"]["route"]["instructions"] = json!("Which route?");
    let mut request = route_request(&project, Some("task-route-demo"));
    request.as_object_mut().unwrap().remove("definition_ref");
    request["definition"] = definition;
    project
        .judge()
        .budget()
        .send(&request)
        .expect("deferred", Some("policy_stale"));
}

#[test]
fn invalid_policy_file_is_an_error() {
    let project = Project::examples();
    let path = project
        .path()
        .join(".snapjudge/policies/task-route-demo.json");
    let mut policy: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    policy["model"] = json!("jev-latest");
    fs::write(&path, policy.to_string()).unwrap();
    project
        .judge()
        .budget()
        .send(&route_request(&project, Some("task-route-demo")))
        .expect("error", Some("invalid_policy"));
}

#[test]
fn shape_subcommands_require_every_output_of_that_shape() {
    let project = Project::examples();
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.95))]);
    let run = project
        .judge()
        .shape("choice")
        .url(&mock.url)
        .budget()
        .send(&route_request(&project, Some("task-route-demo")));
    run.expect("accepted", None);
    for shape in ["noul", "score", "multilabel"] {
        let run = project
            .judge()
            .shape(shape)
            .url(&mock.url)
            .budget()
            .send(&route_request(&project, Some("task-route-demo")));
        run.expect("error", Some("invalid_definition"));
        assert_eq!(run.code, 1);
    }
    project
        .judge()
        .shape("choice")
        .budget()
        .send(&triage_request(&project, Some("ticket-triage-measured")))
        .expect("error", Some("invalid_definition"));
    assert_eq!(mock.count(), 1);
}

type Change = Box<dyn Fn(&mut Value)>;

#[test]
fn invalid_provider_answers_defer_provider_unavailable() {
    let project = Project::examples();
    let legend = ["Angry", "Unhappy", "Neutral", "Satisfied", "Delighted"];
    let cases: Vec<(&str, Change)> = vec![
        (
            "missing answer",
            Box::new(|a| {
                a.as_object_mut().unwrap().remove("urgent");
            }),
        ),
        (
            "wrong type",
            Box::new(
                |a| a["urgent"] = json!({"type": "choice", "choice": "x", "probabilities": {"x": 1.0}, "confidence": 1.0}),
            ),
        ),
        (
            "extra label",
            Box::new(|a| a["team"]["probabilities"]["marketing"] = json!(0.0)),
        ),
        (
            "missing label",
            Box::new(|a| {
                a["team"]["probabilities"]
                    .as_object_mut()
                    .unwrap()
                    .remove("sales");
            }),
        ),
        (
            "probability above 1",
            Box::new(|a| a["team"]["probabilities"]["sales"] = json!(1.5)),
        ),
        (
            "negative probability",
            Box::new(|a| a["tag_bug"] = noul(-0.1)),
        ),
        (
            "confidence above 1",
            Box::new(|a| a["team"]["confidence"] = json!(1.01)),
        ),
        (
            "choice not an argmax",
            Box::new(|a| a["team"]["choice"] = json!("support")),
        ),
        (
            "choice not a label",
            Box::new(|a| a["team"]["choice"] = json!("marketing")),
        ),
        (
            "score levels",
            Box::new(move |a| a["sentiment"] = score(&[0.5, 0.5, 0.0, 0.0], 0.9, &legend[..4])),
        ),
        (
            "score legend",
            Box::new(|a| {
                a["sentiment"]["legend"]
                    .as_object_mut()
                    .unwrap()
                    .remove("4");
            }),
        ),
        (
            "score outside levels",
            Box::new(|a| a["sentiment"]["score"] = json!(4.5)),
        ),
    ];
    for (name, change) in cases {
        let mut answers = triage_answers();
        change(&mut answers);
        let run = triage(&project, answers, "ticket-triage-measured");
        assert_eq!(run.status(), "deferred", "{name}: {}", run.stdout);
        assert_eq!(run.reason(), "provider_unavailable", "{name}");
        assert_eq!(
            run.response["answers"],
            json!({}),
            "{name}: nothing synthesized"
        );
    }
    // No probability-sum check; a choice within 0.005 of the argmax is valid.
    let mut answers = triage_answers();
    answers["team"]["probabilities"] =
        json!({"billing": 0.5, "support": 0.497, "sales": 0.4, "none_of_the_above": 0.4});
    answers["team"]["choice"] = json!("support");
    let run = triage(&project, answers, "ticket-triage-measured");
    run.expect("accepted", None);
    assert_eq!(
        run.response["answers"]["team"]["value"], "billing",
        "argmax is reported"
    );
}

#[test]
fn choice_argmax_ties_break_by_canonical_criteria_order() {
    let project = Project::examples();
    let mut answers = route_answers("review", 0.95);
    answers["route"]["probabilities"] = json!({"review": 0.4, "research": 0.4, "explore": 0.1, "debug": 0.05, "none_of_the_above": 0.05});
    let mock = Mock::start(vec![Scripted::ok(MODEL, answers)]);
    let run = project
        .judge()
        .url(&mock.url)
        .budget()
        .send(&route_request(&project, Some("task-route-demo")));
    run.expect("accepted", None);
    assert_eq!(run.response["answers"]["route"]["value"], "research");
}

// ---- Protocol: stdout purity on every path ----

#[test]
fn request_rejections_write_one_envelope() {
    let project = Project::examples();
    let valid = route_request(&project, Some("task-route-demo"));
    let judge = || project.judge().budget();

    let run = judge().send_bytes(b"{not json");
    run.expect("error", Some("invalid_request"));
    assert_eq!(run.response["request_id"], Value::Null);

    let oversized = vec![b' '; snapjudge::judge::MAX_REQUEST_BYTES + 10];
    judge()
        .send_bytes(&oversized)
        .expect("error", Some("invalid_request"));
    judge()
        .send_bytes(&[0xff, 0xfe])
        .expect("error", Some("invalid_request"));

    let mut request = valid.clone();
    request["protocol_version"] = json!(2);
    let run = judge().send(&request);
    run.expect("error", Some("protocol_mismatch"));
    assert_eq!(run.response["request_id"], "route-1");

    let mut request = valid.clone();
    request["definition"] = example("definition-v1.task-route");
    judge()
        .send(&request)
        .expect("error", Some("invalid_request"));
    let mut request = valid.clone();
    request.as_object_mut().unwrap().remove("definition_ref");
    judge()
        .send(&request)
        .expect("error", Some("invalid_request"));

    let mut request = valid.clone();
    request["site_id"] = json!("runtime:other");
    judge()
        .send(&request)
        .expect("error", Some("invalid_definition"));

    let mut request = valid.clone();
    request["definition_ref"]["id"] = json!("not-installed");
    judge()
        .send(&request)
        .expect("error", Some("invalid_definition"));

    let mut request = valid.clone();
    request["definition_ref"]["definition_revision"] = json!("0".repeat(64));
    judge()
        .send(&request)
        .expect("error", Some("invalid_definition"));

    let mut request = valid.clone();
    request["state"] = json!({"task": 7});
    let run = judge().send(&request);
    run.expect("error", Some("invalid_input"));
    assert_eq!(run.response["site_id"], "runtime:task-route");
    let mut request = valid.clone();
    request["state"] = json!({"task": "x", "extra": MARKER});
    judge()
        .send(&request)
        .expect("error", Some("invalid_input"));
    let mut request = valid;
    request["state"] = json!({});
    judge()
        .send(&request)
        .expect("error", Some("invalid_input"));
}

#[test]
fn request_is_answered_without_waiting_for_stdin_to_close() {
    let project = Project::examples();
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.95))]);
    let request = route_request(&project, Some("task-route-demo"));
    let run = project
        .judge()
        .url(&mock.url)
        .budget()
        .send_open(request.to_string().as_bytes());
    run.expect("accepted", None);
    assert!(
        run.elapsed < Duration::from_millis(1500),
        "{:?}",
        run.elapsed
    );
}

#[test]
fn incomplete_request_with_open_stdin_errors_at_the_deadline() {
    let project = Project::examples();
    let text = route_request(&project, Some("task-route-demo")).to_string();
    let run = project
        .judge()
        .budget()
        .send_open(&text.as_bytes()[..text.len() / 2]);
    run.expect("error", Some("invalid_request"));
    assert_eq!(
        run.response["error"]["message"],
        "request did not arrive within the deadline"
    );
    assert!(
        run.elapsed >= Duration::from_millis(1400),
        "{:?}",
        run.elapsed
    );
    assert!(
        run.elapsed < Duration::from_millis(4000),
        "{:?}",
        run.elapsed
    );

    // A configured `max_timeout_ms` shortens the wait.
    project.config(json!({"max_timeout_ms": 200}));
    let run = project.judge().budget().send_open(b"{");
    run.expect("error", Some("invalid_request"));
    assert!(
        run.elapsed < Duration::from_millis(1200),
        "{:?}",
        run.elapsed
    );
}

#[cfg(unix)]
#[test]
fn deleted_working_directory_is_an_error_envelope() {
    let project = Project::examples();
    let gone = project.path().join("gone");
    fs::create_dir(&gone).unwrap();
    let judge = project.judge().budget();
    let mut command = std::process::Command::new("sh");
    command
        .arg("-c")
        .arg(r#"cd "$1" && rmdir "$1" && exec "$0" judge --json --budget 1 </dev/null"#)
        .arg(env!("CARGO_BIN_EXE_snapjudge"))
        .arg(&gone);
    // The harness environment (no real provider, test key) with the shell as the program.
    let template = judge.command();
    for (key, value) in template.get_envs() {
        match value {
            Some(value) => command.env(key, value),
            None => command.env_remove(key),
        };
    }
    let started = std::time::Instant::now();
    let out = command.output().unwrap();
    let run = Run::check(out, started.elapsed());
    run.expect("error", Some("invalid_policy"));
    assert!(
        run.response["error"]["message"]
            .as_str()
            .unwrap()
            .starts_with("current directory"),
        "{}",
        run.stdout
    );
}

#[test]
fn usage_errors_keep_clap_exit_2_without_envelope() {
    let project = Project::new();
    for args in [
        &["judge"][..],
        &["judge", "--json", "--budget", "-1"],
        &["judge", "--json", "--budget", "NaN"],
        &["judge", "--json", "--unknown"],
        &["judge", "boolean", "--json"],
    ] {
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_snapjudge"))
            .args(args)
            .current_dir(project.path())
            .stdin(std::process::Stdio::null())
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        assert!(out.stdout.is_empty(), "{args:?}");
    }
}

// ---- Cache ----

fn cache_files(project: &Project) -> Vec<PathBuf> {
    let root = project.path().join(".snapjudge/cache");
    let mut files = Vec::new();
    let Ok(dirs) = fs::read_dir(&root) else {
        return files;
    };
    for dir in dirs {
        for file in fs::read_dir(dir.unwrap().path()).unwrap() {
            files.push(file.unwrap().path());
        }
    }
    files.sort();
    files
}

fn without_duration(mut response: Value) -> Value {
    response["metrics"]["duration_ms"] = json!(0);
    response
}

#[test]
fn cache_is_off_by_default() {
    let project = Project::examples();
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.95))]);
    for _ in 0..2 {
        project
            .judge()
            .url(&mock.url)
            .budget()
            .send(&route_request(&project, Some("task-route-demo")))
            .expect("accepted", None);
    }
    assert_eq!(mock.count(), 2);
    assert!(cache_files(&project).is_empty());
}

#[cfg(unix)]
#[test]
fn cache_never_writes_through_a_symlinked_directory() {
    let project = Project::examples();
    project.config(json!({"cache": true}));
    let outside = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(outside.path(), project.path().join(".snapjudge/cache")).unwrap();
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.95))]);
    let request = route_request(&project, Some("task-route-demo"));
    for _ in 0..2 {
        // The failed cache write only costs a refetch.
        project
            .judge()
            .url(&mock.url)
            .budget()
            .send(&request)
            .expect("accepted", None);
    }
    assert_eq!(mock.count(), 2);
    let written: Vec<_> = fs::read_dir(outside.path()).unwrap().collect();
    assert!(written.is_empty(), "nothing created outside: {written:?}");
}

#[test]
fn cache_replays_byte_identically_with_zero_network() {
    let project = Project::examples();
    project.config(json!({"cache": true}));
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.95))]);
    let request = route_request(&project, Some("task-route-demo"));
    let first = project.judge().url(&mock.url).budget().send(&request);
    first.expect("accepted", None);
    assert_eq!(first.response["metrics"]["cache_hit"], false);
    assert_eq!(mock.count(), 1);

    // Replays need neither a key nor a budget, and send nothing.
    let replay = || project.judge().url(&mock.url).no_key().send(&request);
    let second = replay();
    let third = replay();
    second.expect("accepted", None);
    assert_eq!(mock.count(), 1, "the mock saw no request");
    assert_eq!(second.response["metrics"]["cache_hit"], true);
    assert_eq!(second.log["provider_call_attempted"], false);
    assert_eq!(second.log["cache_hit"], true);
    assert_eq!(
        without_duration(second.response.clone()).to_string(),
        without_duration(third.response.clone()).to_string()
    );
    let mut fresh = without_duration(first.response);
    fresh["metrics"]["cache_hit"] = json!(true);
    assert_eq!(fresh, without_duration(second.response));
}

#[test]
fn cache_writes_are_atomic_canonical_entries() {
    let project = Project::examples();
    project.config(json!({"cache": true}));
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.95))]);
    project
        .judge()
        .url(&mock.url)
        .budget()
        .send(&route_request(&project, Some("task-route-demo")))
        .expect("accepted", None);
    let files = cache_files(&project);
    assert_eq!(files.len(), 1, "one entry, no temporary files: {files:?}");
    let path = &files[0];
    let name = path.file_stem().unwrap().to_str().unwrap();
    assert!(snapjudge::decision::jcs::is_revision(name));
    assert_eq!(
        path.parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap(),
        &name[..2]
    );
    let text = fs::read_to_string(path).unwrap();
    assert_eq!(
        text.trim_end(),
        snapjudge::decision::jcs::canonicalize_text(&text).unwrap()
    );
    let entry: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(entry["key"], name);
    assert_eq!(entry["reply"]["model"], MODEL);
    assert!(!text.contains(MARKER), "the cache holds answers, not state");
}

#[test]
fn corrupt_cache_entry_is_a_miss_and_is_replaced() {
    let project = Project::examples();
    project.config(json!({"cache": true}));
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.95))]);
    let request = route_request(&project, Some("task-route-demo"));
    let judge = || project.judge().url(&mock.url).budget().send(&request);
    judge().expect("accepted", None);
    let path = cache_files(&project).remove(0);
    let good = fs::read_to_string(&path).unwrap();
    for corrupt in [
        "",
        "{\"key\":",
        "{\"key\":\"x\",\"reply\":{}}",
        "\u{0}\u{1}",
    ] {
        fs::write(&path, corrupt).unwrap();
        let before = mock.count();
        let run = judge();
        run.expect("accepted", None);
        assert_eq!(run.response["metrics"]["cache_hit"], false);
        assert_eq!(mock.count(), before + 1, "a corrupt entry is a miss");
        assert_eq!(fs::read_to_string(&path).unwrap(), good, "and is replaced");
    }
}

#[test]
fn cache_entry_failing_validation_is_a_miss_and_is_replaced() {
    let project = Project::examples();
    project.config(json!({"cache": true}));
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.95))]);
    let request = route_request(&project, Some("task-route-demo"));
    let judge = || project.judge().url(&mock.url).budget().send(&request);
    judge().expect("accepted", None);
    let path = cache_files(&project).remove(0);
    let good = fs::read_to_string(&path).unwrap();
    let entry: Value = serde_json::from_str(&good).unwrap();
    let mut probability = entry.clone();
    probability["reply"]["answers"]["route"]["probabilities"]["debug"] = json!(2);
    let mut model = entry;
    model["reply"]["model"] = json!("jev-0.0.1");
    for tampered in [probability, model] {
        fs::write(&path, tampered.to_string()).unwrap();
        let before = mock.count();
        let run = judge();
        run.expect("accepted", None);
        assert_eq!(run.response["metrics"]["cache_hit"], false);
        assert_eq!(run.response["provider"]["model"], MODEL);
        assert_eq!(mock.count(), before + 1, "refetched: {tampered}");
        assert_eq!(fs::read_to_string(&path).unwrap(), good, "and overwritten");
    }
}

#[test]
fn model_change_invalidates_the_cache() {
    let project = Project::examples();
    project.config(json!({"cache": true}));
    let mock = Mock::start(vec![
        Scripted::ok(MODEL, route_answers("debug", 0.95)),
        Scripted::ok("jev-1.14.0", route_answers("debug", 0.95)),
    ]);
    let request = route_request(&project, Some("task-route-demo"));
    project
        .judge()
        .url(&mock.url)
        .budget()
        .send(&request)
        .expect("accepted", None);
    let mut policy = example("policy-v1.task-route");
    policy["model"] = json!("jev-1.14.0");
    project.install_policy(&policy);
    let run = project.judge().url(&mock.url).budget().send(&request);
    run.expect("accepted", None);
    assert_eq!(run.response["metrics"]["cache_hit"], false);
    assert_eq!(mock.count(), 2);
    assert_eq!(mock.requests()[1].json()["model"], "jev-1.14.0");
    assert_eq!(cache_files(&project).len(), 2);
}

#[test]
fn policy_change_regates_without_refetch() {
    let project = Project::examples();
    project.config(json!({"cache": true}));
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.9))]);
    let request = route_request(&project, Some("task-route-demo"));
    project
        .judge()
        .url(&mock.url)
        .budget()
        .send(&request)
        .expect("accepted", None);
    let mut policy = example("policy-v1.task-route");
    policy["thresholds"]["route"] = json!(0.95);
    project.install_policy(&policy);
    let run = project.judge().url(&mock.url).budget().send(&request);
    run.expect("deferred", Some("low_confidence"));
    assert_eq!(run.response["metrics"]["cache_hit"], true);
    assert_eq!(mock.count(), 1, "regated from the cached answer");
}
