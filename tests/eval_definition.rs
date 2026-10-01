mod support;

use std::fs;
use std::process::Command;

use serde_json::{Value, json};
use snapjudge::judge::registry::Scope;
use support::{
    KEY, MODEL, Mock, Project, Recorded, Scripted, chat_completion, choice, models_body,
};

fn definition_path() -> std::path::PathBuf {
    std::env::current_dir()
        .unwrap()
        .join("schemas/examples/definition-v1.task-route.json")
}

fn inputs(project: &Project, labelled: bool) -> std::path::PathBuf {
    let path = project.path().join("inputs.jsonl");
    let text = (0..20)
        .map(|index| {
            let mut row = json!({
                "site": "runtime:task-route",
                "input": {"task": format!("explore task {index}")},
            });
            if labelled {
                row["reference"] = json!({"route": "explore"});
            }
            row.to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&path, format!("{text}\n")).unwrap();
    path
}

fn provider(_: usize, request: &Recorded) -> Scripted {
    if request.method == "GET" {
        return Scripted::new(200, models_body());
    }
    if request.path.ends_with("/v1/systemone") {
        return Scripted::ok(
            MODEL,
            json!({"route": choice("explore", json!({
                "explore": 0.99,
                "debug": 0.0025,
                "review": 0.0025,
                "research": 0.0025,
                "none_of_the_above": 0.0025
            }), 0.99)}),
        );
    }
    let body = request.json();
    let rendered: Value =
        serde_json::from_str(body["messages"][0]["content"].as_str().unwrap()).unwrap();
    assert!(rendered["questions"].get("route").is_some());
    Scripted::new(
        200,
        chat_completion("seeded/model", r#"{"route":"explore"}"#, "stop", 0.0001),
    )
}

fn eval(project: &Project, mock: &Mock, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_snapjudge"))
        .arg("eval")
        .args(args)
        .args(["--llm-base-url", &format!("{}/api/v1", mock.url)])
        .current_dir(project.path())
        .env("SNAPJUDGE_CONFIG_DIR", project.user.path())
        .env("SNAPJUDGE_TYPESAFE_URL", &mock.url)
        .env("OPENROUTER_API_KEY", KEY)
        .env("TYPESAFE_API_KEY", KEY)
        .output()
        .unwrap()
}

fn configured_project() -> Project {
    let project = Project::new();
    project.config(json!({"eval": {"designer_model": "seeded/model", "jev_model": MODEL}}));
    project
}

#[test]
fn labelled_definition_does_not_require_a_designer_price() {
    let project = Project::new();
    project.config(json!({"eval": {"designer_model": "unlisted/designer", "jev_model": MODEL}}));
    let mock = Mock::route(provider);
    let inputs = inputs(&project, true);
    let out = project.path().join("out");
    let result = eval(
        &project,
        &mock,
        &[
            "--definition",
            definition_path().to_str().unwrap(),
            "--inputs",
            inputs.to_str().unwrap(),
            "--fixture",
            "--adjudicate",
            "unlisted/adjudicator",
            "--budget",
            "1",
            "--yes",
            "--out",
            out.to_str().unwrap(),
        ],
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(out.join("task-route/policy.json").is_file());
}

#[test]
fn labelled_fixture_needs_only_typesafe_key_not_a_prefetched_catalogue() {
    let project = Project::new();
    project.config(json!({"eval": {"designer_model": "unlisted/designer", "jev_model": MODEL}}));
    let mock = Mock::route(provider);
    let inputs = inputs(&project, true);
    let output = Command::new(env!("CARGO_BIN_EXE_snapjudge"))
        .args(["eval", "--definition", definition_path().to_str().unwrap()])
        .args([
            "--inputs",
            inputs.to_str().unwrap(),
            "--fixture",
            "--target",
            "0",
        ])
        .args([
            "--budget",
            "1",
            "--yes",
            "--llm-base-url",
            &format!("{}/api/v1", mock.url),
        ])
        .current_dir(project.path())
        .env("SNAPJUDGE_CONFIG_DIR", project.user.path())
        .env("SNAPJUDGE_TYPESAFE_URL", &mock.url)
        .env("TYPESAFE_API_KEY", KEY)
        .env_remove("OPENROUTER_API_KEY")
        .env_remove("SNAPJUDGE_LLM_API_KEY")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        mock.requests()
            .iter()
            .any(|request| request.method == "GET")
    );
    assert!(
        mock.requests()
            .iter()
            .all(|request| request.method == "GET" || request.path.ends_with("/v1/systemone"))
    );
    let run: Value = serde_json::from_str(
        &fs::read_to_string(project.path().join(".snapjudge/eval/task-route/run.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        run["replay_only"], false,
        "live Jev calls are not replay-only"
    );
}

#[test]
fn cli_definition_uses_labels_and_exports_fixture_evidence() {
    let project = configured_project();
    let mock = Mock::route(provider);
    let inputs = inputs(&project, true);
    let out = project.path().join("out");
    let result = eval(
        &project,
        &mock,
        &[
            "--definition",
            definition_path().to_str().unwrap(),
            "--inputs",
            inputs.to_str().unwrap(),
            "--fixture",
            "--budget",
            "1",
            "--yes",
            "--out",
            out.to_str().unwrap(),
        ],
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let dir = out.join("task-route");
    let policy: Value =
        serde_json::from_str(&fs::read_to_string(dir.join("policy.json")).unwrap()).unwrap();
    assert_eq!(policy["evidence"], "fixture");
    let dataset = fs::read_to_string(dir.join("dataset.jsonl")).unwrap();
    assert!(
        dataset
            .lines()
            .all(|line| { serde_json::from_str::<Value>(line).unwrap()["origin"] == "synthetic" })
    );
    assert!(
        mock.requests()
            .iter()
            .all(|request| { request.method == "GET" || request.path.ends_with("/v1/systemone") })
    );
}

#[test]
fn cli_definition_teacher_renders_questions_and_policy_verify_detects_staleness() {
    let project = configured_project();
    let mock = Mock::route(provider);
    let inputs = inputs(&project, false);
    let out = project.path().join("out");
    let result = eval(
        &project,
        &mock,
        &[
            "--definition",
            definition_path().to_str().unwrap(),
            "--inputs",
            inputs.to_str().unwrap(),
            "--teacher",
            "seeded/model",
            "--experimental",
            "--min-accepted",
            "1",
            "--budget",
            "1",
            "--yes",
            "--out",
            out.to_str().unwrap(),
        ],
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let definition = fs::read_to_string(definition_path()).unwrap();
    project
        .registry()
        .install_definition(&definition, Scope::Project)
        .unwrap();
    let dir = out.join("task-route");
    let policy = fs::read_to_string(dir.join("policy.json")).unwrap();
    project
        .registry()
        .install_policy(&policy, Scope::Project)
        .unwrap();
    let verify = Command::new(env!("CARGO_BIN_EXE_snapjudge"))
        .args([
            "policies",
            "verify",
            "task-route",
            "--export",
            dir.to_str().unwrap(),
        ])
        .current_dir(project.path())
        .env("SNAPJUDGE_CONFIG_DIR", project.user.path())
        .output()
        .unwrap();
    assert!(
        verify.status.success(),
        "{}",
        String::from_utf8_lossy(&verify.stderr)
    );
    assert_eq!(
        String::from_utf8(verify.stdout).unwrap(),
        "task-route\tok\n"
    );

    let mut changed: Value = serde_json::from_str(&definition).unwrap();
    changed
        .as_object_mut()
        .unwrap()
        .remove("definition_revision");
    changed["questions"]["route"]["instructions"] = json!("Choose a work route.");
    project
        .registry()
        .install_definition(&changed.to_string(), Scope::Project)
        .unwrap();
    let stale = Command::new(env!("CARGO_BIN_EXE_snapjudge"))
        .args(["policies", "verify", "task-route"])
        .current_dir(project.path())
        .env("SNAPJUDGE_CONFIG_DIR", project.user.path())
        .output()
        .unwrap();
    assert!(!stale.status.success());
    assert_eq!(
        String::from_utf8(stale.stdout).unwrap(),
        "task-route\tstale\n"
    );
}
