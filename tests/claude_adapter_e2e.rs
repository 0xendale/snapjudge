mod support;

use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::{Value, json};
use support::{KEY, MODEL, Mock, Project, Scripted, route_answers};

#[test]
fn native_hook_uses_exported_policy_and_keeps_fixture_fallback_silent() {
    let project = Project::new();
    project.config(json!({"eval":{"designer_model":"unused/designer", "jev_model":MODEL}}));
    project.user_config(json!({"spend":{"budget_usd":1}}));
    let source_cache = Path::new("fixtures/policies/claude-code.task-route/cache");
    let destination = project.path().join(".snapjudge/cache/eval");
    fs::create_dir_all(&destination).unwrap();
    for entry in fs::read_dir(source_cache).unwrap() {
        let entry = entry.unwrap();
        if entry.path().is_dir() {
            let shard = destination.join(entry.file_name());
            fs::create_dir(&shard).unwrap();
            for file in fs::read_dir(entry.path()).unwrap() {
                let file = file.unwrap();
                fs::copy(file.path(), shard.join(file.file_name())).unwrap();
            }
        } else {
            fs::copy(entry.path(), destination.join(entry.file_name())).unwrap();
        }
    }
    let definition_path = std::env::current_dir()
        .unwrap()
        .join("fixtures/agent/claude-code.task-route.json");
    let inputs_path = std::env::current_dir()
        .unwrap()
        .join("fixtures/agent/claude-code.task-route.inputs.jsonl");
    let output = Command::new(env!("CARGO_BIN_EXE_snapjudge"))
        .args([
            "eval",
            "--definition",
            definition_path.to_str().unwrap(),
            "--inputs",
            inputs_path.to_str().unwrap(),
            "--target",
            "0.8",
            "--out",
            "export",
        ])
        .current_dir(project.path())
        .env("SNAPJUDGE_CONFIG_DIR", project.user.path())
        .env_remove("TYPESAFE_API_KEY")
        .env_remove("SNAPJUDGE_TYPESAFE_URL")
        .env_remove("OPENROUTER_API_KEY")
        .env_remove("SNAPJUDGE_LLM_API_KEY")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let measured: Value = serde_json::from_str(
        &fs::read_to_string(
            project
                .path()
                .join("export/claude-code.task-route/policy.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(measured["evidence"], "measured");
    let definition: Value =
        serde_json::from_str(&fs::read_to_string(&definition_path).unwrap()).unwrap();
    project.install_definition(&definition);
    project.install_policy(&measured);
    let config = project.path().join("claude-adapter.json");
    fs::write(&config, json!({"route":{
        "definition_revision": definition["definition_revision"], "policy_id": "claude-code.task-route"
    }}).to_string()).unwrap();
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.99))]);
    let send = || {
        let mut child = Command::new(env!("CARGO_BIN_EXE_snapjudge"))
            .args(["adapter", "claude-code", "--event", "UserPromptSubmit"])
            .current_dir(project.path())
            .env("SNAPJUDGE_CLAUDE_CONFIG", &config)
            .env("SNAPJUDGE_CONFIG_DIR", project.user.path())
            .env("SNAPJUDGE_TYPESAFE_URL", &mock.url)
            .env("TYPESAFE_API_KEY", KEY)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(br#"{"hook_event_name":"UserPromptSubmit","session_id":"session-1","prompt":"Investigate a failing test"}"#).unwrap();
        child.wait_with_output().unwrap()
    };
    let accepted = send();
    assert!(accepted.status.success());
    let native: Value = serde_json::from_slice(&accepted.stdout).unwrap();
    assert_eq!(
        native["hookSpecificOutput"]["hookEventName"],
        "UserPromptSubmit"
    );
    assert!(
        native["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap()
            .contains("debug")
    );
    assert!(!String::from_utf8_lossy(&accepted.stdout).contains("Investigate a failing test"));
    assert_eq!(mock.count(), 1);

    let fixture: Value = serde_json::from_str(
        &fs::read_to_string("fixtures/policies/claude-code.task-route/policy.json").unwrap(),
    )
    .unwrap();
    project.install_policy(&fixture);
    let fallback = send();
    assert!(fallback.status.success());
    assert!(fallback.stdout.is_empty());
    assert_eq!(mock.count(), 2);
}
