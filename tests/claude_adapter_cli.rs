use std::io::Write;
use std::process::{Command, Stdio};

use serde_json::{Value, json};

#[test]
fn native_bridge_without_configuration_continues_with_empty_stdout() {
    let project = tempfile::tempdir().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_snapjudge"))
        .args(["adapter", "claude-code", "--event", "UserPromptSubmit"])
        .current_dir(project.path())
        .env_remove("SNAPJUDGE_CLAUDE_CONFIG")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(br#"{"hook_event_name":"UserPromptSubmit","session_id":"session-1","prompt":"Find bug"}"#).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    assert!(output.stdout.is_empty());
}

#[test]
fn configured_tool_events_log_only_metadata_and_never_decide_permissions() {
    let project = tempfile::tempdir().unwrap();
    let config = project.path().join("adapter.json");
    std::fs::write(
        &config,
        json!({
            "route": {"definition_revision": "a".repeat(64), "policy_id": "claude-code.task-route"},
            "advisory_tools": ["Task"], "observations": true
        })
        .to_string(),
    )
    .unwrap();
    for (event, body) in [
        (
            "PreToolUse",
            r#"{"hook_event_name":"PreToolUse","session_id":"session-1","tool_name":"Task","tool_input":{"prompt":"private prompt"}}"#,
        ),
        (
            "PostToolUse",
            r#"{"hook_event_name":"PostToolUse","session_id":"session-1","tool_name":"Task","tool_input":{"prompt":"private prompt"}}"#,
        ),
    ] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_snapjudge"))
            .args(["adapter", "claude-code", "--event", event])
            .env("SNAPJUDGE_CLAUDE_CONFIG", &config)
            .env_remove("TYPESAFE_API_KEY")
            .current_dir(project.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(body.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success());
        assert!(
            output.stdout.is_empty(),
            "{event} cannot set a permission decision"
        );
        let log = String::from_utf8(output.stderr).unwrap();
        assert!(!log.contains("private prompt"));
        assert!(log.contains(if event == "PreToolUse" {
            "invalid_definition"
        } else {
            "observed"
        }));
    }
}

#[test]
fn bundled_launchers_use_explicit_binary_and_clean_mcp_stdout() {
    let project = tempfile::tempdir().unwrap();
    let hook = std::env::current_dir()
        .unwrap()
        .join("adapters/claude-code/scripts/launch-hook.sh");
    let mcp = std::env::current_dir()
        .unwrap()
        .join("adapters/claude-code/scripts/launch-mcp.sh");
    let missing = Command::new(&hook)
        .arg("UserPromptSubmit")
        .env_remove("SNAPJUDGE_BINARY")
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(missing.status.success());
    assert!(missing.stdout.is_empty());
    let unavailable = Command::new(&mcp)
        .arg(project.path())
        .env_remove("SNAPJUDGE_BINARY")
        .output()
        .unwrap();
    assert!(!unavailable.status.success());
    assert!(unavailable.stdout.is_empty());

    let mut child = Command::new(&mcp)
        .arg(project.path())
        .env("SNAPJUDGE_BINARY", env!("CARGO_BIN_EXE_snapjudge"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(br#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25"}}
"#).unwrap();
    let response = child.wait_with_output().unwrap();
    assert!(response.status.success());
    let message: Value = serde_json::from_slice(&response.stdout).unwrap();
    assert_eq!(message["result"]["protocolVersion"], "2025-11-25");
    assert_eq!(message["result"]["serverInfo"]["name"], "snapjudge");
}

#[test]
fn session_start_and_unknown_events_never_call_a_provider_or_block_the_host() {
    let project = tempfile::tempdir().unwrap();
    let config = project.path().join("adapter.json");
    std::fs::write(
        &config,
        json!({
            "route": {"definition_revision": "a".repeat(64), "policy_id": "claude-code.task-route"},
            "observations": true
        })
        .to_string(),
    )
    .unwrap();
    let mut start = Command::new(env!("CARGO_BIN_EXE_snapjudge"))
        .args(["adapter", "claude-code", "--event", "SessionStart"])
        .env("SNAPJUDGE_CLAUDE_CONFIG", &config)
        .env_remove("TYPESAFE_API_KEY")
        .current_dir(project.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    start
        .stdin
        .take()
        .unwrap()
        .write_all(br#"{"hook_event_name":"SessionStart","session_id":"session-1"}"#)
        .unwrap();
    let output = start.wait_with_output().unwrap();
    assert!(output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("route unavailable"));

    let unknown = Command::new(env!("CARGO_BIN_EXE_snapjudge"))
        .args(["adapter", "claude-code", "--event", "UnknownEvent"])
        .env(
            "SNAPJUDGE_CLAUDE_CONFIG",
            project.path().join("missing.json"),
        )
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(unknown.status.success());
    assert!(unknown.stdout.is_empty());
    assert!(unknown.stderr.is_empty());
}
