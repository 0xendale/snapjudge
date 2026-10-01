use std::cell::Cell;

use serde_json::{Value, json};
use snapjudge::adapter::claude::{Binding, Event, advisory_request, handle, selected_tool};
use snapjudge::adapter::config::load_config;
use snapjudge::judge::JudgeResponse;

fn reply(evidence: &str, status: &str, request_id: &str) -> JudgeResponse {
    serde_json::from_value(json!({
        "protocol_version": 1,
        "request_id": request_id,
        "site_id": "agent:claude-code:task-route",
        "status": status,
        "answers": {"route": {"type":"choice", "value":"debug", "probabilities": {"debug": 1}, "confidence":1, "gate_confidence":1, "passed":true, "reasons":[]}},
        "gate": {"policy_id":"claude-code.task-route", "evidence":evidence, "passed": status == "accepted", "reasons": if status == "accepted" { vec![] } else { vec!["low_confidence"] }},
        "provider": {"name":"typesafe", "model":"jev-1.13.0"},
        "fallback_recommended": status != "accepted",
        "fallback_owner":"host", "metrics":{"duration_ms": 3, "cache_hit":false}, "error":null,
    }))
    .unwrap()
}

fn binding() -> Binding {
    Binding {
        definition_revision: "a".repeat(64),
        policy_id: "claude-code.task-route".into(),
    }
}

#[test]
fn measured_acceptance_emits_native_user_prompt_context() {
    let input = br#"{"hook_event_name":"UserPromptSubmit","session_id":"session-1","prompt":"Investigate a failing test"}"#;
    let result = handle(Event::UserPromptSubmit, input, &binding(), |request| {
        reply("measured", "accepted", &request.request_id)
    });
    let output: Value = serde_json::from_str(&result.unwrap()).unwrap();
    assert_eq!(
        output["hookSpecificOutput"]["hookEventName"],
        "UserPromptSubmit"
    );
    let context = output["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .unwrap();
    assert!(context.contains("debug"));
    assert!(context.contains("agent:claude-code:task-route"));
    assert!(context.contains("measured"));
    assert_eq!(output.as_object().unwrap().len(), 1);
}

#[test]
fn fixture_defer_error_and_invalid_inputs_are_silent_without_calls_when_invalid() {
    let input = br#"{"hook_event_name":"UserPromptSubmit","session_id":"session-1","prompt":"Investigate a failing test"}"#;
    for (evidence, status) in [
        ("fixture", "accepted"),
        ("experimental", "accepted"),
        ("measured", "deferred"),
    ] {
        assert_eq!(
            handle(Event::UserPromptSubmit, input, &binding(), |request| reply(
                evidence,
                status,
                &request.request_id
            )),
            None
        );
    }
    let calls = Cell::new(0);
    for invalid in [
        b"not json".as_slice(),
        br#"{"hook_event_name":"UserPromptSubmit","session_id":"x","prompt":""}"#,
    ] {
        assert_eq!(
            handle(Event::UserPromptSubmit, invalid, &binding(), |request| {
                calls.set(calls.get() + 1);
                reply("measured", "accepted", &request.request_id)
            }),
            None
        );
    }
    assert_eq!(calls.get(), 0);
}

#[test]
fn unrelated_events_and_recursive_tool_events_do_not_call_judge() {
    let input = br#"{"hook_event_name":"PreToolUse","session_id":"session-1","tool_name":"mcp__plugin_snapjudge-claude_snapjudge__snapjudge_judge","tool_input":{"prompt":"route this"}}"#;
    let calls = Cell::new(0);
    assert_eq!(
        handle(Event::PreToolUse, input, &binding(), |request| {
            calls.set(calls.get() + 1);
            reply("measured", "accepted", &request.request_id)
        }),
        None
    );
    assert_eq!(calls.get(), 0);
}

#[test]
fn adapter_configuration_requires_explicit_revision_and_bounded_tool_allowlist() {
    let project = tempfile::tempdir().unwrap();
    let path = project.path().join("adapter.json");
    std::fs::write(
        &path,
        json!({
            "route": {"definition_revision": "a".repeat(64), "policy_id": "claude-code.task-route"},
            "advisory_tools": ["Task"],
            "observations": true
        })
        .to_string(),
    )
    .unwrap();
    let config = load_config(&path).unwrap();
    assert_eq!(config.route.unwrap().policy_id, "claude-code.task-route");
    assert_eq!(config.advisory_tools, ["Task"]);
    assert!(config.observations);
    std::fs::write(
        &path,
        r#"{"route":{"definition_revision":"wrong","policy_id":"x"}}"#,
    )
    .unwrap();
    assert!(load_config(&path).is_err());
}

#[test]
fn pre_tool_advisory_projects_only_allowlisted_prompt_without_permissions() {
    let input = br#"{"hook_event_name":"PreToolUse","session_id":"session-1","tool_name":"Task","tool_input":{"prompt":"Investigate failure","extra":"must not be sent"}}"#;
    let allowed = vec!["Task".to_string()];
    let request = advisory_request(input, &binding(), &allowed).unwrap();
    assert_eq!(request.state, json!({"prompt": "Investigate failure"}));
    assert_eq!(request.site_id, "agent:claude-code:task-route");
    let ignored = std::str::from_utf8(input)
        .unwrap()
        .replace("\"Task\"", "\"Other\"");
    assert!(advisory_request(ignored.as_bytes(), &binding(), &allowed).is_none());
    let recursive = br#"{"hook_event_name":"PreToolUse","session_id":"session-1","tool_name":"mcp__plugin_snapjudge-claude_snapjudge__snapjudge_judge","tool_input":{"prompt":"route"}}"#;
    let configured = vec!["mcp__plugin_snapjudge-claude_snapjudge__snapjudge_judge".to_string()];
    assert!(advisory_request(recursive, &binding(), &configured).is_none());
    let after = br#"{"hook_event_name":"PostToolUse","session_id":"session-1","tool_name":"Task","tool_input":{"prompt":"secret"}}"#;
    assert_eq!(
        selected_tool(Event::PostToolUse, after, &allowed),
        Some("Task".into())
    );
}
